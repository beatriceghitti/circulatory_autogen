// coupler.cpp
// Master coupler for coupled 1d-0d simulations: it relays the time step negotiation and the
// coupling data between the 1d solver (Python, src/solver1d/main1D.py) and the 0d solver
// (C++, generated main0d), using MPI. Runs on exactly one rank.
//
// The three programs are started together by a single mpirun (MPMD launch), i.e. on one line:
//   mpirun -np 1 ./coupler <coupler_config.json>
//        : -np 1 <path>/main0d -coupled 1 -ODEsolver ... -T0 ... -nCC ... -networkName ... -initStatePath ...
//        : -np 1 <python> <path>/main1D.py 1 <ODEsolver> <T0> <nCC> <input.ini> <initStatePath>
// src/coupler/launch_coupled_mpi.bash builds this command from the coupler configuration file.
//
// Message protocol (see coupler_mpi_common.h):
//   [start]     coupler -> 0d, 1d : TAG_CONFIG handshake
//               (if volume sum) 1d -> coupler -> 0d : TAG_VOL, initial 1d volume
//   [each global time step]
//               0d -> coupler -> 1d : TAG_DT_STEP {time, dt proposed by 0d}
//               1d -> coupler -> 0d : TAG_DT_STEP {time, dt agreed}
//     [each internal 0d time step / evaluation of the 0d right-hand side]
//               0d -> coupler -> 1d : TAG_DT {time, internal dt}
//               0d -> coupler -> 1d : TAG_DATA (all one-to-one 1d-0d connections)
//               1d -> coupler -> 0d : TAG_DATA
//     [end of the global time step]
//               0d -> coupler -> 1d : TAG_DT {time, -999}
//               1d -> coupler -> 0d : TAG_DT {time, -999}
//               (if volume sum) 1d -> coupler -> 0d : TAG_VOL
//   [end]       0d -> coupler, 1d -> coupler : TAG_DONE

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "coupler_auxFun.h"
#include "coupler_mpi_common.h"

class Coupler1d0d {
private:
    // World ranks of the 0d and 1d solvers (their rank 0).
    int rank0d = -1;
    int rank1d = -1;

    std::string inputFold;
    std::string networkName;
    std::string ODEsolver;
    double T0 = 0.0;
    int nCC = 0;

    size_t N1d0dTot = 0;
    size_t N1d0d = 0;
    AuxCouplerFun auxFun;
    bool couple_volume_sum = false;

    const double tolTime = COUPLING_TOL_TIME; // shared with the 0d solver
    static constexpr int DL = COUPLING_DATA_LENGTH;

    const char* name(int rank) const {
        if (rank == rank0d) return "0d solver";
        if (rank == rank1d) return "1d solver";
        return "unknown process";
    }

    // Receive exactly buf.size() doubles with tag `tag` from world rank `src`.
    // If the next message from `src` is anything else (typically TAG_DONE because that solver
    // stopped before the end of the simulation), abort the whole MPI job
    // (otherwise the other programs would wait forever).
    void recv(std::vector<double>& buf, int src, int tag, const char* what) {
        MPI_Status status;
        MPI_Probe(src, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // Perform a blocking test for a message from source `src` with any tag (MPI_ANY_TAG)
        int n = -1;
        MPI_Get_count(&status, MPI_DOUBLE, &n); // Return the source, tag and number of elements of datatype received.
        if (status.MPI_TAG != tag || n != static_cast<int>(buf.size())) {
            if (status.MPI_TAG == TAG_DONE) {
                std::cerr << "Coupler :: Error :: " << name(src) << " terminated while the coupler was waiting for "
                          << what << ". Aborting." << std::endl;
            } else {
                std::cerr << "Coupler :: Error :: expected " << what << " (tag " << tag << ", " << buf.size()
                          << " doubles) from " << name(src) << ", but received tag " << status.MPI_TAG
                          << " with " << n << " doubles. Aborting." << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Recv(buf.data(), n, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    void send(const std::vector<double>& buf, int dst, int tag) {
        MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);
    }

    // Tag of the next message that `src` sends to the coupler (without receiving it).
    int next_tag(int src) {
        MPI_Status status;
        MPI_Probe(src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        return status.MPI_TAG;
    }

public:
    void init(const std::string& initFilePath, const CouplingLayout& layout) {

        int local_size;
        MPI_Comm_size(layout.local, &local_size);
        rank0d = layout.roots[COLOR_SOLVER0D];
        rank1d = layout.roots[COLOR_SOLVER1D];
        if (local_size != 1 || rank0d < 0 || rank1d < 0) {
            throw std::runtime_error(
                "Coupler1d0d::init() : the coupler needs exactly 1 coupler rank, plus the 0d and the 1d solvers in the same mpirun, e.g.\n"
                "  mpirun -np 1 ./coupler config.json : -np 1 ./main0d -coupled 1 ... : -np 1 python main1D.py 1 ...\n"
                "  (see src/coupler/launch_coupled_mpi.bash)");
        }

        std::ifstream file(initFilePath);
        if (!file.is_open()) {
            throw std::runtime_error("Coupler1d0d::init() : could not open coupler configuration file " + initFilePath);
        }
        json config;
        file >> config;

        inputFold = config.value("inputFold", "./../CA_files/generated_models/aortic_bif_hybrid/");
        networkName = config.value("networkName", "aortic_bif_hybrid");
        ODEsolver = config.value("ODEsolver", "CVODE");
        T0 = config.value("T0", 1.1);
        nCC = config.value("nCC", 5);

        std::string fileJson = inputFold+networkName+"_coupler1d0d.json";
        std::string dataJson = auxFun.readJsonFromFile(fileJson);
        if (dataJson.empty()) {
            throw std::runtime_error("Coupler1d0d::init() : could not read 1d-0d connections file " + fileJson);
        }

        std::map<std::string, std::map<std::string, int>> pipe_1d_0d_info = auxFun.deserializeFromJson(dataJson);
        N1d0dTot = pipe_1d_0d_info.size();
        N1d0d = 0;
        // XXX here we are assuming that these 1d-0d connections are ordered such that the volume sum connection is the last one.
        // TODO make sure this is always the case, by ordering the connections based on their type.
        for (size_t i = 0; i < N1d0dTot; ++i) {
            if (pipe_1d_0d_info[std::to_string(i+1)].find("port_volume_sum") != pipe_1d_0d_info[std::to_string(i+1)].end()){
                if (pipe_1d_0d_info[std::to_string(i+1)]["port_volume_sum"] == 1){
                    // this is a 'global' volume sum connection, for which we will use dedicated messages (as for the time step)
                    couple_volume_sum = true;
                }
            // here we could add other 'special' global connections for which we want dedicated messages
            // we are assuming that all these special connections are stored at the end of the pipe_1d_0d_info file/structure
            } else{
                // this is a 'standard' one-to-one 1d-0d connection
                N1d0d++;
            }
        }
        std::cout << "Coupler1d0d :: ODE solver : " << ODEsolver << " || T0 : " << T0 << " || nCC : " << nCC << std::endl;
        std::cout << "Coupler1d0d :: MPI world ranks : 0d solver " << rank0d << " || 1d solver " << rank1d << std::endl;
        std::cout << "Number of one-to-one 1d-0d connections: " << N1d0d << std::endl;
        std::cout << "Total number of connections (including global connections): " << N1d0dTot << " || With time step connection: "  << N1d0dTot+1 << std::endl;
    }

    // Send the coupling layout to both solvers, which check it against their own and abort the MPI job on any mismatch.
    void handshake() {
        // tolTime: checked by the 0d solver, adopted by the 1d solver (see COUPLING_TOL_TIME)
        std::vector<double> cfg = {static_cast<double>(N1d0d), couple_volume_sum ? 1.0 : 0.0, T0, static_cast<double>(nCC), tolTime};
        send(cfg, rank0d, TAG_CONFIG);
        send(cfg, rank1d, TAG_CONFIG);
    }

    // Relay loop: the coupler orchestrates the communication between the 0d and 1d solvers.
    void run() {

        double dtGlob = 0.;
        double timeGlob = 0.;
        double dtLoc = 0.;
        double timeLoc = 0.;

        std::vector<double> one_buffer(N1d0d*DL);
        std::vector<double> zero_buffer(N1d0d*DL);
        std::vector<double> one_buffer_dt(DL);
        std::vector<double> zero_buffer_dt(DL);
        std::vector<double> one_buffer_vol(DL);
        std::vector<double> zero_done(DL);
        std::vector<double> one_done(DL);

        int count = 0;

        // this is to initialise volume sum exchange from main1d to main0d, to compute total volume in the system
        if (couple_volume_sum){
            recv(one_buffer_vol, rank1d, TAG_VOL, "the initial volume sum");
            send(one_buffer_vol, rank0d, TAG_VOL);
        }

        bool time_step_exchanged = true;

        while (true) {

            // this is to exchange global synchronization time and time step between main0d and main1D 
            // BEFORE advancing each solver in time
            if (time_step_exchanged){
                // Each global time step starts with a message from main0d: either its time step,
                // or the notification that it has completed the simulation.
                if (next_tag(rank0d) == TAG_DONE){
                    recv(zero_done, rank0d, TAG_DONE, "the end of the simulation");
                    std::cout << "Coupler1d0d :: 0d solver has completed the simulation" << std::endl;
                    break;
                }
                // read time step data from main0d and send it to main1D
                recv(zero_buffer_dt, rank0d, TAG_DT_STEP, "the global time step");
                send(zero_buffer_dt, rank1d, TAG_DT_STEP);

                // read time step data from main1D and send it back to main0d
                recv(one_buffer_dt, rank1d, TAG_DT_STEP, "the global time step");
                send(one_buffer_dt, rank0d, TAG_DT_STEP);

                timeGlob = one_buffer_dt[0];
                dtGlob = one_buffer_dt[1];
                std::cout << std::setprecision(8) << "Coupler1d0d :: Global time exchanged : " << timeGlob << " || dt : " << dtGlob << std::endl;
            }

            // this is to exchange local internal time and time step taken by the specific ODE solver between main0d and main1D
            // read time step data from main0d
            recv(zero_buffer_dt, rank0d, TAG_DT, "the internal time step");
            timeLoc = zero_buffer_dt[0];
            dtLoc = zero_buffer_dt[1];

            // std::cout << "Coupler :: Local time step exchanged " << timeLoc << " " << dtLoc << std::endl;

            time_step_exchanged = false;
            if (dtLoc<0.){ // XXX dtLoc = -999 indicates that the ODE solver has completed the evolution to the next time level
                // std::cout << "Coupler :: Received negative 0d internal time step dtLoc " << dtLoc << std::endl;

                // send time step data to main1D
                send(zero_buffer_dt, rank1d, TAG_DT);
                // read time step data from main1D and send it back to main0d
                recv(one_buffer_dt, rank1d, TAG_DT, "the end of the global time step");
                send(one_buffer_dt, rank0d, TAG_DT);

                if (couple_volume_sum){
                    // read volume sum data from main1D and send it to main0d
                    recv(one_buffer_vol, rank1d, TAG_VOL, "the volume sum");
                    send(one_buffer_vol, rank0d, TAG_VOL);
                }

                count++;
                time_step_exchanged = true;
                continue;
            }

            // if (timeLoc>timeGlob+dtGlob){
            if (std::round(timeLoc/tolTime)*tolTime > std::round((timeGlob+dtGlob)/tolTime)*tolTime) {
                // but this should not happen
                dtLoc = dtLoc - (timeLoc-(timeGlob+dtGlob));
                timeLoc = timeGlob+dtGlob;
                timeLoc = std::round(timeLoc/tolTime)*tolTime;
                // std::cout << "Coupler :: LOCAL TIME LARGER THAN GLOBAL TIME " << timeLoc << " " << timeGlob+dtGlob << " || NEW TIME STEP " << " " << dtLoc << std::endl;

                zero_buffer_dt[0] = timeLoc;
                zero_buffer_dt[1] = dtLoc;
            }

            // send time step data to main1D
            send(zero_buffer_dt, rank1d, TAG_DT);

            // std::cout << std::setprecision(16)
            //         << "Coupler :: timeGlobal " <<  timeGlob << " " << dtGlob
            //         << " || timeLocal " <<  timeLoc << " " << dtLoc
            //         << " || timeDiff " << (timeGlob+dtGlob)-timeLoc
            //         << std::endl;

            // this is to exchange solution data between main0d and main1D for all 1D-0D connections (one message each way)
            if (N1d0d > 0) {
                // read data from main0d and send it to main1D
                recv(zero_buffer, rank0d, TAG_DATA, "the 1d-0d coupling data");
                send(zero_buffer, rank1d, TAG_DATA);
                // read data from main1D and send it back to main0d
                recv(one_buffer, rank1d, TAG_DATA, "the 1d-0d coupling data");
                send(one_buffer, rank0d, TAG_DATA);
            }

            count++;
        }

        // main0d has completed the simulation: main1D must have completed it too.
        recv(one_done, rank1d, TAG_DONE, "the end of the simulation");
        std::cout << "Coupler1d0d :: 1d solver has completed the simulation" << std::endl;

        std::cout << "### Coupler1d0d :: Stop execution! ###" << std::endl;
        std::cout << "Global final time : " << std::round((timeGlob+dtGlob)/tolTime)*tolTime << std::endl;
        std::cout << "Final time of 0d solver : " << zero_done[0] << " || of 1d solver : " << one_done[0] << std::endl;
        std::cout << "Number of 1D-0D communications : " << count << std::endl;
    }
};

int main(int argc, char* argv[]) {

    // Only print the MPI version (used by launch_coupled_mpi.bash), before starting MPI.
    if (argc > 1 && std::strcmp(argv[1], MPI_VERSION_OPTION) == 0) {
        print_mpi_version();
        return 0;
    }

    MPI_Init(&argc, &argv);
    // This must come first: it is collective with the 0d and 1d solvers.
    CouplingLayout layout = setup_coupling(COLOR_COUPLER);

    try {
        if (argc < 2) {
            throw std::runtime_error("Input file path needs to be passed as command-line argument when executing your coupler.");
        }
        Coupler1d0d coupler;
        coupler.init(argv[1], layout);
        coupler.handshake();
        coupler.run();
    }
    catch (const std::exception& e) {
        // Abort the whole MPI job, otherwise the solvers would wait forever for the coupler.
        std::cerr << "Coupler :: Error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_free(&layout.local);
    MPI_Finalize();
    std::cout << "### DONE ###" << std::endl;
    return 0;
}

// coupler.cpp
// Master coupler for coupled 1d-0d simulations: it relays the time step negotiation and the
// coupling data between the 1d solver (Python, src/solver1d/main1D.py) and the 0d solver
// (C++, generated main0d), using MPI. Runs on exactly one rank.
//
// The three programs are started together by a single mpirun (MPMD launch), i.e. on one line:
//   mpirun -np 1 ./coupler <coupler_config.json>
//        : -np 1 <path>/main0d -coupled 1 -ODEsolver ... -T0 ... -preTime ... -simTime ... -networkName ... -initStatePath ...
//        : -np 1 <python> <path>/main1D.py 1 <ODEsolver> <T0> <pre_time> <sim_time> <input.ini> <initStatePath>
// src/coupler/launch_coupled_mpi.bash builds this command from the coupler configuration file.
//
// Message protocol (see coupler_mpi_common.h):
//   [start]     coupler -> 0d, 1d : TAG_CONFIG handshake, then TAG_CONN (1d-0d connections, see COUPLING_CONN_FIELDS)
//               (if global connections, e.g. volume sum) 1d -> coupler -> 0d : TAG_VOL, their initial values
//   [each global time step]
//               0d -> coupler -> 1d : TAG_DT_STEP {time, dt proposed by 0d}
//               1d -> coupler -> 0d : TAG_DT_STEP {time, dt agreed}
//     [each internal 0d time step / evaluation of the 0d right-hand side]
//               0d -> coupler -> 1d : TAG_DT {time, internal dt}
//               0d -> coupler -> 1d : TAG_DATA (all one-to-one 1d-0d connections)
//               1d -> coupler -> 0d : TAG_DATA
//     [end of the global time step]
//               0d -> coupler -> 1d : TAG_DT {time reached, -999 = COUPLING_END_OF_STEP}
//               1d -> coupler -> 0d : TAG_DT (echo)
//   The coupler checks all the times and time steps it relays (see "Time synchronisation" in coupler_mpi_common.h).
//               (if global connections, e.g. volume sum) 1d -> coupler -> 0d : TAG_VOL, their values
//   (TAG_VOL: one message with one row per global connection, see "Global connections" in coupler_mpi_common.h)
//   [end]       0d -> coupler, 1d -> coupler : TAG_DONE
//               The 0d solver must reach the end time pre_time + sim_time first: if it stops before, the 1d solver is
//               still waiting for the next time step, so the coupler aborts the job instead of waiting.

#include <iostream>
#include <algorithm>
#include <limits>
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
    std::string petscSolver;                   // PETSc time integration scheme of both solvers (with ODEsolver PETSC)
    int petscScheme = PETSC_SCHEME_NONE;       // its code, sent in the handshake (see CouplingPetscScheme)
    // Time settings of the configuration file (see coupling_end_time() in coupler_mpi_common.h)
    double T0 = 0.0;       // cardiac period: parameter T of the model
    double pre_time = 0.0; // time simulated to reach a periodic steady state (results are not saved)
    double sim_time = 0.0; // time simulated after pre_time, with the results saved
    double tSave = 0.0;    // save time of the coupled simulation, coupling_save_time(pre_time)
    double tEnd = 0.0;     // end time of the coupled simulation, coupling_end_time(pre_time, sim_time)

    size_t N1d0dTot = 0;
    size_t N1d0d = 0;
    std::string connectionsFile;    // connections file of the 0d model (<networkName>_coupler1d0d.json)
    std::vector<double> connTable;  // its connections, N1d0dTot x COUPLING_CONN_NFIELDS values (see COUPLING_CONN_FIELDS)
    AuxCouplerFun auxFun;
    size_t N1d0dGlobal = 0; // global connections (e.g. volume sum), after the one-to-one ones: N1d0dTot - N1d0d

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

    // Time synchronisation checks (see coupler_mpi_common.h): stop the coupled run with a message.
    // The exception is caught in main(), which aborts the MPI job.
    static std::string num(double x) {
        std::ostringstream s;
        s << std::setprecision(15) << x;
        return s.str();
    }
    static void sync_error(int step, const std::string& msg) {
        throw std::runtime_error("time synchronisation error in global time step " + std::to_string(step) + ": " + msg);
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
        // PETSc time integration scheme, written by the generator (petsc_solver in solver_info of the user inputs):
        // required with the PETSc ODE solver, and checked by both solvers in the handshake (see CouplingPetscScheme)
        petscSolver = "";
        petscScheme = PETSC_SCHEME_NONE;
        if (ODEsolver == "PETSC") {
            if (config.contains("PETSc_solver") && config["PETSc_solver"].is_string()) {
                petscSolver = config["PETSc_solver"].get<std::string>();
            }
            petscScheme = coupling_petsc_scheme_code(petscSolver);
            if (petscScheme < 0) {
                throw std::runtime_error("Coupler1d0d::init() : 'PETSc_solver' (CN, BDF1 or BDF2) is missing or not valid in "
                                         + initFilePath + " (needed with ODEsolver PETSC): regenerate the model");
            }
        }
        // Time settings, written by the generator (CVSCppGenerator.py): T0 is the parameter T of the model,
        // pre_time and sim_time come from the user inputs (yaml file). They are required (no default values).
        auto required_number = [&](const char* key) {
            if (!config.contains(key) || !config[key].is_number()) {
                throw std::runtime_error(std::string("Coupler1d0d::init() : '") + key + "' (a number) is missing in "
                                         + initFilePath + ": regenerate the model");
            }
            return config[key].get<double>();
        };
        T0 = required_number("T0");
        pre_time = required_number("pre_time");
        sim_time = required_number("sim_time");
        // Checked here, before the handshake, so that both solvers get a valid time window.
        if (!(std::isfinite(T0) && T0 > 0.0) || !(std::isfinite(pre_time) && pre_time >= 0.0)
            || !(std::isfinite(sim_time) && sim_time > 0.0)) {
            std::ostringstream msg;
            msg << std::setprecision(17) << "Coupler1d0d::init() : T0 > 0, pre_time >= 0 and sim_time > 0 are required in "
                << initFilePath << " (T0 = " << T0 << ", pre_time = " << pre_time << ", sim_time = " << sim_time << " given)";
            throw std::runtime_error(msg.str());
        }
        tSave = coupling_save_time(pre_time);
        tEnd = coupling_end_time(pre_time, sim_time);
        if (!(tEnd > tSave)) {
            std::ostringstream msg;
            msg << std::setprecision(17) << "Coupler1d0d::init() : sim_time = " << sim_time << " in " << initFilePath
                << " is shorter than the time tolerance " << tolTime;
            throw std::runtime_error(msg.str());
        }
        // Rounding times to multiples of tolTime only works while tolTime is larger than the spacing of
        // double precision numbers around the end time (tolTime/epsilon: about 45000 s for 1e-11).
        if (tEnd*std::numeric_limits<double>::epsilon() >= tolTime) {
            std::ostringstream msg;
            msg << "Coupler1d0d::init() : the end time pre_time + sim_time = " << tEnd << " is too large for the time tolerance "
                << tolTime << " (it must be < " << tolTime/std::numeric_limits<double>::epsilon()
                << "): increase COUPLING_TOL_TIME in coupler_mpi_common.h";
            throw std::runtime_error(msg.str());
        }

        std::string fileJson = inputFold+networkName+"_coupler1d0d.json";
        connectionsFile = fileJson;
        std::string dataJson = auxFun.readJsonFromFile(fileJson);
        if (dataJson.empty()) {
            throw std::runtime_error("Coupler1d0d::init() : could not read 1d-0d connections file " + fileJson);
        }

        std::map<std::string, std::map<std::string, int>> pipe_1d_0d_info = auxFun.deserializeFromJson(dataJson);
        N1d0dTot = pipe_1d_0d_info.size();
        N1d0d = 0;
        N1d0dGlobal = 0;
        // one-to-one connections first, then the global connections (e.g. volume sum), which have their own messages
        // (TAG_VOL): see "Global connections" in coupler_mpi_common.h
        for (size_t i = 0; i < N1d0dTot; ++i) {
            const std::string key = std::to_string(i+1);
            if (pipe_1d_0d_info.find(key) == pipe_1d_0d_info.end()) continue; // reported below
            CouplingConnectionType type;
            try {
                type = coupling_connection_type(pipe_1d_0d_info[key]);
            } catch (const std::exception& e) {
                throw std::runtime_error("Coupler1d0d::init() : connection " + key + " of " + fileJson + ": " + e.what());
            }
            if (type == COUPLING_CONN_ONE_TO_ONE) {
                if (N1d0dGlobal > 0) {
                    throw std::runtime_error("Coupler1d0d::init() : connection " + key + " of " + fileJson + " is a one-to-one "
                                             "connection after a global connection: the global connections must come last");
                }
                N1d0d++;
            } else {
                N1d0dGlobal++;
            }
        }
        // connections sent to both solvers at start-up, in the order of the file (see COUPLING_CONN_FIELDS)
        connTable.clear();
        for (size_t i = 0; i < N1d0dTot; ++i) {
            auto conn = pipe_1d_0d_info.find(std::to_string(i+1));
            if (conn == pipe_1d_0d_info.end()) {
                throw std::runtime_error("Coupler1d0d::init() : connection \"" + std::to_string(i+1) + "\" is missing in "
                                         + fileJson + " (the connections must be numbered 1, 2, ...): regenerate the model");
            }
            std::vector<double> row = coupling_connection_row(conn->second);
            connTable.insert(connTable.end(), row.begin(), row.end());
        }
        std::cout << "Coupler1d0d :: 1d-0d connections file : " << fileJson << std::endl;
        std::cout << "Coupler1d0d :: ODE solver : " << ODEsolver << (petscSolver.empty() ? "" : " (" + petscSolver + ")")
                  << " || T0 : " << T0 << " || pre_time : " << pre_time
                  << " || sim_time : " << sim_time << std::endl;
        std::cout << "Coupler1d0d :: results saved from time " << tSave << " || end time : " << tEnd
                  << " || number of cardiac cycles (pre_time + sim_time)/T0 : " << (pre_time + sim_time)/T0 << std::endl;
        std::cout << "Coupler1d0d :: MPI world ranks : 0d solver " << rank0d << " || 1d solver " << rank1d << std::endl;
        std::cout << "Number of one-to-one 1d-0d connections: " << N1d0d << " || global connections (e.g. volume sum): "
                  << N1d0dGlobal << std::endl;
        std::cout << "Total number of connections (including global connections): " << N1d0dTot << " || With time step connection: "  << N1d0dTot+1 << std::endl;
    }

    // Send the coupling layout to both solvers, which check it against their own and abort the MPI job on any mismatch.
    void handshake() {
        // tolTime, tSave and tEnd: checked by the 0d solver, adopted by the 1d solver
        // (see COUPLING_TOL_TIME, coupling_save_time() and coupling_end_time() in coupler_mpi_common.h);
        // petscScheme: checked by both solvers (see CouplingPetscScheme); N1d0dTot: number of connections (one-to-one
        // and global) in the TAG_CONN message that follows; N1d0dGlobal: rows of the TAG_VOL messages
        std::vector<double> cfg = {static_cast<double>(N1d0d), static_cast<double>(N1d0dGlobal), T0, pre_time, sim_time,
                                   tolTime, tSave, tEnd, static_cast<double>(petscScheme), static_cast<double>(N1d0dTot)};
        if (cfg.size() != COUPLING_CONFIG_LENGTH) {
            throw std::runtime_error("Coupler1d0d::handshake() : the handshake message does not have COUPLING_CONFIG_LENGTH values");
        }
        send(cfg, rank0d, TAG_CONFIG);
        send(cfg, rank1d, TAG_CONFIG);
        // then the connections of the connections file: the 0d solver checks them against the ones compiled into it,
        // the 1d solver against its own copy of the file (see COUPLING_CONN_FIELDS)
        send(connTable, rank0d, TAG_CONN);
        send(connTable, rank1d, TAG_CONN);
    }

    // Relay loop: the coupler orchestrates the communication between the 0d and 1d solvers.
    void run() {

        double dtGlob = 0.;
        double timeGlob = 0.;
        double dtLoc = 0.;
        double timeLoc = 0.;

        std::vector<double> one_buffer(N1d0d*DL); // one row per one-to-one connection
        std::vector<double> zero_buffer(N1d0d*DL); // one row per one-to-one connection
        std::vector<double> one_buffer_dt(DL);
        std::vector<double> zero_buffer_dt(DL);
        std::vector<double> one_buffer_vol(N1d0dGlobal*DL); // one row per global connection
        std::vector<double> zero_done(DL);
        std::vector<double> one_done(DL);

        int count = 0;
        int step = 0; // number of global time steps started

        // initial values of the global connections (e.g. the 1d volume, for the total volume in the system), from main1d to
        // main0d: one message, one row per global connection
        if (N1d0dGlobal > 0){
            recv(one_buffer_vol, rank1d, TAG_VOL, "the initial values of the global connections (e.g. volume sum)");
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
                    // Both solvers stop after the same time step only at the end time. If the 0d solver
                    // stopped before (e.g. after an error), the 1d solver is waiting for the next time step
                    // and would never send TAG_DONE: abort instead of waiting forever.
                    const double time0d = coupling_round_time(zero_done[0]);
                    const double timeGlobEnd = coupling_round_time(timeGlob+dtGlob);
                    if (time0d < tEnd-tolTime || timeGlobEnd < tEnd-tolTime) {
                        std::ostringstream msg;
                        msg << std::setprecision(12) << "the 0d solver stopped at time " << time0d
                            << " (global time " << timeGlobEnd << "), before the end time " << tEnd
                            << " (pre_time + sim_time = " << pre_time << " + " << sim_time << ")";
                        throw std::runtime_error(msg.str());
                    }
                    std::cout << "Coupler1d0d :: 0d solver has completed the simulation" << std::endl;
                    break;
                }
                step++;
                // read time step data from main0d and send it to main1D
                recv(zero_buffer_dt, rank0d, TAG_DT_STEP, "the global time step");
                const double time0d = zero_buffer_dt[0];
                const double dt0d = zero_buffer_dt[1];
                if (!std::isfinite(time0d) || !std::isfinite(dt0d) || !(dt0d > 0.0)) {
                    sync_error(step, "the 0d solver proposed time " + num(time0d) + " and time step " + num(dt0d));
                }
                send(zero_buffer_dt, rank1d, TAG_DT_STEP);

                // read time step data from main1D, check it, and send it back to main0d
                recv(one_buffer_dt, rank1d, TAG_DT_STEP, "the global time step");
                const double time1d = one_buffer_dt[0];
                const double dt1d = one_buffer_dt[1];
                if (!std::isfinite(time1d) || !std::isfinite(dt1d) || !(dt1d > 0.0) || dt1d > dt0d) {
                    sync_error(step, "the 1d solver replied with time " + num(time1d) + " and time step " + num(dt1d)
                                     + " (time step proposed by the 0d solver: " + num(dt0d) + ")");
                }
                if (std::abs(time1d - time0d) > tolTime) {
                    sync_error(step, "the 0d solver is at time " + num(time0d) + ", the 1d solver at time " + num(time1d));
                }
                // the 0d solver continues from the agreed time (the 1d solver's)
                send(one_buffer_dt, rank0d, TAG_DT_STEP);

                timeGlob = time1d;
                dtGlob = dt1d;
                std::cout << std::setprecision(8) << "Coupler1d0d :: Global time exchanged : " << timeGlob << " || dt : " << dtGlob << std::endl;
            }

            // this is to exchange local internal time and time step taken by the specific ODE solver between main0d and main1D
            // read time step data from main0d
            recv(zero_buffer_dt, rank0d, TAG_DT, "the internal time step");
            timeLoc = zero_buffer_dt[0];
            dtLoc = zero_buffer_dt[1];
            const double stepEnd = timeGlob + dtGlob;

            // std::cout << "Coupler :: Local time step exchanged " << timeLoc << " " << dtLoc << std::endl;

            if (!std::isfinite(timeLoc) || !std::isfinite(dtLoc)) {
                sync_error(step, "the 0d solver sent internal time " + num(timeLoc) + " and time step " + num(dtLoc));
            }

            time_step_exchanged = false;
            if (dtLoc<0.){ // dtLoc = COUPLING_END_OF_STEP: the ODE solver has completed the evolution to the next time level
                // std::cout << "Coupler :: Received negative 0d internal time step dtLoc " << dtLoc << std::endl;
                if (dtLoc != COUPLING_END_OF_STEP) {
                    sync_error(step, "the 0d solver sent a negative internal time step " + num(dtLoc) + " at time " + num(timeLoc));
                }
                // the 0d solver must have reached the end of the global time step
                if (std::abs(timeLoc - stepEnd) > tolTime) {
                    sync_error(step, "the 0d solver ended the time step at time " + num(timeLoc) + " instead of "
                                     + num(stepEnd) + " (time " + num(timeGlob) + " + time step " + num(dtGlob) + ")");
                }

                // send time step data to main1D
                send(zero_buffer_dt, rank1d, TAG_DT);
                // read time step data from main1D and send it back to main0d
                recv(one_buffer_dt, rank1d, TAG_DT, "the end of the global time step");
                send(one_buffer_dt, rank0d, TAG_DT);

                if (N1d0dGlobal > 0){
                    // values of the global connections (e.g. volume sum) from main1D, sent to main0d
                    recv(one_buffer_vol, rank1d, TAG_VOL, "the values of the global connections (e.g. volume sum)");
                    send(one_buffer_vol, rank0d, TAG_VOL);
                }

                count++;
                time_step_exchanged = true;
                continue;
            }

            // the 0d solver must evaluate its right-hand side within the global time step
            if (timeLoc < timeGlob-tolTime || timeLoc > stepEnd+tolTime) {
                sync_error(step, "the 0d solver sent internal time " + num(timeLoc) + ", outside the global time step ["
                                 + num(timeGlob) + ", " + num(stepEnd) + "]");
            }
            // if (timeLoc>timeGlob+dtGlob){
            if (coupling_round_time(timeLoc) > coupling_round_time(stepEnd)) {
                // beyond the end of the time step by round-off only (checked above): clamp it to the end
                dtLoc = std::max(0.0, dtLoc - (timeLoc-stepEnd)); // never negative, i.e. never COUPLING_END_OF_STEP
                timeLoc = stepEnd;
                timeLoc = coupling_round_time(timeLoc);
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
        if (!(std::abs(one_done[0] - zero_done[0]) <= tolTime)) {
            sync_error(step, "the 1d solver stopped at time " + num(one_done[0]) + ", the 0d solver at time " + num(zero_done[0]));
        }
        std::cout << "Coupler1d0d :: 1d solver has completed the simulation" << std::endl;

        std::cout << "### Coupler1d0d :: Stop execution! ###" << std::endl;
        std::cout << "Global final time : " << coupling_round_time(timeGlob+dtGlob) << std::endl;
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

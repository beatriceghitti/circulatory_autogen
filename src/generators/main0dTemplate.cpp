#include <iostream>
#include <string>
#include <cstring>  
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <random>
#include <thread>
#include <poll.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

// #include <petscsys.h>

#include "model0d.h"
#include "coupler_mpi_common.h"


int main(int argc, char* argv[]) {

    // Only print the MPI version (used by launch_coupled_mpi.bash), before starting MPI and the model.
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], MPI_VERSION_OPTION) == 0) {
            print_mpi_version();
            return 0;
        }
    }

    bool is_coupled = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-coupled") == 0){
            if (std::string(argv[++i]) == "1") is_coupled = true;
        }
    }

    // MPI: in a coupled run, the 0d solver is launched by mpirun together with the coupler and the 1d solver.
    MPI_Init(&argc, &argv);
    MPI_Comm comm0d = MPI_COMM_WORLD;
    int coupler_rank = -1;
    if (is_coupled) {
        // This must come first: it is collective with the coupler and the 1d solver.
        CouplingLayout layout = setup_coupling(COLOR_SOLVER0D);
        comm0d = layout.local;
        coupler_rank = layout.roots[COLOR_COUPLER];
        int size0d;
        MPI_Comm_size(comm0d, &size0d);
        if (coupler_rank < 0 || size0d != 1) {
            std::cerr << "0d solver :: ERROR: coupled mode needs the coupler in the same mpirun, "
                      << "and exactly 1 MPI rank for the 0d solver (" << size0d << " given)" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // PetscErrorCode ierr;
    // The communicator of PETSc must only contain the 0d solver, not the whole coupled MPI job.
    // PETSC_COMM_WORLD = comm0d;
    // ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    // std::cout << "0d solver :: PETSc initialization complete : " << ierr << std::endl;

    std::cout << "0d solver :: is_coupled : " << is_coupled << std::endl;
    
    std::cout << "0d solver :: input arguments" << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "arg[" << i << "]: " << std::quoted(argv[i]) << '\n';
    }

    std::string ODEsolver; // ODE solver: "explEul" / "Heun" / "midpoint" / "RK4" / "CVODE" / "PETSC"
    double T0 = -1.0; // cardiac period: parameter T of the model
    double pre_time = -1.0; // time simulated to reach a periodic steady state (results are not saved)
    double sim_time = -1.0; // time simulated after pre_time, with the results saved
    std::string modelName;
    std::string initStatePath;
    if (is_coupled) {
        // values of the coupler configuration file (coupler_config.json), passed by launch_coupled_mpi.bash
        for (int i = 1; i < argc; i++) {
            if (std::strcmp(argv[i], "-ODEsolver") == 0) ODEsolver = std::string(argv[++i]);
            else if (std::strcmp(argv[i], "-T0") == 0) T0 = std::stod(std::string(argv[++i]));
            else if (std::strcmp(argv[i], "-preTime") == 0) pre_time = std::stod(std::string(argv[++i]));
            else if (std::strcmp(argv[i], "-simTime") == 0) sim_time = std::stod(std::string(argv[++i]));
            else if (std::strcmp(argv[i], "-networkName") == 0) modelName = std::string(argv[++i]);
            else if (std::strcmp(argv[i], "-initStatePath") == 0) initStatePath = std::string(argv[++i]);
        }
        if (!(T0 > 0.0) || !(pre_time >= 0.0) || !(sim_time > 0.0)) {
            std::cerr << "0d solver :: ERROR: -T0 > 0, -preTime >= 0 and -simTime > 0 are required in a coupled run (T0 = "
                      << T0 << ", pre_time = " << pre_time << ", sim_time = " << sim_time << " given)" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else{
        // values of the user inputs at generation (replaced by CVSCppGenerator.py)
        ODEsolver = "CVODE"; // set at generation
        T0 = 1.0; // set at generation
        pre_time = 0.0; // set at generation
        sim_time = 1.0; // set at generation
        modelName = "model0d"; // set at generation
        initStatePath = "None";
    }
    std::cout << "0d solver :: ODE solver : " << ODEsolver << std::endl;
    std::cout << "0d solver :: T0 : " << T0 << " || pre_time : " << pre_time << " || sim_time : " << sim_time << std::endl;
 
    // initialize 0d model.
    Model0d* model0d;
    model0d = new Model0d();

    std::string initStatesFile;
    std::string initVarsFile;
    if (initStatePath != "None"){
        initStatesFile = initStatePath + "sol0D_states.txt";
        initVarsFile = initStatePath + "sol0D_variables.txt";
    } else{
        initStatesFile = "None";
        initVarsFile = "None";
    }
    model0d->initStatesPath = initStatesFile.c_str();
    model0d->initVarsPath = initVarsFile.c_str();

    // set and initialize the ODE solver.
    model0d->set_ode_solver(ODEsolver);
    
    // open output file(s).
    std::string resFold = "./";
    model0d->openOutputFiles(resFold);
    
    // int N1d0d = 0;
    // int N1d0dTot = 0;
    // set up the MPI communication with the coupler for 1d-0d coupling.
    if (is_coupled) {
        // N1d0d = model0d->N1d0d;
        // N1d0dTot = model0d->N1d0dTot;
        int status = model0d->initCoupling(coupler_rank, T0, pre_time, sim_time);
        if (status==1){
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    const double tolTime = COUPLING_TOL_TIME; // shared with the coupler, see coupler_mpi_common.h
    const double tIni = 0.0;
    const double dtSample = 1e-3;
    double tEnd = dtSample;
    const double tEndGlob = coupling_end_time(pre_time, sim_time); // same end time as the coupler, see coupler_mpi_common.h
    const double dt0D = 1e-4;
    const double tSaveRes = coupling_save_time(pre_time); // same save time as the coupler, see coupler_mpi_common.h
    // pre_time and sim_time must be multiples of the output time step (checked at generation, and here again because
    // the coupler configuration file can be edited), so that the results are saved exactly from pre_time to the end time
    for (double t : {pre_time, sim_time}) {
        if (!(std::abs(t - std::round(t/dtSample)*dtSample) <= tolTime)) {
            std::cerr << "0d solver :: ERROR: pre_time = " << pre_time << " and sim_time = " << sim_time
                      << " must be multiples of the output time step dt = " << dtSample << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    int saveResTime = 0;
    
    model0d->voi = tIni;
    model0d->dt = dt0D;
    // initialize variables and compute constants of 0d model.
    model0d->initialiseVariablesAndComputeConstants();

    bool run_0d = true;
    int it = 0;
    double time;
    double dt;
    time = tIni;
    dt = dt0D;
    
    if (std::abs(tIni-tSaveRes)<=tolTime) {
        model0d->writeOutput(time);
    }

    std::cout << "0d solver :: Initialization successful at initial time : " << time << std::endl;
    std::cout << " " << std::endl;

    // enter time loop.
    while (run_0d==true) {
        try {
            if (!is_coupled) {
                dt = dt0D;
                if (time+dt>tEnd) {
                    dt = tEnd-time;
                }
                //std::cout << "0d solver :: IT : " << it << std::setprecision(8) << " || time : " << time << " || dt : " << dt << std::endl;
                model0d->solveOneStep(dt);
            }
            else {  
                model0d->solveOneStep(dt0D);     
                dt = model0d->dt;          
            }

            it++;
            time = model0d->voi; // time already rounded within model0d->solveOneStepCVODE() / model0d->solveOneStepExpl() function
            // time = std::round(time / tolTime) * tolTime;

            saveResTime = 0;
            // if (std::abs(time-tEnd)<=tolTime) {
            if (time>=tEnd-tolTime) {
                tEnd = tEnd + dtSample;
                tEnd = coupling_round_time(tEnd);
                saveResTime = 1;
            }
            if (saveResTime) {
                if (time >= tSaveRes-tolTime) {
                    model0d->writeOutput(time);
                }
            }
            
            if (time>=tEndGlob-tolTime) {
            // if (std::abs(time-tEndGlob)<=tolTime) {
            // if (time>=tEndGlob) {
                std::cout << "### 0d solver :: Stop execution! ###" << std::endl;
                run_0d = false;
            }

        } catch (const std::exception& e) {
            std::cerr << "0d solver :: ERROR: " << e.what() << std::endl;
            break;
        }
    }

    // close output file(s).
    model0d->closeOutputFiles();
    // tell the coupler that the 0d solver has completed its time loop (this replaces closing the pipes).
    if (is_coupled) {
        model0d->finaliseCoupling();
    }
    // free the memory.
    delete model0d;
    std::cout << "### 0d solver :: Object model0d deleted ###" << std::endl;
    // ierr = PetscFinalize();
    // std::cout << "### 0d solver :: PETSc finalization complete : " << ierr << " ###" << std::endl;

    if (comm0d != MPI_COMM_WORLD) {
        MPI_Comm_free(&comm0d);
    }
    MPI_Finalize();

    return 0;
}

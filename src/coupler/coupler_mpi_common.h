// coupler_mpi_common.h
// Shared definitions for the MPI-based 1d-0d coupling.
//
// Used by:
//   - the coupler (src/coupler/coupler.cpp);
//   - the generated 0d solver (main0d.cpp / model0d.cc), which gets a copy of this file
//     in its generated folder (see CVSCppGenerator.py);
//   - the Python 1d solver: src/solver1d/auxMPIFunctions.py duplicates the constants below.
//     KEEP THEM IN SYNC.
//
// The three programs are started together by a single mpirun (MPMD launch), e.g.
//   mpirun -np 1 ./coupler config.json : -np 1 ./main0d -coupled 1 ... : -np 1 python main1D.py 1 ...
// (see src/coupler/launch_coupled_mpi.bash).
// Every message goes through the coupler: the 0d and 1d solvers never talk to each other directly.

#pragma once

#include <mpi.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>

// Each program identifies itself with a fixed "color". Using fixed colors (rather than
// MPI_APPNUM) means that the order of the programs on the mpirun line does not matter.
enum CouplingColor {
    COLOR_COUPLER  = 0,
    COLOR_SOLVER0D = 1,
    COLOR_SOLVER1D = 2,
    NUM_COLORS     = 3
};

// Message tags. The direction of a message is given by its source and destination,
// so the same tag is used both ways (e.g. TAG_DT: 0d -> coupler and coupler -> 0d).
// The time step messages at the start of a global time step (TAG_DT_STEP) have their own tag,
// distinct from the ones within the time step (TAG_DT), so that the coupler detects (and aborts)
// a solver that starts a new time step without completing the previous one, instead of deadlocking.
enum CouplingTag {
    TAG_CONFIG  = 1,   // coupler -> 0d/1d : handshake, COUPLING_CONFIG_LENGTH doubles
                       //                    {N1d0d, couple_volume_sum, T0, pre_time, sim_time, COUPLING_TOL_TIME,
                       //                     coupling_save_time(pre_time), coupling_end_time(pre_time, sim_time)}
    TAG_DONE    = 2,   // 0d/1d -> coupler : the solver has completed its time loop {final time, 0}
                       //                    (the coupler aborts the job if the 0d solver stops before the end time)
    TAG_DT      = 10,  // 0d <-> coupler <-> 1d : {time, time step} within a global time step:
                       //                         internal 0d time step, or {time reached, COUPLING_END_OF_STEP} at its end
    TAG_DT_STEP = 11,  // 0d <-> coupler <-> 1d : {time, time step} at the start of a global time step
    TAG_DATA    = 20,  // 0d <-> coupler <-> 1d : N1d0d x COUPLING_DATA_LENGTH values (row-major),
                       //                         one row per one-to-one 1d-0d connection
    TAG_VOL     = 30   // 1d -> coupler -> 0d : {1d volume sum, 0}
};

constexpr int COUPLING_DATA_LENGTH   = 2;  // doubles per time/time step message and per 1d-0d connection
constexpr int COUPLING_CONFIG_LENGTH = 8;  // doubles in the TAG_CONFIG handshake message
constexpr double COUPLING_END_OF_STEP = -999.0; // time step value of the TAG_DT message that ends a global time step

// Time synchronisation, checked by the coupler (which aborts the job on any mismatch):
// - at the start of each global time step, the 0d solver proposes {its time, its time step}, and the
//   1d solver replies with {its time, agreed time step}: the two times must agree within COUPLING_TOL_TIME,
//   and the agreed time step must be > 0 and not larger than the proposed one. The 0d solver then
//   continues from the agreed time, so that round-off differences between the solvers cannot accumulate;
// - within the global time step, the 0d solver evaluates its right-hand side only at times in [time, time + dt];
// - at the end of the global time step, the 0d solver must have reached time + dt;
// - at the end of the simulation, both solvers must stop at the same time (see coupling_end_time()).

// Tolerance used to round and compare times. The coupler and both solvers must use the same
// value, because the coupler rounds and clamps the 0d times with it.
// - The generated coupled 0d model (main0d.cpp, model0d.cc) uses this constant, and checks in
//   the handshake that the coupler has the same value.
// - The 1d solver receives it from the coupler in the handshake (Python cannot include this
//   header), and uses it instead of tolTime from its input.ini.
// A fully 0d model (no coupler, so no MPI and no copy of this header) uses TOL_TIME from
// CVSCppGenerator.py instead, and a 1d-only run uses tolTime from its input.ini.
constexpr double COUPLING_TOL_TIME = 1e-11; // 1e-13

// Inverse of COUPLING_TOL_TIME, rounded to an integer (1e11 for 1e-11), used to round times.
inline const double COUPLING_INV_TOL_TIME = std::round(1.0 / COUPLING_TOL_TIME);

// Round a time to the grid of COUPLING_TOL_TIME. All the times exchanged by the coupler and the solvers
// are rounded with this function, which gives exactly the same result (bit for bit) in:
// - the coupler and the coupled 0d solver (this function; Model0d::roundTime() calls it);
// - the 1d solver: round_time() in auxMPIFunctions.py, with invTolTime = inv_tol_time(tolTime).
// KEEP THEM IN SYNC. Two details make the C++ and Python results identical:
// - halves are rounded away from zero, as std::round does (Python's round() rounds them to even);
// - the time is multiplied and divided by the integer COUPLING_INV_TOL_TIME (exact in floating point),
//   rather than divided and multiplied by COUPLING_TOL_TIME (inexact), so that e.g. 0.003 stays 0.003.
inline double coupling_round_time(double t)
{
    return std::round(t * COUPLING_INV_TOL_TIME) / COUPLING_INV_TOL_TIME;
}

// Time window of a coupled simulation, set by pre_time and sim_time of the user inputs (yaml file), as for the
// other model types of circulatory_autogen: the solvers run from 0 to the end time pre_time + sim_time, and save
// their results from the save time pre_time. Both times are rounded with COUPLING_TOL_TIME.
// They are computed by the coupler and the 0d solver, which checks in the handshake that the coupler has the same
// values. The 1d solver receives them from the coupler in the handshake (like COUPLING_TOL_TIME), so that all
// three programs use exactly the same times. Each solver:
// - saves its results at the output times >= save time - COUPLING_TOL_TIME;
// - stops after the time step that reaches the end time: time >= end time - COUPLING_TOL_TIME
//   (the coupler aborts the job if the 0d solver stops before).
// The cardiac period T0 (parameter T of the model) does not set the time window: the 1d solver uses it as the
// period of its prescribed boundary conditions (e.g. the inflow), and the number of cardiac cycles simulated is
// (pre_time + sim_time)/T0.
inline double coupling_save_time(double pre_time)
{
    return coupling_round_time(pre_time);
}

inline double coupling_end_time(double pre_time, double sim_time)
{
    return coupling_round_time(pre_time + sim_time);
}

struct CouplingLayout {
    MPI_Comm local;          // communicator containing only this program's ranks
    int roots[NUM_COLORS];   // world rank of each program's rank 0 (-1 if absent): array with one entry per program, indexed by colour
};

// Collective over MPI_COMM_WORLD: every rank of every program must call this, right after
// MPI_Init (the Python side makes the same two calls, in the same order).
inline CouplingLayout setup_coupling(int my_color)
{
    CouplingLayout L;

    // 1) Split the world into one communicator per program.
    MPI_Comm_split(MPI_COMM_WORLD, my_color, 0, &L.local);

    // 2) Share everyone's color so each program can find the others' roots.
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    std::vector<int> colors(world_size);
    MPI_Allgather(&my_color, 1, MPI_INT, colors.data(), 1, MPI_INT, MPI_COMM_WORLD); // every rank sends its colour, and every rank receives the full list

    // Fill the roots array with the world rank of each program's rank 0
    for (int c = 0; c < NUM_COLORS; ++c) L.roots[c] = -1; // First, set all entries to -1
    for (int r = world_size - 1; r >= 0; --r)
        // Then, loop from the highest rank down to 0: the last write for each colour comes from its lowest rank (lowest rank of each color wins)
        if (colors[r] >= 0 && colors[r] < NUM_COLORS) L.roots[colors[r]] = r;

    return L;
}

// Command-line option of the coupler and of main0d: print the MPI version and exit, without starting MPI.
// src/coupler/launch_coupled_mpi.bash uses it to check, before launching a coupled run, that all the
// programs use the same MPI. It must be handled before MPI_Init.
constexpr const char* MPI_VERSION_OPTION = "--mpi-version";

// Print the MPI this program was built with (mpi.h at compile time) and the MPI library it runs with
// (loaded at run time), one per line. MPI_Get_library_version may be called before MPI_Init.
inline void print_mpi_version()
{
#if defined(OMPI_MAJOR_VERSION)
    std::printf("built with: Open MPI %d.%d.%d\n", OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION, OMPI_RELEASE_VERSION);
#elif defined(MPICH_VERSION)
    std::printf("built with: MPICH %s\n", MPICH_VERSION);
#else
    std::printf("built with: unknown MPI implementation (MPI standard %d.%d)\n", MPI_VERSION, MPI_SUBVERSION);
#endif
    char version[MPI_MAX_LIBRARY_VERSION_STRING];
    int length = 0;
    MPI_Get_library_version(version, &length);
    version[std::strcspn(version, "\n")] = '\0';   // first line only
    std::printf("runs with: %s\n", version);
}

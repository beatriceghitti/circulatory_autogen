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
                       //                    {N1d0d, couple_volume_sum, T0, nCC, COUPLING_TOL_TIME}
    TAG_DONE    = 2,   // 0d/1d -> coupler : the solver has completed its time loop {final time, 0}
    TAG_DT      = 10,  // 0d <-> coupler <-> 1d : {time, time step} within a global time step:
                       //                         internal 0d time step, or {time, -999} at its end
    TAG_DT_STEP = 11,  // 0d <-> coupler <-> 1d : {time, time step} at the start of a global time step
    TAG_DATA    = 20,  // 0d <-> coupler <-> 1d : N1d0d x COUPLING_DATA_LENGTH values (row-major),
                       //                         one row per one-to-one 1d-0d connection
    TAG_VOL     = 30   // 1d -> coupler -> 0d : {1d volume sum, 0}
};

constexpr int COUPLING_DATA_LENGTH   = 2;  // doubles per time/time step message and per 1d-0d connection
constexpr int COUPLING_CONFIG_LENGTH = 5;  // doubles in the TAG_CONFIG handshake message

// Tolerance used to round and compare times. The coupler and both solvers must use the same
// value, because the coupler rounds and clamps the 0d times with it.
// - The generated coupled 0d model (main0d.cpp, model0d.cc) uses this constant, and checks in
//   the handshake that the coupler has the same value.
// - The 1d solver receives it from the coupler in the handshake (Python cannot include this
//   header), and uses it instead of tolTime from its input.ini.
// A fully 0d model (no coupler, so no MPI and no copy of this header) uses TOL_TIME from
// CVSCppGenerator.py instead, and a 1d-only run uses tolTime from its input.ini.
constexpr double COUPLING_TOL_TIME = 1e-11; // 1e-13

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

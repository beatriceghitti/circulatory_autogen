#!/bin/bash

### safer to specify absolute paths only
# FOLDERcoupler="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
FOLDERcoupler=""
FOLDERcpp=""
FILEconfig="coupler_config.json"
USE_PETSC=0

### The coupler, the 0d solver and the 1d solver communicate through MPI: they must all use the
### same MPI installation (mpic++/mpicxx, mpirun, and the mpi4py of the Python set in $FILEconfig).
### On a cluster, load the MPI module first, e.g. 'module load mpi/openmpi-x86_64'.

# Stop with a FAILED message and an exit code (1 by default): SUCCESS is printed only if every step succeeded.
fail() {
    echo "*** FAILED $(date +'%Y-%m-%d_%H-%M-%S'): $1 ***" >&2
    exit "${2:-1}"
}

cd "$FOLDERcpp" || fail "cannot enter the 0d solver folder $FOLDERcpp"

if [[ "$USE_PETSC" -eq 1 ]]; then
    make -f MakefilePETSC clean
    make -f MakefilePETSC || fail "the build of the 0d solver failed"
else
    make -f Makefile clean
    make -f Makefile || fail "the build of the 0d solver failed"
    export LD_LIBRARY_PATH=$(spack location -i sundials)/lib:$LD_LIBRARY_PATH
    # export LD_LIBRARY_PATH=$(spack location -i sundials~mpi)/lib:$LD_LIBRARY_PATH
fi


cd "$FOLDERcoupler" || fail "cannot enter the coupler folder $FOLDERcoupler"

make -f MakefileCoupler clean
make -f MakefileCoupler || fail "the build of the coupler failed"


echo "*** RUNNING THE COUPLER NOW ***"

### launch coupler, 0d solver and 1d solver together with mpirun (1 MPI rank each).
### Extra mpirun options can be appended, e.g. --oversubscribe if fewer than 3 cores/slots are available.
### With one of the lines ending with '&', the simulation runs in the background: this script does not wait for it,
### so it cannot check it, and prints STARTED instead of SUCCESS (check the end of the log).
./launch_coupled_mpi.bash "$FOLDERcpp/$FILEconfig"
# ./launch_coupled_mpi.bash "$FOLDERcpp/$FILEconfig" > log_$(date +'%Y-%m-%d_%H-%M-%S').txt
# ./launch_coupled_mpi.bash "$FOLDERcpp/$FILEconfig" > log_$(date +'%Y-%m-%d_%H-%M-%S').txt &
# ./launch_coupled_mpi.bash "$FOLDERcpp/$FILEconfig" > log_$(date +'%Y-%m-%d_%H-%M-%S').txt 2>&1 &
STATUS=$?

if [[ -n "$!" ]]; then
    echo "*** STARTED IN THE BACKGROUND (process $!) $(date +'%Y-%m-%d_%H-%M-%S') ***"
    exit 0
fi
if [[ $STATUS -ne 0 ]]; then
    fail "the coupled simulation stopped with exit code $STATUS" "$STATUS"
fi
echo "*** SUCCESS $(date +'%Y-%m-%d_%H-%M-%S') ***"

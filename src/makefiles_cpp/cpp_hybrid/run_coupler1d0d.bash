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

cd "$FOLDERcpp" || exit 1

if [[ "$USE_PETSC" -eq 1 ]]; then
    make -f MakefilePETSC clean
    make -f MakefilePETSC
else
    make -f Makefile clean
    make -f Makefile
    export LD_LIBRARY_PATH=$(spack location -i sundials)/lib:$LD_LIBRARY_PATH
    # export LD_LIBRARY_PATH=$(spack location -i sundials~mpi)/lib:$LD_LIBRARY_PATH
fi


cd "$FOLDERcoupler" || exit 1

make -f MakefileCoupler clean
make -f MakefileCoupler


echo "*** RUNNING THE COUPLER NOW ***"

### launch coupler, 0d solver and 1d solver together with mpirun (1 MPI rank each).
### Extra mpirun options can be appended, e.g. --oversubscribe if fewer than 3 cores/slots are available.
./launch_coupled_mpi.bash "$FOLDERcpp/$FILEconfig"
# ./launch_coupled_mpi.bash "$FOLDERcpp/$FILEconfig" > log_$(date +'%Y-%m-%d_%H-%M-%S').txt
# ./launch_coupled_mpi.bash "$FOLDERcpp/$FILEconfig" > log_$(date +'%Y-%m-%d_%H-%M-%S').txt &
# ./launch_coupled_mpi.bash "$FOLDERcpp/$FILEconfig" > log_$(date +'%Y-%m-%d_%H-%M-%S').txt 2>&1 &

echo "*** SUCCESS $(date +'%Y-%m-%d_%H-%M-%S') ***"

#!/bin/bash 

### safer to specify absolute paths only
# FOLDERcoupler="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
FOLDERcoupler=""
FOLDERpipes=""
mkdir -p "$FOLDERpipes"
FOLDERcpp=""
FILEconfig="coupler_config.json"
USE_PETSC=0

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
make -f MakefileCoupler clean_pipe PIPE_DIR="$FOLDERpipes"
make -f MakefileCoupler


echo "*** RUNNING THE COUPLER NOW ***"

./coupler "$FOLDERcpp/$FILEconfig"
# ./coupler "$FOLDERcpp/$FILEconfig" > log_$(date +'%Y-%m-%d_%H-%M-%S').txt
# ./coupler "$FOLDERcpp/$FILEconfig" > log_$(date +'%Y-%m-%d_%H-%M-%S').txt & 
# ./coupler "$FOLDERcpp/$FILEconfig" > log_$(date +'%Y-%m-%d_%H-%M-%S').txt 2>&1 &

### clean up when done
rm -rf "$FOLDERpipes"

echo "*** SUCCESS $(date +'%Y-%m-%d_%H-%M-%S') ***"

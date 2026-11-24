#!/bin/bash

# Stop with a FAILED message and an exit code (1 by default): SUCCESS is printed only if every step succeeded.
fail() {
    echo "*** FAILED $(date +'%Y-%m-%d_%H-%M-%S'): $1 ***" >&2
    exit "${2:-1}"
}

# spack load sundials
make clean
make || fail "the build of the 0d solver failed"

export LD_LIBRARY_PATH=$(spack location -i sundials)/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$(spack location -i sundials~mpi)/lib:$LD_LIBRARY_PATH

### With one of the lines ending with '&', the simulation runs in the background: this script does not wait for it,
### so it cannot check it, and prints STARTED instead of SUCCESS (check the end of the log).
./model0d
# ./model0d > log_$(date +'%Y-%m-%d_%H-%M-%S').txt
# ./model0d > log_$(date +'%Y-%m-%d_%H-%M-%S').txt &
# ./model0d > log_$(date +'%Y-%m-%d_%H-%M-%S').txt 2>&1 &
STATUS=$?

if [[ -n "$!" ]]; then
    echo "*** STARTED IN THE BACKGROUND (process $!) $(date +'%Y-%m-%d_%H-%M-%S') ***"
    exit 0
fi
if [[ $STATUS -ne 0 ]]; then
    fail "the 0d simulation stopped with exit code $STATUS" "$STATUS"
fi
echo "*** SUCCESS $(date +'%Y-%m-%d_%H-%M-%S') ***"

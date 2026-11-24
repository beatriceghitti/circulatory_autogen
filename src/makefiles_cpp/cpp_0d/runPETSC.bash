#!/bin/bash

# Stop with a FAILED message and an exit code (1 by default): SUCCESS is printed only if every step succeeded.
fail() {
    echo "*** FAILED $(date +'%Y-%m-%d_%H-%M-%S'): $1 ***" >&2
    exit "${2:-1}"
}

make -f MakefilePETSC clean
make -f MakefilePETSC || fail "the build of the 0d solver failed"

### With one of the lines ending with '&', the simulation runs in the background: this script does not wait for it,
### so it cannot check it, and prints STARTED instead of SUCCESS (check the end of the log).
mpirun -np 1 ./model0d -log_view
# mpirun -np 1 ./model0d -log_view > log_$(date +'%Y-%m-%d_%H-%M-%S').txt
# mpirun -np 1 ./model0d -log_view > log_$(date +'%Y-%m-%d_%H-%M-%S').txt &
# mpirun -np 1 ./model0d -log_view > log_$(date +'%Y-%m-%d_%H-%M-%S').txt 2>&1 &
STATUS=$?

if [[ -n "$!" ]]; then
    echo "*** STARTED IN THE BACKGROUND (process $!) $(date +'%Y-%m-%d_%H-%M-%S') ***"
    exit 0
fi
if [[ $STATUS -ne 0 ]]; then
    fail "the 0d simulation stopped with exit code $STATUS" "$STATUS"
fi
echo "*** SUCCESS $(date +'%Y-%m-%d_%H-%M-%S') ***"

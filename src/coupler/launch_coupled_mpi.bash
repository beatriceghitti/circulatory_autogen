#!/bin/bash
# Launch a coupled 1d-0d simulation as a single MPMD MPI job:
#   1 rank for the coupler : 1 rank for the 0d solver (main0d) : 1 rank for the 1d solver (main1D.py)
#
# The arguments of the two solvers are read from the coupler configuration file (the same file that the coupler reads).
# At start-up, the coupler also sends its view of the coupling to both solvers (handshake),
# and the whole job is aborted if they disagree.
#
# Usage:
#   launch_coupled_mpi.bash <coupler_config.json> [extra mpirun options...]
# e.g.
#   launch_coupled_mpi.bash /path/to/coupler_config.json
#   launch_coupled_mpi.bash /path/to/coupler_config.json --oversubscribe   # if fewer than 3 slots/cores
#
# Optional environment variables:
#   MPIRUN      MPI launcher (default: mpirun). It must belong to the same MPI installation used
#               to build the coupler and main0d, and by mpi4py in the Python of the 1d solver.
#   COUPLER     coupler executable (default: the 'coupler' next to this script)
#   PYTHON_CFG  Python used here only to read the JSON configuration file (default: python3)

# Make the script stop at the first error.
set -euo pipefail

# Read the script arguments: the first one is the coupler config file, and anything after it is passed to mpirun.
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <coupler_config.json> [extra mpirun options...]" >&2
    exit 1
fi
# The first argument is the coupler config file.
CONFIG="$1"
# Drop the first argument and move the others down by one.
shift

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MPIRUN="${MPIRUN:-mpirun}"
COUPLER="${COUPLER:-$SCRIPT_DIR/coupler}"
PYTHON_CFG="${PYTHON_CFG:-python3}"

if [[ ! -f "$CONFIG" ]]; then
    echo "launch_coupled_mpi :: Error: configuration file $CONFIG not found" >&2
    exit 1
fi

# Print the values needed to launch the solvers, one per line.
# Defaults for ODEsolver, T0, nCC and networkName are the same as in coupler.cpp.
read_config() {
    "$PYTHON_CFG" - "$CONFIG" <<'PYEOF'
import json, sys
path = sys.argv[1]
with open(path) as f:
    config = json.load(f)
def get(key, default=None):
    if key in config:
        return config[key]
    if default is None:
        sys.exit(f"launch_coupled_mpi :: Error: key '{key}' missing in {path}")
    return default
values = [
    get("python_path"),
    get("solver1d_path"),
    get("solver0d_path"),
    get("initFile_sim1d_path"),
    get("ODEsolver", "CVODE"),
    repr(float(get("T0", 1.1))),   # full precision: the coupler checks that the solvers use the same T0
    str(int(get("nCC", 5))),
    get("networkName", "aortic_bif_hybrid"),
    get("initStatePath", "None"),
]
print("\n".join(str(v) for v in values))
PYEOF
}

readarray -t V < <(read_config)
if [[ ${#V[@]} -ne 9 ]]; then
    echo "launch_coupled_mpi :: Error: could not read $CONFIG" >&2
    exit 1
fi
PYTHON="${V[0]}"
SOLVER1D="${V[1]}"
SOLVER0D="${V[2]}"
INITFILE1D="${V[3]}"
ODESOLVER="${V[4]}"
T0="${V[5]}"
NCC="${V[6]}"
NETWORK="${V[7]}"
INITSTATE="${V[8]}"

for f in "$COUPLER" "$SOLVER0D" "$PYTHON"; do
    if [[ ! -x "$f" ]]; then
        echo "launch_coupled_mpi :: Error: $f not found or not executable" >&2
        exit 1
    fi
done

echo "launch_coupled_mpi :: coupler   : $COUPLER $CONFIG"
echo "launch_coupled_mpi :: 0d solver : $SOLVER0D"
echo "launch_coupled_mpi :: 1d solver : $PYTHON $SOLVER1D"
echo "launch_coupled_mpi :: ODEsolver=$ODESOLVER T0=$T0 nCC=$NCC networkName=$NETWORK initStatePath=$INITSTATE"

# python -u: unbuffered output, so that the 1d solver's log interleaves with the others.
# $@ : all remaining arguments are passed to mpirun.
exec "$MPIRUN" "$@" \
    -np 1 "$COUPLER" "$CONFIG" \
  : -np 1 "$SOLVER0D" -coupled 1 -ODEsolver "$ODESOLVER" -T0 "$T0" -nCC "$NCC" \
        -networkName "$NETWORK" -initStatePath "$INITSTATE" \
  : -np 1 "$PYTHON" -u "$SOLVER1D" 1 "$ODESOLVER" "$T0" "$NCC" "$INITFILE1D" "$INITSTATE"

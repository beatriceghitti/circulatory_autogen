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
#   SKIP_MPI_CHECK  0 (default): before launching, check that mpirun, the coupler, the 0d solver and
#               the mpi4py of the 1d solver all use the same MPI (implementation and version), and stop
#               with an error otherwise. 1: skip this check.

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

# ---------------------------------------------------------------------------------------------------
# MPI check: all the programs must use the same MPI (implementation and version). With different
# MPIs, the job may hang or crash in MPI_Init, or each program may end up alone in its own MPI job,
# so this is checked here, before mpirun.

MPI_VERSION_OPTION="--mpi-version"  # same as MPI_VERSION_OPTION in coupler_mpi_common.h

# Reduce an MPI version text to "<implementation> <version>", e.g. "Open MPI 4.1.6".
mpi_id() {
    local text="$1"
    local re_ompi='Open MPI\)? v?([0-9]+(\.[0-9]+)*)'
    local re_mpich='(Version:|MPICH)[[:space:]]+v?([0-9]+(\.[0-9]+)*)'
    if [[ "$text" =~ $re_ompi ]]; then
        echo "Open MPI ${BASH_REMATCH[1]}"
    elif [[ "$text" =~ (MPICH|HYDRA) ]] && [[ "$text" =~ $re_mpich ]]; then
        echo "MPICH ${BASH_REMATCH[2]}"
    else
        echo "unknown: ${text%%$'\n'*}"
    fi
}

# Run a command with a time limit, if the 'timeout' command is available.
run_limited() {
    if command -v timeout > /dev/null; then
        timeout 60 "$@"
    else
        "$@"
    fi
}

mpi_check_error() {
    echo "launch_coupled_mpi :: Error: $1" >&2
    echo "launch_coupled_mpi :: (set SKIP_MPI_CHECK=1 to skip the MPI check)" >&2
    exit 1
}

MPI_NAMES=()
MPI_TEXTS=()
add_mpi_entry() {
    MPI_NAMES+=("$1")
    MPI_TEXTS+=("$2")
}

check_mpi_versions() {
    local out exe label built runs libs

    # 1) launcher
    out="$(run_limited "$MPIRUN" --version 2>&1 || true)"
    [[ -n "$out" ]] || mpi_check_error "could not get the version of $MPIRUN"
    add_mpi_entry "mpirun" "$out"

    # 2) coupler and 0d solver: the MPI they were built with, and the MPI library they run with
    for label in coupler "0d solver"; do
        if [[ "$label" == coupler ]]; then exe="$COUPLER"; else exe="$SOLVER0D"; fi
        # do not run an older executable that does not know the option (main0d would start a simulation)
        if ! grep -aqF -e "$MPI_VERSION_OPTION" "$exe"; then
            mpi_check_error "$exe does not support $MPI_VERSION_OPTION: rebuild it from the current sources (for main0d, regenerate the model)"
        fi
        out="$(run_limited "$exe" "$MPI_VERSION_OPTION" < /dev/null 2>&1 || true)"
        built="$(sed -n 's/^built with: //p' <<< "$out")"
        runs="$(sed -n 's/^runs with: //p' <<< "$out")"
        if [[ -z "$built" || -z "$runs" ]]; then
            mpi_check_error "could not get the MPI version of $exe:"$'\n'"$out"
        fi
        add_mpi_entry "$label (built with)" "$built"
        add_mpi_entry "$label (runs with)" "$runs"

        # more than one MPI library linked, e.g. PETSc built with another MPI than main0d
        libs="$( (ldd "$exe" 2> /dev/null || true) | awk '$1 ~ /^lib(mpi|mpich)\.so/ {print $1}' | sort -u)"
        if [[ $(grep -c . <<< "$libs") -gt 1 ]]; then
            mpi_check_error "$exe is linked to more than one MPI library:"$'\n'"$libs"
        fi
    done

    # 3) 1d solver: the MPI library of mpi4py (without starting MPI)
    out="$(run_limited "$PYTHON" -c '
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
version = MPI.Get_library_version().replace("\0", "").strip()  # may end with a null character
print("runs with: " + version.splitlines()[0])
' 2>&1 || true)"
    runs="$(sed -n 's/^runs with: //p' <<< "$out")"
    if [[ -z "$runs" ]]; then
        mpi_check_error "could not get the MPI version of mpi4py in $PYTHON:"$'\n'"$out"
    fi
    add_mpi_entry "1d solver (mpi4py)" "$runs"

    # compare
    local i ref mismatch=0
    ref="$(mpi_id "${MPI_TEXTS[0]}")"
    for i in "${!MPI_NAMES[@]}"; do
        [[ "$(mpi_id "${MPI_TEXTS[$i]}")" == "$ref" ]] || mismatch=1
    done

    if [[ $mismatch -eq 0 ]]; then
        echo "launch_coupled_mpi :: MPI check passed, all programs use $ref:"
        for i in "${!MPI_NAMES[@]}"; do
            printf "    %-22s %s\n" "${MPI_NAMES[$i]}" "$(mpi_id "${MPI_TEXTS[$i]}")"
        done
    else
        {
            echo "launch_coupled_mpi :: Error: the programs do not all use the same MPI:"
            for i in "${!MPI_NAMES[@]}"; do
                printf "    %-22s %-18s [%s]\n" "${MPI_NAMES[$i]}" "$(mpi_id "${MPI_TEXTS[$i]}")" "${MPI_TEXTS[$i]%%$'\n'*}"
            done
            echo "    mpirun     : $(command -v "$MPIRUN" || echo "$MPIRUN")"
            echo "    coupler    : $COUPLER"
            echo "    0d solver  : $SOLVER0D"
            echo "    1d solver  : $PYTHON"
            echo "  Load the same MPI for all of them (e.g. the same 'module load'), rebuild the coupler and main0d,"
            echo "  and use a Python whose mpi4py is built with that MPI."
        } >&2
        mpi_check_error "MPI mismatch"
    fi
}

SKIP_MPI_CHECK="${SKIP_MPI_CHECK:-0}"
case "$SKIP_MPI_CHECK" in
    0) check_mpi_versions ;;
    1) echo "launch_coupled_mpi :: MPI check skipped (SKIP_MPI_CHECK=1)" ;;
    *) echo "launch_coupled_mpi :: Error: SKIP_MPI_CHECK must be 0 or 1, not '$SKIP_MPI_CHECK'" >&2; exit 1 ;;
esac
# ---------------------------------------------------------------------------------------------------

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

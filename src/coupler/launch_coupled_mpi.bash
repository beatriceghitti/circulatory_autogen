#!/bin/bash
# Launch a coupled 1d-0d simulation as a single MPMD MPI job:
#   1 rank for the coupler : 1 rank for the 0d solver (main0d) : 1 rank for the 1d solver (run1DBFmodel.py, the Python
#   1d solver of the configuration file: solver1d_path)
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
#   SKIP_MPI_CHECK  0 (default): before launching, check that mpirun, the coupler, the 0d solver (and the
#               PETSc it is built with, if any) and the mpi4py of the 1d solver all use the same MPI
#               (implementation and version), and that the programs load the same MPI library file; stop
#               with an error otherwise. 1: skip this check.
#
# The programs of the configuration file (python_path, solver0d_path) and COUPLER can be given by their
# name only (e.g. python3): they are then looked up in PATH.

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
# Defaults for ODEsolver and networkName are the same as in coupler.cpp. T0, pre_time and sim_time are required,
# as in coupler.cpp (written by the generator: T0 is the parameter T of the model, pre_time and sim_time come
# from the user inputs).
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
    repr(float(get("T0"))),        # full precision: the solvers check that the coupler uses the same values
    repr(float(get("pre_time"))),
    repr(float(get("sim_time"))),
    get("networkName", "aortic_bif_hybrid"),
    get("initStatePath", "None"),
]
print("\n".join(str(v) for v in values))
PYEOF
}

readarray -t V < <(read_config)
if [[ ${#V[@]} -ne 10 ]]; then
    echo "launch_coupled_mpi :: Error: could not read $CONFIG" >&2
    exit 1
fi
PYTHON="${V[0]}"
SOLVER1D="${V[1]}"
SOLVER0D="${V[2]}"
INITFILE1D="${V[3]}"
ODESOLVER="${V[4]}"
T0="${V[5]}"
PRETIME="${V[6]}"
SIMTIME="${V[7]}"
NETWORK="${V[8]}"
INITSTATE="${V[9]}"

# A program given by its name only (e.g. python_path: python3) is looked up in PATH, as mpirun would; the path found is
# the one checked and launched.
resolve_program() {
    if [[ "$1" != */* ]] && command -v -- "$1" > /dev/null 2>&1; then
        command -v -- "$1"
    else
        echo "$1"
    fi
}
COUPLER="$(resolve_program "$COUPLER")"
SOLVER0D="$(resolve_program "$SOLVER0D")"
PYTHON="$(resolve_program "$PYTHON")"

for f in "$COUPLER" "$SOLVER0D" "$PYTHON"; do
    if [[ ! -x "$f" || -d "$f" ]]; then
        echo "launch_coupled_mpi :: Error: $f not found or not executable" >&2
        exit 1
    fi
done
# the 1d solver script (run1DBFmodel.py), run by the Python of the configuration file
if [[ ! -f "$SOLVER1D" || ! -r "$SOLVER1D" ]]; then
    echo "launch_coupled_mpi :: Error: 1d solver $SOLVER1D (solver1d_path in $CONFIG) not found or not readable" >&2
    exit 1
fi

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

# MPI library files loaded by each program, resolved by the dynamic linker (ldd) and then by realpath: Open MPI 3, 4 and 5
# all have the same library name (libmpi.so.40), so the files are compared, not their names.
MPI_LIB_NAMES=()
MPI_LIB_FILES=()
add_mpi_library_files() {   # <label> <program or shared library>
    local label="$1" bin="$2" name path files=""
    while read -r name path; do
        [[ -n "$name" ]] || continue
        if [[ -z "$path" || "$path" == "not" ]]; then   # "libmpi.so.40 => not found"
            mpi_check_error "the MPI library $name of the $label ($bin) is not found"
        fi
        files+="$(realpath -- "$path")"$'\n'
    done < <( (ldd "$bin" 2> /dev/null || true) | awk '$1 ~ /^lib(mpi|mpich)\.so/ {print $1, $3}')
    files="$(sort -u <<< "$files" | grep . || true)"
    if [[ $(grep -c . <<< "$files") -gt 1 ]]; then
        mpi_check_error "the $label ($bin) loads more than one MPI library:"$'\n'"$files"
    fi
    if [[ -n "$files" ]]; then
        MPI_LIB_NAMES+=("$label")
        MPI_LIB_FILES+=("$files")
    fi
}

# MPI that PETSc was built with, for a 0d solver built with PETSc ("Open MPI x.y.z" or "MPICH x", nothing otherwise):
# from the mpi.h of the MPI include folder of the PETSc installation (MPICC_SHOW in lib/petsc/conf/petscvariables,
# next to libpetsc, or the include folder of PETSc for an MPI installed with it). A PETSc built with another version of
# the same MPI as main0d (e.g. Open MPI 4 and 5, both libmpi.so.40) loads the same library file, so it would not be
# seen otherwise.
petsc_mpi_version() {   # <0d solver>
    local lib conf dirs d h major minor release version
    lib="$( (ldd "$1" 2> /dev/null || true) | awk '$1 ~ /^libpetsc\.so/ {print $3; exit}')"
    [[ -n "$lib" && "$lib" != "not" ]] || return 0
    conf="$(dirname -- "$lib")/petsc/conf/petscvariables"
    if [[ ! -f "$conf" ]]; then
        echo "launch_coupled_mpi :: Warning: $conf not found: the MPI that PETSc was built with is not checked" >&2
        return 0
    fi
    dirs="$(sed -n 's/^MPICC_SHOW *= *//p' "$conf" | tr ' ' '\n' | sed -n 's/^-I//p')"
    dirs+=$'\n'"$(dirname -- "$(dirname -- "$lib")")/include"
    while read -r d; do
        h="$d/mpi.h"
        [[ -n "$d" && -f "$h" ]] || continue
        major="$(sed -n 's/^#define OMPI_MAJOR_VERSION \([0-9]*\).*/\1/p' "$h")"
        if [[ -n "$major" ]]; then
            minor="$(sed -n 's/^#define OMPI_MINOR_VERSION \([0-9]*\).*/\1/p' "$h")"
            release="$(sed -n 's/^#define OMPI_RELEASE_VERSION \([0-9]*\).*/\1/p' "$h")"
            echo "Open MPI $major.$minor.$release"
            return 0
        fi
        version="$(sed -n 's/^#define MPICH_VERSION "\([^"]*\)".*/\1/p' "$h")"
        if [[ -n "$version" ]]; then
            echo "MPICH $version"
            return 0
        fi
    done <<< "$dirs"
    echo "launch_coupled_mpi :: Warning: the mpi.h of the MPI that PETSc was built with was not found (see MPICC_SHOW in" \
         "$conf): it is not checked" >&2
}

check_mpi_versions() {
    local out exe label built runs petsc module

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
        # the MPI library file(s) it loads (more than one: e.g. PETSc built with another MPI than main0d)
        add_mpi_library_files "$label" "$exe"
        # the MPI that PETSc was built with, for a 0d solver built with PETSc
        if [[ "$label" == "0d solver" ]]; then
            petsc="$(petsc_mpi_version "$exe")"
            if [[ -n "$petsc" ]]; then
                add_mpi_entry "0d solver: PETSc (built with)" "$petsc"
            fi
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
print("module: " + MPI.__file__)
' 2>&1 || true)"
    runs="$(sed -n 's/^runs with: //p' <<< "$out")"
    if [[ -z "$runs" ]]; then
        mpi_check_error "could not get the MPI version of mpi4py in $PYTHON:"$'\n'"$out"
    fi
    add_mpi_entry "1d solver (mpi4py)" "$runs"
    # the MPI library file loaded by mpi4py (through its compiled MPI module)
    module="$(sed -n 's/^module: //p' <<< "$out")"
    if [[ -n "$module" && -f "$module" ]]; then
        add_mpi_library_files "1d solver (mpi4py)" "$module"
    fi

    # all the programs must load the same MPI library file
    local j files_mismatch=0
    for j in "${!MPI_LIB_NAMES[@]}"; do
        [[ "${MPI_LIB_FILES[$j]}" == "${MPI_LIB_FILES[0]}" ]] || files_mismatch=1
    done
    if [[ $files_mismatch -ne 0 ]]; then
        {
            echo "launch_coupled_mpi :: Error: the programs do not all load the same MPI library file:"
            for j in "${!MPI_LIB_NAMES[@]}"; do
                printf "    %-30s %s\n" "${MPI_LIB_NAMES[$j]}" "${MPI_LIB_FILES[$j]}"
            done
            echo "  Different installations of an MPI (even of the same version) must not be mixed in one job: load the"
            echo "  same MPI for all of them (e.g. the same 'module load'), rebuild the coupler and main0d, and use a"
            echo "  Python whose mpi4py is built with that MPI."
        } >&2
        mpi_check_error "MPI library mismatch"
    fi

    # compare
    local i ref mismatch=0
    ref="$(mpi_id "${MPI_TEXTS[0]}")"
    for i in "${!MPI_NAMES[@]}"; do
        [[ "$(mpi_id "${MPI_TEXTS[$i]}")" == "$ref" ]] || mismatch=1
    done

    if [[ $mismatch -eq 0 ]]; then
        echo "launch_coupled_mpi :: MPI check passed, all programs use $ref:"
        for i in "${!MPI_NAMES[@]}"; do
            printf "    %-30s %s\n" "${MPI_NAMES[$i]}" "$(mpi_id "${MPI_TEXTS[$i]}")"
        done
    else
        {
            echo "launch_coupled_mpi :: Error: the programs do not all use the same MPI:"
            for i in "${!MPI_NAMES[@]}"; do
                printf "    %-30s %-18s [%s]\n" "${MPI_NAMES[$i]}" "$(mpi_id "${MPI_TEXTS[$i]}")" "${MPI_TEXTS[$i]%%$'\n'*}"
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
echo "launch_coupled_mpi :: ODEsolver=$ODESOLVER T0=$T0 pre_time=$PRETIME sim_time=$SIMTIME networkName=$NETWORK initStatePath=$INITSTATE"

# python -u: unbuffered output, so that the 1d solver's log interleaves with the others.
# $@ : all remaining arguments are passed to mpirun.
exec "$MPIRUN" "$@" \
    -np 1 "$COUPLER" "$CONFIG" \
  : -np 1 "$SOLVER0D" -coupled 1 -ODEsolver "$ODESOLVER" -T0 "$T0" -preTime "$PRETIME" -simTime "$SIMTIME" \
        -networkName "$NETWORK" -initStatePath "$INITSTATE" \
  : -np 1 "$PYTHON" -u "$SOLVER1D" --coupled --initFile "$INITFILE1D" --ODEsolver "$ODESOLVER" --T0 "$T0" \
        --preTime "$PRETIME" --simTime "$SIMTIME" --initStatePath "$INITSTATE"

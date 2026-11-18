# Run this with: . load_openmpi.sh   (sourced, so that the loaded module stays in your shell)
module load mpi/openmpi-x86_64 && echo "succesfully loaded mpi/openmpi-x86_64"

# Print the MPI now in use: mpirun and mpic++ (and the mpi4py of the 1d solver's Python)
# must all come from the same Open MPI installation.
if command -v mpirun > /dev/null; then
    echo "mpirun : $(command -v mpirun)  [$(mpirun --version 2>&1 | head -n 1)]"
else
    echo "mpirun : not found"
fi
if command -v mpic++ > /dev/null; then
    echo "mpic++ : $(command -v mpic++)  [$(mpic++ --showme:version 2>&1)]"
    echo "         C++ compiler: $(mpic++ --version 2>&1 | head -n 1)"
else
    echo "mpic++ : not found"
fi

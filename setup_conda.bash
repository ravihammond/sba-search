# activate the conda environment
#conda activate hanabi

# set path
#echo "Before"
#echo ${CONDA_PREFIX}
CONDA_PREFIX=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
#echo "After"
#echo ${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
#echo ${CONDA_PREFIX}
export CPATH=${CONDA_PREFIX}/include:${CPATH}
export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# avoid tensor operation using all cpu cores
export OMP_NUM_THREADS=1

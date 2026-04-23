#!/bin/bash
#PBS -N bin_run
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_mem=4gb:scratch_local=32gb:gpu_cap=\"sm_75,compute_120\"
#PBS -l walltime=01:00:00
#PBS -o /storage/plzen1/home/xkiszk00/containers/knn_job_log_latest.out
#PBS -e /storage/plzen1/home/xkiszk00/containers/knn_job_log_latest.err

# =========================================
# Paths (edit to match your setup)

HOME_DIR=/storage/plzen1/home/xkiszk00
PROJ_FOLDER="${HOME_DIR}/repos/bin"
CODE_DIR="${PROJ_FOLDER}/src"
DATA_DIR="${PROJ_FOLDER}/data"

PYTHON_SCRIPT="${CODE_DIR}/train.py"

# Conda environment name
CONDA_ENV_NAME=bin_env

# =========================================
# log and error files

CUR_TIME=$(date +%Y-%m-%d_%H-%M-%S)

if [ -n "$NOTE" ]; then
    LOG_SUFFIX="_${NOTE}"
else
    LOG_SUFFIX=""
fi

LOG_DIR="${PROJ_FOLDER}/logs/job_${CUR_TIME}${LOG_SUFFIX}"
mkdir -p "$LOG_DIR"

JOB_LOG_OUT="${LOG_DIR}/log.out"
JOB_LOG_ERR="${LOG_DIR}/log.err"

touch "$JOB_LOG_OUT" "$JOB_LOG_ERR"

# redirect all script outputs (stdout/stderr) to log files
exec > "$JOB_LOG_OUT" 2> "$JOB_LOG_ERR"

# =========================================

echo "JOB START"
echo "PBS job ID: ${PBS_JOBID}"
echo "Node:       $(hostname -f)"
echo "Scratch dir:${SCRATCHDIR}"
test -n "$SCRATCHDIR" || { echo "Variable SCRATCHDIR is not set!" >&2; exit 1; }

# =========================================
# Copy code and data to scratch

echo "Copying code and data to SCRATCH"
cp -R "$CODE_DIR" "$SCRATCHDIR" || { echo "Error copying code!" >&2; exit 2; }
cp -R "$DATA_DIR" "$SCRATCHDIR" || { echo "Error copying data!" >&2; exit 2; }

# Path inside scratch
SCRATCH_CODE_DIR="${SCRATCHDIR}/src"
SCRATCH_DATA_DIR="${SCRATCHDIR}/data"

# =========================================
# Activate conda/mamba environment

echo "Activating conda environment: $CONDA_ENV_NAME"

# load conda/mamba if needed
# module load mambaforge  # uncomment if your cluster uses modules

mamba activate /auto/plzen1/home/xkiszk00/bin_env

# =========================================
# Run python script

PYTHON_LOG="${SCRATCHDIR}/python_job.log"

echo "Executing Python script..."
python "$SCRATCH_CODE_DIR/main.py" --data_dir "$SCRATCH_DATA_DIR" --log_file "$PYTHON_LOG"

# =========================================
# Copy logs/results back

echo "Copying results back to home log directory"
cp "$PYTHON_LOG" "$LOG_DIR/" || { echo "Failed to copy Python log!" >&2; exit 4; }

echo "JOB END"

# clean scratch
clean_scratch

#!/bin/bash
#SBATCH --job-name=cera_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/train_%A_%a.log
#SBATCH --error=slurm_logs/train_err_%A_%a.log
#SBATCH --partition=8gpus
# NOTE: Add your cluster account line here, e.g.:
#   #SBATCH -A YOUR_ACCOUNT

# Job array wrapper for training. Each task reads one line from a config file.
#
# Usage:
#   sbatch --array=1-N%5 slurm/run_train_array.sh CONFIG_FILE
#
#   N         number of lines in CONFIG_FILE (excluding comment lines)
#   %5        max 5 tasks running concurrently (adjust to your QOS limit)
#
# CONFIG_FILE format (one experiment per line, # lines are ignored):
#   CELL_ID  METHOD  RANK  LR  DROPOUT  BASE_MODEL  [DATASET]
# DATASET defaults to "math" if omitted.
#
# Example:
#   sbatch --array=1-32%5 slurm/run_train_array.sh slurm/configs/sweep_1b3b.txt

CONFIG_FILE=${1:?CONFIG_FILE argument required}

# Activate your environment here, e.g.:
#   conda activate cera_env
# or:
#   source /path/to/venv/bin/activate

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — run via sbatch}"
mkdir -p slurm_logs

# Read the Nth non-comment line (1-indexed, matching SLURM_ARRAY_TASK_ID)
LINE=$(grep -v '^\s*#' "$CONFIG_FILE" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$LINE" ]; then
    echo "[ERROR] No config found for task ID=${SLURM_ARRAY_TASK_ID} in ${CONFIG_FILE}"
    exit 1
fi

read -r CELL_ID METHOD RANK LR DROPOUT BASE_MODEL DATASET <<< "$LINE"
DATASET=${DATASET:-math}

echo "[START] Array job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | cell=${CELL_ID} method=${METHOD} rank=${RANK} lr=${LR} dropout=${DROPOUT} dataset=${DATASET} model=${BASE_MODEL}"

python train.py \
    --model_type  "$METHOD" \
    --rank        "$RANK" \
    --lr          "$LR" \
    --dropout     "$DROPOUT" \
    --dataset     "$DATASET" \
    --epochs      3 \
    --base_model  "$BASE_MODEL"

echo "[END] Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"

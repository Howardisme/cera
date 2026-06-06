#!/bin/bash
#SBATCH --job-name=cera_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/eval_%A_%a.log
#SBATCH --error=slurm_logs/eval_err_%A_%a.log
#SBATCH --partition=normal
# NOTE: Add your cluster account line here, e.g.:
#   #SBATCH -A YOUR_ACCOUNT

# Job array wrapper for evaluation. Each task reads one line from a config file.
# Use aftercorr dependency so task N starts only after train task N succeeds.
#
# Usage:
#   TRAIN_JOB=$(sbatch --parsable --array=1-N%5 slurm/run_train_array.sh CONFIG_FILE)
#   sbatch --array=1-N%5 --dependency=aftercorr:$TRAIN_JOB slurm/run_eval_array.sh CONFIG_FILE
#
# CONFIG_FILE format (one experiment per line, # lines are ignored):
#   CELL_ID  METHOD  RANK  LR  DROPOUT  BASE_MODEL
#
# Example:
#   TRAIN_JOB=$(sbatch --parsable --array=1-32%5 slurm/run_train_array.sh slurm/configs/sweep_1b3b.txt)
#   sbatch --array=1-32%5 --dependency=aftercorr:$TRAIN_JOB slurm/run_eval_array.sh slurm/configs/sweep_1b3b.txt

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

read -r CELL_ID METHOD RANK LR DROPOUT BASE_MODEL <<< "$LINE"

echo "[START] Eval array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | cell=${CELL_ID} method=${METHOD} rank=${RANK} lr=${LR} dropout=${DROPOUT} model=${BASE_MODEL}"

# Delegate to the single-cell eval script (handles checkpoint search + all 3 evals)
bash slurm/run_eval.sh "$CELL_ID" "$METHOD" "$RANK" "$LR" "$DROPOUT" "$BASE_MODEL"

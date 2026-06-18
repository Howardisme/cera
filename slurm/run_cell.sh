#!/bin/bash
#SBATCH --job-name=cera_cell
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/cell_%j.log
#SBATCH --error=slurm_logs/cell_err_%j.log
#SBATCH --partition=8gpus
# NOTE: Add your cluster account line here, e.g.:
#   #SBATCH -A YOUR_ACCOUNT

# Combined TRAIN + EVAL for one sweep cell, so each cell occupies exactly ONE
# Slurm job (instead of two). Train runs first; eval runs only if train
# succeeds. The checkpoint produced by train is located automatically by
# run_eval.sh from results/Exp_*.
#
# Usage:
#   sbatch slurm/run_cell.sh CELL_ID METHOD RANK LR DROPOUT DATASET EPOCHS BASE_MODEL
#
# Example:
#   sbatch slurm/run_cell.sh dora_r128_lr3e-4_1b DoRA 128 3e-4 0.0 math 3 meta-llama/Llama-3.2-1B

set -euo pipefail

CELL_ID=${1:?CELL_ID required}
METHOD=${2:?METHOD required}
RANK=${3:?RANK required}
LR=${4:?LR required}
DROPOUT=${5:?DROPOUT required}
DATASET=${6:-math}
EPOCHS=${7:-3}
BASE_MODEL=${8:-meta-llama/Llama-3.1-8B}

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run via sbatch}"
mkdir -p slurm_logs

# Activate your environment here (must match run_train.sh / run_eval.sh), e.g.:
#   conda activate cera_env
# or:
#   source /path/to/venv/bin/activate

echo "[CELL ${CELL_ID}] START $(date) | ${METHOD} R=${RANK} lr=${LR} D=${DROPOUT} ${DATASET} model=${BASE_MODEL}"

# 1) Train
echo "[CELL ${CELL_ID}] === TRAIN ==="
bash slurm/run_train.sh "$METHOD" "$RANK" "$LR" "$DROPOUT" "$DATASET" "$EPOCHS" "$BASE_MODEL"

# 2) Eval (only reached if train succeeded, due to set -e)
echo "[CELL ${CELL_ID}] === EVAL ==="
bash slurm/run_eval.sh "$CELL_ID" "$METHOD" "$RANK" "$LR" "$DROPOUT" "$BASE_MODEL" results/eval_outputs "$DATASET"

echo "[CELL ${CELL_ID}] DONE $(date)"

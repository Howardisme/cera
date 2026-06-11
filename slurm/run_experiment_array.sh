#!/bin/bash
#SBATCH --job-name=cera_exp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/exp_%A_%a.log
#SBATCH --error=slurm_logs/exp_err_%A_%a.log
#SBATCH --partition=8gpus
# NOTE: Add your cluster account line here, e.g.:
#   #SBATCH -A YOUR_ACCOUNT

# Combined train + eval array job. Each task trains a model then immediately
# evaluates it, so the whole cell counts as a single SLURM job. This halves
# the number of submitted jobs compared to separate train and eval arrays,
# making it easier to stay within per-user job submission limits.
#
# Usage:
#   sbatch --array=1-N%5 slurm/run_experiment_array.sh CONFIG_FILE
#
# To submit all tasks within a QOS limit of 10, use run_resubmitter.sh:
#   BATCH_JOB=$(sbatch --parsable --array=1-8%5 slurm/run_experiment_array.sh CONFIG)
#   sbatch --dependency=afterok:$BATCH_JOB slurm/run_resubmitter.sh CONFIG 9 16 TOTAL 8 PARTITION
#
# CONFIG_FILE format (one experiment per line, # lines are ignored):
#   CELL_ID  METHOD  RANK  LR  DROPOUT  BASE_MODEL  [DATASET]
# DATASET defaults to "math" if omitted.

CONFIG_FILE=${1:?CONFIG_FILE argument required}

# Activate your environment here, e.g.:
#   conda activate cera_env
# or:
#   source /path/to/venv/bin/activate

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — run via sbatch}"
mkdir -p slurm_logs

LINE=$(grep -v '^\s*#' "$CONFIG_FILE" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$LINE" ]; then
    echo "[ERROR] No config for task ID=${SLURM_ARRAY_TASK_ID} in ${CONFIG_FILE}"
    exit 1
fi

read -r CELL_ID METHOD RANK LR DROPOUT BASE_MODEL DATASET <<< "$LINE"
DATASET=${DATASET:-math}

echo "[START] Task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | cell=${CELL_ID} method=${METHOD} rank=${RANK} lr=${LR} dropout=${DROPOUT} dataset=${DATASET} model=${BASE_MODEL}"

# ── 1. Train ──────────────────────────────────────────────────────────────────
python train.py \
    --model_type  "$METHOD" \
    --rank        "$RANK" \
    --lr          "$LR" \
    --dropout     "$DROPOUT" \
    --dataset     "$DATASET" \
    --epochs      3 \
    --base_model  "$BASE_MODEL"

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[ERROR] Training failed (exit $TRAIN_EXIT) — skipping eval"
    exit $TRAIN_EXIT
fi

# ── 2. Eval ───────────────────────────────────────────────────────────────────
bash slurm/run_eval.sh "$CELL_ID" "$METHOD" "$RANK" "$LR" "$DROPOUT" "$BASE_MODEL"

echo "[END] Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"

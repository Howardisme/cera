#!/bin/bash
# Submit training + eval jobs for dataset ablation: MathInstruct vs SlimOrca.
#
# Experiment: CeRA and LoRA at R=128, lr=5e-4 on both datasets.
# Goal: compare how the choice of fine-tuning dataset affects downstream MATH/GSM8K.
#
# Usage:
#   bash paper_experiments/submit_dataset_ablation.sh [--dry-run]

set -e

DRY_RUN=0
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=1
    echo "[DRY-RUN] No jobs will actually be submitted."
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."
mkdir -p slurm_logs results/eval_outputs

BASE_MODEL="meta-llama/Llama-3.1-8B"
RANK=128
LR="5e-4"
EVAL_BASE="results/eval_outputs"

eval_done() {
    local out="${EVAL_BASE}/${1}"
    [ -f "${out}/math_pass1.json" ] && \
    [ -f "${out}/math_pass10.json" ] && \
    [ -f "${out}/gsm8k_pass1.json" ]
}

submit_train() {
    local method=$1; local dropout=$2; local dataset=$3
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY_JOB_ID"
        echo "[DRY] sbatch slurm/run_train.sh $method $RANK $LR $dropout $dataset 3 $BASE_MODEL" >&2
        return
    fi
    sbatch --parsable slurm/run_train.sh "$method" "$RANK" "$LR" "$dropout" "$dataset" 3 "$BASE_MODEL"
}

submit_eval() {
    local cell_id=$1; local method=$2; local dropout=$3; local dep=$4
    if eval_done "$cell_id"; then
        echo "[SKIP] ${cell_id}: all eval outputs exist" >&2
        return
    fi
    local dep_arg=""
    [ -n "$dep" ] && dep_arg="--dependency=afterok:${dep}"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY] sbatch $dep_arg slurm/run_eval.sh $cell_id $method $RANK $LR $dropout $BASE_MODEL $EVAL_BASE" >&2
        return
    fi
    sbatch $dep_arg slurm/run_eval.sh \
        "$cell_id" "$method" "$RANK" "$LR" "$dropout" "$BASE_MODEL" "$EVAL_BASE"
}

handle_cell() {
    local cell_id=$1; local method=$2; local dropout=$3; local dataset=$4
    echo "[SUBMIT] ${cell_id}: train (${dataset}) + chained eval" >&2
    local train_jid
    train_jid=$(submit_train "$method" "$dropout" "$dataset")
    echo "  Train job ID: ${train_jid}" >&2
    submit_eval "$cell_id" "$method" "$dropout" "$train_jid"
}

echo "======================================================"
echo " Dataset Ablation: MathInstruct vs SlimOrca"
echo " $(date)"
echo "======================================================"

echo ""
echo "--- CeRA R=128, lr=5e-4 ---"
handle_cell "dataset_cera_math" CeRA 0.1 math
handle_cell "dataset_cera_orca" CeRA 0.1 orca

echo ""
echo "--- LoRA R=128, lr=5e-4 ---"
handle_cell "dataset_lora_math" LoRA 0.0 math
handle_cell "dataset_lora_orca" LoRA 0.0 orca

echo ""
echo "======================================================"
echo " Submission complete. Monitor with: squeue -u \$USER"
echo "======================================================"

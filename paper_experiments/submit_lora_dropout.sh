#!/bin/bash
# Submit LoRA+dropout ablation for reviewer response.
#
# Motivation: reviewer questioned whether the LoRA baseline uses dropout.
# Standard LoRA in this codebase uses dropout=0.0.
# This ablation re-runs LoRA with dropout=0.1 to show that adding dropout
# does not consistently improve LoRA performance.
#
# Cells: R={64,128} x LR={1e-4,3e-4,5e-4}, LoRA dropout=0.1
# Eval: MATH pass@1, GSM8K pass@1
#
# Usage:
#   bash paper_experiments/submit_lora_dropout.sh [--dry-run]

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
DROPOUT="0.1"
EVAL_BASE="results/eval_outputs"

eval_done() {
    local out="${EVAL_BASE}/${1}"
    [ -f "${out}/math_pass1.json" ] && \
    [ -f "${out}/math_pass10.json" ] && \
    [ -f "${out}/gsm8k_pass1.json" ]
}

submit_train() {
    local rank=$1; local lr=$2
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY_JOB_ID"
        echo "[DRY] sbatch slurm/run_train.sh LoRA $rank $lr $DROPOUT math 3 $BASE_MODEL" >&2
        return
    fi
    sbatch --parsable slurm/run_train.sh LoRA "$rank" "$lr" "$DROPOUT" math 3 "$BASE_MODEL"
}

submit_eval() {
    local cell_id=$1; local rank=$2; local lr=$3; local dep=$4
    if eval_done "$cell_id"; then
        echo "[SKIP] ${cell_id}: eval outputs already exist" >&2
        return
    fi
    local dep_arg=""
    [ -n "$dep" ] && dep_arg="--dependency=afterok:${dep}"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY] sbatch $dep_arg slurm/run_eval.sh $cell_id LoRA $rank $lr $DROPOUT $BASE_MODEL $EVAL_BASE" >&2
        return
    fi
    sbatch $dep_arg slurm/run_eval.sh \
        "$cell_id" LoRA "$rank" "$lr" "$DROPOUT" "$BASE_MODEL" "$EVAL_BASE"
}

handle_cell() {
    local cell_id=$1; local rank=$2; local lr=$3
    echo "[SUBMIT] ${cell_id}: train + chained eval" >&2
    local train_jid
    train_jid=$(submit_train "$rank" "$lr")
    echo "  Train job ID: ${train_jid}" >&2
    submit_eval "$cell_id" "$rank" "$lr" "$train_jid"
}

echo "======================================================"
echo " LoRA+Dropout=0.1 Ablation (Reviewer Response)"
echo " $(date)"
echo "======================================================"

echo ""
echo "--- r=64 ---"
handle_cell "lora_dp01_r64_lr1e-4"  64 "1e-4"
handle_cell "lora_dp01_r64_lr3e-4"  64 "3e-4"
handle_cell "lora_dp01_r64_lr5e-4"  64 "5e-4"

echo ""
echo "--- r=128 ---"
handle_cell "lora_dp01_r128_lr1e-4" 128 "1e-4"
handle_cell "lora_dp01_r128_lr3e-4" 128 "3e-4"
handle_cell "lora_dp01_r128_lr5e-4" 128 "5e-4"

echo ""
echo "======================================================"
echo " Submission complete. Monitor with: squeue -u \$USER"
echo "======================================================"

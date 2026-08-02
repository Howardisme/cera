#!/bin/bash
# Scale-matched main comparison: LoRA / DoRA with alpha = rank (effective
# scale = alpha/rank = 1), matching CeRA's s = 1. Appendix D follow-up to
# Table 1: isolates the alpha/rank = 1/... output-scale suppression from
# the linear vs non-linear architectural difference.
#
# Experiment: LoRA / DoRA x R={64,128} x LR={1e-4,3e-4,5e-4,1e-3}
#             + R=512 LoRA (matched-budget baseline, 3 LR)
# Model: Llama-3.1-8B  |  Dataset: MathInstruct
# Eval: MATH pass@1 (greedy, 500), MATH pass@10 (n=10, 500), GSM8K pass@1 (1319)
#
# CeRA cells are unchanged and re-use existing eval outputs from
# submit_main_comparison.sh -- they are not re-run here.
#
# Usage:
#   bash paper_experiments/submit_scale_matched_comparison.sh [--dry-run]
#
# Behavior (idempotent):
#   - Cells with existing best checkpoint (matching alpha=rank folder): eval only.
#   - Cells missing checkpoint: submit train, then chain eval with afterok.
#   - Cells with all 3 eval outputs: skip entirely.

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
EVAL_BASE="results/eval_outputs"
DATASET="math"
EPOCHS=3
DROPOUT="0.0"  # LoRA/DoRA only in this script

# ── Helpers ───────────────────────────────────────────────────────────────────

lr_to_decimal() {
    case "$1" in
      "1e-4") echo "0.0001" ;;
      "3e-4") echo "0.0003" ;;
      "5e-4") echo "0.0005" ;;
      "1e-3") echo "0.001"  ;;
      *)      echo "$1"     ;;
    esac
}

# Locate the alpha=rank checkpoint folder. Mirrors run_eval.sh's alpha filter:
# alpha != 32 → require _A{ALPHA}. This script only calls it with ALPHA==RANK,
# and every rank we care about (64/128/512) is != 32.
find_checkpoint() {
    local method=$1; local rank=$2; local lr=$3; local dropout=$4; local alpha=$5
    local lr_dec=$(lr_to_decimal "$lr")
    local method_lower=$(echo "$method" | tr '[:upper:]' '[:lower:]')
    local dir=$(ls -dt "results/Exp_${method}_${DATASET}_R${rank}_lr${lr_dec}_"* 2>/dev/null \
        | grep "_D${dropout}" | grep "_A${alpha}" | head -1)
    if [ -n "$dir" ]; then
        ls "${dir}/${method}/${method_lower}_ckpt_best_"*.pt 2>/dev/null | head -1
    fi
}

eval_done() {
    local out="${EVAL_BASE}/${1}"
    [ -f "${out}/math_pass1.json" ] && \
    [ -f "${out}/math_pass10.json" ] && \
    [ -f "${out}/gsm8k_pass1.json" ]
}

submit_train() {
    local method=$1; local rank=$2; local lr=$3; local dropout=$4; local alpha=$5
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY_JOB_ID"
        echo "[DRY] sbatch slurm/run_train.sh $method $rank $lr $dropout $DATASET $EPOCHS $BASE_MODEL $alpha" >&2
        return
    fi
    sbatch --parsable slurm/run_train.sh \
        "$method" "$rank" "$lr" "$dropout" "$DATASET" "$EPOCHS" "$BASE_MODEL" "$alpha"
}

submit_eval() {
    local cell_id=$1; local method=$2; local rank=$3; local lr=$4; local dropout=$5; local alpha=$6; local dep=$7
    if eval_done "$cell_id"; then
        echo "[SKIP] ${cell_id}: all eval outputs exist" >&2
        return
    fi
    local dep_arg=""
    [ -n "$dep" ] && dep_arg="--dependency=afterok:${dep}"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY] sbatch $dep_arg slurm/run_eval.sh $cell_id $method $rank $lr $dropout $BASE_MODEL $EVAL_BASE $DATASET $alpha" >&2
        return
    fi
    sbatch $dep_arg slurm/run_eval.sh \
        "$cell_id" "$method" "$rank" "$lr" "$dropout" "$BASE_MODEL" "$EVAL_BASE" "$DATASET" "$alpha"
}

# One scale-matched cell: alpha is fixed to rank so effective scale = 1.
handle_cell() {
    local cell_id=$1; local method=$2; local rank=$3; local lr=$4
    local alpha=$rank
    local ckpt=$(find_checkpoint "$method" "$rank" "$lr" "$DROPOUT" "$alpha")

    if [ -n "$ckpt" ]; then
        echo "[HAVE] ${cell_id}: checkpoint found (alpha=${alpha}) -> eval only" >&2
        submit_eval "$cell_id" "$method" "$rank" "$lr" "$DROPOUT" "$alpha"
    else
        echo "[MISS] ${cell_id}: no checkpoint -> train + chained eval (alpha=${alpha})" >&2
        local train_jid
        train_jid=$(submit_train "$method" "$rank" "$lr" "$DROPOUT" "$alpha")
        echo "  Train job ID: ${train_jid}" >&2
        submit_eval "$cell_id" "$method" "$rank" "$lr" "$DROPOUT" "$alpha" "$train_jid"
    fi
}

echo "=============================================================="
echo " Scale-Matched Comparison (alpha = rank): LoRA / DoRA on Llama-3.1-8B"
echo " $(date)"
echo "=============================================================="

echo ""
echo "--- LoRA r=64 (alpha=64) ---"
handle_cell "r64_lora_a64_lr1e-4"    LoRA  64  "1e-4"
handle_cell "r64_lora_a64_lr3e-4"    LoRA  64  "3e-4"
handle_cell "r64_lora_a64_lr5e-4"    LoRA  64  "5e-4"
handle_cell "r64_lora_a64_lr1e-3"    LoRA  64  "1e-3"

echo ""
echo "--- LoRA r=128 (alpha=128) ---"
handle_cell "r128_lora_a128_lr1e-4"  LoRA 128 "1e-4"
handle_cell "r128_lora_a128_lr3e-4"  LoRA 128 "3e-4"
handle_cell "r128_lora_a128_lr5e-4"  LoRA 128 "5e-4"
handle_cell "r128_lora_a128_lr1e-3"  LoRA 128 "1e-3"

echo ""
echo "--- LoRA r=512 (alpha=512, parameter budget baseline) ---"
handle_cell "r512_lora_a512_lr1e-4"  LoRA 512 "1e-4"
handle_cell "r512_lora_a512_lr3e-4"  LoRA 512 "3e-4"
handle_cell "r512_lora_a512_lr5e-4"  LoRA 512 "5e-4"

echo ""
echo "--- DoRA r=64 (alpha=64) ---"
handle_cell "r64_dora_a64_lr1e-4"    DoRA  64  "1e-4"
handle_cell "r64_dora_a64_lr3e-4"    DoRA  64  "3e-4"
handle_cell "r64_dora_a64_lr5e-4"    DoRA  64  "5e-4"
handle_cell "r64_dora_a64_lr1e-3"    DoRA  64  "1e-3"

echo ""
echo "--- DoRA r=128 (alpha=128) ---"
handle_cell "r128_dora_a128_lr1e-4"  DoRA 128 "1e-4"
handle_cell "r128_dora_a128_lr3e-4"  DoRA 128 "3e-4"
handle_cell "r128_dora_a128_lr5e-4"  DoRA 128 "5e-4"
handle_cell "r128_dora_a128_lr1e-3"  DoRA 128 "1e-3"

echo ""
echo "=============================================================="
echo " Submission complete. Monitor with: squeue -u \$USER"
echo "=============================================================="

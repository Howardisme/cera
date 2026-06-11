#!/bin/bash
# Submit training + eval jobs for the model scale experiment.
#
# Experiment: CeRA / LoRA / DoRA x R={64,128} on Llama-3.2-1B, 3B, and 8B
# LR: "best" -- resolved per model/method/rank by the lookup tables in
#     slurm/run_train.sh / slurm/run_eval.sh (filled from the LR sweeps).
# Dataset: MathInstruct
# Eval: MATH pass@1 (greedy, 500), MATH pass@10 (n=10, 500), GSM8K pass@1 (1319)
#
# Usage:
#   bash paper_experiments/submit_model_scale.sh [--dry-run]

set -e

DRY_RUN=0
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=1
    echo "[DRY-RUN] No jobs will actually be submitted."
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."
mkdir -p slurm_logs results/eval_outputs

LR="best"
EVAL_BASE="results/eval_outputs"

get_dropout() {
    case "$1" in
      CeRA) echo "0.1" ;;
      *)    echo "0.0" ;;
    esac
}

eval_done() {
    local out="${EVAL_BASE}/${1}"
    [ -f "${out}/math_pass1.json" ] && \
    [ -f "${out}/math_pass10.json" ] && \
    [ -f "${out}/gsm8k_pass1.json" ]
}

submit_train() {
    local method=$1; local rank=$2; local dropout=$3; local model=$4
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY_JOB_ID"
        echo "[DRY] sbatch slurm/run_train.sh $method $rank $LR $dropout math 3 $model" >&2
        return
    fi
    sbatch --parsable slurm/run_train.sh "$method" "$rank" "$LR" "$dropout" math 3 "$model"
}

submit_eval() {
    local cell_id=$1; local method=$2; local rank=$3; local dropout=$4; local model=$5; local dep=$6
    if eval_done "$cell_id"; then
        echo "[SKIP] ${cell_id}: all eval outputs exist" >&2
        return
    fi
    local dep_arg=""
    [ -n "$dep" ] && dep_arg="--dependency=afterok:${dep}"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY] sbatch $dep_arg slurm/run_eval.sh $cell_id $method $rank $LR $dropout $model $EVAL_BASE" >&2
        return
    fi
    sbatch $dep_arg slurm/run_eval.sh \
        "$cell_id" "$method" "$rank" "$LR" "$dropout" "$model" "$EVAL_BASE"
}

handle_cell() {
    local cell_id=$1; local method=$2; local rank=$3; local model=$4
    local dropout=$(get_dropout "$method")
    echo "[SUBMIT] ${cell_id}: train + chained eval" >&2
    local train_jid
    train_jid=$(submit_train "$method" "$rank" "$dropout" "$model")
    echo "  Train job ID: ${train_jid}" >&2
    submit_eval "$cell_id" "$method" "$rank" "$dropout" "$model" "$train_jid"
}

echo "======================================================"
echo " Model Scale: Llama-3.2-1B / 3B / Llama-3.1-8B"
echo " $(date)"
echo "======================================================"

echo ""
echo "--- Llama-3.2-1B ---"
handle_cell "scale_1b_r64_cera"  CeRA  64 "meta-llama/Llama-3.2-1B"
handle_cell "scale_1b_r64_lora"  LoRA  64 "meta-llama/Llama-3.2-1B"
handle_cell "scale_1b_r64_dora"  DoRA  64 "meta-llama/Llama-3.2-1B"
handle_cell "scale_1b_r128_cera" CeRA 128 "meta-llama/Llama-3.2-1B"
handle_cell "scale_1b_r128_lora" LoRA 128 "meta-llama/Llama-3.2-1B"
handle_cell "scale_1b_r128_dora" DoRA 128 "meta-llama/Llama-3.2-1B"

echo ""
echo "--- Llama-3.2-3B ---"
handle_cell "scale_3b_r64_cera"  CeRA  64 "meta-llama/Llama-3.2-3B"
handle_cell "scale_3b_r64_lora"  LoRA  64 "meta-llama/Llama-3.2-3B"
handle_cell "scale_3b_r64_dora"  DoRA  64 "meta-llama/Llama-3.2-3B"
handle_cell "scale_3b_r128_cera" CeRA 128 "meta-llama/Llama-3.2-3B"
handle_cell "scale_3b_r128_lora" LoRA 128 "meta-llama/Llama-3.2-3B"
handle_cell "scale_3b_r128_dora" DoRA 128 "meta-llama/Llama-3.2-3B"

echo ""
echo "--- Llama-3.1-8B (use existing checkpoints via main comparison) ---"
echo "[NOTE] 8B results come from submit_main_comparison.sh (best-LR cells per the run_eval.sh lookup)."

echo ""
echo "======================================================"
echo " Submission complete. Monitor with: squeue -u \$USER"
echo "======================================================"

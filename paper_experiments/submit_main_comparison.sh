#!/bin/bash
# Submit training + eval jobs for the main paper comparison table.
#
# Experiment: CeRA / LoRA / DoRA x R={64,128} x LR={1e-4,3e-4,5e-4,1e-3}
#             + R=512 LoRA (parameter budget baseline)
# Model: Llama-3.1-8B  |  Dataset: MathInstruct
# Eval: MATH pass@1 (greedy, 500), MATH pass@10 (n=10, 500), GSM8K pass@1 (1319)
#
# Usage:
#   bash paper_experiments/submit_main_comparison.sh [--dry-run]
#
# Behavior (idempotent):
#   - Cells with existing best checkpoint: submit eval only.
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

# ── Helpers ───────────────────────────────────────────────────────────────────

get_dropout() {
    case "$1" in
      CeRA) echo "0.1" ;;
      *)    echo "0.0" ;;
    esac
}

lr_to_decimal() {
    case "$1" in
      "1e-4") echo "0.0001" ;;
      "3e-4") echo "0.0003" ;;
      "5e-4") echo "0.0005" ;;
      "1e-3") echo "0.001"  ;;
      *)      echo "$1"     ;;
    esac
}

find_checkpoint() {
    local method=$1; local rank=$2; local lr=$3; local dropout=$4
    local lr_dec=$(lr_to_decimal "$lr")
    local method_lower=$(echo "$method" | tr '[:upper:]' '[:lower:]')
    local dir=$(ls -dt "results/Exp_${method}_math_R${rank}_lr${lr_dec}_"* 2>/dev/null \
        | grep "_D${dropout}" | head -1)
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
    local method=$1; local rank=$2; local lr=$3; local dropout=$4
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY_JOB_ID"
        echo "[DRY] sbatch slurm/run_train.sh $method $rank $lr $dropout math 3 $BASE_MODEL" >&2
        return
    fi
    sbatch --parsable slurm/run_train.sh "$method" "$rank" "$lr" "$dropout" math 3 "$BASE_MODEL"
}

submit_eval() {
    local cell_id=$1; local method=$2; local rank=$3; local lr=$4; local dropout=$5; local dep=$6
    if eval_done "$cell_id"; then
        echo "[SKIP] ${cell_id}: all eval outputs exist" >&2
        return
    fi
    local dep_arg=""
    [ -n "$dep" ] && dep_arg="--dependency=afterok:${dep}"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY] sbatch $dep_arg slurm/run_eval.sh $cell_id $method $rank $lr $dropout $BASE_MODEL $EVAL_BASE" >&2
        return
    fi
    sbatch $dep_arg slurm/run_eval.sh \
        "$cell_id" "$method" "$rank" "$lr" "$dropout" "$BASE_MODEL" "$EVAL_BASE"
}

handle_cell() {
    local cell_id=$1; local method=$2; local rank=$3; local lr=$4
    local dropout=$(get_dropout "$method")
    local ckpt=$(find_checkpoint "$method" "$rank" "$lr" "$dropout")

    if [ -n "$ckpt" ]; then
        echo "[HAVE] ${cell_id}: checkpoint found -> eval only" >&2
        submit_eval "$cell_id" "$method" "$rank" "$lr" "$dropout"
    else
        echo "[MISS] ${cell_id}: no checkpoint -> train + chained eval" >&2
        local train_jid
        train_jid=$(submit_train "$method" "$rank" "$lr" "$dropout")
        echo "  Train job ID: ${train_jid}" >&2
        submit_eval "$cell_id" "$method" "$rank" "$lr" "$dropout" "$train_jid"
    fi
}

echo "======================================================"
echo " Main Comparison: CeRA / LoRA / DoRA on Llama-3.1-8B"
echo " $(date)"
echo "======================================================"

echo ""
echo "--- r=64 ---"
handle_cell "r64_cera_lr1e-4"  CeRA  64 "1e-4"
handle_cell "r64_cera_lr3e-4"  CeRA  64 "3e-4"
handle_cell "r64_cera_lr5e-4"  CeRA  64 "5e-4"
handle_cell "r64_cera_lr1e-3"  CeRA  64 "1e-3"
handle_cell "r64_lora_lr1e-4"  LoRA  64 "1e-4"
handle_cell "r64_lora_lr3e-4"  LoRA  64 "3e-4"
handle_cell "r64_lora_lr5e-4"  LoRA  64 "5e-4"
handle_cell "r64_lora_lr1e-3"  LoRA  64 "1e-3"
handle_cell "r64_dora_lr1e-4"  DoRA  64 "1e-4"
handle_cell "r64_dora_lr3e-4"  DoRA  64 "3e-4"
handle_cell "r64_dora_lr5e-4"  DoRA  64 "5e-4"
handle_cell "r64_dora_lr1e-3"  DoRA  64 "1e-3"

echo ""
echo "--- r=128 ---"
handle_cell "r128_cera_lr1e-4" CeRA 128 "1e-4"
handle_cell "r128_cera_lr3e-4" CeRA 128 "3e-4"
handle_cell "r128_cera_lr5e-4" CeRA 128 "5e-4"
handle_cell "r128_cera_lr1e-3" CeRA 128 "1e-3"
handle_cell "r128_lora_lr1e-4" LoRA 128 "1e-4"
handle_cell "r128_lora_lr3e-4" LoRA 128 "3e-4"
handle_cell "r128_lora_lr5e-4" LoRA 128 "5e-4"
handle_cell "r128_lora_lr1e-3" LoRA 128 "1e-3"
handle_cell "r128_dora_lr1e-4" DoRA 128 "1e-4"
handle_cell "r128_dora_lr3e-4" DoRA 128 "3e-4"
handle_cell "r128_dora_lr5e-4" DoRA 128 "5e-4"
handle_cell "r128_dora_lr1e-3" DoRA 128 "1e-3"

echo ""
echo "--- r=512 LoRA (parameter budget baseline) ---"
handle_cell "r512_lora_lr3e-4" LoRA 512 "3e-4"
handle_cell "r512_lora_lr5e-4" LoRA 512 "5e-4"
handle_cell "r512_lora_lr1e-4" LoRA 512 "1e-4"

echo ""
echo "======================================================"
echo " Submission complete. Monitor with: squeue -u \$USER"
echo "======================================================"

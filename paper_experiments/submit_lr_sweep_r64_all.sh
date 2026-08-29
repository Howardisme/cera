#!/bin/bash
# LR-fair sweep for the Phase 2b canonical baseline.
#
# WHY: the "best LR" table baked into slurm/run_train.sh (3e-4 for 8B) was tuned
# on the OLD q/v + MathInstruct regime and NEVER re-validated for all_linear +
# MetaMathQA. r=64 all_linear LoRA currently sits at GSM8K 68.84 — below the
# 72-78 gate — consistent with 3e-4 being too high for the larger all_linear
# param count. This sweep adds the two lower points the old grid never tried
# ({1e-4, 2e-4}); combined with the existing 3e-4 run it gives a 3-point LR
# curve per method to pick each method's fair optimum.
#
# Fixed:  r=64, all_linear, MetaMathQA-40K, 3 epochs, seed=42, scale-matched
#         (alpha = rank = 64 for LoRA/DoRA). Only LR varies.
# Cost:   6 cells x ~NT$325 (train+eval) ~= NT$2,000, ~30 GPU-h.
#
# Usage:
#   bash paper_experiments/submit_lr_sweep_r64_all.sh --dry-run   # preview
#   bash paper_experiments/submit_lr_sweep_r64_all.sh             # submit
#
# NOTE: 3e-4 is intentionally NOT resubmitted (already done in Phase 2b — would
# just retrain and waste money). If 1e-4 wins (boundary), add a 7e-5 point after
# — but 7e-5 => folder lr7e-05, so add "7e-5") LR_DECIMAL="0.00007" to
# run_eval.sh's case block first, same gotcha as 2e-4.

set -euo pipefail

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

BASE_MODEL="meta-llama/Llama-3.1-8B"
RANK=64
DATASET="metamathqa"
EPOCHS=3
TARGETS="all_linear"
SEED=42
EXCLUDE="--exclude=25a-hgpn026"

# method  lr    dropout  alpha
declare -a CELLS=(
    "LoRA 1e-4 0.0 64"
    "LoRA 2e-4 0.0 64"
    "DoRA 1e-4 0.0 64"
    "DoRA 2e-4 0.0 64"
    "CeRA 1e-4 0.1 32"
    "CeRA 2e-4 0.1 32"
)

echo "=== LR-fair sweep r=64 all_linear MetaMathQA (seed=$SEED) ==="
[ "$DRY_RUN" = "1" ] && echo "(dry-run: nothing submitted)"

for line in "${CELLS[@]}"; do
    read -r method lr dropout alpha <<< "$line"
    m=$(echo "$method" | tr '[:upper:]' '[:lower:]')

    if [ "$method" = "CeRA" ]; then
        cell_id="r${RANK}_${m}_lr${lr}_mm_all"
    else
        cell_id="r${RANK}_${m}_a${alpha}_lr${lr}_mm_all"
    fi

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [would submit] $method R=$RANK lr=$lr D=$dropout A=$alpha -> cell=$cell_id"
        continue
    fi

    TRAIN_JID=$(sbatch --parsable $EXCLUDE slurm/run_train.sh \
        "$method" "$RANK" "$lr" "$dropout" "$DATASET" "$EPOCHS" \
        "$BASE_MODEL" "$alpha" "$TARGETS" "$SEED")

    EVAL_JID=$(sbatch --parsable --dependency=afterok:${TRAIN_JID} $EXCLUDE \
        slurm/run_eval.sh "$cell_id" "$method" "$RANK" "$lr" "$dropout" \
        "$BASE_MODEL" results/eval_outputs "$DATASET" "$alpha" "$TARGETS" "$SEED")

    echo "  $method lr=$lr -> train=$TRAIN_JID eval=$EVAL_JID  cell=$cell_id"
done

echo "=== done. 6 cells = 12 jobs (QoS limit is 20, OK to submit at once) ==="

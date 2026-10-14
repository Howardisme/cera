#!/bin/bash
# Appendix-A style LR sweep. CeRA variants use the PEFT-aligned pipeline
# (train_peft.py); LoRA/DoRA use the paper's own wrappers (train.py, legacy trainer).
#
# Grid: METHODS x RANKS x LRS, single seed. Select the best LR per (method, rank)
# by min held-out val loss (test_loss in <METHOD>_log.json, MetaMathQA rows
# 40000:42000), not by MATH-500, then report MATH-500 / GSM8K at that LR.
#
# Defaults (Stage B): Llama-3.1-8B, MetaMathQA-40K, q_proj+v_proj, 3 epochs,
#        alpha = rank (s = 1), same dropout for Identity / SiLU / learned-mix.
# Eval:  MATH-500 pass@1 + GSM8K pass@1 (pass@10 and MATH-Hard skipped).
#
# Usage:
#   bash paper_experiments/submit_appendix_a_lr_sweep_peft.sh --dry-run
#   bash paper_experiments/submit_appendix_a_lr_sweep_peft.sh
#   RANKS="32 64" LRS="1e-4 5e-4" METHODS="CeRA_id CeRA" bash paper_experiments/submit_appendix_a_lr_sweep_peft.sh
#
# METHODS tags: CeRA_id (aligned Identity), CeRA (SiLU), CeRA_mix (learned linear
#               fallback), LoRA / DoRA (paper wrappers in cera/adapters.py).
#
# Rerunning skips cells whose eval outputs already exist; do not resubmit a cell
# whose training finished but eval has not, because the eval selector requires
# exactly one matching checkpoint.

set -euo pipefail

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."
mkdir -p slurm_logs results/eval_outputs

METHODS="${METHODS:-CeRA_id CeRA CeRA_mix}"
RANKS="${RANKS:-32}"
LRS="${LRS:-1e-4 3e-4 5e-4 1e-3}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B}"
DATASET="${DATASET:-metamathqa}"
EPOCHS="${EPOCHS:-3}"
TARGETS="${TARGETS:-q_proj,v_proj}"
SEED="${SEED:-42}"
CERA_DROPOUT="${CERA_DROPOUT:-0.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
GAMMA_INIT="${GAMMA_INIT:-0.0}"
ACCOUNT="${ACCOUNT:-ACD115147}"
EVAL_BASE="results/eval_outputs"
MODEL_TAG="${BASE_MODEL##*/}"

eval_done() {
    [ -f "${EVAL_BASE}/$1/math500_pass1.json" ] && [ -f "${EVAL_BASE}/$1/gsm8k_pass1.json" ]
}

echo "=== Appendix-A LR sweep | model=${BASE_MODEL} data=${DATASET} targets=${TARGETS} seed=${SEED} ==="
[ "$DRY_RUN" = "1" ] && echo "(dry-run: nothing submitted)"

for rank in $RANKS; do
    for tag in $METHODS; do
        for lr in $LRS; do
            alpha="$rank"
            case "$tag" in
                LoRA)     method="LoRA"; dropout="$LORA_DROPOUT"; mix="-"; act="linear"; pipe="paper" ;;
                DoRA)     method="DoRA"; dropout="$LORA_DROPOUT"; mix="-"; act="linear"; pipe="paper" ;;
                CeRA_id)  method="CeRA"; dropout="$CERA_DROPOUT"; mix="pure"; act="identity"; pipe="aligned" ;;
                CeRA)     method="CeRA"; dropout="$CERA_DROPOUT"; mix="pure"; act="silu"; pipe="aligned" ;;
                CeRA_mix) method="CeRA"; dropout="$CERA_DROPOUT"; mix="learned_mix"; act="silu"; pipe="aligned" ;;
                *) echo "[ERROR] Unknown method tag: $tag" >&2; exit 1 ;;
            esac
            tag_lower=$(echo "$tag" | tr '[:upper:]' '[:lower:]')
            [ "$pipe" = "paper" ] && tag_lower="${tag_lower}_paper"
            cell_id="lrsweep_${DATASET}_${MODEL_TAG}_r${rank}_${tag_lower}_lr${lr}_d${dropout}_s${SEED}"

            if eval_done "$cell_id"; then
                echo "  [SKIP] ${cell_id}: eval outputs exist"
                continue
            fi
            if [ "$DRY_RUN" = "1" ]; then
                echo "  [would submit] ${pipe}: ${method}/${act}/${mix} R=${rank} lr=${lr} D=${dropout} A=${alpha} -> ${cell_id}"
                continue
            fi

            if [ "$pipe" = "paper" ]; then
                train_jid=$(sbatch --parsable --account="$ACCOUNT" slurm/run_train.sh \
                    "$method" "$rank" "$lr" "$dropout" "$DATASET" "$EPOCHS" \
                    "$BASE_MODEL" "$alpha" "$TARGETS" "$SEED")
                eval_jid=$(RUN_MATH_HARD=0 sbatch --parsable --account="$ACCOUNT" \
                    --dependency=afterok:"${train_jid}" \
                    slurm/run_eval.sh "$cell_id" "$method" "$rank" "$lr" "$dropout" \
                    "$BASE_MODEL" "$EVAL_BASE" "$DATASET" "$alpha" "$TARGETS" "$SEED")
                echo "  ${cell_id} -> train=${train_jid} eval=${eval_jid}"
                continue
            fi

            export CERA_VARIANT=peft_aligned CERA_MIX_MODE="$mix" GAMMA_INIT="$GAMMA_INIT" ACT_FN="$act"
            train_jid=$(sbatch --parsable --account="$ACCOUNT" slurm/run_train_peft.sh \
                "$method" "$rank" "$lr" "$dropout" "$DATASET" "$EPOCHS" \
                "$BASE_MODEL" "$alpha" "$TARGETS" "$SEED")
            eval_jid=$(sbatch --parsable --account="$ACCOUNT" --dependency=afterok:"${train_jid}" \
                slurm/run_eval_peft.sh "$cell_id" "$method" "$rank" "$lr" "$dropout" \
                "$BASE_MODEL" "$EVAL_BASE" "$DATASET" "$alpha" "$TARGETS" "$SEED")
            echo "  ${cell_id} -> train=${train_jid} eval=${eval_jid}"
        done
    done
done

echo "=== done ==="

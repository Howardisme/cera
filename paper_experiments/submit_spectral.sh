#!/bin/bash
# Submit spectral analysis jobs: SVD spectrum + Effective Rank trajectory.
# Also submits the rank scaling training sweep (SlimOrca, 7 ranks) and the
# rank_scaling plot job that produces Fig. 1 (PPL + manifold dim vs rank).
#
# Workflow:
#   1. Train CeRA and LoRA on SlimOrca across 7 ranks (R8..R512) -- for Fig. 1 left
#   2. SVD spectrum analysis on those checkpoints              -- for Fig. 1 right
#   3. plot_rank_scaling.py: PPL + manifold dim vs rank        -- Fig. 1 output
#   4. SVD spectrum batch analysis across all math experiments -- Fig. 2/3
#   5. ER trajectory: manifold expansion (CeRA vs LoRA R64)   -- Fig. 2/3
#
# Usage:
#   bash paper_experiments/submit_spectral.sh [--dry-run]

set -e

DRY_RUN=0
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=1
    echo "[DRY-RUN] No jobs will actually be submitted."
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."
mkdir -p slurm_logs results

BASE_MODEL="meta-llama/Llama-3.1-8B"
RANK_SCALING_CONFIG="slurm/configs/sweep_orca_rank_scaling.txt"
RANK_SCALING_TOTAL=14
SVD_ORCA_JSON="results/svd_spectra_orca.json"
RANK_SCALING_OUT="results/rank_scaling.pdf"

# -- Edit these paths to your best math checkpoints -----------------------
CERA_R64_PATH="results/Exp_CeRA_math_R64_lr0.0003_silu_q_proj_v_proj_D0.1_E3_*/CeRA"
LORA_R64_PATH="results/Exp_LoRA_math_R64_lr0.0003_silu_q_proj_v_proj_D0.0_E3_*/LoRA"
# -------------------------------------------------------------------------

echo "======================================================"
echo " Spectral Analysis + Rank Scaling (Fig. 1)"
echo " $(date)"
echo "======================================================"

# ── 1. Rank scaling training: SlimOrca x 7 ranks x 2 methods ─────────────────
echo ""
echo "--- Submitting rank scaling training (14 cells, orca) ---"
BATCH1_END=8
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY] sbatch --array=1-${BATCH1_END}%5 --partition=8gpus slurm/run_train_array.sh ${RANK_SCALING_CONFIG}"
    echo "[DRY] sbatch --dependency=afterany:<BATCH1> slurm/run_resubmitter.sh ... (tasks 9-14)"
    TRAIN_JOB="DRY_TRAIN"
else
    BATCH1=$(sbatch --parsable --array=1-${BATCH1_END}%5 --partition=8gpus \
        slurm/run_train_array.sh "$RANK_SCALING_CONFIG")
    echo "[SUBMIT] Rank scaling batch 1 -> job ${BATCH1} (tasks 1-8)"

    RESUB=$(sbatch --parsable --partition=8gpus \
        --dependency=afterany:${BATCH1} \
        slurm/run_resubmitter.sh "$RANK_SCALING_CONFIG" 9 "$RANK_SCALING_TOTAL" \
        "$RANK_SCALING_TOTAL" 8 8gpus)
    echo "[SUBMIT] Resubmitter -> job ${RESUB} (tasks 9-14)"
    TRAIN_JOB="$BATCH1"
fi

# ── 2. SVD spectrum on orca rank scaling checkpoints ─────────────────────────
echo ""
echo "--- Submitting SVD spectrum on orca rank scaling checkpoints ---"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY] sbatch --dependency=afterany:<TRAIN> slurm/run_analysis.sh svd --base_dir results --dataset orca --output_json ${SVD_ORCA_JSON}"
    SVD_JOB="DRY_SVD"
else
    SVD_JOB=$(sbatch --parsable \
        --dependency=afterany:${TRAIN_JOB} \
        slurm/run_analysis.sh svd \
        --base_dir results \
        --dataset orca \
        --base_model "$BASE_MODEL" \
        --output_json "$SVD_ORCA_JSON")
    echo "[SUBMIT] SVD (orca) -> job ${SVD_JOB}"
fi

# ── 3. plot_rank_scaling: Fig. 1 ─────────────────────────────────────────────
echo ""
echo "--- Submitting rank scaling plot (Fig. 1) ---"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY] sbatch --dependency=afterany:<SVD> slurm/run_analysis.sh rank_scaling --metric both --results_dir results --dataset orca --svd_json ${SVD_ORCA_JSON} --output ${RANK_SCALING_OUT}"
else
    PLOT_JOB=$(sbatch --parsable \
        --dependency=afterany:${SVD_JOB} \
        slurm/run_analysis.sh rank_scaling \
        --metric both \
        --results_dir results \
        --dataset orca \
        --svd_json "$SVD_ORCA_JSON" \
        --output "$RANK_SCALING_OUT")
    echo "[SUBMIT] Rank scaling plot -> job ${PLOT_JOB}"
    echo "  Output: ${RANK_SCALING_OUT}"
fi

# ── 4. SVD spectrum batch analysis (math experiments) ────────────────────────
echo ""
echo "--- Submitting SVD spectrum analysis (math) ---"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY] sbatch slurm/run_analysis.sh svd --base_dir results --dataset math --output_json results/svd_spectra.json"
else
    SVD_MATH_JID=$(sbatch --parsable \
        slurm/run_analysis.sh svd \
        --base_dir results \
        --dataset math \
        --base_model "$BASE_MODEL" \
        --output_json results/svd_spectra.json)
    echo "[SUBMIT] SVD (math) -> job ${SVD_MATH_JID}"
fi

# ── 5. ER trajectory: manifold expansion ─────────────────────────────────────
echo ""
echo "--- Submitting ER trajectory (manifold expansion) ---"
CERA_RESOLVED=$(ls -d ${CERA_R64_PATH} 2>/dev/null | head -1)
LORA_RESOLVED=$(ls -d ${LORA_R64_PATH} 2>/dev/null | head -1)

if [ -z "$CERA_RESOLVED" ] || [ -z "$LORA_RESOLVED" ]; then
    echo "[WARN] CeRA or LoRA R64 math path not found -- update CERA_R64_PATH / LORA_R64_PATH."
    echo "  CeRA: ${CERA_RESOLVED:-NOT FOUND}"
    echo "  LoRA: ${LORA_RESOLVED:-NOT FOUND}"
else
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY] sbatch slurm/run_analysis.sh er --mode manifold --rank 64 --cera_path $CERA_RESOLVED --lora_path $LORA_RESOLVED"
    else
        ER_JID=$(sbatch --parsable \
            slurm/run_analysis.sh er \
            --mode manifold \
            --rank 64 \
            --base_model "$BASE_MODEL" \
            --cera_path "$CERA_RESOLVED" \
            --lora_path "$LORA_RESOLVED")
        echo "[SUBMIT] ER manifold expansion -> job ${ER_JID}"
    fi
fi

echo ""
echo "======================================================"
echo " Submission complete. Monitor with: squeue -u \$USER"
echo " Fig. 1 output: ${RANK_SCALING_OUT}"
echo " SVD spectra:   ${SVD_ORCA_JSON}, results/svd_spectra.json"
echo "======================================================"

#!/bin/bash
# Submit spectral analysis jobs: SVD spectrum + Effective Rank trajectory.
#
# Requires training checkpoints to be available in results/.
# Submits two Slurm jobs:
#   1. SVD spectrum batch analysis across all math experiments
#   2. ER trajectory: manifold expansion (CeRA vs LoRA)
#
# Usage:
#   bash paper_experiments/submit_spectral.sh [--dry-run]
#
# Edit the paths below to match your best checkpoints.

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

# -- Edit these paths to your best checkpoints ----------------------------
CERA_R64_PATH="results/Exp_CeRA_math_R64_lr0.0005_silu_q_proj_v_proj_D0.1_E3_*/CeRA"
LORA_R64_PATH="results/Exp_LoRA_math_R64_lr0.0005_silu_q_proj_v_proj_D0.0_E3_*/LoRA"
# -------------------------------------------------------------------------

echo "======================================================"
echo " Spectral Analysis: SVD spectrum + Effective Rank"
echo " $(date)"
echo "======================================================"

# ── 1. SVD spectrum batch analysis ────────────────────────────────────────────
echo ""
echo "--- Submitting SVD spectrum analysis ---"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY] sbatch slurm/run_analysis.sh svd --base_dir results --dataset math --output_json results/svd_spectra.json"
else
    SVD_JID=$(sbatch --parsable \
        slurm/run_analysis.sh svd \
        --base_dir results \
        --dataset math \
        --base_model "$BASE_MODEL" \
        --output_json results/svd_spectra.json)
    echo "[SUBMIT] SVD spectrum -> job ${SVD_JID}"
fi

# ── 2. ER trajectory: manifold expansion ──────────────────────────────────────
echo ""
echo "--- Submitting ER trajectory (manifold expansion) ---"
CERA_RESOLVED=$(ls -d ${CERA_R64_PATH} 2>/dev/null | head -1)
LORA_RESOLVED=$(ls -d ${LORA_R64_PATH} 2>/dev/null | head -1)

if [ -z "$CERA_RESOLVED" ] || [ -z "$LORA_RESOLVED" ]; then
    echo "[WARN] CeRA or LoRA R64 path not found -- update CERA_R64_PATH / LORA_R64_PATH in this script."
    echo "  CeRA path: ${CERA_RESOLVED:-NOT FOUND}"
    echo "  LoRA path: ${LORA_RESOLVED:-NOT FOUND}"
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
echo " Results: results/svd_spectra.json, stdout of ER job"
echo "======================================================"

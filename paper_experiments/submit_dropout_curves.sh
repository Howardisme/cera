#!/bin/bash
# Submit the dropout robustness experiment (paper Fig. 2):
# validation PPL curves on SlimOrca for CeRA D{0.0,0.1,0.2,0.3} + LoRA,
# under suboptimal (1e-4) and optimal (5e-4) learning rates. Rank 128.
#
# Workflow:
#   1. Train the 10 cells in slurm/configs/sweep_orca_dropout_curves.txt
#   2. Plot validation PPL vs data seen, one panel per LR -> Fig. 2
#
# NOTE: 10 training tasks + 1 plot job exceeds the 10-job QOS limit, so the
# plot submission may fail with QOSMaxSubmitJobPerUserLimit if the queue is
# otherwise empty. In that case the script prints the command to run later.
#
# Usage:
#   bash paper_experiments/submit_dropout_curves.sh [--dry-run]

set -e

DRY_RUN=0
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=1
    echo "[DRY-RUN] No jobs will actually be submitted."
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."
mkdir -p slurm_logs results

CONFIG="slurm/configs/sweep_orca_dropout_curves.txt"
TOTAL=$(grep -v '^\s*#' "$CONFIG" | grep -cv '^\s*$')
OUTPUT="results/training_curves_dropout.pdf"

echo "======================================================"
echo " Dropout Robustness Curves (Fig. 2)"
echo " $(date)"
echo "======================================================"

echo ""
echo "--- Submitting training (${TOTAL} cells, orca) ---"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY] sbatch --array=1-${TOTAL}%5 --partition=8gpus slurm/run_train_array.sh ${CONFIG}"
    echo "[DRY] sbatch --dependency=afterany:<TRAIN> slurm/run_analysis.sh curves --results_dir results --dataset orca --rank 128 --lrs 1e-4 5e-4 --output ${OUTPUT}"
    exit 0
fi

TRAIN_JOB=$(sbatch --parsable --array=1-${TOTAL}%5 --partition=8gpus \
    slurm/run_train_array.sh "$CONFIG")
echo "[SUBMIT] Training -> job ${TRAIN_JOB} (tasks 1-${TOTAL})"

if PLOT_JOB=$(sbatch --parsable --dependency=afterany:${TRAIN_JOB} \
        slurm/run_analysis.sh curves \
        --results_dir results --dataset orca --rank 128 \
        --lrs 1e-4 5e-4 --output "$OUTPUT" 2>/dev/null); then
    echo "[SUBMIT] Curves plot -> job ${PLOT_JOB}"
    echo "  Output: ${OUTPUT}"
else
    echo "[WARN] Plot job submission failed (likely QOS limit). Run later:"
    echo "  sbatch --dependency=afterany:${TRAIN_JOB} slurm/run_analysis.sh curves \\"
    echo "      --results_dir results --dataset orca --rank 128 \\"
    echo "      --lrs 1e-4 5e-4 --output ${OUTPUT}"
fi

echo ""
echo "======================================================"
echo " Fig. 2 output: ${OUTPUT}"
echo "======================================================"

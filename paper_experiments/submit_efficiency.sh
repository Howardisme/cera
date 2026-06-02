#!/bin/bash
# Submit inference throughput / latency benchmark job.
#
# Measures token generation speed for CeRA R=64/128 vs LoRA R=512 on the
# base model. No checkpoint loading required -- adapters are randomly initialized
# for the latency benchmark (weights do not affect generation speed).
#
# Usage:
#   bash paper_experiments/submit_efficiency.sh [--dry-run]

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

echo "======================================================"
echo " Efficiency Benchmark: CeRA vs LoRA throughput"
echo " $(date)"
echo "======================================================"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY] sbatch slurm/run_analysis.sh benchmark --base_model $BASE_MODEL --output_csv results/efficiency_results.csv"
else
    JID=$(sbatch --parsable \
        slurm/run_analysis.sh benchmark \
        --base_model "$BASE_MODEL" \
        --output_csv results/efficiency_results.csv)
    echo "[SUBMIT] Efficiency benchmark -> job ${JID}"
    echo "  Output: results/efficiency_results.csv"
fi

echo ""
echo "======================================================"
echo " Submission complete. Monitor with: squeue -u \$USER"
echo "======================================================"

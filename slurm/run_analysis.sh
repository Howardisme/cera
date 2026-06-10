#!/bin/bash
#SBATCH --job-name=cera_analysis
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/analysis_%j.log
#SBATCH --error=slurm_logs/analysis_err_%j.log
#SBATCH --partition=normal
# NOTE: Add your cluster account line here, e.g.:
#   #SBATCH -A YOUR_ACCOUNT

# Dispatch analysis jobs: SVD spectrum, Effective Rank, or benchmark.
#
# Usage:
#   sbatch slurm/run_analysis.sh ANALYSIS_TYPE [args...]
#
# ANALYSIS_TYPE:
#   svd            -- SVD spectrum analysis (analyze_svd.py)
#   er             -- Effective Rank trajectory (analyze_er.py)
#   benchmark      -- Throughput/latency benchmark (benchmark.py)
#   rank_scaling   -- PPL + manifold dimensionality vs rank (plot_rank_scaling.py)
#
# All additional arguments are forwarded to the corresponding Python script.
#
# Examples:
#   sbatch slurm/run_analysis.sh svd --base_dir results --dataset math
#   sbatch slurm/run_analysis.sh er --mode manifold --lora_path results/.../LoRA --cera_path results/.../CeRA
#   sbatch slurm/run_analysis.sh benchmark --output_csv results/efficiency_results.csv

ANALYSIS_TYPE=${1}
shift

if [ -z "$ANALYSIS_TYPE" ]; then
    echo "[ERROR] Usage: $0 ANALYSIS_TYPE [args...]"
    echo "  ANALYSIS_TYPE: svd | er | benchmark"
    exit 1
fi

# Activate your environment here, e.g.:
#   conda activate cera_env
# or:
#   source /path/to/venv/bin/activate

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — run via sbatch}"
mkdir -p slurm_logs results

echo "[START] Analysis: ${ANALYSIS_TYPE} | Job ID: ${SLURM_JOB_ID:-local} | $(date)"

case "$ANALYSIS_TYPE" in
    svd)
        python analysis/analyze_svd.py "$@"
        ;;
    er)
        python analysis/analyze_er.py "$@"
        ;;
    benchmark)
        python analysis/benchmark.py "$@"
        ;;
    rank_scaling)
        python analysis/plot_rank_scaling.py "$@"
        ;;
    *)
        echo "[ERROR] Unknown analysis type: ${ANALYSIS_TYPE}. Choose from svd, er, benchmark, rank_scaling."
        exit 1
        ;;
esac

echo "[END] Finished at $(date)"

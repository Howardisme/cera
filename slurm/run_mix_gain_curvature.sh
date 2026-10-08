#!/bin/bash
#SBATCH --job-name=cera_gain_curv
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/gain_curv_%j.log
#SBATCH --error=slurm_logs/gain_curv_err_%j.log
#SBATCH --partition=normal

# Gain / offset / curvature same-checkpoint interventions on held-out NLL
# for one learned-mix or pure nonlinear (peft_aligned) CeRA checkpoint.
#
# Usage:
#   sbatch slurm/run_mix_gain_curvature.sh RUN_ID CHECKPOINT \
#       [NUM_SAMPLES] [NUM_CALIBRATION] [DTYPE]
#
# DTYPE: float32 | bfloat16

set -euo pipefail

RUN_ID=${1:-}
CHECKPOINT=${2:-}
NUM_SAMPLES=${3:-500}
NUM_CALIBRATION=${4:-200}
DTYPE=${5:-float32}

if [ -z "$RUN_ID" ] || [ -z "$CHECKPOINT" ]; then
    echo "[ERROR] Missing required arguments."
    echo "Usage: $0 RUN_ID CHECKPOINT [NUM_SAMPLES] [NUM_CALIBRATION] [DTYPE]"
    exit 1
fi
case "$DTYPE" in
    float32|bfloat16) ;;
    *) echo "[ERROR] DTYPE must be float32 or bfloat16."; exit 1 ;;
esac

module purge
module load singularity

SIF="${CERA_SIF:-/work/$USER/cera.sif}"
PYPKGS="${CERA_PYPKGS:-/work/$USER/cera_pypkgs}"
HF_CACHE="${CERA_HF_HOME:-/work/$USER/hf_cache}"
OUTPUT_ROOT="${GAIN_CURVATURE_OUTPUT:-results/mix_gain_curvature}"
OUTPUT="${OUTPUT_ROOT}/${RUN_ID}_n${NUM_SAMPLES}_${DTYPE}.json"
BOOTSTRAP=${BOOTSTRAP:-2000}
MAX_LENGTH=${MAX_LENGTH:-512}
BATCH_SIZE=${BATCH_SIZE:-4}
FIT_SCOPE=${FIT_SCOPE:-all}

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set - submit with sbatch}"
mkdir -p slurm_logs "$HF_CACHE" "$OUTPUT_ROOT"

echo "[START] gain/curvature | run=${RUN_ID} n=${NUM_SAMPLES} cal=${NUM_CALIBRATION} dtype=${DTYPE}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Output:     ${OUTPUT}"

singularity exec --nv -B /work \
    --env PYTHONPATH="$PYPKGS" \
    --env PYTHONNOUSERSITE=1 \
    --env HF_HOME="$HF_CACHE" \
    "$SIF" \
    python analysis/analyze_mix_gain_curvature.py \
        --checkpoint "$CHECKPOINT" \
        --output "$OUTPUT" \
        --num_samples "$NUM_SAMPLES" \
        --num_calibration "$NUM_CALIBRATION" \
        --dtype "$DTYPE" \
        --fit_scope "$FIT_SCOPE" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --bootstrap "$BOOTSTRAP"

echo "[END] gain/curvature finished | $(date)"

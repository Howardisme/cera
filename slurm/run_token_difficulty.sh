#!/bin/bash
#SBATCH --job-name=cera_token_diff
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/token_diff_%j.log
#SBATCH --error=slurm_logs/token_diff_err_%j.log
#SBATCH --partition=normal

# Teacher-forced token difficulty analysis for one matched LoRA/CeRA pair.
#
# Usage:
#   sbatch slurm/run_token_difficulty.sh \
#       RUN_ID LORA_CHECKPOINT CERA_CHECKPOINT \
#       [DATASET] [NUM_SAMPLES] [RANK] [ALPHA] [DROPOUT] \
#       [TARGET_MODULES] [BASE_MODEL] [LORA_FORMAT]
#       [LINEAR_ADAPTER_TYPE] [LINEAR_ACT_FN]
#
# DATASET: metamathqa | gsm8k | math500
# LORA_FORMAT: peft | legacy

set -euo pipefail

RUN_ID=${1:-}
LORA_CHECKPOINT=${2:-}
CERA_CHECKPOINT=${3:-}
DATASET=${4:-metamathqa}
NUM_SAMPLES=${5:-200}
RANK=${6:-64}
ALPHA=${7:-64}
DROPOUT=${8:-0.1}
TARGET_MODULES=${9:-q_proj,v_proj}
BASE_MODEL=${10:-meta-llama/Llama-3.1-8B}
LORA_FORMAT=${11:-peft}
LINEAR_ADAPTER_TYPE=${12:-lora}
LINEAR_ACT_FN=${13:-identity}

if [ -z "$RUN_ID" ] || [ -z "$LORA_CHECKPOINT" ] || [ -z "$CERA_CHECKPOINT" ]; then
    echo "[ERROR] Missing required arguments."
    echo "Usage: $0 RUN_ID LORA_CHECKPOINT CERA_CHECKPOINT [DATASET] [NUM_SAMPLES] [RANK] [ALPHA] [DROPOUT] [TARGET_MODULES] [BASE_MODEL] [LORA_FORMAT]"
    exit 1
fi

if [ "$TARGET_MODULES" = "all_linear" ]; then
    TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
fi

case "$DATASET" in
    metamathqa|gsm8k|math500) ;;
    *) echo "[ERROR] Unsupported dataset: $DATASET"; exit 1 ;;
esac
case "$LORA_FORMAT" in
    peft|legacy) ;;
    *) echo "[ERROR] LORA_FORMAT must be peft or legacy."; exit 1 ;;
esac
case "$LINEAR_ADAPTER_TYPE" in
    lora|cera) ;;
    *) echo "[ERROR] LINEAR_ADAPTER_TYPE must be lora or cera."; exit 1 ;;
esac

module purge
module load singularity

SIF="${CERA_SIF:-/work/$USER/cera.sif}"
PYPKGS="${CERA_PYPKGS:-/work/$USER/cera_pypkgs}"
HF_CACHE="${CERA_HF_HOME:-/work/$USER/hf_cache}"
OUTPUT_ROOT="${TOKEN_DIFFICULTY_OUTPUT:-results/token_difficulty}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}_${DATASET}_n${NUM_SAMPLES}"
BOOTSTRAP=${BOOTSTRAP:-2000}
MAX_LENGTH=${MAX_LENGTH:-512}
BATCH_SIZE=${BATCH_SIZE:-1}

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set - submit with sbatch}"
mkdir -p slurm_logs "$HF_CACHE" "$OUTPUT_DIR"

echo "[START] token difficulty | run=${RUN_ID} dataset=${DATASET} n=${NUM_SAMPLES}"
echo "  LoRA:   ${LORA_CHECKPOINT} (${LORA_FORMAT})"
echo "  CeRA:   ${CERA_CHECKPOINT}"
echo "  Output: ${OUTPUT_DIR}"

singularity exec --nv -B /work \
    --env PYTHONPATH="$PYPKGS" \
    --env PYTHONNOUSERSITE=1 \
    --env HF_HOME="$HF_CACHE" \
    "$SIF" \
    python analysis/analyze_token_difficulty.py \
        --base_model "$BASE_MODEL" \
        --dataset "$DATASET" \
        --lora_checkpoint "$LORA_CHECKPOINT" \
        --cera_checkpoint "$CERA_CHECKPOINT" \
        --lora_adapter_format "$LORA_FORMAT" \
        --linear_adapter_type "$LINEAR_ADAPTER_TYPE" \
        --linear_act_fn "$LINEAR_ACT_FN" \
        --rank "$RANK" \
        --alpha "$ALPHA" \
        --dropout "$DROPOUT" \
        --act_fn silu \
        --target_modules "$TARGET_MODULES" \
        --num_samples "$NUM_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --bootstrap "$BOOTSTRAP" \
        --output_dir "$OUTPUT_DIR"

echo "[END] token difficulty finished | $(date)"
#!/bin/bash
#SBATCH --job-name=cera_train_peft
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/train_peft_%j.log
#SBATCH --error=slurm_logs/train_peft_err_%j.log
#SBATCH --partition=8gpus

# Companion to slurm/run_train.sh -- invokes train_peft.py (HF Trainer + PEFT
# + TRL pipeline) instead of train.py.  Same positional interface.
#
# Usage:
#   sbatch slurm/run_train_peft.sh MODEL_TYPE RANK LR DROPOUT DATASET \
#       [EPOCHS] [BASE_MODEL] [ALPHA] [TARGET_MODULES] [SEED] [ATTN_IMPL]

MODEL_TYPE=${1:-LoRA}
RANK=${2:-64}
LR=${3:-1e-4}
DROPOUT=${4:-0.0}
DATASET=${5:-metamathqa}
EPOCHS=${6:-3}
BASE_MODEL=${7:-meta-llama/Llama-3.1-8B}
ALPHA=${8:-64}
TARGET_MODULES=${9:-all_linear}
SEED=${10:-42}
ATTN_IMPL=${11:-sdpa}

if [ "$TARGET_MODULES" = "all_linear" ]; then
    TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
fi

module purge
module load singularity

SIF="${CERA_SIF:-/work/$USER/cera.sif}"
PYPKGS="${CERA_PYPKGS:-/work/$USER/cera_pypkgs}"
HF_CACHE="${CERA_HF_HOME:-/work/$USER/hf_cache}"
mkdir -p "$HF_CACHE"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — run via sbatch}"
mkdir -p slurm_logs

echo "[START] Job ID: ${SLURM_JOB_ID:-local} | pipeline=PEFT | ${MODEL_TYPE} R=${RANK} lr=${LR} D=${DROPOUT} A=${ALPHA} dataset=${DATASET} E=${EPOCHS} model=${BASE_MODEL} targets=${TARGET_MODULES} seed=${SEED} attn=${ATTN_IMPL}"

singularity exec --nv -B /work \
    --env PYTHONPATH="$PYPKGS" \
    --env PYTHONNOUSERSITE=1 \
    --env HF_HOME="$HF_CACHE" \
    "$SIF" \
    python train_peft.py \
        --model_type      "$MODEL_TYPE" \
        --rank            "$RANK" \
        --lr              "$LR" \
        --dropout         "$DROPOUT" \
        --dataset         "$DATASET" \
        --epochs          "$EPOCHS" \
        --base_model      "$BASE_MODEL" \
        --alpha           "$ALPHA" \
        --target_modules  "$TARGET_MODULES" \
        --seed            "$SEED" \
        --attn_impl       "$ATTN_IMPL"

echo "[END] Finished at $(date)"

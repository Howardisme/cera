#!/bin/bash
#SBATCH --job-name=cera_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/train_%j.log
#SBATCH --error=slurm_logs/train_err_%j.log
#SBATCH --partition=normal
# NOTE: Add your cluster account line here, e.g.:
#   #SBATCH -A YOUR_ACCOUNT

# Usage:
#   sbatch slurm/run_train.sh MODEL_TYPE RANK LR DROPOUT DATASET [EPOCHS] [BASE_MODEL]
#
# Arguments:
#   MODEL_TYPE  CeRA | LoRA | DoRA
#   RANK        adapter rank (e.g. 64, 128, 512)
#   LR          learning rate (e.g. 5e-4)
#   DROPOUT     dropout rate (e.g. 0.1 for CeRA, 0.0 for LoRA/DoRA)
#   DATASET     math | code | orca
#   EPOCHS      number of training epochs (default: 3)
#   BASE_MODEL  HuggingFace model ID (default: meta-llama/Llama-3.1-8B)
#
# Example:
#   sbatch slurm/run_train.sh CeRA 128 5e-4 0.1 math 3
#   sbatch slurm/run_train.sh LoRA 128 5e-4 0.0 math 3 meta-llama/Llama-3.2-3B

MODEL_TYPE=${1:-CeRA}
RANK=${2:-128}
LR=${3:-5e-4}
DROPOUT=${4:-0.1}
DATASET=${5:-math}
EPOCHS=${6:-3}
BASE_MODEL=${7:-meta-llama/Llama-3.1-8B}

# Activate your environment here, e.g.:
#   conda activate cera_env
# or:
#   source /path/to/venv/bin/activate

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — run via sbatch}"
mkdir -p slurm_logs

echo "[START] Job ID: ${SLURM_JOB_ID:-local} | ${MODEL_TYPE} R=${RANK} lr=${LR} D=${DROPOUT} dataset=${DATASET} E=${EPOCHS} model=${BASE_MODEL}"

python train.py \
    --model_type  "$MODEL_TYPE" \
    --rank        "$RANK" \
    --lr          "$LR" \
    --dropout     "$DROPOUT" \
    --dataset     "$DATASET" \
    --epochs      "$EPOCHS" \
    --base_model  "$BASE_MODEL"

echo "[END] Finished at $(date)"

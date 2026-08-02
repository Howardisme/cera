#!/bin/bash
#SBATCH --job-name=cera_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/train_%j.log
#SBATCH --error=slurm_logs/train_err_%j.log
#SBATCH --partition=8gpus
# NOTE: Add your cluster account line here, e.g.:
#   #SBATCH -A YOUR_ACCOUNT

# Usage:
#   sbatch slurm/run_train.sh MODEL_TYPE RANK LR DROPOUT DATASET [EPOCHS] [BASE_MODEL] [ALPHA]
#
# Arguments:
#   MODEL_TYPE  CeRA | LoRA | DoRA
#   RANK        adapter rank (e.g. 64, 128, 512)
#   LR          learning rate (e.g. 5e-4), or "best" to use the sweep-selected optimum
#   DROPOUT     dropout rate (e.g. 0.1 for CeRA, 0.0 for LoRA/DoRA)
#   DATASET     math | code | orca
#   EPOCHS      number of training epochs (default: 3)
#   BASE_MODEL  HuggingFace model ID (default: meta-llama/Llama-3.1-8B)
#   ALPHA       LoRA/DoRA alpha; effective scale = alpha / rank (default: 32).
#               Ignored by CeRA. Pass alpha=rank for the scale-matched (s=1) control.
#
# Example:
#   sbatch slurm/run_train.sh CeRA 128 best 0.1 math 3
#   sbatch slurm/run_train.sh LoRA 128 5e-4 0.0 math 3 meta-llama/Llama-3.2-3B
#   sbatch slurm/run_train.sh LoRA 128 5e-4 0.0 math 3 meta-llama/Llama-3.1-8B 128   # scale-matched

MODEL_TYPE=${1:-CeRA}
RANK=${2:-128}
LR=${3:-best}
DROPOUT=${4:-0.1}
DATASET=${5:-math}
EPOCHS=${6:-3}
BASE_MODEL=${7:-meta-llama/Llama-3.1-8B}
ALPHA=${8:-32}

MODEL_TAG=$(echo "$BASE_MODEL" | sed 's|.*/||')

# Best LR per model/method/rank from grid search over {1e-4, 3e-4, 5e-4, 1e-3}.
# 8B: CeRA R64→3e-4, CeRA R128→1e-3, LoRA/DoRA (any rank)→3e-4
# 1B: CeRA R64→3e-4, CeRA R128→3e-4, LoRA R64→3e-4, LoRA R128→5e-4, DoRA R64→1e-3, DoRA R128→1e-3
# 3B: CeRA R64→3e-4, CeRA R128→5e-4, LoRA R64→3e-4, LoRA R128→5e-4, DoRA R64→1e-3, DoRA R128→5e-4
# R512 (1B/3B math): CeRA 1B→3e-4 3B→5e-4, LoRA 1B/3B→5e-4, DoRA 1B/3B→5e-4 (8B R512 math not swept)
if [ "$LR" = "best" ]; then
    case "${MODEL_TAG}_${MODEL_TYPE}_${RANK}" in
        Llama-3.2-1B_CeRA_64)   LR="3e-4" ;;
        Llama-3.2-1B_CeRA_128)  LR="3e-4" ;;
        Llama-3.2-1B_LoRA_64)   LR="3e-4" ;;
        Llama-3.2-1B_LoRA_128)  LR="5e-4" ;;
        Llama-3.2-1B_DoRA_64)   LR="1e-3" ;;
        Llama-3.2-1B_DoRA_128)  LR="1e-3" ;;
        Llama-3.2-3B_CeRA_64)   LR="3e-4" ;;
        Llama-3.2-3B_CeRA_128)  LR="5e-4" ;;
        Llama-3.2-3B_LoRA_64)   LR="3e-4" ;;
        Llama-3.2-3B_LoRA_128)  LR="5e-4" ;;
        Llama-3.2-3B_DoRA_64)   LR="1e-3" ;;
        Llama-3.2-3B_DoRA_128)  LR="5e-4" ;;
        Llama-3.2-1B_CeRA_512)  LR="3e-4" ;;
        Llama-3.2-1B_LoRA_512)  LR="5e-4" ;;
        Llama-3.2-1B_DoRA_512)  LR="5e-4" ;;
        Llama-3.2-3B_CeRA_512)  LR="5e-4" ;;
        Llama-3.2-3B_LoRA_512)  LR="5e-4" ;;
        Llama-3.2-3B_DoRA_512)  LR="5e-4" ;;
        *_CeRA_64)               LR="3e-4" ;;
        *_CeRA_128)              LR="1e-3" ;;
        *_LoRA_*)                LR="3e-4" ;;
        *_DoRA_*)                LR="3e-4" ;;
        *)
            echo "[ERROR] No best LR defined for MODEL=${MODEL_TAG} METHOD=${MODEL_TYPE} RANK=${RANK}. Pass LR explicitly."
            exit 1 ;;
    esac
    echo "[INFO] Resolved LR=best → ${LR} for ${MODEL_TAG} ${MODEL_TYPE} R=${RANK}"
fi

ml load miniconda3 cuda/12.6
source activate cera

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — run via sbatch}"
mkdir -p slurm_logs

echo "[START] Job ID: ${SLURM_JOB_ID:-local} | ${MODEL_TYPE} R=${RANK} lr=${LR} D=${DROPOUT} A=${ALPHA} dataset=${DATASET} E=${EPOCHS} model=${BASE_MODEL}"

python train.py \
    --model_type  "$MODEL_TYPE" \
    --rank        "$RANK" \
    --lr          "$LR" \
    --dropout     "$DROPOUT" \
    --dataset     "$DATASET" \
    --epochs      "$EPOCHS" \
    --base_model  "$BASE_MODEL" \
    --alpha       "$ALPHA"

echo "[END] Finished at $(date)"

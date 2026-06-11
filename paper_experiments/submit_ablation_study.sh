#!/bin/bash
# Submit ablation study jobs: activation function, target modules, dropout.
#
# Variants at R=128 and R=512 on SlimOrca (following the original paper setup):
#   Full          SiLU, q_proj+v_proj, Dropout=0.3  (CeRA default)
#   No-Dropout    SiLU, q_proj+v_proj, Dropout=0.0
#   ReLU          ReLU, q_proj+v_proj, Dropout=0.3
#   Identity      Identity, q_proj+v_proj, Dropout=0.3
#   Granularity   SiLU, o_proj only, Dropout=0.3
#
# After all R=128 training jobs complete, ER analysis is chained automatically.
#
# Usage:
#   bash paper_experiments/submit_ablation_study.sh [--dry-run]

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
LR="5e-4"
DATASET="orca"
EPOCHS=3

submit_train() {
    local name=$1; local rank=$2; local act=$3; local dropout=$4; local targets=$5
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY_JOB_ID"
        echo "[DRY] sbatch --job-name=$name slurm/run_train.sh CeRA $rank $LR $dropout $DATASET $EPOCHS $BASE_MODEL" >&2
        echo "      (act_fn=$act target_modules=$targets)" >&2
        return
    fi
    sbatch --parsable --job-name="$name" \
        slurm/run_train.sh CeRA "$rank" "$LR" "$dropout" "$DATASET" "$EPOCHS" "$BASE_MODEL"
    # NOTE: act_fn and target_modules need to be passed directly to train.py.
    # The generic run_train.sh does not expose these args.
    # For ablation, call train.py directly via sbatch --wrap:
}

submit_ablation_train() {
    local name=$1; local rank=$2; local act=$3; local dropout=$4; local targets=$5
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY_JOB_ID"
        printf "[DRY] sbatch --job-name=%s ... python train.py --model_type CeRA --rank %s --lr %s --dropout %s --act_fn %s --target_modules %s --dataset %s --epochs %s --base_model %s\n" \
            "$name" "$rank" "$LR" "$dropout" "$act" "$targets" "$DATASET" "$EPOCHS" "$BASE_MODEL" >&2
        return
    fi
    sbatch --parsable \
        --job-name="$name" \
        --nodes=1 --gres=gpu:1 --cpus-per-task=4 \
        --time=24:00:00 --partition=8gpus \
        --output="slurm_logs/${name}_%j.log" \
        --error="slurm_logs/${name}_err_%j.log" \
        --wrap="$(cat <<EOF
# Activate your environment here
SCRIPT_DIR=\$(pwd)
cd \$SCRIPT_DIR
python train.py \
    --model_type CeRA \
    --rank $rank \
    --lr $LR \
    --dropout $dropout \
    --act_fn $act \
    --target_modules $targets \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --base_model $BASE_MODEL
EOF
)"
}

echo "======================================================"
echo " Ablation Study: activation, target modules, dropout"
echo " $(date)"
echo "======================================================"

echo ""
echo "--- R=128 variants (orca, lr=5e-4) ---"
JID_FULL=$(submit_ablation_train  "ablation_full"    128 "silu"     "0.3" "q_proj,v_proj")
JID_NODROP=$(submit_ablation_train "ablation_nodrop" 128 "silu"     "0.0" "q_proj,v_proj")
JID_RELU=$(submit_ablation_train  "ablation_relu"    128 "relu"     "0.3" "q_proj,v_proj")
JID_IDENT=$(submit_ablation_train "ablation_ident"   128 "identity" "0.3" "q_proj,v_proj")
JID_OMOD=$(submit_ablation_train  "ablation_omod"    128 "silu"     "0.3" "o_proj")

echo ""
echo "--- Chaining ER analysis after all R=128 training jobs ---"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY] sbatch --dependency=afterok:... slurm/run_analysis.sh er --mode ablation --rank 128 ..."
else
    DEP="afterok:${JID_FULL}:${JID_NODROP}:${JID_RELU}:${JID_IDENT}:${JID_OMOD}"
    ER_JID=$(sbatch --parsable --dependency="$DEP" \
        slurm/run_analysis.sh er \
        --mode ablation \
        --rank 128 \
        --base_model "$BASE_MODEL" \
        | tee /dev/stderr)
    echo "[SUBMIT] ER analysis R=128 -> job ${ER_JID}"
fi

echo ""
echo "--- R=512 variants (missing: ReLU, Identity, Granularity) ---"
JID_RELU512=$(submit_ablation_train  "ablation_relu_r512"  512 "relu"     "0.3" "q_proj,v_proj")
JID_IDENT512=$(submit_ablation_train "ablation_ident_r512" 512 "identity" "0.3" "q_proj,v_proj")
JID_OMOD512=$(submit_ablation_train  "ablation_omod_r512"  512 "silu"     "0.3" "o_proj")

echo ""
echo "--- Chaining ER analysis after all R=512 training jobs ---"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY] sbatch --dependency=afterok:... slurm/run_analysis.sh er --mode ablation --rank 512 ..."
else
    DEP512="afterok:${JID_RELU512}:${JID_IDENT512}:${JID_OMOD512}"
    ER_JID512=$(sbatch --parsable --dependency="$DEP512" \
        slurm/run_analysis.sh er \
        --mode ablation \
        --rank 512 \
        --base_model "$BASE_MODEL" \
        | tee /dev/stderr)
    echo "[SUBMIT] ER analysis R=512 -> job ${ER_JID512}"
fi

echo ""
echo "======================================================"
echo " Submission complete. Monitor with: squeue -u \$USER"
echo "======================================================"

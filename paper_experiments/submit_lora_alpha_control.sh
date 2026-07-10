#!/bin/bash
# LoRA alpha=512 (scaling=1) control for the ER ablation.
#
# Purpose
# -------
# The linear-CeRA control showed that even with an identity activation, CeRA
# (a rank-512 linear map B(A(x))) has ER ~349, far above LoRA (~100). The
# two adapters differ in two ways: (a) init scale (CeRA A Kaiming vs LoRA
# lora_A randn*0.01), and (b) LoRA applies scaling = alpha/rank = 32/512 = 0.0625
# while CeRA uses scaling = 1. ER is L1-normalized so the static scale does not
# change a fixed matrix's ER, but the scale reshapes the training dynamics.
#
# This control isolates the scaling hypothesis by training LoRA with alpha=512
# (scaling = 512/512 = 1), everything else identical to the existing LoRA r512
# reference:
#
#   LoRA, rank=512, alpha=512 (scaling=1), dropout=0.0, q_proj+v_proj, orca, lr=5e-4
#
# Measured through the EXACT SAME pipeline (analyze_svd.py -> get_singular_values:
# eval mode, output last_delta, mean-centered, L1-norm SVD, exp-entropy ER) as the
# references, so the numbers are directly comparable:
#
#   LoRA r512 alpha=32  scaling=0.0625 lr5e-4 : ER ~100  (existing)
#   LoRA r512 alpha=512 scaling=1      lr5e-4 : ER ????  (this control)
#   CeRA ident r512 D0.0               lr5e-4 : ER ~349  (existing)
#   CeRA silu  r512 D0.0               lr5e-4 : ER ~388  (existing)
#
# If the alpha=512 LoRA jumps from ~100 toward ~350, the alpha/rank scaling is the
# dominant driver of the ER gap; if it stays ~100, the gap is from init/structure.
#
# Usage:
#   bash paper_experiments/submit_lora_alpha_control.sh [--dry-run]

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
RANK=512
ALPHA=512
LR="5e-4"
DROPOUT="0.0"
TARGETS="q_proj,v_proj"
DATASET="orca"
EPOCHS=3

# Existing reference folders (measured by the same pipeline).
LORA_REF="results/Exp_LoRA_orca_R512_lr0.0005_silu_q_proj_v_proj_D0.0_E3_20260612_1043"
SILU_REF="results/Exp_CeRA_orca_R512_lr0.0005_silu_q_proj_v_proj_D0.0_E3_20260612_1037"
IDENT_REF="results/Exp_CeRA_orca_R512_lr0.0005_identity_q_proj_v_proj_D0.0_E3_20260622_1315"
OUT_JSON="results/svd_control_lora_alpha512.json"

# --- 1. Train the LoRA alpha=512 control --------------------------------------
TRAIN_NAME="ctrl_lora_a512_r512"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY] sbatch --job-name=${TRAIN_NAME} ... python train.py \\"
    echo "        --model_type LoRA --rank ${RANK} --alpha ${ALPHA} --lr ${LR} \\"
    echo "        --dropout ${DROPOUT} --target_modules ${TARGETS} --dataset ${DATASET} \\"
    echo "        --epochs ${EPOCHS} --base_model ${BASE_MODEL}"
    TRAIN_JID="DRY_TRAIN_JID"
else
    TRAIN_JID=$(sbatch --parsable \
        --job-name="${TRAIN_NAME}" \
        --nodes=1 --gres=gpu:1 --cpus-per-task=4 \
        --time=24:00:00 --partition=8gpus \
        --output="slurm_logs/${TRAIN_NAME}_%j.log" \
        --error="slurm_logs/${TRAIN_NAME}_err_%j.log" \
        --wrap="$(cat <<EOF
cd \$SLURM_SUBMIT_DIR
python train.py \
    --model_type LoRA \
    --rank ${RANK} \
    --alpha ${ALPHA} \
    --lr ${LR} \
    --dropout ${DROPOUT} \
    --target_modules ${TARGETS} \
    --dataset ${DATASET} \
    --epochs ${EPOCHS} \
    --base_model ${BASE_MODEL}
EOF
)")
    echo "[SUBMIT] train control -> job ${TRAIN_JID}"
fi

# --- 2. Chained ER measurement (same pipeline, all configs) -------------------
# The alpha=512 run produces a folder with the SAME name pattern as the existing
# alpha=32 LoRA reference (alpha is not in the folder name), differing only by
# timestamp, so we pick the NEWEST matching folder and pass the old ref by full
# path for a side-by-side comparison.
ANALYZE_NAME="ctrl_lora_a512_er"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY] sbatch --dependency=afterok:${TRAIN_JID} --job-name=${ANALYZE_NAME} ..."
    echo "      glob newest results/Exp_LoRA_orca_R512_lr0.0005_silu_q_proj_v_proj_D0.0_E3_*"
    echo "      python analysis/analyze_svd.py --folders <new_lora> ${LORA_REF} ${IDENT_REF} ${SILU_REF} \\"
    echo "          --dataset ${DATASET} --output_json ${OUT_JSON}"
else
    sbatch --parsable \
        --dependency="afterok:${TRAIN_JID}" \
        --job-name="${ANALYZE_NAME}" \
        --nodes=1 --gres=gpu:1 --cpus-per-task=4 \
        --time=12:00:00 --partition=8gpus \
        --output="slurm_logs/${ANALYZE_NAME}_%j.log" \
        --error="slurm_logs/${ANALYZE_NAME}_err_%j.log" \
        --wrap="$(cat <<EOF
cd \$SLURM_SUBMIT_DIR
NEW_LORA=\$(ls -dt results/Exp_LoRA_orca_R512_lr0.0005_silu_q_proj_v_proj_D0.0_E3_* 2>/dev/null | head -1)
if [ -z "\$NEW_LORA" ]; then
    echo "[ERROR] alpha=512 LoRA control folder not found"
    exit 1
fi
echo "[INFO] alpha=512 LoRA control folder: \$NEW_LORA"
python analysis/analyze_svd.py \
    --folders "\$NEW_LORA" "${LORA_REF}" "${IDENT_REF}" "${SILU_REF}" \
    --dataset ${DATASET} \
    --output_json ${OUT_JSON}
python - <<'PY'
import json, numpy as np
e = json.load(open("${OUT_JSON}"))
def er(s):
    s = np.array(s, float); s = s[s > 0]; s = s / s.sum()
    return float(np.exp(-np.sum(s * np.log(s + 1e-12))))
print("=== LoRA alpha=512 (scaling=1) control: ER (same pipeline) ===")
for x in e:
    c = x["config"]
    t = c.get("type", c.get("model_type"))
    print("%-5s R=%-4s alpha=%-4s D=%-4s act=%-9s ER=%.2f" % (
        t, c.get("rank"), c.get("alpha"), c.get("dropout"),
        c.get("act_fn", "-"), er(x["spectrum"])))
PY
EOF
)"
    echo "[SUBMIT] chained ER analysis (depends on ${TRAIN_JID})"
fi

echo "Done. Monitor with: squeue -u \$USER"

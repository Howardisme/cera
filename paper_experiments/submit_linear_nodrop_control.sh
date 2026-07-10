#!/bin/bash
# Linear + No-Dropout control for the ER ablation.
#
# Purpose
# -------
# Table 3 reports "(b) Identity (Linear)" CeRA with ER ~354, far above LoRA
# (~60). Identity-activation CeRA with no dropout is the same FUNCTIONAL class
# as LoRA (a rank-r linear map B(A(x))), so if its ER stays high
# the gap cannot be attributed to the non-linearity. This control isolates that:
#
#   CeRA, act_fn=identity, dropout=0.0, rank=512, q_proj+v_proj, orca, lr=5e-4
#
# It is measured through the EXACT SAME pipeline (analyze_svd.py ->
# get_singular_values: eval mode, output last_delta, mean-centered, L1-norm SVD,
# exp-entropy ER) as the existing LoRA r512 and CeRA-silu spectra, so the three
# numbers are directly comparable:
#
#   LoRA      r512 D0.0 silu(linear) lr5e-4 : ER ~100  (existing)
#   CeRA silu r512 D0.0           lr5e-4 : ER ~388  (existing)
#   CeRA ident r512 D0.0          lr5e-4 : ER ????  (this control)
#
# Usage:
#   bash paper_experiments/submit_linear_nodrop_control.sh [--dry-run]

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
LR="5e-4"
DROPOUT="${CTRL_DROPOUT:-0.0}"   # override with CTRL_DROPOUT=0.3 to reproduce Table 3 Identity
ACT="identity"
TARGETS="q_proj,v_proj"
DATASET="orca"
EPOCHS=3
DTAG="D${DROPOUT}"

# Existing reference folders (measured by the same pipeline).
LORA_REF="results/Exp_LoRA_orca_R512_lr0.0005_silu_q_proj_v_proj_D0.0_E3_20260612_1043"
SILU_REF="results/Exp_CeRA_orca_R512_lr0.0005_silu_q_proj_v_proj_D0.0_E3_20260612_1037"
OUT_JSON="results/svd_control_linear_${DTAG}.json"

# --- 1. Train the Linear control ----------------------------------------------
TRAIN_NAME="ctrl_linear_${DTAG}_r512"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY] sbatch --job-name=${TRAIN_NAME} ... python train.py \\"
    echo "        --model_type CeRA --rank ${RANK} --lr ${LR} --dropout ${DROPOUT} \\"
    echo "        --act_fn ${ACT} --target_modules ${TARGETS} --dataset ${DATASET} \\"
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
    --model_type CeRA \
    --rank ${RANK} \
    --lr ${LR} \
    --dropout ${DROPOUT} \
    --act_fn ${ACT} \
    --target_modules ${TARGETS} \
    --dataset ${DATASET} \
    --epochs ${EPOCHS} \
    --base_model ${BASE_MODEL}
EOF
)")
    echo "[SUBMIT] train control -> job ${TRAIN_JID}"
fi

# --- 2. Chained ER measurement (same pipeline, all three configs) -------------
ANALYZE_NAME="ctrl_linear_${DTAG}_er"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY] sbatch --dependency=afterok:${TRAIN_JID} --job-name=${ANALYZE_NAME} ..."
    echo "      glob results/Exp_CeRA_orca_R512_lr0.0005_identity_q_proj_v_proj_${DTAG}_E3_*"
    echo "      python analysis/analyze_svd.py --folders <ident> ${LORA_REF} ${SILU_REF} \\"
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
IDENT_DIR=\$(ls -dt results/Exp_CeRA_orca_R512_lr0.0005_identity_q_proj_v_proj_${DTAG}_E3_* 2>/dev/null | head -1)
if [ -z "\$IDENT_DIR" ]; then
    echo "[ERROR] identity control folder not found"
    exit 1
fi
echo "[INFO] identity control folder: \$IDENT_DIR"
python analysis/analyze_svd.py \
    --folders "\$IDENT_DIR" "${LORA_REF}" "${SILU_REF}" \
    --dataset ${DATASET} \
    --output_json ${OUT_JSON}
python - <<'PY'
import json, numpy as np
e = json.load(open("${OUT_JSON}"))
def er(s):
    s = np.array(s, float); s = s / s.sum()
    return float(np.exp(-np.sum(s * np.log(s + 1e-10))))
print("=== Linear + No-Dropout control: ER (same pipeline) ===")
for x in e:
    c = x["config"]
    t = c.get("type", c.get("model_type"))
    print("%-6s R=%-4s D=%-4s act=%-8s ER=%.2f" % (
        t, c.get("rank"), c.get("dropout"), c.get("act_fn","-"), er(x["spectrum"])))
PY
EOF
)"
    echo "[SUBMIT] chained ER analysis (depends on ${TRAIN_JID})"
fi

echo "Done. Monitor with: squeue -u \$USER"

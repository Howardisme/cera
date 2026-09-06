#!/bin/bash
#SBATCH --job-name=cera_eval_peft
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/eval_peft_%j.log
#SBATCH --error=slurm_logs/eval_peft_err_%j.log
#SBATCH --partition=8gpus

# Companion to slurm/run_eval.sh -- evaluates a checkpoint produced by
# train_peft.py, which stores the adapter in PEFT native format
# (peft_adapter_best_<step>/adapter_model.safetensors) rather than the
# legacy .pt used by train.py.  Locates the checkpoint under an
# Exp_PEFT_* results folder and forwards --adapter_format peft to
# evaluate.py.  Runs the same four evaluations as run_eval.sh.
#
# Usage:
#   sbatch slurm/run_eval_peft.sh CELL_ID METHOD RANK LR DROPOUT \
#       [BASE_MODEL] [OUTPUT_DIR] [DATASET] [ALPHA] [TARGET_MODULES] [SEED]
#
# NOTE: METHOD may be LoRA or DoRA only.  CeRA runs from train_peft.py
# still save legacy .pt and must be evaluated with slurm/run_eval.sh.

CELL_ID=${1}
METHOD=${2}
RANK=${3}
LR=${4}
DROPOUT=${5}
BASE_MODEL=${6:-meta-llama/Llama-3.1-8B}
OUTPUT_DIR=${7:-results/eval_outputs}
DATASET=${8:-metamathqa}
ALPHA=${9:-64}
TARGET_MODULES=${10:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
SEED=${11:-42}

if [ "$TARGET_MODULES" = "all_linear" ]; then
    TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
fi

if [ -z "$DROPOUT" ]; then
    echo "[ERROR] Usage: $0 CELL_ID METHOD RANK LR DROPOUT [BASE_MODEL] [OUTPUT_DIR] [DATASET] [ALPHA] [TARGET_MODULES] [SEED]"
    exit 1
fi

if [ "$METHOD" = "CeRA" ]; then
    echo "[ERROR] run_eval_peft.sh does not support CeRA (train_peft.py stores CeRA "
    echo "        checkpoints in legacy .pt format).  Use slurm/run_eval.sh instead."
    exit 1
fi

module purge
module load singularity

SIF="${CERA_SIF:-/work/$USER/cera.sif}"
PYPKGS="${CERA_PYPKGS:-/work/$USER/cera_pypkgs}"
HF_CACHE="${CERA_HF_HOME:-/work/$USER/hf_cache}"
mkdir -p "$HF_CACHE"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — run via sbatch}"
mkdir -p slurm_logs

METHOD_LOWER=$(echo "$METHOD" | tr '[:upper:]' '[:lower:]')
MODEL_TAG=$(echo "$BASE_MODEL" | sed 's|.*/||')

# Convert LR to decimal directory format (matches train_peft.py's exp_name pattern).
case "$LR" in
  "1e-4") LR_DECIMAL="0.0001" ;;
  "2e-4") LR_DECIMAL="0.0002" ;;
  "3e-4") LR_DECIMAL="0.0003" ;;
  "5e-4") LR_DECIMAL="0.0005" ;;
  "1e-3") LR_DECIMAL="0.001"  ;;
  *)      LR_DECIMAL="$LR"    ;;
esac

# Locate the most recent matching Exp_PEFT_ results directory.
# train_peft.py appends _{MODEL_TAG} only for non-default models, _A{ALPHA}
# only for LoRA/DoRA when ALPHA != 32, and _S{seed} only for non-default seed.
DEFAULT_MODEL_TAG="Llama-3.1-8B"
CANDIDATES=$(ls -dt results/Exp_PEFT_${METHOD}_${DATASET}_R${RANK}_lr${LR_DECIMAL}_* 2>/dev/null \
    | grep "_D${DROPOUT}")
if [ "$MODEL_TAG" = "$DEFAULT_MODEL_TAG" ]; then
    CANDIDATES=$(echo "$CANDIDATES" | grep -v "_Llama-3\.")
else
    CANDIDATES=$(echo "$CANDIDATES" | grep "_${MODEL_TAG}")
fi
if [ "$ALPHA" = "32" ]; then
    CANDIDATES=$(echo "$CANDIDATES" | grep -v "_A[0-9]")
else
    CANDIDATES=$(echo "$CANDIDATES" | grep "_A${ALPHA}")
fi
if [ "$SEED" = "42" ]; then
    CANDIDATES=$(echo "$CANDIDATES" | grep -v "_S[0-9]")
else
    CANDIDATES=$(echo "$CANDIDATES" | grep "_S${SEED}")
fi
RESULTS_DIR=$(echo "$CANDIDATES" | head -1)
if [ -z "$RESULTS_DIR" ]; then
    echo "[ERROR] No Exp_PEFT_ results directory found for METHOD=${METHOD} DATASET=${DATASET} RANK=${RANK} LR=${LR_DECIMAL} D=${DROPOUT} A=${ALPHA} MODEL=${MODEL_TAG}"
    exit 1
fi

# PEFT adapter is stored as a directory, not a .pt file.
CHECKPOINT=$(ls -d "${RESULTS_DIR}/${METHOD}/peft_adapter_best_"* 2>/dev/null | head -1)
if [ -z "$CHECKPOINT" ]; then
    echo "[ERROR] No PEFT best-adapter directory found in ${RESULTS_DIR}/${METHOD}/"
    echo "        Expected: peft_adapter_best_<step>/"
    exit 1
fi

REP_PENALTY=${REP_PENALTY:-1.0}
if [ "$REP_PENALTY" != "1.0" ]; then
    CELL_ID="${CELL_ID}_rp${REP_PENALTY}"
    echo "[INFO] repetition_penalty=${REP_PENALTY} -> cell=${CELL_ID}"
fi

OUT_DIR="${OUTPUT_DIR}/${CELL_ID}"
export OUT_DIR_PY="$OUT_DIR"
mkdir -p "$OUT_DIR"

START_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "[START] eval-peft | cell=${CELL_ID} method=${METHOD} rank=${RANK} lr=${LR} A=${ALPHA}"
echo "  Results dir : ${RESULTS_DIR}"
echo "  PEFT adapter: ${CHECKPOINT}"
echo "  Output dir  : ${OUT_DIR}"

# ── 1. MATH-500 pass@1 ────────────────────────────────────────────────────────
if [ ! -f "${OUT_DIR}/math500_pass1.json" ]; then
    echo "[EVAL 1/4] MATH-500 pass@1 (greedy, 500 problems)..."
    singularity exec --nv -B /work \
        --env PYTHONPATH="$PYPKGS" \
        --env PYTHONNOUSERSITE=1 \
        --env HF_HOME="$HF_CACHE" \
        "$SIF" \
        python evaluate.py \
            --base_model              "$BASE_MODEL" \
            --adapter_type            "$METHOD_LOWER" \
            --adapter_format          peft \
            --rank                    "$RANK" \
            --alpha                   "$ALPHA" \
            --dropout                 "$DROPOUT" \
            --checkpoint              "$CHECKPOINT" \
            --dataset                 math500 \
            --num_samples_per_problem 1 \
            --temperature             0 \
            --batch_size              4 \
            --max_new_tokens          1024 \
            --target_modules          "$TARGET_MODULES" \
            --repetition_penalty      "$REP_PENALTY" \
            --output_jsonl            "${OUT_DIR}/math500_pass1.jsonl"

    python3 - <<'PYEOF'
import json, os
out_dir = os.environ["OUT_DIR_PY"]
with open(f"{out_dir}/math500_pass1.jsonl") as fh:
    records = [json.loads(l) for l in fh]
n_correct = sum(1 for r in records if r.get("n_correct", 0) > 0)
n_total = len(records)
pct = round(100.0 * n_correct / n_total, 2) if n_total > 0 else 0.0
with open(f"{out_dir}/math500_pass1.json", "w") as fh:
    json.dump({"n_correct": n_correct, "n_total": n_total, "pass1": pct}, fh)
print(f"[MATH-500 pass@1] {n_correct}/{n_total} = {pct}%")
PYEOF
else
    echo "[SKIP] math500_pass1.json already exists"
fi

# ── 2. MATH-500 pass@10 ───────────────────────────────────────────────────────
if [ ! -f "${OUT_DIR}/math500_pass10.json" ]; then
    echo "[EVAL 2/4] MATH-500 pass@10 (sampling n=10, 500 problems)..."
    singularity exec --nv -B /work \
        --env PYTHONPATH="$PYPKGS" \
        --env PYTHONNOUSERSITE=1 \
        --env HF_HOME="$HF_CACHE" \
        "$SIF" \
        python evaluate.py \
            --base_model              "$BASE_MODEL" \
            --adapter_type            "$METHOD_LOWER" \
            --adapter_format          peft \
            --rank                    "$RANK" \
            --alpha                   "$ALPHA" \
            --dropout                 "$DROPOUT" \
            --checkpoint              "$CHECKPOINT" \
            --dataset                 math500 \
            --num_samples_per_problem 10 \
            --temperature             0.8 \
            --top_p                   0.95 \
            --batch_size              4 \
            --max_new_tokens          1024 \
            --target_modules          "$TARGET_MODULES" \
            --repetition_penalty      "$REP_PENALTY" \
            --output_jsonl            "${OUT_DIR}/math500_pass10.jsonl"

    python3 - <<'PYEOF'
import json, os
from math import comb
out_dir = os.environ["OUT_DIR_PY"]
with open(f"{out_dir}/math500_pass10.jsonl") as fh:
    records = [json.loads(l) for l in fh]
k = 10
pass10_list, any_correct_list = [], []
for r in records:
    n = r.get("num_samples_per_problem", 10)
    c = int(r.get("n_correct", 0))
    if n >= k:
        denom = comb(n, k)
        numer = comb(n - c, k) if n - c >= k else 0
        p = 1.0 - numer / denom if denom > 0 else float(c > 0)
    else:
        p = float(c > 0)
    pass10_list.append(p)
    any_correct_list.append(float(c > 0))
n_total = len(records)
pass10 = round(100.0 * sum(pass10_list) / n_total, 2) if n_total > 0 else 0.0
any_correct = round(100.0 * sum(any_correct_list) / n_total, 2) if n_total > 0 else 0.0
with open(f"{out_dir}/math500_pass10.json", "w") as fh:
    json.dump({"pass10_unbiased": pass10, "any_correct_rate": any_correct, "n_total": n_total}, fh)
print(f"[MATH-500 pass@10] unbiased={pass10}%  any_correct={any_correct}%")
PYEOF
else
    echo "[SKIP] math500_pass10.json already exists"
fi

# ── 3. GSM8K pass@1 ───────────────────────────────────────────────────────────
if [ ! -f "${OUT_DIR}/gsm8k_pass1.json" ]; then
    echo "[EVAL 3/4] GSM8K pass@1 (greedy, 1319 problems)..."
    singularity exec --nv -B /work \
        --env PYTHONPATH="$PYPKGS" \
        --env PYTHONNOUSERSITE=1 \
        --env HF_HOME="$HF_CACHE" \
        "$SIF" \
        python evaluate.py \
            --base_model              "$BASE_MODEL" \
            --adapter_type            "$METHOD_LOWER" \
            --adapter_format          peft \
            --rank                    "$RANK" \
            --alpha                   "$ALPHA" \
            --dropout                 "$DROPOUT" \
            --checkpoint              "$CHECKPOINT" \
            --dataset                 gsm8k \
            --num_samples_per_problem 1 \
            --temperature             0 \
            --batch_size              4 \
            --max_new_tokens          512 \
            --target_modules          "$TARGET_MODULES" \
            --repetition_penalty      "$REP_PENALTY" \
            --output_jsonl            "${OUT_DIR}/gsm8k_pass1.jsonl"

    python3 - <<'PYEOF'
import json, os
out_dir = os.environ["OUT_DIR_PY"]
with open(f"{out_dir}/gsm8k_pass1.jsonl") as fh:
    records = [json.loads(l) for l in fh]
n_correct = sum(1 for r in records if r.get("n_correct", 0) > 0)
n_total = len(records)
pct = round(100.0 * n_correct / n_total, 2) if n_total > 0 else 0.0
with open(f"{out_dir}/gsm8k_pass1.json", "w") as fh:
    json.dump({"n_correct": n_correct, "n_total": n_total, "pass1": pct}, fh)
print(f"[GSM8K pass@1] {n_correct}/{n_total} = {pct}%")
PYEOF
else
    echo "[SKIP] gsm8k_pass1.json already exists"
fi

# ── 4. MATH Level 5 pass@1 ────────────────────────────────────────────────────
if [ ! -f "${OUT_DIR}/math_hard_pass1.json" ]; then
    echo "[EVAL 4/4] MATH Level 5 pass@1 (greedy, ~1324 problems)..."
    singularity exec --nv -B /work \
        --env PYTHONPATH="$PYPKGS" \
        --env PYTHONNOUSERSITE=1 \
        --env HF_HOME="$HF_CACHE" \
        "$SIF" \
        python evaluate.py \
            --base_model              "$BASE_MODEL" \
            --adapter_type            "$METHOD_LOWER" \
            --adapter_format          peft \
            --rank                    "$RANK" \
            --alpha                   "$ALPHA" \
            --dropout                 "$DROPOUT" \
            --checkpoint              "$CHECKPOINT" \
            --dataset                 math_hard \
            --num_samples_per_problem 1 \
            --temperature             0 \
            --batch_size              4 \
            --max_new_tokens          1024 \
            --target_modules          "$TARGET_MODULES" \
            --repetition_penalty      "$REP_PENALTY" \
            --output_jsonl            "${OUT_DIR}/math_hard_pass1.jsonl"

    python3 - <<'PYEOF'
import json, os
out_dir = os.environ["OUT_DIR_PY"]
with open(f"{out_dir}/math_hard_pass1.jsonl") as fh:
    records = [json.loads(l) for l in fh]
n_correct = sum(1 for r in records if r.get("n_correct", 0) > 0)
n_total = len(records)
pct = round(100.0 * n_correct / n_total, 2) if n_total > 0 else 0.0
with open(f"{out_dir}/math_hard_pass1.json", "w") as fh:
    json.dump({"n_correct": n_correct, "n_total": n_total, "pass1": pct}, fh)
print(f"[MATH-Hard (Level 5) pass@1] {n_correct}/{n_total} = {pct}%")
PYEOF
else
    echo "[SKIP] math_hard_pass1.json already exists"
fi

# ── 5. cell_metadata.json ─────────────────────────────────────────────────────
END_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
LOG_FILE="${RESULTS_DIR}/${METHOD}/${METHOD}_log.json"

python3 - <<PYEOF
import json, os

log_path = "${LOG_FILE}"
out_dir  = "${OUT_DIR}"
best_ppl, best_step = None, None
if os.path.exists(log_path):
    try:
        with open(log_path) as fh:
            log = json.load(fh)
        for entry in log.get("history", []):
            ppl = entry.get("new_task_target", {}).get("test_ppl")
            step = entry.get("step")
            if ppl is not None and (best_ppl is None or ppl < best_ppl):
                best_ppl = ppl
                best_step = step
    except (OSError, json.JSONDecodeError) as e:
        print(f"[WARN] Could not parse log: {e}")

meta = {
    "cell_id":          "${CELL_ID}",
    "pipeline":         "peft",
    "rank":             int("${RANK}"),
    "method":           "${METHOD}",
    "lr":               "${LR}",
    "dropout":          float("${DROPOUT}"),
    "alpha":            int("${ALPHA}"),
    "base_model":       "${BASE_MODEL}",
    "best_val_ppl":     round(best_ppl, 4) if best_ppl is not None else None,
    "best_val_step":    best_step,
    "checkpoint_path":  "${CHECKPOINT}",
    "results_dir":      "${RESULTS_DIR}",
    "git_commit":       "${GIT_COMMIT}",
    "start_time":       "${START_TIME}",
    "end_time":         "${END_TIME}",
}
with open(f"{out_dir}/cell_metadata.json", "w") as fh:
    json.dump(meta, fh, indent=2)
print(f"[META] Written: best_val_ppl={best_ppl}, best_val_step={best_step}")
PYEOF

echo "[END] Cell ${CELL_ID} complete at $(date)"

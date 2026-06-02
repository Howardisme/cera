#!/bin/bash
#SBATCH --job-name=cera_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/eval_%j.log
#SBATCH --error=slurm_logs/eval_err_%j.log
#SBATCH --partition=normal
# NOTE: Add your cluster account line here, e.g.:
#   #SBATCH -A YOUR_ACCOUNT

# Run all 3 evaluations (MATH pass@1, MATH pass@10, GSM8K pass@1) for one cell.
#
# Usage:
#   sbatch slurm/run_eval.sh CELL_ID METHOD RANK LR DROPOUT [BASE_MODEL] [OUTPUT_DIR]
#
# Arguments:
#   CELL_ID     unique identifier for this eval cell (e.g. r128_cera_lr5e-4)
#   METHOD      CeRA | LoRA | DoRA
#   RANK        adapter rank (e.g. 64, 128, 512)
#   LR          learning rate string (e.g. 5e-4)
#   DROPOUT     dropout rate (e.g. 0.1 for CeRA, 0.0 for LoRA/DoRA)
#   BASE_MODEL  HuggingFace model ID (default: meta-llama/Llama-3.1-8B)
#   OUTPUT_DIR  base output directory (default: results/eval_outputs)
#
# Checkpoint is located automatically from:
#   results/Exp_<METHOD>_math_R<RANK>_lr<LR_DEC>_*/<METHOD>/*_ckpt_best_*.pt
#
# Outputs written to: OUTPUT_DIR/CELL_ID/
#   math_pass1.json, math_pass1.jsonl
#   math_pass10.json, math_pass10.jsonl
#   gsm8k_pass1.json, gsm8k_pass1.jsonl
#   cell_metadata.json

CELL_ID=${1}
METHOD=${2}
RANK=${3}
LR=${4}
DROPOUT=${5}
BASE_MODEL=${6:-meta-llama/Llama-3.1-8B}
OUTPUT_DIR=${7:-results/eval_outputs}

if [ -z "$DROPOUT" ]; then
    echo "[ERROR] Usage: $0 CELL_ID METHOD RANK LR DROPOUT [BASE_MODEL] [OUTPUT_DIR]"
    exit 1
fi

# Activate your environment here, e.g.:
#   conda activate cera_env
# or:
#   source /path/to/venv/bin/activate

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — run via sbatch}"
mkdir -p slurm_logs

METHOD_LOWER=$(echo "$METHOD" | tr '[:upper:]' '[:lower:]')

# Convert LR to decimal directory format
case "$LR" in
  "1e-4") LR_DECIMAL="0.0001" ;;
  "3e-4") LR_DECIMAL="0.0003" ;;
  "5e-4") LR_DECIMAL="0.0005" ;;
  "1e-3") LR_DECIMAL="0.001"  ;;
  *)      LR_DECIMAL="$LR"    ;;
esac

# Find the most recent matching results/ directory
MODEL_TAG=$(echo "$BASE_MODEL" | sed 's|.*/||')
RESULTS_DIR=$(ls -dt results/Exp_${METHOD}_math_R${RANK}_lr${LR_DECIMAL}_* 2>/dev/null \
    | grep "_D${DROPOUT}" | head -1)
if [ -z "$RESULTS_DIR" ]; then
    echo "[ERROR] No results/ directory found for METHOD=${METHOD} RANK=${RANK} LR=${LR_DECIMAL} D=${DROPOUT}"
    exit 1
fi

CHECKPOINT=$(ls "${RESULTS_DIR}/${METHOD}/${METHOD_LOWER}_ckpt_best_"*.pt 2>/dev/null | head -1)
if [ -z "$CHECKPOINT" ]; then
    echo "[ERROR] No best checkpoint found in ${RESULTS_DIR}/${METHOD}/"
    exit 1
fi

OUT_DIR="${OUTPUT_DIR}/${CELL_ID}"
export OUT_DIR_PY="$OUT_DIR"
mkdir -p "$OUT_DIR"

START_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "[START] eval | cell=${CELL_ID} method=${METHOD} rank=${RANK} lr=${LR}"
echo "  Results dir : ${RESULTS_DIR}"
echo "  Checkpoint  : ${CHECKPOINT}"
echo "  Output dir  : ${OUT_DIR}"

# ── 1. MATH pass@1 (greedy, 500 problems) ─────────────────────────────────────
if [ ! -f "${OUT_DIR}/math_pass1.json" ]; then
    echo "[EVAL 1/3] MATH pass@1 (greedy, 500 problems)..."
    python evaluate.py \
        --base_model              "$BASE_MODEL" \
        --adapter_type            "$METHOD_LOWER" \
        --rank                    "$RANK" \
        --dropout                 "$DROPOUT" \
        --checkpoint              "$CHECKPOINT" \
        --dataset                 math \
        --num_samples_per_problem 1 \
        --temperature             0 \
        --batch_size              4 \
        --max_new_tokens          1024 \
        --num_samples             500 \
        --output_jsonl            "${OUT_DIR}/math_pass1.jsonl"

    python3 - <<'PYEOF'
import json, os
out_dir = os.environ["OUT_DIR_PY"]
with open(f"{out_dir}/math_pass1.jsonl") as fh:
    records = [json.loads(l) for l in fh]
n_correct = sum(1 for r in records if r.get("n_correct", 0) > 0)
n_total = len(records)
pct = round(100.0 * n_correct / n_total, 2) if n_total > 0 else 0.0
with open(f"{out_dir}/math_pass1.json", "w") as fh:
    json.dump({"n_correct": n_correct, "n_total": n_total, "pass1": pct}, fh)
print(f"[MATH pass@1] {n_correct}/{n_total} = {pct}%")
PYEOF
else
    echo "[SKIP] math_pass1.json already exists"
fi

# ── 2. MATH pass@10 (sampling, n=10, 500 problems) ────────────────────────────
if [ ! -f "${OUT_DIR}/math_pass10.json" ]; then
    echo "[EVAL 2/3] MATH pass@10 (sampling n=10, 500 problems)..."
    python evaluate.py \
        --base_model              "$BASE_MODEL" \
        --adapter_type            "$METHOD_LOWER" \
        --rank                    "$RANK" \
        --dropout                 "$DROPOUT" \
        --checkpoint              "$CHECKPOINT" \
        --dataset                 math \
        --num_samples_per_problem 10 \
        --temperature             0.8 \
        --top_p                   0.95 \
        --batch_size              4 \
        --max_new_tokens          1024 \
        --num_samples             500 \
        --output_jsonl            "${OUT_DIR}/math_pass10.jsonl"

    python3 - <<'PYEOF'
import json, os
from math import comb
out_dir = os.environ["OUT_DIR_PY"]
with open(f"{out_dir}/math_pass10.jsonl") as fh:
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
with open(f"{out_dir}/math_pass10.json", "w") as fh:
    json.dump({"pass10_unbiased": pass10, "any_correct_rate": any_correct, "n_total": n_total}, fh)
print(f"[MATH pass@10] unbiased={pass10}%  any_correct={any_correct}%")
PYEOF
else
    echo "[SKIP] math_pass10.json already exists"
fi

# ── 3. GSM8K pass@1 (greedy, full 1319 problems) ──────────────────────────────
if [ ! -f "${OUT_DIR}/gsm8k_pass1.json" ]; then
    echo "[EVAL 3/3] GSM8K pass@1 (greedy, 1319 problems)..."
    python evaluate.py \
        --base_model              "$BASE_MODEL" \
        --adapter_type            "$METHOD_LOWER" \
        --rank                    "$RANK" \
        --dropout                 "$DROPOUT" \
        --checkpoint              "$CHECKPOINT" \
        --dataset                 gsm8k \
        --num_samples_per_problem 1 \
        --temperature             0 \
        --batch_size              4 \
        --max_new_tokens          512 \
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

# ── 4. cell_metadata.json ─────────────────────────────────────────────────────
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
    "rank":             int("${RANK}"),
    "method":           "${METHOD}",
    "lr":               "${LR}",
    "dropout":          float("${DROPOUT}"),
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

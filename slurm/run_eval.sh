#!/bin/bash
#SBATCH --job-name=cera_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/eval_%j.log
#SBATCH --error=slurm_logs/eval_err_%j.log
#SBATCH --partition=8gpus
# NOTE: Add your cluster account line here, e.g.:
#   #SBATCH -A YOUR_ACCOUNT

# Run all 3 evaluations (MATH pass@1, MATH pass@10, GSM8K pass@1) for one cell.
#
# Usage:
#   sbatch slurm/run_eval.sh CELL_ID METHOD RANK LR DROPOUT [BASE_MODEL] [OUTPUT_DIR] [DATASET] [ALPHA]
#
# Arguments:
#   CELL_ID     unique identifier for this eval cell (e.g. r128_cera_lr5e-4)
#   METHOD      CeRA | LoRA | DoRA
#   RANK        adapter rank (e.g. 64, 128, 512)
#   LR          learning rate string (e.g. 5e-4), or "best" to use the sweep-selected optimum
#   DROPOUT     dropout rate (e.g. 0.1 for CeRA, 0.0 for LoRA/DoRA)
#   BASE_MODEL  HuggingFace model ID (default: meta-llama/Llama-3.1-8B)
#   OUTPUT_DIR  base output directory (default: results/eval_outputs)
#   DATASET     training dataset tag for the checkpoint search (default: math)
#   ALPHA       LoRA/DoRA alpha; effective scale = alpha/rank (default: 32).
#               Ignored by CeRA. When ALPHA == 32 the folder search excludes any
#               _A{n} suffix (historical default); when ALPHA != 32 the search
#               requires _A{ALPHA}. Must match the training-time alpha.
#
# Checkpoint is located automatically from:
#   results/Exp_<METHOD>_<DATASET>_R<RANK>_lr<LR_DEC>_*/<METHOD>/*_ckpt_best_*.pt
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
DATASET=${8:-math}   # training dataset tag used to locate the checkpoint dir
ALPHA=${9:-32}

if [ -z "$DROPOUT" ]; then
    echo "[ERROR] Usage: $0 CELL_ID METHOD RANK LR DROPOUT [BASE_MODEL] [OUTPUT_DIR] [DATASET] [ALPHA]"
    exit 1
fi

# Activate your environment here, e.g.:
#   conda activate cera_env
# or:
#   source /path/to/venv/bin/activate

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — run via sbatch}"
mkdir -p slurm_logs

METHOD_LOWER=$(echo "$METHOD" | tr '[:upper:]' '[:lower:]')
MODEL_TAG=$(echo "$BASE_MODEL" | sed 's|.*/||')

# Best LR per model/method/rank from grid search over {1e-4, 3e-4, 5e-4, 1e-3}.
# 8B: CeRA R64→3e-4, CeRA R128→1e-3, LoRA/DoRA (any rank)→3e-4
# 1B: CeRA R64→3e-4, CeRA R128→3e-4, LoRA R64→3e-4, LoRA R128→5e-4, DoRA R64→1e-3, DoRA R128→1e-3
# 3B: CeRA R64→3e-4, CeRA R128→5e-4, LoRA R64→3e-4, LoRA R128→5e-4, DoRA R64→1e-3, DoRA R128→5e-4
# R512 (1B/3B math): CeRA 1B→3e-4 3B→5e-4, LoRA 1B/3B→5e-4, DoRA 1B/3B→5e-4 (8B R512 math not swept)
if [ "$LR" = "best" ]; then
    case "${MODEL_TAG}_${METHOD}_${RANK}" in
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
            echo "[ERROR] No best LR defined for MODEL=${MODEL_TAG} METHOD=${METHOD} RANK=${RANK}. Pass LR explicitly."
            exit 1 ;;
    esac
    echo "[INFO] Resolved LR=best → ${LR} for ${MODEL_TAG} ${METHOD} R=${RANK}"
fi

# Convert LR to decimal directory format
case "$LR" in
  "1e-4") LR_DECIMAL="0.0001" ;;
  "3e-4") LR_DECIMAL="0.0003" ;;
  "5e-4") LR_DECIMAL="0.0005" ;;
  "1e-3") LR_DECIMAL="0.001"  ;;
  *)      LR_DECIMAL="$LR"    ;;
esac

# Find the most recent matching results/ directory.
# train.py appends _{MODEL_TAG} only for non-default models, so we must
# filter by model size to avoid picking up a checkpoint from a different model.
# train.py appends _A{ALPHA} to LoRA/DoRA runs when ALPHA != 32; add a matching
# filter so scale-matched folders do not collide with the default alpha=32 runs.
DEFAULT_MODEL_TAG="Llama-3.1-8B"
CANDIDATES=$(ls -dt results/Exp_${METHOD}_${DATASET}_R${RANK}_lr${LR_DECIMAL}_* 2>/dev/null \
    | grep "_D${DROPOUT}")
if [ "$MODEL_TAG" = "$DEFAULT_MODEL_TAG" ]; then
    # Default model dirs have no model-size suffix → exclude any non-default model dir
    CANDIDATES=$(echo "$CANDIDATES" | grep -v "_Llama-3\.")
else
    # Non-default model dirs contain _<MODEL_TAG> → require it
    CANDIDATES=$(echo "$CANDIDATES" | grep "_${MODEL_TAG}")
fi
# Alpha filter (only meaningful for LoRA/DoRA; CeRA folders never carry _A).
if [ "$METHOD" != "CeRA" ]; then
    if [ "$ALPHA" = "32" ]; then
        # Default alpha → exclude any explicit _A{n} suffix
        CANDIDATES=$(echo "$CANDIDATES" | grep -v "_A[0-9]")
    else
        # Non-default alpha → require the matching _A{ALPHA} tag
        CANDIDATES=$(echo "$CANDIDATES" | grep "_A${ALPHA}")
    fi
fi
RESULTS_DIR=$(echo "$CANDIDATES" | head -1)
if [ -z "$RESULTS_DIR" ]; then
    echo "[ERROR] No results/ directory found for METHOD=${METHOD} DATASET=${DATASET} RANK=${RANK} LR=${LR_DECIMAL} D=${DROPOUT} A=${ALPHA} MODEL=${MODEL_TAG}"
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
echo "[START] eval | cell=${CELL_ID} method=${METHOD} rank=${RANK} lr=${LR} A=${ALPHA}"
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
        --alpha                   "$ALPHA" \
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
        --alpha                   "$ALPHA" \
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
        --alpha                   "$ALPHA" \
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
    "alpha":            int("${ALPHA}") if "${METHOD}" != "CeRA" else None,
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

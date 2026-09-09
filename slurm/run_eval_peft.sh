#!/bin/bash
#SBATCH --job-name=cera_eval_peft
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/eval_peft_%j.log
#SBATCH --error=slurm_logs/eval_peft_err_%j.log
#SBATCH --partition=normal

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
# CeRA uses metadata-backed .pt checkpoints; LoRA/DoRA use PEFT directories.
# Optional argument 12 (or CHECKPOINT_PATH) selects an exact checkpoint.

set -e

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
CHECKPOINT=${12:-${CHECKPOINT_PATH:-}}
ACT_FN=${ACT_FN:-silu}
ADAPTER_FORMAT=peft

if [ "$TARGET_MODULES" = "all_linear" ]; then
    TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
fi

if [ -z "$DROPOUT" ]; then
    echo "[ERROR] Usage: $0 CELL_ID METHOD RANK LR DROPOUT [BASE_MODEL] [OUTPUT_DIR] [DATASET] [ALPHA] [TARGET_MODULES] [SEED]"
    exit 1
fi

if [ "$METHOD" = "CeRA" ]; then
    ADAPTER_FORMAT=legacy
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

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(python3 - "$METHOD" "$DATASET" "$RANK" "$LR" "$DROPOUT" "$BASE_MODEL" "$ALPHA" "$TARGET_MODULES" "$SEED" "$ACT_FN" <<'PYSELECT'
import json
import sys
from pathlib import Path

method, dataset, rank, lr, dropout, base, alpha, targets, seed, activation = sys.argv[1:]
expected = dict(model_type=method, dataset=dataset, rank=int(rank), lr=float(lr),
                dropout=float(dropout), model=base, seed=int(seed))
matches = []
for log_path in Path("results").glob(f"Exp_PEFT_{method}_*/{method}/{method}_log.json"):
    config = json.loads(log_path.read_text())["config"]
    if any(config.get(key) != value for key, value in expected.items()):
        continue
    if set(config.get("target_modules", "").split(",")) != set(targets.split(",")):
        continue
    if method == "CeRA":
        if config.get("cera_format_version") != 1 or config.get("act_fn") != activation:
            continue
        pattern = "cera_ckpt_best_*.pt"
    else:
        if config.get("alpha") != int(alpha):
            continue
        pattern = "peft_adapter_best_*"
    matches.extend(log_path.parent.glob(pattern))
if len(matches) != 1:
    raise SystemExit(f"Expected one matching checkpoint, found {len(matches)}. Set CHECKPOINT_PATH explicitly. Matches: {matches}")
print(matches[0])
PYSELECT
    )
fi
if [ ! -e "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint does not exist: $CHECKPOINT"
    exit 1
fi
RESULTS_DIR=$(dirname "$(dirname "$CHECKPOINT")")

REP_PENALTY=${REP_PENALTY:-1.0}
if [ "$REP_PENALTY" != "1.0" ]; then
    CELL_ID="${CELL_ID}_rp${REP_PENALTY}"
    echo "[INFO] repetition_penalty=${REP_PENALTY} -> cell=${CELL_ID}"
fi

OUT_DIR="${OUTPUT_DIR}/${CELL_ID}"
export OUT_DIR_PY="$OUT_DIR"
mkdir -p "$OUT_DIR"

python3 - "$OUT_DIR" "$CHECKPOINT" "$BASE_MODEL" "$RANK" "$DROPOUT" "$TARGET_MODULES" "$ACT_FN" "$REP_PENALTY" "$ADAPTER_FORMAT" <<'PYREQUEST'
import json
import sys
from pathlib import Path

out_dir, checkpoint, base, rank, dropout, targets, activation, penalty, adapter_format = sys.argv[1:]
root = Path(out_dir)
record = dict(checkpoint=str(Path(checkpoint).resolve()), base_model=base,
              rank=int(rank), dropout=float(dropout), targets=sorted(targets.split(",")),
              activation=activation, repetition_penalty=float(penalty), adapter_format=adapter_format,
              max_new_tokens=1024, num_samples_per_problem=1, batch_size=4)
manifest = root / "evaluation_request.json"
if manifest.exists():
    if json.loads(manifest.read_text()) != record:
        raise SystemExit("Output directory belongs to another evaluation. Use a new CELL_ID.")
elif any(root.iterdir()):
    raise SystemExit("Output directory has untracked results. Use a new CELL_ID; old results are preserved.")
manifest.write_text(json.dumps(record, indent=2))
PYREQUEST

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
            --adapter_format          "$ADAPTER_FORMAT" \
            --act_fn                  "$ACT_FN" \
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
# Disabled during training-pipeline tuning to save time/GPU.
# Re-enable by uncommenting the block below.
# if [ ! -f "${OUT_DIR}/math500_pass10.json" ]; then
#     echo "[EVAL 2/4] MATH-500 pass@10 (sampling n=10, 500 problems)..."
#     singularity exec --nv -B /work \
#         --env PYTHONPATH="$PYPKGS" \
#         --env PYTHONNOUSERSITE=1 \
#         --env HF_HOME="$HF_CACHE" \
#         "$SIF" \
#         python evaluate.py \
#             --base_model              "$BASE_MODEL" \
#             --adapter_type            "$METHOD_LOWER" \
#             --adapter_format          "$ADAPTER_FORMAT" \
#             --act_fn                  "$ACT_FN" \
#             --rank                    "$RANK" \
#             --alpha                   "$ALPHA" \
#             --dropout                 "$DROPOUT" \
#             --checkpoint              "$CHECKPOINT" \
#             --dataset                 math500 \
#             --num_samples_per_problem 10 \
#             --temperature             0.8 \
#             --top_p                   0.95 \
#             --batch_size              4 \
#             --max_new_tokens          1024 \
#             --target_modules          "$TARGET_MODULES" \
#             --repetition_penalty      "$REP_PENALTY" \
#             --output_jsonl            "${OUT_DIR}/math500_pass10.jsonl"
#
#     python3 - <<'PYEOF'
# import json, os
# from math import comb
# out_dir = os.environ["OUT_DIR_PY"]
# with open(f"{out_dir}/math500_pass10.jsonl") as fh:
#     records = [json.loads(l) for l in fh]
# k = 10
# pass10_list, any_correct_list = [], []
# for r in records:
#     n = r.get("num_samples_per_problem", 10)
#     c = int(r.get("n_correct", 0))
#     if n >= k:
#         denom = comb(n, k)
#         numer = comb(n - c, k) if n - c >= k else 0
#         p = 1.0 - numer / denom if denom > 0 else float(c > 0)
#     else:
#         p = float(c > 0)
#     pass10_list.append(p)
#     any_correct_list.append(float(c > 0))
# n_total = len(records)
# pass10 = round(100.0 * sum(pass10_list) / n_total, 2) if n_total > 0 else 0.0
# any_correct = round(100.0 * sum(any_correct_list) / n_total, 2) if n_total > 0 else 0.0
# with open(f"{out_dir}/math500_pass10.json", "w") as fh:
#     json.dump({"pass10_unbiased": pass10, "any_correct_rate": any_correct, "n_total": n_total}, fh)
# print(f"[MATH-500 pass@10] unbiased={pass10}%  any_correct={any_correct}%")
# PYEOF
# else
#     echo "[SKIP] math500_pass10.json already exists"
# fi
echo "[SKIP 2/4] MATH-500 pass@10 disabled (see comment in slurm/run_eval_peft.sh)"

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
            --adapter_format          "$ADAPTER_FORMAT" \
            --act_fn                  "$ACT_FN" \
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
            --adapter_format          "$ADAPTER_FORMAT" \
            --act_fn                  "$ACT_FN" \
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
    "adapter_format":   "${ADAPTER_FORMAT}",
    "target_modules":   "${TARGET_MODULES}",
    "activation":       "${ACT_FN}",
    "rank":             int("${RANK}"),
    "method":           "${METHOD}",
    "lr":               "${LR}",
    "dropout":          float("${DROPOUT}"),
    "alpha":            None if "${METHOD}" == "CeRA" else int("${ALPHA}"),
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

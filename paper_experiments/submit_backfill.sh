#!/bin/bash
# submit_backfill.sh -- Backfill all missing DoRA + R512 LR-sweep cells (1B + 3B).
#
# Each cell = ONE Slurm job (train+eval combined via slurm/run_cell.sh), so the
# job footprint is 1 per cell, not 2. The cluster caps the number of jobs in the
# queue per user (QOSMaxSubmitJobPerUser = 10), so we throttle by POLLING squeue
# and only submitting the next cell when the queue has room (<= MAXQ jobs). This
# respects both the submit cap and a "<= 8 concurrent" preference.
#
# Covers 32 cells (= 32 jobs):
#   A) DoRA missing at R64/R128 (8 cells)
#        1B: R128 lr{3e-4,5e-4,1e-3}
#        3B: R64 lr1e-3 ; R128 lr{1e-4,3e-4,5e-4,1e-3}
#   B) R512 full sweep (24 cells)
#        {1B,3B} x {CeRA,LoRA,DoRA} x lr{1e-4,3e-4,5e-4,1e-3}
#   dataset=math, E=3. dropout: CeRA=0.1, LoRA/DoRA=0.0.
#
# Usage:
#   DRY=1 bash paper_experiments/submit_backfill.sh        # preview only (no waiting)
#   nohup bash paper_experiments/submit_backfill.sh > backfill.log 2>&1 &   # run detached
#   MAXQ=8 POLL=120 bash paper_experiments/submit_backfill.sh               # tune throttle
#   ONLY=r512 bash paper_experiments/submit_backfill.sh    # only R512 cells
#   SKIP="a_cell another_cell" bash ...                    # exclude cells already in queue
#
# Auto-skips cells that already have results (results/eval_outputs/<id>/math_pass1.json).
# Run from the repo root (login node). The loop mostly sleeps; it is not compute.

set -euo pipefail

DATASET=math
EPOCHS=3
ONLY=${ONLY:-all}     # all | dora | r512
SKIP=${SKIP:-}        # space-separated CELL_IDs to exclude (e.g. already in queue)
OUTROOT=results/eval_outputs

dropout_for() { [ "$1" = "CeRA" ] && echo "0.1" || echo "0.0"; }

# ── Build the cell list: "BASE_MODEL SIZE METHOD RANK LR DROPOUT" ──────────────
CELLS=()
add_cell() {  # BASE_MODEL SIZE METHOD RANK LR
    local d; d=$(dropout_for "$3")
    CELLS+=("$1 $2 $3 $4 $5 $d")
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "dora" ]; then
    for lr in 3e-4 5e-4 1e-3; do add_cell meta-llama/Llama-3.2-1B 1b DoRA 128 "$lr"; done
    add_cell meta-llama/Llama-3.2-3B 3b DoRA 64 1e-3
    for lr in 1e-4 3e-4 5e-4 1e-3; do add_cell meta-llama/Llama-3.2-3B 3b DoRA 128 "$lr"; done
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "r512" ]; then
    for ms in "meta-llama/Llama-3.2-1B 1b" "meta-llama/Llama-3.2-3B 3b"; do
        read -r bm size <<< "$ms"
        for method in CeRA LoRA DoRA; do
            for lr in 1e-4 3e-4 5e-4 1e-3; do
                add_cell "$bm" "$size" "$method" 512 "$lr"
            done
        done
    done
fi

# ── Filter: drop cells already done (math_pass1.json exists) or in SKIP list ───
cell_id_of() {  # SIZE METHOD RANK LR -> CELL_ID
    local lc; lc=$(echo "$2" | tr '[:upper:]' '[:lower:]')
    echo "${lc}_r${3}_lr${4}_${1}"
}
in_skip() { for s in $SKIP; do [ "$s" = "$1" ] && return 0; done; return 1; }

FCELLS=()
for cell in "${CELLS[@]}"; do
    read -r BASE_MODEL SIZE METHOD RANK LR DROPOUT <<< "$cell"
    cid=$(cell_id_of "$SIZE" "$METHOD" "$RANK" "$LR")
    if [ -f "${OUTROOT}/${cid}/math_pass1.json" ]; then
        echo "[SKIP] ${cid} (already has eval results)"; continue
    fi
    if in_skip "$cid"; then
        echo "[SKIP] ${cid} (in SKIP list)"; continue
    fi
    FCELLS+=("$cell")
done
CELLS=("${FCELLS[@]}")

TOTAL=${#CELLS[@]}
echo "[INFO] ONLY=${ONLY}  cells=${TOTAL}  jobs=${TOTAL}  throttle MAXQ=${MAXQ:-8} POLL=${POLL:-120}s"
[ "$TOTAL" -eq 0 ] && { echo "[INFO] nothing to submit."; exit 0; }

# ── Submit helper ─────────────────────────────────────────────────────────────
DRY_N=0
submit() {
    if [ "${DRY:-0}" = "1" ]; then
        DRY_N=$((DRY_N + 1))
        echo "[DRY] sbatch $*" >&2
        echo "$((1000 + DRY_N))"
    else
        sbatch "$@" | awk '{print $NF}'
    fi
}

# ── Polling submitter ─────────────────────────────────────────────────────────
# The cluster caps the number of jobs IN THE QUEUE (pending+running) per user
# (QOSMaxSubmitJobPerUser). So we cannot pre-submit everything with dependency
# chains -- pending jobs also consume the quota. Instead we poll squeue and only
# submit the next cell when the queue has room, keeping the total user job count
# at or below MAXQ. With MAXQ <= the run limit, this also bounds concurrency.
#
#   MAXQ  max user jobs allowed in squeue at once (default 8; cluster cap is 10)
#   POLL  seconds between squeue checks while waiting for a slot (default 120)
#
# This loop runs until all cells are submitted -- it can take a long time, so
# launch it detached, e.g.:
#   nohup bash paper_experiments/submit_backfill.sh > backfill.log 2>&1 &
# or inside tmux.
MAXQ=${MAXQ:-8}
POLL=${POLL:-120}

queue_count() { squeue -u "$USER" -h 2>/dev/null | wc -l | tr -d ' '; }

i=0
for cell in "${CELLS[@]}"; do
    read -r BASE_MODEL SIZE METHOD RANK LR DROPOUT <<< "$cell"
    METHOD_LC=$(echo "$METHOD" | tr '[:upper:]' '[:lower:]')
    CELL_ID="${METHOD_LC}_r${RANK}_lr${LR}_${SIZE}"
    i=$((i + 1))

    # Wait for a free queue slot (skipped in DRY mode).
    if [ "${DRY:-0}" != "1" ]; then
        while [ "$(queue_count)" -ge "$MAXQ" ]; do
            echo "[$(date +%H:%M:%S)] queue full ($(queue_count)/${MAXQ}); waiting ${POLL}s before ${CELL_ID}..."
            sleep "$POLL"
        done
    fi

    echo "[${i}/${TOTAL}] submit ${CELL_ID}  (${METHOD} R=${RANK} lr=${LR} D=${DROPOUT}  ${SIZE})"
    JID=$(submit slurm/run_cell.sh "$CELL_ID" "$METHOD" "$RANK" "$LR" "$DROPOUT" "$DATASET" "$EPOCHS" "$BASE_MODEL")
    echo "    job: ${JID}  (queue now ~$([ "${DRY:-0}" = "1" ] && echo DRY || queue_count)/${MAXQ})"
done

echo "[DONE] Submitted all ${TOTAL} cell jobs (throttled to <= ${MAXQ} in queue). squeue -u \$USER"

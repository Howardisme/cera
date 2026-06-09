#!/bin/bash
#SBATCH --job-name=cera_resub
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/resub_%j.log
#SBATCH --error=slurm_logs/resub_err_%j.log
#SBATCH --partition=8gpus
# NOTE: Add your cluster account line here, e.g.:
#   #SBATCH -A YOUR_ACCOUNT

# Cascading resubmitter for job arrays that exceed the per-user job limit.
# Submits the next batch of array tasks after the previous batch completes,
# then submits itself again for the following batch (if any remain).
#
# Usage (called automatically — do not invoke directly):
#   sbatch --dependency=afterok:<PREV_TRAIN_JOB> slurm/run_resubmitter.sh \
#       CONFIG_FILE START END TOTAL BATCH_SIZE PARTITION [RESUB_PARTITION]
#
# Arguments:
#   CONFIG_FILE      path to the array config file
#   START            first task ID of the next batch (1-indexed)
#   END              last task ID of the next batch
#   TOTAL            total number of tasks in the config
#   BATCH_SIZE       number of tasks per batch (default: 5)
#   PARTITION        SLURM partition for experiment (GPU) jobs
#   RESUB_PARTITION  SLURM partition for this resubmitter (CPU-only, default: dev)

CONFIG_FILE=${1:?CONFIG_FILE required}
START=${2:?START required}
END=${3:?END required}
TOTAL=${4:?TOTAL required}
BATCH_SIZE=${5:-5}
PARTITION=${6:-8gpus}
RESUB_PARTITION=${7:-8gpus}

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — run via sbatch}"
mkdir -p slurm_logs

echo "[RESUB] Submitting tasks ${START}-${END} of ${TOTAL} from ${CONFIG_FILE}"

EXP_JOB=$(sbatch --parsable --array=${START}-${END}%${BATCH_SIZE} \
    --partition=${PARTITION} \
    slurm/run_experiment_array.sh "$CONFIG_FILE")
echo "[RESUB] Experiment array job: $EXP_JOB (tasks ${START}-${END})"

# If more tasks remain, submit the next resubmitter after this batch finishes.
# Batch size is kept at BATCH_SIZE-1 (leaving 1 slot for this resubmitter while running).
NEXT_START=$((END + 1))
if [ "$NEXT_START" -le "$TOTAL" ]; then
    NEXT_END=$((NEXT_START + BATCH_SIZE - 1))
    [ "$NEXT_END" -gt "$TOTAL" ] && NEXT_END=$TOTAL
    RESUB_JOB=$(sbatch --parsable \
        --partition=${RESUB_PARTITION} \
        --dependency=afterany:${EXP_JOB} \
        slurm/run_resubmitter.sh "$CONFIG_FILE" "$NEXT_START" "$NEXT_END" "$TOTAL" "$BATCH_SIZE" "$PARTITION" "$RESUB_PARTITION")
    echo "[RESUB] Next resubmitter job: $RESUB_JOB (tasks ${NEXT_START}-${NEXT_END})"
else
    echo "[RESUB] All ${TOTAL} tasks submitted."
fi

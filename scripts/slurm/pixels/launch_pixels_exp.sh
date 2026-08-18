#!/bin/bash
# Pixel-space distillation experiment (none / slurpp / ppg), n=5 seeds x 3 IPC.
#
# Usage:
#   ./launch_pixels_exp.sh                      # 3 priors x 3 ipc x 5 seeds = 45 jobs
#   DRYRUN=1 ./launch_pixels_exp.sh             # preview, submit nothing
#   IPC="1" ./launch_pixels_exp.sh              # one ipc only (15 jobs)
#   PRIORS="slurpp" ./launch_pixels_exp.sh      # one prior only
#   SEEDS="3407" IPC="1 5" ./launch_pixels_exp.sh
#   ONLY=ppg ./launch_pixels_exp.sh             # substring filter on run_name
set -euo pipefail

SLURM_SCRIPT="$(dirname "$0")/distill_pixels.slurm"

OUTPUT_ROOT="$WORK/projects/GradientDistillation/output/pixels"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT/wandb" "$OUTPUT_ROOT/results"

DRYRUN=${DRYRUN:-0}
PRIORS=${PRIORS:-"none slurpp ppg"}
SEEDS=${SEEDS:-"3407 42 1234 2024 7"}
IPC=${IPC:-"1 3 5"}
ONLY=${ONLY:-}
MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-aqua20}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/aqua20}

ACCOUNT="rbw@h100"
CONSTRAINT="h100"
QOS="qos_gpu_h100-t3"

# slurpp paie le forward dual-UNet à l'init ; le coût croît avec IPC.
# Budgets prudents, à recalibrer sur les Elapsed réels (sacct).
time_for() {
    local prior="$1" ipc="$2"
    case "$prior:$ipc" in
        slurpp:1) echo "14:00:00" ;;
        slurpp:3) echo "18:00:00" ;;
        slurpp:5) echo "20:00:00" ;;
        *:1)      echo "10:00:00" ;;
        *:3)      echo "14:00:00" ;;
        *:5)      echo "18:00:00" ;;
    esac
}

njobs=0
submit() {
    local prior="$1" ipc="$2" seed="$3"

    local time
    time=$(time_for "$prior" "$ipc")
    if [ -z "$time" ]; then
        echo "no time budget for prior=$prior ipc=$ipc" >&2
        exit 2
    fi

    # la graine est apposée en aval par le code -> pas de _s${seed} ici
    local run_name="${MODEL}_${DATASET}_${prior}_ipc${ipc}"
    if [ -n "$ONLY" ] && [[ "$run_name" != *"$ONLY"* ]]; then
        return
    fi

    local args=(
        --job-name="pxl_${run_name}_s${seed}"
        --output="$LOG_DIR/%j_${run_name}_s${seed}.out"
        --error="$LOG_DIR/%j_${run_name}_s${seed}.err"
        --account="$ACCOUNT"
        --constraint="$CONSTRAINT"
        --qos="$QOS"
        --time="$time"
        --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",RUN_NAME="$run_name",PRIOR="$prior",SEED="$seed",IPC="$ipc",MODEL="$MODEL",DATASET="$DATASET",DATA_ROOT="$DATA_ROOT"
        "$SLURM_SCRIPT"
    )

    if [ "$DRYRUN" = "1" ]; then
        echo "sbatch ${args[*]}"
    else
        sbatch "${args[@]}"
    fi
    njobs=$((njobs+1))
}

for prior in $PRIORS; do
    for ipc in $IPC; do
        for seed in $SEEDS; do
            submit "$prior" "$ipc" "$seed"
        done
    done
done

echo "-------------------------------------------"
echo "priors($PRIORS) x ipc($IPC) x seeds($SEEDS)"
echo "$njobs jobs $([ "$DRYRUN" = "1" ] && echo 'preview (DRYRUN)' || echo 'submitted')"
[ -n "$ONLY" ] && echo "filter ONLY='$ONLY'"
echo "output root: $OUTPUT_ROOT"
echo "  logs   : $LOG_DIR"
echo "  wandb  : $OUTPUT_ROOT/wandb"
echo "  results: $OUTPUT_ROOT/results/$DATASET/$MODEL/<run_name>_s<seed>/"
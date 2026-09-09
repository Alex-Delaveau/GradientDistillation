#!/bin/bash
# Plain LGM baseline: pixel-space pyramid, no physical reparameterization.
# Same seeds, IPC and evaluation protocol as pixels_exp.
set -euo pipefail

SLURM_SCRIPT="$(dirname "$0")/distill_baseline_f4k.slurm"
OUTPUT_ROOT="$WORK/projects/GradientDistillation/output/baseline_f4k"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT/wandb" "$OUTPUT_ROOT/results"

DRYRUN=${DRYRUN:-0}
SEEDS=${SEEDS:-"3407 42 1234 2024 7"}
IPC=${IPC:-"1 3 5"}
AUGS=${AUGS:-10}
MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-f4k}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/F4K}

njobs=0
submit() {
    local ipc="$1" seed="$2"
    local time
    case "$ipc" in
        1) time="15:00:00" ;;
        3) time="15:00:00" ;;
        5) time="18:00:00" ;;
    esac

    local run_name="${MODEL}_${DATASET}_baseline_ipc${ipc}"
    local args=(
        --job-name="base_${run_name}_s${seed}"
        --output="$LOG_DIR/%j_${run_name}_s${seed}.out"
        --error="$LOG_DIR/%j_${run_name}_s${seed}.err"
        --account="rbw@h100" --constraint="h100" --qos="qos_gpu_h100-t3"
        --time="$time"
        --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",RUN_NAME="$run_name",SEED="$seed",IPC="$ipc",AUGS="$AUGS",MODEL="$MODEL",DATASET="$DATASET",DATA_ROOT="$DATA_ROOT"
        "$SLURM_SCRIPT"
    )
    if [ "$DRYRUN" = "1" ]; then echo "sbatch ${args[*]}"; else sbatch "${args[@]}"; fi
    njobs=$((njobs+1))
}

for ipc in $IPC; do for seed in $SEEDS; do submit "$ipc" "$seed"; done; done

echo "-------------------------------------------"
echo "ipc($IPC) x seeds($SEEDS) -> $njobs jobs"
echo "output root: $OUTPUT_ROOT"
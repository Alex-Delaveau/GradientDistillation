#!/bin/bash
# Learning-rate sweep: fixed parameterization, grid over (lr, lr_T, lr_B).
# Edit the three arrays below to set the grid.
#   Usage: ./launch_lr_exp.sh          submit all
#          DRYRUN=1 ./launch_lr_exp.sh preview only
set -euo pipefail

EXP_NAME=${EXP_NAME:-lr_exp}
OUTPUT_ROOT=$WORK/projects/GradientDistillation/output/$EXP_NAME
SLURM_SCRIPT="$(cd "$(dirname "$0")" && pwd)/lr_exp.slurm"
LOG_DIR=$OUTPUT_ROOT/logs
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT/wandb" "$OUTPUT_ROOT/results"

DRYRUN=${DRYRUN:-0}
MODEL=${MODEL:-dinov2_vitb}

# fixed parameterization for the whole sweep
DISTILL_MODE=pyramid
FORMATION_MODE=physics
PRIOR_INIT=slurpp
SAMPLE_INIT=medoids

# ---- LR grid ----
LR_LIST=(1e-3 2e-3 5e-3)
LRT_LIST=(2e-3 1e-2)
LRB_LIST=(2e-3 1e-2)

ACCOUNT="rbw@h100"; CONSTRAINT="h100"; QOS="qos_gpu_h100-t3"; TIME="10:00:00"

submit() {
    local lr=$1 lrt=$2 lrb=$3
    local run_name="${MODEL}_${DISTILL_MODE}_${PRIOR_INIT}_lr${lr}_lrT${lrt}_lrB${lrb}"
    local args=(
        --job-name="lrexp_${run_name}"
        --output="$LOG_DIR/%j_${run_name}.out"
        --error="$LOG_DIR/%j_${run_name}.err"
        --account="$ACCOUNT" --constraint="$CONSTRAINT" --qos="$QOS" --time="$TIME"
        --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",RUN_NAME="$run_name",LR="$lr",LR_T="$lrt",LR_B="$lrb",DISTILL_MODE="$DISTILL_MODE",FORMATION_MODE="$FORMATION_MODE",PRIOR_INIT="$PRIOR_INIT",SAMPLE_INIT="$SAMPLE_INIT"
        "$SLURM_SCRIPT"
    )
    if [ "$DRYRUN" = "1" ]; then echo "sbatch ${args[*]}"; else sbatch "${args[@]}"; fi
}

n=0
for lr in "${LR_LIST[@]}"; do
    for lrt in "${LRT_LIST[@]}"; do
        for lrb in "${LRB_LIST[@]}"; do
            submit "$lr" "$lrt" "$lrb"
            n=$((n+1))
        done
    done
done

echo "-------------------------------------------"
echo "$n jobs $([ "$DRYRUN" = "1" ] && echo 'preview (DRYRUN)' || echo 'submitted')"
echo "output root: $OUTPUT_ROOT"
echo "  logs   : $LOG_DIR"
echo "  wandb  : $OUTPUT_ROOT/wandb"
echo "  results: $OUTPUT_ROOT/results/$DATASET/$MODEL/<run_name>/"
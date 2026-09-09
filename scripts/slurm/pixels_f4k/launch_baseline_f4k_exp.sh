#!/bin/bash
# Plain LGM baseline: pixel-space pyramid, no physical reparameterization.
# Same seeds, IPC and evaluation protocol as pixels_exp.
#
# Usage:
#   ./launch_baseline_f4k.sh                     # h100, 3 ipc x 5 seeds = 15 jobs
#   ARCH=a100 ./launch_baseline_f4k.sh           # same grid on A100-80G
#   ARCH=a100 DRYRUN=1 ./launch_baseline_f4k.sh  # preview, submit nothing
#   IPC="1" SEEDS="3407" ./launch_baseline_f4k.sh        # single smoke run
#   TIME_IPC1=06:00:00 ./launch_baseline_f4k.sh          # tighter walltime -> backfill
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

# --- resources, resolved per architecture ---
# CPUS follows the node ratio: h100 96c/4gpu, a100 64c/8gpu, v100 40c/4gpu.
# Asking for more than the ratio is rejected by the scheduler.
ARCH=${ARCH:-h100}
case "$ARCH" in
  h100)
    ACCOUNT_DEF="rbw@h100";  CONSTRAINT_DEF="h100"
    QOS_DEF="qos_gpu_h100-t3";  CPUS_DEF=24 ;;
  a100)
    ACCOUNT_DEF="rbw@a100";  CONSTRAINT_DEF="a100"
    QOS_DEF="qos_gpu_a100-t3";  CPUS_DEF=8 ;;
  v100)
    ACCOUNT_DEF="rbw@v100";  CONSTRAINT_DEF="v100-32g"
    QOS_DEF="qos_gpu-t3";    CPUS_DEF=10 ;;
  *)
    echo "ARCH inconnue: '$ARCH' (attendu: h100 | a100 | v100)" >&2; exit 1 ;;
esac

ACCOUNT=${ACCOUNT:-$ACCOUNT_DEF}
CONSTRAINT=${CONSTRAINT:-$CONSTRAINT_DEF}
QOS=${QOS:-$QOS_DEF}
CPUS=${CPUS:-$CPUS_DEF}

# Walltime per IPC. Keep these as tight as your sacct history allows -- a 15h
# request is excluded from almost every backfill window.
TIME_IPC1=${TIME_IPC1:-"15:00:00"}
TIME_IPC3=${TIME_IPC3:-"15:00:00"}
TIME_IPC5=${TIME_IPC5:-"18:00:00"}

njobs=0
submit() {
    local ipc="$1" seed="$2"
    local time
    case "$ipc" in
        1) time="$TIME_IPC1" ;;
        3) time="$TIME_IPC3" ;;
        5) time="$TIME_IPC5" ;;
        *) echo "IPC sans walltime defini: $ipc" >&2; exit 1 ;;
    esac

    local run_name="${MODEL}_${DATASET}_baseline_ipc${ipc}"
    local args=(
        --job-name="base_${run_name}_s${seed}"
        --output="$LOG_DIR/%j_${run_name}_s${seed}.out"
        --error="$LOG_DIR/%j_${run_name}_s${seed}.err"
        --account="$ACCOUNT"
        --constraint="$CONSTRAINT"
        --qos="$QOS"
        --time="$time"
        --cpus-per-task="$CPUS"
        --export=ALL,ARCH="$ARCH",OUTPUT_ROOT="$OUTPUT_ROOT",RUN_NAME="$run_name",SEED="$seed",IPC="$ipc",AUGS="$AUGS",MODEL="$MODEL",DATASET="$DATASET",DATA_ROOT="$DATA_ROOT"
        "$SLURM_SCRIPT"
    )
    if [ "$DRYRUN" = "1" ]; then echo "sbatch ${args[*]}"; else sbatch "${args[@]}"; fi
    njobs=$((njobs+1))
}

for ipc in $IPC; do for seed in $SEEDS; do submit "$ipc" "$seed"; done; done

echo "-------------------------------------------"
echo "arch=$ARCH account=$ACCOUNT qos=$QOS cpus=$CPUS"
echo "walltime: ipc1=$TIME_IPC1 ipc3=$TIME_IPC3 ipc5=$TIME_IPC5"
echo "ipc($IPC) x seeds($SEEDS) -> $njobs jobs"
echo "output root: $OUTPUT_ROOT"
echo "  logs   : $LOG_DIR"
echo "  wandb  : $OUTPUT_ROOT/wandb"
echo "  results: $OUTPUT_ROOT/results/$DATASET/${MODEL}/<run_name>/"
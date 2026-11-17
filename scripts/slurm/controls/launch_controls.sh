#!/bin/bash
# Camera-ready controls (none_frozen / affine / slurpp_t1 / none_t3), IPC 1 x 5 seeds.
# Same seeds, QoS, account and walltime policy as launch_pixels_exp.sh.
#
# Usage:
#   ./launch_controls.sh                          # 4 variants x 1 ipc x 5 seeds = 20 jobs
#   DRYRUN=1 ./launch_controls.sh                 # preview, submit nothing
#   VARIANTS="affine" ./launch_controls.sh        # one variant only
#   SEEDS="3407" ./launch_controls.sh
#   ONLY=slurpp ./launch_controls.sh              # substring filter on run_name
#   NSEG=2 ./launch_controls.sh                   # 2 chained segments per run (afterany);
#                                                 # the 2nd one resumes from ckpt.pth or exits on DONE
#   TIME=08:00:00 ./launch_controls.sh            # same walltime for every job
# Resubmitting a run killed at walltime: rerun the same command filtered with
# VARIANTS/SEEDS (finished runs exit immediately thanks to the DONE marker).
set -euo pipefail

SLURM_SCRIPT="$(dirname "$0")/distill_controls.slurm"

OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORK/projects/GradientDistillation/output/controls"}
LOG_DIR="$OUTPUT_ROOT/logs"

DRYRUN=${DRYRUN:-0}
VARIANTS=${VARIANTS:-"none_frozen affine slurpp_t1 none_t3"}
SEEDS=${SEEDS:-"3407 42 1234 2024 7"}
IPC=${IPC:-"1"}
AUGS=${AUGS:-10}
ONLY=${ONLY:-}
MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-aqua20}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/aqua20}
ARCH=${ARCH:-h100}
NSEG=${NSEG:-1}
TIME=${TIME:-}

ACCOUNT=${ACCOUNT:-"rbw@h100"}
CONSTRAINT=${CONSTRAINT:-"h100"}
QOS=${QOS:-"qos_gpu_h100-t3"}       # max walltime 20h
QOS_MAX_H=20

if [ "$DRYRUN" != "1" ]; then
    mkdir -p "$LOG_DIR" "$OUTPUT_ROOT/wandb" "$OUTPUT_ROOT/results"
fi

# Same budgets as launch_pixels_exp.sh: slurpp_t1 pays the SLURPP forward at init.
time_for() {
    local variant="$1" ipc="$2"
    case "$variant:$ipc" in
        slurpp*:1) echo "14:00:00" ;;
        slurpp*:3) echo "18:00:00" ;;
        slurpp*:5) echo "20:00:00" ;;
        *:1)       echo "10:00:00" ;;
        *:3)       echo "14:00:00" ;;
        *:5)       echo "18:00:00" ;;
    esac
}

njobs=0
submit() {
    local variant="$1" ipc="$2" seed="$3"

    local time="$TIME"
    [ -z "$time" ] && time=$(time_for "$variant" "$ipc")
    if [ -z "$time" ]; then
        echo "no time budget for variant=$variant ipc=$ipc" >&2; exit 2
    fi
    local hours=$((10#${time%%:*}))
    if [ "$hours" -gt "$QOS_MAX_H" ] || { [ "$hours" -eq "$QOS_MAX_H" ] && [ "${time#*:}" != "00:00" ]; }; then
        echo "walltime $time exceeds ${QOS_MAX_H}h ($QOS)" >&2; exit 2
    fi

    # must match the name derived in distill_controls.slurm (seed appended by the code);
    # the job name ctl_<run_name>_s<seed> is what lets a finished run cancel its own
    # pending segments, and only those
    local run_name="${MODEL}_${DATASET}_${variant}_ipc${ipc}"
    if [ -n "$ONLY" ] && [[ "$run_name" != *"$ONLY"* ]]; then
        return
    fi

    local args=(
        --job-name="ctl_${run_name}_s${seed}"
        --output="$LOG_DIR/%j_${run_name}_s${seed}.out"
        --error="$LOG_DIR/%j_${run_name}_s${seed}.err"
        --account="$ACCOUNT"
        --constraint="$CONSTRAINT"
        --qos="$QOS"
        --time="$time"
        --export=ALL,ARCH="$ARCH",OUTPUT_ROOT="$OUTPUT_ROOT",VARIANT="$variant",SEED="$seed",IPC="$ipc",AUGS="$AUGS",MODEL="$MODEL",DATASET="$DATASET",DATA_ROOT="$DATA_ROOT"
        "$SLURM_SCRIPT"
    )

    local prev="" k
    local -a dep
    for k in $(seq 1 "$NSEG"); do
        dep=()
        [ -n "$prev" ] && dep=(--dependency=afterany:"$prev")
        if [ "$DRYRUN" = "1" ]; then
            echo "sbatch ${dep[*]} ${args[*]}"; prev="<seg$k>"
        else
            prev=$(sbatch --parsable "${dep[@]}" "${args[@]}"); prev=${prev%%;*}
            echo "  ${run_name}_s${seed}  seg $k/$NSEG -> $prev"
        fi
        njobs=$((njobs+1))
    done
}

for variant in $VARIANTS; do
    for ipc in $IPC; do
        for seed in $SEEDS; do
            submit "$variant" "$ipc" "$seed"
        done
    done
done

echo "-------------------------------------------"
echo "variants($VARIANTS) x ipc($IPC) x seeds($SEEDS) x nseg($NSEG)"
echo "$njobs jobs $([ "$DRYRUN" = "1" ] && echo 'preview (DRYRUN)' || echo 'submitted')"
[ -n "$ONLY" ] && echo "filter ONLY='$ONLY'"
echo "arch=$ARCH account=$ACCOUNT qos=$QOS"
echo "output root: $OUTPUT_ROOT"
echo "  logs   : $LOG_DIR"
echo "  wandb  : $OUTPUT_ROOT/wandb"
echo "  results: $OUTPUT_ROOT/results/$DATASET/$MODEL/<run_name>_s<seed>/"

#!/bin/bash
# Pixel-space distillation experiment (none / slurpp / ppg), n=5 seeds x 3 IPC.
#
# Usage:
#   ./launch_pixels_fishnet20_exp.sh                      # h100, 3 priors x 3 ipc x 5 seeds = 45 jobs
#   ARCH=a100 ./launch_pixels_fishnet20_exp.sh            # same grid on A100-80G
#   DRYRUN=1 ./launch_pixels_fishnet20_exp.sh             # preview, submit nothing
#   IPC="1" ./launch_pixels_fishnet20_exp.sh              # one ipc only (15 jobs)
#   PRIORS="slurpp" ./launch_pixels_fishnet20_exp.sh      # one prior only
#   SEEDS="3407" IPC="1 5" ./launch_pixels_fishnet20_exp.sh
#   ONLY=ppg ./launch_pixels_fishnet20_exp.sh             # substring filter on run_name
#   TIME=06:00:00 ./launch_pixels_fishnet20_exp.sh        # same walltime for all jobs -> backfill
set -euo pipefail

SLURM_SCRIPT="$(dirname "$0")/distill_pixels_fishnet20.slurm"

OUTPUT_ROOT="$WORK/projects/GradientDistillation/output/pixels_fishnet20"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT/wandb" "$OUTPUT_ROOT/results"

DRYRUN=${DRYRUN:-0}
PRIORS=${PRIORS:-"none slurpp ppg"}
SEEDS=${SEEDS:-"3407 42 1234 2024 7"}
IPC=${IPC:-"1 3 5"}
AUGS=${AUGS:-10}
ONLY=${ONLY:-}
TIME=${TIME:-}
MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-fishnet20}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets}

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


# MODE=t3  : un job long par run (comportement actuel)
# MODE=dev : segments de 2 h chaînés (afterany), reprise sur checkpoint
MODE=${MODE:-t3}
case "$ARCH:$MODE" in
  h100:dev) QOS_DEF="qos_gpu_h100-dev" ;;
  a100:dev) QOS_DEF="qos_gpu_a100-dev" ;;
  v100:dev) QOS_DEF="qos_gpu-dev" ;;
  *:t3)     ;;
  *) echo "MODE inconnu: '$MODE' (attendu: t3 | dev)" >&2; exit 1 ;;
esac

# segments de 2 h par run en mode dev : à recalibrer sur les logs (itérations atteintes en 2 h)
SEG_IPC1=${SEG_IPC1:-4}
SEG_IPC3=${SEG_IPC3:-6}
SEG_IPC5=${SEG_IPC5:-9}
DEV_CAP=10          # MaxSubmitPU des QoS dev
PLAN_ONLY=0

seg_for() {
    case "$1" in
        1) echo "$SEG_IPC1" ;; 3) echo "$SEG_IPC3" ;; 5) echo "$SEG_IPC5" ;;
    esac
}

ACCOUNT=${ACCOUNT:-$ACCOUNT_DEF}
CONSTRAINT=${CONSTRAINT:-$CONSTRAINT_DEF}
QOS=${QOS:-$QOS_DEF}
CPUS=${CPUS:-$CPUS_DEF}

# slurpp paie le forward dual-UNet à l'init ; le coût croît avec IPC.
# Budgets prudents, à recalibrer sur les Elapsed réels (sacct).
# a100/v100 : provisoires, plafonnés à 20h (MaxWall t3).
time_for() {
    local prior="$1" ipc="$2"
    case "$ARCH:$prior:$ipc" in
        h100:slurpp:1) echo "14:00:00" ;;
        h100:slurpp:3) echo "18:00:00" ;;
        h100:slurpp:5) echo "20:00:00" ;;
        h100:*:1)      echo "10:00:00" ;;
        h100:*:3)      echo "14:00:00" ;;
        h100:*:5)      echo "18:00:00" ;;
        a100:*:1)      echo "16:00:00" ;;
        a100:*:*)      echo "20:00:00" ;;
        v100:*:*)      echo "20:00:00" ;;
    esac
}

njobs=0
submit() {
    local prior="$1" ipc="$2" seed="$3"

    # la graine est apposée en aval par le code -> pas de _s${seed} ici
    local run_name="${MODEL}_${DATASET}_${prior}_ipc${ipc}"
    if [ -n "$ONLY" ] && [[ "$run_name" != *"$ONLY"* ]]; then
        return
    fi

    local time nseg
    if [ "$MODE" = "dev" ]; then
        time="02:00:00"; nseg=$(seg_for "$ipc")
    else
        time=${TIME:-$(time_for "$prior" "$ipc")}; nseg=1
    fi
    if [ -z "$time" ] || [ -z "$nseg" ]; then
        echo "no time/segment budget for arch=$ARCH mode=$MODE prior=$prior ipc=$ipc" >&2
        exit 2
    fi

    # passe de planification : on compte sans soumettre
    if [ "$PLAN_ONLY" = "1" ]; then njobs=$((njobs+nseg)); return; fi

    local args=(
        --job-name="pxl_${run_name}_s${seed}"
        --output="$LOG_DIR/%j_${run_name}_s${seed}.out"
        --error="$LOG_DIR/%j_${run_name}_s${seed}.err"
        --account="$ACCOUNT"
        --constraint="$CONSTRAINT"
        --qos="$QOS"
        --time="$time"
        --cpus-per-task="$CPUS"
        --export=ALL,ARCH="$ARCH",OUTPUT_ROOT="$OUTPUT_ROOT",RUN_NAME="$run_name",PRIOR="$prior",SEED="$seed",IPC="$ipc",AUGS="$AUGS",MODEL="$MODEL",DATASET="$DATASET",DATA_ROOT="$DATA_ROOT"
        "$SLURM_SCRIPT"
    )

    local prev="" k
    local -a dep
    for k in $(seq 1 "$nseg"); do
        dep=()
        [ -n "$prev" ] && dep=(--dependency=afterany:"$prev")
        if [ "$DRYRUN" = "1" ]; then
            echo "sbatch ${dep[*]} ${args[*]}"
            prev="<seg$k>"
        else
            prev=$(sbatch --parsable "${dep[@]}" "${args[@]}")
            prev=${prev%%;*}
            echo "  ${run_name}_s${seed}  seg $k/$nseg -> $prev"
        fi
        njobs=$((njobs+1))
    done
}

run_grid() {
    for prior in $PRIORS; do
        for ipc in $IPC; do
            for seed in $SEEDS; do
                submit "$prior" "$ipc" "$seed"
            done
        done
    done
}

PLAN_ONLY=1; njobs=0; run_grid; planned=$njobs
if [ "$MODE" = "dev" ]; then
    current=$(squeue -u "$USER" -h -q "$QOS" | wc -l)
    if (( current + planned > DEV_CAP )); then
        echo "[dev] $planned segments prévus + $current déjà en file > $DEV_CAP (MaxSubmitPU)." >&2
        echo "      Réduis PRIORS/IPC/SEEDS ou les SEG_IPCn." >&2
        [ "$DRYRUN" = "1" ] || exit 3
    fi
fi
PLAN_ONLY=0; njobs=0; run_grid

echo "-------------------------------------------"
echo "arch=$ARCH account=$ACCOUNT qos=$QOS cpus=$CPUS"
echo "priors($PRIORS) x ipc($IPC) x seeds($SEEDS)"
echo "$njobs jobs $([ "$DRYRUN" = "1" ] && echo 'preview (DRYRUN)' || echo 'submitted')"
[ -n "$ONLY" ] && echo "filter ONLY='$ONLY'"
[ -n "$TIME" ] && echo "walltime override: $TIME"
echo "output root: $OUTPUT_ROOT"
echo "  logs   : $LOG_DIR"
echo "  wandb  : $OUTPUT_ROOT/wandb"
echo "  results: $OUTPUT_ROOT/results/$DATASET/$MODEL/<run_name>_s<seed>/"
echo "mode=$MODE $([ "$MODE" = "dev" ] && echo "segments: ipc1=$SEG_IPC1 ipc3=$SEG_IPC3 ipc5=$SEG_IPC5")"
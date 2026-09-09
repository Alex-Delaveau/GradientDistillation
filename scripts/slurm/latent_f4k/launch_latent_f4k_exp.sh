#!/bin/bash
# Launch the latent-space distillation experiment (GLaD-SLURPP).
# Variants:
#   predlatent    : SLURPP dual-UNet init, physics composition (main)
#   decoder_only  : raw SD-VAE latent, no dual-UNet, no physics (ablation)
#  -> 2 variants per seed (x LRs if LRS has several values).
#
# Usage:
#   ./launch_latent_exp.sh                          # h100, 2 modes x 1 seed = 2 jobs
#   ARCH=a100 ./launch_latent_exp.sh                # same runs on A100-80G
#   ARCH=a100 DRYRUN=1 ./launch_latent_exp.sh       # preview, submit nothing
#   MODES="decoder_only" ./launch_latent_exp.sh     # ablation only
#   SEEDS="3407 42 1234 2024 7" ./launch_latent_exp.sh   # multi-seed n=5
#   LRS="1e-4 1e-3 1e-2" MODES="decoder_only" ./launch_latent_exp.sh  # LR mini-sweep
#   ONLY=decoder ./launch_latent_exp.sh             # substring filter on run_name
#   TIME_PREDLATENT=06:00:00 ./launch_latent_exp.sh # shorter walltime -> backfill
set -euo pipefail

SLURM_SCRIPT="$(dirname "$0")/latent_f4k_exp.slurm"

# --- absolute output root (single source of truth, propagated to the job) ---
OUTPUT_ROOT="$WORK/projects/GradientDistillation/output/latent_f4k_exp"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT/wandb" "$OUTPUT_ROOT/results"

DRYRUN=${DRYRUN:-0}
SEEDS=${SEEDS:-"3407"}
MODES=${MODES:-"predlatent decoder_only"}
LRS=${LRS:-"1e-3"}
ONLY=${ONLY:-}

MODEL=${MODEL:-dinov2_vitb}
OPTIM=${OPTIM:-adam}

# --- resources, resolved per architecture ---
# CPUS follows the node ratio: h100 96c/4gpu, a100 64c/8gpu, v100 40c/4gpu.
# Asking for more than the ratio is rejected by the scheduler.
ARCH=${ARCH:-h100}
case "$ARCH" in
  h100)
    ACCOUNT_DEF="rbw@h100";  CONSTRAINT_DEF="h100"
    QOS_DEF="qos_gpu_h100-t3";  CPUS_DEF=24;  CHUNK_DEF=2 ;;
  a100)
    ACCOUNT_DEF="rbw@a100";  CONSTRAINT_DEF="a100"
    QOS_DEF="qos_gpu_a100-t3";  CPUS_DEF=8;   CHUNK_DEF=2 ;;
  v100)
    # 32GB only -- predlatent will very likely OOM at latent_res=512.
    ACCOUNT_DEF="rbw@v100";  CONSTRAINT_DEF="v100-32g"
    QOS_DEF="qos_gpu-t3";    CPUS_DEF=10;  CHUNK_DEF=1 ;;
  *)
    echo "ARCH inconnue: '$ARCH' (attendu: h100 | a100 | v100)" >&2; exit 1 ;;
esac

ACCOUNT=${ACCOUNT:-$ACCOUNT_DEF}
CONSTRAINT=${CONSTRAINT:-$CONSTRAINT_DEF}
QOS=${QOS:-$QOS_DEF}
CPUS=${CPUS:-$CPUS_DEF}
LATENT_CHUNK=${LATENT_CHUNK:-$CHUNK_DEF}

# predlatent pays the dual-UNet forward at init; decoder_only skips it.
# Keep these as tight as your sacct history allows -- a 20h request is
# excluded from almost every backfill window.
TIME_PREDLATENT=${TIME_PREDLATENT:-"20:00:00"}
TIME_DECODER_ONLY=${TIME_DECODER_ONLY:-"10:00:00"}

if [ "$ARCH" = "v100" ] && [[ "$MODES" == *predlatent* ]]; then
    echo "WARNING: predlatent on v100-32g will likely OOM at latent_res=512." >&2
fi

njobs=0

submit() {
    local mode="$1" lr="$2" seed="$3"
    local time="$TIME_DECODER_ONLY"
    [ "$mode" = "predlatent" ] && time="$TIME_PREDLATENT"

    local run_name="${MODEL}_latent_${mode}_medoids_${OPTIM}_lr${lr}"

    if [ -n "$ONLY" ] && [[ "$run_name" != *"$ONLY"* ]]; then
        return
    fi

    local args=(
        --job-name="lexp_${run_name}"
        --output="$LOG_DIR/%j_${run_name}.out"
        --error="$LOG_DIR/%j_${run_name}.err"
        --account="$ACCOUNT"
        --constraint="$CONSTRAINT"
        --qos="$QOS"
        --time="$time"
        --cpus-per-task="$CPUS"
        --export=ALL,ARCH="$ARCH",OUTPUT_ROOT="$OUTPUT_ROOT",RUN_NAME="$run_name",SEED="$seed",MODEL="$MODEL",OPTIM="$OPTIM",LR="$lr",LATENT_MODE="$mode",LATENT_CHUNK="$LATENT_CHUNK"
        "$SLURM_SCRIPT"
    )

    if [ "$DRYRUN" = "1" ]; then
        echo "sbatch ${args[*]}"
    else
        sbatch "${args[@]}"
    fi
    njobs=$((njobs+1))
}

for mode in $MODES; do
    for lr in $LRS; do
        for seed in $SEEDS; do
            submit "$mode" "$lr" "$seed"
        done
    done
done

echo "-------------------------------------------"
echo "arch=$ARCH account=$ACCOUNT qos=$QOS cpus=$CPUS chunk=$LATENT_CHUNK"
echo "walltime: predlatent=$TIME_PREDLATENT decoder_only=$TIME_DECODER_ONLY"
echo "modes($MODES) x lrs($LRS) x seeds($SEEDS)"
echo "$njobs jobs $([ "$DRYRUN" = "1" ] && echo 'preview (DRYRUN)' || echo 'submitted')"
[ -n "$ONLY" ] && echo "filter ONLY='$ONLY'"
echo "output root: $OUTPUT_ROOT"
echo "  logs   : $LOG_DIR"
echo "  wandb  : $OUTPUT_ROOT/wandb"
echo "  results: $OUTPUT_ROOT/results/f4k/${MODEL}/<run_name>/"
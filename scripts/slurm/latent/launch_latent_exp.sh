#!/bin/bash
# Launch the latent-space distillation experiment (GLaD-SLURPP).
# Variants:
#   predlatent    : SLURPP dual-UNet init, physics composition (main)
#   decoder_only  : raw SD-VAE latent, no dual-UNet, no physics (ablation)
#  -> 2 variants per seed (x LRs if LRS has several values).
#
# Usage:
#   ./launch_latent_exp.sh                          # 2 modes x 1 seed = 2 jobs
#   DRYRUN=1 ./launch_latent_exp.sh                 # preview, submit nothing
#   MODES="decoder_only" ./launch_latent_exp.sh     # ablation only
#   SEEDS="3407 42 1234 2024 7" ./launch_latent_exp.sh   # multi-seed n=5
#   LRS="1e-4 1e-3 1e-2" MODES="decoder_only" ./launch_latent_exp.sh  # LR mini-sweep
#   ONLY=decoder ./launch_latent_exp.sh             # substring filter on run_name
set -euo pipefail

SLURM_SCRIPT="$(dirname "$0")/latent_exp.slurm"

# --- absolute output root (single source of truth, propagated to the job) ---
OUTPUT_ROOT="$WORK/projects/GradientDistillation/output/latent_exp"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT/wandb" "$OUTPUT_ROOT/results"

DRYRUN=${DRYRUN:-0}
SEEDS=${SEEDS:-"3407"}
MODES=${MODES:-"predlatent decoder_only"}
LRS=${LRS:-"1e-3"}
ONLY=${ONLY:-}

MODEL=${MODEL:-dinov2_vitb}
OPTIM=${OPTIM:-adam}
LATENT_CHUNK=${LATENT_CHUNK:-2}

# --- resources (h100) ---
ACCOUNT="rbw@h100"
CONSTRAINT="h100"
QOS="qos_gpu_h100-t3"
# predlatent pays the dual-UNet forward at init; decoder_only skips it.
TIME_PREDLATENT="20:00:00"
TIME_DECODER_ONLY="10:00:00"

njobs=0

submit() {
    local mode="$1" lr="$2" seed="$3"
    local time="$TIME_DECODER_ONLY"
    [ "$mode" = "predlatent" ] && time="$TIME_PREDLATENT"

    local run_name="${MODEL}_latent_${mode}_medoids_${OPTIM}_lr${lr}_s${seed}"

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
        --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",RUN_NAME="$run_name",SEED="$seed",MODEL="$MODEL",OPTIM="$OPTIM",LR="$lr",LATENT_MODE="$mode",LATENT_CHUNK="$LATENT_CHUNK"
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
echo "modes($MODES) x lrs($LRS) x seeds($SEEDS)"
echo "$njobs jobs $([ "$DRYRUN" = "1" ] && echo 'preview (DRYRUN)' || echo 'submitted')"
[ -n "$ONLY" ] && echo "filter ONLY='$ONLY'"
echo "output root: $OUTPUT_ROOT"
echo "  logs   : $LOG_DIR"
echo "  wandb  : $OUTPUT_ROOT/wandb"
echo "  results: $OUTPUT_ROOT/results/aqua20/${MODEL}/<run_name>/"
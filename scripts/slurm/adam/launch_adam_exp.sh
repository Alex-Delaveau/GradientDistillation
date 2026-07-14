#!/bin/bash
# Launch the optimizer experiment (pyramid only, medoids only, no freezing).
#   1  identity                (LGM fork baseline, prior unused)
#   1  physics none
#   1  physics slurpp
#  -> 3 variants per optimizer per seed.
#
# Usage:
#   ./launch_adam_exp.sh                      # 3 variants x {sgd, adam} = 6 jobs
#   DRYRUN=1 ./launch_adam_exp.sh             # preview, submit nothing
#   OPTIMS="adam" ./launch_adam_exp.sh        # adam only
#   ONLY=slurpp ./launch_adam_exp.sh          # substring filter on run_name
set -euo pipefail

SLURM_SCRIPT="$(dirname "$0")/adam_exp.slurm"

# --- absolute output root (single source of truth, propagated to the job) ---
OUTPUT_ROOT="$WORK/projects/GradientDistillation/output/adam_exp"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT/wandb" "$OUTPUT_ROOT/results"

DRYRUN=${DRYRUN:-0}
SEEDS=${SEEDS:-"3407"}
OPTIMS=${OPTIMS:-"sgd adam"}
ONLY=${ONLY:-}

# Adam needs its own LR scale; SGD LRs diverge. Set per optimizer.
LR_ARGS_ADAM=${LR_ARGS_ADAM:-""}
LR_ARGS_SGD=${LR_ARGS_SGD:-""}

# --- resources (h100) ---
ACCOUNT="rbw@h100"
CONSTRAINT="h100"
QOS="qos_gpu_h100-t3"
TIME_DEFAULT="10:00:00"
TIME_SLURPP="20:00:00"

submit() {
    local run_name="$1" distill="$2" formation="$3" prior="$4" sample="$5"
    local time="$TIME_DEFAULT"
    [ "$prior" = "slurpp" ] && time="$TIME_SLURPP"

    if [ -n "$ONLY" ] && [[ "$run_name" != *"$ONLY"* ]]; then
        return
    fi

    for optim in $OPTIMS; do
        # optimizer is the studied factor: always in the run name
        local oname="${run_name}_${optim}"
        local lr_args="$LR_ARGS_SGD"
        [ "$optim" = "adam" ] && lr_args="$LR_ARGS_ADAM"

        for seed in $SEEDS; do
            local sname="${oname}_s${seed}"
            local args=(
                --job-name="aexp_${sname}"
                --output="$LOG_DIR/%j_${sname}.out"
                --error="$LOG_DIR/%j_${sname}.err"
                --account="$ACCOUNT"
                --constraint="$CONSTRAINT"
                --qos="$QOS"
                --time="$time"
                --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",RUN_NAME="$sname",SEED="$seed",OPTIM="$optim",LR_ARGS="$lr_args",DISTILL_MODE="$distill",FORMATION_MODE="$formation",PRIOR_INIT="$prior",SAMPLE_INIT="$sample",FREEZE_T="false",FREEZE_B="false"
                "$SLURM_SCRIPT"
            )

            if [ "$DRYRUN" = "1" ]; then
                echo "sbatch ${args[*]}"
            else
                sbatch "${args[@]}"
            fi
            njobs=$((njobs+1))
        done
    done
}

MODEL=${MODEL:-dinov2_vitb}
D="pyramid"
S="medoids"
n=0
njobs=0

# --- identity (prior/sample unused by the fork path) ---
submit "${MODEL}_${D}_identity" "$D" identity none "$S"
n=$((n+1))

# --- physics, no prior ---
submit "${MODEL}_${D}_none_${S}" "$D" physics none "$S"
n=$((n+1))

# --- physics, slurpp prior ---
submit "${MODEL}_${D}_slurpp_${S}" "$D" physics slurpp "$S"
n=$((n+1))

echo "-------------------------------------------"
echo "$n variants x optims($OPTIMS) x seeds($SEEDS)"
echo "$njobs jobs $([ "$DRYRUN" = "1" ] && echo 'preview (DRYRUN)' || echo 'submitted')"
[ -n "$ONLY" ] && echo "filter ONLY='$ONLY'"
echo "output root: $OUTPUT_ROOT"
echo "  logs   : $LOG_DIR"
echo "  wandb  : $OUTPUT_ROOT/wandb"
echo "  results: $OUTPUT_ROOT/results/aqua20/dinov2_vitb/<run_name>/"
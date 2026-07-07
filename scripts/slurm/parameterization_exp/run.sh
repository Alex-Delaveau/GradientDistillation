#!/bin/bash
# Launch the 26 parameterization variants.
#   2  identity        (pixel|pyramid)
#   4  physics none     ({pixel,pyramid} x {medoids,random})
#   4  physics ppg      ({pixel,pyramid} x {medoids,random})
#  16  physics slurpp   ({pixel,pyramid} x {medoids,random} x freeze_T x freeze_B)
#
# Usage:
#   ./launch_parameterization_exp.sh            # submit all
#   DRYRUN=1 ./launch_parameterization_exp.sh   # print sbatch lines, submit nothing
set -euo pipefail

SLURM_SCRIPT="$(dirname "$0")/parameterization_exp.slurm"
EXP_ROOT="$WORK/projects/GradientDistillation/parameterization_exp"
LOG_DIR="$EXP_ROOT/logs"
WANDB_DIR="$EXP_ROOT/wandb"
mkdir -p "$LOG_DIR" "$WANDB_DIR"

# results live under logged_files/parameterization_exp/ (hardcoded in linear_gm);
# expose them inside the exp folder so everything is findable in one place.
ln -sfn "$WORK/projects/GradientDistillation/logged_files/parameterization_exp" "$EXP_ROOT/results"

DRYRUN=${DRYRUN:-0}

# --- resources (bump for slurpp: dual-UNet init is heavier) ---
ACCOUNT="rbw@v100"
CONSTRAINT="v100-32g"
QOS="qos_gpu-t3"
TIME_DEFAULT="10:00:00"
TIME_SLURPP="20:00:00"
# For H100 instead: ACCOUNT="rbw@h100"; CONSTRAINT="h100"; (adjust qos accordingly)

# maps 0/1 -> false/true for the CLI (pydantic-style bool parsing)
bool() { [ "$1" = "1" ] && echo "true" || echo "false"; }

submit() {
    local run_name="$1" distill="$2" formation="$3" prior="$4" sample="$5" ft="$6" fb="$7"
    local time="$TIME_DEFAULT"
    [ "$prior" = "slurpp" ] && time="$TIME_SLURPP"

    local args=(
        --job-name="pexp_${run_name}"
        --output="$LOG_DIR/%j_${run_name}.out"
        --error="$LOG_DIR/%j_${run_name}.err"
        --account="$ACCOUNT"
        --constraint="$CONSTRAINT"
        --qos="$QOS"
        --time="$time"
        --export=ALL,RUN_NAME="$run_name",DISTILL_MODE="$distill",FORMATION_MODE="$formation",PRIOR_INIT="$prior",SAMPLE_INIT="$sample",FREEZE_T="$(bool "$ft")",FREEZE_B="$(bool "$fb")"
        "$SLURM_SCRIPT"
    )

    if [ "$DRYRUN" = "1" ]; then
        echo "sbatch ${args[*]}"
    else
        sbatch "${args[@]}"
    fi
}

MODEL=${MODEL:-dinov2_vitb}
n=0

# --- 2 identity (prior/sample unused by the fork path) ---
for d in pixel pyramid; do
    submit "${MODEL}_${d}_identity" "$d" identity none medoids 0 0
    n=$((n+1))
done

# --- 8 physics none / ppg ---
for prior in none ppg; do
    for d in pixel pyramid; do
        for s in medoids random; do
            submit "${MODEL}_${d}_${prior}_${s}" "$d" physics "$prior" "$s" 0 0
            n=$((n+1))
        done
    done
done

# --- 16 physics slurpp (freeze_T x freeze_B) ---
for d in pixel pyramid; do
    for s in medoids random; do
        for ft in 0 1; do
            for fb in 0 1; do
                submit "${MODEL}_${d}_slurpp_${s}_ft${ft}_fb${fb}" "$d" physics slurpp "$s" "$ft" "$fb"
                n=$((n+1))
            done
        done
    done
done

echo "-------------------------------------------"
echo "$n jobs $([ "$DRYRUN" = "1" ] && echo 'preview (DRYRUN)' || echo 'submitted')"
echo "logs   : $LOG_DIR"
echo "wandb  : $WANDB_DIR"
echo "results: $EXP_ROOT/results -> logged_files/parameterization_exp/"
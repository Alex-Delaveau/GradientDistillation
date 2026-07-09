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

# --- absolute output root (single source of truth, propagated to the job) ---
OUTPUT_ROOT="$WORK/projects/GradientDistillation/output/parameterization_exp"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT/wandb" "$OUTPUT_ROOT/results"

DRYRUN=${DRYRUN:-0}

SEEDS=${SEEDS:-"3407"}

ONLY=${ONLY:-}

# --- resources (h100) ---
ACCOUNT="rbw@h100"
CONSTRAINT="h100"
QOS="qos_gpu_h100-t3"
TIME_DEFAULT="10:00:00"
TIME_SLURPP="20:00:00"

# maps 0/1 -> false/true for the CLI (pydantic-style bool parsing)
bool() { [ "$1" = "1" ] && echo "true" || echo "false"; }

submit() {
    local run_name="$1" distill="$2" formation="$3" prior="$4" sample="$5" ft="$6" fb="$7"
    local time="$TIME_DEFAULT"
    [ "$prior" = "slurpp" ] && time="$TIME_SLURPP"

    if [ -n "$ONLY" ] && [[ "$run_name" != *"$ONLY"* ]]; then
        return
    fi

    # one job per seed, each with its own run_name / logs / wandb dir
    for seed in $SEEDS; do
        local sname="${run_name}_s${seed}"
        local args=(
            --job-name="pexp_${sname}"
            --output="$LOG_DIR/%j_${sname}.out"
            --error="$LOG_DIR/%j_${sname}.err"
            --account="$ACCOUNT"
            --constraint="$CONSTRAINT"
            --qos="$QOS"
            --time="$time"
            --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",RUN_NAME="$sname",SEED="$seed",DISTILL_MODE="$distill",FORMATION_MODE="$formation",PRIOR_INIT="$prior",SAMPLE_INIT="$sample",FREEZE_T="$(bool "$ft")",FREEZE_B="$(bool "$fb")"
            "$SLURM_SCRIPT"
        )

        if [ "$DRYRUN" = "1" ]; then
            echo "sbatch ${args[*]}"
        else
            sbatch "${args[@]}"
        fi
        njobs=$((njobs+1))
    done
}

MODEL=${MODEL:-dinov2_vitb}
n=0
njobs=0

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
echo "$njobs jobs $([ "$DRYRUN" = "1" ] && echo 'preview (DRYRUN)' || echo 'submitted')"
[ -n "$ONLY" ] && echo "filter ONLY='$ONLY'"
echo "output root: $OUTPUT_ROOT"
echo "  logs   : $LOG_DIR"
echo "  wandb  : $OUTPUT_ROOT/wandb"
echo "  results: $OUTPUT_ROOT/results/aqua20/dinov2_vitb/<run_name>/"
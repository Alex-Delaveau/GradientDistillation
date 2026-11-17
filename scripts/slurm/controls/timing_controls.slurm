#!/bin/bash
# Production-cost measurement (reviewer request). TIMING ONLY: these runs are never
# used for any reported performance (--eval_it=0 changes the RNG trajectory, see the
# Evaluator.reset() seeding) and they live in a separate root: output/timing.
#
# One seed per condition. Each condition calls the UNMODIFIED production script:
#   baseline                       -> scripts/slurm/pixels/distill_baseline.slurm (EVAL_IT=0)
#   none | ppg | slurpp            -> scripts/slurm/pixels/distill_pixels.slurm   (--eval_it=0)
#   none_frozen|affine|slurpp_t1|none_t3 -> scripts/slurm/controls/distill_controls.slurm (--eval_it=0)
# Every stdout line is prefixed with a wall-clock epoch timestamp (no Python change);
# scripts/slurm/controls/parse_timing.py turns the .out files into a table
# (init phases: data/backbone, medoids, prior load, prior inference; distillation: s, it/s).
#
#   cd $WORK/projects/GradientDistillation
#   for c in baseline none ppg slurpp none_frozen affine slurpp_t1 none_t3; do
#     sbatch --job-name=ctl_time_$c --export=ALL,COND=$c scripts/slurm/controls/timing_controls.slurm
#   done
#SBATCH --account=rbw@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail

COND=${COND:?COND must be set (baseline|none|ppg|slurpp|none_frozen|affine|slurpp_t1|none_t3)}
SEED=${SEED:-3407}
PROJECT=$WORK/projects/GradientDistillation
export OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT/output/timing}
export DATASET=aqua20 DATA_ROOT=$WORK/datasets/aqua20 MODEL=dinov2_vitb IPC=1 SEED
# "_timing" in the run name: distinct wandb ids, never mixed up with paper/control runs
export RUN_NAME=${MODEL}_${DATASET}_${COND}_timing_ipc${IPC}
export PYTHONUNBUFFERED=1          # timestamps must reflect when lines are printed

RUN_DIR=$OUTPUT_ROOT/results/$DATASET/$MODEL/${RUN_NAME}_s${SEED}
if [ -e "$RUN_DIR" ] && [ "${FORCE:-0}" != "1" ]; then
    # a resumed run would give a meaningless duration
    echo "[timing] $RUN_DIR exists -> refusing (rm -rf it or FORCE=1)." >&2; exit 3
fi
mkdir -p "$OUTPUT_ROOT/logs"

case "$COND" in
  baseline)
    SCRIPT=$PROJECT/scripts/slurm/pixels/distill_baseline.slurm; export EVAL_IT=0; ARGS=() ;;
  none|ppg|slurpp)
    SCRIPT=$PROJECT/scripts/slurm/pixels/distill_pixels.slurm; export PRIOR=$COND; ARGS=(--eval_it=0) ;;
  none_frozen|affine|slurpp_t1|none_t3)
    SCRIPT=$PROJECT/scripts/slurm/controls/distill_controls.slurm; export VARIANT=$COND; ARGS=(--eval_it=0) ;;
  *) echo "unknown COND=$COND" >&2; exit 2 ;;
esac

stamp() { while IFS= read -r line; do printf '%s %s\n' "$EPOCHREALTIME" "$line"; done; }

echo "$EPOCHREALTIME TIMING_BEGIN cond=$COND seed=$SEED script=$SCRIPT job=${SLURM_JOB_ID:-} node=$(hostname)"
bash "$SCRIPT" "${ARGS[@]}" | stamp
echo "$EPOCHREALTIME TIMING_END cond=$COND"

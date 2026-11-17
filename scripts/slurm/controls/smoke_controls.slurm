#!/bin/bash
# Smoke test of the 4 control variants: same script (distill_controls.slurm), same
# flags, but --iterations=50 and a single seed, in a SEPARATE output root.
# Checks that every code path runs (medoids, none/affine/slurpp priors, frozen T/B,
# t_channels 1/3, eval at step 0, save_data), then that the DONE marker short-circuits.
#
#   sbatch scripts/slurm/controls/smoke_controls.slurm
#   VARIANTS="affine" sbatch --export=ALL scripts/slurm/controls/smoke_controls.slurm
#SBATCH --job-name=ctl_smoke
#SBATCH --account=rbw@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail

PROJECT=$WORK/projects/GradientDistillation
SCRIPT=$PROJECT/scripts/slurm/controls/distill_controls.slurm
export OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT/output/controls_smoke}
VARIANTS=${VARIANTS:-"none_frozen affine slurpp_t1 none_t3"}
export SEED=${SEED:-3407}
export IPC=1

status=0
for v in $VARIANTS; do
    echo "================ smoke: $v ================ $(date '+%F %T')"
    if VARIANT=$v bash "$SCRIPT" --iterations=50; then
        d="$OUTPUT_ROOT/results/aqua20/dinov2_vitb/dinov2_vitb_aqua20_${v}_ipc1_s${SEED}"
        if [ -f "$d/data.pth" ] && [ -f "$d/DONE" ]; then
            echo "[smoke] $v OK ($d)"
            # second call must exit immediately on DONE
            VARIANT=$v bash "$SCRIPT" --iterations=50 | grep -q "already finished" \
                && echo "[smoke] $v DONE short-circuit OK" \
                || { echo "[smoke] $v DONE short-circuit FAILED"; status=1; }
        else
            echo "[smoke] $v FAILED: data.pth/DONE missing in $d"; status=1
        fi
    else
        echo "[smoke] $v FAILED (exit code)"; status=1
    fi
done
echo "================ smoke done, status=$status $(date '+%F %T')"
exit $status

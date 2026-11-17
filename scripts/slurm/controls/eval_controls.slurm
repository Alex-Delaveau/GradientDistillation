#!/bin/bash
# Linear-probe evaluation of the control runs with the protocol of Tables 2-3
# (exp/eval_controls.py = script version of exp/benchmark_multipl_cross.ipynb).
#
#   sbatch scripts/slurm/controls/eval_controls.slurm
#   INCLUDE_PAPER=1 sbatch --export=ALL scripts/slurm/controls/eval_controls.slurm
#       -> also re-evaluates the 15 paper IPC-1 runs (output/pixels) in the same job,
#          to check the numbers reproduce the paper's eval_raw.csv.
#   RUNS_DIR=/other/results/aqua20/dinov2_vitb sbatch --export=ALL ...   # other input
#
# Backbone weights must already be in the notebook's caches (no internet on compute
# nodes): $WORK/.cache/torch/hub (DINOv2), $WORK/.cache/clip (CLIP), $WORK/.cache/huggingface (MoCoV3).
# The script loads the 3 backbones once at start (precheck) and fails fast otherwise.
#SBATCH --job-name=ctl_eval
#SBATCH --account=rbw@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail

ARCH=${ARCH:-h100}
module purge
module load arch/$ARCH
module load pytorch-gpu/py3/2.8.0

export PYTHONUSERBASE=$WORK/python_user/$ARCH
export PATH=$PYTHONUSERBASE/bin:$PATH
export XFORMERS_DISABLED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# HOME / TORCH_HOME / HF_HOME are set inside the script, exactly as in the notebook.

PROJECT=$WORK/projects/GradientDistillation
cd $PROJECT/exp

RUNS_DIR=${RUNS_DIR:-$PROJECT/output/controls/results/aqua20/dinov2_vitb}
RUNS_RX=${RUNS_RX:-'^dinov2_vitb_aqua20_(?:none_frozen|affine|slurpp_t1|none_t3)_ipc\d+_s\d+$'}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/aqua20/data/aqua20}
OUT=${OUT:-$PROJECT/output/controls/eval_raw_controls.csv}
INCLUDE_PAPER=${INCLUDE_PAPER:-0}

SRC=(--source "$RUNS_DIR" "$RUNS_RX")
if [ "$INCLUDE_PAPER" = "1" ]; then
    SRC+=(--source "$PROJECT/output/pixels/results/aqua20/dinov2_vitb"
          '^dinov2_vitb_aqua20_(?:none|slurpp|ppg)_ipc1_s\d+$')
fi

echo "[eval] runs=$RUNS_DIR rx=$RUNS_RX include_paper=$INCLUDE_PAPER"
echo "[eval] data_root=$DATA_ROOT out=$OUT gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[eval] start $(date '+%F %T')"

srun python eval_controls.py \
    --work "$WORK" \
    --project-root "$PROJECT" \
    --data-root "$DATA_ROOT" \
    "${SRC[@]}" \
    --out "$OUT" \
    "$@"

echo "[eval] end $(date '+%F %T') -> $OUT"

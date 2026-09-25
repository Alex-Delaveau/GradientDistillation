#!/bin/bash
# Generic SLURM script for the pixel-space distillation experiment.
# Variant axis: PRIOR (none | slurpp | ppg) x SEED.
#
# Resource flags (account / constraint / qos / time / cpus-per-task) are supplied
# by launch_pixels_fishnet20_exp.sh so the same script runs on h100, a100 or v100.
#
# Usage (via the launcher, recommended):
#   ARCH=a100 ./launch_pixels_fishnet20_exp.sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
set -e

# ---- architecture (h100 | a100 | v100) ----
ARCH=${ARCH:-h100}

module purge
module load arch/$ARCH
module load pytorch-gpu/py3/2.8.0

export TORCH_HOME=$WORK/torch_cache
export HF_HOME=$WORK/hf_cache
export TIMM_CACHE_DIR=$WORK/timm_cache
export XDG_CACHE_HOME=$WORK/.cache
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export XFORMERS_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# compiled extensions are NOT portable across architectures -> one prefix per arch
export PYTHONUSERBASE=$WORK/python_user/$ARCH
export PATH=$PYTHONUSERBASE/bin:$PATH

if [ ! -d "$PYTHONUSERBASE" ]; then
    echo "[pixels] ERROR: $PYTHONUSERBASE missing." >&2
    echo "  module purge && module load arch/$ARCH pytorch-gpu/py3/2.8.0" >&2
    echo "  PYTHONUSERBASE=$PYTHONUSERBASE pip install --user --no-cache-dir -r requirements.txt" >&2
    exit 1
fi

cd $WORK/projects/GradientDistillation

PRIOR=${PRIOR:?PRIOR must be set (none|slurpp|ppg)}
MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-fishnet20}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets}
IPC=${IPC:-1}
AUGS=${AUGS:-10}
SEED=${SEED:-3407}

# ---- absolute output root (single source of truth, from launcher via --export) ----
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORK/projects/GradientDistillation/output/pixels_fishnet20}
# the seed is appended downstream by the code -> no _s${SEED} here
RUN_NAME=${RUN_NAME:-${MODEL}_${DATASET}_${PRIOR}_ipc${IPC}}
RUN_DIR="$OUTPUT_ROOT/results/$DATASET/$MODEL/${RUN_NAME}_s${SEED}"
if [ -f "$RUN_DIR/DONE" ]; then
    echo "[pixels] $RUN_DIR déjà terminé, segment ignoré."
    scancel --me --name="$SLURM_JOB_NAME" --state=PENDING || true
    exit 0
fi

# dataloader workers follow the CPU allocation (24 on h100, 8 on a100, 10 on v100)
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}

export WANDB_DIR=$OUTPUT_ROOT/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-pixels_fishnet20}
export WANDB_NAME=${RUN_NAME}_s${SEED}
mkdir -p "$WANDB_DIR" "$OUTPUT_ROOT/results"

# ---- prior-specific flags ----
EXTRA=()
case "$PRIOR" in
  none)
    EXTRA+=(--t_channels=1)
    ;;
  slurpp)
    export BASE_CKPT_DIR=${BASE_CKPT_DIR:-$WORK/models/SLURPP}
    # slurpp_lib exposes the "slurpp" and "utils" packages
    export PYTHONPATH=$WORK/projects/GradientDistillation/slurpp_lib:${PYTHONPATH:-}
    SLURPP_ROOT=${SLURPP_ROOT:-slurpp_lib}
    SLURPP_CKPT=${SLURPP_CKPT:-$WORK/models/SLURPP/slurpp/checkpoint/diffusion}
    EXTRA+=(--t_channels=3
            --slurpp_root=$SLURPP_ROOT
            --slurpp_checkpoint_path=$SLURPP_CKPT)
    ;;
  ppg)
    PPG_CKPT=${PPG_CKPT:-$WORK/models/PPG/I40000_E500_ppg.pth}
    EXTRA+=(--t_channels=1
            --ppg_checkpoint_path=$PPG_CKPT
            --ppg_input_channels=3)
    ;;
  *)
    echo "unknown PRIOR='$PRIOR'" >&2; exit 2
    ;;
esac

echo "[pixels] RUN_NAME=$RUN_NAME"
echo "  prior=$PRIOR ipc=$IPC seed=$SEED model=$MODEL"
echo "  arch=$ARCH cpus=$NUM_WORKERS gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

srun python -m distillation.distill \
    --model=$MODEL \
    --dataset=$DATASET \
    --data_root=$DATA_ROOT \
    --run_name=$RUN_NAME \
    --job_tag=$OUTPUT_ROOT/results \
    --syn_res=224 \
    --real_res=224 \
    --crop_res=224 \
    --train_crop_mode=random \
    --ipc=$IPC \
    --augs_per_batch=$AUGS \
    --num_workers=$NUM_WORKERS \
    --formation_mode=physics \
    --distill_mode=pyramid \
    --prior_init=$PRIOR \
    --sample_init=medoids \
    "${EXTRA[@]}" \
    --b_spatial=False \
    --freeze_T=False \
    --freeze_B=False \
    --clamp_I=True \
    --pyramid_snapshot_it=2500 \
    --image_log_it=2500 \
    --checkpoint_it=100 \
    --eval_it=100 \
    --eval_num_eval=1 \
    --eval_epochs=100 \
    --iterations=5000 \
    --eval_metrics=f1 \
    --seed=$SEED \
    "$@"

mkdir -p "$RUN_DIR" && touch "$RUN_DIR/DONE"
scancel --me --name="$SLURM_JOB_NAME" --state=PENDING || true
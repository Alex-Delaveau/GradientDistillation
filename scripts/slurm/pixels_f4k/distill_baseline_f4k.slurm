#!/bin/bash
# Plain LGM baseline: pixel-space pyramid, no physical reparameterization.
#
# Resource flags (account / constraint / qos / time / cpus-per-task) are supplied
# by launch_baseline_f4k.sh so the same script runs on h100, a100 or v100.
#
# Usage (via the launcher, recommended):
#   ARCH=a100 ./launch_baseline_f4k.sh
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
    echo "[baseline] ERROR: $PYTHONUSERBASE missing." >&2
    echo "  module purge && module load arch/$ARCH pytorch-gpu/py3/2.8.0" >&2
    echo "  PYTHONUSERBASE=$PYTHONUSERBASE pip install --user --no-cache-dir -r requirements.txt" >&2
    exit 1
fi

cd $WORK/projects/GradientDistillation

MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-f4k}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/F4K}
IPC=${IPC:-1}
AUGS=${AUGS:-10}
SEED=${SEED:-3407}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORK/projects/GradientDistillation/output/baseline_f4k}
RUN_NAME=${RUN_NAME:-${MODEL}_${DATASET}_baseline_ipc${IPC}}

# dataloader workers follow the CPU allocation (24 on h100, 8 on a100, 10 on v100)
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}

export WANDB_DIR=$OUTPUT_ROOT/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-baseline_f4k}
export WANDB_NAME=${RUN_NAME}_s${SEED}
mkdir -p "$WANDB_DIR" "$OUTPUT_ROOT/results"

echo "[baseline] RUN_NAME=$RUN_NAME ipc=$IPC seed=$SEED"
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
    --formation_mode=identity \
    --distill_mode=pyramid \
    --sample_init=medoids \
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
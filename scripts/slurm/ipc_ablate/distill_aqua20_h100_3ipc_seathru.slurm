#!/bin/bash
#SBATCH --job-name=distill_aqua20_h100
#SBATCH --account=rbw@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=logs/ipc_ablation/%x_%j.out
#SBATCH --error=logs/ipc_ablation/%x_%j.err
set -e

module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0

export TORCH_HOME=$WORK/torch_cache
export HF_HOME=$WORK/hf_cache
export TIMM_CACHE_DIR=$WORK/timm_cache
export XDG_CACHE_HOME=$WORK/.cache

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XFORMERS_DISABLED=1

# Deps perso via la méthode IDRIS (plus de venv)
export PYTHONUSERBASE=$WORK/python_user/h100
export PATH=$PYTHONUSERBASE/bin:$PATH

cd $WORK/projects/GradientDistillation
mkdir -p logs/ipc_ablation

MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-aqua20}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/aqua20}
RUN_NAME=${RUN_NAME:-${MODEL}_distill_224_ipc3_augs5_seathru_h100_80g}

srun python -m distillation.distill \
    --model=$MODEL \
    --dataset=$DATASET \
    --data_root=$DATA_ROOT \
    --run_name=$RUN_NAME \
    --augs_per_batch=5 \
    --syn_res=224 \
    --real_res=224 \
    --crop_res=224 \
    --train_crop_mode=random \
    --ipc=3 \
    --distill_mode=seathru_pyramid \
    --num_workers=24 \
    "$@"
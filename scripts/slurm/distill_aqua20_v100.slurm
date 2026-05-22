#!/bin/bash
#SBATCH --job-name=distill_aqua20_v100
#SBATCH --account=rbw@v100
#SBATCH --constraint=v100-32g
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -e

module purge
module load pytorch-gpu/py3/2.8.0

export TORCH_HOME=$WORK/torch_cache
export HF_HOME=$WORK/hf_cache
export TIMM_CACHE_DIR=$WORK/timm_cache
export XDG_CACHE_HOME=$WORK/.cache

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XFORMERS_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source $WORK/projects/GradientDistillation/.venv/bin/activate
cd $WORK/projects/GradientDistillation
mkdir -p logs

MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-aqua20}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/aqua20}
RUN_NAME=${RUN_NAME:-${MODEL}_distill_252_ipc1_augs10_v100_32g}

srun python -m distillation.distill \
    --model=$MODEL \
    --dataset=$DATASET \
    --data_root=$DATA_ROOT \
    --run_name=$RUN_NAME \
    --augs_per_batch=10 \
    --syn_res=252 \
    --real_res=252 \
    --crop_res=252 \
    --train_crop_mode=random \
    --ipc=1 \
    --distill_mode=pyramid \
    --num_workers=10 \
    "$@"
#!/bin/bash
#SBATCH --job-name=distill_aqua20_v100_dev
#SBATCH --account=ucw75ke@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -e

# Modules
module purge
module load pytorch-gpu/py3/2.8.0

# Caches (pré-remplis depuis le frontal)
export TORCH_HOME=$WORK/torch_cache
export HF_HOME=$WORK/hf_cache
export TIMM_CACHE_DIR=$WORK/timm_cache
export XDG_CACHE_HOME=$WORK/.cache

# Offline modes
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Venv
source $WORK/projects/GradientDistillation/.venv/bin/activate

cd $WORK/projects/GradientDistillation
mkdir -p logs

# Paramètres (override possible via --export)
MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-aqua20}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/aqua20}
RUN_NAME=${RUN_NAME:-${MODEL}_distill_196_ipc1_augs6_seathru_v100}

srun python -m distillation.distill \
    --model=$MODEL \
    --dataset=$DATASET \
    --data_root=$DATA_ROOT \
    --run_name=$RUN_NAME \
    --augs_per_batch=6 \
    --syn_res=196 \
    --real_res=196 \
    --crop_res=196 \
    --train_crop_mode=random \
    --ipc=1 \
    --distill_mode=seathru_pyramid \
    --num_workers=10 \
    "$@"
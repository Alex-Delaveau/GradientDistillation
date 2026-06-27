#!/bin/bash
#SBATCH --job-name=distill_aqua20_ppg_h100
#SBATCH --account=rbw@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=logs/distill/%x_%j.out
#SBATCH --error=logs/distill/%x_%j.err
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

export PYTHONUSERBASE=$WORK/python_user/h100
export PATH=$PYTHONUSERBASE/bin:$PATH

cd $WORK/projects/GradientDistillation
mkdir -p logs/distill

MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-aqua20}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/aqua20}
IPC=${IPC:-1}
RUN_NAME=${RUN_NAME:-${MODEL}_${DATASET}_ppg_ipc${IPC}_h100}
PPG_CKPT=${PPG_CKPT:-$WORK/models/PPG/I40000_E500_ppg.pth}

srun python -m distillation.distill \
    --model=$MODEL \
    --dataset=$DATASET \
    --data_root=$DATA_ROOT \
    --run_name=$RUN_NAME \
    --augs_per_batch=10 \
    --syn_res=224 \
    --real_res=224 \
    --crop_res=224 \
    --train_crop_mode=random \
    --ipc=$IPC \
    --num_workers=24 \
    --distill_mode=physics_formation \
    --prior_init=ppg \
    --ppg_checkpoint_path=$PPG_CKPT \
    --ppg_input_channels=3 \
    --sample_init=medoids \
    --t_channels=1 \
    --b_spatial=False \
    --freeze_T=False \
    --freeze_B=False \
    --clamp_I=True \
    --pyramid_snapshot_it=500 \
    --image_log_it=500 \
    --checkpoint_it=100 \
    --eval_it=250 \
    --eval_num_eval=1 \
    --eval_epochs=100 \
    --iterations=5000 \
    --eval_metrics=f1 \
    "$@"
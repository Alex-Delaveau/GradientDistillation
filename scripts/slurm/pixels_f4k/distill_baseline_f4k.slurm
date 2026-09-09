#!/bin/bash
#SBATCH --account=rbw@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
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
export HF_HUB_OFFLINE=1
export XFORMERS_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUSERBASE=$WORK/python_user/h100
export PATH=$PYTHONUSERBASE/bin:$PATH

cd $WORK/projects/GradientDistillation

MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-f4k}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/F4K}
IPC=${IPC:-1}
AUGS=${AUGS:-10}
SEED=${SEED:-3407}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORK/projects/GradientDistillation/output/baseline_f4k}
RUN_NAME=${RUN_NAME:-${MODEL}_${DATASET}_baseline_ipc${IPC}}

export WANDB_DIR=$OUTPUT_ROOT/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-baseline_f4k}
export WANDB_NAME=${RUN_NAME}_s${SEED}
mkdir -p "$WANDB_DIR" "$OUTPUT_ROOT/results"

echo "[baseline] RUN_NAME=$RUN_NAME ipc=$IPC seed=$SEED"

srun python -m distillation.distill \
    --model=$MODEL \
    --dataset=$DATASET \
    --data_root=$DATA_ROOT \
    --run_name=$RUN_NAME \
    --job_tag=$OUTPUT_ROOT/results \
    --augs_per_batch=10 \
    --syn_res=224 \
    --real_res=224 \
    --crop_res=224 \
    --train_crop_mode=random \
    --ipc=$IPC \
    --augs_per_batch=$AUGS \
    --num_workers=24 \
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
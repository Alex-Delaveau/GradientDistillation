#!/bin/bash
# Generic SLURM script for the latent-space distillation experiment (GLaD-SLURPP).
# Variant axis: LATENT_MODE (predlatent for now) x LR (to retune from scratch).
#
# Usage (direct, defaults):
#   sbatch --export=ALL,RUN_NAME=dinov2_vitb_latent_predlatent_medoids_adam_lr1e-3_s3407 run_latent_exp.slurm
#
# Usage (LR mini-sweep, 3 jobs):
#   for lr in 1e-3 3e-3 1e-2; do
#     sbatch --export=ALL,LR="$lr",RUN_NAME="dinov2_vitb_latent_predlatent_medoids_adam_lr${lr}_s3407" \
#            run_latent_exp.slurm
#   done
#SBATCH --account=rbw@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=output/latent_exp/logs/%x_%j.out
#SBATCH --error=output/latent_exp/logs/%x_%j.err
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

export BASE_CKPT_DIR=${BASE_CKPT_DIR:-$WORK/models/SLURPP}
export PYTHONUSERBASE=$WORK/python_user/h100
export PATH=$PYTHONUSERBASE/bin:$PATH

# slurpp_lib expose les packages "slurpp" et "utils" (imports du dataset latent)
export PYTHONPATH=$WORK/projects/GradientDistillation/slurpp_lib:${PYTHONPATH:-}

cd $WORK/projects/GradientDistillation

# ---- absolute output root (single source of truth, from launcher via --export) ----
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORK/projects/GradientDistillation/output/latent_exp}
export WANDB_DIR=$OUTPUT_ROOT/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-latent_exp}
export WANDB_NAME=$RUN_NAME
mkdir -p "$WANDB_DIR" "$OUTPUT_ROOT/results"

MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-aqua20}
SEED=${SEED:-3407}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/aqua20}

# ---- axes of this experiment ----
# LATENT_MODE : predlatent (implemented) | independent | z_u (future)
LATENT_MODE=${LATENT_MODE:-predlatent}
# LATENT_CHUNK : images per decode chunk in the two-pass backward.
# 2 is the safe default from the feasibility OOM findings (job 1932676);
# raise it if peak_gb leaves headroom.
LATENT_CHUNK=${LATENT_CHUNK:-2}
# LATENT_RES : SLURPP working resolution (512 -> latent 64x64; 256 loses 8dB on J)
LATENT_RES=${LATENT_RES:-512}
# LR : latent learning rate -- pixel-space LRs are meaningless here, retune.
LR=${LR:-1e-3}
OPTIM=${OPTIM:-adam}

# ---- prior checkpoints (latent mode requires slurpp) ----
SLURPP_ROOT=${SLURPP_ROOT:-slurpp_lib}
SLURPP_CKPT=${SLURPP_CKPT:-$WORK/models/SLURPP/slurpp/checkpoint/diffusion}

echo "[latent_exp] RUN_NAME=$RUN_NAME"
echo "  latent_mode=$LATENT_MODE chunk=$LATENT_CHUNK res=$LATENT_RES lr=$LR optim=$OPTIM seed=$SEED"

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
    --ipc=1 \
    --num_workers=24 \
    --distill_opt=$OPTIM \
    --lr=$LR \
    --formation_mode=latent \
    --latent_mode=$LATENT_MODE \
    --latent_chunk=$LATENT_CHUNK \
    --latent_res=$LATENT_RES \
    --prior_init=slurpp \
    --sample_init=medoids \
    --slurpp_root=$SLURPP_ROOT \
    --slurpp_checkpoint_path=$SLURPP_CKPT \
    --clamp_I=True \
    --image_log_it=100 \
    --checkpoint_it=100 \
    --eval_it=100 \
    --eval_num_eval=1 \
    --eval_epochs=100 \
    --iterations=2000 \
    --pyramid_snapshot_it=100 \
    --eval_metrics=f1 \
    --seed=$SEED \
    "$@"
#!/bin/bash
# Generic SLURM script for the latent-space distillation experiment (GLaD-SLURPP).
# Variant axis: LATENT_MODE (predlatent | decoder_only) x LR (to retune from scratch).
#
# Resource flags (account / constraint / qos / time / cpus-per-task) are supplied
# by launch_latent_exp.sh so the same script runs on h100, a100 or v100.
#
# Usage (via the launcher, recommended):
#   ARCH=a100 ./launch_latent_exp.sh
#
# Usage (direct submission -- you must pass the resource flags yourself):
#   sbatch --account=rbw@a100 --constraint=a100 --qos=qos_gpu_a100-t3 \
#          --time=10:00:00 --cpus-per-task=8 \
#          --export=ALL,ARCH=a100,RUN_NAME=dinov2_vitb_latent_predlatent_medoids_adam_lr1e-3_s3407 \
#          latent_f4k_exp.slurm
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

export BASE_CKPT_DIR=${BASE_CKPT_DIR:-$WORK/models/SLURPP}

# compiled extensions are NOT portable across architectures -> one prefix per arch
export PYTHONUSERBASE=$WORK/python_user/$ARCH
export PATH=$PYTHONUSERBASE/bin:$PATH

if [ ! -d "$PYTHONUSERBASE" ]; then
    echo "[latent_exp] ERROR: $PYTHONUSERBASE missing." >&2
    echo "  module purge && module load arch/$ARCH pytorch-gpu/py3/2.8.0" >&2
    echo "  PYTHONUSERBASE=$PYTHONUSERBASE pip install --user --no-cache-dir -r requirements.txt" >&2
    exit 1
fi

# slurpp_lib expose les packages "slurpp" et "utils" (imports du dataset latent)
export PYTHONPATH=$WORK/projects/GradientDistillation/slurpp_lib:${PYTHONPATH:-}

cd $WORK/projects/GradientDistillation

# ---- absolute output root (single source of truth, from launcher via --export) ----
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORK/projects/GradientDistillation/output/latent_f4k_exp}
export WANDB_DIR=$OUTPUT_ROOT/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-latent_f4k_exp}
export WANDB_NAME=$RUN_NAME
mkdir -p "$WANDB_DIR" "$OUTPUT_ROOT/results"

MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-f4k}
SEED=${SEED:-3407}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/F4K}

# dataloader workers follow the CPU allocation (24 on h100, 8 on a100, 10 on v100)
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}

# ---- axes of this experiment ----
# LATENT_MODE : predlatent (implemented) | decoder_only | independent | z_u (future)
LATENT_MODE=${LATENT_MODE:-predlatent}
# LATENT_CHUNK : images per decode chunk in the two-pass backward.
# 2 is the safe default from the feasibility OOM findings (job 1932676);
# raise it if peak_gb leaves headroom. Lower it on <80GB cards.
LATENT_CHUNK=${LATENT_CHUNK:-2}
# LATENT_RES : SLURPP working resolution (512 -> latent 64x64; 256 loses 8dB on J)
LATENT_RES=${LATENT_RES:-512}
# LR : latent learning rate -- pixel-space LRs are meaningless here, retune.
LR=${LR:-1e-3}
OPTIM=${OPTIM:-adam}

RUN_NAME=${RUN_NAME:-${MODEL}_latent_${DATASET}_${LATENT_MODE}_medoids_${OPTIM}_lr${LR}}

# ---- prior checkpoints (latent mode requires slurpp) ----
SLURPP_ROOT=${SLURPP_ROOT:-slurpp_lib}
SLURPP_CKPT=${SLURPP_CKPT:-$WORK/models/SLURPP/slurpp/checkpoint/diffusion}

echo "[latent_exp] RUN_NAME=$RUN_NAME"
echo "  arch=$ARCH cpus=$NUM_WORKERS gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
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
    --num_workers=$NUM_WORKERS \
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
#!/bin/bash
# Generic SLURM script for the parameterization experiment.
# Job-specific bits (job-name, output, error) are passed by the launcher via
# sbatch CLI flags. Config is passed via environment (see launch script).
# Aligned with the proven h100 slurpp/ppg scripts.
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

export BASE_CKPT_DIR=${BASE_CKPT_DIR:-$WORK/models/SLURPP}
export PYTHONUSERBASE=$WORK/python_user/h100
export PATH=$PYTHONUSERBASE/bin:$PATH

cd $WORK/projects/GradientDistillation

# ---- absolute output root (single source of truth, from launcher via --export) ----
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORK/projects/GradientDistillation/output/parameterization_exp}
export WANDB_DIR=$OUTPUT_ROOT/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-parameterization_exp}
export WANDB_NAME=$RUN_NAME
mkdir -p "$WANDB_DIR" "$OUTPUT_ROOT/results"

MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-aqua20}
SEED=${SEED:-3407}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/aqua20}

# ---- prior checkpoints (match the proven scripts) ----
SLURPP_ROOT=${SLURPP_ROOT:-slurpp_lib}
SLURPP_CKPT=${SLURPP_CKPT:-$WORK/models/SLURPP/slurpp/checkpoint/diffusion}
PPG_CKPT=${PPG_CKPT:-$WORK/models/PPG/I40000_E500_ppg.pth}

# ---- prior-specific args ----
# slurpp: t_channels=3 (keep per-channel transmission), b_spatial=True (keep
# SLURPP's spatial additive backscatter). ppg: t_channels=1 (single-channel T).
PRIOR_ARGS=""
if [ "$PRIOR_INIT" = "slurpp" ]; then
    PRIOR_ARGS="--t_channels=3 --b_spatial=True --slurpp_root=$SLURPP_ROOT --slurpp_checkpoint_path=$SLURPP_CKPT"
elif [ "$PRIOR_INIT" = "ppg" ]; then
    PRIOR_ARGS="--ppg_checkpoint_path=$PPG_CKPT --ppg_input_channels=3 --t_channels=1"
fi

echo "[parameterization_exp] RUN_NAME=$RUN_NAME"
echo "  distill=$DISTILL_MODE formation=$FORMATION_MODE prior=$PRIOR_INIT sample=$SAMPLE_INIT freeze_T=$FREEZE_T freeze_B=$FREEZE_B"

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
    --distill_mode=$DISTILL_MODE \
    --formation_mode=$FORMATION_MODE \
    --prior_init=$PRIOR_INIT \
    --sample_init=$SAMPLE_INIT \
    --freeze_T=$FREEZE_T \
    --freeze_B=$FREEZE_B \
    --clamp_I=True \
    --pyramid_snapshot_it=500 \
    --image_log_it=500 \
    --checkpoint_it=100 \
    --eval_it=100 \
    --eval_num_eval=1 \
    --eval_epochs=100 \
    --iterations=5000 \
    --eval_metrics=f1 \
    --seed=$SEED \
    $PRIOR_ARGS \
    "$@"

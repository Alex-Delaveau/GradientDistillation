#!/bin/bash
# Generic SLURM job for the learning-rate sweep.
# Fixed parameterization; LR / LR_T / LR_B and RUN_NAME passed via env by the launcher.
# All outputs live under an ABSOLUTE path: $WORK/projects/GradientDistillation/output/<EXP_NAME>
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

# ---- absolute output root (passed by launcher) ----
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORK/projects/GradientDistillation/output/lr_exp}
export WANDB_DIR=$OUTPUT_ROOT/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-lr_exp}
export WANDB_NAME=$RUN_NAME
mkdir -p "$WANDB_DIR" "$OUTPUT_ROOT/results"

MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-aqua20}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/aqua20}

# ---- fixed parameterization (override via env if needed) ----
DISTILL_MODE=${DISTILL_MODE:-pyramid}
FORMATION_MODE=${FORMATION_MODE:-physics}
PRIOR_INIT=${PRIOR_INIT:-slurpp}
SAMPLE_INIT=${SAMPLE_INIT:-medoids}

# ---- prior checkpoints ----
SLURPP_ROOT=${SLURPP_ROOT:-slurpp_lib}
SLURPP_CKPT=${SLURPP_CKPT:-$WORK/models/SLURPP/slurpp/checkpoint/diffusion}
PPG_CKPT=${PPG_CKPT:-$WORK/models/PPG/I40000_E500_ppg.pth}

PRIOR_ARGS=""
if [ "$PRIOR_INIT" = "slurpp" ]; then
    PRIOR_ARGS="--t_channels=3 --b_spatial=True --slurpp_root=$SLURPP_ROOT --slurpp_checkpoint_path=$SLURPP_CKPT"
elif [ "$PRIOR_INIT" = "ppg" ]; then
    PRIOR_ARGS="--ppg_checkpoint_path=$PPG_CKPT --ppg_input_channels=3 --t_channels=1"
fi

echo "[lr_exp] RUN_NAME=$RUN_NAME  lr=$LR lr_T=$LR_T lr_B=$LR_B"
echo "  distill=$DISTILL_MODE formation=$FORMATION_MODE prior=$PRIOR_INIT sample=$SAMPLE_INIT"

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
    --freeze_T=False \
    --freeze_B=False \
    --lr=$LR \
    --lr_T=$LR_T \
    --lr_B=$LR_B \
    --clamp_I=True \
    --pyramid_snapshot_it=500 \
    --image_log_it=500 \
    --checkpoint_it=100 \
    --eval_it=100 \
    --eval_num_eval=1 \
    --eval_epochs=100 \
    --iterations=5000 \
    --eval_metrics=f1 \
    $PRIOR_ARGS \
    "$@"
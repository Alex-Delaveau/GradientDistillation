#!/bin/bash
# Generic SLURM script for the pixel-space distillation experiment.
# Variant axis: PRIOR (none | slurpp | ppg) x SEED.
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

PRIOR=${PRIOR:?PRIOR must be set (none|slurpp|ppg)}
MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-aqua20}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/aqua20}
IPC=${IPC:-1}
SEED=${SEED:-3407}

# ---- absolute output root (single source of truth, from launcher via --export) ----
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORK/projects/GradientDistillation/output/pixels}
RUN_NAME=${RUN_NAME:-${MODEL}_${DATASET}_${PRIOR}_ipc${IPC}_s${SEED}}

export WANDB_DIR=$OUTPUT_ROOT/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-pixels}
export WANDB_NAME=$RUN_NAME
mkdir -p "$WANDB_DIR" "$OUTPUT_ROOT/results"

# ---- prior-specific flags ----
EXTRA=()
case "$PRIOR" in
  none)
    EXTRA+=(--t_channels=1)
    ;;
  slurpp)
    export BASE_CKPT_DIR=${BASE_CKPT_DIR:-$WORK/models/SLURPP}
    # slurpp_lib exposes the "slurpp" and "utils" packages
    export PYTHONPATH=$WORK/projects/GradientDistillation/slurpp_lib:${PYTHONPATH:-}
    SLURPP_ROOT=${SLURPP_ROOT:-slurpp_lib}
    SLURPP_CKPT=${SLURPP_CKPT:-$WORK/models/SLURPP/slurpp/checkpoint/diffusion}
    EXTRA+=(--t_channels=3
            --slurpp_root=$SLURPP_ROOT
            --slurpp_checkpoint_path=$SLURPP_CKPT)
    ;;
  ppg)
    PPG_CKPT=${PPG_CKPT:-$WORK/models/PPG/I40000_E500_ppg.pth}
    EXTRA+=(--t_channels=1
            --ppg_checkpoint_path=$PPG_CKPT
            --ppg_input_channels=3)
    ;;
  *)
    echo "unknown PRIOR='$PRIOR'" >&2; exit 2
    ;;
esac

echo "[pixels] RUN_NAME=$RUN_NAME"
echo "  prior=$PRIOR ipc=$IPC seed=$SEED model=$MODEL"

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
    --num_workers=24 \
    --formation_mode=physics \
    --distill_mode=pyramid \
    --prior_init=$PRIOR \
    --sample_init=medoids \
    "${EXTRA[@]}" \
    --b_spatial=False \
    --freeze_T=False \
    --freeze_B=False \
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
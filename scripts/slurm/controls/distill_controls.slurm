#!/bin/bash
# Camera-ready controls (VISAPP 2027, PGDD-UW): separate the physical prior from the
# reparameterization (capacity / composition / dynamic range).
#
# Copy of scripts/slurm/pixels/distill_pixels.slurm: same #SBATCH, same environment,
# same distillation flags. Only the variant-specific flags differ (case "$VARIANT").
# Resume on ckpt.pth (done by LinearGM.load_checkpoint) + DONE marker, taken from
# scripts/slurm/pixels/distill_baseline.slurm -> a run killed at walltime can simply
# be resubmitted with the same command.
#
# Variant axis (VARIANT):
#   none_frozen : I = 0.5 J + 0.25 fixed, T/B not trained   (dynamic-range compression only)
#   affine      : I = J*G + O, G=0.5 O=0.25 trainable        (non-physical, same capacity/init as none)
#   slurpp_t1   : SLURPP init, T averaged to 1 channel        (T channel count under SLURPP)
#   none_t3     : none init, T with 3 channels               (T channel count without physics)
#
# Usage (via the launcher, recommended): ./launch_controls.sh
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

# ---- architecture (default h100 = architecture of the paper runs) ----
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
export PYTHONUSERBASE=$WORK/python_user/$ARCH
export PATH=$PYTHONUSERBASE/bin:$PATH

cd $WORK/projects/GradientDistillation

VARIANT=${VARIANT:?VARIANT must be set (none_frozen|affine|slurpp_t1|none_t3)}
MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-aqua20}
case "$DATASET" in
  aqua20)    DATA_ROOT_DEF="$WORK/datasets/aqua20" ;;
  f4k)       DATA_ROOT_DEF="$WORK/datasets/F4K" ;;
  fishnet20) DATA_ROOT_DEF="$WORK/datasets" ;;
  *)         DATA_ROOT_DEF="" ;;
esac
DATA_ROOT=${DATA_ROOT:-$DATA_ROOT_DEF}
DATA_ROOT=${DATA_ROOT:?no default DATA_ROOT for DATASET=$DATASET, set DATA_ROOT}
IPC=${IPC:-1}
AUGS=${AUGS:-10}
SEED=${SEED:-3407}
NUM_WORKERS=${NUM_WORKERS:-24}

# ---- absolute output root (single source of truth, from launcher via --export) ----
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORK/projects/GradientDistillation/output/controls}
# the seed is appended downstream by the code -> no _s${SEED} here
RUN_NAME=${RUN_NAME:-${MODEL}_${DATASET}_${VARIANT}_ipc${IPC}}
RUN_DIR="$OUTPUT_ROOT/results/$DATASET/$MODEL/${RUN_NAME}_s${SEED}"

if [ -f "$RUN_DIR/DONE" ]; then
    echo "[controls] $RUN_DIR already finished (DONE), nothing to do."
    scancel --me --name="$SLURM_JOB_NAME" --state=PENDING || true
    exit 0
fi
if [ -f "$RUN_DIR/ckpt.pth" ]; then
    echo "[controls] ckpt.pth found in $RUN_DIR -> the code resumes from it."
fi

export WANDB_DIR=$OUTPUT_ROOT/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-controls}
export WANDB_NAME=${RUN_NAME}_s${SEED}
mkdir -p "$WANDB_DIR" "$OUTPUT_ROOT/results"

# ---- variant-specific flags ----
FREEZE_T=False
FREEZE_B=False
EXTRA=()
case "$VARIANT" in
  none_frozen)
    PRIOR=none
    FREEZE_T=True
    FREEZE_B=True
    EXTRA+=(--t_channels=1)
    ;;
  affine)
    PRIOR=affine
    EXTRA+=(--t_channels=1)
    ;;
  none_t3)
    PRIOR=none
    EXTRA+=(--t_channels=3)
    ;;
  slurpp_t1)
    PRIOR=slurpp
    # identical to the slurpp branch of distill_pixels.slurm, except --t_channels=1
    export BASE_CKPT_DIR=${BASE_CKPT_DIR:-$WORK/models/SLURPP}
    export PYTHONPATH=$WORK/projects/GradientDistillation/slurpp_lib:${PYTHONPATH:-}
    SLURPP_ROOT=${SLURPP_ROOT:-slurpp_lib}
    SLURPP_CKPT=${SLURPP_CKPT:-$WORK/models/SLURPP/slurpp/checkpoint/diffusion}
    EXTRA+=(--t_channels=1
            --slurpp_root=$SLURPP_ROOT
            --slurpp_checkpoint_path=$SLURPP_CKPT)
    ;;
  *)
    echo "unknown VARIANT='$VARIANT' (none_frozen|affine|slurpp_t1|none_t3)" >&2; exit 2
    ;;
esac

echo "[controls] RUN_NAME=$RUN_NAME seed=$SEED -> $RUN_DIR"
echo "  variant=$VARIANT prior=$PRIOR freeze_T=$FREEZE_T freeze_B=$FREEZE_B ${EXTRA[*]}"
echo "  dataset=$DATASET data_root=$DATA_ROOT ipc=$IPC model=$MODEL"
echo "  arch=$ARCH workers=$NUM_WORKERS gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  extra args: $*"
echo "[controls] start $(date '+%F %T') epoch=$(date +%s)"

# flags below = distill_pixels.slurm, same order (incl. the duplicated
# --augs_per_batch, last one wins), freeze flags taken from the variant.
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
    --num_workers=$NUM_WORKERS \
    --formation_mode=physics \
    --distill_mode=pyramid \
    --prior_init=$PRIOR \
    --sample_init=medoids \
    "${EXTRA[@]}" \
    --b_spatial=False \
    --freeze_T=$FREEZE_T \
    --freeze_B=$FREEZE_B \
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

echo "[controls] end $(date '+%F %T') epoch=$(date +%s)"
mkdir -p "$RUN_DIR" && touch "$RUN_DIR/DONE"
scancel --me --name="$SLURM_JOB_NAME" --state=PENDING || true

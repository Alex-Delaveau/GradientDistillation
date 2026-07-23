#!/bin/bash
#SBATCH --job-name=distill_grid
#SBATCH --account=rbw@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/ipc_ablation/%x_%A_%a.out
#SBATCH --error=logs/ipc_ablation/%x_%A_%a.err
#
# Ne pas lancer directement : passer par scripts/launch_distill_grid.sh
# qui genere le manifeste et soumet le tableau avec la bonne taille.
#
# Le manifeste est un TSV : RUN_NAME \t FLAGS...
# Une ligne = une tache du tableau (indexee a partir de 1).

set -euo pipefail

MANIFEST=${MANIFEST:?MANIFEST non defini (utiliser le launcher)}
TASK_ID=${SLURM_ARRAY_TASK_ID:?script prevu pour un job array}

module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0

export TORCH_HOME=$WORK/torch_cache
export HF_HOME=$WORK/hf_cache
export TIMM_CACHE_DIR=$WORK/timm_cache
export XDG_CACHE_HOME=$WORK/.cache

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XFORMERS_DISABLED=1

export PYTHONUSERBASE=$WORK/python_user/h100
export PATH=$PYTHONUSERBASE/bin:$PATH

cd "$WORK/projects/GradientDistillation"
mkdir -p logs/ipc_ablation

LINE=$(sed -n "${TASK_ID}p" "$MANIFEST")
if [[ -z "$LINE" ]]; then
    echo "[erreur] ligne ${TASK_ID} absente de ${MANIFEST}" >&2
    exit 1
fi

RUN_NAME=$(cut -f1 <<< "$LINE")
FLAGS=$(cut -f2 <<< "$LINE")

MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-aqua20}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/aqua20}

echo "=================================================================="
echo "array task : ${TASK_ID}  (job ${SLURM_ARRAY_JOB_ID})"
echo "run_name   : ${RUN_NAME}"
echo "flags      : ${FLAGS}"
echo "manifeste  : ${MANIFEST}"
echo "=================================================================="

# shellcheck disable=SC2086
srun python -m distillation.distill \
    --model="$MODEL" \
    --dataset="$DATASET" \
    --data_root="$DATA_ROOT" \
    --run_name="$RUN_NAME" \
    --augs_per_batch=5 \
    --syn_res=224 \
    --real_res=224 \
    --crop_res=224 \
    --train_crop_mode=random \
    --num_workers=24 \
    $FLAGS \
    "$@"

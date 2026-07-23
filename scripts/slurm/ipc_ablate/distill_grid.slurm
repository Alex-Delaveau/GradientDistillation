#!/bin/bash
# Job array pour l'ablation IPC x variante x seed (espace pixel).
#
# Aligne sur latent_exp.slurm : memes flags d'evaluation, meme budget
# d'iterations, meme augs_per_batch, meme mecanisme de sortie (--job_tag).
# C'est ce qui rend la comparaison pixel <-> latent appariee.
#
# Ne pas lancer directement : passer par launch_distill_grid.sh, qui genere
# le manifeste TSV (RUN_NAME \t FLAGS) et soumet le tableau.
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
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export XFORMERS_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export BASE_CKPT_DIR=${BASE_CKPT_DIR:-$WORK/models/SLURPP}
export PYTHONUSERBASE=$WORK/python_user/h100
export PATH=$PYTHONUSERBASE/bin:$PATH

# necessaire pour les variantes physics_slurpp / physics_ppg, inoffensif sinon
export PYTHONPATH=$WORK/projects/GradientDistillation/slurpp_lib:${PYTHONPATH:-}

cd "$WORK/projects/GradientDistillation"

# ---- racine de sortie (source unique de verite, via --export du launcher) ----
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORK/projects/GradientDistillation/output/ipc_ablation}

# ---- lecture de la tache dans le manifeste ----
LINE=$(sed -n "${TASK_ID}p" "$MANIFEST")
if [[ -z "$LINE" ]]; then
    echo "[erreur] ligne ${TASK_ID} absente de ${MANIFEST}" >&2
    exit 1
fi
RUN_NAME=$(cut -f1 <<< "$LINE")
FLAGS=$(cut -f2 <<< "$LINE")

export WANDB_DIR=$OUTPUT_ROOT/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-ipc_ablation}
export WANDB_NAME=$RUN_NAME
mkdir -p "$WANDB_DIR" "$OUTPUT_ROOT/results"

MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-aqua20}
DATA_ROOT=${DATA_ROOT:-$WORK/datasets/aqua20}

SLURPP_ROOT=${SLURPP_ROOT:-slurpp_lib}
SLURPP_CKPT=${SLURPP_CKPT:-$WORK/models/SLURPP/slurpp/checkpoint/diffusion}

echo "=================================================================="
echo "array task  : ${TASK_ID}  (job ${SLURM_ARRAY_JOB_ID})"
echo "run_name    : ${RUN_NAME}"
echo "flags       : ${FLAGS}"
echo "output root : ${OUTPUT_ROOT}"
echo "manifeste   : ${MANIFEST}"
echo "=================================================================="

# shellcheck disable=SC2086
srun python -m distillation.distill \
    --model="$MODEL" \
    --dataset="$DATASET" \
    --data_root="$DATA_ROOT" \
    --run_name="$RUN_NAME" \
    --job_tag="$OUTPUT_ROOT/results" \
    --augs_per_batch=10 \
    --syn_res=224 \
    --real_res=224 \
    --crop_res=224 \
    --train_crop_mode=random \
    --num_workers=24 \
    --iterations=2000 \
    --clamp_I=True \
    --image_log_it=100 \
    --checkpoint_it=100 \
    --eval_it=100 \
    --eval_num_eval=1 \
    --eval_epochs=100 \
    --eval_metrics=f1 \
    --pyramid_snapshot_it=100 \
    --slurpp_root="$SLURPP_ROOT" \
    --slurpp_checkpoint_path="$SLURPP_CKPT" \
    $FLAGS \
    "$@"
#!/bin/bash
#SBATCH --job-name=latent_feas
#SBATCH --account=rbw@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=logs/latent_feas_%j.out
#SBATCH --error=logs/latent_feas_%j.err

# ============================================================
# Test de faisabilite : optimisation en espace latent SLURPP
# (T1 round-trip VAE, T2/T3 gradient flow, T4 benchmark, T5 resolution)
#
# Usage:
#   sbatch run_latent_feasibility.slurm
#
# Variables a adapter dans la section CONFIG ci-dessous avant de lancer.
# ============================================================

set -euo pipefail
mkdir -p logs

echo "== Job $SLURM_JOB_ID demarre sur $(hostname) a $(date) =="

# ---- Environnement IDRIS ----
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0

# ---- caches / offline (evite tout appel reseau depuis un noeud de calcul) ----
export TORCH_HOME=$WORK/torch_cache
export HF_HOME=$WORK/hf_cache
export XDG_CACHE_HOME=$WORK/.cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export XFORMERS_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- PYTHONUSERBASE (pattern H100 du projet) ----
export PYTHONUSERBASE=$WORK/python_user/h100
export PATH=$PYTHONUSERBASE/bin:$PATH

# slurpp_lib expose les packages "slurpp" et "utils" -- ajout explicite,
# independant de la version Python (pas de chemin site-packages en dur)
export PYTHONPATH=$WORK/projects/GradientDistillation/slurpp_lib:$PYTHONPATH


# ---- CONFIG : a adapter ----
PROJECT_DIR=$WORK/projects/GradientDistillation
SLURPP_ROOT=${SLURPP_ROOT:-slurpp_lib}
SLURPP_CKPT=${SLURPP_CKPT:-$WORK/models/SLURPP/slurpp/checkpoint/diffusion}
MEDOID_IMAGE=$WORK/datasets/aqua20/data/aqua20/train/fish/01816.png
N_IMAGES=20
OUTDIR=$PROJECT_DIR/output/latent_feasibility/$SLURM_JOB_ID

export BASE_CKPT_DIR=${BASE_CKPT_DIR:-$WORK/models/SLURPP}

mkdir -p "$OUTDIR"

cd "$PROJECT_DIR"

echo "== Config =="
echo "  PROJECT_DIR   = $PROJECT_DIR"
echo "  SLURPP_ROOT   = $SLURPP_ROOT"
echo "  SLURPP_CKPT   = $SLURPP_CKPT"
echo "  MEDOID_IMAGE  = $MEDOID_IMAGE"
echo "  N_IMAGES      = $N_IMAGES"
echo "  OUTDIR        = $OUTDIR"
echo "  BASE_CKPT_DIR = $BASE_CKPT_DIR"

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

echo "== Lancement du script de faisabilite =="
python exp/glad_slurpp.py \
    --image "$MEDOID_IMAGE" \
    --slurpp-root "$SLURPP_ROOT" \
    --checkpoint "$SLURPP_CKPT" \
    --n-images "$N_IMAGES" \
    --outdir "$OUTDIR"

echo "== Termine a $(date) =="
echo "Rapport : $OUTDIR/report.json"

#!/bin/bash
# Usage: ./run.sh <command> [options]
# Commands: distill, eval, neighbors, centroids, random, full

MODEL=${MODEL:-dinov2_vitb}
EVAL_MODEL=${EVAL_MODEL:-clip_vitb}
DATASET=${DATASET:-imagenet-birds}
JOB_TAG=${JOB_TAG:-distillation}
SEED=${SEED:-0}

case "$1" in
  distill)
    uv run -m distillation.distill --model=$MODEL --dataset=$DATASET --job_tag=$JOB_TAG "${@:2}"
    ;;
  eval)
    uv run -m distillation.eval --model=$MODEL --eval_model=$EVAL_MODEL --dataset=$DATASET --job_tag=$JOB_TAG "${@:2}"
    ;;
  neighbors)
    uv run -m baselines.neighbors --model=$MODEL --dataset=$DATASET --job_tag=$JOB_TAG "${@:2}"
    ;;
  centroids)
    uv run -m baselines.centroids --model=$MODEL --dataset=$DATASET "${@:2}"
    ;;
  random)
    uv run -m baselines.random_reals --model=$MODEL --dataset=$DATASET --random_seed=$SEED "${@:2}"
    ;;
  full)
    uv run -m baselines.full_dataset --model=$MODEL --dataset=$DATASET "${@:2}"
    ;;
  *)
    echo "Usage: ./run.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  distill     Distill a dataset"
    echo "  eval        Evaluate distilled data"
    echo "  neighbors   Find nearest real neighbors"
    echo "  centroids   Find real centroid images"
    echo "  random      Train on random real images"
    echo "  full        Train on full dataset"
    echo ""
    echo "Default variables (override via env):"
    echo "  MODEL=$MODEL"
    echo "  EVAL_MODEL=$EVAL_MODEL"
    echo "  DATASET=$DATASET"
    echo "  JOB_TAG=$JOB_TAG"
    echo "  SEED=$SEED"
    echo ""
    echo "Examples:"
    echo "  ./run.sh distill"
    echo "  ./run.sh distill --augs_per_batch=3"
    echo "  MODEL=clip_vitb DATASET=imagenet-1k ./run.sh distill"
    echo "  ./run.sh eval"
    ;;
esac
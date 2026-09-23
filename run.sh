#!/usr/bin/env bash
# Runner LOCAL (pas Jean Zay). Usage : ./run.sh <commande> [--flag=valeur ...]
# Les flags passés en argument sont ajoutés en dernier : ils écrasent les défauts du runner.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=$PWD

# ---------------- paramètres (surchargeables par env) ----------------
MODEL=${MODEL:-dinov2_vitb}
EVAL_MODELS=${EVAL_MODELS:-$MODEL}            # ex: "dinov2_vitb clip_vitb mocov3_vitb"
DATASET=${DATASET:-F4K}
DATA_ROOT=${DATA_ROOT:-$ROOT/data/datasets/$DATASET}
FORMATION=${FORMATION:-physics}               # identity | physics | latent
PRIOR=${PRIOR:-none}                          # none | slurpp | ppg
IPC=${IPC:-1}
SEED=${SEED:-3407}
RES=${RES:-224}
ITERATIONS=${ITERATIONS:-5000}
NUM_WORKERS=${NUM_WORKERS:-8}
MODELS_ROOT=${MODELS_ROOT:-$ROOT/models}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/output/local}
QUICK=${QUICK:-0}
DRYRUN=${DRYRUN:-0}

[[ $FORMATION == identity ]] && PRIOR=none
SUFFIX=""
if [[ $QUICK == 1 ]]; then SUFFIX="_quick"; fi
RUN_NAME=${RUN_NAME:-${MODEL}_${DATASET}_${FORMATION}-${PRIOR}_ipc${IPC}_s${SEED}${SUFFIX}}
JOB_TAG=${JOB_TAG:-$OUTPUT_ROOT/results}

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$OUTPUT_ROOT/wandb}
export WANDB_PROJECT=${WANDB_PROJECT:-local}
export WANDB_NAME=$RUN_NAME

# ---------------- helpers ----------------
die() { echo "erreur: $*" >&2; exit 1; }

check_data() {
  [[ -d $DATA_ROOT ]] || die "DATA_ROOT introuvable : $DATA_ROOT (DATASET=$DATASET)"
}

HELP_MOD=""; HELP_TXT=""
# Remplit ARGS avec les flags "clé=valeur" que <module> accepte (lu depuis son -h)
build_args() {
  local mod=$1; shift
  if [[ $HELP_MOD != "$mod" ]]; then
    HELP_TXT=$(uv run python -m "$mod" -h 2>&1) || true
    grep -q "usage:" <<<"$HELP_TXT" || { echo "$HELP_TXT" >&2; die "impossible de lire l'aide de $mod"; }
    HELP_MOD=$mod
  fi
  ARGS=()
  local kv k
  for kv in "$@"; do
    k=${kv%%=*}
    if grep -qE -- "--${k}([^A-Za-z0-9_]|\$)" <<<"$HELP_TXT"; then
      ARGS+=("--$kv")
    else
      echo "  [skip] --$k non accepté par $mod" >&2
    fi
  done
}

# launch <module> <défauts clé=valeur...>   (flags utilisateur : tableau EXTRA)
launch() {
  local mod=$1; shift
  build_args "$mod" "$@"
  local cmd=(uv run python -m "$mod" "${ARGS[@]}" "${EXTRA[@]}")
  printf '+'; printf ' %q' "${cmd[@]}"; echo
  if [[ $DRYRUN != 1 ]]; then
    mkdir -p "$JOB_TAG" "$WANDB_DIR"
    "${cmd[@]}"
  fi
}

set_prior() {
  PRIOR_ARGS=()
  case $PRIOR in
    none)
      PRIOR_ARGS=(t_channels=1) ;;
    slurpp)
      local ckpt=${SLURPP_CKPT:-$MODELS_ROOT/SLURPP/slurpp/checkpoint/diffusion}
      [[ -d $ckpt ]] || die "checkpoint SLURPP introuvable : $ckpt"
      export BASE_CKPT_DIR=${BASE_CKPT_DIR:-$MODELS_ROOT/SLURPP}
      export PYTHONPATH=$ROOT/slurpp_lib${PYTHONPATH:+:$PYTHONPATH}
      PRIOR_ARGS=(t_channels=3 "slurpp_root=${SLURPP_ROOT:-slurpp_lib}" "slurpp_checkpoint_path=$ckpt") ;;
    ppg)
      local ckpt=${PPG_CKPT:-$MODELS_ROOT/PPG/I40000_E500_ppg.pth}
      [[ -f $ckpt ]] || die "checkpoint PPG introuvable : $ckpt"
      PRIOR_ARGS=(t_channels=1 "ppg_checkpoint_path=$ckpt" ppg_input_channels=3) ;;
    *) die "PRIOR inconnu : $PRIOR (none|slurpp|ppg)" ;;
  esac
}

print_cfg() {
  cat >&2 <<EOF
[run.sh] $1
  RUN_NAME=$RUN_NAME
  MODEL=$MODEL  EVAL_MODELS=$EVAL_MODELS  DATASET=$DATASET
  DATA_ROOT=$DATA_ROOT
  FORMATION=$FORMATION  PRIOR=$PRIOR  IPC=$IPC  SEED=$SEED  RES=$RES  QUICK=$QUICK
  OUTPUT_ROOT=$OUTPUT_ROOT
EOF
}

# ---------------- commandes ----------------
cmd_distill() {
  check_data; set_prior
  local d=(
    "model=$MODEL" "dataset=$DATASET" "data_root=$DATA_ROOT"
    "run_name=$RUN_NAME" "job_tag=$JOB_TAG" "seed=$SEED"
    "ipc=$IPC" "iterations=$ITERATIONS" "num_workers=$NUM_WORKERS"
    "syn_res=$RES" "real_res=$RES" "crop_res=$RES" train_crop_mode=random augs_per_batch=10
    "formation_mode=$FORMATION" distill_mode=pyramid "prior_init=$PRIOR" sample_init=medoids
    "${PRIOR_ARGS[@]}"
    checkpoint_it=100 image_log_it=500 pyramid_snapshot_it=0
    eval_it=100 eval_num_eval=1 eval_epochs=100 eval_metrics=f1
  )
  if [[ $QUICK == 1 ]]; then
    d+=(iterations=50 augs_per_batch=2 pyramid_extent_it=5
        checkpoint_it=25 image_log_it=25 eval_it=0)
  fi
  launch distillation.distill "${d[@]}"
}

cmd_eval() {
  check_data
  local m
  for m in $EVAL_MODELS; do
    local d=(
      "model=$MODEL" "eval_model=$m" "dataset=$DATASET" "data_root=$DATA_ROOT"
      "job_tag=$JOB_TAG" "run_name=$RUN_NAME"
      "real_res=$RES" "crop_res=$RES" "num_workers=$NUM_WORKERS" eval_metrics=f1
    )
    if [[ $QUICK == 1 ]]; then d+=(num_eval=1 eval_epochs=20); fi
    launch distillation.eval "${d[@]}"
  done
}

cmd_baseline() {  # cmd_baseline <module> [défauts supplémentaires...]
  check_data
  local mod=$1; shift
  launch "$mod" \
    "model=$MODEL" "dataset=$DATASET" "data_root=$DATA_ROOT" "job_tag=$JOB_TAG" \
    "ipc=$IPC" "seed=$SEED" "real_res=$RES" "crop_res=$RES" "num_workers=$NUM_WORKERS" "$@"
}

usage() {
  cat <<EOF
Usage : [VAR=...] ./run.sh <commande> [--flag=valeur ...]

Commandes :
  distill     Distille un dataset (RUN_NAME déterministe)
  eval        Évalue le run RUN_NAME sur chaque backbone de EVAL_MODELS
  pipeline    distill puis eval (les flags CLI vont à distill ; EVAL_ARGS pour l'eval)
  neighbors   Plus proches voisins réels du run RUN_NAME
  centroids   Centroïdes réels
  random      Images réelles aléatoires (random_seed=SEED)
  full        Dataset complet

Variables :
  MODEL=$MODEL  EVAL_MODELS="$EVAL_MODELS"  DATASET=$DATASET
  DATA_ROOT=$DATA_ROOT
  FORMATION=$FORMATION  PRIOR=$PRIOR  IPC=$IPC  SEED=$SEED  RES=$RES
  ITERATIONS=$ITERATIONS  NUM_WORKERS=$NUM_WORKERS
  MODELS_ROOT=$MODELS_ROOT  (SLURPP_CKPT, PPG_CKPT pour surcharger)
  OUTPUT_ROOT=$OUTPUT_ROOT  RUN_NAME=<auto>  JOB_TAG=<OUTPUT_ROOT>/results
  QUICK=1   run de test court (suffixe _quick sur RUN_NAME)
  DRYRUN=1  affiche les commandes sans les lancer

Exemples :
  QUICK=1 ./run.sh pipeline
  PRIOR=slurpp IPC=5 ./run.sh distill --lr=1e-3
  EVAL_MODELS="dinov2_vitb clip_vitb mocov3_vitb" ./run.sh eval
  FORMATION=identity ./run.sh distill                # baseline LGM
  DATASET=aqua20 DATA_ROOT=~/datasets/aqua20 ./run.sh random
  DRYRUN=1 PRIOR=ppg ./run.sh pipeline
EOF
}

CMD=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
EXTRA=("$@")

case $CMD in
  distill)   print_cfg distill;  cmd_distill ;;
  eval)      print_cfg eval;     cmd_eval ;;
  pipeline)  print_cfg pipeline; cmd_distill
             read -r -a EXTRA <<<"${EVAL_ARGS:-}"; cmd_eval ;;
  neighbors) print_cfg neighbors; cmd_baseline baselines.neighbors "run_name=$RUN_NAME" ;;
  centroids) print_cfg centroids; cmd_baseline baselines.centroids ;;
  random)    print_cfg random;    cmd_baseline baselines.random_reals "random_seed=$SEED" ;;
  full)      print_cfg full;      cmd_baseline baselines.full_dataset ;;
  help|-h|--help) usage ;;
  *) usage; exit 1 ;;
esac
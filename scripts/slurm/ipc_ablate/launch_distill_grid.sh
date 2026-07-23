#!/usr/bin/env bash
#
# Construit la grille (IPC x variante x seed), ecrit un manifeste TSV,
# et soumet un unique job array.
#
# Exemples :
#   ./launch_distill_grid.sh --dry-run
#   ./launch_distill_grid.sh --ipcs "1 3 5" --variants pixel --seeds "0 1 2 3 4"
#   ./launch_distill_grid.sh --variants "pixel physics_slurpp" --concurrency 6
#
set -euo pipefail

REPO=${REPO:-$WORK/projects/GradientDistillation}
SLURM_SCRIPT=${SLURM_SCRIPT:-scripts/slurm/ipc_ablate/distill_grid.slurm}

MODEL=${MODEL:-dinov2_vitb}
DATASET=${DATASET:-aqua20}
IPCS=${IPCS:-"1 3 5 8"}
VARIANTS=${VARIANTS:-"pixel physics physics_slurpp"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
SAMPLE_INIT=${SAMPLE_INIT:-medoids}
OPT=${OPT:-adam}
TIME=${TIME:-20:00:00}
CONCURRENCY=${CONCURRENCY:-4}
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2";        shift 2 ;;
        --dataset)      DATASET="$2";      shift 2 ;;
        --ipcs)         IPCS="$2";         shift 2 ;;
        --variants)     VARIANTS="$2";     shift 2 ;;
        --seeds)        SEEDS="$2";        shift 2 ;;
        --sample-init)  SAMPLE_INIT="$2";  shift 2 ;;
        --opt)          OPT="$2";          shift 2 ;;
        --time)         TIME="$2";         shift 2 ;;
        --concurrency)  CONCURRENCY="$2";  shift 2 ;;
        --dry-run)      DRY_RUN=1;         shift ;;
        -h|--help)      sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "option inconnue : $1" >&2; exit 1 ;;
    esac
done

# --- source unique de verite pour le mapping variante -> flags ------------
# A verifier contre la signature courante de DistillCfg avant le premier run.
variant_flags() {
    case "$1" in
        pixel)
            echo "--distill_mode=pyramid" ;;
        physics)
            echo "--distill_mode=physics_pyramid --prior_init=none" ;;
        physics_slurpp)
            echo "--distill_mode=physics_pyramid --prior_init=slurpp" ;;
        physics_slurpp_frozen)
            echo "--distill_mode=physics_pyramid --prior_init=slurpp --freeze_T" ;;
        physics_ppg)
            echo "--distill_mode=physics_pyramid --prior_init=ppg" ;;
        latent_predlatent)
            echo "--distill_mode=latent --latent_mode=predlatent --prior_init=slurpp" ;;
        latent_decoder_only)
            echo "--distill_mode=latent --latent_mode=decoder_only --prior_init=slurpp" ;;
        *)
            echo "variante inconnue : $1" >&2; return 1 ;;
    esac
}

# lr par famille : les lr pixel n'ont aucun sens dans l'espace latent
variant_lr() {
    case "$1" in
        latent_*) echo "1e-3" ;;
        *)        echo "0.002" ;;
    esac
}

# --- construction du manifeste -------------------------------------------
cd "$REPO"
mkdir -p logs/manifests logs/ipc_ablation

STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="$REPO/logs/manifests/grid_${STAMP}.tsv"
: > "$MANIFEST"

for ipc in $IPCS; do
    for variant in $VARIANTS; do
        flags=$(variant_flags "$variant")
        lr=$(variant_lr "$variant")
        for seed in $SEEDS; do
            run_name="${MODEL}_ipc${ipc}_${variant}_${SAMPLE_INIT}_${OPT}_lr${lr}_s${seed}"
            full_flags="--ipc=${ipc} ${flags} --sample_init=${SAMPLE_INIT}"
            full_flags+=" --distill_opt=${OPT} --lr=${lr} --seed=${seed}"
            printf '%s\t%s\n' "$run_name" "$full_flags" >> "$MANIFEST"
        done
    done
done

N=$(wc -l < "$MANIFEST")

echo "manifeste : $MANIFEST"
echo "taches    : $N  (concurrence ${CONCURRENCY}, walltime ${TIME})"
echo "------------------------------------------------------------------"
nl -ba "$MANIFEST" | sed 's/\t/  |  /'
echo "------------------------------------------------------------------"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] rien de soumis."
    exit 0
fi

JOBID=$(sbatch --parsable \
    --array="1-${N}%${CONCURRENCY}" \
    --time="$TIME" \
    --job-name="distill_${DATASET}_${STAMP}" \
    --export=ALL,MANIFEST="$MANIFEST",MODEL="$MODEL",DATASET="$DATASET" \
    "$SLURM_SCRIPT")

echo "soumis : job array ${JOBID}"
echo "suivi  : squeue -j ${JOBID}"
echo "relance d'une tache : sbatch --array=<id> --export=ALL,MANIFEST=${MANIFEST} ${SLURM_SCRIPT}"
#!/usr/bin/env bash
#
# Construit la grille (IPC x variante x seed), ecrit un manifeste TSV,
# et soumet un unique job array.
#
# Exemples :
#   ./launch_distill_grid.sh --dry-run
#   ./launch_distill_grid.sh --ipcs "1 3 5" --variants pixel --seeds "0 1 2 3 4"
#   ./launch_distill_grid.sh --variants "pixel physics_slurpp" --concurrency 6
#   ./launch_distill_grid.sh --output-root $WORK/.../output/autre_campagne
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
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO/output/ipc_ablation}
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
        --output-root)  OUTPUT_ROOT="$2";  shift 2 ;;
        --dry-run)      DRY_RUN=1;         shift ;;
        -h|--help)      sed -n '2,13p' "$0"; exit 0 ;;
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

# lr par famille : les lr pixel n'ont aucun sens dans l'espace latent.
# Valeurs heritees de runs deja valides dans leurs espaces respectifs.
variant_lr() {
    case "$1" in
        latent_*) echo "1e-3" ;;
        *)        echo "0.002" ;;
    esac
}

# --- arborescence de sortie (miroir du launcher latent) -------------------
cd "$REPO"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT/wandb" "$OUTPUT_ROOT/results" logs/manifests

# --- construction du manifeste -------------------------------------------
STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="$REPO/logs/manifests/grid_${STAMP}.tsv"
: > "$MANIFEST"

for ipc in $IPCS; do
    for variant in $VARIANTS; do
        flags=$(variant_flags "$variant")
        lr=$(variant_lr "$variant")
        for seed in $SEEDS; do
            run_name="${MODEL}_ipc${ipc}_${variant}_${SAMPLE_INIT}_${OPT}_lr${lr}_s${seed}"

            # garde-fou : deux runs de meme nom s'ecraseraient silencieusement
            if [[ -d "$OUTPUT_ROOT/results/${DATASET}/${MODEL}/${run_name}" ]]; then
                echo "[erreur] ${run_name} existe deja sous ${OUTPUT_ROOT} —" \
                     "supprimer, renommer, ou changer --output-root" >&2
                exit 1
            fi

            full_flags="--ipc=${ipc} ${flags} --sample_init=${SAMPLE_INIT}"
            full_flags+=" --distill_opt=${OPT} --lr=${lr} --seed=${seed}"
            printf '%s\t%s\n' "$run_name" "$full_flags" >> "$MANIFEST"
        done
    done
done

N=$(wc -l < "$MANIFEST")

echo "manifeste   : $MANIFEST"
echo "output root : $OUTPUT_ROOT"
echo "taches      : $N  (concurrence ${CONCURRENCY}, walltime ${TIME})"
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
    --output="$LOG_DIR/%A_%a.out" \
    --error="$LOG_DIR/%A_%a.err" \
    --export=ALL,MANIFEST="$MANIFEST",MODEL="$MODEL",DATASET="$DATASET",OUTPUT_ROOT="$OUTPUT_ROOT" \
    "$SLURM_SCRIPT")

echo "soumis  : job array ${JOBID}"
echo "suivi   : squeue -j ${JOBID}"
echo "logs    : ${LOG_DIR}/${JOBID}_<task>.out"
echo "results : ${OUTPUT_ROOT}/results/${DATASET}/${MODEL}/<run_name>/"
echo "relance d'une tache :"
echo "  sbatch --array=<id> --export=ALL,MANIFEST=${MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT} ${SLURM_SCRIPT}"   
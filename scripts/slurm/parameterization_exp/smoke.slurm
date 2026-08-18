#!/bin/bash
# Quick smoke test BEFORE launching the 26-variant sweep.
# Covers every refactor risk point cheaply, cheapest-first, fail-fast (set -e):
#   - factory routing (identity -> fork)
#   - physics host: pyramid decode + extend + optimizer rebuild + physics_metrics
#   - checkpoint round-trip (get_save_dict / load_from_dict) via a resume
#   - pixel decode path, ppg prior (t_channels=1)
#   - slurpp init (dual-UNet), t_channels=3, b_spatial=True, freeze groups
#   - random sample_init branch
#   - f1 eval path + save_data (get_to_save / "save_dict" condition)
# Uses the h100 dev qos so it schedules fast. Results go under job_tag=smoke_test
# (disposable, isolated from the real experiment).
#SBATCH --job-name=smoke_refactor
#SBATCH --account=rbw@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=00:40:00
#SBATCH --output=parameterization_exp/logs/smoke_%j.out
#SBATCH --error=parameterization_exp/logs/smoke_%j.err
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
export BASE_CKPT_DIR=$WORK/models/SLURPP
export PYTHONUSERBASE=$WORK/python_user/h100
export PATH=$PYTHONUSERBASE/bin:$PATH

cd $WORK/projects/GradientDistillation
mkdir -p parameterization_exp/logs

SLURPP_CKPT=$WORK/models/SLURPP/slurpp/checkpoint/diffusion
PPG_CKPT=$WORK/models/PPG/I40000_E500_ppg.pth

# tiny + aggressive so refactor code paths actually fire fast:
#   pyramid_extent_it=5 -> extension at step 6 & 11 (tests extend + optim rebuild)
#   checkpoint_it=4     -> ckpt.pth written (tests get_save_dict)
#   eval_it=6, epochs=3 -> f1 eval path runs, cheap
COMMON="--model=dinov2_vitb --dataset=aqua20 --data_root=$WORK/datasets/aqua20 \
--job_tag=smoke_test --augs_per_batch=2 --syn_res=224 --real_res=224 --crop_res=224 \
--train_crop_mode=random --ipc=1 --num_workers=8 --clamp_I=True \
--pyramid_extent_it=5 --pyramid_snapshot_it=6 --image_log_it=6 \
--checkpoint_it=4 --eval_it=6 --eval_num_eval=1 --eval_epochs=3 --eval_metrics=f1"

run() {  # run <run_name> <iterations> <extra args...>
    local name=$1; local iters=$2; shift 2
    echo ">>> SMOKE: $name (iters=$iters)"
    srun python -m distillation.distill --run_name=$name --iterations=$iters $COMMON "$@"
    echo "<<< OK: $name"
}

# 1. factory routing: identity -> fork (cheapest sanity)
run smoke_identity_pyramid 8 --distill_mode=pyramid --formation_mode=identity

# 2. physics core: pyramid + none  (decode, extend, physics_metrics, none prior)
run smoke_phys_none_pyr 8 --distill_mode=pyramid --formation_mode=physics \
    --prior_init=none --sample_init=medoids --freeze_T=false --freeze_B=false

# 3. checkpoint round-trip: resume the SAME run to 16 (loads ckpt after an extension)
run smoke_phys_none_pyr 16 --distill_mode=pyramid --formation_mode=physics \
    --prior_init=none --sample_init=medoids --freeze_T=false --freeze_B=false

# 4. pixel path + ppg prior (t_channels=1) + random sample_init branch
run smoke_phys_ppg_pix_rand 8 --distill_mode=pixel --formation_mode=physics \
    --prior_init=ppg --sample_init=random --freeze_T=false --freeze_B=false \
    --ppg_checkpoint_path=$PPG_CKPT --ppg_input_channels=3 --t_channels=1

# 5. heaviest: slurpp init, t_channels=3, b_spatial=True, both frozen (optim groups)
run smoke_phys_slurpp_pyr_frozen 8 --distill_mode=pyramid --formation_mode=physics \
    --prior_init=slurpp --sample_init=medoids --freeze_T=true --freeze_B=true \
    --t_channels=3 --b_spatial=True \
    --slurpp_root=slurpp_lib --slurpp_checkpoint_path=$SLURPP_CKPT

echo "========================================"
echo "SMOKE TEST PASSED (all configs ran + saved)"
echo "========================================"

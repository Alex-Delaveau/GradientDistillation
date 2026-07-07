# Learning-Rate Sweep

Fixed parameterization; grid over the three learning rates (base `lr`, `lr_T`, `lr_B`)
to study T/B optimization dynamics. AQUA20, DINOv2 ViT-B, IPC=1, macro F1.

## Fixed parameterization
distill_mode=pyramid · formation_mode=physics · prior_init=slurpp · sample_init=medoids
freeze_T=False · freeze_B=False   (MUST stay False, else lr_T/lr_B are ignored:
                                   frozen T/B are excluded from the optimizer)
slurpp: t_channels=3, b_spatial=True

## Swept axes (edit arrays in launch_lr_exp.sh)
lr    in {1e-3, 2e-3, 5e-3}
lr_T  in {2e-3, 1e-2}
lr_B  in {2e-3, 1e-2}
-> 12 jobs. run_name: <model>_pyramid_slurpp_lr<lr>_lrT<lrT>_lrB<lrB>

## Config prerequisite
DistillCfg must declare annotated fields:
    lr: float = 2e-3
    lr_T: float | None = None
    lr_B: float | None = None
and init_optimizer must fall back to lr when lr_T/lr_B is None.

## Fixed run settings
res=224 · iterations=5000 · augs_per_batch=10 · eval_it=100 · eval_metrics=f1 (macro) ·
checkpoint_it=100 · pyramid_snapshot_it=500 · clamp_I=True
Hardware: H100 (rbw@h100, qos_gpu_h100-t3); torch from IDRIS module (NOT uv, wheel incompat).

## Outputs — ABSOLUTE path (no relative dirs, no symlink)
$WORK/projects/GradientDistillation/output/lr_exp/
├── logs/     <jobid>_<run_name>.out / .err
├── wandb/    offline runs (sync: uv run wandb sync)
└── results/aqua20/dinov2_vitb/<run_name>/
    ├── data.pth   (final deliverable, written at step 5000 only)
    └── ckpt.pth   (resume checkpoint, from step 100)
Redirection trick: --job_tag=$OUTPUT_ROOT/results is absolute, so linear_gm's
os.path.join discards the "logged_files/" prefix and writes under the absolute root.

## Run / monitor
DRYRUN=1 ./launch_lr_exp.sh    # preview
./launch_lr_exp.sh             # submit
squeue -u $USER
tail -f output/lr_exp/logs/*.out   # init is slow (medoids DINOv2 pass + slurpp dual-UNet load)

## Caveats
- Single-seed (n=1) -> directional; wrap in a seed loop for CI.
- data.pth appears only at the end; empty results/ mid-run is normal.
- b_spatial=True: B-drift metrics NOT comparable to pre-refactor slurpp runs.
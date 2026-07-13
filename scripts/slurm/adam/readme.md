# Adam Experiment


Our current hypothesis is that hte convergence advantages we get from the physics does not come from physics but by a better coefficient in the update of $\Delta{I}$ with gradient $\partial{L}/\partial{I}$


To test this hypothesis, we aim to replace the use of SGD by Adam, allowing us to normalize to gradients of the parameters
## Axes

- `distill_mode`   : pyramid
- `formation_mode` : identity | physics        (identity = LGM fork baseline, prior unused)
- `prior_init`     : none | slurpp        (T/B init, physics only)
- `sample_init`    : medoids

## Fixed settings (in the .slurm)

syn/real/crop_res=224, ipc=1, iterations=5000, augs_per_batch=10, eval_it=100, eval_metrics=f1 (macro), eval_num_eval=1, checkpoint_it=100,
pyramid_snapshot_it=500, clamp_I=True
- slurpp: `t_channels=3`, `b_spatial=True` (keep per-channel + spatial B)

Hardware: H100 (`rbw@h100`, `qos_gpu_h100-t3`), torch from IDRIS module
`pytorch-gpu/py3/2.8.0` (NOT uv — wheel incompatibility), `PYTHONUSERBASE=$WORK/python_user/h100`.
Model ckpts: SLURPP `$WORK/models/SLURPP`

## Files

- `adam_exp.slurm`        — generic job, config passed via env
- `launch_adam_exp.sh`    — generates + submits the 26 variants (`DRYRUN=1` to preview)
- `smoke_test.slurm`                  — quick pre-flight (h100 dev qos) covering all code paths

## Outputs — all under `adam_exp/`

```
adam_exp/
├── logs/     <jobid>_<run_name>.out / .err
├── wandb/    offline runs (sync from login node: `uv run wandb sync`)
└── results/  -> logged_files/adam_exp/   (symlink)
```

Per-run results (written by `linear_gm`, hardcoded under `logged_files/`):
`logged_files/adam_exp/aqua20/dinov2_vitb/<run_name>/`
- `data.pth` — final deliverable (distilled I, J/T/B, snapshots), written at step 5000 only
- `ckpt.pth` — resume checkpoint (from step 100 onward)

`run_name` encodes the variant, e.g. `dinov2_vitb_pyramid_slurpp_medoids_ft1_fb1`.

## Run / monitor / resume

```bash
DRYRUN=1 ./launch_adam_exp.sh   # preview
./launch_adam_exp.sh            # submit 26
squeue -u $USER                             # status
tail -f adam_exp/logs/*.out     # live (init is slow: medoids=DINOv2 pass, slurpp=dual-UNet load)
```

No auto-requeue (no `--signal`/`--requeue`): a job killed at walltime is resumed by
**re-submitting the same config** — it finds `ckpt.pth` via the deterministic `run_name`.

## Caveats

- Single-seed (n=1) → directional only. Wrap the launcher in a seed loop for CI intervals.
- `data.pth` appears only at the end; empty `logged_files/` mid-run is normal.
- Bool CLI syntax (`freeze_T` etc.): validated by the smoke test — the sweep mixes forms.
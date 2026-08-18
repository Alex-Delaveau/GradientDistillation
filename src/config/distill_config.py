from typing import Literal

from .base_config import BaseCfg


class DistillCfg(BaseCfg):
    num_workers: int = 16
    seed : int = 3407
    job_tag: str = "distillation"

    ipc: int = 1
    lr: float = 2e-3
    iterations: int = 5000
    augs_per_batch: int = 10

    distill_mode: Literal["pixel", "pyramid"] = "pyramid"
    aug_mode: Literal["standard", "none"] = "standard"
    decorrelate_color: bool = True
    distill_opt: Literal["sgd", "adam"] = "adam"

    init_mode: Literal["noise", "zero"] = "noise"

    pyramid_extent_it: int = 200
    pyramid_start_res: int = 1
    pyramid_snapshot_it: int = 0  # save decoded pyramid every N iterations (0 = disabled)

    image_log_it: int = 500

    run_name: str | None = None

    checkpoint_it: int = 100

    syn_res: int = 256 
    real_res: int = 256 
    crop_res: int = 224 

    train_crop_mode: Literal["center", "random"] = "random"

    # --- Eval dataset during the distillation ---
    eval_it: int = 0          # run linear probe every N steps (0 = disabled)
    eval_epochs: int = 1000   # max epochs per probe run; early stopping usually terminates earlier
    eval_patience: int = 5    # early stopping patience for the periodic probe
    eval_num_eval: int = 3    # number of probe runs to average
    eval_metrics: Literal["accuracy", "f1"] = "accuracy"

    # --- physics formation model (PhysicsFormationDataset) ---
    formation_mode: Literal["identity", "physics", "latent"] = "physics"
    prior_init: Literal["none", "ppg", "slurpp"] = "none"
    sample_init : Literal["medoids", "random"] = "medoids"
    freeze_T: bool = False
    freeze_B: bool = False
    t_channels: Literal[1, 3] = 1
    b_spatial: bool = False
    clamp_I: bool = True
    lr_T: float | None = None
    lr_B: float | None = None

    # SLURPP (requis seulement si prior_init="slurpp")
    slurpp_root: str = ""
    slurpp_checkpoint_path: str = ""

    # PPG (requis si prior_init="ppg")
    ppg_checkpoint_path: str = ""
    ppg_input_channels: int = 3

    # Latent
    latent_mode: Literal["predlatent", "decoder_only"] = "predlatent" 
    latent_chunk: int = 2 
    latent_res: int = 512
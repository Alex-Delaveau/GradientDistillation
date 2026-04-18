from typing import Literal

import torch
from .base_config import BaseCfg


class DistillCfg(BaseCfg):
    num_workers: int = 16

    job_tag: str = "distillation"

    ipc: int = 1
    lr = 2e-3
    iterations: int = 5000
    augs_per_batch: int = 10

    distill_mode: Literal["pixel", "pyramid"] = "pyramid"
    aug_mode: Literal["standard", "none"] = "standard"
    decorrelate_color: bool = True

    init_mode: Literal["noise", "zero"] = "noise"

    pyramid_extent_it: int = 200
    pyramid_start_res: int = 1

    image_log_it: int = 500

    run_name: str | None = None

    checkpoint_it: int = 100

    syn_res: int = 126
    real_res: int = 126
    crop_res: int = 126

    train_crop_mode: Literal["center", "random"] = "random"

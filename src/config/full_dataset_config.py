from typing import Literal

from .base_config import EvalTrainCfg


class FullDatasetCfg(EvalTrainCfg):
    workers_per_gpu: int = 16
    batch_size_per_gpu: int = 256
    real_res: int = 126
    crop_res: int = 126
    num_eval: int = 5
    eval_epochs: int = 100
    train_crop_mode: Literal["center", "random"] = "random"

    checkpoint_it: int = 10
    do_f1: bool = False
from typing import Literal

import torch
from tap import Tap
from my_utils.device import DeviceSingleton

class FullDatasetCfg(Tap):
    dataset: str
    model: str
    data_root: str = "data/datasets"
    workers_per_gpu: int = 16
    batch_size_per_gpu: int = 256
    real_res: int = 256
    crop_res: int = 224
    num_eval: int = 5
    eval_epochs: int = 100
    train_crop_mode: Literal["center", "random"] = "random"

    device_count: int = DeviceSingleton.device_count()
    patience: int = 5
    eval_it: int = -1

    checkpoint_it: int = 10

    skip_if_exists: bool = True

from typing import Literal

from .base_config import EvalTrainCfg


class EvalCfg(EvalTrainCfg):
    eval_model: str
    job_tag: str = "distillation"
    num_workers: int = 0
    real_batch_size: int = 100
    real_res: int = 256
    crop_res: int = 224
    num_eval: int = 5
    train_crop_mode: Literal["center", "random"] = "random"

    job_id: str | None = None
    run_name: str | None = None
    do_f1: bool = False

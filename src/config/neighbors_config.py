from .base_config import BaseCfg


class NeighborsCfg(BaseCfg):
    job_tag: str = "distillation"
    num_workers: int = 16
    real_batch_size: int = 100
    real_res: int = 126
    crop_res: int = 126

    run_name: str | None = None
    job_id: str | None = None

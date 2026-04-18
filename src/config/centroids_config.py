from .base_config import BaseCfg


class CentroidRealsCfg(BaseCfg):
    num_workers: int = 16
    real_batch_size: int = 100
    real_res: int = 256
    crop_res: int = 224

    job_tag: str | None = None
    job_id: str | None = None

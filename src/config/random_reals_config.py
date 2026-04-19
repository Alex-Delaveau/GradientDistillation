from .base_config import EvalTrainCfg


class RandomRealsConfig(EvalTrainCfg):
    random_seed: int
    workers_per_gpu: int = 16
    batch_size_per_gpu: int = 256
    real_res: int = 256
    crop_res: int = 224

    job_tag: str | None = None
    job_id: str | None = None

    ipc: int = 1
    do_f1: bool = False
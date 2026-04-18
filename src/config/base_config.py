from tap import Tap
from my_utils.device import DeviceSingleton


class BaseCfg(Tap):
    """Universal fields shared by all configs."""
    dataset: str
    model: str
    data_root: str = "data/datasets"
    skip_if_exists: bool = True
    device_count: int = DeviceSingleton.device_count()


class EvalTrainCfg(BaseCfg):
    """Adds evaluation training loop fields (EvalCfg, FullDatasetCfg, RandomRealsConfig)."""
    eval_epochs: int = 1000
    patience: int = 5
    eval_it: int = -1
    checkpoint_it: int = 100

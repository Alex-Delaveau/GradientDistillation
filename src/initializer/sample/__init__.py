from typing import Tuple
from .medoids_init import MedoidInitializer
from .random_init import RandomInitializer


def build_sample_init(backbone, num_feat, cfg, train_dataset):
    if cfg.sample_init == "medoids":
        return MedoidInitializer(cfg, train_dataset, backbone, num_feat, cfg.seed)
    if cfg.sample_init == "random":
        return RandomInitializer(cfg, train_dataset, seed=cfg.seed)
    raise ValueError(f"sample_init unknown: {cfg.sample_init}")



def ordered_indices(sample_idx: dict, train_dataset, ipc: int) -> Tuple[list, list]:
    """Flatten {class: [global indices]} into class-major order."""
    ordered, labels = [], []
    for c in range(train_dataset.num_classes):
        idxs = list(sample_idx[c])
        if len(idxs) < ipc:                 # class too small -> repeat
            idxs = (idxs * ipc)[:ipc]
        ordered += idxs[:ipc]
        labels += [c] * ipc
    return ordered, labels
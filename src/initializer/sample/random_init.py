from .base_sample_init import BaseSampleInitializer
from config import DistillCfg
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np


class RandomInitializer(BaseSampleInitializer):

    def __init__(self, cfg: DistillCfg, train_dataset: Dataset,
                 backbone: nn.Module = None, num_feat: int = None,
                 seed: int | None = None):
        super().__init__(cfg, train_dataset, backbone, num_feat, seed)
        seed = seed if seed is not None else getattr(cfg, "seed", None)
        self.rng = np.random.default_rng(seed)

    def get_indices(self) -> dict:
        return self.find_random()

    def find_random(self, labels: torch.Tensor = None) -> dict:
        if labels is None:
            labels = self.get_labels()
        labels_np = labels.numpy()

        random_global_idx = {}
        for c in range(self.num_classes):
            cls_idx = np.where(labels_np == c)[0]
            k = min(self.cfg.ipc, len(cls_idx))
            chosen = self.rng.choice(cls_idx, size=k, replace=False)
            random_global_idx[c] = chosen.tolist()

        return random_global_idx
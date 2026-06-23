from config import DistillCfg
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np


class RandomInitializer():

    def __init__(self, cfg: DistillCfg, train_dataset: Dataset,
                 backbone: nn.Module = None, num_feat: int = None,
                 seed: int | None = None):
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=8)
        seed = seed if seed is not None else getattr(cfg, "seed", None)
        self.rng = np.random.default_rng(seed)

    def get_indices(self) -> dict:
        return self.find_random()

    def get_labels(self) -> torch.Tensor:
        """Collect labels for the whole training set (no backbone forward needed)."""
        labels = []
        for _, y in self.train_loader:
            labels.append(y)
        return torch.cat(labels)

    def find_random(self, labels: torch.Tensor = None) -> dict:
        if labels is None:
            labels = self.get_labels()
        labels_np = labels.numpy()

        random_global_idx = {}
        for c in range(self.train_loader.dataset.num_classes):
            cls_idx = np.where(labels_np == c)[0]
            k = min(self.cfg.ipc, len(cls_idx))
            chosen = self.rng.choice(cls_idx, size=k, replace=False)
            random_global_idx[c] = chosen.tolist()

        return random_global_idx
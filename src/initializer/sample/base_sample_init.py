from abc import ABC, abstractmethod
from config import DistillCfg
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn


class BaseSampleInitializer(ABC):

    def __init__(self, cfg: DistillCfg, train_dataset: Dataset,
                 backbone: nn.Module = None, num_feat: int = None,
                 seed: int | None = None):
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=32,
                                       shuffle=False, num_workers=8)
        self.backbone = backbone
        self.num_feat = num_feat
        self.seed = seed

    @property
    def num_classes(self) -> int:
        return self.train_loader.dataset.num_classes

    def get_labels(self) -> torch.Tensor:
        """Collect labels for the whole training set (no backbone forward needed)."""
        labels = [y for _, y in self.train_loader]
        return torch.cat(labels)

    @abstractmethod
    def get_indices(self) -> dict:
        """Return {class_idx: [global_sample_indices]}."""
        ...
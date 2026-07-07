from config import DistillCfg
from data.dataloaders import BaseRealDataset

from .base import BaseDistilledDataset
from .pixels import PixelDataset
from .pyramid import PyramidDataset
from .physics_formation_pyramid import PhysicsFormationDataset
import torch.nn as nn



    
def get_distilled_dataset(
    train_dataset: BaseRealDataset, cfg: DistillCfg, backbone: nn.Module | None = None, num_feat: int | None = None
) -> BaseDistilledDataset:
    if cfg.formation_mode == "identity":
        # LGM base code, prior_init not used
        if cfg.distill_mode == "pixel":
            return PixelDataset(train_dataset=train_dataset, cfg=cfg)
        if cfg.distill_mode == "pyramid":
            return PyramidDataset(train_dataset=train_dataset, cfg=cfg)
        raise ValueError(f"distill_mode {cfg.distill_mode} invalide")
    if cfg.formation_mode == "physics":
        return PhysicsFormationDataset(train_dataset, cfg, backbone, num_feat)
    raise ValueError(f"formation_mode {cfg.formation_mode} invalide")

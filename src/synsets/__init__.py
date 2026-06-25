from config import DistillCfg
from data.dataloaders import BaseRealDataset

from .base import BaseDistilledDataset
from .pixels import PixelDataset
from .pyramid import PyramidDataset
from .physics import PhysicsDataset
from .physics_pyramid import PhysicsPyramidDataset
from .seathru_physics_pyramid import SeaThruPyramidDataset
from .physics_formation_pyramid import PhysicsFormationDataset
import torch.nn as nn


def get_distilled_dataset(
    train_dataset: BaseRealDataset, cfg: DistillCfg, backbone: nn.Module | None = None, num_feat: int | None = None
) -> BaseDistilledDataset:

    match cfg.distill_mode:

        case "pixel":
            ds = PixelDataset(train_dataset=train_dataset, cfg=cfg)

        case "pyramid":
            ds = PyramidDataset(train_dataset=train_dataset, cfg=cfg)

        case "physics":
            ds = PhysicsDataset(train_dataset=train_dataset, cfg=cfg)

        case "physics_pyramid":
            ds = PhysicsPyramidDataset(train_dataset=train_dataset, cfg=cfg, backbone=backbone, num_feat=num_feat)

        case "seathru_pyramid":
            ds = SeaThruPyramidDataset(train_dataset=train_dataset, cfg=cfg)

        case "physics_formation":
            ds = PhysicsFormationDataset(train_dataset=train_dataset, cfg=cfg, backbone=backbone, num_feat=num_feat)
        case _:
            raise NotImplementedError(
                "Distillation mode {} not implemented".format(cfg.distill_mode)
            )

    return ds

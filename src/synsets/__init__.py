from config import DistillCfg
from data.dataloaders import BaseRealDataset

from .base import BaseDistilledDataset
from .pixels import PixelDataset
from .pyramid import PyramidDataset
from .physics import PhysicsDataset
from .physics_pyramid import PhysicsPyramidDataset


def get_distilled_dataset(
    train_dataset: BaseRealDataset, cfg: DistillCfg
) -> BaseDistilledDataset:

    match cfg.distill_mode:

        case "pixel":
            ds = PixelDataset(train_dataset=train_dataset, cfg=cfg)

        case "pyramid":
            ds = PyramidDataset(train_dataset=train_dataset, cfg=cfg)

        case "physics":
            ds = PhysicsDataset(train_dataset=train_dataset, cfg=cfg)

        case "physics_pyramid":
            ds = PhysicsPyramidDataset(train_dataset=train_dataset, cfg=cfg)

        case _:
            raise NotImplementedError(
                "Distillation mode {} not implemented".format(cfg.distill_mode)
            )

    return ds

from torch import nn

from .basic_aug import AugBasic
from .none_aug import AugNone
from .standard_aug import AugStandard
from my_utils import DeviceSingleton


def get_augmentor(aug_mode: str, crop_res: int) -> nn.Module:

    if aug_mode == "standard":
        return AugStandard(crop_res=crop_res).to(DeviceSingleton.get())
    elif aug_mode == "none":
        return AugNone(crop_res=crop_res)
    else:
        raise NotImplementedError("Unknown augmentation mode: {}".format(aug_mode))

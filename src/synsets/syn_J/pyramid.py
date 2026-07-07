from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor

from .base import JParameterization

from src.my_utils.log_utils import log
from src.synsets.color import linear_decorrelate_color

class PyramidJ(JParameterization):

    def __init__(self, cfg, N: int, device):
        self.cfg = cfg
        self.N = N
        self.device = device
        self.pyramid_J: List[Tensor] = self._init_pyramid()

    def _init_pyramid(self) -> List[Tensor]:
        levels, res = [], 1
        while res <= self.cfg.pyramid_start_res:
            level = torch.randn((self.N, 3, res, res), device=self.device)
            if self.cfg.init_mode == "zero":
                level = level * 0
            levels.insert(0, level)
            res *= 2
            if res > self.cfg.syn_res:
                res = self.cfg.syn_res
        levels = [p / len(levels) for p in levels]
        for p in levels:
            p.requires_grad_(True)
        log(type(self).__name__, f"pyramid initialized with {len(levels)} level(s), "
            f"start_res={self.cfg.pyramid_start_res}")
        return levels

    def decode(self, n_levels: int | None = None) -> Tensor:
        levels = self.pyramid_J if n_levels is None else self.pyramid_J[-n_levels:]
        result = torch.sum(torch.stack([
            F.interpolate(p, (self.cfg.syn_res, self.cfg.syn_res),
                          antialias=False, mode="bilinear")
            for p in levels
        ]), dim=0)
        if self.cfg.decorrelate_color:
            result = linear_decorrelate_color(result)
        return torch.sigmoid(2 * result)

    def parameters(self) -> list[Tensor]:
        return self.pyramid_J

    def extend(self) -> bool:
        old_len = len(self.pyramid_J)
        new_len = old_len + 1
        old_res = self.pyramid_J[0].shape[-1]
        if old_res == self.cfg.syn_res:
            log(type(self).__name__, "pyramid already at max resolution, no extension")
            return False
        new_res = min(old_res * 2, self.cfg.syn_res)
        num_images = self.pyramid_J[-1].shape[0]

        # rescale in-place -> keep object identity (= Adam state of existing J levels)
        with torch.no_grad():
            for p in self.pyramid_J:
                p.mul_(old_len / new_len)

            if self.cfg.init_mode == "zero":
                new_layer = torch.sum(torch.stack([
                    F.interpolate(p, (new_res, new_res), antialias=False, mode="bilinear")
                    for p in self.pyramid_J
                ]), dim=0) / old_len
            else:
                new_layer = torch.randn((num_images, 3, new_res, new_res),
                                        device=self.device) / new_len

        new_layer.requires_grad_(True)
        self.pyramid_J.insert(0, new_layer)

        log(type(self).__name__, f"pyramid extended: {old_len} -> {new_len} levels (new res {new_res})")
        return True

    def snapshot_levels(self):
        J_levels = [self.decode(n_levels=k) for k in range(1, len(self.pyramid_J) + 1)]
        level_res = [p.shape[-1] for p in reversed(self.pyramid_J)]
        return J_levels, level_res

    def state_dict(self) -> dict:
        return {"pyramid_J": self.pyramid_J}

    def load_state_dict(self, d: dict) -> None:
        while len(self.pyramid_J) < len(d["pyramid_J"]):
            self.extend()
        with torch.no_grad():
            for p, loaded in zip(self.pyramid_J, d["pyramid_J"]):
                p.copy_(loaded)
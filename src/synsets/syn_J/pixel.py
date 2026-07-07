import torch
from torch import Tensor
from .base import JParameterization
from src.synsets.color import linear_decorrelate_color


class PixelJ(JParameterization):

    def __init__(self, cfg, N: int, device):
        self.cfg = cfg
        self.J = self._init_J(N, device)

    def _init_J(self, N: int, device) -> Tensor:
        syn_J = torch.randn((N, 3, self.cfg.syn_res, self.cfg.syn_res), device=device)
        if self.cfg.init_mode == "zero":
            syn_J = syn_J * 0
        syn_J.requires_grad_(True)
        return syn_J

    def decode(self, n_levels: int | None = None) -> Tensor:
        result = self.J
        if self.cfg.decorrelate_color:
            result = linear_decorrelate_color(result)
        return torch.sigmoid(2 * result)

    def parameters(self) -> list[Tensor]:
        return [self.J]

    def state_dict(self) -> dict:
        return {"pixels_J": self.J}

    def load_state_dict(self, d: dict) -> None:
        with torch.no_grad():
            self.J.copy_(d["pixels_J"])
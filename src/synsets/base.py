from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor
from my_utils.device import DeviceSingleton

class BaseDistilledDataset:
    optimizer: torch.optim.Optimizer
    syn_lr: Tensor
    res: int
    num_samples: int

    def __init__(self):
        self.color_correlation_svd_sqrt = torch.tensor(
            [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
        ).to(DeviceSingleton.get())

        self.max_norm_svd_sqrt = torch.max(
            torch.linalg.norm(self.color_correlation_svd_sqrt, axis=0)
        )

        self.color_mean = torch.tensor([0.48, 0.46, 0.41]).to(DeviceSingleton.get())

    def build_optimizer(self, param_groups: list) -> torch.optim.Optimizer:
        opt = getattr(self.cfg, "distill_opt", "sgd")
        if opt == "sgd":
            return torch.optim.SGD(param_groups, momentum=0.5)
        if opt == "adam":
            return torch.optim.Adam(param_groups)
        raise NotImplementedError(f"unknown distill_opt: {opt}")

    def get_data(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
    
    def get_to_save(self) -> dict:
        return {
            "syn_data": self.get_data()
        }

    def log_images(self, step: int = None):
        raise NotImplementedError

    def upkeep(self, step: int = None):
        return

    def get_save_dict(self):
        return
    
    @torch.no_grad()
    def gradient_metrics(self, grads: dict) -> dict:
        gs = [
            p.grad.reshape(-1)
            for g in self.optimizer.param_groups
            for p in g["params"]
            if p.grad is not None
        ]
        out = dict(grads)
        out["grad/norm_total"] = torch.cat(gs).norm().item() if gs else 0.0
        return out

    def load_from_dict(self, load_dict: dict):
        return

    def linear_decorrelate_color(self, im: Tensor):
        b, c, h, w = im.shape
        im = rearrange(im, "b c h w -> (b h w) c")
        color_correlation_normalized = (
            self.color_correlation_svd_sqrt / self.max_norm_svd_sqrt
        )
        im = im @ color_correlation_normalized.T
        im = rearrange(im, "(b h w) c -> b c h w", h=h, w=w)
        return im

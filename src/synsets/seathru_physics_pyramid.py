from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from config import DistillCfg
from data.dataloaders import BaseRealDataset
from my_utils.device import DeviceSingleton
from my_utils.log_utils import log_images

from .base import BaseDistilledDataset


class SeaThruPyramidDataset(BaseDistilledDataset):
    """
    Sea-Thru reparametrization of the Koschmieder model.

    Physical model (Akkaynak & Treibitz, CVPR 2019):
        T^c(x, y) = exp(-beta^c * d(x, y))
        I^c       = J^c * T^c + (1 - T^c) * B^c

    Parameters optimized:
        - pyramid_J:      clean radiance, multiscale (like LGM)
        - syn_d_logit:    depth map, 1 channel (softplus -> positive)
        - syn_beta_logit: per-image per-channel attenuation (softplus -> positive)
        - syn_B:          per-image ambient background colour (sigmoid -> [0, 1])
    """

    def __init__(self, train_dataset: BaseRealDataset, cfg: DistillCfg):
        super().__init__()
        self.train_dataset = train_dataset
        self.cfg = cfg

        (
            (self.pyramid_J, self.syn_d_logit, self.syn_beta_logit, self.syn_B),
            self.syn_labels,
        ) = self.init_synset()
        self.optimizer = self.init_optimizer()

    def init_synset(
        self,
    ) -> Tuple[Tuple[List[Tensor], Tensor, Tensor, Tensor], Tensor]:
        N = self.cfg.ipc * self.train_dataset.num_classes
        H = W = self.cfg.syn_res
        device = DeviceSingleton.get()

        syn_labels = torch.cat(
            [
                torch.tensor([c] * self.cfg.ipc, dtype=torch.long)
                for c in range(self.train_dataset.num_classes)
            ],
            dim=0,
        ).to(device)

        # --- J pyramid (unchanged from PhysicsPyramidDataset) ---
        pyramid_J = []
        res = 1
        while res <= self.cfg.pyramid_start_res:
            level = torch.randn((N, 3, res, res), device=device)
            if self.cfg.init_mode == "zero":
                level = level * 0
            pyramid_J.insert(0, level)
            res *= 2
            if res > self.cfg.syn_res:
                res = self.cfg.syn_res

        pyramid_J = [p / len(pyramid_J) for p in pyramid_J]
        for p in pyramid_J:
            p.requires_grad_(True)

        # --- Sea-Thru parameters ---
        # softplus(0) ≈ 0.693, so d ≈ 0.69 and beta ≈ 0.69 at init.
        # => T = exp(-0.69 * 0.69) ≈ 0.62 at init. Centered, no collapse.
        syn_d_logit = torch.zeros((N, 1, H, W), device=device, requires_grad=True)
        syn_beta_logit = torch.zeros((N, 3, 1, 1), device=device, requires_grad=True)

        # B: ambient background colour (sigmoid -> [0, 1])
        syn_B = torch.zeros((N, 3, 1, 1), device=device, requires_grad=True)

        return (pyramid_J, syn_d_logit, syn_beta_logit, syn_B), syn_labels

    def init_optimizer(self):
        lr_d = getattr(self.cfg, "lr_d", self.cfg.lr)
        lr_beta = getattr(self.cfg, "lr_beta", self.cfg.lr)
        lr_B = getattr(self.cfg, "lr_B", self.cfg.lr)
        param_groups = [
            {"params": self.pyramid_J, "lr": self.cfg.lr},
            {"params": [self.syn_d_logit], "lr": lr_d},
            {"params": [self.syn_beta_logit], "lr": lr_beta},
            {"params": [self.syn_B], "lr": lr_B},
        ]
        return torch.optim.Adam(param_groups)

    def decode_J(self) -> Tensor:
        result = torch.sum(
            torch.stack(
                [
                    F.interpolate(
                        p,
                        (self.cfg.syn_res, self.cfg.syn_res),
                        antialias=False,
                        mode="bilinear",
                    )
                    for p in self.pyramid_J
                ]
            ),
            dim=0,
        )

        if self.cfg.decorrelate_color:
            result = self.linear_decorrelate_color(result)

        return torch.sigmoid(2 * result)

    def decode_depth(self) -> Tensor:
        """d in [0, +inf), shape (N, 1, H, W)."""
        return F.softplus(self.syn_d_logit)

    def decode_beta(self) -> Tensor:
        """beta in [0, +inf), shape (N, 3, 1, 1)."""
        return F.softplus(self.syn_beta_logit)

    def decode_T(self) -> Tensor:
        """T = exp(-beta * d) in (0, 1], shape (N, 3, H, W) by broadcasting.
        -beta * d <= 0, so T <= 1 and T > 0 since exp never reaches 0.
        """

        d = self.decode_depth()       # (N, 1, H, W)
        beta = self.decode_beta()     # (N, 3, 1, 1)
        return torch.exp(-beta * d)   # (N, 3, H, W)

    def decode_B(self) -> Tensor:
        """B in [0, 1], shape (N, 3, 1, 1)."""
        return torch.sigmoid(self.syn_B)

    def extend_pyramid(self) -> bool:
        print("extending J pyramid...")

        old_len = len(self.pyramid_J)
        new_len = old_len + 1
        old_res = self.pyramid_J[0].shape[-1]

        if old_res == self.cfg.syn_res:
            print("already max res")
            return False

        new_res = min(old_res * 2, self.cfg.syn_res)
        print("new res: {}".format(new_res))

        num_images = self.pyramid_J[-1].shape[0]

        self.pyramid_J = [
            p.detach().clone() * old_len / new_len for p in self.pyramid_J
        ]
        if self.cfg.init_mode == "zero":
            new_layer = (
                torch.sum(
                    torch.stack(
                        [
                            F.interpolate(
                                p,
                                (new_res, new_res),
                                antialias=False,
                                mode="bilinear",
                            )
                            for p in self.pyramid_J
                        ]
                    ),
                    dim=0,
                )
                / old_len
            )
        else:
            new_layer = (
                torch.randn(
                    (num_images, 3, new_res, new_res),
                    device=DeviceSingleton.get(),
                )
                / new_len
            )

        self.pyramid_J.insert(0, new_layer)
        for p in self.pyramid_J:
            p.requires_grad_(True)

        self.optimizer = self.init_optimizer()
        return True

    def get_data(self) -> Tuple[Tensor, Tensor]:
        J = self.decode_J()              # (N, 3, H, W) in [0, 1]
        T = self.decode_T()              # (N, 3, H, W) in (0, 1]
        B = self.decode_B()              # (N, 3, 1, 1) in [0, 1]
        I = J * T + (1.0 - T) * B
        return I, self.syn_labels

    def get_to_save(self) -> dict:
        data = self.get_data()
        return {
            "syn_data": data,
            "save_dict": {
                "syn_J": self.decode_J(),
                "syn_T": self.decode_T(),
                "syn_d": self.decode_depth(),
                "syn_beta": self.decode_beta(),
                "syn_B": self.decode_B(),
            },
        }

    @torch.no_grad()
    def log_images(self, step: int = None):
        if len(self.pyramid_J[0]) > 100:
            print("Warning: too many images to log")
            return
        I, _ = self.get_data()
        log_images(syn_images=I.detach().clone(), step=step)

    def upkeep(self, step: int = None):
        if (step - 1) % self.cfg.pyramid_extent_it == 0 and step > 1:
            if self.extend_pyramid():
                self.log_images(step=step)

    def get_save_dict(self):
        return {
            "pyramid_J": self.pyramid_J,
            "d_logit": self.syn_d_logit,
            "beta_logit": self.syn_beta_logit,
            "B": self.syn_B,
            "opt_state": self.optimizer.state_dict(),
        }

    def load_from_dict(self, load_dict: dict):
        loaded_pyramid = load_dict["pyramid_J"]
        while len(self.pyramid_J) < len(loaded_pyramid):
            self.extend_pyramid()

        with torch.no_grad():
            for p, loaded_p in zip(self.pyramid_J, loaded_pyramid):
                p.copy_(loaded_p)
            self.syn_d_logit.copy_(load_dict["d_logit"])
            self.syn_beta_logit.copy_(load_dict["beta_logit"])
            self.syn_B.copy_(load_dict["B"])

        self.optimizer.load_state_dict(load_dict["opt_state"])
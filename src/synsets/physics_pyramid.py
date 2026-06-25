from typing import Tuple

import torch
from torch import Tensor

from config import DistillCfg
from data.dataloaders import BaseRealDataset
from my_utils.device import DeviceSingleton
from my_utils.log_utils import log_images

from .base import BaseDistilledDataset
import torch.nn.functional as F
from src.initializer.sample.medoids_init import MedoidInitializer
from src.initializer.priors.ppg_init import PPGInitializer
from src.initializer.sample.random_init import RandomInitializer

class PhysicsPyramidDataset(BaseDistilledDataset):
    """
    Atmospheric model (I = J*T + (1-T)*B) where J is encoded as a
    coarse-to-fine pyramid (like PyramidDataset) instead of a single
    full-resolution tensor.
    """

    def __init__(self, train_dataset: BaseRealDataset, cfg: DistillCfg, backbone: torch.nn.Module, num_feat: int, do_ppg_init: bool = False):
        super().__init__()
        self.train_dataset = train_dataset
        self.cfg = cfg
        self.do_ppg_init = do_ppg_init
        if self.do_ppg_init:
            if self.cfg.sample_init == "medoids":
                self.sample_init = MedoidInitializer(
                    self.cfg, self.train_dataset, backbone, num_feat, self.cfg.seed
                )
            elif self.cfg.sample_init == "random":
                self.sample_init = RandomInitializer(
                    self.cfg, self.train_dataset, seed=self.cfg.seed
                )
            
            self.ppg_init = PPGInitializer(self.cfg)

        (self.pyramid_J, self.syn_T, self.syn_B), self.syn_labels = self.init_synset()
        self.optimizer = self.init_optimizer()

    def init_synset(self):
        N = self.cfg.ipc * self.train_dataset.num_classes
        device = DeviceSingleton.get()

        if self.do_ppg_init:
            # 1) medoids/random en ordre classe-major (= ordre de syn_labels)
            sample_idx = self.sample_init.get_indices()

            ordered_idx, label_list = [], []
            for c in range(self.train_dataset.num_classes):
                idxs = sample_idx[c]
                if len(idxs) < self.cfg.ipc:                # classe trop petite -> on repete
                    idxs = (idxs * self.cfg.ipc)[:self.cfg.ipc]
                ordered_idx += idxs[:self.cfg.ipc]
                label_list  += [c] * self.cfg.ipc
            syn_labels = torch.tensor(label_list, dtype=torch.long, device=device)

            # 2) PPG sur chaque medoid -> inits T, B
            T_list, B_list = [], []
            for i in ordered_idx:
                path = self.train_dataset.get_path(i)
                img  = self.ppg_init.load_img(path)         # [1,3,256,256] en [-1,1]
                syn_T_i, syn_B_i = self.ppg_init.run_physical_model(img)
                T_list.append(syn_T_i); B_list.append(syn_B_i)

            syn_T = torch.cat(T_list, 0).detach().clone().requires_grad_(True)   # [N,1,H,W]
            syn_B = torch.cat(B_list, 0).detach().clone().requires_grad_(True)   # [N,3,1,1]
        else:
            # init aléatoire (T proche de 1 via logit=0, B sombre)
            H = W = self.cfg.syn_res
            syn_labels = torch.cat(
                [
                    torch.tensor([c] * self.cfg.ipc, dtype=torch.long)
                    for c in range(self.train_dataset.num_classes)
                ],
                dim=0,
            ).to(device)
            syn_T = torch.full((N, 1, H, W), 0.0, device=device, requires_grad=True)
            syn_B = torch.zeros(N, 3, 1, 1, device=device, requires_grad=True)

        # 3) pyramide J (inchangee)
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

        return (pyramid_J, syn_T, syn_B), syn_labels

    def init_optimizer(self):
        lr_T = getattr(self.cfg, "lr_T", self.cfg.lr)
        lr_B = getattr(self.cfg, "lr_B", self.cfg.lr)
        param_groups = [
            {"params": self.pyramid_J, "lr": self.cfg.lr},
            {"params": [self.syn_T], "lr": lr_T},
            {"params": [self.syn_B], "lr": lr_B},
        ]
        return torch.optim.Adam(param_groups)

    def decode_J(self, n_levels: int = None) -> Tensor:
        # n_levels=None -> tous les niveaux (comportement inchangé)
        # n_levels=k    -> seulement les k niveaux les plus grossiers
        levels = self.pyramid_J if n_levels is None else self.pyramid_J[-n_levels:]
        result = torch.sum(
            torch.stack(
                [
                    F.interpolate(
                        p,
                        (self.cfg.syn_res, self.cfg.syn_res),
                        antialias=False,
                        mode="bilinear",
                    )
                    for p in levels
                ]
            ),
            dim=0,
        )

        if self.cfg.decorrelate_color:
            result = self.linear_decorrelate_color(result)

        return torch.sigmoid(2 * result)
    
    @torch.no_grad()
    def get_snapshot(self) -> dict:
        T = torch.sigmoid(self.syn_T)
        B = torch.sigmoid(self.syn_B)
        J_levels = [self.decode_J(n_levels=k) for k in range(1, len(self.pyramid_J) + 1)]
        J = J_levels[-1]
        I = J * T + (1.0 - T) * B
        return {
            "I": I.cpu(),
            "J": J.cpu(),
            "T": T.cpu(),
            "B": B.cpu(),
            "J_levels": [j.cpu() for j in J_levels],
            "level_res": [p.shape[-1] for p in reversed(self.pyramid_J)],
        }

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

        self.pyramid_J = [p.detach().clone() * old_len / new_len for p in self.pyramid_J]
        if self.cfg.init_mode == "zero":
            new_layer = torch.sum(
                torch.stack(
                    [
                        torch.nn.functional.interpolate(
                            p, (new_res, new_res), antialias=False, mode="bilinear"
                        )
                        for p in self.pyramid_J
                    ]
                ),
                dim=0,
            ) / old_len
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
        J = self.decode_J()
        T = torch.sigmoid(self.syn_T)
        B = torch.sigmoid(self.syn_B)
        I = J * T + (1.0 - T) * B
        return I, self.syn_labels
    
    def get_to_save(self) -> dict:
        data = self.get_data()
        return {
            "syn_data": data,
            "save_dict": {
                "syn_J": self.decode_J(),
                "syn_T": self.syn_T,
                "syn_B": self.syn_B
            }
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
            "T": self.syn_T,
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
            self.syn_T.copy_(load_dict["T"])
            self.syn_B.copy_(load_dict["B"])

        self.optimizer.load_state_dict(load_dict["opt_state"])

from typing import Tuple

import torch
from torch import Tensor

from config import DistillCfg
from data.dataloaders import BaseRealDataset
from my_utils.device import DeviceSingleton
from my_utils.log_utils import log_images

from .base import BaseDistilledDataset


class PhysicsDataset(BaseDistilledDataset):
    """
    Paramétrisation physique : I = J * T + (1 - T) * B
    Les paramètres optimisés sont (J, T, B), pas I directement.
    I est re-rendu différentiablement à chaque get_data().
    """

    def __init__(self, train_dataset: BaseRealDataset, cfg: DistillCfg):
        super().__init__()
        self.train_dataset = train_dataset
        self.cfg = cfg

        (self.syn_J, self.syn_T, self.syn_B), self.syn_labels = self.init_synset()
        self.optimizer = self.init_optimizer()

    def init_synset(self) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]:
        N = self.cfg.ipc * self.train_dataset.num_classes
        H = W = self.cfg.syn_res
        device = DeviceSingleton.get()

        syn_labels = torch.cat(
            [torch.tensor([c] * self.cfg.ipc, dtype=torch.long)
             for c in range(self.train_dataset.num_classes)],
            dim=0,
        ).to(device)

        if self.cfg.init_mode == "real":
            # init J depuis des images réelles, T et B neutres
            reals = self.train_dataset.get_random_reals(self.cfg.ipc).to(device)
            # inverse-sigmoid pour que sigmoid(2*J) ≈ reals
            reals = reals.clamp(1e-4, 1 - 1e-4)
            syn_J = (0.5 * torch.log(reals / (1 - reals))).detach()
            # T initialisé à ~1 (eau claire) → logit grand positif
            syn_T = torch.full((N, 1, H, W), 2.0, device=device)
            # B initialisé à la couleur moyenne de l'eau (à toi de fixer un prior)
            syn_B = torch.zeros(N, 3, 1, 1, device=device)
        else:
            syn_J = torch.randn(N, 3, H, W, device=device)
            syn_T = torch.full((N, 1, H, W), 2.0, device=device)  # logit
            syn_B = torch.zeros(N, 3, 1, 1, device=device)

        syn_J.requires_grad_(True)
        syn_T.requires_grad_(True)
        syn_B.requires_grad_(True)

        return (syn_J, syn_T, syn_B), syn_labels

    def init_optimizer(self):
        # lr séparés : T et B bougent typiquement bien plus lentement que J
        param_groups = [
            {"params": [self.syn_J], "lr": self.cfg.lr},
            {"params": [self.syn_T], "lr": getattr(self.cfg, "lr_T", self.cfg.lr)},
            {"params": [self.syn_B], "lr": getattr(self.cfg, "lr_B", self.cfg.lr)},
        ]
        return torch.optim.Adam(param_groups)

    def get_data(self) -> Tuple[Tensor, Tensor]:
        # J ∈ [0,1] via la même reparam sigmoid que PixelDataset
        J = self.syn_J
        if self.cfg.decorrelate_color:
            J = self.linear_decorrelate_color(J)
        J = torch.sigmoid(2 * J)

        # T ∈ [0,1], broadcast sur les 3 canaux
        T = torch.sigmoid(self.syn_T)

        # B ∈ [0,1], une couleur ambiante par image (broadcast spatial)
        B = torch.sigmoid(self.syn_B)

        # Rendu Koschmieder différentiable
        I = J * T + (1.0 - T) * B
        return I, self.syn_labels
    
    def get_to_save(self) -> dict:
        data = self.get_data()
        save_dict = self.get_save_dict()
        return {
            "syn_lr": self.syn_lr,
            "syn_data": data,
            "save_dict": save_dict
        }

    def log_images(self, step: int = None):
        with torch.no_grad():
            I, _ = self.get_data()
            log_images(syn_images=I.detach().clone(), step=step)

    def get_save_dict(self):
        return {
            "J": self.syn_J,
            "T": self.syn_T,
            "B": self.syn_B,
            "opt_state": self.optimizer.state_dict(),
        }

    def load_from_dict(self, load_dict: dict):
        with torch.no_grad():
            self.syn_J.copy_(load_dict["J"])
            self.syn_T.copy_(load_dict["T"])
            self.syn_B.copy_(load_dict["B"])
        self.optimizer.load_state_dict(load_dict["opt_state"])
import torch
from torch import Tensor

from config import DistillCfg
from my_utils.device import DeviceSingleton
from .base_prior_init import BasePriorInitializer


class NonePriorInitializer(BasePriorInitializer):
    """
    Priors triviaux, sans modèle physique : T proche de 1 (dégradation
    minimale, I ≈ J), B sombre. Reproduit le comportement de l'ancien
    PhysicsPyramidDataset sans init PPG. get_priors ignore l'image.
    """

    def __init__(self, cfg: DistillCfg):
        self.cfg = cfg
        self.device = DeviceSingleton.get()

    def load_img(self, path: str) -> Tensor:
        # aucune image n'est lue ; on renvoie un placeholder ignoré par get_priors
        return torch.empty(0)

    @torch.no_grad()
    def get_priors(self, image) -> dict:
        """
        Priors constants T = B = 0.5 (logit 0 -> sigmoid 0.5). image est ignoré.
        """
        H = W = self.cfg.syn_res
        T = torch.full((1, 1, H, W), 0.5, device=self.device)
        B = torch.full((1, 3, 1, 1), 0.5, device=self.device)
        return {"T": T, "B": B}

    @staticmethod
    def compose(J, T, B):
        return J * T + (1.0 - T) * B
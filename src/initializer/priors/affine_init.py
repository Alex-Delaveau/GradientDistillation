import torch
from torch import Tensor

from config import DistillCfg
from my_utils.device import DeviceSingleton
from .base_prior_init import BasePriorInitializer


class AffinePriorInitializer(BasePriorInitializer):
    """
    Contrôle NON physique (camera-ready, réponse aux reviewers).

    Reparamétrisation affine générique de l'image distillée :

        I = J · G + O

    avec G un gain spatial [N,1,H,W] (stocké dans le slot T) et O un offset
    global par canal [N,3,1,1] (stocké dans le slot B), tous deux passés par
    une sigmoïde comme T/B. Contrairement à l'Eq. (1) du papier
    (I = J·T + (1-T)·B), il n'y a PAS de couplage convexe (1 - T) entre le
    gain et l'offset : aucune contrainte issue du modèle de formation sous-marine.

    Les priors sont constants et choisis pour que l'image initiale soit
    EXACTEMENT celle du prior `none` (T = B = 0.5) :

        J · 0.5 + 0.25  ==  J · 0.5 + (1 - 0.5) · 0.5

    Capacité identique à `none` avec t_channels=1 (mêmes formes de paramètres),
    même image initiale ; seuls les gradients diffèrent
    (dI/dG = J, dI/dO = 1, contre dI/dT = J - B, dI/dB = 1 - T).
    Ce contrôle isole donc l'effet du prior physique de celui de la seule
    reparamétrisation. get_priors ignore l'image.
    """

    GAIN_INIT = 0.5
    OFFSET_INIT = 0.25

    def __init__(self, cfg: DistillCfg):
        self.cfg = cfg
        self.device = DeviceSingleton.get()

    def load_img(self, path: str) -> Tensor:
        # aucune image n'est lue ; placeholder ignoré par get_priors
        return torch.empty(0)

    @torch.no_grad()
    def get_priors(self, image) -> dict:
        """Gain G = 0.5 (slot T, [1,1,H,W]) et offset O = 0.25 (slot B, [1,3,1,1])."""
        H = W = self.cfg.syn_res
        G = torch.full((1, 1, H, W), self.GAIN_INIT, device=self.device)
        O = torch.full((1, 3, 1, 1), self.OFFSET_INIT, device=self.device)
        return {"T": G, "B": O}

    @staticmethod
    def compose(J, T, B):
        # T = gain G, B = offset O ; pas de terme (1 - T)
        return J * T + B

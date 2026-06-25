from ppg.core.PPGB import PM
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

from config import DistillCfg
from my_utils.device import DeviceSingleton
from .base_prior_init import BasePriorInitializer


class PPGInitializer(BasePriorInitializer):

    def __init__(self, cfg: DistillCfg):
        self.cfg = cfg
        self.physical_model = self._load_physical_model(
            self.cfg.ppg_input_channels, self.cfg.ppg_checkpoint_path
        )

    def _load_physical_model(self, input_channels, checkpoint_path):
        model = PM(input_channels=input_channels)
        ckpt = torch.load(checkpoint_path,
                          map_location=DeviceSingleton.get(), weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(DeviceSingleton.get())
        return model

    def load_img(self, path: str) -> torch.Tensor:
        """PNG -> tensor [1,3,256,256] dans [-1,1]."""
        pil = Image.open(path).convert("RGB").resize((256, 256), Image.BICUBIC)
        arr = np.array(pil, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float().to(DeviceSingleton.get())

    @torch.no_grad()
    def get_priors(self, image: torch.Tensor) -> dict:
        """
        image : [1,3,256,256] en [-1,1] (sortie de load_img)
        -> {"T":[1,1,h,w], "B":[1,3,h,w]} en [0,1], résolution native.
        Le dataset fait interpolation -> syn_res, reduction de canaux/spatial, logit.
        """
        p_a, p_t = self.physical_model(image)
        T = p_t.clamp(0, 1)                  # [1,1,h,w]  (suppose p_t en [0,1])
        B = ((p_a + 1) / 2).clamp(0, 1)      # [1,3,h,w]  (p_a en [-1,1])
        return {"T": T, "B": B}

    @staticmethod
    def compose(J, T, B):
        return J * T + (1.0 - T) * B          # Koschmieder
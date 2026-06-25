import os
import numpy as np
import torch
from PIL import Image
from .base_prior_init import BasePriorInitializer 

from slurpp import load_stage1
from slurpp.io import normalize_imgs
from utils.config_util import recursive_load_config


class SLURPPInitializer(BasePriorInitializer):
    """
    Loads SLURPP once and exposes get_priors() for repeated inference.

    Args:
        slurpp_root:      path to the slurpp_lib directory (or SLURPP/slurpp/)
        checkpoint_path:  path to the SLURPP fine-tuned checkpoint directory
        device:           "cuda" or "cpu"
    """

    def __init__(
        self,
        slurpp_root: str,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.dual = True

        base_ckpt_dir = os.environ.get("BASE_CKPT_DIR")
        if base_ckpt_dir is None:
            raise EnvironmentError("BASE_CKPT_DIR environment variable is not set.")
        model_path = os.path.join(base_ckpt_dir, "stable-diffusion-2")

        config_name = "dual.yaml"
        config_path = os.path.join(slurpp_root, "config", config_name)
        cfg = recursive_load_config(config_path)

        self.pipe, self.inputs_fields, self.outputs_fields, _ = load_stage1(
            model_path, checkpoint_path, cfg
        )

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except (ImportError, NotImplementedError, RuntimeError):
            pass

        self.pipe = self.pipe.to(self.device)

        # Set one-step scheduler spacing
        self.pipe.scheduler.config.timestep_spacing = "trailing"

        # Build field → output index mapping
        self.field_idx = {f: i for i, f in enumerate(self.outputs_fields)}

    def _to_tensor(self, image) -> torch.Tensor:
        """Converts PIL / numpy / tensor to [1, 3, H, W] float32 in [0, 1]."""
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image.float().clamp(0, 1)

    @torch.no_grad()
    def get_priors(self, image) -> dict:
        """
        Run SLURPP on a single image.

        Args:
            image: PIL.Image | np.ndarray [H, W, 3] uint8 | torch.Tensor [1, 3, H, W] float [0, 1]
                   H and W must be multiples of 8. Recommended: 512x512.

        Returns:
            Dual mode:     {"J": Tensor [1,3,H,W], "T": Tensor [1,3,H,W], "B": Tensor [1,3,H,W]}
            All tensors are float32 in [0, 1] on CPU.
        """
        tensor = self._to_tensor(image).to(self.device)
        normalized = normalize_imgs(tensor, device=self.device)

        # Dual mode feeds the same image to both UNets
        inputs = [normalized] * len(self.inputs_fields)

        output = self.pipe(
            inputs,
            denoising_steps=1,
            show_progress_bar=False,
            is_dual=self.dual,
        )

        result = {}
        if "ill" in self.field_idx:
            result["T"] = output[self.field_idx["ill"]:self.field_idx["ill"] + 1].clamp(0, 1).cpu()
        if "bc" in self.field_idx:
            result["B"] = output[self.field_idx["bc"]:self.field_idx["bc"] + 1].clamp(0, 1).cpu()
        if "clear" in self.field_idx:
            result["J"] = output[self.field_idx["clear"]:self.field_idx["clear"] + 1].clamp(0, 1).cpu()
        return result
    
    def load_img(self, path: str) -> torch.Tensor:
        """path -> tensor [1,3,512,512] en [0,1]."""
        pil = Image.open(path).convert("RGB").resize((512, 512), Image.BICUBIC)
        arr = np.array(pil, dtype=np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()

    @staticmethod
    def compose(J, T, B):
        return J * T + B
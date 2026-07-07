from .none_init import NonePriorInitializer
from .ppg_init import PPGInitializer
from .slurpp_init import SLURPPInitializer
import torch
from torch import Tensor
import torch.nn.functional as F

def build_prior(cfg, device):
    """"
    Build the prior initializer based on the configuration.
    """
    if cfg.prior_init == "none":
        return NonePriorInitializer(cfg)
    if cfg.prior_init == "ppg":
        return PPGInitializer(cfg)
    if cfg.prior_init == "slurpp":
        return SLURPPInitializer(cfg.slurpp_root, cfg.slurpp_checkpoint_path, device=device)
    raise ValueError(f"prior_init unknown: {cfg.prior_init}")

def compute_TB_priors(cfg, train_dataset, prior_init, sample_indices, device):
    """Selected samples -> (T01, B01) in [0,1], shape [N, t_channels|3, syn_res, syn_res]."""
    T_list, B_list = [], []
    for i in sample_indices:
        priors = prior_init.get_priors_from_path(train_dataset.get_path(i))
        T_list.append(_prep_prior(priors["T"], cfg.t_channels, spatial=True, syn_res=cfg.syn_res, device=device))
        B_list.append(_prep_prior(priors["B"], channels=3, spatial=cfg.b_spatial, syn_res=cfg.syn_res, device=device))
    return torch.cat(T_list, 0), torch.cat(B_list, 0)

def _prep_prior(x: Tensor, channels: int, spatial: bool, syn_res, device) -> Tensor:
    """native [0,1] -> [1, channels, syn_res, syn_res] (or [...,1,1] if not spatial)."""
    x = x.to(device).float()
    x = F.interpolate(x, (syn_res, syn_res),
                        mode="bilinear", align_corners=False)
    if x.shape[1] == 1 and channels == 3:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] == 3 and channels == 1:
        x = x.mean(1, keepdim=True)
    if not spatial:
        x = x.mean(dim=(2, 3), keepdim=True)
    return x.clamp(1e-4, 1 - 1e-4)


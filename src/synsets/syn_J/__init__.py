from .pixel import PixelJ
from .pyramid import PyramidJ

def build_syn_J(cfg, N, device):
    if cfg.distill_mode == "pixel":
        return PixelJ(cfg, N, device)
    if cfg.distill_mode == "pyramid":
        return PyramidJ(cfg, N, device)
    raise ValueError(f"distill_mode unknown: {cfg.distill_mode}")
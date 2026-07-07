from einops import rearrange
import torch
from torch import Tensor
from my_utils.device import DeviceSingleton



color_correlation_svd_sqrt = torch.tensor(
        [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
    ).to(DeviceSingleton.get())

max_norm_svd_sqrt = torch.max(
        torch.linalg.norm(color_correlation_svd_sqrt, axis=0)
    )

def linear_decorrelate_color(im: Tensor):
        b, c, h, w = im.shape
        im = rearrange(im, "b c h w -> (b h w) c")
        color_correlation_normalized = (
            color_correlation_svd_sqrt / max_norm_svd_sqrt
        )
        im = im @ color_correlation_normalized.T
        im = rearrange(im, "(b h w) c -> b c h w", h=h, w=w)
        return im
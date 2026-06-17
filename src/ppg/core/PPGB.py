import torch
from torch import nn
import torchvision.transforms.functional as TF
from .ConditionNet import ConditionNet

def get_A(x):
    h, w = x.shape[2], x.shape[3]
    sigma = float(h + w) / 2.0   # matches PIL GaussianBlur(radius=...)
    ks = (min(h, w) - 1) | 1     # largest odd integer <= min(h, w)
    return TF.gaussian_blur(x, kernel_size=[ks, ks], sigma=sigma)

class TNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.aNet = ConditionNet(support_size=input_channels)
        self.final = nn.Conv2d(128, output_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        a = self.final(self.aNet(x))
        return a
 
class PM(nn.Module):
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.tNet = TNet(input_channels,1)
        self.aNet = TNet(input_channels,3)

    def forward(self, x): 
        a = self.aNet(get_A(x)+x)
        t = self.tNet(x)
        return a, t
    

from config import DistillCfg
from torch.utils.data import Dataset, DataLoader
from my_utils.device import DeviceSingleton
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import kmedoids


class MedoidInitializer():

    def __init__(self, cfg: DistillCfg, train_dataset: Dataset, backbone: nn.Module, num_feat: int):
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=8)
        self.backbone = self.init_backbone(backbone)
        self.num_feat = num_feat
        

    def init_backbone(self, backbone: nn.Module) -> nn.Module:
        """Set the backbone to eval mode and freeze its parameters."""
        backbone = backbone.to(DeviceSingleton.get())
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
        return backbone
    
    def compute_features(self) -> torch.Tensor:
        """Compute features for all images in the training dataset using the backbone."""
        feats, labels = [], []
        with torch.no_grad():
            for x, y in self.train_loader:
                z = self.backbone(x.to(DeviceSingleton.get()))
                feats.append(z.cpu())
                labels.append(y)

        features = torch.cat(feats)
        labels   = torch.cat(labels)
        features = F.normalize(features, dim=1)
        return features, labels

    def find_medoids(self, features: torch.Tensor, labels: torch.Tensor) -> dict:
        features_np = features.numpy()
        labels_np   = labels.numpy()

        medoid_global_idx = {}
        for c in range(self.train_loader.dataset.num_classes):
            cls_idx = np.where(labels_np == c)[0]
            Fc = features_np[cls_idx]
            k  = min(self.cfg.ipc, len(cls_idx))

            D = np.clip(1.0 - Fc @ Fc.T, 0.0, 2.0)
            np.fill_diagonal(D, 0.0)

            res = kmedoids.fasterpam(D, k) 
            medoid_global_idx[c] = cls_idx[res.medoids].tolist()
        
        return medoid_global_idx
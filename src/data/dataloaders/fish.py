


import torch
from torchvision import transforms
import torchvision.datasets as tv_datasets

from torchvision.datasets.folder import default_loader
from typing import Literal
from data.dataloaders.base import BaseRealDataset

from torch.utils.data import DataLoader, random_split
from my_utils.device import DeviceSingleton
from torch import Tensor

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

class Fish4Knowledge(BaseRealDataset):

    NUM_CLASSES = 23
    TRAIN_RATIO = 0.8

    def __init__(
    self,
    split: str = "train",
    res: int = 126,
    crop_res: int = 126,
    crop_mode: Literal["center", "random"] = "center",
    data_root: str = "data/datasets",
    seed: int = 42,
    ):
        
        super().__init__()

        self.num_classes = self.NUM_CLASSES

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose(
            [
                transforms.Resize(res),
                (
                    transforms.CenterCrop(crop_res)
                    if crop_mode == "center"
                    else transforms.RandomCrop(crop_res)
                ),
                transforms.ToTensor(),
            ]
        )

        self.mean = torch.tensor(mean, device=DeviceSingleton.get()).reshape(1, 3, 1, 1)
        self.std = torch.tensor(std, device=DeviceSingleton.get()).reshape(1, 3, 1, 1)

        root = f"{data_root}/fish4knowledge"
        full_ds = tv_datasets.ImageFolder(
            root,
            transform=self.transform,
            target_transform=None,
            loader=default_loader,
            is_valid_file=None,
        )

        self.class_names = full_ds.classes  # derives from folder names
        assert len(self.class_names) == self.NUM_CLASSES, (
            f"Expected {self.NUM_CLASSES} classes, found {len(self.class_names)}. "
            f"Check your directory structure at {root}."
        )

        # Stratified train/test split (preserves per-class proportions)
        targets_all = np.asarray(full_ds.targets)
        splitter = StratifiedShuffleSplit(
            n_splits=1, train_size=self.TRAIN_RATIO, random_state=seed
        )
        train_idx, test_idx = next(
            splitter.split(np.zeros(len(targets_all)), targets_all)
        )
        split_indices = train_idx if split == "train" else test_idx
        split_indices = split_indices.tolist()

        self.full_ds = full_ds
        self.ds = torch.utils.data.Subset(full_ds, split_indices)

        # Materialise targets for the active split (needed by get_single_class)
        self.targets = torch.tensor(
            [full_ds.targets[i] for i in self.ds.indices]
        )

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        image, label = self.ds.__getitem__(index)
        return image, label

    def get_single_class(self, cls: int) -> Tensor:
        # self.ds.indices : List[int] -> np.ndarray pour pouvoir masquer
        split_indices = np.asarray(self.ds.indices)
        mask = (self.targets == cls).numpy()
        indices = split_indices[mask].tolist()

        subset = torch.utils.data.Subset(self.full_ds, indices)

        loader = DataLoader(subset, batch_size=64, num_workers=8)
        images, labels = [], []
        for x, y in loader:
            images.append(x)
            labels.append(y)

        images = torch.cat(images)
        labels = torch.cat(labels)

        # Garde-fou : vérifie que tout est bien de la classe demandée
        assert (labels == cls).all(), (
            f"get_single_class({cls}) a renvoyé des labels {labels.unique().tolist()}"
        )
        return images
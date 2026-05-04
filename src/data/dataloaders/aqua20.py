

import copy
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

class Aqua20(BaseRealDataset):

    NUM_CLASSES = 20

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


        root = f"{data_root}/data/aqua20"

        train_ds = tv_datasets.ImageFolder(
            f"{root}/train",
            transform=self.transform,
            target_transform=None,
            loader=default_loader,
            is_valid_file=None,
        )


        test_ds = tv_datasets.ImageFolder(
            f"{root}/test",
            transform=self.transform,
            target_transform=None,
            loader=default_loader,
            is_valid_file=None,
        )

        self.class_names = train_ds.classes  # derives from folder names
        assert len(self.class_names) == self.NUM_CLASSES, (
            f"Expected {self.NUM_CLASSES} classes, found {len(self.class_names)}. "
            f"Check your directory structure at {root}."
        )

        self.full_ds = train_ds if split == "train" else test_ds

        self.ds = copy.deepcopy(self.full_ds)

        self.targets = torch.tensor(self.ds.targets)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        image, label = self.ds.__getitem__(index)
        return image, label

    def get_single_class(self, cls: int) -> Tensor:
        copy_ds = copy.deepcopy(self.full_ds)
        copy_ds.samples = [s for s in copy_ds.samples if s[1] == cls]
        copy_ds.targets = [s[1] for s in copy_ds.samples]

        num_samples = len(copy_ds.samples)
        loader = DataLoader(copy_ds, batch_size=64, num_workers=8)
        images = []
        labels = []
        print(f"Loading all {num_samples} images for class {cls}...")
        for x, y in loader:
            images.append(x)
            labels.append(y)
        images = torch.cat(images)
        labels = torch.cat(labels)
        print("Done.")

        return images
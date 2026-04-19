import os

import kornia
import torch
from torch import Tensor, nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from tqdm import tqdm

from config import CentroidRealsCfg
from data.dataloaders import (
    BaseRealDataset,
    get_dataset,
)
from models import get_model

from my_utils import DeviceSingleton

def _build_metrics(num_classes: int, do_f1: bool):
    """Builds top-1 / top-5 metrics, mirroring eval.py's logic."""
    device = DeviceSingleton.get()

    if do_f1:
        top1_metric = MulticlassF1Score(
            average="micro", num_classes=num_classes
        ).to(device)
        top5_metric = (
            MulticlassF1Score(average="micro", num_classes=num_classes, top_k=5).to(device)
            if num_classes >= 5 else None
        )
    else:
        top1_metric = MulticlassAccuracy(
            average="micro", num_classes=num_classes, top_k=1
        ).to(device)
        top5_metric = (
            MulticlassAccuracy(average="micro", num_classes=num_classes, top_k=5).to(device)
            if num_classes >= 5 else None
        )
    return top1_metric, top5_metric


@torch.no_grad()
def get_centroid_images(
    labels: Tensor,
    model: nn.Module,
    train_dataset: BaseRealDataset,
) -> Tensor:
    crop = kornia.augmentation.CenterCrop(224)
    real_neighbors = []
    for y in tqdm(labels):
        real_images = train_dataset.get_single_class(y.item()).to(DeviceSingleton.get())

        normalized_real_images = train_dataset.normalize(real_images)
        cropped_real_images = crop(normalized_real_images)

        real_embeddings = torch.cat(
            [model(chunk) for chunk in torch.split(cropped_real_images, 100)]
        )
        mean_embedding = torch.mean(real_embeddings, dim=0, keepdim=True)

        distances = torch.norm(real_embeddings - mean_embedding, dim=1)

        nearest_idx = torch.argmin(distances)
        nearest_image = real_images[nearest_idx].clone()

        real_neighbors.append(nearest_image)

    real_neighbors = torch.stack(real_neighbors)

    return real_neighbors


def main(cfg: CentroidRealsCfg):
    save_directory = os.path.join(
        "logged_files", "real_centroids", cfg.dataset, cfg.model, "run"
    )
    save_file = os.path.join(save_directory, "data.pth")
    if os.path.exists(save_file) and cfg.skip_if_exists:
        print("This eval already done.")
        print("Exiting...")
        exit()

    train_dataset, test_dataset = get_dataset(
        name=cfg.dataset,
        res=cfg.real_res,
        crop_res=cfg.crop_res,
        train_crop_mode="center",
        data_root=cfg.data_root,
    )

    eval_model, num_feats = get_model(
        cfg.model, distributed=DeviceSingleton.is_distributed()
    )

    labels = torch.cat(
        [
            torch.tensor([c] * 1, dtype=torch.long)
            for c in range(train_dataset.num_classes)
        ],
        dim=0,
    ).to(DeviceSingleton.get())

    real_neighbors = get_centroid_images(
        labels=labels, model=eval_model, train_dataset=train_dataset
    )

    os.makedirs(save_directory, exist_ok=True)

    save_dict = {
        "images": real_neighbors.cpu(),
        "labels": labels.cpu(),
    }
    torch.save(save_dict, os.path.join(save_directory, "data.pth"))


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    cfg = CentroidRealsCfg(explicit_bool=True).parse_args()
    main(cfg)

"""Where we put cataract dataset for classification"""

import json
import pathlib
import random
from typing import Callable

import albumentations as A
import cv2
import einops
import torch
from omegaconf import DictConfig
from torch.utils import data as torch_data
from torch.utils.data import DataLoader

from utils.helpers import instantiate_from_config


def basic_preprocess_image(image_path: str, max_side: int = 256):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load the image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (max_side, max_side), interpolation=cv2.INTER_AREA)
    return image


def prepare_data(
    image_path: pathlib.Path,
    phase: str,
    class_to_idx: dict,
    max_side: int = 64,
    transform_fn: Callable | None = None,
):
    label = class_to_idx[phase]
    image = basic_preprocess_image(str(image_path), max_side)
    if transform_fn is not None:
        image = transform_fn(image=image)["image"]
    image = einops.rearrange(image, "h w c -> c h w")
    return image, torch.tensor(label, dtype=torch.long), str(image_path)


class ImageNumbersDataset(torch_data.Dataset):
    def __init__(
        self,
        root: str,
        class_names: list[str],
        max_side: int = 256,
        transform_fn: Callable | None = None,
        seed: int = 42,
        others_classes: list[str] | None = None,
    ):
        super().__init__()
        self.root = pathlib.Path(root)
        self.max_side = max_side
        self.transform_fn = transform_fn
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # Remap des phases rares vers "Others"
        self.others_remap = {cls: "Others" for cls in (others_classes or [])}

        labels_json_path = self.root.parent / "labels.json"
        with open(labels_json_path) as f:
            all_labels = json.load(f)

        split_name = self.root.name  # "train", "val" ou "test"
        self.path_to_phase = {
            k: v for k, v in all_labels.items()
            if k.startswith(split_name + "/")
        }

        self.all_path_images = [self.root.parent / k for k in self.path_to_phase]
        random.Random(seed).shuffle(self.all_path_images)

    def _get_phase(self, path: pathlib.Path) -> str:
        rel_key = str(path.relative_to(self.root.parent)).replace("\\", "/")
        phase = self.path_to_phase[rel_key]
        return self.others_remap.get(phase, phase)

    def __len__(self) -> int:
        return len(self.all_path_images)

    def __getitem__(self, index: int):
        path = self.all_path_images[index]
        phase = self._get_phase(path)
        image, label, path_str = prepare_data(
            path,
            phase=phase,
            class_to_idx=self.class_to_idx,
            max_side=self.max_side,
            transform_fn=self.transform_fn,
        )
        return image, label, path_str


def build_transforms(transforms_config: DictConfig | dict | None) -> Callable | None:
    if not transforms_config:
        return None
    aug_list = [instantiate_from_config(cfg) for cfg in transforms_config.values()]
    return A.Compose(aug_list)


def build_weighted_sampler(dataset: ImageNumbersDataset) -> torch.utils.data.WeightedRandomSampler:
    """Sampler qui équilibre les classes en sur-échantillonnant les rares."""
    num_classes = len(dataset.class_to_idx)
    counts = torch.zeros(num_classes)
    for path in dataset.all_path_images:
        counts[dataset.class_to_idx[dataset._get_phase(path)]] += 1

    class_weights = 1.0 / counts.clamp(min=1)
    sample_weights = torch.tensor([
        class_weights[dataset.class_to_idx[dataset._get_phase(p)]].item()
        for p in dataset.all_path_images
    ])
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


def compute_class_weights(dataset: ImageNumbersDataset) -> torch.Tensor:
    """Retourne un tenseur de weights inversement proportionnel à la fréquence de chaque classe."""
    num_classes = len(dataset.class_to_idx)
    counts = torch.zeros(num_classes)
    for path in dataset.all_path_images:
        counts[dataset.class_to_idx[dataset._get_phase(path)]] += 1
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights


def instantiate_dataloader(
    split_config: DictConfig | dict,
    class_names: list[str],
    others_classes: list[str] | None = None,
    use_sampler: bool = False,
) -> DataLoader:
    transform_fn = build_transforms(split_config.get("transforms"))
    dataset = ImageNumbersDataset(
        **split_config["params"],
        class_names=class_names,
        transform_fn=transform_fn,
        others_classes=others_classes,
    )
    loader_params = dict(split_config["loader_params"])
    if use_sampler:
        loader_params["sampler"] = build_weighted_sampler(dataset)
        loader_params.pop("shuffle", None)
    return DataLoader(dataset, **loader_params)


def instantiate_loaders(
    dataset_dict: DictConfig | dict,
) -> tuple[DataLoader, DataLoader]:
    class_names = list(dataset_dict["class_names"])
    others_classes = list(dataset_dict.get("others_classes") or [])
    train_loader = instantiate_dataloader(dataset_dict["train"], class_names, others_classes, use_sampler=True)
    val_loader = instantiate_dataloader(dataset_dict["val"], class_names, others_classes, use_sampler=False)
    return train_loader, val_loader

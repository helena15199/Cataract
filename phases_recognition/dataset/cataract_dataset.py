"""Where we put cataract dataset for classification"""

import pathlib
import random
from typing import Callable

import cv2
import einops
import torch
from torch.utils import data as torch_data


def basic_preprocess_image(
    image_path: str,
    max_side: int = 256,
):
    """
    Resize image to a square (max_side x max_side).
    """

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load the image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (max_side, max_side), interpolation=cv2.INTER_AREA)

    return image


def prepare_data(
    image_path: pathlib.Path,
    max_side: int = 64,
    transform_fn: Callable | None = None,
    do_get_label: bool = True,
):
    label = -1
    if do_get_label:
        label = int(image_path.parent.name)

    image = basic_preprocess_image(str(image_path), max_side)

    if transform_fn is not None:
        image = transform_fn(image=image)["image"]

    image = torch.from_numpy(image).float() / 255.0
    image = einops.rearrange(image, "h w c -> c h w")

    label = torch.tensor(label, dtype=torch.long)

    return image, label, str(image_path)


class ImageNumbersDataset(torch_data.Dataset):
    def __init__(
        self,
        root: str,
        max_side: int = 256,
        transform_fn: Callable | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.root = pathlib.Path(root)
        self.max_side = max_side
        self.transform_fn = transform_fn

        self.all_path_images = list(self.root.glob("*/*.jpg"))
        random.Random(seed).shuffle(self.all_path_images)

    def __len__(self) -> int:
        return len(self.all_path_images)

    def __getitem__(self, index: int):
        path = self.all_path_images[index]
        image, label, path = prepare_data(
            path,
            max_side=self.max_side,
            transform_fn=self.transform_fn,
        )
        return image, label, path

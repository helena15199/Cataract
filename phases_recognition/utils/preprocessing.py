"""Useful functions for preprocessing"""

import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=image.device, dtype=torch.float32).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device, dtype=torch.float32).reshape(3, 1, 1)
    image = image.to(torch.float32) / 255.0
    return (image - mean) / std


def unormalize_image(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=image.device, dtype=image.dtype).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device, dtype=image.dtype).reshape(3, 1, 1)
    image = image * std + mean
    return torch.clamp(image * 255, 0, 255)


class NormalizeImage:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return normalize_image(image)


class UnormalizeImage:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return unormalize_image(image)


class Identity:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image
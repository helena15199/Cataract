"""Dataset that loads pre-extracted ResNet features for MS-TCN++ training.

Each sample is a full video: features (T, feature_dim) + labels (T,).
Since videos have different lengths, the DataLoader must use batch_size=1
with the provided collate_fn (which just unwraps the single-item batch).
"""

import pathlib

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


class VideoFeatureDataset(Dataset):
    """
    One sample = one video.

    Expects a directory containing pairs of files:
        {video_name}.npy         — shape (T, feature_dim), float32
        {video_name}_labels.npy  — shape (T,),             int64

    Returns:
        features : (T, feature_dim) FloatTensor
        labels   : (T,)             LongTensor
        name     : str  (video folder name, useful for evaluation)
    """

    def __init__(self, root: str):
        self.root = pathlib.Path(root)
        # All feature files, excluding the *_labels.npy companions
        self.video_names = sorted(
            p.stem
            for p in self.root.glob("*.npy")
            if not p.stem.endswith("_labels") and not p.stem.endswith("_binary_ch")
        )
        if not self.video_names:
            raise ValueError(f"No feature files found in {self.root}")

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx: int):
        name = self.video_names[idx]
        features = np.load(self.root / f"{name}.npy")         # (T, D)
        labels   = np.load(self.root / f"{name}_labels.npy")  # (T,)
        return (
            torch.from_numpy(features).float(),  # (T, D)
            torch.from_numpy(labels).long(),     # (T,)
            name,
        )


def _collate_single_video(batch):
    """Unwrap the single-video batch — no padding needed."""
    features, labels, name = batch[0]
    return features, labels, name


def instantiate_feature_loaders(
    dataset_config: DictConfig | dict,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = VideoFeatureDataset(root=dataset_config["train"]["params"]["root"])
    val_dataset   = VideoFeatureDataset(root=dataset_config["val"]["params"]["root"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,  # .npy loading is fast; workers would add overhead
        collate_fn=_collate_single_video,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_single_video,
    )
    return train_loader, val_loader

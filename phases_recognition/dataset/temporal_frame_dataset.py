"""Dataset for end-to-end backbone + MS-TCN++ training.

Loads frames in temporal order as contiguous windows.

Train: random window of T_window consecutive frames per video per call.
Val:   full video split into non-overlapping T_window chunks.
"""

import json
import pathlib
import random
import re
from collections import defaultdict

import einops
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from dataset.cataract_dataset import basic_preprocess_image, build_transforms

FRAME_RE = re.compile(r"Frame_(\d+)")


def _frame_number(key: str) -> int:
    m = FRAME_RE.search(key)
    if m is None:
        raise ValueError(f"No frame number in: {key}")
    return int(m.group(1))


def _build_video_index(
    labels_json: pathlib.Path,
    split: str,
    class_to_idx: dict[str, int],
    others_remap: dict[str, str],
) -> list[list[tuple[pathlib.Path, int]]]:
    """
    Returns a list of videos. Each video is a list of (frame_path, label)
    sorted in temporal order.
    """
    with open(labels_json) as f:
        all_labels = json.load(f)

    video_keys: dict[str, list[str]] = defaultdict(list)
    for key in all_labels:
        parts = key.split("/")
        if parts[0] != split:
            continue
        video_keys[parts[1]].append(key)

    dataset_root = labels_json.parent
    videos = []
    for video, keys in video_keys.items():
        keys_sorted = sorted(keys, key=_frame_number)
        frames = []
        for k in keys_sorted:
            phase = all_labels[k]
            phase = others_remap.get(phase, phase)
            if phase not in class_to_idx:
                continue
            frames.append((dataset_root / k, class_to_idx[phase]))
        if frames:
            videos.append(frames)
    return videos


class TemporalFrameDataset(Dataset):
    """
    Train dataset — one sample = a random contiguous window of T_window frames
    from a randomly chosen video.

    Each call to __getitem__ samples a different window, providing implicit
    data augmentation across epochs.
    """

    def __init__(
        self,
        dataset_root: str,
        split: str,
        class_names: list[str],
        others_classes: list[str],
        T_window: int = 128,
        max_side: int = 224,
        transform_fn=None,
        seed: int = 42,
    ):
        self.T_window    = T_window
        self.max_side    = max_side
        self.transform_fn = transform_fn

        class_to_idx = {n: i for i, n in enumerate(class_names)}
        others_remap = {c: "Others" for c in others_classes}

        labels_json = pathlib.Path(dataset_root) / "labels.json"
        all_videos  = _build_video_index(labels_json, split, class_to_idx, others_remap)

        # Keep only videos long enough for at least one full window
        self.videos = [v for v in all_videos if len(v) >= T_window]
        if not self.videos:
            raise ValueError(
                f"No video with ≥{T_window} frames found in split='{split}' of {dataset_root}"
            )

    def __len__(self):
        # Approximate number of non-overlapping windows across all videos
        return sum(len(v) // self.T_window for v in self.videos)

    def __getitem__(self, _idx: int):
        # Pick a random video and a random start position
        video  = random.choice(self.videos)
        start  = random.randint(0, len(video) - self.T_window)
        window = video[start : start + self.T_window]

        images, labels = [], []
        for path, label in window:
            img = basic_preprocess_image(str(path), self.max_side)
            if self.transform_fn is not None:
                img = self.transform_fn(image=img)["image"]
            images.append(torch.from_numpy(einops.rearrange(img, "h w c -> c h w")))
            labels.append(label)

        return torch.stack(images), torch.tensor(labels, dtype=torch.long)


class TemporalFrameValDataset(Dataset):
    """
    Val dataset — one sample = one non-overlapping T_window chunk of a video.
    Each video is split into as many full chunks as possible (remainder dropped).

    Returns (frames, labels, video_name, chunk_idx) so the trainer can
    reconstruct per-video predictions.
    """

    def __init__(
        self,
        dataset_root: str,
        split: str,
        class_names: list[str],
        others_classes: list[str],
        T_window: int = 128,
        max_side: int = 224,
    ):
        self.T_window = T_window
        self.max_side = max_side

        class_to_idx = {n: i for i, n in enumerate(class_names)}
        others_remap = {c: "Others" for c in others_classes}

        labels_json = pathlib.Path(dataset_root) / "labels.json"
        all_videos  = _build_video_index(labels_json, split, class_to_idx, others_remap)

        # Build flat list of (video_name_idx, start) chunks
        self._videos = all_videos
        self._video_names = self._build_names(labels_json, split)
        self._chunks: list[tuple[int, int]] = []
        for vid_idx, video in enumerate(all_videos):
            n_chunks = len(video) // T_window
            for c in range(n_chunks):
                self._chunks.append((vid_idx, c * T_window))

        if not self._chunks:
            raise ValueError(f"No chunks found for split='{split}'")

    @staticmethod
    def _build_names(labels_json: pathlib.Path, split: str) -> list[str]:
        with open(labels_json) as f:
            all_labels = json.load(f)
        seen, names = set(), []
        for key in all_labels:
            parts = key.split("/")
            if parts[0] == split and parts[1] not in seen:
                seen.add(parts[1])
                names.append(parts[1])
        return names

    def __len__(self):
        return len(self._chunks)

    def __getitem__(self, idx: int):
        vid_idx, start = self._chunks[idx]
        video  = self._videos[vid_idx]
        window = video[start : start + self.T_window]

        images, labels = [], []
        for path, label in window:
            img = basic_preprocess_image(str(path), self.max_side)
            images.append(torch.from_numpy(einops.rearrange(img, "h w c -> c h w")))
            labels.append(label)

        chunk_idx  = start // self.T_window
        video_name = self._video_names[vid_idx] if vid_idx < len(self._video_names) else str(vid_idx)
        return torch.stack(images), torch.tensor(labels, dtype=torch.long), video_name, chunk_idx


def instantiate_e2e_loaders(
    dataset_config: DictConfig | dict,
    T_window: int = 128,
) -> tuple[DataLoader, DataLoader]:
    class_names    = list(dataset_config["class_names"])
    others_classes = list(dataset_config.get("others_classes") or [])
    dataset_root   = str(pathlib.Path(dataset_config["train"]["params"]["root"]).parent)
    max_side       = dataset_config["train"]["params"].get("max_side", 224)
    transform_fn   = build_transforms(dataset_config["train"].get("transforms"))

    train_dataset = TemporalFrameDataset(
        dataset_root   = dataset_root,
        split          = "train",
        class_names    = class_names,
        others_classes = others_classes,
        T_window       = T_window,
        max_side       = max_side,
        transform_fn   = transform_fn,
    )
    val_dataset = TemporalFrameValDataset(
        dataset_root   = dataset_root,
        split          = "val",
        class_names    = class_names,
        others_classes = others_classes,
        T_window       = T_window,
        max_side       = max_side,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = 1,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = _unwrap_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = _unwrap_batch,
    )
    return train_loader, val_loader


def _unwrap_batch(batch):
    """Collate function: unwrap the single-item batch (batch_size=1)."""
    return batch[0]

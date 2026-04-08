"""Extract ResNet backbone features for all videos and save as .npy files.

Usage:
    cd /home/helena/Cataract
    python phases_recognition/extract_features.py \
        --checkpoint /home/helena/experiments_cataract/<experiment>/ckpt/best.pt \
        --config phases_recognition/configs/config.yaml \
        --output_dir /home/helena/UCL_video_cataract/features/
"""

import argparse
import json
import pathlib
import re
from collections import defaultdict

import einops
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from dataset.cataract_dataset import basic_preprocess_image
from models import instantiate_model
from utils.preprocessing import normalize_image

FRAME_RE = re.compile(r"Frame_(\d+)")


def _extract_frame_number(path: pathlib.Path) -> int:
    m = FRAME_RE.search(path.name)
    if m is None:
        raise ValueError(f"No frame number found in filename: {path.name}")
    return int(m.group(1))


class _VideoFrameDataset(Dataset):
    """Loads all frames of a single video in temporal order, without augmentation."""

    def __init__(self, frame_paths: list[pathlib.Path], max_side: int):
        self.frame_paths = frame_paths
        self.max_side = max_side

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        image = basic_preprocess_image(str(self.frame_paths[idx]), self.max_side)
        image = einops.rearrange(image, "h w c -> c h w")
        return torch.from_numpy(image)


def main(args):
    config = OmegaConf.load(args.config)
    dataset_root = pathlib.Path(config.dataset.train.params.root).parent
    # dataset_root = /home/helena/UCL_video_cataract/dataset_temporal
    output_dir = pathlib.Path(args.output_dir)
    device = torch.device(args.device)
    max_side = config.dataset.train.params.max_side

    # --- Load model and keep only the backbone ---
    model = instantiate_model(config.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    backbone = model.backbone  # ResNet without FC head
    backbone.eval().to(device)

    # --- Build class mapping (same logic as ImageNumbersDataset) ---
    class_names = list(config.dataset.class_names)
    others_classes = list(config.dataset.get("others_classes") or [])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    others_remap = {cls: "Others" for cls in others_classes}

    with open(dataset_root / "labels.json") as f:
        all_labels = json.load(f)

    # Group frame keys by (split, video_folder)
    video_frames: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in all_labels:
        parts = key.split("/")
        split, video = parts[0], parts[1]
        video_frames[(split, video)].append(key)

    total_videos = len(video_frames)
    pbar = tqdm.tqdm(sorted(video_frames.items()), total=total_videos, desc="Extracting features")

    for (split, video), frame_keys in pbar:
        pbar.set_postfix(video=video[:40])
        out_dir = output_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        # Sort frames by temporal order using the frame number in the filename
        frame_keys_sorted = sorted(
            frame_keys,
            key=lambda k: _extract_frame_number(pathlib.Path(k)),
        )

        # Build label array
        label_list = []
        for k in frame_keys_sorted:
            phase = all_labels[k]
            phase = others_remap.get(phase, phase)
            if phase not in class_to_idx:
                raise ValueError(f"Unknown phase '{phase}' in {k}. Known: {list(class_to_idx)}")
            label_list.append(class_to_idx[phase])
        labels = np.array(label_list, dtype=np.int64)  # (T,)

        # Load frames and extract features
        frame_paths = [dataset_root / k for k in frame_keys_sorted]
        frame_dataset = _VideoFrameDataset(frame_paths, max_side=max_side)
        loader = DataLoader(
            frame_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        features_list = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                batch = normalize_image(batch)
                feats = backbone(batch)  # (B, 2048)
                features_list.append(feats.cpu().numpy())

        features = np.concatenate(features_list, axis=0)  # (T, 2048)

        np.save(out_dir / f"{video}.npy", features)
        np.save(out_dir / f"{video}_labels.npy", labels)

    print(f"\nDone. Features saved to {output_dir}")
    print("Structure:")
    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        if split_dir.exists():
            n = len(list(split_dir.glob("*_labels.npy")))
            print(f"  {split}/  — {n} videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract ResNet backbone features for all videos")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained ResNet checkpoint (best.pt)",
    )
    parser.add_argument(
        "--config", type=str, default="phases_recognition/configs/config.yaml",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/home/helena/UCL_video_cataract/features/",
        help="Root directory where features/{train,val,test}/*.npy will be written",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)

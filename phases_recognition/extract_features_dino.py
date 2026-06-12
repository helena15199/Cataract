"""Extract DINOv2 CLS-token features for all videos and save as .npy files.

No checkpoint needed — DINOv2 is loaded pretrained from torch.hub.
Output format is identical to extract_features.py so the rest of the
pipeline (VideoFeatureDataset, train_temporal.py, evaluate_temporal.py)
works unchanged.

Usage:
    cd /home/helena/Cataract
    python phases_recognition/extract_features_dino.py \
        --config phases_recognition/configs/config.yaml \
        --output_dir /home/helena/UCL_video_cataract/features_dino/
"""

import argparse
import json
import pathlib
import re
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as T
import tqdm
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset

FRAME_RE = re.compile(r"Frame_(\d+)")

# DINOv2 expects ImageNet normalisation
_DINO_TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _extract_frame_number(path: pathlib.Path) -> int:
    m = FRAME_RE.search(path.name)
    if m is None:
        raise ValueError(f"No frame number in filename: {path.name}")
    return int(m.group(1))


class _VideoFrameDataset(Dataset):
    def __init__(self, frame_paths: list[pathlib.Path]):
        self.frame_paths = frame_paths

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img = Image.open(self.frame_paths[idx]).convert("RGB")
        return _DINO_TRANSFORM(img)


def main(args):
    config = OmegaConf.load(args.config)
    dataset_root = pathlib.Path(config.dataset.train.params.root).parent
    output_dir   = pathlib.Path(args.output_dir)
    device       = torch.device(args.device)

    # --- Load DINOv2 (frozen, no grad) ---
    print(f"Loading DINOv2 ({args.model}) from torch.hub ...")
    dino = torch.hub.load("facebookresearch/dinov2", args.model, verbose=False)
    dino.eval().to(device)
    for p in dino.parameters():
        p.requires_grad_(False)

    # --- Build class mapping from config ---
    class_names     = list(config.dataset.class_names)
    others_classes  = set(config.dataset.get("others_classes")  or [])
    exclude_classes = set(config.dataset.get("exclude_classes") or [])
    binary_phase    = config.dataset.get("binary_phase") or None
    class_to_idx    = {name: idx for idx, name in enumerate(class_names)}
    if binary_phase:
        class_to_idx[binary_phase] = -1

    labels_file = config.dataset.get("labels_file", "labels.json")
    with open(dataset_root / labels_file) as f:
        all_labels = json.load(f)

    # Group frame keys by (split, video)
    video_frames: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in all_labels:
        parts = key.split("/")
        split, video = parts[0], parts[1]
        video_frames[(split, video)].append(key)

    pbar = tqdm.tqdm(sorted(video_frames.items()), total=len(video_frames), desc="Extracting DINOv2 features")

    for (split, video), frame_keys in pbar:
        pbar.set_postfix(video=video[:40])
        out_dir = output_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already extracted (allows resuming a crashed run)
        if (out_dir / f"{video}.npy").exists():
            pbar.write(f"  Skipping {video} — already extracted")
            continue

        frame_keys_sorted = sorted(frame_keys, key=lambda k: _extract_frame_number(pathlib.Path(k)))

        label_list, frame_paths_final = [], []
        for k in frame_keys_sorted:
            phase = all_labels[k]
            if phase in exclude_classes:
                continue
            img_path = dataset_root / k
            if not img_path.exists():
                found = False
                for alt in ("train", "val", "test"):
                    alt_path = dataset_root / alt / k.split("/", 1)[1]
                    if alt_path.exists():
                        img_path = alt_path
                        found = True
                        break
                if not found:
                    continue
            label_list.append(-1 if (phase in others_classes or phase not in class_to_idx) else class_to_idx[phase])
            frame_paths_final.append(img_path)

        if not frame_paths_final:
            pbar.write(f"  Skipping {video} — no valid frames")
            continue

        loader = DataLoader(
            _VideoFrameDataset(frame_paths_final),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        features_list = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                cls_token = dino(batch)          # (B, 768) for ViT-B, (B, 1024) for ViT-L
                features_list.append(cls_token.cpu().numpy())

        features = np.concatenate(features_list, axis=0).astype(np.float32)  # (T, D)
        labels   = np.array(label_list, dtype=np.int64)                       # (T,)

        np.save(out_dir / f"{video}.npy",        features)
        np.save(out_dir / f"{video}_labels.npy", labels)

    print(f"\nDone. Features saved to {output_dir}")
    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        if split_dir.exists():
            n = len(list(split_dir.glob("*_labels.npy")))
            print(f"  {split}/  — {n} videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract DINOv2 CLS-token features for all videos")
    parser.add_argument("--config",     type=str, default="phases_recognition/configs/config.yaml")
    parser.add_argument("--output_dir", type=str, default="/home/helena/UCL_video_cataract/features_dino/")
    parser.add_argument("--model",      type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                        help="DINOv2 variant. vitb14 → 768-dim, vitl14 → 1024-dim")
    parser.add_argument("--device",     type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers",type=int, default=8)
    args = parser.parse_args()
    main(args)

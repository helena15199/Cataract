"""Evaluation script for the end-to-end ResNet50 + MS-TCN++ model.

Différence vs evaluate_temporal.py : l'input est des frames brutes (images),
pas des features .npy pré-extraites. Le backbone est appliqué on-the-fly
par chunks pour gérer la mémoire.

Usage (from repo root):
    python phases_recognition/evaluate_e2e.py \
        --config phases_recognition/configs/config_e2e.yaml \
        --ckpt /home/helena/experiments_cataract/<exp>/ckpt/best.pt \
        --out_dir /home/helena/experiments_cataract/<exp>/eval_test/ \
        [--smooth_window 15]
        [--backbone_chunk 256]
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

# Réutilise toutes les fonctions de metrics et plots de evaluate_temporal
from evaluate_temporal import (
    majority_vote_smooth,
    compute_all_metrics,
    plot_phase_timeline,
    plot_confusion_matrix,
    plot_per_video_f1,
)

FRAME_RE = re.compile(r"Frame_(\d+)")


# ---------------------------------------------------------------------------
# Dataset : charge toutes les frames d'une vidéo en ordre temporel
# ---------------------------------------------------------------------------

class _FullVideoDataset(Dataset):
    """Charge toutes les frames d'une vidéo en ordre temporel."""

    def __init__(self, frame_paths: list[pathlib.Path], max_side: int = 224):
        self.frame_paths = frame_paths
        self.max_side = max_side

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img = basic_preprocess_image(str(self.frame_paths[idx]), self.max_side)
        return torch.from_numpy(einops.rearrange(img, "h w c -> c h w"))


def _load_test_videos(
    dataset_root: pathlib.Path,
    class_names: list[str],
    others_classes: list[str],
) -> list[tuple[list[pathlib.Path], list[int], str]]:
    """
    Retourne une liste de (frame_paths, labels, video_name) pour chaque
    vidéo du split test, frames triées temporellement.
    """
    import json
    from collections import defaultdict

    with open(dataset_root / "labels.json") as f:
        all_labels = json.load(f)

    class_to_idx = {n: i for i, n in enumerate(class_names)}
    others_remap = {c: "Others" for c in others_classes}

    video_keys: dict[str, list[str]] = defaultdict(list)
    for key in all_labels:
        parts = key.split("/")
        if parts[0] == "test":
            video_keys[parts[1]].append(key)

    results = []
    for video_name, keys in sorted(video_keys.items()):
        keys_sorted = sorted(keys, key=lambda k: int(FRAME_RE.search(k).group(1)))
        frame_paths, labels = [], []
        for k in keys_sorted:
            phase = all_labels[k]
            phase = others_remap.get(phase, phase)
            if phase not in class_to_idx:
                continue
            frame_paths.append(dataset_root / k)
            labels.append(class_to_idx[phase])
        if frame_paths:
            results.append((frame_paths, labels, video_name))

    return results


# ---------------------------------------------------------------------------
# Inférence : backbone par chunks + TCN sur séquence complète
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_e2e(
    model,
    test_videos: list[tuple[list[pathlib.Path], list[int], str]],
    device: torch.device,
    max_side: int = 224,
    backbone_chunk: int = 256,
) -> list[tuple[list[int], list[int], str]]:
    """
    Pour chaque vidéo :
      1. Charge toutes les frames
      2. Passe le backbone par chunks de `backbone_chunk` frames
      3. Passe le TCN sur la séquence complète de features
      4. Retourne (gt_seq, pred_seq, video_name)
    """
    model.eval()
    results = []

    for frame_paths, gt_labels, video_name in tqdm.tqdm(test_videos, desc="Inference"):
        # Charge les frames via DataLoader (parallélisme I/O)
        frame_ds  = _FullVideoDataset(frame_paths, max_side=max_side)
        frame_dl  = DataLoader(frame_ds, batch_size=backbone_chunk,
                               shuffle=False, num_workers=4, pin_memory=True)

        # Backbone par chunks → accumule les features
        all_features = []
        for batch in frame_dl:
            batch = batch.to(device, non_blocking=True)
            batch = normalize_image(batch)
            feats = model.backbone(batch)          # (chunk, 2048)
            all_features.append(feats.cpu())

        features = torch.cat(all_features, dim=0)  # (T, 2048)
        features = features.unsqueeze(0).to(device)  # (1, T, 2048)

        # TCN sur la séquence complète
        stage_logits = model.tcn(features)         # list of (1, C, T)
        last_logits  = stage_logits[-1].squeeze(0).T  # (T, C)
        preds = last_logits.argmax(dim=1).cpu().tolist()

        results.append((gt_labels, preds, video_name))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str, ckpt_path: str, out_dir: str,
         smooth_window: int, backbone_chunk: int):
    config  = OmegaConf.load(config_path)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(config.train.device)

    # Dataset root (frames, pas features)
    dataset_root = pathlib.Path(config.dataset.train.params.root).parent
    if not (dataset_root / "labels.json").exists():
        raise FileNotFoundError(f"labels.json introuvable dans {dataset_root}")

    # Modèle
    model = instantiate_model(config.model)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    print(f"Loaded checkpoint: epoch {state.get('epoch', '?')}")

    class_names    = list(config.dataset.class_names)
    others_classes = list(config.dataset.get("others_classes") or [])
    others_indices = {class_names.index(c) for c in config.metrics.others_classes
                      if c in class_names}
    eval_indices   = [i for i in range(len(class_names)) if i not in others_indices]
    num_classes    = config.model.num_classes
    max_side       = config.dataset.train.params.get("max_side", 224)

    # Charge les vidéos test en ordre temporel
    test_videos = _load_test_videos(dataset_root, class_names, others_classes)
    print(f"{len(test_videos)} test videos trouvées.")

    # Inférence
    video_results_raw = run_inference_e2e(
        model, test_videos, device,
        max_side=max_side,
        backbone_chunk=backbone_chunk,
    )

    # Lissage
    video_results_smooth = [
        (gt, majority_vote_smooth(pred, smooth_window), name)
        for gt, pred, name in video_results_raw
    ]

    # Métriques
    raw_metrics, raw_vf1, raw_preds_flat, raw_labels_flat = compute_all_metrics(
        video_results_raw, num_classes, class_names,
        list(config.metrics.others_classes), prefix="raw")
    smooth_metrics, smooth_vf1, smooth_preds_flat, _ = compute_all_metrics(
        video_results_smooth, num_classes, class_names,
        list(config.metrics.others_classes), prefix="smoothed")

    all_metrics = {**raw_metrics, **smooth_metrics}

    # Affichage
    print(f"\n=== Test metrics E2E (smooth_window={smooth_window}) ===")
    print(f"{'Metric':<35} {'Raw':>8}  {'Smoothed':>10}")
    print("-" * 58)
    for key in ["global/accuracy", "global/f1_macro", "global/auroc",
                "segment/edit_score", "segment/f1@10",
                "segment/f1@25",     "segment/f1@50"]:
        r = raw_metrics.get(f"raw/{key}", float("nan"))
        s = smooth_metrics.get(f"smoothed/{key}", float("nan"))
        print(f"  {key:<33} {r:>8.4f}  {s:>10.4f}")

    print("\nPer-class F1 (raw → smoothed):")
    for c in class_names:
        if c in list(config.metrics.others_classes):
            continue
        r = raw_metrics.get(f"raw/per_class/f1/{c}", 0.0)
        s = smooth_metrics.get(f"smoothed/per_class/f1/{c}", 0.0)
        arrow = "↑" if s - r > 0.01 else ("↓" if r - s > 0.01 else "~")
        print(f"  {c:<35} {r:.3f} → {s:.3f}  {arrow}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({k: round(v, 6) for k, v in all_metrics.items()}, f, indent=2)
    print(f"\n  Saved: metrics.json")

    # Plots
    print("\nGenerating plots...")
    plot_phase_timeline(video_results_raw, video_results_smooth,
                        class_names, out_dir / "phase_timeline.png", smooth_window)
    plot_confusion_matrix(raw_preds_flat, raw_labels_flat, class_names, eval_indices,
                          out_dir / "confusion_matrix_raw.png",
                          "Confusion matrix — raw (test set, e2e)")
    plot_confusion_matrix(smooth_preds_flat, raw_labels_flat, class_names, eval_indices,
                          out_dir / "confusion_matrix_smoothed.png",
                          f"Confusion matrix — smoothed w={smooth_window} (test set, e2e)")
    plot_per_video_f1(raw_vf1, smooth_vf1, out_dir / "per_video_f1.png")

    print(f"\nDone. Résultats sauvegardés dans {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate E2E ResNet50 + MS-TCN++ on test set")
    parser.add_argument("--config",  type=str,
                        default="phases_recognition/configs/config_e2e.yaml")
    parser.add_argument("--ckpt",    type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--smooth_window",  type=int, default=15)
    parser.add_argument("--backbone_chunk", type=int, default=256,
                        help="Frames passées au backbone à la fois (gestion mémoire)")
    args = parser.parse_args()
    main(args.config, args.ckpt, args.out_dir, args.smooth_window, args.backbone_chunk)

"""Compute class-conditional Mahalanobis stats on MSTCN internal features (32D).

These features come from the last refinement stage, just before the classification
head. They encode temporal context and are a much better novelty signal than
ResNet features: a truly unknown phase will be far from all class centroids even
if the classifier confidently assigns it to a known class.

Usage:
    python phases_recognition/utils/compute_mstcn_mahal_stats.py \
        --config  phases_recognition/configs/config_mstcn.yaml \
        --ckpt    /home/helena/experiments_cataract/.../ckpt/best.pt \
        --out_path /home/helena/UCL_video_cataract/features_v1.9/mstcn_mahal_stats.npz
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from dataset.feature_dataset import VideoFeatureDataset, _collate_single_video
from models import instantiate_model


def main(args):
    config     = OmegaConf.load(args.config)
    train_root = pathlib.Path(config.dataset.train.params.root)
    class_names = list(config.dataset.class_names)
    n_classes   = len(class_names)
    device      = torch.device(config.train.device)

    # Load MSTCN
    model = instantiate_model(config.model)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval().to(device)

    # Collect 32D features per frame
    loader = DataLoader(
        VideoFeatureDataset(root=str(train_root)),
        batch_size=1, shuffle=False, collate_fn=_collate_single_video,
    )

    all_features, all_labels = [], []
    with torch.no_grad():
        for features, labels, video_name in loader:
            features = features.unsqueeze(0).to(device)   # (1, T, 2048)
            _, feat_32d = model.forward_with_features(features)
            # feat_32d: (1, num_f_maps, T) → (T, num_f_maps)
            feat_32d = feat_32d.squeeze(0).T.cpu().numpy()
            labels_np = labels.numpy()

            # Keep only in-distribution frames (label >= 0)
            mask = labels_np >= 0
            all_features.append(feat_32d[mask])
            all_labels.append(labels_np[mask])

    features = np.concatenate(all_features, axis=0)  # (N, 32)
    labels   = np.concatenate(all_labels,   axis=0)  # (N,)
    print(f"Total in-dist frames : {len(features):,}  |  dim={features.shape[1]}")

    # Class-conditional means
    class_means  = np.zeros((n_classes, features.shape[1]), dtype=np.float32)
    class_counts = np.zeros(n_classes, dtype=int)
    for c in range(n_classes):
        mask = labels == c
        class_counts[c] = mask.sum()
        if mask.sum() > 0:
            class_means[c] = features[mask].mean(axis=0)
        print(f"  {class_names[c]:<35} {class_counts[c]:>6} frames")

    # Shared covariance (LedoitWolf) on 32D — very fast
    print("\nFitting shared covariance (LedoitWolf) on 32D features...")
    centered = features.copy()
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            centered[mask] -= class_means[c]

    lw = LedoitWolf(assume_centered=True)
    lw.fit(centered)
    precision = lw.precision_.astype(np.float32)  # (32, 32)

    # OOD threshold: 95th percentile of POSITIVE in-dist distances
    # (flags only the 5% of training frames that are most anomalous)
    # NOTE: do NOT use mahalanobis_scores() here — that function returns NEGATIVE
    # distances (high = in-dist), which is the opposite convention of _mahal_ood_signal
    # in evaluate_temporal.py (positive, high = OOD).
    T_id = len(features)
    C_id = len(class_means)
    min_dist2 = np.full(T_id, np.inf, dtype=np.float64)
    for c in range(C_id):
        diff  = features - class_means[c]
        dist2 = (diff @ precision.astype(np.float64) * diff).sum(axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    threshold = float(np.percentile(min_dist2, 95))
    print(f"  Threshold (95th percentile of train dist²) : {threshold:.4f}")

    out_path = pathlib.Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        class_means  = class_means,
        precision    = precision,
        class_names  = np.array(class_names),
        class_counts = class_counts,
        threshold    = np.float32(threshold),
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--ckpt",     type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    main(parser.parse_args())

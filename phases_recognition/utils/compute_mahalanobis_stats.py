"""Compute class-conditional means and shared covariance from training features.

Saves:
    mahal_stats.npz  — class_means (C, D), precision_matrix (D, D), class_names

Usage:
    python phases_recognition/compute_mahalanobis_stats.py \
        --config phases_recognition/configs/config_mstcn.yaml \
        --out_path /home/helena/UCL_video_cataract/features_v1.8/mahal_stats.npz
"""

import argparse
import pathlib

import numpy as np
from omegaconf import OmegaConf
from sklearn.covariance import LedoitWolf


def main(args):
    config      = OmegaConf.load(args.config)
    train_root  = pathlib.Path(config.dataset.train.params.root)
    class_names = list(config.dataset.class_names)
    n_classes   = len(class_names)

    # Load all training features and labels
    all_features, all_labels = [], []
    for feat_file in sorted(train_root.glob("*.npy")):
        if feat_file.stem.endswith("_labels") or feat_file.stem.endswith("_binary_ch"):
            continue
        label_file = train_root / f"{feat_file.stem}_labels.npy"
        if not label_file.exists():
            continue
        feats  = np.load(feat_file)   # (T, D)
        labels = np.load(label_file)  # (T,)
        # Exclude CH frames (label=-1)
        mask = labels >= 0
        all_features.append(feats[mask])
        all_labels.append(labels[mask])

    features = np.concatenate(all_features, axis=0)  # (N, D)
    labels   = np.concatenate(all_labels,   axis=0)  # (N,)
    print(f"Total training frames : {len(features):,}  |  dim={features.shape[1]}")

    # "Others" est exclu des centroïdes — c'est un mix de phases rares sans lien
    # entre elles, son centroïde ne serait pas représentatif. Ces frames seront
    # naturellement détectées comme OOD, ce qui est le comportement voulu.
    others_idx = class_names.index("Others") if "Others" in class_names else -1
    in_dist_mask = (labels != others_idx) if others_idx >= 0 else np.ones(len(labels), dtype=bool)
    features_id = features[in_dist_mask]
    labels_id   = labels[in_dist_mask]

    # Class-conditional means (sur les classes in-distribution uniquement)
    class_means = np.zeros((n_classes, features.shape[1]), dtype=np.float32)
    class_counts = np.zeros(n_classes, dtype=int)
    for c in range(n_classes):
        if c == others_idx:
            print(f"  {class_names[c]:<35} {'(exclu)':>6}")
            continue
        mask = labels_id == c
        class_counts[c] = mask.sum()
        if mask.sum() > 0:
            class_means[c] = features_id[mask].mean(axis=0)
        print(f"  {class_names[c]:<35} {mask.sum():>6} frames")

    # Shared covariance sur les frames in-distribution uniquement
    print("\nFitting shared covariance (LedoitWolf) — Others exclus...")
    centered = features_id.copy()
    for c in range(n_classes):
        if c == others_idx:
            continue
        mask = labels_id == c
        if mask.sum() > 0:
            centered[mask] -= class_means[c]

    lw = LedoitWolf(assume_centered=True)
    lw.fit(centered)
    precision = lw.precision_.astype(np.float32)  # (D, D) inverse covariance

    out_path = pathlib.Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, class_means=class_means, precision=precision,
             class_names=np.array(class_names), class_counts=class_counts)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute Mahalanobis statistics from training features")
    parser.add_argument("--config",   type=str,
                        default="phases_recognition/configs/config_mstcn.yaml")
    parser.add_argument("--out_path", type=str, required=True)
    main(parser.parse_args())

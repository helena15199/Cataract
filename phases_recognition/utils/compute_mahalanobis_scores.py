"""Compute per-frame Mahalanobis OOD scores for a given split.

For each frame: score = -min_c ( (f - mu_c)^T * P * (f - mu_c) )
High score (close to 0) = in-distribution.
Low score (very negative) = OOD / anomaly.

Saves one *_mahal.npy per video alongside the feature files.

Usage:
    python phases_recognition/compute_mahalanobis_scores.py \
        --stats   /home/helena/UCL_video_cataract/features_v1.8/mahal_stats.npz \
        --feat_dir /home/helena/UCL_video_cataract/features_v1.8/test/
"""

import argparse
import pathlib

import numpy as np


def mahalanobis_scores(features: np.ndarray,
                       class_means: np.ndarray,
                       precision: np.ndarray) -> np.ndarray:
    """
    features    : (T, D)
    class_means : (C, D)
    precision   : (D, D)
    Returns     : (T,) — score = -min_c Mahalanobis distance² (higher = more normal)
    """
    T = len(features)
    C = len(class_means)
    scores = np.full(T, -np.inf, dtype=np.float32)

    for c in range(C):
        diff = features - class_means[c]          # (T, D)
        # Mahalanobis² = diag( diff @ P @ diff^T )
        maha2 = (diff @ precision * diff).sum(axis=1)  # (T,)
        scores = np.maximum(scores, -maha2)

    return scores


def main(args):
    stats       = np.load(args.stats, allow_pickle=True)
    all_means   = stats["class_means"].astype(np.float32)   # (C, D)
    class_names = stats["class_names"].tolist()
    precision   = stats["precision"].astype(np.float32)     # (D, D)

    # Exclure Others : son centroïde est vide (vecteur nul), pas représentatif
    others_idx  = class_names.index("Others") if "Others" in class_names else -1
    valid_idx   = [i for i in range(len(class_names)) if i != others_idx]
    class_means = all_means[valid_idx]

    feat_dir = pathlib.Path(args.feat_dir)
    feat_files = sorted(
        f for f in feat_dir.glob("*.npy")
        if not f.stem.endswith("_labels") and not f.stem.endswith("_binary_ch")
        and not f.stem.endswith("_mahal")
    )

    print(f"Computing Mahalanobis scores for {len(feat_files)} videos in {feat_dir.name}/")
    for feat_file in feat_files:
        features = np.load(feat_file).astype(np.float32)  # (T, D)
        scores   = mahalanobis_scores(features, class_means, precision)
        out_path = feat_dir / f"{feat_file.stem}_mahal.npy"
        np.save(out_path, scores)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute Mahalanobis OOD scores")
    parser.add_argument("--stats",    type=str, required=True,
                        help="Path to mahal_stats.npz")
    parser.add_argument("--feat_dir", type=str, required=True,
                        help="Directory containing feature .npy files")
    main(parser.parse_args())

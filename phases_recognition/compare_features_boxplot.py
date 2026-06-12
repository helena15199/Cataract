"""Compare ResNet (2048D) vs MS-TCN (64D) feature spaces on correct vs wrong
frame predictions, using Mahalanobis distance to the predicted-class centroid.

For each frame we compute:
  - d_resnet = Mahalanobis²(resnet_feature, centroid of predicted class)   in 2048D
  - d_mstcn  = Mahalanobis²(mstcn_feature,  centroid of predicted class)   in 64D
and bucket frames into "correct" (pred == GT) / "wrong" (pred != GT).

Box plots let us see which feature space better separates correct from
wrong predictions — i.e. which one carries the most signal for detecting
"the model is wrong".

Usage:
    python phases_recognition/compare_features_boxplot.py \
        --config phases_recognition/configs/config_mstcn.yaml \
        --ckpt /path/to/best.pt \
        --out_dir /path/to/eval_boxplot/
"""

import argparse
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset.feature_dataset import VideoFeatureDataset, _collate_single_video
from models import instantiate_model

matplotlib.use("Agg")

OOD_LABEL = 12  # sentinel index for unseen phases in GT (label -1 in features)


def _mahal_to_predicted_class(features: np.ndarray, preds: np.ndarray, stats: dict) -> np.ndarray:
    """Mahalanobis² distance of each frame to the centroid of ITS PREDICTED class.

    Unlike `_mahal_ood_signal` (min over all classes), this measures
    "how far is this frame from where the model thinks it belongs" —
    the relevant question for "is the model wrong".

    Returns (T,) float32.
    """
    class_means = stats["class_means"].astype(np.float64)  # (C, D)
    precision   = stats["precision"].astype(np.float64)    # (D, D)
    feats       = features.astype(np.float64)              # (T, D)

    mu    = class_means[preds]                 # (T, D) — centroid of predicted class per frame
    diff  = feats - mu                          # (T, D)
    dist2 = (diff @ precision * diff).sum(axis=1)  # (T,)
    return dist2.astype(np.float32)


@torch.no_grad()
def collect_distances(model, test_root, device, resnet_stats, mstcn_stats):
    """Run inference once, return per-frame (d_resnet, d_mstcn, is_correct)."""
    dataset = VideoFeatureDataset(root=str(test_root))
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         collate_fn=_collate_single_video)
    model.eval()

    d_resnet_all, d_mstcn_all, correct_all = [], [], []

    for features, labels, video_name in tqdm.tqdm(loader, desc="Inference"):
        resnet_np    = features.numpy()                       # (T, 2048)
        features_gpu = features.unsqueeze(0).to(device)

        stage_logits, mstcn_feats = model.forward_with_features(features_gpu)
        last_logits = stage_logits[-1].squeeze(0).T           # (T, C)
        preds       = last_logits.argmax(dim=1).cpu().numpy()  # (T,)
        mstcn_np    = mstcn_feats.squeeze(0).T.cpu().numpy()   # (T, num_f_maps)

        gt = np.asarray(labels.tolist())
        known_mask = gt != -1
        if not known_mask.any():
            continue

        d_resnet = _mahal_to_predicted_class(resnet_np[known_mask], preds[known_mask], resnet_stats)
        d_mstcn  = _mahal_to_predicted_class(mstcn_np[known_mask],  preds[known_mask], mstcn_stats)
        is_correct = (preds[known_mask] == gt[known_mask])

        d_resnet_all.append(d_resnet)
        d_mstcn_all.append(d_mstcn)
        correct_all.append(is_correct)

    return (np.concatenate(d_resnet_all), np.concatenate(d_mstcn_all),
            np.concatenate(correct_all))


def plot_boxplots(d_resnet, d_mstcn, is_correct, out_path: pathlib.Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    for ax, d, title in zip(axes, [d_resnet, d_mstcn], ["ResNet (2048D)", "MS-TCN (feature space)"]):
        data = [d[is_correct], d[~is_correct]]
        bp = ax.boxplot(data, labels=["Correct", "Wrong"], showfliers=True,
                        patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], ["#4C78A8", "#E15759"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Mahalanobis² to predicted-class centroid")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Distance to predicted-class centroid — correct vs wrong predictions", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main(config_path: str, ckpt_path: str, out_dir: str, split: str = "test"):
    config  = OmegaConf.load(config_path)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(config.train.device)

    val_root   = pathlib.Path(config.dataset.val.params.root)
    split_root = val_root.parent / split
    if not split_root.exists():
        raise FileNotFoundError(f"Features not found at {split_root}.")

    model = instantiate_model(config.model)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    print(f"Loaded checkpoint: epoch {state.get('epoch', '?')}")

    # ResNet 2048D Mahalanobis stats
    resnet_stats_path = split_root.parent / "mahal_stats.npz"
    _r = np.load(resnet_stats_path, allow_pickle=True)
    resnet_stats = {
        "class_means": _r["class_means"].astype(np.float32),
        "precision":   _r["precision"].astype(np.float32),
    }
    print(f"  ResNet Mahalanobis stats loaded — {len(resnet_stats['class_means'])} classes, "
          f"D={resnet_stats['class_means'].shape[1]}")

    # MS-TCN Mahalanobis stats (must match the model's num_f_maps — recompute if stale)
    mstcn_stats_path = split_root.parent / "mstcn_mahal_stats.npz"
    _s = np.load(mstcn_stats_path, allow_pickle=True)
    mstcn_stats = {
        "class_means": _s["class_means"].astype(np.float32),
        "precision":   _s["precision"].astype(np.float32),
    }
    print(f"  MS-TCN Mahalanobis stats loaded — {len(mstcn_stats['class_means'])} classes, "
          f"D={mstcn_stats['class_means'].shape[1]}")

    expected_dim = config.model.num_f_maps
    if mstcn_stats["class_means"].shape[1] != expected_dim:
        raise ValueError(
            f"mstcn_mahal_stats.npz has D={mstcn_stats['class_means'].shape[1]} but model "
            f"num_f_maps={expected_dim} — recompute stats with compute_mstcn_mahal_stats.py"
        )

    d_resnet, d_mstcn, is_correct = collect_distances(model, split_root, device, resnet_stats, mstcn_stats)
    print(f"\n  Total frames: {len(is_correct)}  |  correct: {is_correct.sum()}  "
          f"wrong: {(~is_correct).sum()}")

    print("\n=== Median Mahalanobis² to predicted-class centroid ===")
    print(f"  ResNet  — correct: {np.median(d_resnet[is_correct]):.2f}   "
          f"wrong: {np.median(d_resnet[~is_correct]):.2f}")
    print(f"  MS-TCN  — correct: {np.median(d_mstcn[is_correct]):.2f}   "
          f"wrong: {np.median(d_mstcn[~is_correct]):.2f}")

    plot_boxplots(d_resnet, d_mstcn, is_correct, out_dir / "feature_distance_boxplot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compare ResNet vs MS-TCN feature distances for correct/wrong predictions")
    parser.add_argument("--config", type=str, default="phases_recognition/configs/config_mstcn.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    main(args.config, args.ckpt, args.out_dir, args.split)

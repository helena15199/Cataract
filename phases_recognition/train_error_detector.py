"""Train a logistic regression to predict when the MS-TCN model is wrong.

For each frame we compute 4 signals from the model's outputs:
  - d_resnet  : Mahalanobis² (ResNet 2048D) to centroid of predicted class
  - d_mstcn   : Mahalanobis² (MS-TCN 64D)  to centroid of predicted class
  - entropy   : entropy of last-stage softmax
  - kl        : KL divergence between stage-1 and stage-4 softmax (inter-stage disagreement)

A logistic regression is trained on the VAL set (label = pred != GT) and
evaluated on the TEST set. This gives the optimal linear combination of all
signals, plus individual AUROC scores for comparison.

Usage:
    python phases_recognition/train_error_detector.py \
        --config phases_recognition/configs/config_mstcn.yaml \
        --ckpt /path/to/best.pt \
        --out_dir /path/to/eval_error_detector/
"""

import argparse
import json
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset.feature_dataset import VideoFeatureDataset, _collate_single_video
from models import instantiate_model

matplotlib.use("Agg")


def _mahal_to_predicted_class(features: np.ndarray, preds: np.ndarray, stats: dict) -> np.ndarray:
    """Mahalanobis² of each frame to the centroid of its predicted class."""
    class_means = stats["class_means"].astype(np.float64)
    precision   = stats["precision"].astype(np.float64)
    feats       = features.astype(np.float64)
    mu    = class_means[preds]
    diff  = feats - mu
    dist2 = (diff @ precision * diff).sum(axis=1)
    return dist2.astype(np.float32)


@torch.no_grad()
def collect_features(model, data_root, device, resnet_stats, mstcn_stats):
    """Collect 4 per-frame signals + correctness label for all videos in data_root.

    Returns X (N, 4) and y (N,) where y=1 means the model was WRONG.
    Columns: [d_resnet, d_mstcn, entropy, kl_inter_stage]
    """
    dataset = VideoFeatureDataset(root=str(data_root))
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         collate_fn=_collate_single_video)
    model.eval()

    X_list, y_list, X_ood_list = [], [], []

    for features, labels, video_name in tqdm.tqdm(loader, desc=f"  {data_root.name}"):
        resnet_np    = features.numpy()                        # (T, 2048)
        features_gpu = features.unsqueeze(0).to(device)       # (1, T, 2048)

        stage_logits, mstcn_feats = model.forward_with_features(features_gpu)

        last_logits = stage_logits[-1].squeeze(0).T            # (T, C)
        preds       = last_logits.argmax(dim=1).cpu().numpy()  # (T,)
        mstcn_np    = mstcn_feats.squeeze(0).T.cpu().numpy()   # (T, num_f_maps)

        probs   = torch.softmax(last_logits, dim=1).cpu()
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=1).numpy()

        p_first = torch.softmax(stage_logits[0].squeeze(0).T, dim=1).cpu()
        kl      = np.abs((p_first * ((p_first + 1e-9) / (probs + 1e-9)).log()).sum(dim=1).numpy())

        gt = np.asarray(labels.tolist())
        known_mask = gt != -1
        ood_mask   = gt == -1   # unknown phase frames

        if known_mask.any():
            d_resnet = _mahal_to_predicted_class(resnet_np[known_mask], preds[known_mask], resnet_stats)
            d_mstcn  = _mahal_to_predicted_class(mstcn_np[known_mask],  preds[known_mask], mstcn_stats)
            X = np.stack([d_resnet, d_mstcn, entropy[known_mask], kl[known_mask]], axis=1)
            y = (preds[known_mask] != gt[known_mask]).astype(np.int32)
            X_list.append(X)
            y_list.append(y)

        # Collect signals on OOD frames (gt == -1) separately
        if ood_mask.any():
            d_resnet_ood = _mahal_to_predicted_class(resnet_np[ood_mask], preds[ood_mask], resnet_stats)
            d_mstcn_ood  = _mahal_to_predicted_class(mstcn_np[ood_mask],  preds[ood_mask], mstcn_stats)
            X_ood = np.stack([d_resnet_ood, d_mstcn_ood, entropy[ood_mask], kl[ood_mask]], axis=1)
            X_ood_list.append(X_ood)

    X_ood = np.concatenate(X_ood_list, axis=0) if X_ood_list else np.empty((0, 4), dtype=np.float32)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0), X_ood


def eval_signal(y_true, score, name, threshold_pct=85):
    """Print AUROC + recall/FPR at given percentile threshold for a single signal."""
    auroc  = roc_auc_score(y_true, score)
    thr    = float(np.percentile(score[y_true == 0], threshold_pct))
    recall = float((score[y_true == 1] > thr).mean())
    fpr    = float((score[y_true == 0] > thr).mean())
    print(f"  {name:<30}  AUROC={auroc:.3f}   "
          f"recall={recall*100:5.1f}%   FPR={fpr*100:5.1f}%  (thr={thr:.3f}, pct={threshold_pct})")
    return {"auroc": auroc, "recall": recall, "fpr": fpr, "threshold": thr}


def plot_auroc_comparison(results: dict, out_path: pathlib.Path):
    names  = list(results.keys())
    aurocs = [results[n]["auroc"] for n in names]
    colors = ["#4C78A8" if n != "combined (LR)" else "#F58518" for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, aurocs, color=colors, alpha=0.8)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random")
    ax.set_xlabel("AUROC")
    ax.set_title("Error detection — AUROC per signal (test set)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main(config_path: str, ckpt_path: str, out_dir: str):
    config  = OmegaConf.load(config_path)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(config.train.device)

    val_root  = pathlib.Path(config.dataset.val.params.root)
    test_root = val_root.parent / "test"
    if not test_root.exists():
        raise FileNotFoundError(f"Test features not found at {test_root}.")

    model = instantiate_model(config.model)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    print(f"Loaded checkpoint: epoch {state.get('epoch', '?')}\n")

    # Load Mahalanobis stats
    base = val_root.parent
    _r = np.load(base / "mahal_stats.npz", allow_pickle=True)
    resnet_stats = {"class_means": _r["class_means"].astype(np.float32),
                    "precision":   _r["precision"].astype(np.float32)}
    _s = np.load(base / "mstcn_mahal_stats.npz", allow_pickle=True)
    mstcn_stats  = {"class_means": _s["class_means"].astype(np.float32),
                    "precision":   _s["precision"].astype(np.float32)}
    print(f"ResNet stats: D={resnet_stats['class_means'].shape[1]}")
    print(f"MS-TCN stats: D={mstcn_stats['class_means'].shape[1]}\n")

    expected_dim = config.model.num_f_maps
    if mstcn_stats["class_means"].shape[1] != expected_dim:
        raise ValueError(
            f"mstcn_mahal_stats.npz D={mstcn_stats['class_means'].shape[1]} != "
            f"num_f_maps={expected_dim} — recompute stats."
        )

    # Collect features
    print("=== Collecting val features (train logistic regression) ===")
    X_val, y_val, _ = collect_features(model, val_root, device, resnet_stats, mstcn_stats)
    print(f"  Val: {len(y_val)} frames — {y_val.sum()} wrong ({y_val.mean()*100:.1f}%)\n")

    print("=== Collecting test features (evaluate) ===")
    X_test, y_test, X_ood_test = collect_features(model, test_root, device, resnet_stats, mstcn_stats)
    print(f"  Test: {len(y_test)} frames — {y_test.sum()} wrong ({y_test.mean()*100:.1f}%)")
    print(f"  Test OOD frames (GT=unknown): {len(X_ood_test)}\n")

    # Train logistic regression on val, evaluate on test
    feature_names = ["d_resnet", "d_mstcn", "entropy", "kl_inter_stage"]
    scaler = StandardScaler()
    X_val_s  = scaler.fit_transform(X_val)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_val_s, y_val)
    combined_score = lr.predict_proba(X_test_s)[:, 1]

    print("=== Logistic regression coefficients (standardized features) ===")
    for name, coef in zip(feature_names, lr.coef_[0]):
        print(f"  {name:<20}  coef={coef:+.4f}")

    # Evaluate all signals on test set
    print("\n=== Error detection — test set (threshold @ 85th pct of correct frames) ===")
    FEATURE_IDX = {"d_resnet": 0, "d_mstcn": 1, "entropy": 2, "kl_inter_stage": 3}
    results = {}
    for name, idx in FEATURE_IDX.items():
        results[name] = eval_signal(y_test, X_test[:, idx], name)
    results["combined (LR)"] = eval_signal(y_test, combined_score, "combined (LR)")

    # Complementarity analysis: which errors does each signal catch uniquely?
    print("\n=== Complementarity analysis — ResNet vs MS-TCN (at their 85th pct thresholds) ===")
    thr_resnet = results["d_resnet"]["threshold"]
    thr_mstcn  = results["d_mstcn"]["threshold"]

    flagged_resnet = X_test[:, 0] > thr_resnet  # (N,)
    flagged_mstcn  = X_test[:, 1] > thr_mstcn   # (N,)
    is_wrong       = y_test == 1

    n_errors = int(is_wrong.sum())
    only_resnet = is_wrong &  flagged_resnet & ~flagged_mstcn
    only_mstcn  = is_wrong & ~flagged_resnet &  flagged_mstcn
    both        = is_wrong &  flagged_resnet &  flagged_mstcn
    neither     = is_wrong & ~flagged_resnet & ~flagged_mstcn
    union       = is_wrong & (flagged_resnet |  flagged_mstcn)

    print(f"  Total error frames : {n_errors}")
    print(f"  Caught by ResNet only   : {only_resnet.sum():5d}  ({only_resnet.sum()/n_errors*100:5.1f}%)")
    print(f"  Caught by MS-TCN only   : {only_mstcn.sum():5d}  ({only_mstcn.sum()/n_errors*100:5.1f}%)")
    print(f"  Caught by BOTH          : {both.sum():5d}  ({both.sum()/n_errors*100:5.1f}%)")
    print(f"  Caught by NEITHER       : {neither.sum():5d}  ({neither.sum()/n_errors*100:5.1f}%)")
    print(f"  Caught by UNION (OR)    : {union.sum():5d}  ({union.sum()/n_errors*100:5.1f}%)  "
          f"FPR={(~is_wrong & (flagged_resnet | flagged_mstcn)).sum()/(~is_wrong).sum()*100:.1f}%")

    # Save metrics
    with open(out_dir / "error_detector_metrics.json", "w") as f:
        json.dump({k: {kk: round(v, 6) for kk, v in vv.items()} for k, vv in results.items()}, f, indent=2)
    print(f"\n  Saved: error_detector_metrics.json")

    plot_auroc_comparison(results, out_dir / "auroc_comparison.png")
    plot_complementarity(only_resnet.sum(), only_mstcn.sum(), both.sum(), neither.sum(),
                         n_errors, out_dir / "complementarity.png")

    # Key question: are unknown phase frames detectable by each signal?
    if len(X_ood_test) > 0:
        print("\n=== Signal recall on GT=unknown frames (the real target) ===")
        print(f"  {len(X_ood_test)} unknown-phase frames in test set")
        for name, idx in FEATURE_IDX.items():
            thr = results[name]["threshold"]
            recall_ood = float((X_ood_test[:, idx] > thr).mean())
            print(f"  {name:<30}  recall on unknown frames = {recall_ood*100:5.1f}%  (thr={thr:.3f})")
        thr_lr = results["combined (LR)"]["threshold"]
        ood_lr_score = lr.predict_proba(scaler.transform(X_ood_test))[:, 1]
        recall_ood_lr = float((ood_lr_score > thr_lr).mean())
        print(f"  {'combined (LR)':<30}  recall on unknown frames = {recall_ood_lr*100:5.1f}%  (thr={thr_lr:.3f})")
    else:
        print("\n  No OOD frames found in test set.")


def plot_complementarity(only_r, only_m, both, neither, total, out_path: pathlib.Path):
    labels = [
        f"ResNet only\n{only_r} ({only_r/total*100:.0f}%)",
        f"MS-TCN only\n{only_m} ({only_m/total*100:.0f}%)",
        f"Both\n{both} ({both/total*100:.0f}%)",
        f"Neither\n{neither} ({neither/total*100:.0f}%)",
    ]
    sizes  = [only_r, only_m, both, neither]
    colors = ["#4C78A8", "#E15759", "#F28E2B", "#CCCCCC"]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.legend(wedges, labels, loc="lower center", ncol=2, fontsize=9,
              bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.set_title(f"Error frames ({total}) — which signal catches them\n"
                 f"(ResNet & MS-TCN at their 85th pct threshold)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train logistic regression to detect MS-TCN wrong predictions")
    parser.add_argument("--config", type=str, default="phases_recognition/configs/config_mstcn.yaml")
    parser.add_argument("--ckpt",   type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.ckpt, args.out_dir)

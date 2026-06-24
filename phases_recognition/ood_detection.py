"""Open-set detection: identify unknown/novel surgical phases.

Compares three OOD signals on DINOv2 features:
  1. Mahalanobis distance in feature space (class-conditional)
  2. Entropy of ensemble weighted probabilities
  3. Disagreement between MSTCN and LSTM (L1 of softmax probs)

Evaluation: binary classification unknown (GT=-1) vs known.
Metrics: AUROC, AUPR, ROC + PR curves, score distributions.

Usage (from repo root):
    python phases_recognition/ood_detection.py
"""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    roc_curve, precision_recall_curve,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset.feature_dataset import VideoFeatureDataset, _collate_single_video
from models import instantiate_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TRAIN_ROOT = pathlib.Path("/home/helena/UCL_video_cataract/features_dino/train/")
TEST_ROOT  = pathlib.Path("/home/helena/UCL_video_cataract/features_dino/test/")

MSTCN_EXP = "/home/helena/experiments_cataract/mstcn_dino_v1_date=2026_06_11_17_02_41"
LSTM_EXP  = "/home/helena/experiments_cataract/lstm_dino_v1_date=2026_06_11_17_40_27"

OUT_DIR = pathlib.Path("/home/helena/experiments_cataract/ood_detection/")

CLASS_NAMES = [
    "Capsule_polishing", "Hydrodissection", "Incision",
    "Irrigation_and_aspiration", "Lens_implant_settingup",
    "Phacoemulsification", "Rhexis", "Tonifying_and_antibiotics",
    "Viscous_agent_injection", "Viscous_agent_removal", "Wound_hydration",
]

MSTCN_VAL_F1 = np.array([0.156, 0.634, 0.888, 0.587, 0.862, 0.920,
                          0.799, 0.327, 0.540, 0.801, 0.766])
LSTM_VAL_F1  = np.array([0.595, 0.607, 0.928, 0.527, 0.818, 0.874,
                          0.666, 0.271, 0.380, 0.692, 0.758])

# ---------------------------------------------------------------------------
# 1. Mahalanobis statistics from training features
# ---------------------------------------------------------------------------

def load_split_features(root: pathlib.Path, exclude_unknown: bool = True):
    feats_list, labels_list = [], []
    for feat_file in sorted(root.glob("*.npy")):
        stem = feat_file.stem
        if stem.endswith("_labels") or stem.endswith("_mahal"):
            continue
        label_file = root / f"{stem}_labels.npy"
        if not label_file.exists():
            continue
        feats  = np.load(feat_file).astype(np.float32)
        labels = np.load(label_file).astype(np.int32)
        if exclude_unknown:
            mask = labels >= 0
            feats, labels = feats[mask], labels[mask]
        feats_list.append(feats)
        labels_list.append(labels)
    return np.concatenate(feats_list), np.concatenate(labels_list)


def compute_mahal_stats(features: np.ndarray, labels: np.ndarray, class_names: list):
    n_classes = len(class_names)
    D = features.shape[1]

    class_means = np.zeros((n_classes, D), dtype=np.float32)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_means[c] = features[mask].mean(axis=0)

    centered = features.copy()
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            centered[mask] -= class_means[c]

    lw = LedoitWolf(assume_centered=True)
    lw.fit(centered)
    precision = lw.precision_.astype(np.float32)
    return class_means, precision


def mahal_scores(features: np.ndarray,
                 class_means: np.ndarray,
                 precision: np.ndarray) -> np.ndarray:
    """Returns (T,) — score = -min_c Mahal² (higher = more in-distribution)."""
    scores = np.full(len(features), -np.inf, dtype=np.float32)
    for c in range(len(class_means)):
        diff  = features - class_means[c]
        maha2 = (diff @ precision * diff).sum(axis=1)
        scores = np.maximum(scores, -maha2)
    return scores


# ---------------------------------------------------------------------------
# 2. Ensemble signals (entropy + disagreement)
# ---------------------------------------------------------------------------

def load_model(exp_dir: str, device: torch.device):
    cfg   = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_model(cfg.model)
    state = torch.load(f"{exp_dir}/ckpt/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def compute_ensemble_signals(mstcn, lstm, loader, device,
                              w_mstcn: np.ndarray, w_lstm: np.ndarray):
    """Returns arrays (N_total_frames,): entropy, disagreement, gt_binary."""
    entropy_all, disagree_all, gt_binary_all = [], [], []

    for features, labels, _ in loader:
        feat = features.unsqueeze(0).to(device)

        pm = F.softmax(mstcn(feat)[-1].squeeze(0).T, dim=1).cpu().numpy()  # (T, C)
        pl = F.softmax(lstm(feat)[-1].squeeze(0).T,  dim=1).cpu().numpy()

        pw = pm * w_mstcn[None, :] + pl * w_lstm[None, :]  # weighted ensemble
        H  = -(pw * np.log(pw + 1e-9)).sum(axis=1)          # entropy (T,)
        D  = np.abs(pm - pl).sum(axis=1)                     # L1 disagreement (T,)

        gt = labels.numpy().astype(np.int32)
        gt_bin = (gt == -1).astype(np.int32)                 # 1 = unknown

        entropy_all.append(H)
        disagree_all.append(D)
        gt_binary_all.append(gt_bin)

    return (np.concatenate(entropy_all),
            np.concatenate(disagree_all),
            np.concatenate(gt_binary_all))


# ---------------------------------------------------------------------------
# 3. Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_signal(scores: np.ndarray, gt_binary: np.ndarray, name: str):
    """Higher score = more likely OOD."""
    auroc = roc_auc_score(gt_binary, scores)
    aupr  = average_precision_score(gt_binary, scores)
    print(f"  {name:<25}  AUROC={auroc:.4f}   AUPR={aupr:.4f}")
    return auroc, aupr


def plot_roc_pr(signals: dict, gt_binary: np.ndarray, out_dir: pathlib.Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    for (name, scores), color in zip(signals.items(), colors):
        fpr, tpr, _ = roc_curve(gt_binary, scores)
        auroc = roc_auc_score(gt_binary, scores)
        axes[0].plot(fpr, tpr, color=color, label=f"{name} (AUROC={auroc:.3f})")

        prec, rec, _ = precision_recall_curve(gt_binary, scores)
        aupr = average_precision_score(gt_binary, scores)
        axes[1].plot(rec, prec, color=color, label=f"{name} (AUPR={aupr:.3f})")

    baseline = gt_binary.mean()
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].axhline(baseline, color="k", ls="--", lw=0.8, label=f"Baseline ({baseline:.3f})")

    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC curves — unknown phase detection")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR curves — unknown phase detection")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'roc_pr_curves.png'}")


def plot_score_distributions(signals: dict, gt_binary: np.ndarray, out_dir: pathlib.Path):
    n = len(signals)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, scores) in zip(axes, signals.items()):
        known   = scores[gt_binary == 0]
        unknown = scores[gt_binary == 1]
        bins = np.linspace(scores.min(), scores.max(), 60)
        ax.hist(known,   bins=bins, alpha=0.6, color="#377eb8", label="Known", density=True)
        ax.hist(unknown, bins=bins, alpha=0.6, color="#e41a1c", label="Unknown", density=True)
        ax.set(title=name, xlabel="Score", ylabel="Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Score distributions — known vs unknown phases", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'score_distributions.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -- Load features
    print("Loading training features...")
    train_feats, train_labels = load_split_features(TRAIN_ROOT, exclude_unknown=True)
    print(f"  {len(train_feats):,} frames  |  dim={train_feats.shape[1]}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:<35} {(train_labels == c).sum():>6} frames")

    print("\nLoading test features...")
    test_feats, test_labels = load_split_features(TEST_ROOT, exclude_unknown=False)
    gt_binary = (test_labels == -1).astype(np.int32)
    print(f"  {len(test_feats):,} test frames  |  unknown: {gt_binary.sum():,} ({gt_binary.mean()*100:.1f}%)")

    # -- Full 768-dim Mahalanobis
    print("\nFitting Mahalanobis (full 768 dims)...")
    class_means_full, prec_full = compute_mahal_stats(train_feats, train_labels, CLASS_NAMES)
    mahal_full_ood = -mahal_scores(test_feats, class_means_full, prec_full)

    # -- PCA + Mahalanobis at several dims
    pca_dims = [32, 64, 128]
    mahal_pca = {}
    for n_comp in pca_dims:
        print(f"Fitting PCA({n_comp}) + Mahalanobis...")
        pca = PCA(n_components=n_comp, random_state=42)
        train_pca = pca.fit_transform(train_feats).astype(np.float32)
        test_pca  = pca.transform(test_feats).astype(np.float32)
        cm, pr = compute_mahal_stats(train_pca, train_labels, CLASS_NAMES)
        mahal_pca[f"Mahal PCA-{n_comp}"] = -mahal_scores(test_pca, cm, pr)
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # -- Ensemble signals
    print("\nLoading ensemble models...")
    mstcn = load_model(MSTCN_EXP, device)
    lstm  = load_model(LSTM_EXP,  device)

    total = MSTCN_VAL_F1 + LSTM_VAL_F1
    w_mstcn = MSTCN_VAL_F1 / np.maximum(total, 1e-6)
    w_lstm  = LSTM_VAL_F1  / np.maximum(total, 1e-6)

    dataset = VideoFeatureDataset(root=str(TEST_ROOT))
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         collate_fn=_collate_single_video)

    print("Running ensemble inference...")
    entropy, disagree, gt_binary_ens = compute_ensemble_signals(
        mstcn, lstm, loader, device, w_mstcn, w_lstm
    )

    # Sanity check: frame counts should match
    assert len(mahal_full_ood) == len(entropy), \
        f"Frame count mismatch: Mahal={len(mahal_full_ood)}, ensemble={len(entropy)}"
    assert (gt_binary == gt_binary_ens).all(), "GT binary mismatch between loaders"

    # -- Combined signal: normalize each to [0,1] then average
    def norm01(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-9)

    # Use PCA-64 as the Mahalanobis signal in the combined score
    combined = (norm01(mahal_pca["Mahal PCA-64"]) + norm01(entropy) + norm01(disagree)) / 3.0

    # -- Evaluate
    print("\n--- OOD Detection Results ---")
    signals = {
        "Mahal full-768": mahal_full_ood,
        **mahal_pca,
        "Entropy":        entropy,
        "Disagreement":   disagree,
        "Combined":       combined,
    }

    print()
    for name, scores in signals.items():
        evaluate_signal(scores, gt_binary, name)

    # -- Plots
    print()
    plot_roc_pr(signals, gt_binary, OUT_DIR)
    plot_score_distributions(signals, gt_binary, OUT_DIR)

    # -- Save scores for further analysis
    np.savez(OUT_DIR / "ood_scores.npz",
             mahal_full=mahal_full_ood,
             mahal_pca64=mahal_pca["Mahal PCA-64"],
             entropy=entropy, disagree=disagree,
             combined=combined, gt_binary=gt_binary)
    print(f"Saved: {OUT_DIR / 'ood_scores.npz'}")
    print("\nDone.")


if __name__ == "__main__":
    main()

"""Open-set surgical phase detection — v2.

Compares KNN and RMDS (Relative Mahalanobis) OOD scores across 4 feature
spaces (DINOv2 768, MSTCN 64, LSTM 512, TeCNO 32).

Pipeline:
  1. Extract / load features from all 4 spaces (train + val + test)
  2. Leave-one-class-out (LOCO) on train/val → select best space
  3. Final evaluation on test set: per-unknown-phase AUROC
  4. Temporal smoothing via run-length thresholding

Usage (from repo root):
    python phases_recognition/ood_detection_v2.py
"""

import pathlib
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset.feature_dataset import VideoFeatureDataset, _collate_single_video
from models import instantiate_model

# ── Paths ──────────────────────────────────────────────────────────────────
FEAT_ROOT  = pathlib.Path("/home/helena/UCL_video_cataract/features_dino/")
MSTCN_EXP  = "/home/helena/experiments_cataract/mstcn_dino_v1_date=2026_06_11_17_02_41"
LSTM_EXP   = "/home/helena/experiments_cataract/lstm_dino_v1_date=2026_06_11_17_40_27"
OUT_DIR    = pathlib.Path("/home/helena/experiments_cataract/ood_detection_v2/")

CLASS_NAMES = [
    "Capsule_polishing", "Hydrodissection", "Incision",
    "Irrigation_and_aspiration", "Lens_implant_settingup",
    "Phacoemulsification", "Rhexis", "Tonifying_and_antibiotics",
    "Viscous_agent_injection", "Viscous_agent_removal", "Wound_hydration",
]
N_CLASSES = len(CLASS_NAMES)

UNKNOWN_PHASES = [
    "Malyugin_ring_insertion", "Malyugin_ring_removal",
    "Suture", "Iris_manipulation", "Trypan_blue_injection",
]

KNN_K = 10
SMOOTH_MIN_RUN = 5  # minimum run length to keep an OOD segment


# ── 1. Feature extraction ─────────────────────────────────────────────────

def load_dino_features(split: str):
    """Load pre-extracted DINOv2 features. Returns (feats, labels) per video."""
    root = FEAT_ROOT / split
    videos = []
    for feat_file in sorted(root.glob("*.npy")):
        if feat_file.stem.endswith(("_labels", "_mahal", "_binary_ch")):
            continue
        label_file = root / f"{feat_file.stem}_labels.npy"
        if not label_file.exists():
            continue
        feats = np.load(feat_file).astype(np.float32)
        labels = np.load(label_file).astype(np.int32)
        videos.append((feat_file.stem, feats, labels))
    return videos


def load_model(exp_dir: str, device: torch.device):
    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_model(cfg.model)
    state = torch.load(f"{exp_dir}/ckpt/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def extract_temporal_features(mstcn, lstm, split: str, device):
    """Extract MSTCN-64, LSTM-512, TeCNO-32 features for a split."""
    dataset = VideoFeatureDataset(root=str(FEAT_ROOT / split))
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=_collate_single_video)

    mstcn_feats, lstm_feats, tecno_feats = [], [], []

    for features, _, _ in loader:
        feat = features.unsqueeze(0).to(device)

        # MSTCN: 64-dim from last stage
        _, f_mstcn = mstcn.forward_with_features(feat)
        mstcn_feats.append(f_mstcn.squeeze(0).T.cpu().numpy())  # (T, 64)

        # LSTM: 512-dim hidden state before output_proj
        captured_lstm = []
        h1 = lstm.lstm_stage.output_proj.register_forward_hook(
            lambda m, inp, out: captured_lstm.append(inp[0].detach().cpu()))
        # TeCNO: 32-dim before last refinement output_proj
        captured_tecno = []
        h2 = lstm.refinement_stages[-1].output_proj.register_forward_hook(
            lambda m, inp, out: captured_tecno.append(inp[0].detach().cpu()))

        lstm(feat)

        lstm_feats.append(captured_lstm[0].squeeze(0).numpy())       # (T, 512)
        tecno_feats.append(captured_tecno[0].squeeze(0).T.numpy())   # Conv1d: (B,C,T)→(T,C)

        h1.remove()
        h2.remove()

    return mstcn_feats, lstm_feats, tecno_feats


def build_feature_dict(split: str, mstcn, lstm, device):
    """Returns dict[space_name] → list of (feats_array, labels_array) per video."""
    print(f"  Loading {split} features...")
    dino_videos = load_dino_features(split)
    mstcn_feats, lstm_feats, tecno_feats = extract_temporal_features(
        mstcn, lstm, split, device)

    spaces = {"DINOv2-768": [], "MSTCN-64": [], "LSTM-512": [], "TeCNO-32": []}
    for i, (name, dino_f, labels) in enumerate(dino_videos):
        spaces["DINOv2-768"].append((dino_f, labels))
        spaces["MSTCN-64"].append((mstcn_feats[i], labels))
        spaces["LSTM-512"].append((lstm_feats[i], labels))
        spaces["TeCNO-32"].append((tecno_feats[i], labels))

    return spaces


def concat_space(video_list, exclude_unknown=True):
    """Concatenate per-video (feats, labels) into flat arrays."""
    feats_all, labels_all = [], []
    for feats, labels in video_list:
        if exclude_unknown:
            mask = labels >= 0
            feats_all.append(feats[mask])
            labels_all.append(labels[mask])
        else:
            feats_all.append(feats)
            labels_all.append(labels)
    return np.concatenate(feats_all), np.concatenate(labels_all)


# ── 2. OOD scoring ────────────────────────────────────────────────────────

def fit_knn(train_feats: np.ndarray, k: int = KNN_K):
    nn_model = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn_model.fit(train_feats)
    return nn_model


def knn_scores(nn_model: NearestNeighbors, test_feats: np.ndarray):
    """Higher = more OOD."""
    dists, _ = nn_model.kneighbors(test_feats)
    return dists[:, -1]  # distance to k-th neighbor


def fit_rmds(train_feats: np.ndarray, train_labels: np.ndarray, n_classes: int):
    """Fit class-conditional Gaussians + background Gaussian with Ledoit-Wolf."""
    class_means = np.zeros((n_classes, train_feats.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        mask = train_labels == c
        if mask.sum() > 0:
            class_means[c] = train_feats[mask].mean(axis=0)

    # Class-conditional shared covariance
    centered = train_feats.copy()
    for c in range(n_classes):
        mask = train_labels == c
        if mask.sum() > 0:
            centered[mask] -= class_means[c]
    lw_class = LedoitWolf(assume_centered=True)
    lw_class.fit(centered)
    prec_class = lw_class.precision_.astype(np.float32)

    # Background Gaussian (all data, no labels)
    bg_mean = train_feats.mean(axis=0).astype(np.float32)
    lw_bg = LedoitWolf()
    lw_bg.fit(train_feats)
    prec_bg = lw_bg.precision_.astype(np.float32)

    return class_means, prec_class, bg_mean, prec_bg


def _mahal_min_class(feats, class_means, precision):
    """Min Mahalanobis² across classes. Returns (T,) — lower = closer to a class."""
    best = np.full(len(feats), np.inf, dtype=np.float32)
    for c in range(len(class_means)):
        diff = feats - class_means[c]
        maha2 = (diff @ precision * diff).sum(axis=1)
        best = np.minimum(best, maha2)
    return best


def rmds_scores(feats, class_means, prec_class, bg_mean, prec_bg):
    """RMDS = D_class - D_background. Higher = more OOD."""
    d_class = _mahal_min_class(feats, class_means, prec_class)
    diff_bg = feats - bg_mean
    d_bg = (diff_bg @ prec_bg * diff_bg).sum(axis=1)
    return d_class - d_bg


# ── 3. Leave-one-class-out selection ──────────────────────────────────────

def loco_evaluate(spaces_train: dict, spaces_val: dict, n_classes: int):
    """Leave-one-class-out: hold out each known class as pseudo-OOD.

    Uses train split to fit KNN/RMDS, val split for evaluation.
    Returns dict[space_name][method] → mean AUROC over held-out classes.
    """
    results = {}

    for space_name in spaces_train:
        print(f"\n  LOCO — {space_name}")
        train_feats, train_labels = concat_space(spaces_train[space_name])
        val_feats, val_labels = concat_space(spaces_val[space_name])

        aurocs_knn = []
        aurocs_rmds = []

        for held_out in range(n_classes):
            # Fit on train without held-out class
            fit_mask = train_labels != held_out
            fit_feats = train_feats[fit_mask]
            fit_labels = train_labels[fit_mask]

            # Remap labels to be contiguous (not strictly necessary for KNN)
            remaining_classes = sorted(set(fit_labels.tolist()))
            n_remaining = len(remaining_classes)

            # Eval: val frames of held-out = pseudo-OOD, rest = ID
            val_ood_mask = val_labels == held_out
            val_id_mask = (val_labels >= 0) & (val_labels != held_out)

            if val_ood_mask.sum() < 5 or val_id_mask.sum() < 5:
                continue

            eval_feats = np.concatenate([val_feats[val_id_mask], val_feats[val_ood_mask]])
            gt_binary = np.concatenate([
                np.zeros(val_id_mask.sum(), dtype=np.int32),
                np.ones(val_ood_mask.sum(), dtype=np.int32),
            ])

            # KNN
            nn_model = fit_knn(fit_feats)
            scores_knn = knn_scores(nn_model, eval_feats)
            try:
                aurocs_knn.append(roc_auc_score(gt_binary, scores_knn))
            except ValueError:
                pass

            # RMDS
            class_means, prec_c, bg_mean, prec_bg = fit_rmds(
                fit_feats, fit_labels, n_classes)
            scores_rmds = rmds_scores(eval_feats, class_means, prec_c, bg_mean, prec_bg)
            try:
                aurocs_rmds.append(roc_auc_score(gt_binary, scores_rmds))
            except ValueError:
                pass

        mean_knn = np.mean(aurocs_knn) if aurocs_knn else 0.0
        mean_rmds = np.mean(aurocs_rmds) if aurocs_rmds else 0.0
        std_knn = np.std(aurocs_knn) if aurocs_knn else 0.0
        std_rmds = np.std(aurocs_rmds) if aurocs_rmds else 0.0

        print(f"    KNN  AUROC: {mean_knn:.4f} ± {std_knn:.4f}  ({len(aurocs_knn)} folds)")
        print(f"    RMDS AUROC: {mean_rmds:.4f} ± {std_rmds:.4f}  ({len(aurocs_rmds)} folds)")

        results[space_name] = {
            "KNN": {"mean": mean_knn, "std": std_knn, "per_class": aurocs_knn},
            "RMDS": {"mean": mean_rmds, "std": std_rmds, "per_class": aurocs_rmds},
        }

    return results


# ── 4. Final test evaluation ──────────────────────────────────────────────

def map_unknown_phases(test_videos):
    """Map -1 labels to specific unknown phase names using labels_ood.json."""
    import json, re
    ood_path = pathlib.Path("/home/helena/UCL_video_cataract/dataset_temporal/labels_ood.json")
    with open(ood_path) as f:
        ood_data = json.load(f)

    unk_names_all = []
    for name, feats, labels in test_videos:
        T = len(labels)
        prefix = f"test/{name}/"
        frame_phase = {}
        for key, phase in ood_data.items():
            if not key.startswith(prefix):
                continue
            m = re.search(r"Frame_(\d+)", key)
            if m:
                frame_phase[int(m.group(1))] = phase

        unk_names = np.full(T, "", dtype=object)
        if frame_phase:
            sorted_frames = sorted(frame_phase.keys())
            step = max(1, sorted_frames[-1] // T)
            for t in range(T):
                if labels[t] != -1:
                    continue
                approx_frame = t * step
                closest = min(sorted_frames, key=lambda f: abs(f - approx_frame))
                unk_names[t] = frame_phase[closest]
        unk_names_all.append(unk_names)

    return unk_names_all


def temporal_smooth(scores: np.ndarray, min_run: int = SMOOTH_MIN_RUN):
    """Run-length thresholding: suppress isolated OOD spikes."""
    smoothed = scores.copy()
    median = np.median(scores)
    high = scores > median

    i = 0
    while i < len(high):
        if high[i]:
            j = i
            while j < len(high) and high[j]:
                j += 1
            if (j - i) < min_run:
                smoothed[i:j] = median
            i = j
        else:
            i += 1
    return smoothed


def evaluate_test(spaces_train, spaces_test, test_dino_videos,
                  method: str, space_name: str):
    """Evaluate on test set with per-unknown-phase AUROC."""
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION — {space_name} / {method}")
    print(f"{'='*60}")

    # Fit on full train set (all 11 known classes)
    train_feats, train_labels = concat_space(spaces_train[space_name])

    # Test: include unknowns
    test_feats_list = [f for f, _ in spaces_test[space_name]]
    test_labels_list = [l for _, l in spaces_test[space_name]]
    test_feats = np.concatenate(test_feats_list)
    test_labels = np.concatenate(test_labels_list)

    # Map unknown phases
    unk_names_all = map_unknown_phases(test_dino_videos)
    unk_names = np.concatenate(unk_names_all)

    # Exclude Corneal_hydration
    keep_mask = ~((test_labels == -1) & (unk_names == "Corneal_hydration"))
    test_feats = test_feats[keep_mask]
    test_labels = test_labels[keep_mask]
    unk_names = unk_names[keep_mask]

    gt_binary = (test_labels == -1).astype(np.int32)

    # Score
    if method == "KNN":
        nn_model = fit_knn(train_feats)
        raw_scores = knn_scores(nn_model, test_feats)
    else:
        class_means, prec_c, bg_mean, prec_bg = fit_rmds(
            train_feats, train_labels, N_CLASSES)
        raw_scores = rmds_scores(test_feats, class_means, prec_c, bg_mean, prec_bg)

    # Temporal smoothing (per-video)
    smoothed_scores = np.empty_like(raw_scores)
    offset = 0
    for feats_v, labels_v in spaces_test[space_name]:
        T = len(labels_v)
        keep_v = ~((labels_v == -1) &
                    np.array([unk_names[offset + t] == "Corneal_hydration"
                              for t in range(T) if offset + t < len(unk_names)],
                             dtype=bool)) if False else np.ones(T, dtype=bool)
        n_keep = keep_v.sum()
        smoothed_scores[offset:offset + n_keep] = temporal_smooth(
            raw_scores[offset:offset + n_keep])
        offset += n_keep

    # Overall AUROC
    for label, scores in [("raw", raw_scores), ("smoothed", smoothed_scores)]:
        auroc = roc_auc_score(gt_binary, scores)
        aupr = average_precision_score(gt_binary, scores)
        print(f"\n  [{label}] Overall: AUROC={auroc:.4f}  AUPR={aupr:.4f}")

    # Per-unknown-phase AUROC
    print(f"\n  Per-phase AUROC (smoothed):")
    per_phase = {}
    for phase_name in UNKNOWN_PHASES:
        phase_mask = (test_labels == -1) & (unk_names == phase_name)
        n_phase = phase_mask.sum()
        if n_phase < 5:
            print(f"    {phase_name:<30} SKIP (n={n_phase})")
            continue

        # Binary: this unknown phase vs all known
        eval_mask = (test_labels >= 0) | phase_mask
        gt_phase = phase_mask[eval_mask].astype(np.int32)
        scores_phase = smoothed_scores[eval_mask]

        try:
            auroc = roc_auc_score(gt_phase, scores_phase)
            aupr = average_precision_score(gt_phase, scores_phase)
            per_phase[phase_name] = auroc
            print(f"    {phase_name:<30} AUROC={auroc:.4f}  AUPR={aupr:.4f}  (n={n_phase})")
        except ValueError:
            print(f"    {phase_name:<30} ERROR")

    return raw_scores, smoothed_scores, gt_binary, unk_names, per_phase


# ── 5. Plots ──────────────────────────────────────────────────────────────

def plot_loco_results(loco_results: dict, out_dir: pathlib.Path):
    """Bar chart: LOCO AUROC per space, KNN vs RMDS."""
    spaces = list(loco_results.keys())
    knn_means = [loco_results[s]["KNN"]["mean"] for s in spaces]
    knn_stds = [loco_results[s]["KNN"]["std"] for s in spaces]
    rmds_means = [loco_results[s]["RMDS"]["mean"] for s in spaces]
    rmds_stds = [loco_results[s]["RMDS"]["std"] for s in spaces]

    x = np.arange(len(spaces))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, knn_means, w, yerr=knn_stds, label="KNN", capsize=4,
           color="#1f77b4")
    ax.bar(x + w/2, rmds_means, w, yerr=rmds_stds, label="RMDS", capsize=4,
           color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(spaces, fontsize=11)
    ax.set_ylabel("AUROC (leave-one-class-out)", fontsize=12)
    ax.set_title("Feature space selection — LOCO validation", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "loco_selection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'loco_selection.png'}")


def plot_per_phase_auroc(per_phase: dict, space: str, method: str,
                         out_dir: pathlib.Path):
    """Horizontal bar chart: AUROC per unknown phase."""
    phases = list(per_phase.keys())
    aurocs = [per_phase[p] for p in phases]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e41a1c" if a < 0.7 else "#4daf4a" if a > 0.85 else "#ff7f00"
              for a in aurocs]
    ax.barh(phases, aurocs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUROC", fontsize=12)
    ax.set_title(f"Per-phase AUROC — {space} / {method}", fontsize=13)
    ax.axvline(0.5, color="grey", ls="--", lw=0.8)
    for i, v in enumerate(aurocs):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "per_phase_auroc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'per_phase_auroc.png'}")


def plot_all_spaces_test(spaces_train, spaces_test, test_dino_videos, out_dir):
    """Full comparison table: 4 spaces × 2 methods on test set."""
    print("\n" + "=" * 60)
    print("FULL COMPARISON — all spaces × all methods on test set")
    print("=" * 60)

    rows = []
    for space_name in ["DINOv2-768", "MSTCN-64", "LSTM-512", "TeCNO-32"]:
        train_feats, train_labels = concat_space(spaces_train[space_name])

        test_feats_list = [f for f, _ in spaces_test[space_name]]
        test_labels_list = [l for _, l in spaces_test[space_name]]
        test_feats = np.concatenate(test_feats_list)
        test_labels = np.concatenate(test_labels_list)

        unk_names_all = map_unknown_phases(test_dino_videos)
        unk_names = np.concatenate(unk_names_all)

        keep = ~((test_labels == -1) & (unk_names == "Corneal_hydration"))
        test_feats, test_labels, unk_names = test_feats[keep], test_labels[keep], unk_names[keep]
        gt_binary = (test_labels == -1).astype(np.int32)

        for method in ["KNN", "RMDS"]:
            if method == "KNN":
                nn_model = fit_knn(train_feats)
                scores = knn_scores(nn_model, test_feats)
            else:
                cm, pc, bm, pb = fit_rmds(train_feats, train_labels, N_CLASSES)
                scores = rmds_scores(test_feats, cm, pc, bm, pb)

            auroc = roc_auc_score(gt_binary, scores)
            aupr = average_precision_score(gt_binary, scores)
            rows.append((space_name, method, auroc, aupr))
            print(f"  {space_name:<12} {method:<6}  AUROC={auroc:.4f}  AUPR={aupr:.4f}")

    # Save as table
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table_data = [["Space", "Method", "AUROC", "AUPR"]]
    for s, m, auroc, aupr in rows:
        table_data.append([s, m, f"{auroc:.4f}", f"{aupr:.4f}"])
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    # Bold header
    for j in range(4):
        table[0, j].set_text_props(fontweight="bold")
    # Highlight best AUROC
    best_auroc = max(r[2] for r in rows)
    for i, (s, m, auroc, aupr) in enumerate(rows, start=1):
        if auroc == best_auroc:
            for j in range(4):
                table[i, j].set_facecolor("#d4edda")
    ax.set_title("OOD Detection — Full comparison (test set)", fontsize=14, pad=20)
    fig.tight_layout()
    fig.savefig(out_dir / "full_comparison_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'full_comparison_table.png'}")

    return rows


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading models...")
    mstcn = load_model(MSTCN_EXP, device)
    lstm = load_model(LSTM_EXP, device)

    # Extract features for all splits
    print("\nExtracting features...")
    spaces_train = build_feature_dict("train", mstcn, lstm, device)
    spaces_val = build_feature_dict("val", mstcn, lstm, device)
    spaces_test = build_feature_dict("test", mstcn, lstm, device)

    # Quick stats
    for split_name, spaces in [("train", spaces_train), ("val", spaces_val), ("test", spaces_test)]:
        n_videos = len(spaces["DINOv2-768"])
        n_frames = sum(len(l) for _, l in spaces["DINOv2-768"])
        n_unk = sum((l == -1).sum() for _, l in spaces["DINOv2-768"])
        print(f"  {split_name}: {n_videos} videos, {n_frames:,} frames, {n_unk:,} unknown")

    # ── Step 1: LOCO selection ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LEAVE-ONE-CLASS-OUT SELECTION")
    print("=" * 60)
    loco_results = loco_evaluate(spaces_train, spaces_val, N_CLASSES)
    plot_loco_results(loco_results, OUT_DIR)

    # Find best (space, method)
    best_auroc, best_space, best_method = 0, None, None
    for space, methods in loco_results.items():
        for method, vals in methods.items():
            if vals["mean"] > best_auroc:
                best_auroc = vals["mean"]
                best_space = space
                best_method = method
    print(f"\n>>> Best LOCO: {best_space} / {best_method}  (AUROC={best_auroc:.4f})")

    # ── Step 2: Full comparison table on test set ─────────────────────────
    test_dino_videos = load_dino_features("test")
    all_rows = plot_all_spaces_test(spaces_train, spaces_test, test_dino_videos, OUT_DIR)

    # ── Step 3: Detailed evaluation for LOCO winner ───────────────────────
    raw, smoothed, gt, unk, per_phase = evaluate_test(
        spaces_train, spaces_test, test_dino_videos,
        best_method, best_space)

    if per_phase:
        plot_per_phase_auroc(per_phase, best_space, best_method, OUT_DIR)

    # ── Save all scores ───────────────────────────────────────────────────
    np.savez(OUT_DIR / "ood_scores_v2.npz",
             raw_scores=raw, smoothed_scores=smoothed,
             gt_binary=gt, unk_names=unk,
             best_space=best_space, best_method=best_method)
    print(f"\nSaved: {OUT_DIR / 'ood_scores_v2.npz'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  LOCO winner     : {best_space} / {best_method} (AUROC={best_auroc:.4f})")
    print(f"  Test overall    : see full_comparison_table.png")
    print(f"  Per-phase AUROC : see per_phase_auroc.png")
    print(f"  Output dir      : {OUT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()

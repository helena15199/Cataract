"""OOD fusion experiment: combine DINOv2 + MSTCN-64 scores.

DINOv2 preserves visual novelty (appearance), MSTCN-64 preserves temporal
novelty (phase structure). The Malyugin phases that MSTCN absorbs might
still be visually atypical in DINOv2 → late fusion could recover them.

Pipeline:
  Étage 0: Per-phase AUROC in DINOv2 and MSTCN-64 separately (KNN + RMDS)
           → diagnose complementarity before fusing
  Étage 1: Rank normalization calibrated on val known frames (no leakage)
  Étage 2: Fusion rules — mean and max of normalized scores
  Étage 3: Per-phase AUROC on fused scores (Malyugin↑ without Suture↓?)

Usage:
    python phases_recognition/ood_fusion.py
"""

import json
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset.feature_dataset import VideoFeatureDataset, _collate_single_video
from models import instantiate_model

# ── Paths ──────────────────────────────────────────────────────────────────
FEAT_ROOT = pathlib.Path("/home/helena/UCL_video_cataract/features_dino/")
MSTCN_EXP = "/home/helena/experiments_cataract/mstcn_dino_v1_date=2026_06_11_17_02_41"
LSTM_EXP  = "/home/helena/experiments_cataract/lstm_dino_v1_date=2026_06_11_17_40_27"
OOD_JSON  = pathlib.Path("/home/helena/UCL_video_cataract/dataset_temporal/labels_ood.json")
OUT_DIR   = pathlib.Path("/home/helena/experiments_cataract/ood_fusion/")

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


# ── Feature loading ───────────────────────────────────────────────────────

def load_dino_split(split: str):
    root = FEAT_ROOT / split
    videos = []
    for f in sorted(root.glob("*.npy")):
        if f.stem.endswith(("_labels", "_mahal", "_binary_ch")):
            continue
        lf = root / f"{f.stem}_labels.npy"
        if not lf.exists():
            continue
        videos.append((f.stem, np.load(f).astype(np.float32),
                        np.load(lf).astype(np.int32)))
    return videos


def load_model(exp_dir, device):
    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_model(cfg.model)
    state = torch.load(f"{exp_dir}/ckpt/best.pt", map_location="cpu",
                       weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def extract_mstcn_features(mstcn, split, device):
    ds = VideoFeatureDataset(root=str(FEAT_ROOT / split))
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=_collate_single_video)
    out = []
    for features, _, _ in loader:
        _, f = mstcn.forward_with_features(features.unsqueeze(0).to(device))
        out.append(f.squeeze(0).T.cpu().numpy())  # (T, 64)
    return out


def map_unknown_phases(test_videos):
    with open(OOD_JSON) as f:
        ood_data = json.load(f)
    all_unk = []
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
        unk = np.full(T, "", dtype=object)
        if frame_phase:
            sf = sorted(frame_phase.keys())
            step = max(1, sf[-1] // T)
            for t in range(T):
                if labels[t] != -1:
                    continue
                approx = t * step
                closest = min(sf, key=lambda x: abs(x - approx))
                unk[t] = frame_phase[closest]
        all_unk.append(unk)
    return all_unk


# ── Scoring functions ─────────────────────────────────────────────────────

def fit_knn(train_feats, k=KNN_K):
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn.fit(train_feats)
    return nn


def score_knn(nn_model, feats):
    """Higher = more OOD."""
    dists, _ = nn_model.kneighbors(feats)
    return dists[:, -1]


def fit_rmds(train_feats, train_labels, n_classes):
    D = train_feats.shape[1]
    class_means = np.zeros((n_classes, D), dtype=np.float32)
    for c in range(n_classes):
        mask = train_labels == c
        if mask.sum() > 0:
            class_means[c] = train_feats[mask].mean(0)

    centered = train_feats.copy()
    for c in range(n_classes):
        mask = train_labels == c
        if mask.sum() > 0:
            centered[mask] -= class_means[c]
    lw = LedoitWolf(assume_centered=True)
    lw.fit(centered)
    prec_class = lw.precision_.astype(np.float32)

    bg_mean = train_feats.mean(0).astype(np.float32)
    lw_bg = LedoitWolf()
    lw_bg.fit(train_feats)
    prec_bg = lw_bg.precision_.astype(np.float32)

    return class_means, prec_class, bg_mean, prec_bg


def score_rmds(feats, class_means, prec_class, bg_mean, prec_bg):
    """RMDS = D_class - D_bg. Higher = more OOD."""
    best_class = np.full(len(feats), np.inf, dtype=np.float32)
    for c in range(len(class_means)):
        diff = feats - class_means[c]
        d = (diff @ prec_class * diff).sum(1)
        best_class = np.minimum(best_class, d)

    diff_bg = feats - bg_mean
    d_bg = (diff_bg @ prec_bg * diff_bg).sum(1)
    return best_class - d_bg


# ── Rank normalization (calibrated on val known frames) ───────────────────

def rank_normalize(test_scores: np.ndarray, val_known_scores: np.ndarray):
    """Normalize test scores by their rank in the val-known distribution.

    For each test frame: what fraction of val-known frames have a lower score?
    Result in [0, 1], calibrated on ID data only → no leakage.
    """
    sorted_val = np.sort(val_known_scores)
    n_val = len(sorted_val)
    normalized = np.searchsorted(sorted_val, test_scores, side="right") / n_val
    return normalized.astype(np.float32)


# ── Evaluation ────────────────────────────────────────────────────────────

def eval_per_phase(scores, test_labels, unk_names, label=""):
    gt = (test_labels == -1).astype(np.int32)
    overall_auroc = roc_auc_score(gt, scores)
    overall_aupr = average_precision_score(gt, scores)

    print(f"\n  {label}")
    print(f"    Overall:  AUROC={overall_auroc:.4f}  AUPR={overall_aupr:.4f}")

    per_phase = {}
    for phase in UNKNOWN_PHASES:
        mask_p = (test_labels == -1) & (unk_names == phase)
        n = mask_p.sum()
        if n < 5:
            print(f"    {phase:<30} SKIP (n={n})")
            continue
        eval_mask = (test_labels >= 0) | mask_p
        gt_p = mask_p[eval_mask].astype(np.int32)
        s_p = scores[eval_mask]
        try:
            auroc = roc_auc_score(gt_p, s_p)
            per_phase[phase] = auroc
            print(f"    {phase:<30} AUROC={auroc:.4f}  (n={n})")
        except ValueError:
            print(f"    {phase:<30} ERROR")

    return overall_auroc, per_phase


# ── Plots ─────────────────────────────────────────────────────────────────

def plot_comparison(all_results, out_dir):
    methods = list(all_results.keys())
    active_phases = [p for p in UNKNOWN_PHASES
                     if any(p in all_results[m][1] for m in methods)]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(active_phases))
    n_methods = len(methods)
    w = 0.8 / n_methods
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#17becf"]

    for i, method in enumerate(methods):
        _, per_phase = all_results[method]
        vals = [per_phase.get(p, 0) for p in active_phases]
        offset = (i - n_methods / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=method,
                      color=colors[i % len(colors)],
                      edgecolor="black", linewidth=0.3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in active_phases], fontsize=9)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Per-phase AUROC — DINOv2 vs MSTCN-64 vs Fusion", fontsize=14)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="chance")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "fusion_per_phase_auroc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")


def plot_summary_table(all_results, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    header = ["Method", "Overall"] + [p.replace("_", " ") for p in UNKNOWN_PHASES]
    rows = [header]
    for method in all_results:
        overall, per_phase = all_results[method]
        row = [method, f"{overall:.4f}"]
        for p in UNKNOWN_PHASES:
            if p in per_phase:
                v = per_phase[p]
                row.append(f"{v:.4f}")
            else:
                row.append("—")
        rows.append(row)

    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    for j in range(len(header)):
        table[0, j].set_text_props(fontweight="bold")

    best_overall = max(all_results[m][0] for m in all_results)
    for i, method in enumerate(all_results, start=1):
        if all_results[method][0] == best_overall:
            for j in range(len(header)):
                table[i, j].set_facecolor("#d4edda")

    ax.set_title("OOD Detection — Fusion experiment", fontsize=14, pad=20)
    fig.tight_layout()
    path = out_dir / "fusion_summary_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_diagnostic(diag_results, out_dir):
    """Étage 0 diagnostic: side-by-side per-phase for DINOv2 vs MSTCN."""
    spaces = ["DINOv2-768", "MSTCN-64"]
    methods_per_space = ["KNN", "RMDS"]
    active_phases = [p for p in UNKNOWN_PHASES
                     if any(p in diag_results.get(f"{s} / {m}", ({}, {}))[1]
                            for s in spaces for m in methods_per_space)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    colors_method = {"KNN": "#1f77b4", "RMDS": "#ff7f0e"}

    for ax, space in zip(axes, spaces):
        x = np.arange(len(active_phases))
        w = 0.35
        for i, method in enumerate(methods_per_space):
            key = f"{space} / {method}"
            _, per_phase = diag_results.get(key, (0, {}))
            vals = [per_phase.get(p, 0) for p in active_phases]
            offset = (i - 0.5) * w
            bars = ax.bar(x + offset, vals, w, label=method,
                          color=colors_method[method], edgecolor="black", lw=0.3)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                            f"{v:.2f}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_", "\n") for p in active_phases], fontsize=8)
        ax.set_title(space, fontsize=13, fontweight="bold")
        ax.axhline(0.5, color="grey", ls="--", lw=0.8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("AUROC", fontsize=12)
    fig.suptitle("Étage 0 — Diagnostic per-phase: DINOv2 vs MSTCN-64",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = out_dir / "diagnostic_per_phase.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Load models + features ────────────────────────────────────────────
    print("Loading models...")
    mstcn = load_model(MSTCN_EXP, device)

    print("Extracting features (train / val / test)...")
    train_dino = load_dino_split("train")
    val_dino = load_dino_split("val")
    test_dino = load_dino_split("test")

    train_mstcn = extract_mstcn_features(mstcn, "train", device)
    val_mstcn = extract_mstcn_features(mstcn, "val", device)
    test_mstcn = extract_mstcn_features(mstcn, "test", device)

    # ── Concatenate splits ────────────────────────────────────────────────
    # Train: known frames only (for fitting KNN / RMDS)
    train_dino_f = np.concatenate([f for _, f, l in train_dino])
    train_labels = np.concatenate([l for _, _, l in train_dino])
    train_mstcn_f = np.concatenate(train_mstcn)
    mask_train_known = train_labels >= 0
    train_dino_f = train_dino_f[mask_train_known]
    train_mstcn_f = train_mstcn_f[mask_train_known]
    train_labels = train_labels[mask_train_known]

    # Val: known frames only (for calibrating rank normalization)
    val_dino_f = np.concatenate([f for _, f, l in val_dino])
    val_labels = np.concatenate([l for _, _, l in val_dino])
    val_mstcn_f = np.concatenate(val_mstcn)
    mask_val_known = val_labels >= 0
    val_dino_f = val_dino_f[mask_val_known]
    val_mstcn_f = val_mstcn_f[mask_val_known]

    # Test: keep unknowns, exclude Corneal_hydration
    test_dino_f = np.concatenate([f for _, f, l in test_dino])
    test_labels = np.concatenate([l for _, _, l in test_dino])
    test_mstcn_f = np.concatenate(test_mstcn)
    unk_names = np.concatenate(map_unknown_phases(test_dino))

    keep = ~((test_labels == -1) & (unk_names == "Corneal_hydration"))
    test_dino_f = test_dino_f[keep]
    test_mstcn_f = test_mstcn_f[keep]
    test_labels = test_labels[keep]
    unk_names = unk_names[keep]

    n_known = (test_labels >= 0).sum()
    n_unk = (test_labels == -1).sum()
    print(f"\nSplits ready:")
    print(f"  Train: {len(train_dino_f):,} known frames")
    print(f"  Val:   {mask_val_known.sum():,} known frames (for normalization)")
    print(f"  Test:  {n_known:,} known + {n_unk:,} unknown frames")

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAGE 0 — Diagnostic per-phase per-space
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("ÉTAGE 0 — DIAGNOSTIC (per-phase, before fusion)")
    print("=" * 60)

    # Fit scorers on train
    print("\nFitting DINOv2-768...")
    knn_dino = fit_knn(train_dino_f)
    rmds_dino_params = fit_rmds(train_dino_f, train_labels, N_CLASSES)

    print("Fitting MSTCN-64...")
    knn_mstcn = fit_knn(train_mstcn_f)
    rmds_mstcn_params = fit_rmds(train_mstcn_f, train_labels, N_CLASSES)

    # Score test
    s_dino_knn = score_knn(knn_dino, test_dino_f)
    s_dino_rmds = score_rmds(test_dino_f, *rmds_dino_params)
    s_mstcn_knn = score_knn(knn_mstcn, test_mstcn_f)
    s_mstcn_rmds = score_rmds(test_mstcn_f, *rmds_mstcn_params)

    # Score val (known only — for rank normalization calibration)
    val_s_dino_knn = score_knn(knn_dino, val_dino_f)
    val_s_dino_rmds = score_rmds(val_dino_f, *rmds_dino_params)
    val_s_mstcn_knn = score_knn(knn_mstcn, val_mstcn_f)
    val_s_mstcn_rmds = score_rmds(val_mstcn_f, *rmds_mstcn_params)

    # Diagnostic evaluation (raw scores)
    diag = {}
    for name, scores in [("DINOv2-768 / KNN", s_dino_knn),
                          ("DINOv2-768 / RMDS", s_dino_rmds),
                          ("MSTCN-64 / KNN", s_mstcn_knn),
                          ("MSTCN-64 / RMDS", s_mstcn_rmds)]:
        overall, per_phase = eval_per_phase(scores, test_labels, unk_names, name)
        diag[name] = (overall, per_phase)

    plot_diagnostic(diag, OUT_DIR)

    # ── Decide: complementary, redundant, or mixed? ───────────────────────
    print("\n" + "-" * 60)
    print("COMPLEMENTARITY CHECK")
    print("-" * 60)
    for phase in ["Malyugin_ring_insertion", "Malyugin_ring_removal"]:
        best_dino = max(diag["DINOv2-768 / KNN"][1].get(phase, 0),
                        diag["DINOv2-768 / RMDS"][1].get(phase, 0))
        best_mstcn = max(diag["MSTCN-64 / KNN"][1].get(phase, 0),
                         diag["MSTCN-64 / RMDS"][1].get(phase, 0))
        delta = abs(best_dino - best_mstcn)
        status = ("COMPLEMENTARY" if delta > 0.15
                  else "MIXED" if delta > 0.05
                  else "REDUNDANT")
        print(f"  {phase}: DINOv2={best_dino:.3f}  MSTCN={best_mstcn:.3f}"
              f"  Δ={delta:.3f}  → {status}")

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAGE 1 — Rank normalization calibrated on val known frames
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("ÉTAGE 1 — RANK NORMALIZATION (calibrated on val known)")
    print("=" * 60)

    # For each score: normalize test scores by their rank in val-known distribution
    n_dino_knn = rank_normalize(s_dino_knn, val_s_dino_knn)
    n_dino_rmds = rank_normalize(s_dino_rmds, val_s_dino_rmds)
    n_mstcn_knn = rank_normalize(s_mstcn_knn, val_s_mstcn_knn)
    n_mstcn_rmds = rank_normalize(s_mstcn_rmds, val_s_mstcn_rmds)

    print(f"  Rank-normalized test scores (should be ~uniform for known frames):")
    known_mask = test_labels >= 0
    for name, ns in [("DINOv2 KNN", n_dino_knn), ("DINOv2 RMDS", n_dino_rmds),
                      ("MSTCN KNN", n_mstcn_knn), ("MSTCN RMDS", n_mstcn_rmds)]:
        print(f"    {name:<15} known: mean={ns[known_mask].mean():.3f}"
              f"  std={ns[known_mask].std():.3f}"
              f"  |  unknown: mean={ns[~known_mask].mean():.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAGE 2 — Fusion (mean + max)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("ÉTAGE 2 — FUSION")
    print("=" * 60)

    # Pick best per-space score for Malyugin (from diagnostic)
    # Use KNN for both by default, but check RMDS too
    fusions = {
        "Fusion MEAN (DINOv2-KNN + MSTCN-KNN)":
            (n_dino_knn + n_mstcn_knn) / 2,
        "Fusion MAX (DINOv2-KNN + MSTCN-KNN)":
            np.maximum(n_dino_knn, n_mstcn_knn),
        "Fusion MEAN (DINOv2-RMDS + MSTCN-RMDS)":
            (n_dino_rmds + n_mstcn_rmds) / 2,
        "Fusion MAX (DINOv2-RMDS + MSTCN-RMDS)":
            np.maximum(n_dino_rmds, n_mstcn_rmds),
        "Fusion MEAN (DINOv2-KNN + MSTCN-RMDS)":
            (n_dino_knn + n_mstcn_rmds) / 2,
        "Fusion MAX (DINOv2-KNN + MSTCN-RMDS)":
            np.maximum(n_dino_knn, n_mstcn_rmds),
    }

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAGE 3 — Per-phase evaluation
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("ÉTAGE 3 — FUSION RESULTS (per-phase)")
    print("=" * 60)

    all_results = dict(diag)  # start with individual scores
    for name, scores in fusions.items():
        overall, per_phase = eval_per_phase(scores, test_labels, unk_names, name)
        all_results[name] = (overall, per_phase)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_comparison(all_results, OUT_DIR)
    plot_summary_table(all_results, OUT_DIR)

    # ── Save ──────────────────────────────────────────────────────────────
    np.savez(OUT_DIR / "fusion_scores.npz",
             s_dino_knn=s_dino_knn, s_dino_rmds=s_dino_rmds,
             s_mstcn_knn=s_mstcn_knn, s_mstcn_rmds=s_mstcn_rmds,
             n_dino_knn=n_dino_knn, n_dino_rmds=n_dino_rmds,
             n_mstcn_knn=n_mstcn_knn, n_mstcn_rmds=n_mstcn_rmds,
             test_labels=test_labels, unk_names=unk_names)
    print(f"\nSaved: {OUT_DIR / 'fusion_scores.npz'}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — Malyugin↑ without Suture↓ ?")
    print("=" * 60)

    # Best individual for reference
    for phase in ["Malyugin_ring_insertion", "Malyugin_ring_removal", "Suture"]:
        print(f"\n  {phase}:")
        for method, (_, pp) in sorted(all_results.items()):
            if phase in pp:
                marker = " ◄" if "Fusion" in method else ""
                print(f"    {method:<45} {pp[phase]:.4f}{marker}")

    print(f"\nOutput: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()

"""Diagnostic: Malyugin_ring_insertion composition + missing unknowns.

1. What are the ~1326 unaccounted unknown frames?
2. Malyugin_insertion: how many videos, where in the workflow?
3. Frame-by-frame correlation between DINOv2 and MSTCN scores
   (do they fail on the same frames?)

Usage:
    python phases_recognition/ood_diagnostic.py
"""

import json
import pathlib
import re
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset.feature_dataset import VideoFeatureDataset, _collate_single_video
from models import instantiate_model

# ── Paths ──────────────────────────────────────────────────────────────────
FEAT_ROOT = pathlib.Path("/home/helena/UCL_video_cataract/features_dino/")
MSTCN_EXP = "/home/helena/experiments_cataract/mstcn_dino_v1_date=2026_06_11_17_02_41"
OOD_JSON  = pathlib.Path("/home/helena/UCL_video_cataract/dataset_temporal/labels_ood.json")
OUT_DIR   = pathlib.Path("/home/helena/experiments_cataract/ood_fusion/")

CLASS_NAMES = [
    "Capsule_polishing", "Hydrodissection", "Incision",
    "Irrigation_and_aspiration", "Lens_implant_settingup",
    "Phacoemulsification", "Rhexis", "Tonifying_and_antibiotics",
    "Viscous_agent_injection", "Viscous_agent_removal", "Wound_hydration",
]
N_CLASSES = len(CLASS_NAMES)
KNN_K = 10


def load_dino_split(split):
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
    from models import instantiate_model
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
        out.append(f.squeeze(0).T.cpu().numpy())
    return out


def map_unknown_phases_per_video(test_videos):
    """Returns list of unk_names arrays (one per video), plus raw frame_phase dicts."""
    with open(OOD_JSON) as f:
        ood_data = json.load(f)
    all_unk = []
    all_frame_phase = []
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
        all_frame_phase.append(frame_phase)
    return all_unk, all_frame_phase


def fit_knn(train_feats, k=KNN_K):
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn.fit(train_feats)
    return nn


def score_knn(nn_model, feats):
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
    best_class = np.full(len(feats), np.inf, dtype=np.float32)
    for c in range(len(class_means)):
        diff = feats - class_means[c]
        d = (diff @ prec_class * diff).sum(1)
        best_class = np.minimum(best_class, d)
    diff_bg = feats - bg_mean
    d_bg = (diff_bg @ prec_bg * diff_bg).sum(1)
    return best_class - d_bg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load
    print("Loading...")
    mstcn = load_model(MSTCN_EXP, device)
    train_dino = load_dino_split("train")
    test_dino = load_dino_split("test")
    test_mstcn = extract_mstcn_features(mstcn, "test", device)

    train_feats_d = np.concatenate([f for _, f, l in train_dino])
    train_labels = np.concatenate([l for _, _, l in train_dino])
    mask_known = train_labels >= 0
    train_feats_d = train_feats_d[mask_known]
    train_mstcn_list = extract_mstcn_features(mstcn, "train", device)
    train_feats_m = np.concatenate(train_mstcn_list)[mask_known]
    train_labels = train_labels[mask_known]

    unk_names_per_video, frame_phases = map_unknown_phases_per_video(test_dino)

    # ══════════════════════════════════════════════════════════════════════
    # 1. Missing unknowns — what are the ~1326 unaccounted frames?
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("1. UNACCOUNTED UNKNOWN FRAMES")
    print("=" * 60)

    NAMED_PHASES = {
        "Malyugin_ring_insertion", "Malyugin_ring_removal",
        "Suture", "Iris_manipulation", "Trypan_blue_injection",
    }

    all_unk_names = np.concatenate(unk_names_per_video)
    all_labels = np.concatenate([l for _, _, l in test_dino])

    unk_mask = all_labels == -1
    corneal_mask = unk_mask & (all_unk_names == "Corneal_hydration")
    after_corneal_mask = unk_mask & ~corneal_mask

    print(f"\nTotal unknown frames (label=-1): {unk_mask.sum()}")
    print(f"  Corneal_hydration (excluded): {corneal_mask.sum()}")
    print(f"  Remaining unknown: {after_corneal_mask.sum()}")

    # Count by assigned phase name
    remaining_unk = all_unk_names[after_corneal_mask]
    counter = Counter(remaining_unk)
    print(f"\nPhase assignment for remaining {len(remaining_unk)} unknown frames:")
    for phase, count in counter.most_common():
        in_named = "✓ NAMED" if phase in NAMED_PHASES else "✗ NOT in eval list"
        print(f"  {phase:<35} {count:>5}  {in_named}")

    named_total = sum(count for phase, count in counter.items()
                      if phase in NAMED_PHASES)
    unnamed_total = sum(count for phase, count in counter.items()
                        if phase not in NAMED_PHASES)
    print(f"\n  Named (in eval):     {named_total}")
    print(f"  NOT in eval list:    {unnamed_total}")
    print(f"  → These {unnamed_total} frames are in the overall AUROC but not "
          f"in any per-phase bar.")

    # ══════════════════════════════════════════════════════════════════════
    # 2. Malyugin_ring_insertion: composition per video
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("2. MALYUGIN_RING_INSERTION — VIDEO COMPOSITION")
    print("=" * 60)

    for i, (name, feats, labels) in enumerate(test_dino):
        T = len(labels)
        unk = unk_names_per_video[i]
        mal_mask = (labels == -1) & (unk == "Malyugin_ring_insertion")
        n_mal = mal_mask.sum()
        if n_mal == 0:
            continue

        # Where in the video? (as % of total length)
        mal_indices = np.where(mal_mask)[0]
        start_pct = mal_indices[0] / T * 100
        end_pct = mal_indices[-1] / T * 100

        # What known phases surround it?
        first_mal = mal_indices[0]
        last_mal = mal_indices[-1]
        before_labels = labels[max(0, first_mal - 20):first_mal]
        after_labels = labels[last_mal + 1:min(T, last_mal + 21)]
        before_phases = [CLASS_NAMES[l] for l in before_labels if l >= 0]
        after_phases = [CLASS_NAMES[l] for l in after_labels if l >= 0]

        before_str = Counter(before_phases).most_common(1)
        after_str = Counter(after_phases).most_common(1)

        print(f"\n  {name}")
        print(f"    Frames: {n_mal} ({n_mal/T*100:.1f}% of video)")
        print(f"    Position: {start_pct:.1f}% — {end_pct:.1f}% of video")
        print(f"    Duration: {mal_indices[-1] - mal_indices[0] + 1} consecutive feature frames")
        print(f"    Phase before: {before_str[0][0] if before_str else '?'}")
        print(f"    Phase after:  {after_str[0][0] if after_str else '?'}")

        # Check if contiguous or fragmented
        gaps = np.diff(mal_indices)
        n_segments = 1 + (gaps > 1).sum()
        print(f"    Segments: {n_segments} (1=contiguous, >1=fragmented)")

    # ══════════════════════════════════════════════════════════════════════
    # 3. Frame-by-frame score correlation DINOv2 vs MSTCN on Malyugin
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("3. SCORE CORRELATION — DINOv2 vs MSTCN on Malyugin_insertion")
    print("=" * 60)

    # Fit scorers
    print("\nFitting scorers...")
    knn_d = fit_knn(train_feats_d)
    knn_m = fit_knn(train_feats_m)
    rmds_d_params = fit_rmds(train_feats_d, train_labels, N_CLASSES)
    rmds_m_params = fit_rmds(train_feats_m, train_labels, N_CLASSES)

    # Score test per video, focus on Malyugin_insertion
    offset = 0
    all_s_dino_knn, all_s_mstcn_knn = [], []
    all_s_dino_rmds, all_s_mstcn_rmds = [], []
    all_is_mal = []

    for i, (name, feats_d, labels) in enumerate(test_dino):
        T = len(labels)
        feats_m = test_mstcn[i]
        unk = unk_names_per_video[i]

        sd_knn = score_knn(knn_d, feats_d)
        sm_knn = score_knn(knn_m, feats_m)
        sd_rmds = score_rmds(feats_d, *rmds_d_params)
        sm_rmds = score_rmds(feats_m, *rmds_m_params)

        mal_mask = (labels == -1) & (unk == "Malyugin_ring_insertion")

        all_s_dino_knn.append(sd_knn)
        all_s_mstcn_knn.append(sm_knn)
        all_s_dino_rmds.append(sd_rmds)
        all_s_mstcn_rmds.append(sm_rmds)
        all_is_mal.append(mal_mask)

    s_dino_knn = np.concatenate(all_s_dino_knn)
    s_mstcn_knn = np.concatenate(all_s_mstcn_knn)
    s_dino_rmds = np.concatenate(all_s_dino_rmds)
    s_mstcn_rmds = np.concatenate(all_s_mstcn_rmds)
    is_mal = np.concatenate(all_is_mal)
    all_labels_flat = np.concatenate([l for _, _, l in test_dino])

    # Exclude corneal
    all_unk_flat = np.concatenate(unk_names_per_video)
    keep = ~((all_labels_flat == -1) & (all_unk_flat == "Corneal_hydration"))

    # Correlation on Malyugin_insertion frames
    mal_final = is_mal[keep]
    sd_k = s_dino_knn[keep][mal_final]
    sm_k = s_mstcn_knn[keep][mal_final]
    sd_r = s_dino_rmds[keep][mal_final]
    sm_r = s_mstcn_rmds[keep][mal_final]

    rho_knn, p_knn = spearmanr(sd_k, sm_k)
    rho_rmds, p_rmds = spearmanr(sd_r, sm_r)

    print(f"\n  Malyugin_insertion frames: {mal_final.sum()}")
    print(f"  Spearman correlation (DINOv2 vs MSTCN):")
    print(f"    KNN:  ρ={rho_knn:.4f}  (p={p_knn:.2e})")
    print(f"    RMDS: ρ={rho_rmds:.4f}  (p={p_rmds:.2e})")

    if rho_knn > 0.5:
        print("    → HIGH correlation: they fail on the SAME frames. Fusion cannot help.")
    elif rho_knn > 0.2:
        print("    → MODERATE correlation: partial overlap in failures.")
    else:
        print("    → LOW correlation: they fail on DIFFERENT frames. Fusion could help.")

    # Where do known frames fall vs Malyugin?
    known_final = (all_labels_flat[keep] >= 0)
    print(f"\n  Score distributions (KNN, higher=more OOD):")
    print(f"    Known frames:      mean={s_dino_knn[keep][known_final].mean():.2f}"
          f"  ±{s_dino_knn[keep][known_final].std():.2f}  (DINOv2)")
    print(f"    Malyugin_insert:   mean={sd_k.mean():.2f}"
          f"  ±{sd_k.std():.2f}  (DINOv2)")
    print(f"    Known frames:      mean={s_mstcn_knn[keep][known_final].mean():.2f}"
          f"  ±{s_mstcn_knn[keep][known_final].std():.2f}  (MSTCN)")
    print(f"    Malyugin_insert:   mean={sm_k.mean():.2f}"
          f"  ±{sm_k.std():.2f}  (MSTCN)")

    # ── Plot: scatter DINOv2 vs MSTCN scores for Malyugin vs known ──────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (sd, sm, title) in zip(axes, [
        (s_dino_knn[keep], s_mstcn_knn[keep], "KNN scores"),
        (s_dino_rmds[keep], s_mstcn_rmds[keep], "RMDS scores"),
    ]):
        # Subsample known for readability
        known_idx = np.where(known_final)[0]
        if len(known_idx) > 2000:
            known_idx = np.random.RandomState(42).choice(known_idx, 2000, replace=False)
        mal_idx = np.where(mal_final)[0]

        ax.scatter(sd[known_idx], sm[known_idx], c="#aaaaaa", s=5, alpha=0.3,
                   label=f"Known (n={known_final.sum():,})")
        ax.scatter(sd[mal_idx], sm[mal_idx], c="#FF0000", s=15, alpha=0.6,
                   label=f"Malyugin insert (n={len(mal_idx)})")
        ax.set_xlabel("DINOv2 score", fontsize=11)
        ax.set_ylabel("MSTCN score", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Do DINOv2 and MSTCN fail on the same Malyugin frames?",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "malyugin_score_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    # ── Plot: Malyugin position in workflow per video ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    y_pos = 0
    for i, (name, feats, labels) in enumerate(test_dino):
        T = len(labels)
        unk = unk_names_per_video[i]
        mal_mask = (labels == -1) & (unk == "Malyugin_ring_insertion")
        if mal_mask.sum() == 0:
            continue
        mal_idx = np.where(mal_mask)[0]

        # Draw full video as grey bar
        ax.barh(y_pos, T, height=0.6, color="#eeeeee", edgecolor="grey", lw=0.5)
        # Overlay known phases
        for c in range(N_CLASSES):
            c_mask = labels == c
            if c_mask.sum() == 0:
                continue
            c_idx = np.where(c_mask)[0]
            segments = np.split(c_idx, np.where(np.diff(c_idx) > 1)[0] + 1)
            for seg in segments:
                ax.barh(y_pos, len(seg), left=seg[0], height=0.6,
                        color="#bbbbbb", linewidth=0)
        # Overlay Malyugin
        segments = np.split(mal_idx, np.where(np.diff(mal_idx) > 1)[0] + 1)
        for seg in segments:
            ax.barh(y_pos, len(seg), left=seg[0], height=0.6,
                    color="#FF0000", linewidth=0)
        ax.text(-50, y_pos, name[-15:], ha="right", va="center", fontsize=8)
        y_pos += 1

    ax.set_xlabel("Frame index", fontsize=11)
    ax.set_title("Malyugin_ring_insertion position in each test video (red = insertion)",
                 fontsize=12)
    ax.set_yticks([])
    ax.set_xlim(-200, None)
    fig.tight_layout()
    path = OUT_DIR / "malyugin_workflow_position.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

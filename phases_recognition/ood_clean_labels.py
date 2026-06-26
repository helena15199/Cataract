"""Clean OOD labels: reclassify mislabeled known phases.

Problem: label=-1 encodes "no annotation", not "semantically novel".
Some -1 frames are actually known phases (Irrigation, Viscous_agent, etc.)
that were missed in annotation. They pollute the OOD evaluation.

Fix: a frame is unknown IFF its phase name (from labels_ood.json) does NOT
appear in any training video. Everything else is known/ID.

Then recalculate AUROC on the cleaned pool.

Usage:
    python phases_recognition/ood_clean_labels.py
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
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset.feature_dataset import VideoFeatureDataset, _collate_single_video
from models import instantiate_model

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
KNOWN_SET = set(CLASS_NAMES)
N_CLASSES = len(CLASS_NAMES)
KNN_K = 10

UNKNOWN_PHASES = [
    "Malyugin_ring_insertion", "Malyugin_ring_removal",
    "Suture", "Iris_manipulation", "Trypan_blue_injection",
]


# ── Feature loading (same as before) ─────────────────────────────────────

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


def load_phase_names(split, video_names):
    """Load phase names from pre-saved _phases.npy files."""
    all_phase_names = []
    for name in video_names:
        path = FEAT_ROOT / split / f"{name}_phases.npy"
        all_phase_names.append(np.load(path, allow_pickle=True))
    return all_phase_names


# ── Scoring ───────────────────────────────────────────────────────────────

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

def rank_normalize(test_scores, val_known_scores):
    sorted_val = np.sort(val_known_scores)
    return (np.searchsorted(sorted_val, test_scores, side="right")
            / len(sorted_val)).astype(np.float32)


def eval_per_phase(scores, is_truly_unknown, phase_names, label=""):
    gt = is_truly_unknown.astype(np.int32)
    overall_auroc = roc_auc_score(gt, scores)
    overall_aupr = average_precision_score(gt, scores)
    print(f"\n  {label}")
    print(f"    Overall:  AUROC={overall_auroc:.4f}  AUPR={overall_aupr:.4f}"
          f"  (n_unk={gt.sum()}, n_known={(~is_truly_unknown).sum()})")

    per_phase = {}
    for phase in UNKNOWN_PHASES:
        mask_p = is_truly_unknown & (phase_names == phase)
        n = mask_p.sum()
        if n < 5:
            print(f"    {phase:<30} SKIP (n={n})")
            continue
        eval_mask = (~is_truly_unknown) | mask_p
        gt_p = mask_p[eval_mask].astype(np.int32)
        s_p = scores[eval_mask]
        try:
            auroc = roc_auc_score(gt_p, s_p)
            per_phase[phase] = auroc
            print(f"    {phase:<30} AUROC={auroc:.4f}  (n={n})")
        except ValueError:
            print(f"    {phase:<30} ERROR")
    return overall_auroc, per_phase


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ══════════════════════════════════════════════════════════════════════
    # 1. Define unknown by construction: phase name NOT in training set
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("1. PHASE INVENTORY — train vs test")
    print("=" * 60)

    with open(OOD_JSON) as f:
        ood_data = json.load(f)

    # Collect all phase names per split
    train_phases = set()
    test_phases = set()
    for key, phase in ood_data.items():
        if key.startswith("train/"):
            train_phases.add(phase)
        elif key.startswith("test/"):
            test_phases.add(phase)

    print(f"\nPhases in TRAIN ({len(train_phases)}):")
    for p in sorted(train_phases):
        print(f"  {p}")

    print(f"\nPhases in TEST ({len(test_phases)}):")
    for p in sorted(test_phases):
        status = "KNOWN" if p in train_phases else "UNKNOWN"
        print(f"  {p:<35} → {status}")

    truly_unknown_names = test_phases - train_phases
    print(f"\nTruly unknown phases (in test but NOT in train):")
    for p in sorted(truly_unknown_names):
        print(f"  {p}")

    # ══════════════════════════════════════════════════════════════════════
    # 2. Build clean labels for test set
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("2. CLEAN LABELS — reclassify mislabeled frames")
    print("=" * 60)

    test_dino = load_dino_split("test")
    video_names = [name for name, _, _ in test_dino]
    phase_names_per_video = load_phase_names("test", video_names)

    old_labels = np.concatenate([l for _, _, l in test_dino])
    phase_names = np.concatenate(phase_names_per_video)

    # Old labeling
    old_unk_mask = old_labels == -1
    print(f"\nOLD labeling (label=-1 = 'unknown'):")
    print(f"  Total: {len(old_labels):,} frames")
    print(f"  Known (label>=0): {(~old_unk_mask).sum():,}")
    print(f"  Unknown (label=-1): {old_unk_mask.sum():,}")

    # Breakdown of -1 frames by actual phase name
    print(f"\nBreakdown of label=-1 frames:")
    unk_phases = Counter(phase_names[old_unk_mask])
    n_truly_unk = 0
    n_mislabeled = 0
    for p, count in unk_phases.most_common():
        if p in truly_unknown_names:
            status = "TRUE UNKNOWN"
            n_truly_unk += count
        else:
            status = "MISLABELED (known phase)"
            n_mislabeled += count
        print(f"  {p:<35} {count:>5}  {status}")

    print(f"\n  True unknowns:  {n_truly_unk}")
    print(f"  Mislabeled:     {n_mislabeled}")
    print(f"  → {n_mislabeled} frames will be reclassified as KNOWN")

    # New clean mask: unknown = phase name not in train
    is_truly_unknown = np.array([p in truly_unknown_names for p in phase_names])

    # Exclude Corneal_hydration (too short, too similar to Wound_hydration)
    corneal_mask = phase_names == "Corneal_hydration"
    exclude = corneal_mask  # frames to drop entirely
    keep = ~exclude

    is_truly_unknown = is_truly_unknown[keep]
    phase_names_clean = phase_names[keep]

    print(f"\nCLEAN labeling:")
    print(f"  Total (after excluding Corneal): {keep.sum():,}")
    print(f"  Known: {(~is_truly_unknown).sum():,}")
    print(f"  Unknown: {is_truly_unknown.sum():,}")
    print(f"  Unknown breakdown:")
    clean_unk = Counter(phase_names_clean[is_truly_unknown])
    for p, count in clean_unk.most_common():
        print(f"    {p:<35} {count:>5}")

    # ══════════════════════════════════════════════════════════════════════
    # 3. Recalculate AUROC with clean labels
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("3. RECALCULATE AUROC — old vs clean labels")
    print("=" * 60)

    print("\nLoading models and features...")
    mstcn = load_model(MSTCN_EXP, device)

    # Train features (known only, for fitting)
    train_dino = load_dino_split("train")
    train_mstcn = extract_mstcn_features(mstcn, "train", device)
    train_d = np.concatenate([f for _, f, _ in train_dino])
    train_l = np.concatenate([l for _, _, l in train_dino])
    train_m = np.concatenate(train_mstcn)
    tk = train_l >= 0
    train_d, train_m, train_l = train_d[tk], train_m[tk], train_l[tk]

    # Val features (for rank normalization)
    val_dino = load_dino_split("val")
    val_mstcn = extract_mstcn_features(mstcn, "val", device)
    val_d = np.concatenate([f for _, f, _ in val_dino])
    val_l = np.concatenate([l for _, _, l in val_dino])
    val_m = np.concatenate(val_mstcn)
    vk = val_l >= 0
    val_d, val_m = val_d[vk], val_m[vk]

    # Test features (apply keep mask)
    test_d = np.concatenate([f for _, f, _ in test_dino])[keep]
    test_m = np.concatenate(extract_mstcn_features(mstcn, "test", device))[keep]

    # Fit scorers
    print("Fitting scorers...")
    knn_d = fit_knn(train_d)
    knn_m = fit_knn(train_m)
    rmds_d_p = fit_rmds(train_d, train_l, N_CLASSES)
    rmds_m_p = fit_rmds(train_m, train_l, N_CLASSES)

    # Score test
    s_dino_knn = score_knn(knn_d, test_d)
    s_mstcn_knn = score_knn(knn_m, test_m)
    s_dino_rmds = score_rmds(test_d, *rmds_d_p)
    s_mstcn_rmds = score_rmds(test_m, *rmds_m_p)

    # ── Compare old vs clean ──────────────────────────────────────────────
    # Old mask (for comparison): label=-1, minus Corneal
    old_unk_clean = old_labels[keep] == -1

    print("\n--- OLD labels (label=-1 = unknown) ---")
    for name, scores in [("DINOv2 KNN", s_dino_knn), ("DINOv2 RMDS", s_dino_rmds),
                          ("MSTCN KNN", s_mstcn_knn), ("MSTCN RMDS", s_mstcn_rmds)]:
        gt = old_unk_clean.astype(np.int32)
        auroc = roc_auc_score(gt, scores)
        print(f"  {name:<15} AUROC={auroc:.4f}  (n_unk={gt.sum()})")

    print("\n--- CLEAN labels (truly unknown phases only) ---")
    for name, scores in [("DINOv2 KNN", s_dino_knn), ("DINOv2 RMDS", s_dino_rmds),
                          ("MSTCN KNN", s_mstcn_knn), ("MSTCN RMDS", s_mstcn_rmds)]:
        overall, per_phase = eval_per_phase(
            scores, is_truly_unknown, phase_names_clean, name)

    # ── Best fusion on clean labels ───────────────────────────────────────
    print("\n--- FUSION on clean labels ---")
    # Score val for rank normalization
    val_s_dk = score_knn(knn_d, val_d)
    val_s_mk = score_knn(knn_m, val_m)
    val_s_dr = score_rmds(val_d, *rmds_d_p)
    val_s_mr = score_rmds(val_m, *rmds_m_p)

    n_dk = rank_normalize(s_dino_knn, val_s_dk)
    n_mk = rank_normalize(s_mstcn_knn, val_s_mk)
    n_dr = rank_normalize(s_dino_rmds, val_s_dr)
    n_mr = rank_normalize(s_mstcn_rmds, val_s_mr)

    fusions = {
        "Fusion MAX KNN": np.maximum(n_dk, n_mk),
        "Fusion MEAN KNN": (n_dk + n_mk) / 2,
        "Fusion MAX RMDS": np.maximum(n_dr, n_mr),
        "Fusion MEAN RMDS": (n_dr + n_mr) / 2,
    }
    for name, scores in fusions.items():
        overall, per_phase = eval_per_phase(
            scores, is_truly_unknown, phase_names_clean, name)

    # ── Summary comparison table ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON: OLD vs CLEAN overall AUROC")
    print("=" * 60)
    print(f"\n  {'Method':<20} {'OLD':>8} {'CLEAN':>8} {'Δ':>8}")
    print(f"  {'-'*46}")
    for name, scores in [("DINOv2 KNN", s_dino_knn), ("DINOv2 RMDS", s_dino_rmds),
                          ("MSTCN KNN", s_mstcn_knn), ("MSTCN RMDS", s_mstcn_rmds)]:
        old_auroc = roc_auc_score(old_unk_clean.astype(np.int32), scores)
        clean_auroc = roc_auc_score(is_truly_unknown.astype(np.int32), scores)
        delta = clean_auroc - old_auroc
        print(f"  {name:<20} {old_auroc:>8.4f} {clean_auroc:>8.4f} {delta:>+8.4f}")

    # Save clean labels for future use
    np.savez(OUT_DIR / "clean_labels.npz",
             is_truly_unknown=is_truly_unknown,
             phase_names=phase_names_clean,
             keep_mask=keep,
             truly_unknown_names=np.array(sorted(truly_unknown_names)))
    print(f"\nSaved: {OUT_DIR / 'clean_labels.npz'}")
    print("\nDone.")


if __name__ == "__main__":
    main()

"""Action Distribution Flow for open-set surgical phase recognition.

Replicates the ADF pipeline (paper) on cataract surgery phases:
  Stage 1: Gaussian per phase on MSTCN logits → Mahalanobis → initial open-set predictions
  Stage 2: Wasserstein-2 geodesic flow between consecutive phases → reclassify transitions

Usage:
    python phases_recognition/action_distribution_flow.py
"""

import pathlib
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset.feature_dataset import VideoFeatureDataset, _collate_single_video
from models import instantiate_model

# ── Config ─────────────────────────────────────────────────────────────────
FEAT_ROOT = pathlib.Path("/home/helena/UCL_video_cataract/features_dino/")
MSTCN_EXP = "/home/helena/experiments_cataract/mstcn_dino_v1_date=2026_06_11_17_02_41"
OUT_DIR   = pathlib.Path("/home/helena/experiments_cataract/adf/")

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

THRESHOLD_PERCENTILE = 95  # FPR95 calibration on val
N_RHO = 11  # discretization of flow: ρ ∈ {0, 0.1, ..., 1.0}


# ── Model + feature loading ───────────────────────────────────────────────

def load_model(exp_dir, device):
    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_model(cfg.model)
    state = torch.load(f"{exp_dir}/ckpt/best.pt", map_location="cpu",
                       weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def extract_logits_per_video(model, split, device):
    """Extract MSTCN logits (11-dim, last stage before softmax) per video.

    Returns list of (logits_array, labels_array, phase_names_array, video_name).
    """
    ds = VideoFeatureDataset(root=str(FEAT_ROOT / split))
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=_collate_single_video)
    results = []
    for features, labels, name in loader:
        out = model(features.unsqueeze(0).to(device))
        logits = out[-1].squeeze(0).T.cpu().numpy()  # (T, C) last stage logits

        phases_path = FEAT_ROOT / split / f"{name}_phases.npy"
        if phases_path.exists():
            phase_names = np.load(phases_path, allow_pickle=True)
        else:
            phase_names = np.full(len(labels), "", dtype=object)

        results.append((logits, labels.numpy(), phase_names, name))
    return results


# ── Stage 1: Action Distribution Modeling ─────────────────────────────────

def fit_class_gaussians(train_videos, n_classes):
    """Fit one Gaussian N(μ_a, Σ_a) per phase on train logits."""
    per_class = defaultdict(list)
    for logits, labels, _, _ in train_videos:
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() > 0:
                per_class[c].append(logits[mask])

    means = np.zeros((n_classes, n_classes), dtype=np.float64)
    covs = np.zeros((n_classes, n_classes, n_classes), dtype=np.float64)
    precisions = np.zeros_like(covs)

    for c in range(n_classes):
        feats_c = np.concatenate(per_class[c]).astype(np.float64)
        means[c] = feats_c.mean(axis=0)
        lw = LedoitWolf()
        lw.fit(feats_c)
        covs[c] = lw.covariance_
        precisions[c] = lw.precision_

    return means, covs, precisions


def mahalanobis_to_predicted(logits, predictions, means, precisions):
    """Mahalanobis distance of each frame to its predicted class gaussian."""
    T = len(logits)
    dists = np.zeros(T, dtype=np.float64)
    for t in range(T):
        c = predictions[t]
        diff = logits[t] - means[c]
        dists[t] = np.sqrt(max(0, diff @ precisions[c] @ diff))
    return dists


def calibrate_threshold(val_videos, means, precisions, percentile):
    """Set threshold at given percentile of val known-frame distances."""
    all_dists = []
    for logits, labels, _, _ in val_videos:
        known_mask = labels >= 0
        if known_mask.sum() == 0:
            continue
        preds = logits[known_mask].argmax(axis=1)
        dists = mahalanobis_to_predicted(logits[known_mask], preds,
                                          means, precisions)
        all_dists.append(dists)
    all_dists = np.concatenate(all_dists)
    threshold = np.percentile(all_dists, percentile)
    print(f"  Threshold (p{percentile}): {threshold:.4f}")
    return threshold


# ── Stage 2: Action Distribution Flow ─────────────────────────────────────

def find_transition_pairs(train_videos, n_classes):
    """Find all consecutive phase pairs (A→B) in train videos."""
    pairs = set()
    for _, labels, _, _ in train_videos:
        known = labels[labels >= 0]
        for i in range(len(known) - 1):
            if known[i] != known[i + 1]:
                pairs.add((known[i], known[i + 1]))
    print(f"  Found {len(pairs)} transition pairs in train")
    return pairs


def wasserstein_geodesic(mu_A, cov_A, mu_B, cov_B, rho):
    """Wasserstein-2 geodesic between N(mu_A, cov_A) and N(mu_B, cov_B) at ρ.

    Returns (mu_flow, cov_flow) — the interpolated Gaussian.
    """
    mu_flow = (1 - rho) * mu_A + rho * mu_B

    if rho == 0:
        return mu_flow, cov_A.copy()
    if rho == 1:
        return mu_flow, cov_B.copy()

    S_A_half = sqrtm(cov_A).real
    S_A_inv_half = np.linalg.inv(S_A_half)

    M = S_A_half @ cov_B @ S_A_half
    M_half = sqrtm(M).real

    T = S_A_inv_half @ M_half @ S_A_inv_half

    mix = (1 - rho) * np.eye(len(mu_A)) + rho * T
    cov_flow = mix @ cov_A @ mix

    return mu_flow, cov_flow


def precompute_flows(means, covs, pairs, n_rho=N_RHO):
    """Pre-compute flow distributions for each transition pair."""
    rhos = np.linspace(0, 1, n_rho)
    flows = {}
    for (a, b) in pairs:
        flow_dists = []
        for rho in rhos:
            mu_f, cov_f = wasserstein_geodesic(means[a], covs[a],
                                                means[b], covs[b], rho)
            prec_f = np.linalg.inv(cov_f)
            flow_dists.append((mu_f, prec_f))
        flows[(a, b)] = flow_dists
    return flows, rhos


def flow_distance(frame_logits, flow_dists):
    """Min Mahalanobis distance of a frame to any point on a flow."""
    min_dist = np.inf
    for mu_f, prec_f in flow_dists:
        diff = frame_logits - mu_f
        d = np.sqrt(max(0, diff @ prec_f @ diff))
        if d < min_dist:
            min_dist = d
    return min_dist


def find_neighboring_transition(t, predictions, labels_or_preds, window=15):
    """Find which transition A→B a frame at position t might belong to.

    Looks at the predicted classes in a window around t to determine
    the transition pair. Returns (A, B) or None.
    """
    T = len(labels_or_preds)
    left_start = max(0, t - window)
    right_end = min(T, t + window + 1)

    left_classes = []
    right_classes = []
    for i in range(left_start, t):
        if labels_or_preds[i] >= 0:
            left_classes.append(labels_or_preds[i])
    for i in range(t + 1, right_end):
        if labels_or_preds[i] >= 0:
            right_classes.append(labels_or_preds[i])

    if not left_classes or not right_classes:
        return None

    a = Counter(left_classes).most_common(1)[0][0]
    b = Counter(right_classes).most_common(1)[0][0]
    if a == b:
        return None
    return (int(a), int(b))


# ── Evaluation ────────────────────────────────────────────────────────────

def eval_auroc(scores, labels, phase_names, label=""):
    """Higher score = more OOD. Evaluate overall + per-phase AUROC."""
    gt = (labels == -1).astype(np.int32)
    if gt.sum() == 0 or gt.sum() == len(gt):
        return 0, {}

    overall = roc_auc_score(gt, scores)
    aupr = average_precision_score(gt, scores)
    print(f"\n  {label}")
    print(f"    Overall: AUROC={overall:.4f}  AUPR={aupr:.4f}"
          f"  (n_unk={gt.sum()}, n_known={len(gt)-gt.sum()})")

    per_phase = {}
    for phase in UNKNOWN_PHASES:
        mask_p = (labels == -1) & (phase_names == phase)
        n = mask_p.sum()
        if n < 5:
            print(f"    {phase:<30} SKIP (n={n})")
            continue
        eval_mask = (labels >= 0) | mask_p
        gt_p = mask_p[eval_mask].astype(np.int32)
        s_p = scores[eval_mask]
        try:
            auroc = roc_auc_score(gt_p, s_p)
            per_phase[phase] = auroc
            print(f"    {phase:<30} AUROC={auroc:.4f}  (n={n})")
        except ValueError:
            print(f"    {phase:<30} ERROR")

    return overall, per_phase


# ── Plots ─────────────────────────────────────────────────────────────────

def plot_comparison(stage1_results, stage2_results, out_dir):
    """Side-by-side per-phase AUROC: Stage 1 vs Stage 1+2."""
    phases = [p for p in UNKNOWN_PHASES
              if p in stage1_results[1] or p in stage2_results[1]]
    if not phases:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(phases))
    w = 0.35

    v1 = [stage1_results[1].get(p, 0) for p in phases]
    v2 = [stage2_results[1].get(p, 0) for p in phases]

    bars1 = ax.bar(x - w/2, v1, w, label="Stage 1 (Mahalanobis only)",
                   color="#1f77b4", edgecolor="black", lw=0.3)
    bars2 = ax.bar(x + w/2, v2, w, label="Stage 1 + 2 (with ADF)",
                   color="#ff7f0e", edgecolor="black", lw=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in phases], fontsize=9)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Action Distribution Flow — Stage 1 vs Stage 1+2", fontsize=14)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "adf_stage1_vs_stage2.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")


def plot_transition_distances(diag, out_dir):
    """Diagnostic: mean distance of transition frames with/without flow."""
    if not diag:
        return

    pairs = list(diag.keys())
    d_no_flow = [diag[p]["mean_dist_no_flow"] for p in pairs]
    d_flow = [diag[p]["mean_dist_flow"] for p in pairs]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(pairs))
    w = 0.35
    ax.bar(x - w/2, d_no_flow, w, label="Without flow", color="#e41a1c")
    ax.bar(x + w/2, d_flow, w, label="With flow", color="#4daf4a")

    labels = [f"{CLASS_NAMES[a][:8]}→{CLASS_NAMES[b][:8]}" for a, b in pairs]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Mean Mahalanobis distance", fontsize=11)
    ax.set_title("Transition frames: distance with vs without flow", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "adf_transition_distances.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading MSTCN model...")
    mstcn = load_model(MSTCN_EXP, device)

    print("Extracting logits...")
    train_videos = extract_logits_per_video(mstcn, "train", device)
    val_videos = extract_logits_per_video(mstcn, "val", device)
    test_videos = extract_logits_per_video(mstcn, "test", device)

    for split, vids in [("train", train_videos), ("val", val_videos), ("test", test_videos)]:
        n_frames = sum(len(l) for _, l, _, _ in vids)
        n_unk = sum((l == -1).sum() for _, l, _, _ in vids)
        print(f"  {split}: {len(vids)} videos, {n_frames:,} frames, {n_unk:,} unknown")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Action Distribution Modeling
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 1 — Action Distribution Modeling")
    print("=" * 60)

    print("\nFitting class Gaussians on train logits...")
    means, covs, precisions = fit_class_gaussians(train_videos, N_CLASSES)

    print("\nCalibrating threshold on val...")
    threshold = calibrate_threshold(val_videos, means, precisions,
                                     THRESHOLD_PERCENTILE)

    # Apply Stage 1 on test
    all_logits, all_labels, all_phases, all_preds = [], [], [], []
    all_dists_s1 = []

    for logits, labels, phase_names, name in test_videos:
        preds = logits.argmax(axis=1)
        dists = mahalanobis_to_predicted(logits, preds, means, precisions)

        all_logits.append(logits)
        all_labels.append(labels)
        all_phases.append(phase_names)
        all_preds.append(preds)
        all_dists_s1.append(dists)

    logits_flat = np.concatenate(all_logits)
    labels_flat = np.concatenate(all_labels)
    phases_flat = np.concatenate(all_phases)
    preds_flat = np.concatenate(all_preds)
    dists_s1_flat = np.concatenate(all_dists_s1)

    # Stage 1 predictions: unknown if distance > threshold
    s1_unknown = dists_s1_flat > threshold
    n_s1_unk = s1_unknown.sum()
    print(f"\nStage 1: {n_s1_unk} frames marked unknown "
          f"({n_s1_unk/len(s1_unknown)*100:.1f}%)")

    # Stage 1 AUROC (using distance as OOD score)
    print("\nStage 1 evaluation:")
    s1_results = eval_auroc(dists_s1_flat, labels_flat, phases_flat,
                            "Stage 1 (Mahalanobis on logits)")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Action Distribution Flow
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 2 — Action Distribution Flow")
    print("=" * 60)

    print("\nFinding transition pairs in train...")
    pairs = find_transition_pairs(train_videos, N_CLASSES)

    print("Pre-computing Wasserstein flows...")
    flows, rhos = precompute_flows(means, covs, pairs, N_RHO)
    print(f"  {len(flows)} flows × {N_RHO} points each")

    # Apply flow to candidate-unknown frames
    dists_s2_flat = dists_s1_flat.copy()
    n_reclassified = 0
    transition_diag = defaultdict(lambda: {"no_flow": [], "flow": []})

    offset = 0
    for vid_idx, (logits, labels, phase_names, name) in enumerate(test_videos):
        T = len(labels)
        preds = all_preds[vid_idx]
        dists_s1 = all_dists_s1[vid_idx]

        for t in range(T):
            global_t = offset + t
            if not s1_unknown[global_t]:
                continue

            pair = find_neighboring_transition(t, preds, preds)
            if pair is None:
                continue

            # Try both directions
            d_flow = np.inf
            matched_pair = None
            for p in [pair, (pair[1], pair[0])]:
                if p in flows:
                    d = flow_distance(logits[t].astype(np.float64), flows[p])
                    if d < d_flow:
                        d_flow = d
                        matched_pair = p

            if matched_pair is not None:
                transition_diag[matched_pair]["no_flow"].append(dists_s1[t])
                transition_diag[matched_pair]["flow"].append(d_flow)

                dists_s2_flat[global_t] = d_flow
                if d_flow <= threshold:
                    n_reclassified += 1

        offset += T

    print(f"\n  Frames reclassified (unknown → known by flow): {n_reclassified}")
    s2_unknown = dists_s2_flat > threshold
    print(f"  Stage 2 unknown: {s2_unknown.sum()} "
          f"(was {s1_unknown.sum()} at Stage 1)")

    # ── Transition distance diagnostic ────────────────────────────────────
    print("\n" + "-" * 60)
    print("TRANSITION DISTANCE DIAGNOSTIC")
    print("-" * 60)
    diag_summary = {}
    for pair in sorted(transition_diag.keys()):
        d_nf = transition_diag[pair]["no_flow"]
        d_f = transition_diag[pair]["flow"]
        if len(d_nf) == 0:
            continue
        mean_nf = np.mean(d_nf)
        mean_f = np.mean(d_f)
        reduction = (mean_nf - mean_f) / mean_nf * 100
        a, b = pair
        print(f"  {CLASS_NAMES[a][:15]:>15}→{CLASS_NAMES[b][:15]:<15}"
              f"  n={len(d_nf):>4}  "
              f"no_flow={mean_nf:.2f}  flow={mean_f:.2f}  "
              f"reduction={reduction:+.1f}%")
        diag_summary[pair] = {"mean_dist_no_flow": mean_nf,
                               "mean_dist_flow": mean_f,
                               "n_frames": len(d_nf)}

    # ── Stage 2 AUROC ─────────────────────────────────────────────────────
    print("\nStage 2 evaluation:")
    s2_results = eval_auroc(dists_s2_flat, labels_flat, phases_flat,
                            "Stage 1 + 2 (with ADF)")

    # ── Comparison ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON — Stage 1 vs Stage 1+2")
    print("=" * 60)
    print(f"\n  {'Phase':<30} {'Stage1':>8} {'Stage1+2':>8} {'Δ':>8}")
    print(f"  {'-'*56}")
    print(f"  {'Overall':<30} {s1_results[0]:>8.4f} {s2_results[0]:>8.4f} "
          f"{s2_results[0]-s1_results[0]:>+8.4f}")
    for phase in UNKNOWN_PHASES:
        v1 = s1_results[1].get(phase, None)
        v2 = s2_results[1].get(phase, None)
        if v1 is not None and v2 is not None:
            print(f"  {phase:<30} {v1:>8.4f} {v2:>8.4f} {v2-v1:>+8.4f}")
        elif v1 is not None:
            print(f"  {phase:<30} {v1:>8.4f} {'—':>8}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_comparison(s1_results, s2_results, OUT_DIR)
    plot_transition_distances(diag_summary, OUT_DIR)

    # ── Save ──────────────────────────────────────────────────────────────
    np.savez(OUT_DIR / "adf_results.npz",
             dists_s1=dists_s1_flat, dists_s2=dists_s2_flat,
             labels=labels_flat, phases=phases_flat,
             threshold=threshold, preds=preds_flat)
    print(f"\nSaved: {OUT_DIR / 'adf_results.npz'}")
    print(f"Output: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()

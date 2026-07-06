"""Evaluate Action Distribution Flow on TCL model features.

Same ADF pipeline as action_distribution_flow.py but operating on the 64-dim
internal features (forward_with_features) instead of logits. Runs on all models
(baseline + TCL betas) so we can compare whether ADF helps in the TCL feature space.

Usage:
    python phases_recognition/eval_tcl_adf.py
"""

import pathlib
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (average_precision_score, auc, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset.feature_dataset import VideoFeatureDataset, _collate_single_video
from models import instantiate_model

# ── Config ─────────────────────────────────────────────────────────────────
FEAT_ROOT = pathlib.Path("/home/helena/UCL_video_cataract/features_dino/")
EXPERIMENTS = {
    "Baseline": "/home/helena/experiments_cataract/baseline_detection_phases_unknown_mstcn_dino_v1_date=2026_06_11_17_02_41",
    "TCL β=0.02": "/home/helena/experiments_cataract/mstcn_dino_tcl_beta0.02_date=2026_06_29_16_32_06",
}
OUT_DIR = pathlib.Path("/home/helena/experiments_cataract/tcl_adf_eval/")

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

THRESHOLD_PERCENTILE = 95
N_RHO = 11


# ── Model + feature extraction ────────────────────────────────────────────

def load_model(exp_dir, device):
    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_model(cfg.model)
    state = torch.load(f"{exp_dir}/ckpt/best.pt", map_location="cpu",
                       weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def extract_features_per_video(model, split, device):
    """Extract 64-dim internal features + logit predictions per video."""
    ds = VideoFeatureDataset(root=str(FEAT_ROOT / split))
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=_collate_single_video)
    results = []
    for features, labels, name in loader:
        inp = features.unsqueeze(0).to(device)
        logits_list, feats = model.forward_with_features(inp)
        # feats: (1, 64, T) → (T, 64)
        feats_np = feats.squeeze(0).T.cpu().numpy().astype(np.float64)
        # predictions from last stage logits: (1, C, T) → argmax over C
        preds = logits_list[-1].squeeze(0).T.argmax(dim=1).cpu().numpy()

        phases_path = FEAT_ROOT / split / f"{name}_phases.npy"
        if phases_path.exists():
            phase_names = np.load(phases_path, allow_pickle=True)
        else:
            phase_names = np.full(len(labels), "", dtype=object)

        results.append((feats_np, labels.numpy(), preds, phase_names, name))
    return results


# ── Stage 1: Gaussian per class ───────────────────────────────────────────

def fit_class_gaussians(train_videos):
    per_class = defaultdict(list)
    for feats, labels, _, _, _ in train_videos:
        for c in range(N_CLASSES):
            mask = labels == c
            if mask.sum() > 0:
                per_class[c].append(feats[mask])

    D = train_videos[0][0].shape[1]
    means = np.zeros((N_CLASSES, D), dtype=np.float64)
    covs = np.zeros((N_CLASSES, D, D), dtype=np.float64)
    precisions = np.zeros_like(covs)

    for c in range(N_CLASSES):
        X = np.concatenate(per_class[c]).astype(np.float64)
        means[c] = X.mean(axis=0)
        lw = LedoitWolf()
        lw.fit(X)
        covs[c] = lw.covariance_
        precisions[c] = lw.precision_

    return means, covs, precisions


def mahalanobis_to_predicted(feats, preds, means, precisions):
    T = len(feats)
    dists = np.zeros(T, dtype=np.float64)
    for t in range(T):
        c = preds[t]
        diff = feats[t] - means[c]
        dists[t] = np.sqrt(max(0.0, diff @ precisions[c] @ diff))
    return dists


def calibrate_threshold(val_videos, means, precisions, percentile):
    all_dists = []
    for feats, labels, preds, _, _ in val_videos:
        known = labels >= 0
        if known.sum() == 0:
            continue
        dists = mahalanobis_to_predicted(feats[known], preds[known], means, precisions)
        all_dists.append(dists)
    threshold = np.percentile(np.concatenate(all_dists), percentile)
    print(f"  Threshold (p{percentile}): {threshold:.4f}")
    return threshold


# ── Stage 2: Wasserstein flow ─────────────────────────────────────────────

def find_transition_pairs(train_videos):
    pairs = set()
    for _, labels, _, _, _ in train_videos:
        known = labels[labels >= 0]
        for i in range(len(known) - 1):
            if known[i] != known[i + 1]:
                pairs.add((int(known[i]), int(known[i + 1])))
    print(f"  Found {len(pairs)} transition pairs in train")
    return pairs


def wasserstein_geodesic(mu_A, cov_A, mu_B, cov_B, rho):
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
    rhos = np.linspace(0, 1, n_rho)
    flows = {}
    for (a, b) in pairs:
        flow_dists = []
        for rho in rhos:
            mu_f, cov_f = wasserstein_geodesic(
                means[a], covs[a], means[b], covs[b], rho)
            prec_f = np.linalg.inv(cov_f + 1e-6 * np.eye(len(mu_f)))
            flow_dists.append((mu_f, prec_f))
        flows[(a, b)] = flow_dists
    return flows


def flow_distance(feat, flow_dists):
    min_dist = np.inf
    for mu_f, prec_f in flow_dists:
        diff = feat - mu_f
        d = np.sqrt(max(0.0, diff @ prec_f @ diff))
        if d < min_dist:
            min_dist = d
    return min_dist


def find_neighboring_transition(t, preds, window=15):
    T = len(preds)
    left_classes = [preds[i] for i in range(max(0, t - window), t) if preds[i] >= 0]
    right_classes = [preds[i] for i in range(t + 1, min(T, t + window + 1)) if preds[i] >= 0]
    if not left_classes or not right_classes:
        return None
    a = Counter(left_classes).most_common(1)[0][0]
    b = Counter(right_classes).most_common(1)[0][0]
    if a == b:
        return None
    return (int(a), int(b))


# ── Evaluation ────────────────────────────────────────────────────────────

def eval_f1(scores, labels, phase_names, threshold, label=""):
    gt = (labels == -1).astype(np.int32)
    pred = (scores > threshold).astype(np.int32)

    f1_unk = f1_score(gt, pred, pos_label=1, zero_division=0)
    prec   = precision_score(gt, pred, pos_label=1, zero_division=0)
    rec    = recall_score(gt, pred, pos_label=1, zero_division=0)
    f1_kn  = f1_score(gt, pred, pos_label=0, zero_division=0)

    print(f"\n  {label}")
    print(f"    Known   — F1={f1_kn:.4f}")
    print(f"    Unknown — F1={f1_unk:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

    per_phase = {}
    for phase in UNKNOWN_PHASES:
        mask_p = (labels == -1) & (phase_names == phase)
        n = mask_p.sum()
        if n < 5:
            print(f"    {phase:<30} SKIP (n={n})")
            continue
        eval_mask = (labels >= 0) | mask_p
        gt_p   = mask_p[eval_mask].astype(np.int32)
        pred_p = pred[eval_mask]
        f1_p   = f1_score(gt_p, pred_p, zero_division=0)
        prec_p = precision_score(gt_p, pred_p, zero_division=0)
        rec_p  = recall_score(gt_p, pred_p, zero_division=0)
        per_phase[phase] = {"f1": f1_p, "precision": prec_p, "recall": rec_p}
        print(f"    {phase:<30} F1={f1_p:.4f}  Prec={prec_p:.4f}  Rec={rec_p:.4f}  (n={n})")

    return {"f1_known": f1_kn, "f1_unknown": f1_unk,
            "prec": prec, "rec": rec, "per_phase": per_phase}


def eval_auroc(scores, labels, phase_names, label=""):
    gt = (labels == -1).astype(np.int32)
    if gt.sum() == 0 or gt.sum() == len(gt):
        return 0.0, {}

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


def run_adf(train_videos, val_videos, test_videos, exp_name):
    print(f"\nFitting Gaussians on 64-dim features...")
    means, covs, precisions = fit_class_gaussians(train_videos)

    print(f"Calibrating threshold on val...")
    threshold = calibrate_threshold(val_videos, means, precisions, THRESHOLD_PERCENTILE)

    # Stage 1: Mahalanobis on test
    all_feats, all_labels, all_preds, all_phases = [], [], [], []
    all_dists_s1 = []

    for feats, labels, preds, phase_names, _ in test_videos:
        dists = mahalanobis_to_predicted(feats, preds, means, precisions)
        all_feats.append(feats)
        all_labels.append(labels)
        all_preds.append(preds)
        all_phases.append(phase_names)
        all_dists_s1.append(dists)

    labels_flat = np.concatenate(all_labels)
    phases_flat = np.concatenate(all_phases)
    preds_flat = np.concatenate(all_preds)
    dists_s1_flat = np.concatenate(all_dists_s1)

    s1_unknown = dists_s1_flat > threshold
    print(f"Stage 1: {s1_unknown.sum()} frames marked unknown "
          f"({s1_unknown.sum()/len(s1_unknown)*100:.1f}%)")

    print(f"\nStage 1 eval:")
    s1_results = eval_auroc(dists_s1_flat, labels_flat, phases_flat,
                            f"{exp_name} — Stage 1 (Mahalanobis)")
    print(f"\nStage 1 F1 (threshold=p{THRESHOLD_PERCENTILE}):")
    f1_s1 = eval_f1(dists_s1_flat, labels_flat, phases_flat, threshold,
                    f"{exp_name} — Stage 1 F1")

    # Stage 2: Wasserstein flow
    print(f"\nFinding transition pairs...")
    pairs = find_transition_pairs(train_videos)

    print(f"Pre-computing Wasserstein flows ({N_RHO} points each)...")
    flows = precompute_flows(means, covs, pairs, N_RHO)
    print(f"  {len(flows)} flows ready")

    dists_s2_flat = dists_s1_flat.copy()
    n_reclassified = 0
    transition_diag = defaultdict(lambda: {"no_flow": [], "flow": []})

    offset = 0
    for vid_idx, (feats, labels, preds, _, _) in enumerate(test_videos):
        T = len(labels)
        dists_s1 = all_dists_s1[vid_idx]

        for t in range(T):
            global_t = offset + t
            if not s1_unknown[global_t]:
                continue

            pair = find_neighboring_transition(t, preds)
            if pair is None:
                continue

            d_flow = np.inf
            matched_pair = None
            for p in [pair, (pair[1], pair[0])]:
                if p in flows:
                    d = flow_distance(feats[t], flows[p])
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

    print(f"\n  Frames reclassified (unknown→known by flow): {n_reclassified}")

    print(f"\nStage 2 eval:")
    s2_results = eval_auroc(dists_s2_flat, labels_flat, phases_flat,
                            f"{exp_name} — Stage 1+2 (with flow)")
    print(f"\nStage 2 F1 (threshold=p{THRESHOLD_PERCENTILE}):")
    f1_s2 = eval_f1(dists_s2_flat, labels_flat, phases_flat, threshold,
                    f"{exp_name} — Stage 1+2 F1")

    # Transition distance diagnostic
    diag_summary = {}
    for pair in sorted(transition_diag.keys()):
        d_nf = transition_diag[pair]["no_flow"]
        d_f = transition_diag[pair]["flow"]
        if not d_nf:
            continue
        mean_nf = np.mean(d_nf)
        mean_f = np.mean(d_f)
        reduction = (mean_nf - mean_f) / mean_nf * 100
        a, b = pair
        print(f"  {CLASS_NAMES[a][:15]:>15}→{CLASS_NAMES[b][:15]:<15}"
              f"  n={len(d_nf):>4}  no_flow={mean_nf:.2f}  flow={mean_f:.2f}"
              f"  reduction={reduction:+.1f}%")
        diag_summary[pair] = {"mean_dist_no_flow": mean_nf,
                               "mean_dist_flow": mean_f,
                               "n_frames": len(d_nf)}

    gt_binary = (labels_flat == -1).astype(np.int32)
    return (s1_results, s2_results, diag_summary, threshold, f1_s1, f1_s2,
            dists_s1_flat, dists_s2_flat, gt_binary)


# ── Plots ─────────────────────────────────────────────────────────────────

def plot_pr_and_roc_curves(all_model_results, out_dir):
    """PR curves and ROC curves for Stage 1 and Stage 1+2 of all models."""
    entries = []
    for exp_name, tup in all_model_results.items():
        s1, s2, _, thr, f1_s1, f1_s2, scores_s1, scores_s2, gt_bin = tup
        entries.append((f"{exp_name} S1",   scores_s1, gt_bin, thr,
                        f1_s1["prec"], f1_s1["rec"]))
        entries.append((f"{exp_name} S1+2", scores_s2, gt_bin, thr,
                        f1_s2["prec"], f1_s2["rec"]))

    COLORS = plt.cm.tab10(np.linspace(0, 1, len(entries)))
    STYLES = ["-", "--"] * (len(entries) // 2 + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax_pr, ax_roc = axes

    for (label, scores, gt, thr, op_prec, op_rec), color, ls in zip(entries, COLORS, STYLES):
        ap = average_precision_score(gt, scores)
        prec_c, rec_c, _ = precision_recall_curve(gt, scores)
        ax_pr.plot(rec_c, prec_c, color=color, lw=1.5, ls=ls,
                   label=f"{label} (AP={ap:.3f})")
        ax_pr.scatter(op_rec, op_prec, color=color, s=60, zorder=5, marker="o")

        auroc = roc_auc_score(gt, scores)
        fpr_c, tpr_c, _ = roc_curve(gt, scores)
        ax_roc.plot(fpr_c, tpr_c, color=color, lw=1.5, ls=ls,
                    label=f"{label} (AUC={auroc:.3f})")
        fp = ((scores > thr) & (gt == 0)).sum()
        tn = ((scores <= thr) & (gt == 0)).sum()
        tp = ((scores > thr) & (gt == 1)).sum()
        fn = ((scores <= thr) & (gt == 1)).sum()
        ax_roc.scatter(fp / max(fp + tn, 1), tp / max(tp + fn, 1),
                       color=color, s=60, zorder=5, marker="o")

    ax_pr.set_xlabel("Recall", fontsize=12)
    ax_pr.set_ylabel("Precision", fontsize=12)
    ax_pr.set_title("Precision-Recall curves (• = p95 threshold)", fontsize=13)
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)
    ax_pr.legend(fontsize=7, loc="upper right")
    ax_pr.grid(alpha=0.3)

    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax_roc.set_xlabel("FPR", fontsize=12)
    ax_roc.set_ylabel("TPR", fontsize=12)
    ax_roc.set_title("ROC curves (• = p95 threshold)", fontsize=13)
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02)
    ax_roc.legend(fontsize=7, loc="lower right")
    ax_roc.grid(alpha=0.3)

    fig.tight_layout()
    path = out_dir / "tcl_adf_pr_roc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_stage1_vs_stage2_all_models(all_model_results, out_dir):
    """Per-phase AUROC: Stage1 vs Stage1+2 for each model."""
    active_phases = [p for p in UNKNOWN_PHASES
                     if any(p in r[0][1] or p in r[1][1]
                            for r in all_model_results.values())]
    if not active_phases:
        return

    n_models = len(all_model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (exp_name, (s1, s2, _, _, _, _, _, _, _)) in zip(axes, all_model_results.items()):
        x = np.arange(len(active_phases))
        w = 0.35
        v1 = [s1[1].get(p, 0) for p in active_phases]
        v2 = [s2[1].get(p, 0) for p in active_phases]
        ax.bar(x - w/2, v1, w, label="Stage 1", color="#1f77b4",
               edgecolor="black", lw=0.3)
        ax.bar(x + w/2, v2, w, label="Stage 1+2", color="#ff7f0e",
               edgecolor="black", lw=0.3)
        for vals, offset in [(v1, -w/2), (v2, w/2)]:
            for i, v in enumerate(vals):
                if v > 0.01:
                    ax.text(x[i] + offset, v + 0.01, f"{v:.2f}",
                            ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_", "\n") for p in active_phases], fontsize=7)
        ax.set_title(exp_name, fontsize=10, fontweight="bold")
        ax.axhline(0.5, color="grey", ls="--", lw=0.7)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("AUROC", fontsize=12)
    fig.suptitle("ADF on 64-dim features: Stage 1 vs Stage 1+2 per model",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = out_dir / "tcl_adf_stage1_vs_stage2.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_stage1_comparison(all_model_results, out_dir):
    """Stage 1 only: per-phase AUROC across all models (grouped by phase)."""
    active_phases = [p for p in UNKNOWN_PHASES
                     if any(p in r[0][1] for r in all_model_results.values())]
    if not active_phases:
        return

    model_names = list(all_model_results.keys())
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(active_phases))
    n = len(model_names)
    w = 0.8 / n
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    for i, exp_name in enumerate(model_names):
        s1, _, _, _, _, _, _, _, _ = all_model_results[exp_name]
        vals = [s1[1].get(p, 0) for p in active_phases]
        offset = (i - n / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=exp_name,
                      color=colors[i], edgecolor="black", lw=0.3)
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.2f}", ha="center", fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in active_phases], fontsize=9)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Stage 1 (Mahalanobis on 64-dim features) — all models", fontsize=13)
    ax.axhline(0.5, color="grey", ls="--", lw=0.7)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "tcl_adf_stage1_all_models.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_summary_table(all_model_results, out_dir):
    rows = []
    header = ["Model", "Stage", "Overall"] + [p.replace("_", "\n")[:12]
                                               for p in UNKNOWN_PHASES]
    for exp_name, (s1, s2, _, _, _, _, _, _, _) in all_model_results.items():
        for stage_label, res in [("S1", s1), ("S1+2", s2)]:
            row = [exp_name, stage_label, f"{res[0]:.4f}"]
            for p in UNKNOWN_PHASES:
                v = res[1].get(p)
                row.append(f"{v:.3f}" if v is not None else "—")
            rows.append(row)

    fig, ax = plt.subplots(figsize=(18, 1 + len(rows) * 0.45))
    ax.axis("off")
    table = ax.table(cellText=[header] + rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)
    for j in range(len(header)):
        table[0, j].set_text_props(fontweight="bold")

    # Highlight S1+2 rows where it beats S1
    row_idx = 1
    for exp_name, (s1, s2, _, _, _, _, _, _, _) in all_model_results.items():
        if s2[0] > s1[0]:
            for j in range(len(header)):
                table[row_idx + 1, j].set_facecolor("#d4edda")
        row_idx += 2

    ax.set_title("ADF on 64-dim features — all models summary", fontsize=13, pad=20)
    fig.tight_layout()
    path = out_dir / "tcl_adf_summary_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_model_results = {}

    for exp_name, exp_path in EXPERIMENTS.items():
        print("\n" + "=" * 60)
        print(f"MODEL: {exp_name}")
        print("=" * 60)

        model = load_model(exp_path, device)

        print("Extracting 64-dim features (train / val / test)...")
        train_videos = extract_features_per_video(model, "train", device)
        val_videos = extract_features_per_video(model, "val", device)
        test_videos = extract_features_per_video(model, "test", device)

        for split, vids in [("train", train_videos), ("val", val_videos),
                             ("test", test_videos)]:
            n_frames = sum(len(l) for _, l, _, _, _ in vids)
            n_unk = sum((l == -1).sum() for _, l, _, _, _ in vids)
            print(f"  {split}: {len(vids)} videos, {n_frames:,} frames, {n_unk:,} unknown")

        del model
        torch.cuda.empty_cache()

        s1, s2, diag, thr, f1_s1, f1_s2, scores_s1, scores_s2, gt_bin = run_adf(
            train_videos, val_videos, test_videos, exp_name)
        all_model_results[exp_name] = (s1, s2, diag, thr, f1_s1, f1_s2,
                                       scores_s1, scores_s2, gt_bin)

    # ── Final comparison table ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON — Stage 1 vs Stage 1+2, all models")
    print("=" * 60)
    print(f"\n  {'Model':<20} {'Stage':>6} {'Overall':>8}", end="")
    for p in UNKNOWN_PHASES:
        print(f" {p[:10]:>10}", end="")
    print()
    print(f"  {'-'*80}")

    for exp_name, (s1, s2, _, _, _, _, _, _, _) in all_model_results.items():
        for stage_label, res in [("S1", s1), ("S1+2", s2)]:
            marker = " ▲" if stage_label == "S1+2" and res[0] > s1[0] else ""
            print(f"  {exp_name:<20} {stage_label:>6} {res[0]:>8.4f}{marker}", end="")
            for p in UNKNOWN_PHASES:
                v = res[1].get(p)
                print(f" {v:>10.4f}" if v is not None else f" {'—':>10}", end="")
            print()
        print()

    # ── F1 / Precision / Recall summary ───────────────────────────────────
    print("\n" + "=" * 60)
    print("F1 / PRECISION / RECALL (threshold=p95 val)")
    print("=" * 60)

    for metric, key in [("F1", "f1"), ("Precision", "precision"), ("Recall", "recall")]:
        print(f"\n  Per-phase {metric}:")
        print(f"  {'Model':<20} {'Stage':>6}", end="")
        for p in UNKNOWN_PHASES:
            print(f" {p[:10]:>10}", end="")
        print()
        print(f"  {'-'*80}")
        for exp_name, (_, _, _, _, f1_s1, f1_s2, _, _, _) in all_model_results.items():
            for stage_label, f1_res in [("S1", f1_s1), ("S1+2", f1_s2)]:
                print(f"  {exp_name:<20} {stage_label:>6}", end="")
                for p in UNKNOWN_PHASES:
                    pp = f1_res["per_phase"].get(p)
                    print(f" {pp[key]:>10.4f}" if pp else f" {'—':>10}", end="")
                print()
        print()

    print(f"\n  Overall (unknown class):")
    print(f"  {'Model':<20} {'Stage':>6} {'F1_unk':>8} {'Prec':>8} {'Rec':>8} {'F1_kn':>8}")
    print(f"  {'-'*62}")
    for exp_name, (_, _, _, _, f1_s1, f1_s2, _, _, _) in all_model_results.items():
        for stage_label, f1_res in [("S1", f1_s1), ("S1+2", f1_s2)]:
            print(f"  {exp_name:<20} {stage_label:>6}"
                  f" {f1_res['f1_unknown']:>8.4f}"
                  f" {f1_res['prec']:>8.4f}"
                  f" {f1_res['rec']:>8.4f}"
                  f" {f1_res['f1_known']:>8.4f}")
        print()

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_pr_and_roc_curves(all_model_results, OUT_DIR)
    plot_stage1_vs_stage2_all_models(all_model_results, OUT_DIR)
    plot_stage1_comparison(all_model_results, OUT_DIR)
    plot_summary_table(all_model_results, OUT_DIR)

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()

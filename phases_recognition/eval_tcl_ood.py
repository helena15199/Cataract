"""Evaluate OOD detection on MSTCN trained with Temporal Clustering Loss.

Compares features from the TCL model vs the baseline MSTCN:
  1. Extract 64-dim internal features from both models
  2. Mahalanobis / KNN / RMDS on each
  3. Per-phase AUROC comparison
  4. t-SNE visualization: before vs after TCL

Usage:
    python phases_recognition/eval_tcl_ood.py
"""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.covariance import LedoitWolf
from sklearn.manifold import TSNE
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              f1_score, precision_score, recall_score,
                              classification_report, precision_recall_curve,
                              roc_curve, auc)
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset.feature_dataset import VideoFeatureDataset, _collate_single_video
from models import instantiate_model

# ── Config ─────────────────────────────────────────────────────────────────
FEAT_ROOT = pathlib.Path("/home/helena/UCL_video_cataract/features_dino/")
EXPERIMENTS = {
    "Baseline": "/home/helena/experiments_cataract/baseline_detection_phases_unknown_mstcn_dino_v1_date=2026_06_11_17_02_41",
    "TCL β=0.01": "/home/helena/experiments_cataract/mstcn_dino_tcl_beta0.01_date=2026_06_29_13_50_35",
    "TCL β=0.02": "/home/helena/experiments_cataract/mstcn_dino_tcl_beta0.02_date=2026_06_29_16_32_06",
    "TCL β=0.05": "/home/helena/experiments_cataract/mstcn_dino_tcl_beta0.05_date=2026_06_29_16_44_54",
    "TCL β=0.01+FT": "/home/helena/experiments_cataract/mstcn_dino_tcl_beta0.01_ft_date=2026_07_06_15_46_02",
    "TCL β=0.02+FT": "/home/helena/experiments_cataract/mstcn_dino_tcl_beta0.02_ft_date=2026_07_06_15_47_59",
    "TCL β=0.05+FT": "/home/helena/experiments_cataract/mstcn_dino_tcl_beta0.05_ft_date=2026_07_06_15_50_00",
}
OUT_DIR = pathlib.Path("/home/helena/experiments_cataract/tcl_ood_eval/")

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


# ── Model + features ──────────────────────────────────────────────────────

def load_model(exp_dir, device):
    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_model(cfg.model)
    state = torch.load(f"{exp_dir}/ckpt/best.pt", map_location="cpu",
                       weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def extract_features(model, split, device):
    """Extract 64-dim internal features (last stage before output_proj)."""
    ds = VideoFeatureDataset(root=str(FEAT_ROOT / split))
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=_collate_single_video)
    all_feats, all_labels, all_phases, all_names = [], [], [], []
    for features, labels, name in loader:
        _, f = model.forward_with_features(features.unsqueeze(0).to(device))
        all_feats.append(f.squeeze(0).T.cpu().numpy())  # (T, 64)
        all_labels.append(labels.numpy())
        phases_path = FEAT_ROOT / split / f"{name}_phases.npy"
        if phases_path.exists():
            all_phases.append(np.load(phases_path, allow_pickle=True))
        else:
            all_phases.append(np.full(len(labels), "", dtype=object))
        all_names.append(name)
    return all_feats, all_labels, all_phases, all_names


def concat_known(feats_list, labels_list):
    feats = np.concatenate(feats_list)
    labels = np.concatenate(labels_list)
    mask = labels >= 0
    return feats[mask], labels[mask]


def concat_all(feats_list, labels_list, phases_list):
    feats = np.concatenate(feats_list)
    labels = np.concatenate(labels_list)
    phases = np.concatenate(phases_list)
    return feats, labels, phases


# ── Scoring ────────────────────────────────────────────────────────────────

def fit_and_score_knn(train_feats, test_feats, k=KNN_K):
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn.fit(train_feats)
    dists, _ = nn.kneighbors(test_feats)
    return dists[:, -1]


def fit_and_score_rmds(train_feats, train_labels, test_feats, n_classes):
    D = train_feats.shape[1]
    class_means = np.zeros((n_classes, D), dtype=np.float64)
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
    prec_class = lw.precision_.astype(np.float64)

    bg_mean = train_feats.mean(0).astype(np.float64)
    lw_bg = LedoitWolf()
    lw_bg.fit(train_feats)
    prec_bg = lw_bg.precision_.astype(np.float64)

    best_class = np.full(len(test_feats), np.inf, dtype=np.float64)
    for c in range(n_classes):
        diff = test_feats - class_means[c]
        d = (diff @ prec_class * diff).sum(1)
        best_class = np.minimum(best_class, d)
    diff_bg = test_feats - bg_mean
    d_bg = (diff_bg @ prec_bg * diff_bg).sum(1)
    return best_class - d_bg


# ── Threshold calibration ──────────────────────────────────────────────────

def calibrate_threshold(scores_val_known, percentile=95):
    """Set OOD threshold at given percentile of known val scores."""
    return np.percentile(scores_val_known, percentile)


# ── Evaluation ─────────────────────────────────────────────────────────────

def eval_f1(scores, labels, phases, threshold, label=""):
    """Evaluate with F1: known vs unknown (binary) + per unknown phase."""
    gt_unknown = (labels == -1).astype(np.int32)
    pred_unknown = (scores > threshold).astype(np.int32)

    # Binary F1: known vs unknown
    f1_unk = f1_score(gt_unknown, pred_unknown, pos_label=1)
    prec_unk = precision_score(gt_unknown, pred_unknown, pos_label=1, zero_division=0)
    rec_unk = recall_score(gt_unknown, pred_unknown, pos_label=1, zero_division=0)
    f1_known = f1_score(gt_unknown, pred_unknown, pos_label=0)
    auroc = roc_auc_score(gt_unknown, scores)

    print(f"\n  {label}")
    print(f"    Threshold: {threshold:.4f}")
    print(f"    Known   — F1={f1_known:.4f}")
    print(f"    Unknown — F1={f1_unk:.4f}  Prec={prec_unk:.4f}  Rec={rec_unk:.4f}")
    print(f"    AUROC={auroc:.4f}")

    # Per unknown phase F1
    per_phase = {}
    print(f"    Per-phase:")
    for phase in UNKNOWN_PHASES:
        mask_p = (labels == -1) & (phases == phase)
        n = mask_p.sum()
        if n < 5:
            print(f"      {phase:<30} SKIP (n={n})")
            continue
        # Binary eval: this phase vs known (1-vs-rest)
        eval_mask = (labels >= 0) | mask_p
        gt_p = mask_p[eval_mask].astype(np.int32)
        pred_p = pred_unknown[eval_mask]
        f1_p = f1_score(gt_p, pred_p, zero_division=0)
        prec_p = precision_score(gt_p, pred_p, zero_division=0)
        recall_p = recall_score(gt_p, pred_p, zero_division=0)
        per_phase[phase] = {"f1": f1_p, "precision": prec_p, "recall": recall_p, "n": int(n)}
        print(f"      {phase:<30} F1={f1_p:.4f}  Prec={prec_p:.4f}  Rec={recall_p:.4f}  (n={n})")

    return {"f1_known": f1_known, "f1_unknown": f1_unk,
            "prec_unknown": prec_unk, "rec_unknown": rec_unk,
            "auroc": auroc, "per_phase": per_phase}


# ── t-SNE visualization ───────────────────────────────────────────────────

def plot_tsne_comparison(feats_base, feats_tcl, labels, phases, out_dir):
    """t-SNE before vs after TCL, with unknown phases highlighted."""
    np.random.seed(42)
    N_PER_CLASS = 200
    N_PER_UNK = 300

    idx_keep = []
    for c in range(N_CLASSES):
        idx_c = np.where(labels == c)[0]
        if len(idx_c) > 0:
            idx_keep.append(np.random.choice(idx_c, min(N_PER_CLASS, len(idx_c)),
                                              replace=False))
    for phase in UNKNOWN_PHASES:
        idx_p = np.where((labels == -1) & (phases == phase))[0]
        if len(idx_p) > 0:
            idx_keep.append(np.random.choice(idx_p, min(N_PER_UNK, len(idx_p)),
                                              replace=False))
    idx = np.sort(np.concatenate(idx_keep))
    labels_s = labels[idx]
    phases_s = phases[idx]

    COLORS = [
        '#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#e377c2',
        '#17becf', '#bcbd22', '#d4a017', '#f07800', '#3a7d44', '#6b5b95',
    ]
    UNK_COLORS = {
        'Malyugin_ring_insertion': '#FF0000',
        'Malyugin_ring_removal': '#FF8800',
        'Suture': '#FF00FF',
        'Iris_manipulation': '#00CCFF',
        'Trypan_blue_injection': '#00FF44',
    }

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    for ax, feats, title in [(axes[0], feats_base[idx], "Baseline MSTCN"),
                              (axes[1], feats_tcl[idx], "MSTCN + TCL")]:
        print(f"  Computing t-SNE for {title}...")
        emb = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(feats)

        for c in range(N_CLASSES):
            mask = labels_s == c
            if mask.sum() > 0:
                ax.scatter(emb[mask, 0], emb[mask, 1], c=COLORS[c], s=12,
                           alpha=0.5, linewidths=0)

        for phase, color in UNK_COLORS.items():
            mask = (labels_s == -1) & (phases_s == phase)
            if mask.sum() > 0:
                ax.scatter(emb[mask, 0], emb[mask, 1], c=color, s=80,
                           alpha=0.8, marker='*', edgecolors='black',
                           linewidths=0.5, zorder=3)

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=n) for c, n in zip(COLORS, CLASS_NAMES)]
    handles += [plt.scatter([], [], c=c, s=80, marker='*', edgecolors='black',
                            linewidths=0.5, label=n) for n, c in UNK_COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("t-SNE: feature space before vs after Temporal Clustering Loss",
                 fontsize=18, y=1.02)
    fig.tight_layout()
    path = out_dir / "tsne_baseline_vs_tcl.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pr_and_roc_curves(results, out_dir):
    """PR curves and ROC curves for all methods, with p95 operating point marked."""
    COLORS = plt.cm.tab10(np.linspace(0, 1, len(results)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax_pr, ax_roc = axes

    for (label, res), color in zip(results.items(), COLORS):
        scores = res["scores"]
        gt = res["labels_binary"]
        thr = res["threshold"]

        # PR curve
        prec, rec, thresholds_pr = precision_recall_curve(gt, scores)
        ap = average_precision_score(gt, scores)
        ax_pr.plot(rec, prec, color=color, lw=1.5, label=f"{label} (AP={ap:.3f})")
        # Mark p95 operating point
        op_prec = res["prec_unknown"]
        op_rec = res["rec_unknown"]
        ax_pr.scatter(op_rec, op_prec, color=color, s=60, zorder=5, marker="o")

        # ROC curve
        fpr, tpr, _ = roc_curve(gt, scores)
        auroc = roc_auc_score(gt, scores)
        ax_roc.plot(fpr, tpr, color=color, lw=1.5, label=f"{label} (AUC={auroc:.3f})")
        # Mark p95 operating point: FPR = 1 - specificity at threshold
        fp_at_thr = ((scores > thr) & (gt == 0)).sum()
        tn_at_thr = ((scores <= thr) & (gt == 0)).sum()
        fpr_op = fp_at_thr / max(fp_at_thr + tn_at_thr, 1)
        tp_at_thr = ((scores > thr) & (gt == 1)).sum()
        fn_at_thr = ((scores <= thr) & (gt == 1)).sum()
        tpr_op = tp_at_thr / max(tp_at_thr + fn_at_thr, 1)
        ax_roc.scatter(fpr_op, tpr_op, color=color, s=60, zorder=5, marker="o")

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
    path = out_dir / "pr_roc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_comparison_bars(results, out_dir):
    """Bar chart comparing recall per unknown phase: baseline vs TCL."""
    phases = [p for p in UNKNOWN_PHASES
              if any(p in results[m]["per_phase"] for m in results)]
    if not phases:
        return

    methods = list(results.keys())
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(phases))
    n = len(methods)
    w = 0.8 / n
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, method in enumerate(methods):
        vals = [results[method]["per_phase"].get(p, {}).get("f1", 0) for p in phases]
        offset = (i - n / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=method,
                      color=colors[i % len(colors)], edgecolor="black", lw=0.3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.2f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in phases], fontsize=9)
    ax.set_ylabel("F1", fontsize=12)
    ax.set_title("OOD Detection — F1 per unknown phase (threshold=p95 val)", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "tcl_ood_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ══════════════════════════════════════════════════════════════════════
    # CLOSED-SET CLASSIFICATION (accuracy + F1 macro on 11 known phases)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CLOSED-SET CLASSIFICATION — all models on test set")
    print("=" * 60)

    classif_results = {}
    models_data = {}

    for exp_name, exp_path in EXPERIMENTS.items():
        print(f"\n--- {exp_name} ---")
        model = load_model(exp_path, device)

        # Classification: run inference, get predictions on known frames
        ds = VideoFeatureDataset(root=str(FEAT_ROOT / "test"))
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            collate_fn=_collate_single_video)

        all_preds, all_gt = [], []
        with torch.no_grad():
            for features, labels, name in loader:
                logits = model(features.unsqueeze(0).to(device))
                preds = logits[-1].squeeze(0).T.argmax(dim=1).cpu().numpy()
                labels_np = labels.numpy()
                known = labels_np >= 0
                all_preds.append(preds[known])
                all_gt.append(labels_np[known])

        all_preds = np.concatenate(all_preds)
        all_gt = np.concatenate(all_gt)
        acc = (all_preds == all_gt).mean()
        f1_macro = f1_score(all_gt, all_preds, average="macro")
        f1_per_class = f1_score(all_gt, all_preds, average=None)

        classif_results[exp_name] = {"accuracy": acc, "f1_macro": f1_macro,
                                      "f1_per_class": f1_per_class}
        print(f"  Accuracy={acc:.4f}  F1 macro={f1_macro:.4f}")

        # Extract features for OOD eval
        train_f, train_l, _, _ = extract_features(model, "train", device)
        val_f, val_l, _, _ = extract_features(model, "val", device)
        test_f, test_l, test_p, _ = extract_features(model, "test", device)

        tr_feats, tr_labels = concat_known(train_f, train_l)
        val_feats, _ = concat_known(val_f, val_l)
        te_feats, te_labels, te_phases = concat_all(test_f, test_l, test_p)

        models_data[exp_name] = {
            "train_feats": tr_feats, "train_labels": tr_labels,
            "val_feats": val_feats,
            "test_feats": te_feats, "test_labels": te_labels, "test_phases": te_phases,
        }
        print(f"  Features: train={len(tr_feats):,}, val={len(val_feats):,}, test={len(te_feats):,}")
        del model
        torch.cuda.empty_cache()

    # Classification summary table
    print("\n" + "-" * 60)
    print(f"  {'Model':<20} {'Accuracy':>10} {'F1 macro':>10}")
    print(f"  {'-'*42}")
    for name, res in classif_results.items():
        print(f"  {name:<20} {res['accuracy']:>10.4f} {res['f1_macro']:>10.4f}")
    print(f"\n  Per-class F1:")
    print(f"  {'Model':<20}", end="")
    for c in CLASS_NAMES:
        print(f" {c[:8]:>8}", end="")
    print()
    for name, res in classif_results.items():
        print(f"  {name:<20}", end="")
        for f in res["f1_per_class"]:
            print(f" {f:>8.3f}", end="")
        print()

    # ── OOD scores + F1 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OOD DETECTION — All betas (F1 with threshold at p95 of val)")
    print("=" * 60)

    all_results = {}

    for exp_name, data in models_data.items():
        tr_f, tr_l = data["train_feats"], data["train_labels"]
        val_f = data["val_feats"]
        te_f, te_l, te_p = data["test_feats"], data["test_labels"], data["test_phases"]

        for method in ["KNN", "RMDS"]:
            label = f"{exp_name} {method}"
            if method == "KNN":
                nn = NearestNeighbors(n_neighbors=KNN_K, metric="euclidean").fit(tr_f)
                val_scores = nn.kneighbors(val_f)[0][:, -1]
                test_scores = nn.kneighbors(te_f)[0][:, -1]
            else:
                test_scores = fit_and_score_rmds(tr_f, tr_l, te_f, N_CLASSES)
                val_scores = fit_and_score_rmds(tr_f, tr_l, val_f, N_CLASSES)

            threshold = calibrate_threshold(val_scores, percentile=95)
            result = eval_f1(test_scores, te_l, te_p, threshold, label)
            result["scores"] = test_scores
            result["labels_binary"] = (te_l == -1).astype(np.int32)
            result["threshold"] = threshold
            all_results[label] = result

    # ── Comparison table ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"\n  {'Method':<25} {'AUROC':>7} {'F1_kn':>7} {'F1_unk':>7} {'Prec':>7} {'Rec':>7}")
    print(f"  {'-'*63}")
    for method, res in all_results.items():
        print(f"  {method:<25} {res['auroc']:>7.4f} {res['f1_known']:>7.4f} "
              f"{res['f1_unknown']:>7.4f} {res['prec_unknown']:>7.4f} {res['rec_unknown']:>7.4f}")

    for metric, key in [("Per-phase F1", "f1"), ("Per-phase Precision", "precision"), ("Per-phase Recall", "recall")]:
        print(f"\n  {metric}:")
        print(f"  {'Method':<25}", end="")
        for p in UNKNOWN_PHASES:
            print(f" {p[:10]:>10}", end="")
        print()
        print(f"  {'-'*77}")
        for method, res in all_results.items():
            print(f"  {method:<25}", end="")
            for p in UNKNOWN_PHASES:
                pp = res["per_phase"].get(p)
                if pp:
                    print(f" {pp[key]:>10.4f}", end="")
                else:
                    print(f" {'—':>10}", end="")
            print()

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    # ── Figure 1: Classification table (accuracy + F1 macro + F1 per-class)
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis("off")
    header = ["Model", "Accuracy", "F1 macro"] + [c.replace("_", "\n")[:12] for c in CLASS_NAMES]
    rows = []
    for name, res in classif_results.items():
        row = [name, f"{res['accuracy']:.4f}", f"{res['f1_macro']:.4f}"]
        row += [f"{f:.3f}" for f in res["f1_per_class"]]
        rows.append(row)
    table = ax.table(cellText=[header] + rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    for j in range(len(header)):
        table[0, j].set_text_props(fontweight="bold")
    # Highlight best F1 macro
    best_f1 = max(r["f1_macro"] for r in classif_results.values())
    for i, (name, res) in enumerate(classif_results.items(), start=1):
        if res["f1_macro"] == best_f1:
            for j in range(len(header)):
                table[i, j].set_facecolor("#d4edda")
    ax.set_title("Closed-set classification — all models on test set", fontsize=14, pad=20)
    fig.tight_layout()
    path = OUT_DIR / "classif_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 2: OOD summary table (AUROC + F1 known/unknown)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    header = ["Method", "AUROC", "F1 known", "F1 unknown", "Precision", "Recall"]
    rows = []
    for method, res in all_results.items():
        rows.append([method, f"{res['auroc']:.4f}", f"{res['f1_known']:.4f}",
                      f"{res['f1_unknown']:.4f}", f"{res['prec_unknown']:.4f}",
                      f"{res['rec_unknown']:.4f}"])
    table = ax.table(cellText=[header] + rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    for j in range(len(header)):
        table[0, j].set_text_props(fontweight="bold")
    best_auroc = max(r["auroc"] for r in all_results.values())
    for i, (method, res) in enumerate(all_results.items(), start=1):
        if res["auroc"] == best_auroc:
            for j in range(len(header)):
                table[i, j].set_facecolor("#d4edda")
    ax.set_title("OOD Detection — all models (threshold = p95 val)", fontsize=14, pad=20)
    fig.tight_layout()
    path = OUT_DIR / "ood_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 3: Per-phase AUROC bars (grouped by phase, colored by model)
    phases = [p for p in UNKNOWN_PHASES
              if any(p in all_results[m]["per_phase"] for m in all_results)]
    # Only keep KNN methods for cleaner plot (or all)
    methods = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(phases))
    n = len(methods)
    w = 0.8 / n
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    for i, method in enumerate(methods):
        vals = [all_results[method]["per_phase"].get(p, {}).get("f1", 0) for p in phases]
        offset = (i - n / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=method,
                      color=colors[i], edgecolor="black", lw=0.3)
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.2f}", ha="center", fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in phases], fontsize=9)
    ax.set_ylabel("F1", fontsize=12)
    ax.set_title("Per unknown phase F1 — all models (threshold = p95 val)", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "per_phase_f1.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 4: AUROC per phase (same layout but AUROC instead of F1)
    fig, ax = plt.subplots(figsize=(16, 6))
    for i, method in enumerate(methods):
        # Compute AUROC per phase
        data = models_data[method.rsplit(" ", 1)[0]] if " " in method else None
        vals = []
        for p in phases:
            pp = all_results[method]["per_phase"].get(p)
            if pp:
                vals.append(pp.get("f1", 0))  # already have F1
            else:
                vals.append(0)
        # We need AUROC per phase — recompute from scores
    # Actually let's store AUROC per phase in eval_f1
    # For now, use the per-phase F1 plot above and add AUROC to the summary

    # ── Figure 4: Combined view — classif F1 macro vs OOD AUROC per beta
    model_names = list(EXPERIMENTS.keys())
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(model_names))
    f1_vals = [classif_results[m]["f1_macro"] for m in model_names]
    # Best AUROC across KNN/RMDS for each model
    auroc_vals = []
    for m in model_names:
        aucs = [all_results[f"{m} {method}"]["auroc"]
                for method in ["KNN", "RMDS"] if f"{m} {method}" in all_results]
        auroc_vals.append(max(aucs) if aucs else 0)

    bars1 = ax1.bar(x - 0.2, f1_vals, 0.35, label="Classif F1 macro", color="#1f77b4",
                    edgecolor="black", lw=0.5)
    bars2 = ax2.bar(x + 0.2, auroc_vals, 0.35, label="Best OOD AUROC", color="#ff7f0e",
                    edgecolor="black", lw=0.5)

    for bar, v in zip(bars1, f1_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                 ha="center", fontsize=9)
    for bar, v in zip(bars2, auroc_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                 ha="center", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=10)
    ax1.set_ylabel("Classification F1 macro", fontsize=12, color="#1f77b4")
    ax2.set_ylabel("OOD AUROC", fontsize=12, color="#ff7f0e")
    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax1.axhline(f1_vals[0], color="#1f77b4", ls="--", lw=0.8, alpha=0.5)
    ax2.axhline(auroc_vals[0], color="#ff7f0e", ls="--", lw=0.8, alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right")

    ax1.set_title("Trade-off: Classification F1 vs OOD AUROC per β", fontsize=14)
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "tradeoff_f1_vs_auroc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Figure 5: PR curves + ROC curves for all OOD methods
    plot_pr_and_roc_curves(all_results, OUT_DIR)

    print(f"\nAll figures saved to: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()

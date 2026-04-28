"""Evaluation script for the MS-TCN++ temporal model.

Usage (from repo root):
    python phases_recognition/evaluate_temporal.py \
        --config phases_recognition/configs/config_mstcn.yaml \
        --ckpt /home/helena/experiments_cataract/<exp>/ckpt/best.pt \
        --out_dir /home/helena/experiments_cataract/<exp>/eval_test/

Options:
    --smooth_window  Taille de la fenêtre de lissage majority-vote (défaut: 15).
                     0 = pas de lissage.

Outputs:
    metrics.json            — métriques raw + smoothed (frame + segment)
    confusion_matrix.png    — matrice raw
    confusion_matrix_smoothed.png
    phase_timeline.png      — 3 barres par vidéo : GT / Raw / Smoothed
    per_video_f1.png        — F1 par vidéo (raw vs smoothed)
"""

import argparse
import json
import pathlib
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import tqdm

from dataset.feature_dataset import VideoFeatureDataset, _collate_single_video
from metrics.cataract_metrics import CataractMetrics
from models import instantiate_model
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Post-processing: majority-vote temporal smoothing
# ---------------------------------------------------------------------------

def majority_vote_smooth(seq: list[int], window: int) -> list[int]:
    """
    Sliding-window majority vote on a frame-level prediction sequence.
    Each frame takes the most common label in [t - window//2, t + window//2].
    Eliminates short spurious segments without changing phase boundaries much.
    """
    if window <= 1:
        return seq
    n = len(seq)
    half = window // 2
    smoothed = []
    for t in range(n):
        lo = max(0, t - half)
        hi = min(n, t + half + 1)
        smoothed.append(Counter(seq[lo:hi]).most_common(1)[0][0])
    return smoothed


# ---------------------------------------------------------------------------
# Segment-level metrics
# ---------------------------------------------------------------------------

def _get_segments(seq: list[int]) -> list[tuple[int, int, int]]:
    """(label, start, end) segments from a frame-level sequence."""
    if not seq:
        return []
    segments, start = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            segments.append((seq[start], start, i))
            start = i
    return segments


def edit_score(pred_seq: list[int], gt_seq: list[int]) -> float:
    """Normalised Levenshtein distance on segment label sequences. ∈ [0,1], ↑ better."""
    pred_labels = [s[0] for s in _get_segments(pred_seq)]
    gt_labels   = [s[0] for s in _get_segments(gt_seq)]
    n, m = len(pred_labels), len(gt_labels)
    if n == 0 and m == 0:
        return 1.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred_labels[i - 1] == gt_labels[j - 1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return 1.0 - dp[n][m] / max(n, m)


def f1_at_overlap(pred_seq: list[int], gt_seq: list[int],
                  threshold: float) -> float:
    pred_segs = _get_segments(pred_seq)
    gt_segs   = _get_segments(gt_seq)
    tp, gt_matched = 0, [False] * len(gt_segs)
    for p_label, p_start, p_end in pred_segs:
        best_iou, best_j = 0.0, -1
        for j, (g_label, g_start, g_end) in enumerate(gt_segs):
            if gt_matched[j] or g_label != p_label:
                continue
            inter = max(0, min(p_end, g_end) - max(p_start, g_start))
            union = (p_end - p_start) + (g_end - g_start) - inter
            iou   = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= threshold and best_j >= 0:
            tp += 1
            gt_matched[best_j] = True
    precision = tp / len(pred_segs) if pred_segs else 0.0
    recall    = tp / len(gt_segs)   if gt_segs   else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_all_metrics(
    video_results: list[tuple[list[int], list[int], str]],
    num_classes: int,
    class_names: list[str],
    others_classes: list[str],
    prefix: str = "",
) -> tuple[dict, dict, list[int], list[int]]:
    """
    Compute frame-level + segment-level metrics for a set of (gt, pred) sequences.
    Returns (all_metrics, per_video_f1, all_preds_flat, all_labels_flat).
    """
    metrics_fn = CataractMetrics(num_classes=num_classes,
                                  class_names=class_names,
                                  others_classes=others_classes)
    per_video_fn = CataractMetrics(num_classes=num_classes,
                                    class_names=class_names,
                                    others_classes=others_classes)
    all_preds, all_labels = [], []
    edit_scores, f1_10, f1_25, f1_50 = [], [], [], []
    video_f1s = {}

    for gt_seq, pred_seq, video_name in video_results:
        t = len(gt_seq)
        dummy = torch.zeros(t, num_classes)
        dummy[range(t), pred_seq] = 10.0
        metrics_fn.update(dummy, torch.tensor(gt_seq))
        all_preds.extend(pred_seq)
        all_labels.extend(gt_seq)

        edit_scores.append(edit_score(pred_seq, gt_seq))
        f1_10.append(f1_at_overlap(pred_seq, gt_seq, 0.10))
        f1_25.append(f1_at_overlap(pred_seq, gt_seq, 0.25))
        f1_50.append(f1_at_overlap(pred_seq, gt_seq, 0.50))

        per_video_fn.reset()
        per_video_fn.update(dummy, torch.tensor(gt_seq))
        video_f1s[video_name] = per_video_fn.compute().get("global/f1_macro", 0.0)

    p = (prefix + "/") if prefix else ""
    frame_m = {f"{p}{k}": v for k, v in metrics_fn.compute().items()}
    seg_m = {
        f"{p}segment/edit_score": float(np.mean(edit_scores)),
        f"{p}segment/f1@10":      float(np.mean(f1_10)),
        f"{p}segment/f1@25":      float(np.mean(f1_25)),
        f"{p}segment/f1@50":      float(np.mean(f1_50)),
    }
    return {**frame_m, **seg_m}, video_f1s, all_preds, all_labels


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _draw_bars(ax, sequences: list[tuple[list[int], str]], colors, n_frames):
    """Draw stacked horizontal bars for each sequence on a single ax."""
    for bar_idx, (seq, _) in enumerate(sequences):
        start = 0
        for i in range(1, n_frames + 1):
            if i == n_frames or seq[i] != seq[start]:
                left  = start / n_frames
                width = (i - start) / n_frames
                ax.barh(bar_idx, width, left=left,
                        color=colors[seq[start]], height=0.8, align="center")
                start = i


PHASE_COLORS = [
    "#90EE90",  # vert clair
    "#228B22",  # vert foncé
    "#ADD8E6",  # bleu très clair
    "#00008B",  # bleu foncé
    "#E74C3C",  # rouge
    "#F1C40F",  # jaune
    "#808080",  # gris
    "#8B4513",  # marron
    "#E67E22",  # orange
    "#8E44AD",  # violet
    "#FF69B4",  # rose
    "#111111",  # noir
]


def plot_phase_timeline(
    video_results_raw: list[tuple[list[int], list[int], str]],
    class_names: list[str],
    out_path: pathlib.Path,
):
    n_classes  = len(class_names)
    colors     = PHASE_COLORS[:n_classes]
    n_videos   = len(video_results_raw)
    bar_labels = ["GT", "Pred"]

    fig, axes = plt.subplots(n_videos, 1,
                             figsize=(14, n_videos * 1.2 + 1.5),
                             squeeze=False)

    for row, (gt_seq, pred_raw, video_name) in enumerate(video_results_raw):
        ax = axes[row, 0]
        n_frames = len(gt_seq)
        _draw_bars(ax, [(gt_seq, "GT"), (pred_raw, "Pred")], colors, n_frames)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(bar_labels, fontsize=7)
        ax.set_xticks([])
        ax.set_title(video_name, loc="left", fontsize=8, pad=2)
        for spine in ["top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(n_classes)]
    fig.legend(handles, class_names,
               loc="lower center", ncol=min(n_classes, 7),
               fontsize=7, bbox_to_anchor=(0.5, 0), frameon=False)
    fig.suptitle("Phase timeline — GT vs Predicted (test set)", fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_binary_ch_timeline(
    video_results_raw: list[tuple[list[int], list[int], str]],
    test_root: pathlib.Path,
    out_path: pathlib.Path,
):
    """One bar per video: GT CH (label=-1) vs predicted CH (from _binary_ch.npy)."""
    color_0 = "#4C9BE8"   # bleu  — non CH
    color_1 = "#E8734C"   # rouge — CH

    n_videos = len(video_results_raw)
    fig, axes = plt.subplots(n_videos, 1,
                             figsize=(14, n_videos * 1.0 + 1.5),
                             squeeze=False)

    for row, (gt_phase_seq, _, video_name) in enumerate(video_results_raw):
        ax = axes[row, 0]

        # GT binary: frames where original label was -1 (before masking) — reload from file
        label_file = test_root / f"{video_name}_labels.npy"
        ch_file    = test_root / f"{video_name}_binary_ch.npy"

        gt_labels_full = np.load(label_file)          # includes -1 for CH
        gt_binary  = (gt_labels_full == -1).astype(int).tolist()
        pred_binary = np.load(ch_file).tolist() if ch_file.exists() else [0] * len(gt_binary)

        n_frames = len(gt_binary)
        seqs = [(gt_binary, "GT CH"), (pred_binary, "Pred CH")]
        for bar_idx, (seq, _) in enumerate(seqs):
            start = 0
            for i in range(1, n_frames + 1):
                if i == n_frames or seq[i] != seq[start]:
                    left  = start / n_frames
                    width = (i - start) / n_frames
                    color = color_1 if seq[start] == 1 else color_0
                    ax.barh(bar_idx, width, left=left, color=color, height=0.8, align="center")
                    start = i

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["GT CH", "Pred CH"], fontsize=7)
        ax.set_xticks([])
        ax.set_title(video_name, loc="left", fontsize=8, pad=2)
        for spine in ["top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_0),
               plt.Rectangle((0, 0), 1, 1, color=color_1)]
    fig.legend(handles, ["Non Corneal_hydration", "Corneal_hydration"],
               loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, 0), frameon=False)
    fig.suptitle("Binary Corneal Hydration — GT vs Predicted (test set)", fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_confusion_matrix(all_preds, all_labels, class_names,
                          eval_indices, out_path, title):
    eval_names = [class_names[i] for i in eval_indices]
    cm = confusion_matrix(all_labels, all_preds, labels=eval_indices)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(cm_norm, display_labels=eval_names).plot(
        ax=ax, colorbar=True, xticks_rotation=45, values_format=".2f")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_per_video_f1(video_f1s_raw: dict, video_f1s_smooth: dict, out_path: pathlib.Path):
    names = list(video_f1s_raw.keys())
    raw   = [video_f1s_raw[n] for n in names]
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    names = [names[i][:40] for i in order]
    raw   = [raw[i] for i in order]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.45)))
    ax.barh(y, raw, height=0.6, color="#5b9bd5", label="Pred")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 macro (frame-level)")
    ax.axvline(x=np.mean(raw), color="#5b9bd5", linestyle="--", linewidth=1)
    ax.set_title("Per-video F1 macro — test set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, test_root, device):
    dataset = VideoFeatureDataset(root=str(test_root))
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         collate_fn=_collate_single_video)
    model.eval()
    results = []
    for features, labels, video_name in tqdm.tqdm(loader, desc="Inference"):
        features     = features.unsqueeze(0).to(device)
        stage_logits = model(features)
        last_logits  = stage_logits[-1].squeeze(0).T   # (T, C)
        preds = last_logits.argmax(dim=1).cpu().tolist()
        gt = labels.tolist()
        # Mask out binary-phase frames (label=-1) from GT and preds
        mask = [i for i, l in enumerate(gt) if l != -1]
        gt_clean   = [gt[i]   for i in mask]
        pred_clean = [preds[i] for i in mask]
        results.append((gt_clean, pred_clean, video_name))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str, ckpt_path: str, out_dir: str, smooth_window: int):
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
    print(f"Loaded checkpoint: epoch {state.get('epoch', '?')}")

    class_names    = list(config.dataset.class_names)
    others_classes = list(config.metrics.others_classes)
    others_indices = {class_names.index(c) for c in others_classes if c in class_names}
    eval_indices   = [i for i in range(len(class_names)) if i not in others_indices]
    num_classes    = config.model.num_classes

    # Raw predictions
    video_results_raw = run_inference(model, test_root, device)

    # Metrics
    raw_metrics, raw_vf1, raw_preds_flat, raw_labels_flat = compute_all_metrics(
        video_results_raw, num_classes, class_names, others_classes, prefix="raw")

    all_metrics = {**raw_metrics}

    # Print
    print(f"\n=== Test metrics ===")
    for key in ["global/accuracy", "global/f1_macro", "global/auroc",
                "segment/edit_score", "segment/f1@10", "segment/f1@25", "segment/f1@50"]:
        v = raw_metrics.get(f"raw/{key}", float("nan"))
        print(f"  {key:<33} {v:.4f}")

    print("\nPer-class F1:")
    for c in class_names:
        if c in others_classes:
            continue
        v = raw_metrics.get(f"raw/per_class/f1/{c}", 0.0)
        print(f"  {c:<35} {v:.3f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({k: round(v, 6) for k, v in all_metrics.items()}, f, indent=2)
    print(f"\n  Saved: metrics.json")

    # Plots
    print("\nGenerating plots...")
    plot_phase_timeline(video_results_raw, class_names, out_dir / "phase_timeline.png")
    plot_binary_ch_timeline(video_results_raw, test_root, out_dir / "binary_ch_timeline.png")
    plot_confusion_matrix(raw_preds_flat, raw_labels_flat, class_names, eval_indices,
                          out_dir / "confusion_matrix.png",
                          "Confusion matrix — predictions (test set)")
    plot_per_video_f1(raw_vf1, {}, out_dir / "per_video_f1.png")

    print(f"\nDone. Résultats sauvegardés dans {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate MS-TCN++ temporal model on test set")
    parser.add_argument("--config", type=str,
                        default="phases_recognition/configs/config_mstcn.yaml")
    parser.add_argument("--ckpt",   type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--smooth_window", type=int, default=15,
                        help="Fenêtre majority-vote (0 = désactivé)")
    args = parser.parse_args()
    main(args.config, args.ckpt, args.out_dir, args.smooth_window)

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


def plot_phase_timeline(
    video_results_raw: list[tuple[list[int], list[int], str]],
    video_results_smooth: list[tuple[list[int], list[int], str]],
    class_names: list[str],
    out_path: pathlib.Path,
    smooth_window: int,
):
    n_classes = len(class_names)
    cmap      = matplotlib.colormaps.get_cmap("tab20")
    colors    = [cmap(i / max(n_classes - 1, 1)) for i in range(n_classes)]
    n_videos  = len(video_results_raw)
    has_smooth = smooth_window > 1

    n_bars = 3 if has_smooth else 2
    bar_labels = ["GT", "Raw", f"Smooth (w={smooth_window})"] if has_smooth else ["GT", "Raw"]

    fig, axes = plt.subplots(n_videos, 1,
                             figsize=(14, n_videos * (0.5 * n_bars + 0.4) + 1.5),
                             squeeze=False)

    for row, ((gt_seq, pred_raw, video_name), (_, pred_smooth, _)) in enumerate(
        zip(video_results_raw, video_results_smooth)
    ):
        ax = axes[row, 0]
        n_frames = len(gt_seq)
        seqs = [(gt_seq, "GT"), (pred_raw, "Raw")]
        if has_smooth:
            seqs.append((pred_smooth, bar_labels[2]))
        _draw_bars(ax, seqs, colors, n_frames)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, n_bars - 0.5)
        ax.set_yticks(range(n_bars))
        ax.set_yticklabels(bar_labels, fontsize=7)
        ax.set_xticks([])
        ax.set_title(video_name, loc="left", fontsize=8, pad=2)
        for spine in ["top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(n_classes)]
    fig.legend(handles, class_names,
               loc="lower center", ncol=min(n_classes, 7),
               fontsize=7, bbox_to_anchor=(0.5, 0), frameon=False)
    fig.suptitle("Phase timeline — GT / Raw / Smoothed (test set)", fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
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
    names  = list(video_f1s_raw.keys())
    raw    = [video_f1s_raw[n]    for n in names]
    smooth = [video_f1s_smooth[n] for n in names]
    order  = sorted(range(len(raw)), key=lambda i: raw[i])
    names  = [names[i][:40] for i in order]
    raw    = [raw[i]    for i in order]
    smooth = [smooth[i] for i in order]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.45)))
    ax.barh(y - 0.2, raw,    height=0.35, color="#5b9bd5", label="Raw")
    ax.barh(y + 0.2, smooth, height=0.35, color="#ed7d31", label="Smoothed")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 macro (frame-level)")
    ax.axvline(x=np.mean(raw),    color="#5b9bd5", linestyle="--", linewidth=1)
    ax.axvline(x=np.mean(smooth), color="#ed7d31", linestyle="--", linewidth=1)
    ax.legend(fontsize=8)
    ax.set_title("Per-video F1 macro — test set (raw vs smoothed)")
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
        results.append((labels.tolist(), preds, video_name))
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

    # Smoothed predictions
    video_results_smooth = [
        (gt, majority_vote_smooth(pred, smooth_window), name)
        for gt, pred, name in video_results_raw
    ]

    # Metrics
    raw_metrics, raw_vf1, raw_preds_flat, raw_labels_flat = compute_all_metrics(
        video_results_raw, num_classes, class_names, others_classes, prefix="raw")
    smooth_metrics, smooth_vf1, smooth_preds_flat, _ = compute_all_metrics(
        video_results_smooth, num_classes, class_names, others_classes, prefix="smoothed")

    all_metrics = {**raw_metrics, **smooth_metrics}

    # Print
    print(f"\n=== Test metrics (smooth_window={smooth_window}) ===")
    print(f"{'Metric':<35} {'Raw':>8}  {'Smoothed':>10}")
    print("-" * 58)
    key_pairs = [
        ("global/accuracy", "global/accuracy"),
        ("global/f1_macro", "global/f1_macro"),
        ("global/auroc", "global/auroc"),
        ("segment/edit_score", "segment/edit_score"),
        ("segment/f1@10", "segment/f1@10"),
        ("segment/f1@25", "segment/f1@25"),
        ("segment/f1@50", "segment/f1@50"),
    ]
    for key, _ in key_pairs:
        r = raw_metrics.get(f"raw/{key}", float("nan"))
        s = smooth_metrics.get(f"smoothed/{key}", float("nan"))
        print(f"  {key:<33} {r:>8.4f}  {s:>10.4f}")

    print("\nPer-class F1 (raw → smoothed):")
    for c in class_names:
        if c in others_classes:
            continue
        r = raw_metrics.get(f"raw/per_class/f1/{c}", 0.0)
        s = smooth_metrics.get(f"smoothed/per_class/f1/{c}", 0.0)
        delta = s - r
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "~")
        print(f"  {c:<35} {r:.3f} → {s:.3f}  {arrow}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({k: round(v, 6) for k, v in all_metrics.items()}, f, indent=2)
    print(f"\n  Saved: metrics.json")

    # Plots
    print("\nGenerating plots...")
    plot_phase_timeline(video_results_raw, video_results_smooth,
                        class_names, out_dir / "phase_timeline.png", smooth_window)
    plot_confusion_matrix(raw_preds_flat, raw_labels_flat, class_names, eval_indices,
                          out_dir / "confusion_matrix_raw.png",
                          "Confusion matrix — raw predictions (test set)")
    plot_confusion_matrix(smooth_preds_flat, raw_labels_flat, class_names, eval_indices,
                          out_dir / "confusion_matrix_smoothed.png",
                          f"Confusion matrix — smoothed w={smooth_window} (test set)")
    plot_per_video_f1(raw_vf1, smooth_vf1, out_dir / "per_video_f1.png")

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

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

    for gt_seq, pred_seq, *_rest, video_name in video_results:
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
    "#4E79A7",  # bleu acier
    "#F28E2B",  # orange
    "#E15759",  # rouge
    "#76B7B2",  # teal
    "#59A14F",  # vert
    "#EDC948",  # jaune doré
    "#B07AA1",  # violet
    "#FF9DA7",  # rose saumon
    "#9C755F",  # marron
    "#BAB0AC",  # gris chaud
    "#499894",  # teal foncé
    "#D37295",  # rose foncé
    "#2D2D2D",  # gris foncé — OOD / phase inconnue
]

OOD_LABEL = 12  # sentinel index used in GT for unseen phases (label -1 in features)


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def _segment_ood_signal(ood_signal: np.ndarray, pred_seq: list[int]) -> np.ndarray:
    """Replace each frame's OOD score with the median over its predicted segment.
    This gives a step-function view: whole segments appear as OOD or not, avoiding
    single-frame noise, and directly answers 'is this PHASE anomalous?'
    """
    n = len(pred_seq)
    out = np.zeros(n, dtype=np.float32)
    segments = _get_segments(pred_seq)
    for _, start, end in segments:
        out[start:end] = np.median(ood_signal[start:end])
    return out


def plot_phase_timeline(
    video_results_raw: list,
    class_names: list[str],
    out_path: pathlib.Path,
    ood_threshold: float | None = None,
    smooth_window: int = 100,
    error_recalls: dict[str, float] | None = None,
    error_fprs: dict[str, float] | None = None,
):
    """GT bar + OOD signal per video. The signal is z-scored per video so a
    peak at +1σ means 'this segment is more uncertain than this video's baseline'.
    Dark gray in the GT bar = ground-truth unknown phase.
    """
    n_classes = len(class_names)
    colors = PHASE_COLORS[:n_classes] + ["#000000"] * (OOD_LABEL - n_classes) + [PHASE_COLORS[OOD_LABEL]]
    n_videos  = len(video_results_raw)

    height_ratios = [3, 1] * n_videos
    fig, axes = plt.subplots(
        n_videos * 2, 1,
        figsize=(14, n_videos * 1.8 + 1.5),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
    )

    for row, (gt_seq, pred_seq, conf_seq, ood_seq, video_name) in enumerate(video_results_raw):
        ax_row   = row * 2
        n_frames = len(gt_seq)
        t        = np.linspace(0, 1, n_frames)

        # ── GT + Pred bars ───────────────────────────────────────────────────
        ax_gt = axes[ax_row, 0]
        _draw_bars(ax_gt, [(gt_seq, "GT"), (pred_seq, "Pred")], colors, n_frames)
        ax_gt.set_xlim(0, 1)
        ax_gt.set_ylim(-0.5, 1.5)
        ax_gt.set_yticks([0, 1])
        ax_gt.set_yticklabels(["Pred", "GT"], fontsize=7)
        ax_gt.set_xticks([])
        title = video_name
        if error_recalls is not None and video_name in error_recalls:
            title += f"   —   erreurs détectées: {error_recalls[video_name] * 100:.0f}%"
            if error_fprs is not None and video_name in error_fprs:
                title += f"   |   FPR: {error_fprs[video_name] * 100:.0f}%"
        ax_gt.set_title(title, loc="left", fontsize=8, pad=2)
        for spine in ["top", "right", "bottom"]:
            ax_gt.spines[spine].set_visible(False)

        # ── OOD signal ──────────────────────────────────────────────────────
        ax_ood  = axes[ax_row + 1, 0]
        raw_sig = np.array(ood_seq, dtype=np.float32)

        # smooth curve — wide window for phase-level trends
        smooth_sig = _smooth(raw_sig, smooth_window)

        # GT-aligned shading: highlight regions where GT is OOD
        gt_ood_mask = np.array([g == OOD_LABEL for g in gt_seq])
        ax_ood.fill_between(t, ax_ood.get_ylim()[0] if False else -10, 10,
                            where=gt_ood_mask,
                            alpha=0.12, color="#2D2D2D", zorder=0,
                            label="GT OOD region")

        ax_ood.fill_between(t, 0, smooth_sig, alpha=0.2, color="#E15759", zorder=1)
        ax_ood.plot(t, smooth_sig, color="#E15759", linewidth=0.9, zorder=2)

        if ood_threshold is not None:
            ax_ood.axhline(ood_threshold, color="black", linestyle="--", linewidth=0.8)
            ax_ood.fill_between(t, ood_threshold, smooth_sig,
                                where=(smooth_sig > ood_threshold),
                                alpha=0.5, color="#E15759", zorder=3)

        ax_ood.set_xlim(0, 1)
        ax_ood.set_xticks([])
        ax_ood.set_ylabel("Incert.\n(↑=tort)", fontsize=6, rotation=0, labelpad=38, va="center")
        ax_ood.tick_params(labelsize=5)
        ax_ood.axhline(0, color="#AAAAAA", linewidth=0.5, linestyle=":")
        for spine in ["top", "right"]:
            ax_ood.spines[spine].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(n_classes)]
    handles += [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[OOD_LABEL])]
    labels_legend = class_names + ["Phase inconnue (GT)"]
    fig.legend(handles, labels_legend,
               loc="lower center", ncol=min(n_classes + 1, 7),
               fontsize=7, bbox_to_anchor=(0.5, 0), frameon=False)
    fig.suptitle("GT / Pred + signal d'incertitude (entropie × KL inter-stages) — test set", fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
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

    for row, (gt_phase_seq, *_rest, video_name) in enumerate(video_results_raw):
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

def _mahal_ood_signal(features: np.ndarray, stats: dict) -> np.ndarray:
    """Compute per-frame min-class Mahalanobis² OOD score.
    Works for any feature dimension; uses an identity expansion to avoid
    materialising the full (T, D) @ (D, D) @ (D, T) product per class.

    Returns (T,) — higher = further from all class centroids = more OOD.
    """
    class_means = stats["class_means"].astype(np.float64)  # (C, D)
    precision   = stats["precision"].astype(np.float64)    # (D, D)
    features    = features.astype(np.float64)              # (T, D)
    T, D = features.shape
    C    = len(class_means)

    # Precompute x @ P once — O(T × D²) but a single BLAS gemm, fast
    xP   = features @ precision          # (T, D)
    xPx  = (xP * features).sum(axis=1)  # (T,)  — quadratic term

    min_dist2 = np.full(T, np.inf, dtype=np.float64)
    for c in range(C):
        mu = class_means[c]                          # (D,)
        muPmu = float(mu @ precision @ mu)           # scalar
        xPmu  = xP @ mu                              # (T,)
        dist2 = xPx - 2.0 * xPmu + muPmu            # (T,)
        np.minimum(min_dist2, dist2, out=min_dist2)
    return min_dist2.astype(np.float32)  # high = OOD


def _entropy_ood_signal(logits: torch.Tensor, logit_norm_tau: float = 0.04) -> np.ndarray:
    """LogitNorm OOD signal: entropy of softmax(logits / (||logits|| * tau)).
    This reproduces the exact same operation as the training loss, so the
    model's learned confidence is correctly expressed.
    - In-dist: logit vector strongly aligned with one class → low entropy
    - OOD: logit vector diffuse → high entropy
    Returns (T,) numpy array, high = OOD.
    """
    norm  = logits.norm(p=2, dim=1, keepdim=True).clamp(min=1e-7)
    scaled = logits / (norm * logit_norm_tau)                 # same as training
    probs  = torch.softmax(scaled, dim=1)                     # (T, C)
    entropy = -(probs * (probs + 1e-9).log()).sum(dim=1)      # (T,)
    return entropy.cpu().numpy().astype(np.float32)


@torch.no_grad()
def run_inference(model, test_root, device, use_logit_norm: bool = False,
                  logit_norm_tau: float = 0.04,
                  resnet_mahal_stats=None, mstcn_mahal_stats=None):
    dataset = VideoFeatureDataset(root=str(test_root))
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         collate_fn=_collate_single_video)
    model.eval()
    results = []
    for features, labels, video_name in tqdm.tqdm(loader, desc="Inference"):
        resnet_np    = features.numpy()                        # (T, 2048)
        features_gpu = features.unsqueeze(0).to(device)
        stage_logits = model(features_gpu)

        # Probs from last stage
        last_logits  = stage_logits[-1].squeeze(0).T          # (T, C)
        probs        = torch.softmax(last_logits, dim=1).cpu()
        confidence   = probs.max(dim=1).values.tolist()
        preds        = probs.argmax(dim=1).tolist()

        # --- Signal 1: entropie frame-level (last stage) ---
        # Capture l'incertitude sur toutes les classes, pas juste la max
        p_last  = probs                                        # (T, C)
        entropy = -(p_last * (p_last + 1e-9).log()).sum(dim=1).numpy()  # (T,)

        # --- Signal 2: désaccord inter-stages (stage 1 vs stage 4) ---
        # KL(p1 || p4) par frame — élevé si le raffinement temporel
        # contredit fortement la prédiction initiale → instabilité temporelle
        p_first = torch.softmax(stage_logits[0].squeeze(0).T, dim=1).cpu()  # (T, C)
        kl_div  = (p_first * ((p_first + 1e-9) / (p_last + 1e-9)).log()).sum(dim=1).numpy()  # (T,)
        kl_div  = np.abs(kl_div)  # symétrique

        # --- Signal combiné: produit normalisé ---
        # Haut quand les DEUX signaux sont élevés simultanément
        entropy_norm = (entropy - entropy.min()) / (entropy.std() + 1e-7)
        kl_norm      = (kl_div  - kl_div.min())  / (kl_div.std()  + 1e-7)
        raw_signal   = (entropy_norm * kl_norm).astype(np.float32)

        ood_signal = raw_signal.tolist()

        gt = labels.tolist()
        gt_full    = [OOD_LABEL if l == -1 else l for l in gt]
        mask       = [i for i, l in enumerate(gt) if l != -1]
        gt_clean   = [gt[i]         for i in mask]
        pred_clean = [preds[i]      for i in mask]
        conf_clean = [confidence[i] for i in mask]
        results.append((gt_full, preds, confidence, ood_signal, gt_clean, pred_clean, conf_clean, video_name))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str, ckpt_path: str, out_dir: str, smooth_window: int, split: str = "test"):
    config  = OmegaConf.load(config_path)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(config.train.device)

    val_root   = pathlib.Path(config.dataset.val.params.root)
    split_root = val_root.parent / split
    if not split_root.exists():
        raise FileNotFoundError(f"Features not found at {split_root}.")

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

    # Load ResNet 2048D Mahalanobis stats (primary OOD signal — better separation)
    resnet_mahal_stats = None
    resnet_stats_path = split_root.parent / "mahal_stats.npz"
    if resnet_stats_path.exists():
        _r = np.load(resnet_stats_path, allow_pickle=True)
        resnet_mahal_stats = {
            "class_means": _r["class_means"].astype(np.float32),
            "precision":   _r["precision"].astype(np.float32),
        }
        print(f"  ResNet 2048D Mahalanobis stats loaded — {len(resnet_mahal_stats['class_means'])} classes")
    else:
        print(f"  No mahal_stats.npz at {resnet_stats_path} — will try MSTCN 32D fallback")

    # Load MSTCN 32D Mahalanobis stats (fallback)
    mstcn_mahal_stats = None
    mstcn_stats_path = split_root.parent / "mstcn_mahal_stats.npz"
    if mstcn_stats_path.exists():
        _s = np.load(mstcn_stats_path, allow_pickle=True)
        mstcn_mahal_stats = {
            "class_means": _s["class_means"].astype(np.float32),
            "precision":   _s["precision"].astype(np.float32),
            "threshold":   float(_s["threshold"]),
        }
        print(f"  MSTCN 32D Mahalanobis stats loaded — threshold: {mstcn_mahal_stats['threshold']:.4f}")

    # Use LogitNorm entropy if the config has logit_norm_tau > 0
    logit_norm_tau = float(config.loss.get("logit_norm_tau", 0.0))
    use_logit_norm = logit_norm_tau > 0
    if use_logit_norm:
        print(f"  LogitNorm model detected (tau={logit_norm_tau}) — using entropy OOD signal")
    else:
        print("  Standard CE model — using Mahalanobis OOD signal")

    # Raw predictions
    video_results_raw = run_inference(model, split_root, device,
                                      use_logit_norm=use_logit_norm,
                                      logit_norm_tau=logit_norm_tau,
                                      resnet_mahal_stats=resnet_mahal_stats,
                                      mstcn_mahal_stats=mstcn_mahal_stats)

    # Split into viz sequences (full, with OOD frames) and metric sequences (clean)
    video_results_viz     = [(gt_full, preds_full, conf_full, entropy, vname)
                             for gt_full, preds_full, conf_full, entropy, _, _, _, vname in video_results_raw]
    video_results_metrics = [(gt_clean, pred_clean, conf_clean, vname)
                             for _, _, _, _, gt_clean, pred_clean, conf_clean, vname in video_results_raw]

    # Split metrics by whether the video contains OOD frames
    videos_with_ood    = {vname for gt_full, _, _, _, _, _, _, vname in video_results_raw
                          if OOD_LABEL in gt_full}
    metrics_no_ood  = [(gt, pred, conf, vname) for gt, pred, conf, vname in video_results_metrics
                       if vname not in videos_with_ood]
    metrics_with_ood = [(gt, pred, conf, vname) for gt, pred, conf, vname in video_results_metrics
                        if vname in videos_with_ood]
    print(f"\n  Videos without OOD frames : {len(metrics_no_ood)}")
    print(f"  Videos with OOD frames    : {len(metrics_with_ood)}")

    # Threshold: 95th percentile of in-dist frames on test set
    id_signals = []
    for gt_full, _, _, ood_signal, _, _, _, _ in video_results_raw:
        id_signals.extend(s for s, g in zip(ood_signal, gt_full) if g != OOD_LABEL)
    ood_entropy_threshold = float(np.percentile(id_signals, 85)) if id_signals else None
    print(f"  Uncertainty threshold (85th pct): {ood_entropy_threshold:.4f}")

    # Threshold sweep: for each percentile, global recall (errors caught) vs
    # FPR (correct frames wrongly flagged) — helps pick the recall/precision compromise.
    print("\n=== Threshold sweep — global error-detection recall vs FPR ===")
    all_signals, all_is_error, all_is_known = [], [], []
    for gt_full, preds_full, _, ood_signal, _, _, _, _ in video_results_raw:
        gt_arr   = np.asarray(gt_full)
        pred_arr = np.asarray(preds_full)
        sig_arr  = np.asarray(ood_signal)
        known    = gt_arr != OOD_LABEL
        all_signals.append(sig_arr)
        all_is_error.append(known & (pred_arr != gt_arr))
        all_is_known.append(known)
    all_signals  = np.concatenate(all_signals)
    all_is_error = np.concatenate(all_is_error)
    all_is_known = np.concatenate(all_is_known)
    all_correct  = all_is_known & ~all_is_error

    for pct in [80, 85, 90, 95, 99]:
        thr = float(np.percentile(id_signals, pct))
        recall = float((all_signals[all_is_error] > thr).mean()) if all_is_error.any() else float("nan")
        fpr    = float((all_signals[all_correct] > thr).mean()) if all_correct.any() else float("nan")
        print(f"  pct={pct:3d}  thr={thr:7.4f}   recall(errors caught)={recall*100:5.1f}%   "
              f"FPR(correct flagged)={fpr*100:5.1f}%")

    # Error-detection recall: among frames where the model is wrong (on known phases),
    # what fraction has an uncertainty signal above the threshold?
    print("\n=== Error detection — recall & FPR @ threshold (per video) ===")
    error_recalls_by_video = {}
    error_fprs_by_video = {}
    for gt_full, preds_full, _, ood_signal, _, _, _, vname in video_results_raw:
        gt_arr   = np.asarray(gt_full)
        pred_arr = np.asarray(preds_full)
        sig_arr  = np.asarray(ood_signal)
        known          = gt_arr != OOD_LABEL
        is_known_error = known & (pred_arr != gt_arr)
        is_correct     = known & (pred_arr == gt_arr)
        n_errors = int(is_known_error.sum())
        fpr = float((sig_arr[is_correct] > ood_entropy_threshold).mean()) if is_correct.any() else float("nan")
        error_fprs_by_video[vname] = fpr
        if n_errors == 0:
            print(f"  {vname:<35} no errors   (FPR={fpr*100:5.1f}%)")
            continue
        recall = float((sig_arr[is_known_error] > ood_entropy_threshold).mean())
        error_recalls_by_video[vname] = recall
        print(f"  {vname:<35} recall={recall*100:5.1f}%  FPR={fpr*100:5.1f}%  ({n_errors} error frames)")

    error_recalls = list(error_recalls_by_video.values())
    error_fprs = list(error_fprs_by_video.values())
    mean_error_recall = float(np.mean(error_recalls)) if error_recalls else float("nan")
    mean_error_fpr = float(np.mean(error_fprs)) if error_fprs else float("nan")
    print(f"\n  Mean error-detection recall @ threshold: {mean_error_recall*100:.1f}%")
    print(f"  Mean FPR (correct frames flagged) @ threshold: {mean_error_fpr*100:.1f}%")
    all_metrics_extra = {
        "error_detection_recall_mean": mean_error_recall,
        "error_detection_fpr_mean": mean_error_fpr,
    }

    # Metrics — all videos + split by OOD presence
    raw_metrics,     raw_vf1,    raw_preds_flat,    raw_labels_flat    = compute_all_metrics(
        video_results_metrics, num_classes, class_names, others_classes, prefix="raw")
    metrics_no_ood_d,  _, _, _ = compute_all_metrics(
        metrics_no_ood,  num_classes, class_names, others_classes, prefix="no_ood") \
        if metrics_no_ood  else ({}, {}, [], [])
    metrics_with_ood_d, _, _, _ = compute_all_metrics(
        metrics_with_ood, num_classes, class_names, others_classes, prefix="with_ood") \
        if metrics_with_ood else ({}, {}, [], [])

    all_metrics = {**raw_metrics, **metrics_no_ood_d, **metrics_with_ood_d, **all_metrics_extra}

    def _print_metrics(metrics, prefix, title):
        print(f"\n=== {title} ===")
        for key in ["global/accuracy", "global/f1_macro", "global/auroc",
                    "segment/edit_score", "segment/f1@10", "segment/f1@25", "segment/f1@50"]:
            v = metrics.get(f"{prefix}/{key}", float("nan"))
            print(f"  {key:<33} {v:.4f}")

    _print_metrics(raw_metrics,      "raw",      "Test metrics — all videos")
    _print_metrics(metrics_no_ood_d,  "no_ood",  "Test metrics — videos WITHOUT unknown phases")
    _print_metrics(metrics_with_ood_d,"with_ood","Test metrics — videos WITH unknown phases")

    print("\nPer-class F1 (all videos):")
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
    plot_phase_timeline(video_results_viz, class_names, out_dir / "phase_timeline.png",
                        ood_threshold=ood_entropy_threshold,
                        error_recalls=error_recalls_by_video,
                        error_fprs=error_fprs_by_video)
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
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    args = parser.parse_args()
    main(args.config, args.ckpt, args.out_dir, args.smooth_window, args.split)

"""Per-video timeline: GT / predictions / OOD scores (Baseline vs TCL β=0.02).

For each test video, plots 4 subplots:
  1. Ground truth (colored bars by phase)
  2. Closed-set prediction — TCL β=0.02 (same palette)
  3. OOD score — Baseline KNN (curve + threshold)
  4. OOD score — TCL β=0.02 KNN (curve + threshold)
Unknown GT segments are highlighted in transparent red on all subplots.

Usage:
    python phases_recognition/eval_tcl_timeline.py
"""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset.feature_dataset import VideoFeatureDataset, _collate_single_video
from models import instantiate_model

# ── Config ─────────────────────────────────────────────────────────────────
FEAT_ROOT = pathlib.Path("/home/helena/UCL_video_cataract/features_dino/")
EXP_BASELINE = "/home/helena/experiments_cataract/baseline_detection_phases_unknown_mstcn_dino_v1_date=2026_06_11_17_02_41"
EXP_TCL      = "/home/helena/experiments_cataract/mstcn_dino_tcl_beta0.02_date=2026_06_29_16_32_06"
OUT_DIR      = pathlib.Path("/home/helena/experiments_cataract/tcl_timeline_eval/")
KNN_K = 10

CLASS_NAMES = [
    "Capsule_polishing", "Hydrodissection", "Incision",
    "Irrigation_and_aspiration", "Lens_implant_settingup",
    "Phacoemulsification", "Rhexis", "Tonifying_and_antibiotics",
    "Viscous_agent_injection", "Viscous_agent_removal", "Wound_hydration",
]
UNKNOWN_PHASES = [
    "Malyugin_ring_insertion", "Malyugin_ring_removal",
    "Suture", "Iris_manipulation", "Trypan_blue_injection",
]

# One distinct color per known phase
PHASE_COLORS = [
    "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2",
    "#17becf", "#bcbd22", "#d4a017", "#f07800", "#3a7d44", "#6b5b95",
]
UNKNOWN_COLOR = "#888888"


# ── Model helpers ─────────────────────────────────────────────────────────

def load_model(exp_dir, device):
    cfg = OmegaConf.load(f"{exp_dir}/config.yaml")
    model = instantiate_model(cfg.model)
    state = torch.load(f"{exp_dir}/ckpt/best.pt", map_location="cpu",
                       weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def extract_per_video(model, split, device):
    """Returns list of (name, features_64dim, logit_preds, labels, phase_names)."""
    ds = VideoFeatureDataset(root=str(FEAT_ROOT / split))
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=_collate_single_video)
    results = []
    for features, labels, name in loader:
        inp = features.unsqueeze(0).to(device)
        logits_list, feats = model.forward_with_features(inp)
        feats_np = feats.squeeze(0).T.cpu().numpy().astype(np.float64)
        preds = logits_list[-1].squeeze(0).T.argmax(dim=1).cpu().numpy()
        ph_path = FEAT_ROOT / split / f"{name}_phases.npy"
        phase_names = (np.load(ph_path, allow_pickle=True) if ph_path.exists()
                       else np.full(len(labels), "", dtype=object))
        results.append((name, feats_np, preds, labels.numpy(), phase_names))
    return results


def fit_knn(videos):
    """Fit KNN on known train frames."""
    feats = np.concatenate([f for _, f, _, l, _ in videos])
    labels = np.concatenate([l for _, _, _, l, _ in videos])
    known = labels >= 0
    nn = NearestNeighbors(n_neighbors=KNN_K, metric="euclidean").fit(feats[known])
    return nn


def get_threshold(nn_model, val_videos, percentile=95):
    feats = np.concatenate([f for _, f, _, l, _ in val_videos])
    labels = np.concatenate([l for _, _, _, l, _ in val_videos])
    known = labels >= 0
    scores = nn_model.kneighbors(feats[known])[0][:, -1]
    return float(np.percentile(scores, percentile))


# ── Timeline plot ─────────────────────────────────────────────────────────

def label_to_color(label, phase_name):
    if label >= 0:
        return PHASE_COLORS[label % len(PHASE_COLORS)]
    return UNKNOWN_COLOR


def plot_phase_bar(ax, labels, phase_names, title):
    """Horizontal stacked bar showing phase sequence."""
    T = len(labels)
    x = 0
    while x < T:
        lbl = labels[x]
        ph = phase_names[x] if x < len(phase_names) else ""
        # Find end of this segment
        xend = x + 1
        while xend < T and labels[xend] == lbl:
            xend += 1
        color = PHASE_COLORS[lbl] if lbl >= 0 else UNKNOWN_COLOR
        ax.barh(0, xend - x, left=x, height=0.8, color=color, linewidth=0)
        x = xend
    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_ylabel(title, fontsize=9, rotation=0, ha="right", va="center",
                  labelpad=5)
    ax.tick_params(axis="x", labelsize=8)


def shade_unknown_regions(ax, labels, phase_names, ymin, ymax):
    """Red transparent overlay on all unknown segments."""
    T = len(labels)
    x = 0
    while x < T:
        if labels[x] == -1:
            xend = x + 1
            while xend < T and labels[xend] == -1:
                xend += 1
            ax.axvspan(x, xend, ymin=ymin, ymax=ymax,
                       color="red", alpha=0.15, linewidth=0)
        x += 1 if labels[x] != -1 else (xend - x)


def plot_score_curve(ax, scores, threshold, title, color):
    T = len(scores)
    ax.plot(range(T), scores, color=color, lw=0.8, alpha=0.85)
    ax.axhline(threshold, color="black", ls="--", lw=1.0, label=f"p95={threshold:.2f}")
    ax.fill_between(range(T), scores, threshold,
                    where=scores > threshold, color=color, alpha=0.25)
    ax.set_xlim(0, T)
    ax.set_ylabel(title, fontsize=9, rotation=0, ha="right", va="center",
                  labelpad=5)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, loc="upper right")


def make_legend(fig):
    handles = [mpatches.Patch(color=PHASE_COLORS[i], label=CLASS_NAMES[i])
               for i in range(len(CLASS_NAMES))]
    handles.append(mpatches.Patch(color=UNKNOWN_COLOR, label="Unknown"))
    handles.append(mpatches.Patch(color="red", alpha=0.3, label="GT unknown region"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.04))


def plot_video(video_name, gt_labels, gt_phases, tcl_preds,
               scores_base, thr_base, scores_tcl, thr_tcl, out_dir):
    fig, axes = plt.subplots(4, 1, figsize=(18, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 2, 2],
                                          "hspace": 0.35})

    T = len(gt_labels)

    # ── Subplot 1: Ground truth ──────────────────────────────────────────
    plot_phase_bar(axes[0], gt_labels, gt_phases, "GT")
    shade_unknown_regions(axes[0], gt_labels, gt_phases, 0, 1)

    # ── Subplot 2: TCL β=0.02 closed-set predictions ─────────────────────
    plot_phase_bar(axes[1], tcl_preds, gt_phases, "Pred\nTCL")
    shade_unknown_regions(axes[1], gt_labels, gt_phases, 0, 1)

    # ── Subplot 3: Baseline OOD score ────────────────────────────────────
    plot_score_curve(axes[2], scores_base, thr_base, "OOD\nBaseline", "#1f77b4")
    shade_unknown_regions(axes[2], gt_labels, gt_phases, 0, 1)

    # ── Subplot 4: TCL β=0.02 OOD score ──────────────────────────────────
    plot_score_curve(axes[3], scores_tcl, thr_tcl, "OOD\nTCL β=0.02", "#d62728")
    shade_unknown_regions(axes[3], gt_labels, gt_phases, 0, 1)

    axes[3].set_xlabel("Frame", fontsize=9)

    # Count unknown phases in this video
    unk_phases_in_video = sorted(set(gt_phases[gt_labels == -1]) - {""})
    unk_str = ", ".join(unk_phases_in_video) if unk_phases_in_video else "none"
    fig.suptitle(f"{video_name}\nUnknown phases: {unk_str}", fontsize=11, y=1.01)

    make_legend(fig)
    fig.tight_layout()

    safe_name = video_name.replace("/", "_").replace(" ", "_")[:60]
    path = out_dir / f"timeline_{safe_name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    model_base = load_model(EXP_BASELINE, device)
    model_tcl  = load_model(EXP_TCL, device)

    print("Extracting features (train / val / test) for both models...")
    train_base = extract_per_video(model_base, "train", device)
    val_base   = extract_per_video(model_base, "val", device)
    test_base  = extract_per_video(model_base, "test", device)

    train_tcl  = extract_per_video(model_tcl, "train", device)
    val_tcl    = extract_per_video(model_tcl, "val", device)
    test_tcl   = extract_per_video(model_tcl, "test", device)

    del model_base, model_tcl
    torch.cuda.empty_cache()

    print("Fitting KNN...")
    nn_base = fit_knn(train_base)
    nn_tcl  = fit_knn(train_tcl)

    print("Calibrating thresholds (p95 val)...")
    thr_base = get_threshold(nn_base, val_base)
    thr_tcl  = get_threshold(nn_tcl, val_tcl)
    print(f"  Baseline threshold: {thr_base:.4f}")
    print(f"  TCL β=0.02 threshold: {thr_tcl:.4f}")

    print(f"\nGenerating timelines for {len(test_base)} test videos...")

    # Build index: test_tcl by video name for alignment
    tcl_by_name = {name: (feats, preds, labels, phases)
                   for name, feats, preds, labels, phases in test_tcl}

    for name, feats_b, preds_b, labels, phases in test_base:
        n_unk = (labels == -1).sum()
        unk_set = set(phases[labels == -1]) - {""}
        print(f"\n  {name}  ({n_unk} unknown frames: {', '.join(unk_set) or 'none'})")

        scores_b = nn_base.kneighbors(feats_b)[0][:, -1]

        feats_t, preds_t, _, _ = tcl_by_name[name]
        scores_t = nn_tcl.kneighbors(feats_t)[0][:, -1]

        plot_video(name, labels, phases, preds_t,
                   scores_b, thr_base, scores_t, thr_tcl, OUT_DIR)

    print(f"\nAll timelines saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

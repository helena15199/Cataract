"""Per-video timeline: GT / predictions / OOD scores (Baseline vs TCL β=0.02 vs TCL+Mixup).

For each test video, plots 6 subplots:
  1. Ground truth (colored bars by phase)
  2. OOD-masked prediction — TCL β=0.02
  3. OOD-masked prediction — TCL β=0.02+Mixup
  4. OOD score — Baseline KNN (curve + threshold)
  5. OOD score — TCL β=0.02 KNN (curve + threshold)
  6. OOD score — TCL β=0.02+Mixup KNN (curve + threshold)
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
from collections import defaultdict
import torch
from omegaconf import OmegaConf
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset.feature_dataset import VideoFeatureDataset, _collate_single_video
from models import instantiate_model

# ── Config ─────────────────────────────────────────────────────────────────
FEAT_ROOT    = pathlib.Path("/home/helena/UCL_video_cataract/features_dino/")
EXP_BASELINE = "/home/helena/experiments_cataract/baseline_detection_phases_unknown_mstcn_dino_v1_date=2026_06_11_17_02_41"
EXP_TCL      = "/home/helena/experiments_cataract/mstcn_dino_tcl_beta0.02_date=2026_06_29_16_32_06"
EXP_MIXUP    = "/home/helena/experiments_cataract/mstcn_dino_tcl_beta0.02_mixup_date=2026_07_22_16_32_18"
OUT_DIR      = pathlib.Path("/home/helena/experiments_cataract/tcl_timeline_eval/")
KNN_K = 10
MIN_DURATION = 30   # min consecutive frames above threshold to count as a predicted segment

# Videos with no unknown GT annotation but known to be clinically complex
COMPLICATED_VIDEOS = {"Video 28 (cat127)"}  # phaco compliqué, anomalies non étiquetées

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


def plot_video(video_name, gt_labels, gt_phases,
               tcl_preds, mixup_preds,
               scores_base, thr_base,
               scores_tcl, thr_tcl,
               scores_mixup, thr_mixup,
               out_dir):
    fig, axes = plt.subplots(6, 1, figsize=(18, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 2, 2, 2],
                                          "hspace": 0.35})

    T = len(gt_labels)

    # ── Subplot 1: Ground truth ──────────────────────────────────────────
    plot_phase_bar(axes[0], gt_labels, gt_phases, "GT")
    shade_unknown_regions(axes[0], gt_labels, gt_phases, 0, 1)

    # ── Subplot 2: TCL β=0.02 OOD-masked predictions ─────────────────────
    eff_preds_tcl = np.where(scores_tcl > thr_tcl, -1, tcl_preds)
    plot_phase_bar(axes[1], eff_preds_tcl, gt_phases, "Pred\nTCL")
    shade_unknown_regions(axes[1], gt_labels, gt_phases, 0, 1)

    # ── Subplot 3: TCL β=0.02+Mixup OOD-masked predictions ───────────────
    eff_preds_mixup = np.where(scores_mixup > thr_mixup, -1, mixup_preds)
    plot_phase_bar(axes[2], eff_preds_mixup, gt_phases, "Pred\nTCL+Mixup")
    shade_unknown_regions(axes[2], gt_labels, gt_phases, 0, 1)

    # ── Subplot 4: Baseline OOD score ────────────────────────────────────
    plot_score_curve(axes[3], scores_base, thr_base, "OOD\nBaseline", "#1f77b4")
    shade_unknown_regions(axes[3], gt_labels, gt_phases, 0, 1)

    # ── Subplot 5: TCL β=0.02 OOD score ──────────────────────────────────
    plot_score_curve(axes[4], scores_tcl, thr_tcl, "OOD\nTCL β=0.02", "#d62728")
    shade_unknown_regions(axes[4], gt_labels, gt_phases, 0, 1)

    # ── Subplot 6: TCL β=0.02+Mixup OOD score ────────────────────────────
    plot_score_curve(axes[5], scores_mixup, thr_mixup, "OOD\nTCL+Mixup", "#2ca02c")
    shade_unknown_regions(axes[5], gt_labels, gt_phases, 0, 1)

    axes[5].set_xlabel("Frame", fontsize=9)

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


# ── Segment-level metrics ─────────────────────────────────────────────────

def get_predicted_segments(scores, threshold, min_duration=MIN_DURATION):
    """Consecutive frames above threshold → list of (start, end) segments."""
    binary = scores > threshold
    segments = []
    in_seg, start = False, 0
    for t, flag in enumerate(binary):
        if flag and not in_seg:
            start, in_seg = t, True
        elif not flag and in_seg:
            if t - start >= min_duration:
                segments.append((start, t))
            in_seg = False
    if in_seg and len(scores) - start >= min_duration:
        segments.append((start, len(scores)))
    return segments


def get_gt_segments(labels, phase_names):
    """Extract GT unknown segments as list of (start, end, phase_name)."""
    segments = []
    T = len(labels)
    t = 0
    while t < T:
        if labels[t] == -1:
            start = t
            ph = phase_names[t] if t < len(phase_names) else ""
            while t < T and labels[t] == -1:
                t += 1
            segments.append((start, t, ph))
        else:
            t += 1
    return segments


def seg_iou(pred, gt):
    inter = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return inter / union if union > 0 else 0.0


def f1_at_k(pred_segs, gt_segs, k):
    """Segment-level F1 at IoU threshold k."""
    tp, matched = 0, set()
    for p in pred_segs:
        for i, g in enumerate(gt_segs):
            if i not in matched and seg_iou(p, g) >= k:
                tp += 1
                matched.add(i)
                break
    prec = tp / len(pred_segs) if pred_segs else 0.0
    rec  = tp / len(gt_segs)   if gt_segs   else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return f1, prec, rec


def compute_segment_metrics(all_video_results, threshold, model_label):
    """Aggregate segment-level metrics across all test videos.

    all_video_results: list of (scores, gt_labels, gt_phases)
    Returns per-phase dict with F1@{10,25,50} and event recall/precision.
    """
    IOU_THRESHOLDS = [0.10, 0.25, 0.50]

    # Collect per-phase pred/gt segments
    per_phase_pred = defaultdict(list)   # phase → list of pred segments (pooled)
    per_phase_gt   = defaultdict(list)   # phase → list of gt segments

    # Event-level: per gt segment, was it hit by ≥1 pred segment (any IoU > 0)?
    per_phase_event_tp  = defaultdict(int)
    per_phase_event_tot = defaultdict(int)
    per_phase_fp        = defaultdict(int)  # pred segs with no GT match

    for scores, gt_labels, gt_phases in all_video_results:
        pred_segs = get_predicted_segments(scores, threshold)
        gt_segs   = get_gt_segments(gt_labels, gt_phases)

        # Split GT by phase
        for (gs, ge, gph) in gt_segs:
            per_phase_gt[gph].append((gs, ge))
            # Event detection: is there any pred segment overlapping this GT?
            hit = any(seg_iou(p, (gs, ge)) > 0 for p in pred_segs)
            per_phase_event_tp[gph]  += int(hit)
            per_phase_event_tot[gph] += 1

        # False positives: pred segs that don't overlap any GT unknown segment
        all_gt = [(gs, ge) for gs, ge, _ in gt_segs]
        for p in pred_segs:
            if not any(seg_iou(p, g) > 0 for g in all_gt):
                per_phase_fp["__all__"] += 1
            # Assign pred seg to the GT phase with max IoU (for per-phase F1)
            best_iou, best_ph = 0, None
            for (gs, ge, gph) in gt_segs:
                iou_val = seg_iou(p, (gs, ge))
                if iou_val > best_iou:
                    best_iou, best_ph = iou_val, gph
            if best_ph:
                per_phase_pred[best_ph].append(p)
            else:
                per_phase_pred["__fp__"].append(p)

    # Compute and return metrics dict
    results = {"event_tp": per_phase_event_tp, "event_tot": per_phase_event_tot,
               "fp_total": per_phase_fp.get("__all__", 0)}
    for k in IOU_THRESHOLDS:
        results[f"f1@{k}"] = {}
        for phase in UNKNOWN_PHASES:
            gt_s   = per_phase_gt.get(phase, [])
            pred_s = per_phase_pred.get(phase, [])
            if not gt_s:
                continue
            f1, prec, rec = f1_at_k(pred_s, gt_s, k)
            results[f"f1@{k}"][phase] = {"f1": f1, "prec": prec, "rec": rec,
                                          "n_gt": len(gt_s), "n_pred": len(pred_s)}
    return results


def plot_metrics_tables(base_res, tcl_res, out_dir, suffix="", label_b="TCL β=0.02"):
    """Save segment metrics as PNG tables (event recall + F1@IoU)."""
    IOU_THRESHOLDS = [0.10, 0.25, 0.50]
    phases = UNKNOWN_PHASES

    # ── Table 1: Event recall ────────────────────────────────────────────
    header = ["Phase", "GT segs",
              "Baseline\nDetected", "Baseline\nRecall",
              f"{label_b}\nDetected", f"{label_b}\nRecall"]
    rows = []
    for ph in phases:
        tot  = base_res["event_tot"].get(ph, 0)
        b_tp = base_res["event_tp"].get(ph, 0)
        t_tp = tcl_res["event_tp"].get(ph, 0)
        if tot == 0:
            continue
        rows.append([ph.replace("_", " "), str(tot),
                     str(b_tp), f"{b_tp/tot:.2f}",
                     str(t_tp), f"{t_tp/tot:.2f}"])

    fig, ax = plt.subplots(figsize=(13, 1 + len(rows) * 0.55))
    ax.axis("off")
    tbl = ax.table(cellText=[header] + rows, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.6)
    for j in range(len(header)):
        tbl[0, j].set_text_props(fontweight="bold")
        tbl[0, j].set_facecolor("#dce6f1")
    for i, row in enumerate(rows, start=1):
        b_rec, t_rec = float(row[3]), float(row[5])
        col = 5 if t_rec >= b_rec else 3
        tbl[i, col].set_facecolor("#d4edda")
    fp_b = base_res["fp_total"]
    fp_t = tcl_res["fp_total"]
    ax.set_title(f"Event-level detection (any IoU > 0, min_duration={MIN_DURATION} frames)\n"
                 f"False alarm segments — Baseline: {fp_b}  |  {label_b}: {fp_t}",
                 fontsize=11, pad=14)
    fig.tight_layout()
    path = out_dir / f"segment_event_recall{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Table 2: F1@IoU for each threshold ───────────────────────────────
    for k in IOU_THRESHOLDS:
        header = ["Phase",
                  "Baseline F1", "Baseline Prec", "Baseline Rec",
                  f"{label_b} F1", f"{label_b} Prec", f"{label_b} Rec",
                  "#GT segs"]
        rows = []
        key = f"f1@{k}"
        for ph in phases:
            b = base_res[key].get(ph)
            t = tcl_res[key].get(ph)
            if b is None:
                continue
            t = t or {"f1": 0, "prec": 0, "rec": 0, "n_gt": b["n_gt"], "n_pred": 0}
            rows.append([ph.replace("_", " "),
                         f"{b['f1']:.3f}", f"{b['prec']:.3f}", f"{b['rec']:.3f}",
                         f"{t['f1']:.3f}", f"{t['prec']:.3f}", f"{t['rec']:.3f}",
                         str(b["n_gt"])])

        fig, ax = plt.subplots(figsize=(16, 1 + len(rows) * 0.55))
        ax.axis("off")
        tbl = ax.table(cellText=[header] + rows, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.6)
        for j in range(len(header)):
            tbl[0, j].set_text_props(fontweight="bold")
            tbl[0, j].set_facecolor("#dce6f1")
        for i, row in enumerate(rows, start=1):
            b_f1, t_f1 = float(row[1]), float(row[4])
            col = 4 if t_f1 >= b_f1 else 1
            tbl[i, col].set_facecolor("#d4edda")
        ax.set_title(f"F1 @ IoU ≥ {k:.2f}  (min_duration={MIN_DURATION} frames)",
                     fontsize=11, pad=14)
        fig.tight_layout()
        path = out_dir / f"segment_f1_iou{int(k*100)}{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_segment_metrics(base_results, tcl_results, out_dir, suffix="", label_b="TCL β=0.02"):
    """Bar chart: event recall per phase — Baseline vs TCL."""
    base_tp, base_tot = base_results["event_tp"], base_results["event_tot"]
    tcl_tp,  tcl_tot  = tcl_results["event_tp"],  tcl_results["event_tot"]

    phases = [p for p in UNKNOWN_PHASES if base_tot.get(p, 0) > 0 or tcl_tot.get(p, 0) > 0]
    x = np.arange(len(phases))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    base_rec = [base_tp.get(p, 0) / max(base_tot.get(p, 1), 1) for p in phases]
    tcl_rec  = [tcl_tp.get(p, 0)  / max(tcl_tot.get(p, 1), 1)  for p in phases]

    bars1 = ax.bar(x - w/2, base_rec, w, label="Baseline", color="#1f77b4",
                   edgecolor="black", lw=0.4)
    bars2 = ax.bar(x + w/2, tcl_rec,  w, label=label_b, color="#d62728",
                   edgecolor="black", lw=0.4)
    for bars, vals in [(bars1, base_rec), (bars2, tcl_rec)]:
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                        f"{v:.2f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in phases], fontsize=9)
    ax.set_ylabel("Event recall", fontsize=12)
    ax.set_title(f"Segment detection recall — Baseline vs {label_b}\n"
                 f"(min_duration={MIN_DURATION} frames, any IoU > 0)", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"segment_event_recall{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_false_alarms_and_delay(all_video_results, threshold, video_names):
    """Compute per-video false alarms and per-phase detection delay.

    Returns:
        fa_rows  : list of (video_name, category, n_fa) for videos without unknowns
        delay_rows: list of (phase, delay_frames) for detected GT segments
    """
    fa_rows, delay_rows = [], []

    for (scores, gt_labels, gt_phases), name in zip(all_video_results, video_names):
        pred_segs = get_predicted_segments(scores, threshold)
        gt_segs   = get_gt_segments(gt_labels, gt_phases)
        has_unknown = (gt_labels == -1).any()

        if not has_unknown:
            # Count predicted segments = false alarms (no GT unknown to match)
            n_fa = len(pred_segs)
            if any(c in name for c in COMPLICATED_VIDEOS):
                category = "complicated"
            else:
                category = "routine"
            fa_rows.append((name, category, n_fa))
        else:
            # Detection delay: for each GT segment detected, when does the alert fire?
            for (gs, ge, gph) in gt_segs:
                # Find first pred segment overlapping this GT
                overlapping = [p for p in pred_segs if seg_iou(p, (gs, ge)) > 0]
                if not overlapping:
                    continue
                first_alert = min(p[0] for p in overlapping)
                delay = max(0, first_alert - gs)  # 0 if pred starts before GT
                delay_rows.append((gph, delay))

    return fa_rows, delay_rows


def plot_false_alarms(fa_rows_base, fa_rows_tcl, out_dir,
                      suffix="", label_b="TCL β=0.02 KNN"):
    """PNG table: false alarms per video (routine vs complicated)."""
    all_names = sorted(set(r[0] for r in fa_rows_base + fa_rows_tcl))
    base_map  = {r[0]: r for r in fa_rows_base}
    tcl_map   = {r[0]: r for r in fa_rows_tcl}

    header = ["Video", "Category", "Baseline FA", f"{label_b} FA"]
    rows = []
    for name in all_names:
        cat  = base_map[name][1] if name in base_map else tcl_map[name][1]
        b_fa = base_map[name][2] if name in base_map else "—"
        t_fa = tcl_map[name][2]  if name in tcl_map  else "—"
        short = name[:40]
        rows.append([short, cat, str(b_fa), str(t_fa)])

    fig, ax = plt.subplots(figsize=(13, 1 + len(rows) * 0.6))
    ax.axis("off")
    tbl = ax.table(cellText=[header] + rows, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.7)
    for j in range(len(header)):
        tbl[0, j].set_text_props(fontweight="bold")
        tbl[0, j].set_facecolor("#dce6f1")
    for i, row in enumerate(rows, start=1):
        if row[1] == "complicated":
            for j in range(len(header)):
                tbl[i, j].set_facecolor("#fff3cd")  # yellow = ambiguous
        b, t = row[2], row[3]
        if b.isdigit() and t.isdigit() and int(t) < int(b):
            tbl[i, 3].set_facecolor("#d4edda")  # green = TCL fewer FA

    ax.set_title(f"False alarms per video (no GT unknowns, min_duration={MIN_DURATION})\n"
                 "Yellow = clinically complex (anomalies possibly unlabelled)",
                 fontsize=11, pad=14)
    fig.tight_layout()
    path = out_dir / f"false_alarms_per_video{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_detection_delay(delay_rows_base, delay_rows_tcl, out_dir,
                         suffix="", label_b="TCL β=0.02 KNN"):
    """PNG table: mean detection delay per phase."""
    from collections import defaultdict
    def aggregate(rows):
        d = defaultdict(list)
        for ph, delay in rows:
            d[ph].append(delay)
        return d

    base_d = aggregate(delay_rows_base)
    tcl_d  = aggregate(delay_rows_tcl)
    phases  = [p for p in UNKNOWN_PHASES if p in base_d or p in tcl_d]

    header = ["Phase", "Baseline\nmean delay (frames)", "Baseline\ndetections",
              f"{label_b}\nmean delay (frames)", f"{label_b}\ndetections"]
    rows = []
    for ph in phases:
        bd = base_d.get(ph, [])
        td = tcl_d.get(ph, [])
        b_str = f"{np.mean(bd):.0f}" if bd else "—"
        t_str = f"{np.mean(td):.0f}" if td else "—"
        rows.append([ph.replace("_", " "), b_str, str(len(bd)), t_str, str(len(td))])

    fig, ax = plt.subplots(figsize=(13, 1 + len(rows) * 0.6))
    ax.axis("off")
    tbl = ax.table(cellText=[header] + rows, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.7)
    for j in range(len(header)):
        tbl[0, j].set_text_props(fontweight="bold")
        tbl[0, j].set_facecolor("#dce6f1")
    for i, row in enumerate(rows, start=1):
        b, t = row[1], row[3]
        if b.isdigit() and t.isdigit() and int(t) <= int(b):
            tbl[i, 3].set_facecolor("#d4edda")

    ax.set_title("Detection delay — frames from GT segment start to first alert\n"
                 "(only detected segments counted; — = none detected)",
                 fontsize=11, pad=14)
    fig.tight_layout()
    path = out_dir / f"detection_delay{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    model_base  = load_model(EXP_BASELINE, device)
    model_tcl   = load_model(EXP_TCL, device)
    model_mixup = load_model(EXP_MIXUP, device)

    print("Extracting features (train / val / test) for all models...")
    train_base = extract_per_video(model_base,  "train", device)
    val_base   = extract_per_video(model_base,  "val",   device)
    test_base  = extract_per_video(model_base,  "test",  device)

    train_tcl  = extract_per_video(model_tcl,   "train", device)
    val_tcl    = extract_per_video(model_tcl,   "val",   device)
    test_tcl   = extract_per_video(model_tcl,   "test",  device)

    train_mix  = extract_per_video(model_mixup, "train", device)
    val_mix    = extract_per_video(model_mixup, "val",   device)
    test_mix   = extract_per_video(model_mixup, "test",  device)

    del model_base, model_tcl, model_mixup
    torch.cuda.empty_cache()

    print("Fitting KNN...")
    nn_base  = fit_knn(train_base)
    nn_tcl   = fit_knn(train_tcl)
    nn_mixup = fit_knn(train_mix)

    print("Calibrating thresholds (p95 val)...")
    thr_base  = get_threshold(nn_base,  val_base)
    thr_tcl   = get_threshold(nn_tcl,   val_tcl)
    thr_mixup = get_threshold(nn_mixup, val_mix)
    print(f"  Baseline threshold:      {thr_base:.4f}")
    print(f"  TCL β=0.02 threshold:    {thr_tcl:.4f}")
    print(f"  TCL+Mixup threshold:     {thr_mixup:.4f}")

    print(f"\nGenerating timelines for {len(test_base)} test videos...")

    # Build index by video name for alignment
    tcl_by_name   = {name: (feats, preds, labels, phases)
                     for name, feats, preds, labels, phases in test_tcl}
    mixup_by_name = {name: (feats, preds, labels, phases)
                     for name, feats, preds, labels, phases in test_mix}

    video_results_base, video_results_tcl, video_results_mixup = [], [], []
    video_names_list = []

    for name, feats_b, preds_b, labels, phases in test_base:
        n_unk = (labels == -1).sum()
        unk_set = set(phases[labels == -1]) - {""}
        print(f"\n  {name}  ({n_unk} unknown frames: {', '.join(unk_set) or 'none'})")

        scores_b = nn_base.kneighbors(feats_b)[0][:, -1]

        feats_t, preds_t, _, _   = tcl_by_name[name]
        scores_t = nn_tcl.kneighbors(feats_t)[0][:, -1]

        feats_m, preds_m, _, _   = mixup_by_name[name]
        scores_m = nn_mixup.kneighbors(feats_m)[0][:, -1]

        plot_video(name, labels, phases,
                   preds_t, preds_m,
                   scores_b, thr_base,
                   scores_t, thr_tcl,
                   scores_m, thr_mixup,
                   OUT_DIR)

        video_results_base.append((scores_b, labels, phases))
        video_results_tcl.append((scores_t, labels, phases))
        video_results_mixup.append((scores_m, labels, phases))
        video_names_list.append(name)

    print(f"\nAll timelines saved to: {OUT_DIR}")

    # ── Segment-level metrics ─────────────────────────────────────────────
    base_res  = compute_segment_metrics(video_results_base,  thr_base,  "Baseline KNN")
    tcl_res   = compute_segment_metrics(video_results_tcl,   thr_tcl,   "TCL β=0.02 KNN")
    mixup_res = compute_segment_metrics(video_results_mixup, thr_mixup, "TCL+Mixup KNN")
    plot_metrics_tables(base_res, tcl_res, OUT_DIR)
    plot_segment_metrics(base_res, tcl_res, OUT_DIR)
    plot_metrics_tables(base_res, mixup_res,
                        OUT_DIR, suffix="_vs_mixup")
    plot_segment_metrics(base_res, mixup_res,
                         OUT_DIR, suffix="_vs_mixup")

    # ── False alarms and detection delay ─────────────────────────────────
    fa_rows_base,  delay_rows_base  = compute_false_alarms_and_delay(
        video_results_base,  thr_base,  video_names_list)
    fa_rows_tcl,   delay_rows_tcl   = compute_false_alarms_and_delay(
        video_results_tcl,   thr_tcl,   video_names_list)
    fa_rows_mixup, delay_rows_mixup = compute_false_alarms_and_delay(
        video_results_mixup, thr_mixup, video_names_list)
    plot_false_alarms(fa_rows_base, fa_rows_tcl,   OUT_DIR)
    plot_false_alarms(fa_rows_base, fa_rows_mixup, OUT_DIR,
                      suffix="_vs_mixup", label_b="TCL+Mixup KNN")
    plot_detection_delay(delay_rows_base, delay_rows_tcl,   OUT_DIR)
    plot_detection_delay(delay_rows_base, delay_rows_mixup, OUT_DIR,
                         suffix="_vs_mixup", label_b="TCL+Mixup KNN")


if __name__ == "__main__":
    main()

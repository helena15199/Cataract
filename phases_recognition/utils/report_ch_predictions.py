"""Report frames where the binary CH head predicts CH with no GT annotation.

Generates an Excel file with:
  - Sheet "Summary"  : one row per video with TP/FP/FN counts
  - Sheet "Details"  : one row per zone (missing annotation or FN)

Usage:
    python phases_recognition/utils/report_ch_predictions.py \
        --feat_dir /home/helena/UCL_video_cataract/features_v1.8/test/
"""

import argparse
import json
import pathlib
import re

import cv2
import numpy as np
import pandas as pd

FRAME_RE = re.compile(r"Frame_(\d+)")


def build_fps_map(video_source_dir: pathlib.Path) -> dict[str, float]:
    """Return {video_name: fps} by reading original mp4 files."""
    fps_map = {}
    for mp4 in video_source_dir.rglob("*.mp4"):
        cap = cv2.VideoCapture(str(mp4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0:
            video_name = mp4.parent.name.split("_json_")[0]
            fps_map[video_name] = 60.0 if fps > 30 else 25.0
    return fps_map


def get_sorted_frame_numbers(labels: dict, video_key_fragment: str) -> list[int]:
    """Return frame numbers sorted the same way as extract_features.py — with duplicates."""
    keys = [k for k in labels if video_key_fragment in k]
    keyed = []
    for k in keys:
        m = FRAME_RE.search(k)
        if m:
            keyed.append((int(m.group(1)), k))
    # Same sort as extract_features: by frame number, duplicates kept
    keyed.sort(key=lambda x: x[0])
    return [fn for fn, _ in keyed]


def frames_to_segments(frame_list: list[int], fps: float) -> list[tuple]:
    """Group consecutive frames — max 2s gap between frames."""
    if not frame_list:
        return []
    max_gap = int(fps * 2)
    segments = []
    start = frame_list[0]
    prev  = frame_list[0]
    for fn in frame_list[1:]:
        if fn - prev > max_gap:
            segments.append((start, prev, start / fps, prev / fps))
            start = fn
        prev = fn
    segments.append((start, prev, start / fps, prev / fps))
    return segments


def fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def process_split(feat_dir: pathlib.Path, labels: dict, fps: int,
                  fps_map: dict[str, float] | None = None) -> list[dict]:
    feat_files = sorted(
        p for p in feat_dir.glob("*.npy")
        if not any(p.stem.endswith(s) for s in
                   ["_labels", "_binary_ch", "_frame_numbers", "_mahal"])
    )
    results = []
    for feat_file in feat_files:
        video_name = feat_file.stem
        ch_file    = feat_dir / f"{video_name}_binary_ch.npy"
        label_file = feat_dir / f"{video_name}_labels.npy"
        if not ch_file.exists() or not label_file.exists():
            continue

        pred_ch   = np.load(ch_file)
        gt_labels = np.load(label_file)
        base_name  = video_name.split("_json_")[0]
        video_fps  = (fps_map or {}).get(base_name, fps)
        frame_nums = get_sorted_frame_numbers(labels, base_name)
        if len(frame_nums) != len(pred_ch):
            frame_nums = list(range(len(pred_ch)))

        # Si un numéro de frame est annoté CH même une fois → GT=1 pour toutes ses positions
        ch_frame_set = {fn for fn, label in zip(frame_nums, gt_labels) if label == -1}
        gt_ch_effective = [1 if frame_nums[i] in ch_frame_set else 0 for i in range(len(pred_ch))]

        pred_only = [frame_nums[i] for i in range(len(pred_ch))
                     if pred_ch[i] == 1 and gt_ch_effective[i] == 0]
        gt_only   = [frame_nums[i] for i in range(len(pred_ch))
                     if pred_ch[i] == 0 and gt_ch_effective[i] == 1]
        true_pos  = [frame_nums[i] for i in range(len(pred_ch))
                     if pred_ch[i] == 1 and gt_ch_effective[i] == 1]

        results.append({
            "name":        base_name,
            "gt_total":    int((gt_labels == -1).sum()),
            "pred_total":  int(pred_ch.sum()),
            "tp":          len(true_pos),
            "fp":          len(pred_only),
            "fn":          len(gt_only),
            "fps":         video_fps,
            "pred_segs":   frames_to_segments(sorted(pred_only), video_fps),
            "fn_segs":     frames_to_segments(sorted(gt_only),   video_fps),
            "pred_frames": pred_only,
        })
    return results


def build_dataframes(all_results: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    detail_rows  = []

    for split, results in all_results.items():
        for r in results:
            summary_rows.append({
                "Split":       split,
                "Vidéo":       r["name"],
                "FPS":         r.get("fps", 25),
                "GT CH":       r["gt_total"],
                "Pred CH":     r["pred_total"],
                "TP":          r["tp"],
                "FP (pred sans GT)": r["fp"],
                "FN (GT raté)":      r["fn"],
            })

            for s, e, ts, te in r["pred_segs"]:
                n = len([f for f in r["pred_frames"] if s <= f <= e])
                detail_rows.append({
                    "Split":   split,
                    "Vidéo":   r["name"],
                    "Type":    "Pred=CH sans GT",
                    "Début":   fmt_time(ts),
                    "Fin":     fmt_time(te),
                    "Frame début": s,
                    "Frame fin":   e,
                    "Nb frames":   n,
                    "Note": "Annotation potentiellement manquante",
                })

            for s, e, ts, te in r["fn_segs"]:
                detail_rows.append({
                    "Split":   split,
                    "Vidéo":   r["name"],
                    "Type":    "FN (GT raté)",
                    "Début":   fmt_time(ts),
                    "Fin":     fmt_time(te),
                    "Frame début": s,
                    "Frame fin":   e,
                    "Nb frames":   e - s + 1,
                    "Note": "Modèle n'a pas détecté CH annoté",
                })

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def main(args):
    feat_root = pathlib.Path(args.feat_dir).parent
    with open(args.labels_json) as f:
        labels = json.load(f)

    fps_map = {}
    source_dir = pathlib.Path("/home/helena/UCL_video_cataract/video_matching_json_ayushi")
    if source_dir.exists():
        fps_map = build_fps_map(source_dir)
        print(f"FPS map chargée : {len(fps_map)} vidéos")

    all_results = {}
    for split in ["train", "val", "test"]:
        split_dir = feat_root / split
        if split_dir.exists():
            results = process_split(split_dir, labels, args.fps, fps_map)
            if results:
                all_results[split] = results

    df_summary, df_details = build_dataframes(all_results)

    out_path = feat_root / "report_ch_predictions.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_details.to_excel(writer, sheet_name="Details", index=False)

        # Auto-adjust column widths
        for sheet_name, df in [("Summary", df_summary), ("Details", df_details)]:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col) + 2
                ws.column_dimensions[col[0].column_letter].width = min(max_len, 40)

    print(f"Excel sauvegardé : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Report CH binary prediction vs GT")
    parser.add_argument("--feat_dir",    type=str, required=True)
    parser.add_argument("--labels_json", type=str,
                        default="/home/helena/UCL_video_cataract/dataset_temporal/labels.json")
    parser.add_argument("--fps",         type=int, default=25,
                        help="FPS original de la vidéo (défaut: 25)")
    main(parser.parse_args())

"""Detect frames with duplicate files (same frame number, different phases) per video.

Usage:
    python datasets/detect_alternating.py
    python datasets/detect_alternating.py --video "Video 61 (cat25)"
"""

import argparse
import pathlib
import re
from collections import defaultdict, Counter

DATASET_ROOT = pathlib.Path("/home/helena/UCL_video_cataract/dataset_temporal")
FRAME_RE     = re.compile(r"Frame_(\d+)")
PHASE_RE     = re.compile(r"_Phase_(.+)\.png$")


def main():
    parser = argparse.ArgumentParser("Detect duplicate frames on disk")
    parser.add_argument("--video", default=None,
                        help='Limiter à une vidéo, ex: "Video 61 (cat25)"')
    args = parser.parse_args()

    video_frames: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))

    for p in DATASET_ROOT.rglob("*.png"):
        key = str(p.relative_to(DATASET_ROOT))
        if args.video and args.video not in key:
            continue
        parts = key.split("/")
        video = parts[1].split("_json_")[0] if len(parts) > 1 else parts[0]
        m = FRAME_RE.search(p.name)
        pm = PHASE_RE.search(p.name)
        if m and pm:
            video_frames[video][int(m.group(1))].append(pm.group(1))

    total = 0
    for video in sorted(video_frames):
        duplicates = {fn: phases for fn, phases in video_frames[video].items() if len(phases) > 1}
        if not duplicates:
            continue
        total += len(duplicates)
        pairs = Counter(tuple(sorted(p)) for p in duplicates.values())
        print(f"\n{video}  →  {len(duplicates)} frames en double")
        for pair, count in pairs.most_common():
            print(f"  {count:4d}x  {pair[0]}  ↔  {pair[1]}")

    if total == 0:
        print("Aucun doublon détecté.")
    else:
        print(f"\nTotal : {total} frames en double")


if __name__ == "__main__":
    main()

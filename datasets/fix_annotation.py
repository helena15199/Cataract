"""Fix IA/CP annotation issues in labels.json + image files.

Two problems handled:
  1. Duplicate frames: same frame number has TWO files (one IA, one CP).
     → Delete the file + JSON entry for the wrong phase.

  2. Alternating single-label frames: only one file per frame, but labels
     flip rapidly between IA and CP.
     → Reassign the JSON label to the chosen phase (file stays, label changes).

Usage:
    python datasets/fix_annotation.py \
        --video "Video 32 (cat144)" \
        --phase "Capsule_polishing" \
        --dry_run

    python datasets/fix_annotation.py \
        --video "Video 32 (cat144)" \
        --phase "Capsule_polishing"
"""

import argparse
import json
import pathlib
import re
import shutil
from collections import defaultdict
from datetime import datetime

LABELS_PATH  = pathlib.Path("/home/helena/UCL_video_cataract/dataset_temporal/labels.json")
DATASET_ROOT = LABELS_PATH.parent
FRAME_RE     = re.compile(r"Frame_(\d+)")
IA = "Irrigation_and_aspiration"
CP = "Capsule_polishing"


def frame_number(key: str) -> int:
    m = FRAME_RE.search(key)
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser("Fix IA/CP annotation issues")
    parser.add_argument("--video",   required=True,
                        help='Ex: "Video 32 (cat144)"')
    parser.add_argument("--phase",   required=True,
                        choices=["Irrigation_and_aspiration", "Capsule_polishing"],
                        help="Phase à garder")
    parser.add_argument("--dry_run", action="store_true",
                        help="Affiche les changements sans rien modifier")
    parser.add_argument("--alternating_only", action="store_true",
                        help="Step 2 : réassigne uniquement les frames dans la zone d'alternance "
                             "(avant la dernière frame de --phase), pas la section stable après")
    args = parser.parse_args()

    other_phase = CP if args.phase == IA else IA

    with open(LABELS_PATH) as f:
        labels = json.load(f)

    # All JSON keys for this video
    video_keys = {k: v for k, v in labels.items() if args.video in k}

    # Group by frame number
    by_frame: dict[int, list[str]] = defaultdict(list)
    for k in video_keys:
        by_frame[frame_number(k)].append(k)

    print(f"\nVidéo   : {args.video}")
    print(f"Garder  : {args.phase}")
    print(f"Enlever : {other_phase}")
    print(f"Frames totales dans le JSON : {len(video_keys)}")

    # ---------------------------------------------------------------
    # Step 1 — Duplicate frames (two files, two JSON entries per frame)
    # ---------------------------------------------------------------
    files_to_delete  = []   # physical files to remove
    json_keys_to_del = []   # JSON keys to remove

    for fn, keys in sorted(by_frame.items()):
        phases = {labels[k] for k in keys}
        if args.phase in phases and other_phase in phases:
            for k in keys:
                if labels[k] == other_phase:
                    json_keys_to_del.append(k)
                    # Resolve physical path
                    img_path = DATASET_ROOT / k
                    if img_path.exists():
                        files_to_delete.append(img_path)

    print(f"\n[Step 1] Frames en double (IA+CP) : {len(json_keys_to_del)}")
    for k in sorted(json_keys_to_del):
        img = DATASET_ROOT / k
        exists = "✓" if img.exists() else "✗ (fichier absent)"
        print(f"  DELETE file  {exists}  {pathlib.Path(k).name}")

    # ---------------------------------------------------------------
    # Step 2 — Alternating single-label frames
    # ---------------------------------------------------------------
    # After removing duplicates, find the last frame still labeled as args.phase
    # (= the boundary of the stable zone). CP frames beyond that boundary are
    # stable and must NOT be touched; only those before it are alternating noise.
    if args.alternating_only:
        remaining_phase_frames = [
            frame_number(k) for k, v in video_keys.items()
            if v == args.phase and k not in json_keys_to_del
        ]
        last_phase_frame = max(remaining_phase_frames) if remaining_phase_frames else 0
        # Also consider IA frames from duplicates that will be kept
        kept_ia = [
            frame_number(k) for k, v in video_keys.items()
            if v == args.phase
        ]
        boundary = max(kept_ia) if kept_ia else 0

        remaining_other = {
            k: v for k, v in video_keys.items()
            if v == other_phase
            and k not in json_keys_to_del
            and frame_number(k) <= boundary
        }
        skipped = {
            k: v for k, v in video_keys.items()
            if v == other_phase
            and k not in json_keys_to_del
            and frame_number(k) > boundary
        }
        print(f"\n[Step 2] Boundary (dernière frame '{args.phase}') : Frame {boundary}")
        print(f"[Step 2] Frames alternantes à réassigner (avant boundary) : {len(remaining_other)}")
        for k in sorted(remaining_other):
            print(f"  REASSIGN  {pathlib.Path(k).name}  →  {args.phase}")
        print(f"[Step 2] Frames '{other_phase}' stables conservées (après boundary) : {len(skipped)}")
    else:
        remaining_other = {
            k: v for k, v in video_keys.items()
            if v == other_phase and k not in json_keys_to_del
        }
        print(f"\n[Step 2] Frames alternantes à réassigner : {len(remaining_other)}")
        for k in sorted(remaining_other):
            print(f"  REASSIGN  {pathlib.Path(k).name}  →  {args.phase}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\nRésumé :")
    print(f"  {len(files_to_delete):4d} fichiers à supprimer du disque")
    print(f"  {len(json_keys_to_del):4d} entrées à supprimer du JSON")
    print(f"  {len(remaining_other):4d} entrées à réassigner dans le JSON")

    if not files_to_delete and not json_keys_to_del and not remaining_other:
        print("\nRien à modifier.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Aucune modification effectuée.")
        return

    # ---------------------------------------------------------------
    # Apply
    # ---------------------------------------------------------------
    backup = LABELS_PATH.with_suffix(
        f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    shutil.copy(LABELS_PATH, backup)
    print(f"\nBackup JSON : {backup}")

    # Delete files
    for img in files_to_delete:
        img.unlink()
    print(f"{len(files_to_delete)} fichiers supprimés.")

    # Delete JSON entries
    for k in json_keys_to_del:
        del labels[k]

    # Reassign alternating labels
    for k in remaining_other:
        labels[k] = args.phase

    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"labels.json mis à jour.")


if __name__ == "__main__":
    main()

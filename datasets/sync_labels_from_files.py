"""Sync labels.json from actual image files on disk.

The phase is encoded in each filename: ..._Phase_<PhaseName>.png
This script makes labels.json match what's physically present,
adding missing entries, fixing wrong labels, and removing stale entries.

Usage:
    python datasets/sync_labels_from_files.py --dry_run
    python datasets/sync_labels_from_files.py
    python datasets/sync_labels_from_files.py --video "Video 32 (cat144)"
"""

import argparse
import json
import pathlib
import re
import shutil
from datetime import datetime

LABELS_PATH  = pathlib.Path("/home/helena/UCL_video_cataract/dataset_temporal/labels.json")
DATASET_ROOT = LABELS_PATH.parent
PHASE_RE     = re.compile(r"_Phase_(.+)\.png$")

PHASE_MAP = {
    "Capsule_polish":                                     "Capsule_polishing",
    "Capusule_polishing":                                 "Capsule_polishing",
    "Cornel_hydration":                                   "Corneal_hydration",
    "Hydrate_Cornea":                                     "Corneal_hydration",
    "Hydrate_cornea":                                     "Corneal_hydration",
    "Wound_Hydration":                                    "Wound_hydration",
    "IrrigationAspiration":                               "Irrigation_and_aspiration",
    "Irrigation_and_aspirationIrrigation_and_aspiration": "Irrigation_and_aspiration",
    "irrigation_and_aspiration":                          "Irrigation_and_aspiration",
    "Injection":                                          "Incision",
    "Lens_Implant":                                       "Lens_implant_settingup",
    "Lens_implant_setting_up":                            "Lens_implant_settingup",
    "Maliugan_ring_insertion":                            "Malyugin_ring_insertion",
    "Maliugan_ring_removal":                              "Malyugin_ring_removal",
    "Phaecoemulsification":                               "Phacoemulsification",
    "Tonifying_and_Antibiotics":                          "Tonifying_and_antibiotics",
    "Tonifying_and_antibitoics":                          "Tonifying_and_antibiotics",
    "Tonifying_and_antiobiotics":                         "Tonifying_and_antibiotics",
    "Viscous_Agent_Injection":                            "Viscous_agent_injection",
    "Viscous_Agent_Removal":                              "Viscous_agent_removal",
}


def phase_from_path(p: pathlib.Path) -> str | None:
    m = PHASE_RE.search(p.name)
    if not m:
        return None
    return PHASE_MAP.get(m.group(1), m.group(1))


def main():
    parser = argparse.ArgumentParser("Sync labels.json from image files on disk")
    parser.add_argument("--video",   default=None,
                        help='Limiter à une vidéo, ex: "Video 32 (cat144)"')
    parser.add_argument("--dry_run", action="store_true",
                        help="Affiche les changements sans rien modifier")
    args = parser.parse_args()

    with open(LABELS_PATH) as f:
        labels = json.load(f)

    # Scan all .png files on disk
    all_files = DATASET_ROOT.rglob("*.png")
    disk: dict[str, str] = {}
    for p in all_files:
        phase = phase_from_path(p)
        if phase is None:
            continue
        key = str(p.relative_to(DATASET_ROOT))
        if args.video and args.video not in key:
            continue
        disk[key] = phase

    # Filter JSON to the same scope
    if args.video:
        scoped_labels = {k: v for k, v in labels.items() if args.video in k}
    else:
        scoped_labels = dict(labels)

    added   = {k: v for k, v in disk.items() if k not in scoped_labels}
    removed = {k: v for k, v in scoped_labels.items() if k not in disk}
    fixed   = {k: (scoped_labels[k], disk[k]) for k in disk
               if k in scoped_labels and scoped_labels[k] != disk[k]}

    scope = f'"{args.video}"' if args.video else "tout le dataset"
    print(f"\nScope    : {scope}")
    print(f"Fichiers sur le disque  : {len(disk)}")
    print(f"Entrées dans le JSON    : {len(scoped_labels)}")

    all_changes = (
        [(k, "+", None,  v)        for k, v       in added.items()]   +
        [(k, "~", old,   new)      for k, (old, new) in fixed.items()] +
        [(k, "-", v,     None)     for k, v       in removed.items()]
    )
    all_changes.sort(key=lambda x: pathlib.Path(x[0]).name)

    print(f"\n{'ACTION':<6}  {'FICHIER':<80}  DÉTAIL")
    print("-" * 110)
    for k, action, old, new in all_changes:
        name = pathlib.Path(k).name
        if action == "+":
            detail = f"→ ajout dans JSON  ({new})"
        elif action == "~":
            detail = f"→ {old}  corrigé en  {new}"
        else:
            detail = f"→ supprimé du JSON  ({old})"
        print(f"  [{action}]  {name:<80}  {detail}")

    print(f"\n[ADD] {len(added)}  [FIX] {len(fixed)}  [REMOVE] {len(removed)}")

    print(f"\nRésumé : +{len(added)} ajouts, ~{len(fixed)} corrections, -{len(removed)} suppressions")

    affected_videos = sorted({
        pathlib.Path(k).parent.name.split("_json_")[0]
        for k in list(added) + list(fixed) + list(removed)
    })
    if affected_videos:
        print(f"\nVidéos concernées ({len(affected_videos)}) :")
        for v in affected_videos:
            print(f"  {v}")

    known = {"Irrigation_and_aspiration", "Capsule_polishing"}
    other_phases = sorted({
        v for k, v in list(added.items()) + list(removed.items())
    } | {
        old for _, (old, _) in fixed.items()
    } | {
        new for _, (_, new) in fixed.items()
    } - known)
    if other_phases:
        print(f"\nAutres phases concernées :")
        for p in other_phases:
            print(f"  {p}")

    if not added and not fixed and not removed:
        print("\nJSON déjà synchronisé.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Aucune modification effectuée.")
        return

    backup = LABELS_PATH.with_suffix(
        f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    shutil.copy(LABELS_PATH, backup)
    print(f"\nBackup : {backup}")

    for k, v in added.items():
        labels[k] = v
    for k, (_, new) in fixed.items():
        labels[k] = new
    for k in removed:
        del labels[k]

    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    print("labels.json mis à jour.")


if __name__ == "__main__":
    main()

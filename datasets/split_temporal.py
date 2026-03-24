import os
import shutil
import random
import re
import argparse
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd


PHASE_MAP = {
    "Capsule_polish":                                    "Capsule_polishing",
    "Capusule_polishing":                                "Capsule_polishing",
    "Cornel_hydration":                                  "Corneal_hydration",
    "Hydrate_Cornea":                                    "Corneal_hydration",
    "Hydrate_cornea":                                    "Corneal_hydration",
    "Wound_Hydration":                                   "Wound_hydration",
    "IrrigationAspiration":                              "Irrigation_and_aspiration",
    "Irrigation_and_aspirationIrrigation_and_aspiration": "Irrigation_and_aspiration",
    "irrigation_and_aspiration":                         "Irrigation_and_aspiration",
    "Injection":                                         "Incision",
    "Lens_Implant":                                      "Lens_implant_settingup",
    "Lens_implant_setting_up":                           "Lens_implant_settingup",
    "Maliugan_ring_insertion":                           "Malyugin_ring_insertion",
    "Maliugan_ring_removal":                             "Malyugin_ring_removal",
    "Phaecoemulsification":                              "Phacoemulsification",
    "Tonifying_and_Antibiotics":                         "Tonifying_and_antibiotics",
    "Tonifying_and_antibitoics":                         "Tonifying_and_antibiotics",
    "Tonifying_and_antiobiotics":                        "Tonifying_and_antibiotics",
    "Viscous_Agent_Injection":                           "Viscous_agent_injection",
    "Viscous_Agent_Removal":                             "Viscous_agent_removal",
}


def load_grading_mapping(excel_path):
    """
    Load grading from Excel column 'grading'.
    Returns a mapping: video_number (int) -> grading (str)
    ex: {1: "2", 22: "3", ...}
    """
    df = pd.read_excel(excel_path)
    mapping = {}
    for _, row in df.iterrows():
        video_name = str(row["Videos"]).strip()
        raw_grading = row["grading"]

        if pd.isna(raw_grading):
            continue

        match = re.search(r'\d+', video_name)
        if not match:
            continue

        video_number = int(match.group())
        mapping[video_number] = str(int(float(raw_grading)))

    return mapping


def extract_level(video_name, grading_mapping):
    """
    Find grading level of a video folder by extracting its number.
    'Video 22_json_Video 22' -> grading_mapping[22]
    """
    match = re.search(r'[Vv]ideo\s*(\d+)', video_name)
    if match:
        video_number = int(match.group(1))
        return grading_mapping.get(video_number, "unknown")
    return "unknown"


def extract_phases(video_path):
    return [
        item for item in os.listdir(video_path)
        if os.path.isdir(os.path.join(video_path, item))
    ]


def build_video_phase_mapping(source_dir):
    video_phases = {}
    phase_to_videos = defaultdict(list)
    video_phase_frame_counts = defaultdict(lambda: defaultdict(int))

    for video in tqdm(os.listdir(source_dir), desc="Scanning videos"):
        video_path = os.path.join(source_dir, video)

        if not os.path.isdir(video_path):
            continue

        phases = extract_phases(video_path)

        if not phases:
            continue

        normalized_phases = {PHASE_MAP.get(p, p) for p in phases}
        video_phases[video] = normalized_phases

        for raw_phase in phases:
            norm_phase = PHASE_MAP.get(raw_phase, raw_phase)
            phase_path = os.path.join(video_path, raw_phase)
            n_frames = len([f for f in os.listdir(phase_path) if os.path.isfile(os.path.join(phase_path, f))])
            video_phase_frame_counts[video][norm_phase] += n_frames

        for phase in normalized_phases:
            phase_to_videos[phase].append(video)

    return video_phases, phase_to_videos, video_phase_frame_counts


def smart_split(video_phases, phase_to_videos, video_phase_frame_counts, seed, ratios):

    random.seed(seed)

    train = set()
    val = set()
    test = set()
    used_videos = set()

    sorted_phases = sorted(
        phase_to_videos.keys(),
        key=lambda p: len(phase_to_videos[p])
    )

    for phase in tqdm(sorted_phases, desc="Processing phases"):

        videos = [v for v in phase_to_videos[phase] if v not in used_videos]

        if not videos:
            continue

        n = len(videos)

        if n <= 3:
            # Pour les phases rares, on met la vidéo avec le plus de frames en train
            videos.sort(key=lambda v: video_phase_frame_counts[v][phase], reverse=True)
        else:
            random.shuffle(videos)

        if n == 1:
            train.add(videos[0])
            used_videos.add(videos[0])

        elif n == 2:
            train.add(videos[0])
            test.add(videos[1])
            used_videos.update(videos)

        elif n == 3:
            train.add(videos[0])
            val.add(videos[1])
            test.add(videos[2])
            used_videos.update(videos)

        else:
            n_train = int(ratios[0] * n)
            n_remaining = n - n_train
            n_val = n_remaining // 2
            n_test = n_remaining - n_val

            train.update(videos[:n_train])
            val.update(videos[n_train:n_train + n_val])
            test.update(videos[n_train + n_val:])
            used_videos.update(videos)

    return {
        "train": train,
        "val": val,
        "test": test
    }


def fix_missing_train_phases(split, video_phases, phase_to_videos, video_phase_frame_counts):
    """
    Après le split, vérifie que chaque phase a au moins une vidéo dans le train.
    Si une phase est absente du train, déplace la vidéo avec le plus de frames
    de val ou test vers train.
    """
    train_phases = set()
    for v in split["train"]:
        train_phases.update(video_phases[v])

    missing = [p for p in phase_to_videos if p not in train_phases]
    if missing:
        print(f"\n⚠️  Phases absentes du train : {missing}")

    for phase in missing:
        moved = False
        for src_split in ("val", "test"):
            candidates = [v for v in phase_to_videos[phase] if v in split[src_split]]
            if not candidates:
                continue
            candidates.sort(key=lambda v: video_phase_frame_counts[v][phase], reverse=True)
            best = candidates[0]
            split[src_split].remove(best)
            split["train"].add(best)
            n = video_phase_frame_counts[best][phase]
            print(f"  → '{best}' déplacée de {src_split} → train (phase '{phase}', {n} frames)")
            moved = True
            break
        if not moved:
            print(f"  ✗ Impossible de corriger la phase '{phase}' (aucune vidéo trouvée)")

    return split


def create_structure(dest_dir, split):
    for split_name in split.keys():
        split_path = os.path.join(dest_dir, split_name)
        os.makedirs(split_path, exist_ok=True)


def populate_dataset_flat(source_dir, dest_dir, split):

    labels_dict = {}

    for split_name, videos in split.items():

        print(f"\nProcessing {split_name} set")
        split_dir = os.path.join(dest_dir, split_name)

        for video in tqdm(videos, desc=f"{split_name} videos"):

            video_path = os.path.join(source_dir, video)
            dest_video_dir = os.path.join(split_dir, video)

            os.makedirs(dest_video_dir, exist_ok=True)

            for phase in os.listdir(video_path):

                phase_path = os.path.join(video_path, phase)

                if not os.path.isdir(phase_path):
                    continue

                files = sorted(os.listdir(phase_path),
                               key=lambda x: int(re.search(r"Frame_(\d+)", x).group(1)))

                for file in files:

                    src_file = os.path.join(phase_path, file)

                    if not os.path.isfile(src_file):
                        continue

                    dst_file = os.path.join(dest_video_dir, file)
                    
                    if os.path.exists(dst_file):
                        continue
                    shutil.copy2(src_file, dst_file)

                    labels_dict[f"{split_name}/{video}/{file}"] = PHASE_MAP.get(phase, phase)

    return labels_dict


def save_labels_json(dest_dir, labels_dict):
    json_path = os.path.join(dest_dir, "labels.json")
    with open(json_path, "w") as f:
        json.dump(labels_dict, f, indent=4)
    print(f"\nLabels saved to {json_path}")


def save_labels_excel(dest_dir, labels_dict):
    df = pd.DataFrame(
        [(k, v) for k, v in labels_dict.items()],
        columns=["image_path", "phase"]
    )
    excel_path = os.path.join(dest_dir, "labels.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Labels saved to {excel_path}")


def analyze_split(split, video_phases, grading_mapping):

    print("\n===== SPLIT SIZE =====")
    for s in split:
        print(f"{s}: {len(split[s])} videos")

    print("\n===== GRADING DISTRIBUTION =====")
    for s in split:
        levels = [extract_level(v, grading_mapping) for v in split[s]]
        print(f"\n{s.upper()}")
        print(Counter(levels))

    print("\n===== PHASE DISTRIBUTION PER SPLIT =====")
    phase_distribution = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter()
    }
    for split_name in split:
        for v in split[split_name]:
            for phase in video_phases[v]:
                phase_distribution[split_name][phase] += 1

    for split_name in phase_distribution:
        print(f"\n{split_name.upper()} PHASES:")
        print(phase_distribution[split_name])


def main():

    parser = argparse.ArgumentParser(
        description="Build flat video-level phase classification dataset"
    )

    parser.add_argument("--source_dir", type=str, required=True,
                        help="Folder containing video subfolders with extracted frames")
    parser.add_argument("--dest_dir", type=str, required=True,
                        help="Destination dataset directory")
    parser.add_argument("--excel_path", type=str, required=True,
                        help="Excel file with video grading info")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio_split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument("--save_excel", action="store_true")

    args = parser.parse_args()

    # CHANGED: load grading from Excel instead of scanning grading folders
    grading_mapping = load_grading_mapping(args.excel_path)

    video_phases, phase_to_videos, video_phase_frame_counts = build_video_phase_mapping(args.source_dir)

    split = smart_split(
        video_phases,
        phase_to_videos,
        video_phase_frame_counts,
        args.seed,
        args.ratio_split
    )

    split = fix_missing_train_phases(split, video_phases, phase_to_videos, video_phase_frame_counts)

    create_structure(args.dest_dir, split)

    labels_dict = populate_dataset_flat(
        args.source_dir,
        args.dest_dir,
        split
    )

    print("Saving labels")
    save_labels_json(args.dest_dir, labels_dict)

    if args.save_excel:
        save_labels_excel(args.dest_dir, labels_dict)

    print("Analyzing split")
    analyze_split(split, video_phases, grading_mapping)

    print("\nDataset successfully created!")


if __name__ == "__main__":
    main()
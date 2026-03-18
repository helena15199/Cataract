import os
import shutil
import random
import re
import argparse
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd


def build_grading_mapping(grading_root):
    """
    Scan grading_1, grading_2, grading_3, grading_4 folders
    and build a mapping: normalized mp4_name -> grading level (str)
    ex: {"video22": "1", "video55": "3", ...}
    """
    mapping = {}
    for folder in os.listdir(grading_root):
        folder_path = os.path.join(grading_root, folder)
        if not os.path.isdir(folder_path):
            continue

        match = re.search(r'(\d+)', folder)
        if not match:
            continue
        level = match.group(1)

        for f in os.listdir(folder_path):
            if f.endswith(".mp4"):
                name = os.path.splitext(f)[0].lower().replace(" ", "").replace("_", "")
                mapping[name] = level

    return mapping


def extract_level(video_name, grading_mapping):
    """
    Find the grading level of a video by matching its name
    against the grading mapping.
    """
    normalized = video_name.lower().replace(" ", "").replace("_", "")
    for key, level in grading_mapping.items():
        if key in normalized or normalized in key:
            return level
    return "unknown"


def extract_phases(video_path):
    return [
        item for item in os.listdir(video_path)
        if os.path.isdir(os.path.join(video_path, item))
    ]


def build_video_phase_mapping(source_dir):
    video_phases = {}
    phase_to_videos = defaultdict(list)

    for video in tqdm(os.listdir(source_dir), desc="Scanning videos"):
        video_path = os.path.join(source_dir, video)

        if not os.path.isdir(video_path):
            continue

        phases = extract_phases(video_path)

        if not phases:
            continue

        video_phases[video] = set(phases)

        for phase in phases:
            phase_to_videos[phase].append(video)

    return video_phases, phase_to_videos


def smart_split(video_phases, phase_to_videos, seed, ratios):

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

        random.shuffle(videos)
        n = len(videos)

        if n == 1:
            train.add(videos[0])
            used_videos.add(videos[0])

        elif n == 2:
            train.add(videos[0])
            test.add(videos[1])
            used_videos.update(videos)

        elif n == 3:
            train.update(videos[:2])
            val.add(videos[2])
            used_videos.update(videos)

        else:
            n_train = int(ratios[0] * n)
            n_val = int(ratios[1] * n)

            train.update(videos[:n_train])
            val.update(videos[n_train:n_train + n_val])
            test.update(videos[n_train + n_val:])
            used_videos.update(videos)

    return {
        "train": train,
        "val": val,
        "test": test
    }


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
                    shutil.copy2(src_file, dst_file)

                    labels_dict[f"{split_name}/{video}/{file}"] = phase

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
    parser.add_argument("--grading_dir", type=str, required=True,
                        help="Folder containing grading_1, grading_2, grading_3, grading_4 subfolders")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio_split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument("--save_excel", action="store_true")

    args = parser.parse_args()

    grading_mapping = build_grading_mapping(args.grading_dir)

    video_phases, phase_to_videos = build_video_phase_mapping(args.source_dir)

    split = smart_split(
        video_phases,
        phase_to_videos,
        args.seed,
        args.ratio_split
    )

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
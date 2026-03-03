import os
import shutil
import random
import argparse
from collections import defaultdict, Counter

from tqdm import tqdm

def extract_level(video_name: str) -> str:
    if "VR_Senior_Fellow" in video_name:
        return "senior"
    elif "VR_junior_Fellow" in video_name:
        return "junior"
    elif "VR_Consultant" in video_name:
        return "consultant"
    else:
        return "unknown"


def extract_phases(video_path):
    phases = []
    for item in os.listdir(video_path):
        full_path = os.path.join(video_path, item)
        if os.path.isdir(full_path):
            phases.append(item)
    return phases


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

    print("\nApplying phase-aware smart split...")

    # On traite les phases dans l'ordre croissant d'apparition
    sorted_phases = sorted(
        phase_to_videos.keys(),
        key=lambda p: len(phase_to_videos[p])
    )

    for phase in tqdm(sorted_phases, desc="Processing phases"):

        videos = phase_to_videos[phase]
        videos = [v for v in videos if v not in used_videos]

        if not videos:
            continue

        random.shuffle(videos)
        n = len(videos)

        if n == 1:
            train.add(videos[0])
            used_videos.add(videos[0])

        elif n ==2:
            train.update(videos[:1])
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


def create_structure(dest_dir, all_phases):
    for split in ["train", "val", "test"]:
        for phase in all_phases:
            os.makedirs(
                os.path.join(dest_dir, split, phase),
                exist_ok=True
            )

def populate_dataset(source_dir, dest_dir, split, use_symlink):

    for split_name, videos in split.items():

        print(f"\nProcessing {split_name} set...")

        for video in tqdm(videos, desc=f"{split_name} videos"):

            video_path = os.path.join(source_dir, video)

            for phase in os.listdir(video_path):

                phase_path = os.path.join(video_path, phase)

                if not os.path.isdir(phase_path):
                    continue

                dest_phase_dir = os.path.join(dest_dir, split_name, phase)

                for file in os.listdir(phase_path):

                    src_file = os.path.join(phase_path, file)

                    new_name = f"{video}_{file}"
                    dst_file = os.path.join(dest_phase_dir, new_name)

                    if use_symlink:
                        if not os.path.exists(dst_file):
                            os.symlink(src_file, dst_file)
                    else:
                        shutil.copy(src_file, dst_file)


def analyze_split(split, video_phases):

    print("\n===== SPLIT SIZE =====")
    for s in split:
        print(f"{s}: {len(split[s])} videos")

    print("\n===== EXPERTISE DISTRIBUTION =====")
    for s in split:
        levels = [extract_level(v) for v in split[s]]
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
        description="Build phase-aware train/val/test dataset"
    )

    parser.add_argument("--source_dir", type=str, required=True, default='/home/helena/UCL_video_cataract/videos_matching_json_with_extracted_frames')
    parser.add_argument("--dest_dir", type=str, required=True, default='/home/helena/UCL_video_cataract')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_symlink", action="store_true",
                        help="Use symlinks instead of copying files")
    parser.add_argument("--ratio_split",type=float,nargs=3,default=[0.8, 0.1, 0.1],
                        help="Train/Val/Test split ratios (must sum to 1.0)")

    args = parser.parse_args()

    print("Building mappings...")
    video_phases, phase_to_videos = build_video_phase_mapping(args.source_dir)

    print("Splitting dataset...")
    split = smart_split(video_phases,phase_to_videos,args.seed,args.ratio_split)    

    all_phases = set()
    for phases in video_phases.values():
        all_phases.update(phases)

    print("Creating folder structure...")
    create_structure(args.dest_dir, all_phases)

    print("Populating dataset...")
    populate_dataset(
        args.source_dir,
        args.dest_dir,
        split,
        args.use_symlink
    )

    print("Analyzing split...")
    analyze_split(split, video_phases)

    print("\nDataset successfully created")


if __name__ == "__main__":
    main()
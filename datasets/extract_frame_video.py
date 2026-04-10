import argparse
import cv2
import json
import os
import pandas as pd
import re

from tqdm import tqdm


def extract_video_number_from_folder(folder_name):
    match = re.search(r'[Vv]ideo\s*(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None


def load_video_id_mapping(excel_path):
    df = pd.read_excel(excel_path)
    mapping = {}
    for _, row in df.iterrows():
        video_name = str(row["Videos"]).strip()
        raw_id = row["mp4 used in the json file"]

        if pd.isna(raw_id):
            print(f"Warning: no video_id for '{video_name}', skipping")
            continue

        match = re.search(r'\d+', video_name)
        if not match:
            print(f"Warning: can't extract number from '{video_name}', skipping")
            continue

        video_number = int(match.group())
        video_id = str(int(float(raw_id)))
        mapping[video_number] = video_id

    return mapping


def clean_string(s):
    s = s.strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_]", "", s)
    return s


def process_folder(folder_path, fps_target, mapping, corrupted_frames, remaining):
    folder_name = os.path.basename(folder_path)

    print(f"\n[{remaining} videos remaining] Processing: {folder_name}")

    video_file = None
    json_file = None

    for f in os.listdir(folder_path):
        if f.endswith(".mp4"):
            video_file = f
        if f.endswith(".json"):
            json_file = f

    if video_file is None or json_file is None:
        print(f"Missing video or json in {folder_path}")
        return

    video_number = extract_video_number_from_folder(folder_name)

    if video_number is None:
        print(f"Warning: can't extract video number from '{folder_name}', skipping")
        return

    if video_number not in mapping:
        print(f"Warning: Video {video_number} not found in Excel, skipping")
        return

    video_id = mapping[video_number]

    json_path = os.path.join(folder_path, json_file)
    video_path = os.path.join(folder_path, video_file)

    with open(json_path, "r") as f:
        data = json.load(f)

    if video_id not in data["file"]:
        print(f"Warning: video_id '{video_id}' not found in JSON for '{folder_name}', skipping")
        return

    metadata = data["metadata"]
    segments = []

    for meta in metadata.values():
        if meta["vid"] == video_id:
            z = meta.get("z", [])
            if len(z) != 2:
                print(f"Warning: malformed annotation in {folder_name} -> {z}")
                continue
            start, end = z
            phase = meta["av"]["1"]
            segments.append((start, end, phase))

    segments.sort(key=lambda x: x[0])

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps == 0:
        print(f"Error reading FPS for {video_file}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(original_fps / fps_target))
    video_name = clean_string(os.path.splitext(video_file)[0])

    for frame_idx in tqdm(range(total_frames), desc=f"{folder_name}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval != 0:
            continue

        current_time = frame_idx / original_fps

        for start, end, phase in segments:
            if start <= current_time <= end:
                safe_phase = clean_string(phase)
                phase_folder = os.path.join(folder_path, safe_phase)
                os.makedirs(phase_folder, exist_ok=True)

                filename = (
                    f"Video_{video_name}_"
                    f"Frame_{frame_idx:06d}_"
                    f"Phase_{safe_phase}.png"
                )
                save_path = os.path.join(phase_folder, filename)

                if os.path.exists(save_path):
                    continue

                success = cv2.imwrite(save_path, frame)
                if not success:
                    corrupted_frames.append({
                        "video": folder_name,
                        "frame_idx": frame_idx,
                        "time_sec": round(current_time, 2),
                        "phase": safe_phase,
                        "filename": filename
                    })

                break

    cap.release()
    print(f"Done: {folder_name}")


def save_corrupted_report(dataset_path, corrupted_frames):
    if not corrupted_frames:
        print("\nNo corrupted frames found.")
        return

    report_path = os.path.join(dataset_path, "corrupted_frames.csv")
    df = pd.DataFrame(corrupted_frames)
    df.to_csv(report_path, index=False)
    print(f"\n{len(corrupted_frames)} corrupted frames saved to {report_path}")

    print("\n===== CORRUPTED FRAMES PER VIDEO =====")
    print(df.groupby("video")["frame_idx"].count().to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from cataract surgery videos using VIA JSON annotations"
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--single_folder", type=str, default=None)
    parser.add_argument("--excel_path", type=str, required=True)

    args = parser.parse_args()

    mapping = load_video_id_mapping(args.excel_path)
    corrupted_frames = []

    if args.single_folder:
        folder_path = os.path.join(args.dataset_path, args.single_folder)
        process_folder(folder_path, args.fps, mapping, corrupted_frames, remaining=1)
    else:
        folders = [
            f for f in os.listdir(args.dataset_path)
            if os.path.isdir(os.path.join(args.dataset_path, f))
        ]
        total = len(folders)
        print(f"Found {total} folders to process")

        for i, folder in enumerate(folders):
            folder_path = os.path.join(args.dataset_path, folder)
            remaining = total - i
            process_folder(folder_path, args.fps, mapping, corrupted_frames, remaining)

    save_corrupted_report(args.dataset_path, corrupted_frames)


if __name__ == "__main__":
    main()

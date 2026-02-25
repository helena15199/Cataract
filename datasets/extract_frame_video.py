import argparse
import os
import json
import cv2
import re

from tqdm import tqdm

def extract_video_id_from_folder(folder_name):
    """
    If folder contains [number], return that number.
    Otherwise return "1".
    Works for any number: [2], [5], [12], etc.
    """
    match = re.search(r"\[(\d+)\]", folder_name)
    if match:
        return match.group(1)
    return "1"


def clean_string(s):
    """
    Clean string for safe filenames:
    - replace spaces with _
    - remove special characters
    """
    s = s.strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_]", "", s)
    return s


def process_folder(folder_path, fps_target):
    folder_name = os.path.basename(folder_path)

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

    video_id = extract_video_id_from_folder(folder_name)

    json_path = os.path.join(folder_path, json_file)
    video_path = os.path.join(folder_path, video_file)

    with open(json_path, "r") as f:
        data = json.load(f)

    if video_id not in data["file"]:
        raise ValueError(
            f"Video id {video_id} not found in JSON 'file' section for {folder_name}"
        )

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

    for frame_idx in tqdm(range(total_frames),desc=f"{folder_name}",leave=False):

        ret, frame = cap.read()
        if not ret:
            break

        # On ne traite que les frames correspondant à 5 fps
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

                cv2.imwrite(save_path, frame)
                break

    cap.release()
    print(f"Done: {folder_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from cataract surgery videos using VIA JSON annotations."
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset root folder"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Target FPS for frame extraction (default=5)"
    )

    args = parser.parse_args()

    dataset_path = args.dataset_path
    fps_target = args.fps

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            process_folder(folder_path, fps_target)
    
if __name__ == "__main__":
    main()
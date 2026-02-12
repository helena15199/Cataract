import os
import csv
import cv2
import argparse
import random
from collections import defaultdict


def load_phase_names(csv_file):
    phase_dict = {}
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            phase_id = int(row[0])
            name = row[1].strip().lower().replace(" ", "_")
            phase_dict[phase_id] = name
    return phase_dict


def load_video_info(csv_file):
    video_info = {}
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            vid = int(row[0])
            frames = int(row[1])
            fps = float(row[2])
            video_info[vid] = {"frames": frames, "fps": fps}
    return video_info


def load_annotations(csv_file):
    annotations = defaultdict(list)
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            vid = int(row[0])
            frame = int(row[1])
            phase = int(row[2])
            annotations[vid].append((frame, phase))

    for vid in annotations:
        annotations[vid].sort(key=lambda x: x[0])

    return annotations


def extract_frames_1fps(
    video_path,
    start_frame,
    end_frame,
    fps,
    output_dir,
    vid,
    phase_name
):

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return

    step = int(fps)  # 1 fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    index = 1

    while current_frame <= end_frame:

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            break

        filename = f"case_{vid}_{phase_name}_frame_{current_frame}_{index}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, frame)

        current_frame += step
        index += 1

    cap.release()


def main(args):

    phase_dict = load_phase_names(args.phases_csv)
    video_info = load_video_info(args.videos_csv)
    annotations = load_annotations(args.annotation_csv)

    video_ids = list(video_info.keys())

    random.seed(args.seed)
    random.shuffle(video_ids)

    n = len(video_ids)
    train_ids = video_ids[:int(0.8*n)]
    val_ids   = video_ids[int(0.8*n):int(0.9*n)]
    test_ids  = video_ids[int(0.9*n):]

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    for split_name, vids in splits.items():

        print(f"\nProcessing {split_name}...")

        for vid in vids:

            video_path = os.path.join(args.video_dir, f"{vid}.mp4")

            fps = video_info[vid]["fps"]
            total_frames = video_info[vid]["frames"]

            segments = annotations[vid]

            for i in range(len(segments)):

                start_frame, phase_id = segments[i]

                if i < len(segments) - 1:
                    end_frame = segments[i+1][0] - 1
                else:
                    end_frame = total_frames - 1

                phase_name = phase_dict[phase_id]

                output_dir = os.path.join(
                    args.output_dir,
                    split_name,
                    phase_name
                )

                extract_frames_1fps(
                    video_path,
                    start_frame,
                    end_frame,
                    fps,
                    output_dir,
                    vid,
                    phase_name
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation_csv", type=str, default="C:/Users/helen/Documents/UCL/video_cataract_101/annotations.csv", required=True)
    parser.add_argument("--phases_csv",type=str,default="C:/Users/helen/Documents/UCL/video_cataract_101/phases.csv")
    parser.add_argument("--videos_csv", type=str, default="C:/Users/helen/Documents/UCL/video_cataract_101/videos.csv")
    parser.add_argument("--video_dir", type=str, default="C:/Users/helen/Documents/UCL/video_cataract_101/videos")
    parser.add_argument("--output_dir", type=str, default="C:/Users/helen/Documents/UCL/dataset_cataract_101/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)

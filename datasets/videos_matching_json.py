import os
import shutil
import argparse

VIDEO_EXT = [".mp4", ".avi", ".mov", ".mkv"]

def normalize(name):
    """Normalize filename to make matching robust."""
    return name.lower().replace(" ", "").replace("_", "")


def collect_files(parent_dir):

    videos = {}
    jsons = {}

    for root, _, files in os.walk(parent_dir):

        for f in files:

            path = os.path.join(root, f)
            name, ext = os.path.splitext(f)
            key = normalize(name)

            if ext.lower() in VIDEO_EXT:
                videos[key] = path

            elif ext.lower() == ".json":
                jsons[key] = path

    return videos, jsons


def build_dataset(parent_dir, dest_dir):

    os.makedirs(dest_dir, exist_ok=True)

    videos, jsons = collect_files(parent_dir)

    matches = set(videos.keys()) & set(jsons.keys())

    for key in matches:

        video = videos[key]
        json_file = jsons[key]

        video_name = os.path.splitext(os.path.basename(video))[0]
        json_name = os.path.splitext(os.path.basename(json_file))[0]

        folder_name = f"{video_name}_json_{json_name}"
        folder_path = os.path.join(dest_dir, folder_name)

        os.makedirs(folder_path, exist_ok=True)

        shutil.copy(video, os.path.join(folder_path, os.path.basename(video)))
        shutil.copy(json_file, os.path.join(folder_path, os.path.basename(json_file)))

        print(f"Created: {folder_name}")

    print("\nSummary")
    print(f"Matched pairs: {len(matches)}")
    print(f"Videos without json: {len(videos) - len(matches)}")
    print(f"Json without video: {len(jsons) - len(matches)}")


def main():

    parser = argparse.ArgumentParser(
        description="Create dataset pairing videos with corresponding JSON annotations"
    )

    parser.add_argument(
        "--parent_dir",
        type=str,
        required=True,
        help="Parent directory containing multiple folders with videos and json files",
    )

    parser.add_argument(
        "--dest_dir",
        type=str,
        required=True,
        help="Destination dataset directory",
    )

    args = parser.parse_args()

    build_dataset(args.parent_dir, args.dest_dir)


if __name__ == "__main__":
    main()
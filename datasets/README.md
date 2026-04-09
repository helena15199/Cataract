# Cataract Phase Classification Dataset

This repository provides a complete pipeline to:

1. Extract frames from cataract surgery videos using VIA annotations  
2. Organize extracted frames by surgical phase  
3. Create a phase-aware train / validation / test split for temporal and non temporal models 

---

## Step 0 — Folder Structure

Your raw videos must be organized as:

```
dataset_raw/
├── Video_01/
│   ├── video.mp4
│   └── annotations.json
├── Video_02/
│   └── ...
```

Each folder contains **one video** and its **VIA JSON annotations**.
The JSON must include:
- `z = [start_time, end_time]` — segment timestamps
- `av["1"] = phase_name` — surgical phase label
- `vid = video_id` — video identifier


## Step 1 — Frame Extraction

Here the goal is to extract frames at a target FPS (default 5 FPS) and automatically organize them by surgical phase.

```bash
python extract_frame_video.py \
    --dataset_path /path/to/dataset_raw \
    --fps 5
```
Arguments :
- --dataset_path : root folder containing raw videos (each video in a separate folder)
- --fps : target FPS for frame extraction (default: 5)
- --single_folder (optional) : process only this folder

## Step 2 — Phase-Aware Dataset Split

To create the final dataset, two scripts can be used depending on whether you want a non-temporal or temporal organization of your data.

### Temporal Model (split_temporal.py)

This script prepares a dataset for models that use temporal sequences (video-level models):

- Video-level structure: each video’s frames are kept in order in the same folder, preserving the temporal sequence.

- Phase-aware splitting: ensures all surgical phases are represented in train, val, and test.

- Labels: a labels.json file is automatically created to assign a surgical phase label to each frame.

You can run this :

```bash
python split_temporal.py \
    --source_dir /path/to/extracted_frames \
    --dest_dir /path/to/dataset_cataract_temporal \
    --ratio_split 0.8 0.1 0.1 \
```
After running, the folder structure looks like this:

```
dataset_cataract_temporal/
├── train/
│   ├── Video_01/
│   │   ├── Frame_0001.jpg
│   │   ├── Frame_0002.jpg
│   │   └── ...
│   └── ...
├── val/
└── test/
```

### Non-Temporal Model (split_non_temporal.py)

This script prepares a dataset for frame-level models, where each frame is treated independently:

- Phase-aware splitting: assigns videos to train, val, and test sets, ensuring balanced representation of all surgical phases.

- Flattened structure: within each split, frames are copied into folders by phase.

You can run this :

```bash
python split_non_temporal.py \
    --source_dir /path/to/extracted_frames \
    --dest_dir /path/to/dataset_cataract \
    --ratio_split 0.8 0.1 0.1 \
```
After running, the folder structure looks like this:

```
dataset_cataract/
├── train/
│   ├── Incision/
│   │   ├── Video_01_Frame_0001.jpg
│   │   ├── Video_01_Frame_0002.jpg
│   │   └── ...
│   └── ...
├── val/
└── test/
```
--- 

Both scripts also print a summary of the dataset with the number of videos per split, the distribution of surgeon expertise (senior, junior, consultant) and the phase distribution across splits.

---

## Step 3 — Annotation Fix (IA / Capsule Polishing overlap)

Some videos contain frames double-annotated as both `Irrigation_and_aspiration` and `Capsule_polishing` simultaneously. This creates duplicate image files per frame and corrupts training labels.

`fix_annotation.py` resolves this for a given video by:
1. **Duplicate frames**: deletes the image file and JSON entry for the wrong phase.
2. **Alternating single-label frames**: reassigns the JSON label to the chosen phase.

Only `labels.json` and the duplicate image files are modified — no other files are touched. A timestamped backup of `labels.json` is created automatically before any change.

```bash
# Dry run — see what would change without modifying anything
python datasets/fix_annotation.py \
    --video "Video 32 (cat144)" \
    --phase "Capsule_polishing" \
    --dry_run

# Apply
python datasets/fix_annotation.py \
    --video "Video 32 (cat144)" \
    --phase "Capsule_polishing"
```

Arguments:
- `--video` : video name as it appears in `labels.json` (e.g. `"Video 32 (cat144)"`)
- `--phase` : phase to keep — either `Irrigation_and_aspiration` or `Capsule_polishing`
- `--dry_run` : print changes without applying them

**24 videos** are affected by this issue (across train/val/test splits). Run the script once per video after receiving the annotation decision from the clinical expert.
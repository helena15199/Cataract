# Cataract Phase Classification Dataset

This repository provides a complete pipeline to:

1. Extract frames from cataract surgery videos using VIA annotations  
2. Organize extracted frames by surgical phase  
3. Create a phase-aware train / validation / test split  

---

# Dataset Preparation Workflow

## Step 0 — Folder Structure

Your raw videos must be organized as:


dataset_raw/
├── Video_01/
│ ├── surgery.mp4
│ └── surgery.json
├── Video_02/
│ ├── surgery.mp4
│ └── surgery.json


- Each folder contains **one video** and its **VIA JSON annotations**.  
- The JSON must include:  
  - `z = [start_time, end_time]`  
  - `av["1"] = phase_name`  
  - `vid = video_id`  

---

## Step 1 — Frame Extraction

1) Objective

Extract frames at a target FPS (default 5 FPS) and automatically organize them by surgical phase.

2) Command

```bash
python extract_frame_video.py \
    --dataset_path /path/to/dataset_raw \
    --fps 5

Argument	Description
--dataset_path	Root folder containing raw videos (each video in a separate folder)
--fps	Target FPS for frame extraction (default: 5)
--single_folder	Optional. Process only this folder

Output structure:

dataset_raw/
└── Video_01/
    ├── Incision/
    ├── Rhexis/
    ├── Phacoemulsification/

### Step 2 — Phase-Aware Dataset Split

This script creates a train/validation/test dataset from videos annotated by surgical phase, ensuring balanced phase representation. Files can be copied or symlinked.

Features

Detects surgeon level from video name: senior, junior, consultant

Extracts phases from video subfolders

Builds video → phases and phase → videos mappings

Splits dataset intelligently per phase with configurable ratios

Creates folder structure and populates dataset

Prints split stats: number of videos, expertise distribution, phase distribution

Usage
python dataset_splitter.py \
    --source_dir /path/to/videos \
    --dest_dir /path/to/dataset \
    --seed 42 \
    --use_symlink \
    --ratio_split 0.8 0.1 0.1
Arguments
Argument	Type	Default	Description
--source_dir	str	required	Source folder containing videos with phase subfolders
--dest_dir	str	required	Destination folder for the dataset
--seed	int	42	Random seed for reproducibility
--use_symlink	flag	False	Use symlinks instead of copying files
--ratio_split	float float float	0.8 0.1 0.1	Train/Val/Test split ratios (sum=1.0)
Example Structure

Before:

videos/
├─ VR_Senior_Fellow_01/
│  ├─ Incision/
│  └─ Phacoemulsification/
├─ VR_junior_Fellow_02/
│  ├─ Incision/
│  └─ Lens_Implant/

After:

dataset/
├─ train/Incision/
├─ train/Phacoemulsification/
├─ val/Incision/
├─ test/Lens_Implant/

Files are renamed as:

<video_name>_<original_file>
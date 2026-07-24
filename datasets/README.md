# Cataract Phase Classification Dataset

This folder contains the full pipeline to go from raw surgery videos + VIA
annotations to the `dataset_temporal/` directory consumed by the training
code in `phases_recognition/`.

Pipeline overview:

```
raw videos + VIA json  →  Step 0  →  paired folders
                        →  Step 1  →  extracted frames (by phase)
                        →  Step 2  →  dataset_temporal/ (train/val/test + labels.json)
                        →  Step 3  →  annotation fixes (IA/Capsule_polishing overlap)
```

Two inputs are required throughout and are **not derivable from anything
else** — keep them safe:
- The raw videos + their VIA JSON annotations (one JSON per video).
- `analysis_data_Ayushi.xlsx` — one row per video, with at least the columns
  `Videos`, `mp4 used in the json file` and `grading`. Used to map video
  folder names to the video id used inside the VIA JSON, and to attach a
  grading/expertise label to each video.

Everything else (`labels.json`, extracted frames, `dataset_temporal/`) is
regenerated from those two inputs and can be rebuilt if lost.

---

## Step 0 — Pair raw videos with their VIA JSON

```bash
python datasets/videos_matching_json.py \
    --parent_dir /path/to/raw_videos_and_jsons \
    --dest_dir   /path/to/dataset_raw
```

Scans `--parent_dir` recursively, matches video files (`.mp4`, `.avi`,
`.mov`, `.mkv`) with `.json` files that share the same normalized filename,
and copies each matched pair into its own folder:

```
dataset_raw/
├── Video_01_json_Video_01/
│   ├── Video_01.mp4
│   └── Video_01.json
├── Video_02_json_Video_02/
│   └── ...
```

Prints a summary of matched pairs, videos without a JSON, and JSONs without
a video — check this output, unmatched files are silently skipped.

## Step 1 — Frame extraction

```bash
python datasets/extract_frame_video.py \
    --dataset_path /path/to/dataset_raw \
    --excel_path   /path/to/analysis_data_Ayushi.xlsx \
    --fps 5
```

Arguments:
- `--dataset_path` : root folder from Step 0 (one subfolder per video)
- `--excel_path` : `analysis_data_Ayushi.xlsx`, used to map the video folder
  number to the video id referenced inside the VIA JSON (`mp4 used in the
  json file` column)
- `--fps` : target FPS for frame extraction (default: 5)
- `--single_folder` (optional) : process only this one folder

Each frame is saved as `Video_<name>_Frame_<idx>_Phase_<phase>.png` inside a
per-phase subfolder of the video's folder. Frames that fail to write are
logged to `corrupted_frames.csv` in `--dataset_path`.

## Step 2 — Phase-aware temporal split

```bash
python datasets/split_temporal.py \
    --source_dir /path/to/dataset_raw \
    --dest_dir   /path/to/dataset_temporal \
    --excel_path /path/to/analysis_data_Ayushi.xlsx \
    --ratio_split 0.8 0.1 0.1
```

- Video-level structure: each video's frames stay together and in temporal
  order, split across `train/`, `val/`, `test/`.
- Phase-aware splitting: every phase gets at least one video in `train`
  (`fix_missing_train_phases`), and every grading/expertise level present in
  the data is represented in both `val` and `test` (`fix_grading_coverage`).
- `--excel_path` supplies the `grading` column used for the expertise checks
  above.
- Writes `labels.json` at the root of `--dest_dir`, mapping every frame path
  (`split/video_folder/filename.png`) to its (normalized) phase. This is the
  file the training/eval code actually reads — everything downstream in
  `phases_recognition/` depends on it.
- `--save_excel` additionally writes `labels.xlsx` (same content as
  `labels.json`, easier to skim).

Resulting structure:

```
dataset_temporal/
├── labels.json
├── train/
│   ├── Video_01_json_Video_01/
│   │   ├── Frame_000001_Phase_Incision.png
│   │   └── ...
│   └── ...
├── val/
└── test/
```

Prints split sizes, grading distribution, and phase distribution per split
— check this before training, especially for rare phases.

## Step 3 — Annotation fixes (Irrigation_and_aspiration / Capsule_polishing overlap)

Some videos have frames double-annotated as both `Irrigation_and_aspiration`
and `Capsule_polishing` (duplicate files for the same frame number), or a
rapid back-and-forth between the two labels around the transition. Two
scripts handle this, always in this order:

1. **Diagnose** — list which videos/frames are affected:
   ```bash
   python datasets/detect_alternating.py
   python datasets/detect_alternating.py --video "Video 32 (cat144)"
   ```
2. **Fix** — after getting the clinical expert's decision on which phase to
   keep for a given video:
   ```bash
   # Dry run first
   python datasets/fix_annotation.py \
       --video "Video 32 (cat144)" \
       --phase "Capsule_polishing" \
       --dry_run

   # Apply
   python datasets/fix_annotation.py \
       --video "Video 32 (cat144)" \
       --phase "Capsule_polishing"
   ```
   - Duplicate frames (two files, two JSON entries): deletes the file + JSON
     entry for the phase **not** passed via `--phase`.
   - Alternating single-label frames: reassigns the JSON label to `--phase`
     (file is kept, only the label changes). Add `--alternating_only` to
     restrict this to the alternating zone before the last stable frame of
     `--phase`, leaving a stable trailing section of the other phase
     untouched.
   - `labels.json` is backed up (timestamped) before any change.

As of writing, 24 videos across train/val/test are known to be affected.
Run `fix_annotation.py` once per affected video after receiving the
annotation decision from the clinical expert.

---

## Maintenance

**`sync_labels_from_files.py`** — makes `labels.json` match whatever `.png`
files actually exist on disk (adds missing entries, fixes wrong labels from
the filename, removes stale entries for deleted files). Use this if
`labels.json` and the frame files on disk have drifted apart (e.g. after a
manual file deletion/move).

```bash
python datasets/sync_labels_from_files.py --dry_run
python datasets/sync_labels_from_files.py
python datasets/sync_labels_from_files.py --video "Video 32 (cat144)"
```

**`analyze_dataset.ipynb`** — exploration notebook, reads
`analysis_data_Ayushi.xlsx` directly (`EXCEL_PATH` constant at the top) for
grading/phase distribution analysis outside of the split scripts.

---

## Notes for reproducing on a new machine

- `detect_alternating.py`, `fix_annotation.py` and `sync_labels_from_files.py`
  currently hardcode `DATASET_ROOT` / `LABELS_PATH` to
  `/home/helena/UCL_video_cataract/dataset_temporal/` at the top of each
  file — update these constants (or the `EXCEL_PATH` constant in
  `analyze_dataset.ipynb`) if you're not on the original machine.
- There can be more than one `labels*.json` in `dataset_temporal/` (e.g. an
  `_ood` variant used for out-of-distribution experiments). These do **not**
  duplicate videos or frames — they only re-assign some videos to a
  different split. Because the physical frame folders live under a single
  fixed split, loading an alternate labels file against the "wrong" root
  will either silently drop the reassigned videos or raise
  `FileNotFoundError`, depending on the dataset class used in
  `phases_recognition/dataset/`. Check which labels file a given experiment
  config expects (`dataset.labels_file` in the training config) before
  assuming `labels.json` is the one in use.

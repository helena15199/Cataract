# Cataract Phases Recognition

Frame-level surgical phase classification using a pretrained CNN backbone + MLP head.

---

## Installation

```bash
uv sync
```

---

## Folder Structure

```
phases_recognition/
├── configs/config.yaml       # Main config (model, dataset, training, metrics)
├── dataset/                  # Dataset class and dataloader factory
├── losses/                   # Cross-entropy loss with label smoothing
├── metrics/                  # Accuracy, F1, precision, recall, AUROC per class
├── models/                   # CataractPredictor (backbone + MLP head)
├── utils/                    # Trainer, LR scheduler, visualizer, preprocessing
├── train.py                  # Training entry point
└── evaluate.py               # Evaluation entry point
```

---

## Config

All hyperparameters live in `configs/config.yaml`. Key sections:

| Section | Description |
|---|---|
| `model` | Backbone name (`resnet50`), number of classes, freeze flag |
| `dataset` | `class_names`, `others_classes` (remapped to *Others* during training), train/val roots, augmentations |
| `loss` | `label_smoothing` |
| `optimizer` | Separate `lr` for the head and `backbone_lr` for the backbone (AdamW) |
| `lr_scheduler` | Linear warmup + cosine decay |
| `metrics` | Classes excluded from evaluation metrics (e.g. `Others`) |
| `train` | Epochs, device, AMP, gradient clipping, checkpoint frequency |

### Classes

The 13 evaluated surgical phases are:

`Capsule_polishing`, `Corneal_hydration`, `Hydrodissection`, `Incision`,
`Irrigation_and_aspiration`, `Lens_implant_settingup`, `Phacoemulsification`,
`Rhexis`, `Tonifying_and_antibiotics`, `Viscous_agent_injection`,
`Viscous_agent_removal`, `Wound_hydration`, `Others`

Rare phases (`Malyugin_ring_insertion`, `Malyugin_ring_removal`, `Suture`, `Trypan_blue_injection`) are remapped to `Others` at load time and excluded from metrics.

---

## Model Architecture

`CataractPredictor` = pretrained backbone (e.g. ResNet50) + MLP classification head:

```
backbone → Linear(2048, 512) → ReLU → Dropout(0.5) → Linear(512, num_classes)
```

---

## Training

```bash
cd phases_recognition
uv run python train.py --config configs/config.yaml
```

Checkpoints and TensorBoard logs are saved under `root_dir/experiment_name/`.
The best checkpoint is saved as `best.pt`.

---

## Evaluation

```bash
cd phases_recognition
uv run python evaluate.py \
    --config configs/config.yaml \
    --ckpt /path/to/best.pt \
    --out_dir /path/to/results
```

The test set is automatically inferred from the val root (`.../val/` → `.../test/`).

### Outputs

| File | Description |
|---|---|
| `metrics.json` | Global metrics (accuracy, F1, precision, recall, AUROC, ECE) and per-class F1 |
| `confusion_matrix.png` | Normalized confusion matrix (Others excluded) |
| `confusion_matrix_counts.png` | Raw counts confusion matrix |
| `confidence_histogram.png` | Distribution of max softmax confidence, correct vs incorrect |
| `phase_timeline.png` | Per-video timeline: GT (top bar) vs predicted phases (bottom bar) |

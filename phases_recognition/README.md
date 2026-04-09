# Cataract Phases Recognition

Surgical phase recognition pipeline with two stages:
1. **Frame-level** : ResNet50 backbone trained independently on each frame
2. **Temporal** : MS-TCN++ / ASFormer / TeCNO on pre-extracted ResNet features

---

## Installation

```bash
uv sync
```

---

## Folder Structure

```
phases_recognition/
├── configs/
│   ├── config.yaml            # ResNet frame-level training
│   ├── config_mstcn.yaml      # MS-TCN++ temporal model
│   ├── config_asformer.yaml   # ASFormer temporal model
│   └── config_tecno.yaml      # LSTM-TCN temporal model
├── dataset/                   # Dataset classes (frame-level + feature-level)
├── losses/                    # CE / focal loss + TMSE smoothness loss
├── metrics/                   # Accuracy, F1, edit score, F1@{10,25,50}
├── models/
│   ├── cataract_predictor.py  # ResNet + MLP head
│   ├── mstcn.py               # MS-TCN++ (DualDilated refinement stages)
│   ├── asformer.py            # ASFormer (Transformer + positional encoding)
│   └── lstm_tcn.py            # LSTM + TCN refinement stages
├── train.py                   # ResNet frame-level training
├── evaluate.py                # ResNet frame-level evaluation
├── extract_features.py        # Extract ResNet features → .npy files
├── train_temporal.py          # MS-TCN++ / ASFormer / TeCNO training
├── evaluate_temporal.py       # Temporal model evaluation (raw + smoothed)
├── train_e2e.py               # End-to-end ResNet + TCN joint training
├── evaluate_e2e.py            # End-to-end evaluation
└── sweep_mstcn.py             # Hyperparameter sweep (λ, τ, L, γ)
```

---

## Pipeline

### Stage 1 — ResNet frame-level training

```bash
python train.py --config configs/config.yaml
```

### Stage 2 — Feature extraction

```bash
python extract_features.py --config configs/config.yaml
```

Saves `{video}.npy` (T, 2048) and `{video}_labels.npy` (T,) per video in `features/train|val|test/`.

### Stage 3 — Temporal model training

```bash
# MS-TCN++ (best sweep result: λ=0.05, τ=4, L=10)
python train_temporal.py --config configs/config_mstcn.yaml

# ASFormer
python train_temporal.py --config configs/config_asformer.yaml
```

### Stage 3 (alternative) — End-to-end joint training

```bash
python train_e2e.py --config configs/config_e2e.yaml
```

---

## Hyperparameter Sweep

```bash
python sweep_mstcn.py \
    --lambdas 0.05 0.15 0.25 \
    --taus 4 \
    --layers 8 10 12 \
    --gammas 0.0 2.0
```

Results saved to `{root_dir}/sweep_results.csv`, ranked by val F1 macro.
Best config found: **λ=0.05, τ=4, L=10, γ=0** → val F1_macro=0.703.

---

## Models

| Model | Architecture | Key advantage |
|---|---|---|
| MS-TCN++ | Dilated conv (dilation 1→512) + DualDilated refinement | Fast, well-studied |
| ASFormer | Transformer encoder + cross-attention decoders | Global temporal context, positional encoding |
| LSTM-TCN | Bidirectional LSTM + TCN refinement | Sequential surgical state memory |

All temporal models share the same training script (`train_temporal.py`) and loss (`MSTCNLoss`: CE + λ·TMSE). Input: pre-extracted ResNet50 features (2048-dim per frame).

---

## Config

### ResNet (`config.yaml`)

| Section | Description |
|---|---|
| `model` | Backbone name (`resnet50`), number of classes, freeze flag |
| `dataset` | `class_names`, train/val roots, augmentations |
| `loss` | `label_smoothing` |
| `optimizer` | Separate `lr` for head and `backbone_lr` for backbone (AdamW) |
| `train` | Epochs, device, AMP, gradient clipping |

### Temporal models (`config_mstcn.yaml`, etc.)

| Section | Description |
|---|---|
| `model` | Architecture params (num_stages, num_layers, d_model, dropout…) |
| `loss` | `lambda_smoothing`, `tau`, `label_smoothing`, `use_class_weights`, `focal_gamma` |
| `optimizer` | AdamW lr + weight_decay |
| `train` | Epochs, feature_noise_std |

### Classes

The 13 surgical phases:

`Capsule_polishing`, `Corneal_hydration`, `Hydrodissection`, `Incision`,
`Irrigation_and_aspiration`, `Lens_implant_settingup`, `Phacoemulsification`,
`Rhexis`, `Tonifying_and_antibiotics`, `Viscous_agent_injection`,
`Viscous_agent_removal`, `Wound_hydration`, `Others`

---

## Evaluation

### Temporal model
```bash
python evaluate_temporal.py \
    --config configs/config_mstcn.yaml \
    --ckpt /path/to/best.pt \
    --out_dir /path/to/results/
```

### End-to-end model
```bash
python evaluate_e2e.py \
    --config configs/config_e2e.yaml \
    --ckpt /path/to/best.pt \
    --out_dir /path/to/results/
```

### Outputs

| File | Description |
|---|---|
| `metrics.json` | Frame-level + segment-level metrics (edit score, F1@{10,25,50}), raw and smoothed |
| `confusion_matrix.png` | Normalized confusion matrix (raw predictions) |
| `confusion_matrix_smoothed.png` | Confusion matrix after majority-vote smoothing |
| `phase_timeline.png` | Per-video: GT / Raw / Smoothed phase bars |
| `per_video_f1.png` | Per-video F1 comparison raw vs smoothed |

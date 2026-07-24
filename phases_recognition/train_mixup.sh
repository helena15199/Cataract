#!/bin/bash
# Train Baseline and TCL β=0.02 with soft-label Mixup (α=0.2).

set -e
cd /home/helena/Cataract

echo "=========================================="
echo "Run 1/2 — Baseline + Mixup α=0.2"
echo "=========================================="

python phases_recognition/train_temporal.py \
    --config phases_recognition/configs/config_mstcn_dino.yaml \
    --override \
        experiment_name=baseline_mstcn_dino_mixup \
        train.mixup_alpha=0.2

echo ""
echo "=========================================="
echo "Run 2/2 — TCL β=0.02 + Mixup α=0.2"
echo "=========================================="

python phases_recognition/train_temporal.py \
    --config phases_recognition/configs/config_mstcn_dino_tcl.yaml \
    --override \
        experiment_name=mstcn_dino_tcl_beta0.02_mixup \
        train.tcl_beta=0.02 \
        train.mixup_alpha=0.2

echo ""
echo "All done."

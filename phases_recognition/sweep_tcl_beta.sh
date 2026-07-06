#!/bin/bash
# Sweep TCL beta: find the sweet spot where OOD improves without breaking closed-set
# Beta values: 0.01, 0.02, 0.05 (0.1 already done, too aggressive)

set -e
cd /home/helena/Cataract

for BETA in 0.02 0.05; do
    echo "=========================================="
    echo "Training with tcl_beta=${BETA}"
    echo "=========================================="

    python phases_recognition/train_temporal.py \
        --config phases_recognition/configs/config_mstcn_dino_tcl.yaml \
        --override experiment_name=mstcn_dino_tcl_beta${BETA} train.tcl_beta=${BETA} train.early_stopping_patience=20

    echo "Done with beta=${BETA}"
    echo ""
done

echo "All sweeps complete."

#!/bin/bash
# Fine-tune TCL models: freeze all except output_proj of last refinement stage.
# Only the linear classification head (64 → num_classes) is updated,
# preserving the 64-dim feature space shaped by TCL.

set -e
cd /home/helena/Cataract

CKPTS=(
    "0.01:/home/helena/experiments_cataract/mstcn_dino_tcl_beta0.01_date=2026_06_29_13_50_35/ckpt/best.pt"
    "0.02:/home/helena/experiments_cataract/mstcn_dino_tcl_beta0.02_date=2026_06_29_16_32_06/ckpt/best.pt"
    "0.05:/home/helena/experiments_cataract/mstcn_dino_tcl_beta0.05_date=2026_06_29_16_44_54/ckpt/best.pt"
)

for entry in "${CKPTS[@]}"; do
    BETA="${entry%%:*}"
    CKPT="${entry#*:}"

    echo "=========================================="
    echo "Fine-tuning TCL beta=${BETA} — last proj only"
    echo "=========================================="

    python phases_recognition/train_temporal.py \
        --config phases_recognition/configs/config_mstcn_dino_tcl.yaml \
        --resume "${CKPT}" \
        --override \
            experiment_name=mstcn_dino_tcl_beta${BETA}_ft \
            freeze_except_last_proj=true \
            train.tcl_beta=0.0 \
            train.epochs=80 \
            train.early_stopping_patience=20 \
            train.mixup_alpha=0.0 \
            optimizer.params.lr=5e-4 \
            lr_scheduler.params.n_warmup_steps=0 \
            lr_scheduler.params.n_total_steps=5200

    echo "Done with beta=${BETA}"
    echo ""
done

echo "All fine-tuning complete."

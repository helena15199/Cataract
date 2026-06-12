"""Hyperparameter sweep for the DINOv2 temporal transformer.

Grid:
    hidden_dim  × num_layers × dropout × train_crop_len

All other settings come from config_dino_transformer.yaml.
Already-completed runs (best.pt exists) are skipped automatically.
A CSV summary is written after every completed run.

Usage (from repo root):
    python phases_recognition/sweep_transformer.py [--dry_run]

    # custom grid
    python phases_recognition/sweep_transformer.py \
        --hidden_dims 64 128 256 \
        --layers 1 2 \
        --dropouts 0.1 0.2 0.3 \
        --crops 256 512
"""

import argparse
import copy
import csv
import pathlib
import sys

from loguru import logger
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Default grid
# ---------------------------------------------------------------------------

DEFAULT_HIDDEN       = [64, 128]
DEFAULT_LAYERS       = [1, 2]
DEFAULT_DROPOUT      = [0.3, 0.4]
DEFAULT_CROPS        = [0]
DEFAULT_WEIGHT_DECAY = [1e-2, 1e-1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exp_name(base: str, hidden: int, layers: int, dropout: float, crop: int, wd: float) -> str:
    return f"{base}__h{hidden}_L{layers}_do{dropout:.2f}_wd{wd:.0e}"


def _best_ckpt(root_dir: str, exp_name: str) -> pathlib.Path | None:
    for candidate in sorted(pathlib.Path(root_dir).glob(f"{exp_name}*/ckpt/best.pt")):
        return candidate
    return None


def _read_metrics(ckpt_path: pathlib.Path) -> dict:
    import torch
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return state.get("metric_dict", {})


def _append_csv(csv_path: pathlib.Path, row: dict):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _make_row(exp_name, hidden, layers, dropout, wd, metrics, failed=False):
    return {
        "experiment":   exp_name,
        "hidden_dim":   hidden,
        "num_layers":   layers,
        "dropout":      dropout,
        "weight_decay": wd,
        "failed":       failed,
        "val_f1_macro": round(metrics.get("global/f1_macro", float("nan")), 4),
        "val_accuracy": round(metrics.get("global/accuracy", float("nan")), 4),
    }


def _print_summary(csv_path: pathlib.Path):
    if not csv_path.exists():
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    def _safe_f1(r):
        try:
            return float(r.get("val_f1_macro") or 0)
        except (ValueError, TypeError):
            return 0.0

    valid = [r for r in rows if _safe_f1(r) > 0]
    if not valid:
        return

    rows_sorted = sorted(valid, key=_safe_f1, reverse=True)
    print(f"\n{'Rank':<5} {'hidden':<8} {'layers':<8} {'dropout':<10} {'wd':<10} {'F1_macro':<10} {'Accuracy'}")
    print("-" * 65)
    for rank, r in enumerate(rows_sorted, 1):
        print(
            f"{rank:<5} {r.get('hidden_dim','?'):<8} {r.get('num_layers','?'):<8} "
            f"{r.get('dropout','?'):<10} {r.get('weight_decay','?'):<10} "
            f"{r['val_f1_macro']:<10} {r.get('val_accuracy','?')}"
        )


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------

def run_sweep(
    base_config_path: str,
    hidden_dims: list[int],
    layers: list[int],
    dropouts: list[float],
    crops: list[int],
    weight_decays: list[float],
    dry_run: bool = False,
):
    base_cfg = OmegaConf.load(base_config_path)
    base_exp = base_cfg.experiment_name
    root_dir = base_cfg.root_dir
    csv_path = pathlib.Path(root_dir) / "sweep_transformer_results.csv"

    grid = [
        (h, L, do, wd)
        for h  in hidden_dims
        for L  in layers
        for do in dropouts
        for wd in weight_decays
    ]

    logger.info(f"Sweep grid: {len(grid)} experiments")
    logger.info(f"  hidden_dim   ∈ {hidden_dims}")
    logger.info(f"  num_layers   ∈ {layers}")
    logger.info(f"  dropout      ∈ {dropouts}")
    logger.info(f"  weight_decay ∈ {weight_decays}")
    logger.info(f"  Base config: {base_config_path}")
    logger.info(f"  Results CSV: {csv_path}")

    if dry_run:
        print("\n--- DRY RUN: experiment list ---")
        for i, (h, L, do, wd) in enumerate(grid):
            name = _exp_name(base_exp, h, L, do, 0, wd)
            done = _best_ckpt(root_dir, name) is not None
            print(f"  [{i+1:2d}/{len(grid)}] [{'DONE' if done else 'TODO'}] {name}")
        return

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from train_temporal import main as train_main

    for i, (h, L, do, wd) in enumerate(grid):
        exp_name = _exp_name(base_exp, h, L, do, 0, wd)

        existing = _best_ckpt(root_dir, exp_name)
        if existing is not None:
            logger.info(f"[{i+1}/{len(grid)}] SKIP (already done): {exp_name}")
            metrics = _read_metrics(existing)
            _append_csv(csv_path, _make_row(exp_name, h, L, do, wd, metrics))
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"[{i+1}/{len(grid)}] START: {exp_name}")
        logger.info(f"  hidden={h}  layers={L}  dropout={do}  weight_decay={wd}")
        logger.info(f"{'='*70}\n")

        cfg = copy.deepcopy(base_cfg)
        cfg.experiment_name             = exp_name
        cfg.model.hidden_dim            = int(h)
        cfg.model.num_layers            = int(L)
        cfg.model.dropout               = float(do)
        cfg.optimizer.params.weight_decay = float(wd)

        try:
            train_main(cfg)
        except Exception as e:
            logger.error(f"Run {exp_name} FAILED: {e}")
            _append_csv(csv_path, _make_row(exp_name, h, L, do, wd, {}, failed=True))
            continue

        best = _best_ckpt(root_dir, exp_name)
        metrics = _read_metrics(best) if best else {}
        _append_csv(csv_path, _make_row(exp_name, h, L, do, wd, metrics))
        logger.info(f"[{i+1}/{len(grid)}] DONE — val f1_macro={metrics.get('global/f1_macro', float('nan')):.4f}")

    logger.info(f"\nSweep complete. Results: {csv_path}")
    _print_summary(csv_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameter sweep — DINOv2 temporal transformer")
    parser.add_argument("--config",      type=str,   default="phases_recognition/configs/config_dino_transformer.yaml")
    parser.add_argument("--hidden_dims",   type=int,   nargs="+", default=DEFAULT_HIDDEN)
    parser.add_argument("--layers",        type=int,   nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--dropouts",      type=float, nargs="+", default=DEFAULT_DROPOUT)
    parser.add_argument("--crops",         type=int,   nargs="+", default=DEFAULT_CROPS)
    parser.add_argument("--weight_decays", type=float, nargs="+", default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--dry_run",       action="store_true")
    args = parser.parse_args()

    run_sweep(
        base_config_path=args.config,
        hidden_dims=args.hidden_dims,
        layers=args.layers,
        dropouts=args.dropouts,
        crops=args.crops,
        weight_decays=args.weight_decays,
        dry_run=args.dry_run,
    )

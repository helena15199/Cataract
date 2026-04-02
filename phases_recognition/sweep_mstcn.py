"""Hyperparameter sweep for MS-TCN++ on pre-extracted features.

Sweeps over λ (lambda_smoothing), τ (tau), and L (num_layers).
All other settings come from the base config (config_mstcn.yaml).

Each combination is run sequentially on the same GPU.
Already-completed runs (best.pt exists) are skipped automatically.

A CSV summary is written after every completed run to:
    {root_dir}/sweep_results.csv

Usage (from repo root):
    python phases_recognition/sweep_mstcn.py \
        [--config phases_recognition/configs/config_mstcn.yaml] \
        [--lambdas 0.05 0.15 0.25 0.30] \
        [--taus 3 4 5] \
        [--layers 8 10 12] \
        [--dry_run]   # print experiment list without training
"""

import argparse
import copy
import csv
import pathlib
import sys

from loguru import logger
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

DEFAULT_LAMBDAS = [0.05, 0.15, 0.25, 0.30]
DEFAULT_TAUS    = [3, 4]          # τ=5 crushes everything per paper — skip unless needed
DEFAULT_LAYERS  = [8, 10, 12]
DEFAULT_GAMMAS  = [0.0]           # 0 = standard CE; add 1.0 or 2.0 to test focal loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exp_name(base_name: str, lam: float, tau: float, L: int, gamma: float) -> str:
    gamma_str = f"_focal{gamma:.1f}" if gamma > 0 else ""
    return f"{base_name}__lam{lam:.2f}_tau{tau}_L{L}{gamma_str}"


def _best_ckpt(root_dir: str, exp_name: str) -> pathlib.Path | None:
    p = pathlib.Path(root_dir)
    # instantiate_dirs creates  root_dir / exp_name_date=... / ckpt/best.pt
    # We glob for any matching prefix
    for candidate in sorted(p.glob(f"{exp_name}*/ckpt/best.pt")):
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


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------

def run_sweep(
    base_config_path: str,
    lambdas: list[float],
    taus: list[float],
    layers: list[int],
    gammas: list[float],
    dry_run: bool = False,
):
    base_cfg = OmegaConf.load(base_config_path)
    base_exp  = base_cfg.experiment_name
    root_dir  = base_cfg.root_dir
    csv_path  = pathlib.Path(root_dir) / "sweep_results.csv"

    # Build full grid
    grid = [
        (lam, tau, L, gamma)
        for lam   in lambdas
        for tau   in taus
        for L     in layers
        for gamma in gammas
    ]

    logger.info(f"Sweep grid: {len(grid)} experiments")
    logger.info(f"  λ     ∈ {lambdas}")
    logger.info(f"  τ     ∈ {taus}")
    logger.info(f"  L     ∈ {layers}")
    logger.info(f"  γ     ∈ {gammas}  (focal loss; 0=standard CE)")
    logger.info(f"  Base config: {base_config_path}")
    logger.info(f"  Results CSV: {csv_path}")

    if dry_run:
        print("\n--- DRY RUN: experiment list ---")
        for i, (lam, tau, L, gamma) in enumerate(grid):
            name = _exp_name(base_exp, lam, tau, L, gamma)
            done = _best_ckpt(root_dir, name) is not None
            status = "DONE" if done else "TODO"
            print(f"  [{i+1:2d}/{len(grid)}] [{status}] {name}")
        return

    # Import training main lazily (keeps sweep importable without heavy deps)
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from train_temporal import main as train_main

    for i, (lam, tau, L, gamma) in enumerate(grid):
        exp_name = _exp_name(base_exp, lam, tau, L, gamma)

        # --- Skip if already done ---
        existing = _best_ckpt(root_dir, exp_name)
        if existing is not None:
            logger.info(f"[{i+1}/{len(grid)}] SKIP (already done): {exp_name}")
            metrics = _read_metrics(existing)
            row = _make_row(exp_name, lam, tau, L, gamma, metrics)
            _append_csv(csv_path, row)
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"[{i+1}/{len(grid)}] START: {exp_name}")
        logger.info(f"  λ={lam}  τ={tau}  L={L}  γ={gamma}")
        logger.info(f"{'='*70}\n")

        # Build modified config
        cfg = copy.deepcopy(base_cfg)
        cfg.experiment_name          = exp_name
        cfg.loss.lambda_smoothing    = float(lam)
        cfg.loss.tau                 = float(tau)
        cfg.loss.focal_gamma         = float(gamma)
        cfg.model.num_layers         = int(L)

        try:
            train_main(cfg)
        except Exception as e:
            logger.error(f"Run {exp_name} FAILED: {e}")
            row = _make_row(exp_name, lam, tau, L, gamma, {}, failed=True)
            _append_csv(csv_path, row)
            continue

        # Read best val metrics from the saved checkpoint
        best = _best_ckpt(root_dir, exp_name)
        metrics = _read_metrics(best) if best else {}
        row = _make_row(exp_name, lam, tau, L, gamma, metrics)
        _append_csv(csv_path, row)

        logger.info(f"[{i+1}/{len(grid)}] DONE: {exp_name}")
        logger.info(f"  val f1_macro={metrics.get('global/f1_macro', float('nan')):.4f}")

    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info(f"Sweep complete. Results: {csv_path}")
    _print_summary(csv_path)


def _make_row(exp_name, lam, tau, L, gamma, metrics, failed=False):
    return {
        "experiment":        exp_name,
        "lambda":            lam,
        "tau":               tau,
        "L":                 L,
        "focal_gamma":       gamma,
        "failed":            failed,
        "val_f1_macro":      round(metrics.get("global/f1_macro",   float("nan")), 4),
        "val_accuracy":      round(metrics.get("global/accuracy",   float("nan")), 4),
        "val_f1_Others":     round(metrics.get("per_class/f1/Others", float("nan")), 4),
        "val_f1_IA":         round(metrics.get("per_class/f1/Irrigation_and_aspiration", float("nan")), 4),
        "val_f1_Corneal":    round(metrics.get("per_class/f1/Corneal_hydration", float("nan")), 4),
    }


def _print_summary(csv_path: pathlib.Path):
    if not csv_path.exists():
        return
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("failed") != "True"]

    if not rows:
        return

    rows_sorted = sorted(rows, key=lambda r: float(r.get("val_f1_macro") or 0), reverse=True)

    print(f"\n{'Rank':<5} {'λ':<6} {'τ':<4} {'L':<4} {'γ':<5} {'F1_macro':<10} {'Acc':<8} {'F1_IA':<8} {'F1_Corneal'}")
    print("-" * 72)
    for rank, r in enumerate(rows_sorted[:10], 1):
        print(
            f"{rank:<5} {r['lambda']:<6} {r['tau']:<4} {r['L']:<4} "
            f"{r.get('focal_gamma', 0):<5} "
            f"{r['val_f1_macro']:<10} {r['val_accuracy']:<8} "
            f"{r['val_f1_IA']:<8} {r['val_f1_Corneal']}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MS-TCN++ hyperparameter sweep")
    parser.add_argument("--config",  type=str,
                        default="phases_recognition/configs/config_mstcn.yaml")
    parser.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument("--taus",    type=float, nargs="+", default=DEFAULT_TAUS)
    parser.add_argument("--layers",  type=int,   nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--gammas",  type=float, nargs="+", default=DEFAULT_GAMMAS,
                        help="Focal loss gamma values. 0=standard CE, 2=typical focal.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print experiment list without running anything")
    args = parser.parse_args()

    run_sweep(
        base_config_path = args.config,
        lambdas          = args.lambdas,
        taus             = args.taus,
        layers           = args.layers,
        gammas           = args.gammas,
        dry_run          = args.dry_run,
    )

"""Training script for the MS-TCN++ temporal model.

Usage (from repo root):
    python phases_recognition/train_temporal.py \
        --config phases_recognition/configs/config_mstcn.yaml

Differences vs train.py (ResNet frame-level):
  - Loads pre-extracted feature sequences (.npy), not raw images.
  - One step = one full video (batch_size=1, variable T).
  - Model outputs a list of logits (one per stage); loss sums over all stages.
  - No image normalisation or visualisation.
"""

import argparse
import collections
import pathlib

import numpy as np
import torch
import tqdm
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter

from dataset.feature_dataset import instantiate_feature_loaders
from losses.mstcn_loss import MSTCNLoss
from metrics.cataract_metrics import CataractMetrics
from models import instantiate_model
from utils.helpers import (
    instantiate_dirs,
    save_python_code,
    get_commit_hash,
)
from utils.visualizer import TemporalVisualizer
from utils.lr_scheduler import get_linear_warmup_cosine_decay_lr_scheduler


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TemporalTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: MSTCNLoss,
        metrics_fn: CataractMetrics,
        visualizer: TemporalVisualizer,
        log_dir: str,
        ckpt_dir: str,
        epochs: int = 50,
        device: str = "cuda:0",
        max_norm: float = 1.0,
        log_every_n_steps: int = 5,
        val_every_n_epoch: int = 1,
        keep_ckpt: int = 3,
        **_,  # absorb unknown config keys
    ):
        self.model       = model
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.loss_fn     = loss_fn
        self.metrics_fn  = metrics_fn
        self.visualizer  = visualizer
        self.log_dir     = log_dir
        self.ckpt_dir    = ckpt_dir
        self.epochs      = epochs
        self.device      = torch.device(device)
        self.max_norm    = max_norm
        self.log_every_n_steps = log_every_n_steps
        self.val_every_n_epoch = val_every_n_epoch
        self.keep_ckpt   = keep_ckpt

        pathlib.Path(log_dir).mkdir(exist_ok=True, parents=True)
        pathlib.Path(ckpt_dir).mkdir(exist_ok=True, parents=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.visualizer.writer = self.writer

        self._global_step = 0
        self.best_f1      = -1.0

        self.model.to(self.device)
        self.loss_fn.to(self.device)

    # ------------------------------------------------------------------

    def _run_step(
        self,
        features: torch.Tensor,  # (T, D)
        labels:   torch.Tensor,  # (T,)
    ) -> tuple[torch.Tensor, dict, list[torch.Tensor]]:
        features = features.unsqueeze(0).to(self.device)  # (1, T, D)
        labels   = labels.to(self.device)                 # (T,)

        stage_logits = self.model(features)               # list of (1, C, T)
        total_loss, loss_dict = self.loss_fn(stage_logits, labels)
        return total_loss, loss_dict, stage_logits

    def _run_epoch(self, loader, epoch: int, tag: str):
        is_train = tag == "train"
        self.model.train() if is_train else self.model.eval()
        self.metrics_fn.reset()

        loss_history = collections.defaultdict(list)
        video_sequences = []  # pour le visualizer en val : (gt_seq, pred_seq, name)
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader), desc=f"{tag} epoch={epoch}")

        for i, (features, labels, video_name) in pbar:
            if is_train:
                self.optimizer.zero_grad(set_to_none=True)
                total_loss, loss_dict, stage_logits = self._run_step(features, labels)
                if torch.isnan(total_loss):
                    logger.error(f"NaN loss on {video_name}, skipping.")
                    continue
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_norm
                )
                self.optimizer.step()
                self.scheduler.step()
            else:
                with torch.no_grad():
                    total_loss, loss_dict, stage_logits = self._run_step(features, labels)

            # Metrics: use last-stage logits, reshape to (T, C)
            last_logits = stage_logits[-1].squeeze(0).T  # (T, C)
            self.metrics_fn.update(last_logits, labels.to(self.device))

            # Accumule pour le visualizer (val seulement)
            if not is_train:
                preds = last_logits.argmax(dim=1).cpu().tolist()
                video_sequences.append((labels.tolist(), preds, video_name))

            pbar.set_postfix(loss=f"{total_loss.item():.3f}", video=video_name[:30])
            loss_history["total_loss"].append(total_loss.item())
            for k, v in loss_dict.items():
                loss_history[k].append(v.item())

            if is_train and self._global_step % self.log_every_n_steps == 0:
                self.writer.add_scalar("lr", self.scheduler.get_last_lr()[-1], self._global_step)
                self.writer.add_scalar("train/total_loss_step", total_loss.item(), self._global_step)
                self.writer.add_scalar("train/grad_norm", grad_norm.item(), self._global_step)
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f"train/{k}_step", v.item(), self._global_step)

            if is_train:
                self._global_step += 1

        for k, vals in loss_history.items():
            self.writer.add_scalar(f"{tag}/{k}_epoch", np.mean(vals), epoch)

        metric_dict = self.metrics_fn.compute()
        for k, v in metric_dict.items():
            self.writer.add_scalar(f"{tag}/{k}_epoch", v, epoch)

        if not is_train and video_sequences:
            self.visualizer.log_epoch(video_sequences, epoch)

        return loss_history, metric_dict

    # ------------------------------------------------------------------

    def _save_ckpt(self, epoch: int, loss_history: dict, metric_dict: dict):
        ckpt_path = pathlib.Path(self.ckpt_dir) / f"model_{epoch:06d}.pt"
        state = {
            "epoch":              epoch,
            "model_state_dict":   self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric_dict":        metric_dict,
        }
        torch.save(state, ckpt_path)

        f1 = metric_dict.get("global/f1_macro", -1.0)
        if f1 > self.best_f1:
            logger.info(f"New best f1_macro={f1:.4f} at epoch={epoch}")
            self.best_f1 = f1
            torch.save(state, pathlib.Path(self.ckpt_dir) / "best.pt")

        # Prune old checkpoints
        ckpts = sorted(pathlib.Path(self.ckpt_dir).glob("model_*.pt"))
        for old in ckpts[: -self.keep_ckpt]:
            old.unlink()

    # ------------------------------------------------------------------

    def fit(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            self._run_epoch(train_loader, epoch, "train")
            if epoch % self.val_every_n_epoch == 0:
                loss_history, metric_dict = self._run_epoch(val_loader, epoch, "val")
                self._save_ckpt(epoch, loss_history, metric_dict)
                f1 = metric_dict.get("global/f1_macro", 0.0)
                logger.info(
                    f"Epoch {epoch} | val f1_macro={f1:.4f} | "
                    f"val accuracy={metric_dict.get('global/accuracy', 0.0):.4f}"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config: DictConfig):
    log_dir, ckpt_dir, img_dir, code_dir = instantiate_dirs(
        config.root_dir, config.experiment_name
    )
    root_code_path = pathlib.Path(__file__).parent
    save_python_code(
        root_code_path,
        pathlib.Path(code_dir),
        root_code_path.parent / ".gitignore",
    )
    config.commit_hash = get_commit_hash()
    OmegaConf.save(config, pathlib.Path(log_dir).parent / "config.yaml")

    model = instantiate_model(config.model)
    logger.info(
        f"Model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params"
    )

    train_loader, val_loader = instantiate_feature_loaders(config.dataset)
    logger.info(
        f"Dataset: {len(train_loader)} train videos, {len(val_loader)} val videos"
    )

    loss_fn = MSTCNLoss(
        lambda_smoothing=config.loss.get("lambda_smoothing", 0.15),
        tau=config.loss.get("tau", 4.0),
    )

    optimizer = getattr(torch.optim, config.optimizer.target.split(".")[-1])(
        model.parameters(),
        **OmegaConf.to_container(config.optimizer.params, resolve=True),
    )
    scheduler = get_linear_warmup_cosine_decay_lr_scheduler(
        optimizer, **OmegaConf.to_container(config.lr_scheduler.params, resolve=True)
    )

    class_names = list(config.dataset.class_names)

    metrics_fn = CataractMetrics(
        num_classes=config.model.num_classes,
        class_names=class_names,
        others_classes=list(config.metrics.others_classes),
    )

    visualizer = TemporalVisualizer(img_dir=img_dir, class_names=class_names)

    trainer = TemporalTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        visualizer=visualizer,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        **OmegaConf.to_container(config.train, resolve=True),
    )
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train MS-TCN++ temporal model")
    parser.add_argument(
        "--config", type=str,
        default="phases_recognition/configs/config_mstcn.yaml",
    )
    args = parser.parse_args()
    main(OmegaConf.load(args.config))

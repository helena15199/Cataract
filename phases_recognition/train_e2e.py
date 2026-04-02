"""End-to-end training: ResNet50 backbone + MS-TCN++ jointly.

Key differences vs train_temporal.py:
  - Input: raw frames (T, 3, H, W), not pre-extracted features.
  - Backbone is frozen for the first `freeze_backbone_epochs` epochs,
    then unfrozen at a much lower LR to avoid destroying ImageNet features.
  - Two parameter groups: backbone (backbone_lr) and TCN (lr).
  - AMP enabled by default (fp16 — makes backbone forward manageable in memory).
  - Val: per-chunk metrics (T_window frames each), not full videos.

Usage (from repo root):
    python phases_recognition/train_e2e.py \
        --config phases_recognition/configs/config_e2e.yaml
"""

import argparse
import collections
import pathlib

import numpy as np
import torch
import tqdm
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from dataset.temporal_frame_dataset import instantiate_e2e_loaders
from losses.mstcn_loss import MSTCNLoss
from metrics.cataract_metrics import CataractMetrics
from models import instantiate_model
from utils.helpers import instantiate_dirs, save_python_code, get_commit_hash
from utils.preprocessing import normalize_image
from utils.visualizer import TemporalVisualizer


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class E2ETrainer:
    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: MSTCNLoss,
        metrics_fn: CataractMetrics,
        visualizer: TemporalVisualizer,
        log_dir: str,
        ckpt_dir: str,
        epochs: int = 80,
        device: str = "cuda:0",
        use_amp: bool = True,
        max_norm: float = 1.0,
        freeze_backbone_epochs: int = 10,
        log_every_n_steps: int = 10,
        val_every_n_epoch: int = 1,
        keep_ckpt: int = 3,
        **_,
    ):
        self.model      = model
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.loss_fn    = loss_fn
        self.metrics_fn = metrics_fn
        self.visualizer = visualizer
        self.epochs     = epochs
        self.device     = torch.device(device)
        self.max_norm   = max_norm
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.log_every_n_steps  = log_every_n_steps
        self.val_every_n_epoch  = val_every_n_epoch
        self.keep_ckpt  = keep_ckpt

        self.grad_scaler = GradScaler(enabled=use_amp)
        self.use_amp     = use_amp

        pathlib.Path(log_dir).mkdir(exist_ok=True, parents=True)
        pathlib.Path(ckpt_dir).mkdir(exist_ok=True, parents=True)
        self.writer  = SummaryWriter(log_dir=log_dir)
        self.visualizer.writer = self.writer
        self.log_dir  = log_dir
        self.ckpt_dir = ckpt_dir

        self._global_step = 0
        self.best_f1      = -1.0

        self.model.to(self.device)
        self.loss_fn.to(self.device)

    # ------------------------------------------------------------------
    # One gradient step
    # ------------------------------------------------------------------

    def _run_step(self, frames: torch.Tensor, labels: torch.Tensor):
        frames = frames.to(self.device, non_blocking=True)
        labels = labels.to(self.device)
        frames = normalize_image(frames)                  # ImageNet normalisation

        with autocast(device_type="cuda", enabled=self.use_amp):
            stage_logits = self.model(frames)             # list of (1, C, T)
            total_loss, loss_dict = self.loss_fn(stage_logits, labels)

        return total_loss, loss_dict, stage_logits

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def _run_epoch(self, loader, epoch: int, tag: str):
        is_train = tag == "train"
        self.model.train() if is_train else self.model.eval()
        self.metrics_fn.reset()

        loss_history   = collections.defaultdict(list)
        video_sequences = []   # val only — for timeline visualizer

        pbar = tqdm.tqdm(enumerate(loader), total=len(loader),
                         desc=f"{tag} epoch={epoch}")

        for i, batch in pbar:
            if len(batch) == 2:
                # Train: (frames, labels)
                frames, labels = batch
                video_name, chunk_idx = "train", 0
            else:
                # Val: (frames, labels, video_name, chunk_idx)
                frames, labels, video_name, chunk_idx = batch

            if is_train:
                self.optimizer.zero_grad(set_to_none=True)
                total_loss, loss_dict, stage_logits = self._run_step(frames, labels)
                if torch.isnan(total_loss):
                    logger.error(f"NaN loss on {video_name} chunk {chunk_idx}, skipping.")
                    continue
                self.grad_scaler.scale(total_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_norm)
                scale_before = self.grad_scaler.get_scale()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                # N'avance le scheduler que si l'optimizer a vraiment stepé
                # (GradScaler saute l'optimizer.step si gradients inf/nan)
                if self.grad_scaler.get_scale() >= scale_before:
                    self.scheduler.step()
            else:
                with torch.no_grad():
                    total_loss, loss_dict, stage_logits = self._run_step(frames, labels)

            last_logits = stage_logits[-1].squeeze(0).T   # (T, C)
            self.metrics_fn.update(last_logits, labels.to(self.device))

            if not is_train:
                preds = last_logits.argmax(dim=1).cpu().tolist()
                video_sequences.append((labels.tolist(), preds, f"{video_name}_{chunk_idx}"))

            pbar.set_postfix(loss=f"{total_loss.item():.3f}")
            loss_history["total_loss"].append(total_loss.item())
            for k, v in loss_dict.items():
                loss_history[k].append(v.item())

            if is_train and self._global_step % self.log_every_n_steps == 0:
                self.writer.add_scalar("lr/tcn",      self.scheduler.get_last_lr()[-1], self._global_step)
                self.writer.add_scalar("lr/backbone", self.scheduler.get_last_lr()[0],  self._global_step)
                self.writer.add_scalar("train/total_loss", total_loss.item(), self._global_step)
                self.writer.add_scalar("train/grad_norm",  grad_norm.item(), self._global_step)

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
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_ckpt(self, epoch: int, metric_dict: dict):
        ckpt_path = pathlib.Path(self.ckpt_dir) / f"model_{epoch:06d}.pt"
        state = {
            "epoch":               epoch,
            "model_state_dict":    self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric_dict":         metric_dict,
        }
        torch.save(state, ckpt_path)

        f1 = metric_dict.get("global/f1_macro", -1.0)
        if f1 > self.best_f1:
            logger.info(f"New best f1_macro={f1:.4f} at epoch={epoch}")
            self.best_f1 = f1
            torch.save(state, pathlib.Path(self.ckpt_dir) / "best.pt")

        ckpts = sorted(pathlib.Path(self.ckpt_dir).glob("model_*.pt"))
        for old in ckpts[: -self.keep_ckpt]:
            old.unlink()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(self, train_loader, val_loader):
        for epoch in range(self.epochs):

            # Gèle / dégèle le backbone selon l'epoch
            if epoch == 0:
                self.model.freeze_backbone()
                logger.info("Backbone frozen — training TCN only.")
            elif epoch == self.freeze_backbone_epochs:
                self.model.unfreeze_backbone()
                logger.info(f"Backbone unfrozen at epoch {epoch} — fine-tuning jointly.")

            self._run_epoch(train_loader, epoch, "train")

            if epoch % self.val_every_n_epoch == 0:
                _, metric_dict = self._run_epoch(val_loader, epoch, "val")
                self._save_ckpt(epoch, metric_dict)
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"val f1={metric_dict.get('global/f1_macro', 0):.4f} | "
                    f"val acc={metric_dict.get('global/accuracy', 0):.4f} | "
                    f"backbone {'frozen' if epoch < self.freeze_backbone_epochs else 'unfrozen'}"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: DictConfig):
    log_dir, ckpt_dir, img_dir, code_dir = instantiate_dirs(
        config.root_dir, config.experiment_name)
    root_code_path = pathlib.Path(__file__).parent
    save_python_code(root_code_path, pathlib.Path(code_dir),
                     root_code_path.parent / ".gitignore")
    config.commit_hash = get_commit_hash()
    OmegaConf.save(config, pathlib.Path(log_dir).parent / "config.yaml")

    model = instantiate_model(config.model)
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    n_tcn      = sum(p.numel() for p in model.tcn.parameters())
    logger.info(f"Backbone params: {n_backbone:,} | TCN params: {n_tcn:,}")

    T_window = config.dataset.get("T_window", 128)
    train_loader, val_loader = instantiate_e2e_loaders(config.dataset, T_window=T_window)
    logger.info(f"Train: {len(train_loader)} windows/epoch | Val: {len(val_loader)} chunks")

    loss_fn = MSTCNLoss(
        lambda_smoothing = config.loss.get("lambda_smoothing", 0.15),
        tau              = config.loss.get("tau", 4.0),
        label_smoothing  = config.loss.get("label_smoothing", 0.0),
    )

    class_names = list(config.dataset.class_names)

    # Two LR groups: backbone (very slow) + TCN (normal)
    backbone_lr = config.optimizer.params.get("backbone_lr", 1e-5)
    tcn_lr      = config.optimizer.params.get("lr", 1e-4)
    weight_decay = config.optimizer.params.get("weight_decay", 1e-3)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": backbone_lr},
        {"params": model.tcn.parameters(),      "lr": tcn_lr},
    ], weight_decay=weight_decay)

    from utils.lr_scheduler import get_linear_warmup_cosine_decay_lr_scheduler
    scheduler = get_linear_warmup_cosine_decay_lr_scheduler(
        optimizer,
        **OmegaConf.to_container(config.lr_scheduler.params, resolve=True),
    )

    metrics_fn = CataractMetrics(
        num_classes    = config.model.num_classes,
        class_names    = class_names,
        others_classes = list(config.metrics.others_classes),
    )
    visualizer = TemporalVisualizer(img_dir=img_dir, class_names=class_names)

    trainer = E2ETrainer(
        model      = model,
        optimizer  = optimizer,
        scheduler  = scheduler,
        loss_fn    = loss_fn,
        metrics_fn = metrics_fn,
        visualizer = visualizer,
        log_dir    = log_dir,
        ckpt_dir   = ckpt_dir,
        **OmegaConf.to_container(config.train, resolve=True),
    )
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("End-to-end backbone + MS-TCN++ training")
    parser.add_argument("--config", type=str,
                        default="phases_recognition/configs/config_e2e.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))

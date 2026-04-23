import collections
import pathlib
from typing import Iterable, Literal

import einops
from loguru import logger
import numpy as np
import torch
from torch import nn
from torch import optim
import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from losses.basic_loss import CataractLoss
from metrics.cataract_metrics import CataractMetrics
from utils.preprocessing import normalize_image
from utils.visualizer import ClassificationVisualizer


class CataractTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        loss_fn: CataractLoss,
        log_dir: str,
        ckpt_dir: str,
        visualizer: ClassificationVisualizer,
        metrics_fn: CataractMetrics,
        epochs: int = 20,
        device: str = "cuda:0",
        use_amp: bool = True,
        max_norm: float = 1.0,
        log_every_n_steps: int = 1,
        log_image_every_n_epoch: int = 5,
        val_every_n_epoch: int = 5,
        keep_ckpt: int = 3,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.visualizer = visualizer
        self.metrics_fn = metrics_fn
        self.epochs = epochs
        self.device = torch.device(device=device)
        self.use_amp = use_amp
        self.max_norm = max_norm
        self.log_every_n_steps = log_every_n_steps
        self.log_image_every_n_epoch = log_image_every_n_epoch
        self.val_every_n_epoch = val_every_n_epoch
        self.keep_ckpt = keep_ckpt

        self.grad_scaler = GradScaler(enabled=self.use_amp)

        pathlib.Path(log_dir).mkdir(exist_ok=True, parents=True)
        pathlib.Path(ckpt_dir).mkdir(exist_ok=True, parents=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        self._start_epoch = 0
        self._global_step = 0
        self.best_metric = -1

        self.model.to(self.device)
        self.loss_fn.to(self.device)
        self.visualizer.writer = self.writer

    def run_step(
        self,
        images: torch.Tensor,
        phase_labels: torch.Tensor,
        binary_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
        images = images.to(self.device, non_blocking=True)
        phase_labels = phase_labels.to(self.device)
        binary_labels = binary_labels.to(self.device)
        images = normalize_image(images)

        with autocast(device_type="cuda", enabled=self.use_amp):
            phase_logits, binary_logits = self.model(images)
            final_loss, loss_dict = self.loss_fn(
                phase_logits, binary_logits, phase_labels, binary_labels
            )

        return final_loss, loss_dict, phase_logits, binary_logits

    def run_epoch(
        self,
        dataloader: Iterable[tuple[torch.Tensor, torch.Tensor, list]],
        epoch: int,
        tag: Literal["train", "val"] = "train",
    ):
        if tag == "train":
            self.model.train()
        else:
            self.model.eval()

        loss_dict_per_epoch = collections.defaultdict(list)
        self.metrics_fn.reset()

        pbar = tqdm.tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"{tag} {epoch=}"
        )
        for i, (images, phase_labels, binary_labels, paths) in pbar:
            if tag == "train":
                self.optimizer.zero_grad(set_to_none=True)
                final_loss, loss_dict, phase_logits, binary_logits = self.run_step(
                    images, phase_labels, binary_labels
                )
                if torch.isnan(final_loss):
                    logger.error("Loss is nan, skip this batch.")
                    continue
                self.grad_scaler.scale(final_loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_norm
                )
                grad_norm_after_clip = min(grad_norm.item(), self.max_norm)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.scheduler.step()
            else:
                with torch.no_grad():
                    final_loss, loss_dict, phase_logits, binary_logits = self.run_step(
                        images, phase_labels, binary_labels
                    )

            self.metrics_fn.update(phase_logits, phase_labels)

            if epoch % self.log_image_every_n_epoch == 0:
                self.visualizer(
                    logits,
                    labels,
                    paths,
                    tag=tag,
                    batch_idx=i,
                    global_step=self._global_step,
                    epoch=epoch,
                )

            pbar.set_description_str(f"{tag} {epoch=} loss={final_loss.item():.3f}")
            loss_dict_per_epoch["final_loss"].append(final_loss.item())
            for loss_name, loss_value in loss_dict.items():
                loss_dict_per_epoch[loss_name].append(loss_value.item())

            if tag == "train" and self._global_step % self.log_every_n_steps == 0:
                self.writer.add_scalar("lr", self.scheduler.get_last_lr()[-1], self._global_step)
                self.writer.add_scalar("train/final_loss_per_step", final_loss.item(), self._global_step)
                self.writer.add_scalar("train/grad_norm_before_clip", grad_norm.item(), self._global_step)
                self.writer.add_scalar("train/grad_norm_after_clip", grad_norm_after_clip, self._global_step)
                self.writer.add_scalar(
                    "train/grad_clipping_active",
                    1.0 if grad_norm.item() > self.max_norm else 0.0,
                    self._global_step,
                )
                for loss_name, loss_value in loss_dict.items():
                    self.writer.add_scalar(f"train/{loss_name}_per_step", loss_value.item(), self._global_step)

            if tag == "train":
                self._global_step += 1

        for loss_name, loss_values in loss_dict_per_epoch.items():
            self.writer.add_scalar(f"{tag}/{loss_name}_per_epoch", np.mean(loss_values), epoch)

        metric_dict = self.metrics_fn.compute()
        for metric_name, metric_value in metric_dict.items():
            self.writer.add_scalar(f"{tag}/{metric_name}_per_epoch", metric_value, epoch)

        return loss_dict_per_epoch, metric_dict

    def save_ckpt(self, epoch: int, loss_dict: dict, metric_dict: dict):
        ckpt_path = pathlib.Path(self.ckpt_dir) / f"model_{epoch:06d}.pt"
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_dict": loss_dict,
            "metric_dict": metric_dict,
        }
        torch.save(state, ckpt_path)

        current_f1 = metric_dict.get("global/f1_macro", -1)
        if self.best_metric == -1 or current_f1 > self.best_metric:
            logger.info(f"New best f1_macro={current_f1:.4f} at {epoch=}")
            self.best_metric = current_f1
            torch.save(state, pathlib.Path(self.ckpt_dir) / "best.pt")

        # Supprime les vieux checkpoints, garde les keep_ckpt derniers + best.pt
        ckpts = sorted(pathlib.Path(self.ckpt_dir).glob("model_*.pt"))
        for old_ckpt in ckpts[: -self.keep_ckpt]:
            old_ckpt.unlink()

    def set_start_epoch(self, start_epoch: int):
        self._start_epoch = start_epoch

    def fit(
        self,
        train_loader: Iterable[tuple[torch.Tensor, torch.Tensor, list]],
        val_loader: Iterable[tuple[torch.Tensor, torch.Tensor, list]],
    ):
        for epoch in range(self._start_epoch, self.epochs):
            self.run_epoch(train_loader, epoch, "train")
            if epoch % self.val_every_n_epoch == 0:
                loss_dict, metric_dict = self.run_epoch(val_loader, epoch, "val")
                self.save_ckpt(epoch, loss_dict, metric_dict)
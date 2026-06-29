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

from dataset.feature_dataset import instantiate_feature_loaders, VideoFeatureDataset
from losses.mstcn_loss import MSTCNLoss
from losses.temporal_clustering_loss import TemporalClusteringLoss
from metrics.cataract_metrics import CataractMetrics
from models import instantiate_model
from utils.helpers import (
    instantiate_dirs,
    save_python_code,
    get_commit_hash,
)
from utils.visualizer import TemporalVisualizer
from utils.lr_scheduler import get_linear_warmup_cosine_decay_lr_scheduler


def compute_class_weights(train_root: str, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights computed from all training label files."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for p in sorted(pathlib.Path(train_root).glob("*_labels.npy")):
        labels = np.load(p)
        for c in range(num_classes):
            counts[c] += int((labels == c).sum())
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1))
    logger.info("Class weights (inverse-freq):")
    for i, w in enumerate(weights):
        logger.info(f"  class {i:2d}: count={counts[i]:7d}  weight={w:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


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
        feature_noise_std: float = 0.0,
        early_stopping_patience: int = 0,   # 0 = disabled
        train_crop_len: int = 0,            # 0 = full sequence ; >0 = random temporal crop
        mixup_alpha: float = 0.0,           # 0 = disabled ; 0.4 recommended
        **kwargs,  # absorb unknown config keys (tcl_beta, tcl_momentum, etc.)
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
        self.keep_ckpt                = keep_ckpt
        self.feature_noise_std        = feature_noise_std
        self.early_stopping_patience  = early_stopping_patience
        self.train_crop_len           = train_crop_len
        self.mixup_alpha              = mixup_alpha
        self.tcl_beta                 = kwargs.get("tcl_beta", 0.0)
        self.tcl                      = None
        self._train_dataset           = None   # set in fit()

        pathlib.Path(log_dir).mkdir(exist_ok=True, parents=True)
        pathlib.Path(ckpt_dir).mkdir(exist_ok=True, parents=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.visualizer.writer = self.writer

        self._global_step    = 0
        self.best_f1         = -1.0
        self._no_improve     = 0

        self.model.to(self.device)
        self.loss_fn.to(self.device)

    # ------------------------------------------------------------------

    def _run_step(
        self,
        features:  torch.Tensor,  # (T, D)
        labels:    torch.Tensor,  # (T,)
        start_pos: int = 0,
        total_len: int = 0,
    ) -> tuple[torch.Tensor, dict, list[torch.Tensor]]:
        features = features.unsqueeze(0).to(self.device)  # (1, T, D)
        if self.feature_noise_std > 0 and self.model.training:
            features = features + torch.randn_like(features) * self.feature_noise_std
        labels = labels.to(self.device)

        if self.tcl is not None and hasattr(self.model, "forward_with_features"):
            stage_logits, internal_feats = self.model.forward_with_features(features)
            # internal_feats: (B, F, T) → (T, F)
            internal_feats = internal_feats.squeeze(0).T
        else:
            stage_logits = self.model(features, start_pos=start_pos, total_len=total_len)
            internal_feats = None

        total_loss, loss_dict = self.loss_fn(stage_logits, labels)

        if self.tcl is not None and internal_feats is not None:
            l_intra, l_inter = self.tcl(internal_feats, labels)
            tcl_loss = self.tcl_beta * (l_intra + l_inter)
            total_loss = total_loss + tcl_loss
            loss_dict["tcl_intra"] = l_intra.detach()
            loss_dict["tcl_inter"] = l_inter.detach()
            loss_dict["tcl_total"] = tcl_loss.detach()

        return total_loss, loss_dict, stage_logits

    def _run_epoch(self, loader, epoch: int, tag: str):
        is_train = tag == "train"
        self.model.train() if is_train else self.model.eval()
        self.metrics_fn.reset()

        loss_history = collections.defaultdict(list)
        video_sequences = []  # pour le visualizer en val : (gt_seq, pred_seq, name)
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader), desc=f"{tag} epoch={epoch}")

        for i, (features, labels, video_name) in pbar:
            # Mixup : interpoler avec une autre vidéo aléatoire par position relative
            if is_train and self.mixup_alpha > 0 and self._train_dataset is not None:
                lam = float(torch.distributions.Beta(
                    torch.tensor(self.mixup_alpha),
                    torch.tensor(self.mixup_alpha),
                ).sample())
                mix_idx = torch.randint(len(self._train_dataset), (1,)).item()
                feat_b, labels_b, _ = self._train_dataset[mix_idx]
                T_a, T_b = features.shape[0], feat_b.shape[0]
                # aligner feat_b sur la longueur de feat_a par position relative (t/T)
                idx_b = (torch.linspace(0, 1, T_a) * (T_b - 1)).long().clamp(0, T_b - 1)
                feat_b_aligned   = feat_b[idx_b]
                labels_b_aligned = labels_b[idx_b]
                features = lam * features + (1.0 - lam) * feat_b_aligned
                # labels : garder ceux de la vidéo dominante
                labels = labels if lam >= 0.5 else labels_b_aligned

            # Temporal crop : avant tout le reste pour que labels soit cohérent avec logits
            total_len = features.shape[0]
            start_pos = 0
            if is_train and self.train_crop_len > 0 and total_len > self.train_crop_len:
                start_pos = torch.randint(0, total_len - self.train_crop_len, (1,)).item()
                features  = features[start_pos : start_pos + self.train_crop_len]
                labels    = labels  [start_pos : start_pos + self.train_crop_len]

            if is_train:
                self.optimizer.zero_grad(set_to_none=True)
                total_loss, loss_dict, stage_logits = self._run_step(
                    features, labels, start_pos=start_pos, total_len=total_len
                )
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
                    total_loss, loss_dict, stage_logits = self._run_step(
                        features, labels, start_pos=0, total_len=0
                    )

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
        self._train_dataset = train_loader.dataset
        for epoch in range(self.epochs):
            self._run_epoch(train_loader, epoch, "train")
            if epoch % self.val_every_n_epoch == 0:
                loss_history, metric_dict = self._run_epoch(val_loader, epoch, "val")
                f1 = metric_dict.get("global/f1_macro", 0.0)
                prev_best = self.best_f1
                self._save_ckpt(epoch, loss_history, metric_dict)
                logger.info(
                    f"Epoch {epoch} | val f1_macro={f1:.4f} | "
                    f"val accuracy={metric_dict.get('global/accuracy', 0.0):.4f}"
                )

                if self.early_stopping_patience > 0:
                    if f1 > prev_best:
                        self._no_improve = 0
                    else:
                        self._no_improve += 1
                        logger.info(
                            f"No improvement for {self._no_improve}/{self.early_stopping_patience} epochs"
                        )
                    if self._no_improve >= self.early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch} "
                            f"(best val f1_macro={self.best_f1:.4f})"
                        )
                        break


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

    resume_ckpt = config.get("resume_ckpt", None)
    if resume_ckpt:
        state = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        logger.info(f"Resumed from {resume_ckpt} (epoch {state.get('epoch', '?')})")

    logger.info(
        f"Model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params"
    )

    train_loader, val_loader = instantiate_feature_loaders(config.dataset)
    logger.info(
        f"Dataset: {len(train_loader)} train videos, {len(val_loader)} val videos"
    )

    class_weights = None
    if config.loss.get("use_class_weights", False):
        class_weights = compute_class_weights(
            train_root=config.dataset["train"]["params"]["root"],
            num_classes=config.model.num_classes,
        )

    loss_fn = MSTCNLoss(
        lambda_smoothing=config.loss.get("lambda_smoothing", 0.15),
        tau=config.loss.get("tau", 4.0),
        label_smoothing=config.loss.get("label_smoothing", 0.0),
        class_weights=class_weights,
        focal_gamma=config.loss.get("focal_gamma", 0.0),
        logit_norm_tau=config.loss.get("logit_norm_tau", 0.0),
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

    train_params = OmegaConf.to_container(config.train, resolve=True)

    trainer = TemporalTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        visualizer=visualizer,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        **train_params,
    )

    tcl_beta = train_params.get("tcl_beta", 0.0)
    if tcl_beta > 0:
        feature_dim = config.model.get("num_f_maps", 64)
        tcl_momentum = train_params.get("tcl_momentum", 0.9)
        tcl_delta = train_params.get("tcl_delta", 1.0)
        trainer.tcl = TemporalClusteringLoss(
            num_classes=config.model.num_classes,
            feature_dim=feature_dim,
            momentum=tcl_momentum,
            delta=tcl_delta,
        ).to(trainer.device)
        logger.info(f"TCL enabled: beta={tcl_beta}, feature_dim={feature_dim}, "
                    f"momentum={tcl_momentum}, delta={tcl_delta}")

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train MS-TCN++ temporal model")
    parser.add_argument(
        "--config", type=str,
        default="phases_recognition/configs/config_mstcn.yaml",
    )
    parser.add_argument(
        "--override", type=str, nargs="*", default=[],
        help="Override config values, e.g. --override train.tcl_beta=0.01",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume/fine-tune from",
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if args.override:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))
    if args.resume:
        config.resume_ckpt = args.resume
    main(config)

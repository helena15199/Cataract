
import argparse
import pathlib

from omegaconf import OmegaConf, DictConfig
import torch

from dataset import instantiate_loaders
from dataset.cataract_dataset import compute_class_weights

from utils.helpers import (
    instantiate_dirs,
    instantiate_from_config,
    save_python_code,
    get_commit_hash,
)
from utils.trainer import CataractTrainer
from utils.visualizer import instantiate_visualizer
from models import instantiate_model
from losses.basic_loss import CataractLoss
from metrics.cataract_metrics import CataractMetrics


def main(config: dict | DictConfig | OmegaConf):
    log_dir, ckpt_dir, img_dir, code_dir = instantiate_dirs(
        config.root_dir, config.experiment_name
    )
    root_code_path = pathlib.Path(__file__).parent

    save_python_code(
        root_code_path, pathlib.Path(code_dir), root_code_path.parent / ".gitignore"
    )
    config.commit_hash = get_commit_hash()
    OmegaConf.save(config, pathlib.Path(log_dir).parent / "config.yaml")

    model = instantiate_model(config.model)

    train_loader, val_loader = instantiate_loaders(config.dataset)

    class_weights = compute_class_weights(train_loader.dataset)
    config.loss.params.weight = class_weights.tolist()
    loss_fn: CataractLoss = instantiate_from_config(config.loss)

    opt_params = OmegaConf.to_container(config.optimizer.params, resolve=True)
    backbone_lr = opt_params.pop("backbone_lr")
    optimizer: torch.optim.Optimizer = getattr(torch.optim, config.optimizer.target.split(".")[-1])(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.fc_complete_number.parameters()},
        ],
        **opt_params,
    )
    lr_scheduler = instantiate_from_config(config.lr_scheduler, optimizer=optimizer)

    metrics_fn = CataractMetrics(
        num_classes=config.model.num_classes,
        class_names=list(config.dataset.class_names),
        others_classes=list(config.metrics.others_classes),
    )

    visualizer = instantiate_visualizer(
        img_dir,
        **config.visualizer,
    )

    trainer = CataractTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loss_fn=loss_fn,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        visualizer=visualizer,
        metrics_fn=metrics_fn,
        **config.train,
    )

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run training")
    parser.add_argument("--config", type=str, default="phases_recognition/configs/config.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(config=config)




import argparse
import pathlib

from omegaconf import OmegaConf, DictConfig
import torch

from dataset import instantiate_loaders

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
    loss_fn: CataractLoss = instantiate_from_config(config.loss)

    train_loader, val_loader = instantiate_loaders(config.dataset)

    optimizer: torch.optim.Optimizer = instantiate_from_config(
        config.optimizer, params=model.parameters()
    )
    lr_scheduler = instantiate_from_config(config.lr_scheduler, optimizer=optimizer)

    metrics_fn = CataractMetrics(num_classes=17)

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
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(config=config)



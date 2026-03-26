import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.amp import autocast
import tqdm

from dataset.cataract_dataset import instantiate_dataloader
from metrics.cataract_metrics import CataractMetrics
from models import instantiate_model
from utils.preprocessing import normalize_image


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    confidences, preds = probs.max(dim=1)
    accuracies = preds.eq(labels)

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for low, high in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > low) & (confidences <= high)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].float().mean().item()
        bin_conf = confidences[mask].mean().item()
        ece += mask.float().mean().item() * abs(bin_acc - bin_conf)
    return ece


def plot_confidence_histogram(probs: torch.Tensor, labels: torch.Tensor, out_path: pathlib.Path):
    confidences, preds = probs.max(dim=1)
    correct = preds.eq(labels)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(confidences[correct].numpy(), bins=30, alpha=0.6, label="Correct", color="green")
    ax.hist(confidences[~correct].numpy(), bins=30, alpha=0.6, label="Incorrect", color="red")
    ax.set_xlabel("Confidence (max prob)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    preds: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    out_path: pathlib.Path,
    normalize: bool = True,
    eval_indices: list[int] | None = None,
):
    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=eval_indices)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45, values_format=".2f" if normalize else "d")
    ax.set_title("Confusion matrix (normalized)" if normalize else "Confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_inference(model, dataloader, device, use_amp):
    model.eval()
    all_probs, all_labels, all_paths = [], [], []

    for images, labels, paths in tqdm.tqdm(dataloader, desc="Inference"):
        images = images.to(device, non_blocking=True)
        images = normalize_image(images)
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
        probs = F.softmax(logits.float(), dim=1).cpu()
        all_probs.append(probs)
        all_labels.append(labels)
        all_paths.extend(paths)

    return torch.cat(all_probs), torch.cat(all_labels), all_paths


def main(config_path: str, ckpt_path: str, out_dir: str):
    config = OmegaConf.load(config_path)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.train.device)

    # Modèle
    model = instantiate_model(config.model)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Compatibilité ancienne tête (Linear simple) vs nouvelle (Sequential MLP)
    ckpt_keys = state["model_state_dict"].keys()
    if "fc_complete_number.weight" in ckpt_keys:
        model.fc_complete_number = torch.nn.Linear(model.final_size, model.num_classes)
        model.to(device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)

    # Test loader (même config que val, root swappé vers test/)
    test_config = OmegaConf.to_container(config.dataset.val, resolve=True)
    test_config["params"]["root"] = str(
        pathlib.Path(config.dataset.val.params.root).parent / "test"
    )
    class_names = list(config.dataset.class_names)
    others_classes = list(config.dataset.get("others_classes") or [])
    test_loader = instantiate_dataloader(test_config, class_names, others_classes, use_sampler=False)

    # Inférence
    probs, labels, paths = run_inference(model, test_loader, device, config.train.use_amp)
    preds = probs.argmax(dim=1)

    # Métriques principales
    metrics_fn = CataractMetrics(
        num_classes=config.model.num_classes,
        class_names=class_names,
        others_classes=list(config.metrics.others_classes),
    )
    metrics_fn.update(probs / probs.sum(dim=1, keepdim=True), labels)  # probs déjà softmax
    metrics = metrics_fn.compute()

    # ECE
    metrics["global/ece"] = compute_ece(probs, labels)

    # Affichage
    print("\n=== Test metrics ===")
    for k, v in sorted(metrics.items()):
        if k.startswith("global/"):
            print(f"  {k}: {v:.4f}")
    print("\nPer-class F1:")
    for k, v in sorted(metrics.items()):
        if k.startswith("per_class/f1/"):
            print(f"  {k.replace('per_class/f1/', '')}: {v:.4f}")

    # Sauvegarde JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({k: round(v, 6) for k, v in metrics.items()}, f, indent=2)

    # Filtrer Others pour les plots
    others_indices = {class_names.index(c) for c in config.metrics.others_classes if c in class_names}
    eval_mask = torch.ones(len(labels), dtype=torch.bool)
    for idx in others_indices:
        eval_mask &= (labels != idx)
    eval_class_names = [n for n in class_names if n not in config.metrics.others_classes]
    eval_indices = [i for i, n in enumerate(class_names) if n not in config.metrics.others_classes]

    # Plots
    plot_confusion_matrix(preds[eval_mask], labels[eval_mask], eval_class_names, out_dir / "confusion_matrix.png", eval_indices=eval_indices)
    plot_confusion_matrix(preds[eval_mask], labels[eval_mask], eval_class_names, out_dir / "confusion_matrix_counts.png", normalize=False, eval_indices=eval_indices)
    plot_confidence_histogram(probs[eval_mask], labels[eval_mask], out_dir / "confidence_histogram.png")

    print(f"\nRésultats sauvegardés dans {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate cataract model on test set")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Chemin vers best.pt ou model_XXXXXX.pt")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    main(args.config, args.ckpt, args.out_dir)
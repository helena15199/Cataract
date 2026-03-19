"""Metrics for cataract phase classification."""

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F


class CataractMetrics:
    def __init__(self, num_classes: int = 17, eps: float = 1e-7):
        self.num_classes = num_classes
        self.eps = eps
        self.reset()

    def reset(self):
        self.all_probs = []
        self.all_labels = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        probs = F.softmax(logits, dim=1).detach().cpu()
        self.all_probs.append(probs)
        self.all_labels.append(labels.detach().cpu())

    def compute(self) -> dict[str, float]:
        probs = torch.cat(self.all_probs, dim=0)   # (N, 17)
        gt = torch.cat(self.all_labels, dim=0)      # (N,)
        preds = probs.argmax(dim=1)

        eps = self.eps
        n = self.num_classes

        accuracy = (preds == gt).float().mean().item()

        tp = torch.zeros(n)
        fp = torch.zeros(n)
        fn = torch.zeros(n)
        for c in range(n):
            tp[c] = ((preds == c) & (gt == c)).sum()
            fp[c] = ((preds == c) & (gt != c)).sum()
            fn[c] = ((preds != c) & (gt == c)).sum()

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)

        metrics = {
            "global/accuracy": accuracy,
            "global/f1_macro": f1.mean().item(),
            "global/precision_macro": precision.mean().item(),
            "global/recall_macro": recall.mean().item(),
        }

        try:
            auroc = roc_auc_score(
                F.one_hot(gt, num_classes=n).numpy(),
                probs.numpy(),
                average="macro",
                multi_class="ovr",
            )
            metrics["global/auroc"] = auroc
        except ValueError:
            pass  # peut arriver si une classe est absente du batch de val

        for c in range(n):
            metrics[f"per_class/f1/{c}"] = f1[c].item()
            metrics[f"per_class/precision/{c}"] = precision[c].item()
            metrics[f"per_class/recall/{c}"] = recall[c].item()

        return metrics
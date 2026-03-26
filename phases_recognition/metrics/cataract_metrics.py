"""Metrics for cataract phase classification."""

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F


class CataractMetrics:
    def __init__(
        self,
        num_classes: int = 16,
        class_names: list[str] | None = None,
        others_classes: list[str] | None = None,
        eps: float = 1e-7,
    ):
        self.num_classes = num_classes
        self.eps = eps
        self.class_names = class_names or [str(i) for i in range(num_classes)]

        name_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.others_indices = set()
        if others_classes:
            self.others_indices = {name_to_idx[n] for n in others_classes if n in name_to_idx}
        self.eval_indices = [i for i in range(num_classes) if i not in self.others_indices]

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
        probs = torch.cat(self.all_probs, dim=0)   # (N, num_classes)
        gt = torch.cat(self.all_labels, dim=0)      # (N,)
        preds = probs.argmax(dim=1)

        # Filtrer les samples des classes "Others" pour les métriques
        mask = torch.ones(len(gt), dtype=torch.bool)
        for idx in self.others_indices:
            mask &= (gt != idx)
        gt_eval = gt[mask]
        preds_eval = preds[mask]
        probs_eval = probs[mask]

        eps = self.eps
        n_eval = len(self.eval_indices)

        accuracy = (preds_eval == gt_eval).float().mean().item() if len(gt_eval) > 0 else 0.0

        tp = torch.zeros(self.num_classes)
        fp = torch.zeros(self.num_classes)
        fn = torch.zeros(self.num_classes)
        for c in self.eval_indices:
            tp[c] = ((preds_eval == c) & (gt_eval == c)).sum()
            fp[c] = ((preds_eval == c) & (gt_eval != c)).sum()
            fn[c] = ((preds_eval != c) & (gt_eval == c)).sum()

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)

        eval_f1        = f1[self.eval_indices]
        eval_precision = precision[self.eval_indices]
        eval_recall    = recall[self.eval_indices]

        metrics = {
            "global/accuracy": accuracy,
            "global/f1_macro": eval_f1.mean().item(),
            "global/precision_macro": eval_precision.mean().item(),
            "global/recall_macro": eval_recall.mean().item(),
        }

        try:
            # AUROC uniquement sur les classes évaluées
            eval_idx_list = self.eval_indices
            gt_oh = F.one_hot(gt_eval, num_classes=self.num_classes).numpy()[:, eval_idx_list]
            probs_np = probs_eval.numpy()[:, eval_idx_list]
            auroc = roc_auc_score(gt_oh, probs_np, average="macro", multi_class="ovr")
            metrics["global/auroc"] = auroc
        except ValueError:
            pass

        for c in self.eval_indices:
            name = self.class_names[c]
            metrics[f"per_class/f1/{name}"] = f1[c].item()
            metrics[f"per_class/precision/{name}"] = precision[c].item()
            metrics[f"per_class/recall/{name}"] = recall[c].item()

        return metrics

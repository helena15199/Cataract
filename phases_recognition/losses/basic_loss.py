import torch
from torch import nn


class CataractLoss(nn.Module):
    def __init__(self, weight: list[float] | None = None, label_smoothing: float = 0):
        super().__init__()
        weight = None if weight is None else torch.tensor(weight)
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=label_smoothing
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        loss = self.cross_entropy(logits, targets)
        return loss, {"phase": loss}
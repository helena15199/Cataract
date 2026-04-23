import torch
from torch import nn


class PhaseLoss(nn.Module):
    """CrossEntropy sur les frames non-CH (ignore_index=-1)."""

    def __init__(self, weight: list[float] | None = None, label_smoothing: float = 0):
        super().__init__()
        weight = None if weight is None else torch.tensor(weight)
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=label_smoothing, ignore_index=-1
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(logits, targets)


class BinaryPhaseLoss(nn.Module):
    """BCEWithLogits pour la tête binaire (ex: Corneal_hydration)."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets)


class CataractLoss(nn.Module):
    def __init__(self, weight: list[float] | None = None, label_smoothing: float = 0):
        super().__init__()
        self.phase_loss = PhaseLoss(weight=weight, label_smoothing=label_smoothing)
        self.binary_loss = BinaryPhaseLoss()

    def forward(
        self,
        phase_logits: torch.Tensor,
        binary_logits: torch.Tensor,
        phase_targets: torch.Tensor,
        binary_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        loss_phase = self.phase_loss(phase_logits, phase_targets)
        loss_binary = self.binary_loss(binary_logits, binary_targets)
        return loss_phase + loss_binary, {"phase": loss_phase, "binary": loss_binary}
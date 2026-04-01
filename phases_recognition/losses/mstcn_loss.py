"""MS-TCN++ loss: cross-entropy + truncated MSE smoothness loss.

For each stage:
    L_stage = CE(logits, targets) + lambda_smoothing * TMSE(log_softmax(logits))

Total loss = sum over all stages (equal weight, as in the original paper).

TMSE (Truncated MSE) penalises abrupt changes between consecutive frame
predictions, but clips the squared difference at tau to avoid over-smoothing
at genuine phase boundaries.

Reference: Li et al., MS-TCN++, TPAMI 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TMSELoss(nn.Module):
    """Truncated MSE smoothness loss applied to log-probabilities."""

    def __init__(self, tau: float = 4.0):
        super().__init__()
        self.tau = tau

    def forward(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_probs: (B, C, T)
        Returns:
            scalar loss
        """
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]  # (B, C, T-1)
        return torch.clamp(diff ** 2, max=self.tau).mean()


class MSTCNLoss(nn.Module):
    """
    Loss for MS-TCN++.

    Args:
        lambda_smoothing: weight of the TMSE term (0.15 in the original paper).
        tau:              clipping threshold for TMSE (4.0 in the original paper).
    """

    def __init__(self, lambda_smoothing: float = 0.15, tau: float = 4.0):
        super().__init__()
        self.lambda_smoothing = lambda_smoothing
        self.ce   = nn.CrossEntropyLoss(ignore_index=-1)
        self.tmse = _TMSELoss(tau=tau)

    def forward(
        self,
        stage_logits: list[torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            stage_logits: list of (B, C, T) — one per MS-TCN stage.
            targets:      (T,) or (B, T)   — frame-level class indices.
        Returns:
            total_loss: scalar
            loss_dict:  {"ce": ..., "tmse": ...}  (averaged over stages)
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)  # → (1, T)

        total_loss = targets.new_zeros((), dtype=torch.float32)
        ce_sum     = targets.new_zeros((), dtype=torch.float32)
        tmse_sum   = targets.new_zeros((), dtype=torch.float32)

        for logits in stage_logits:
            # CrossEntropyLoss accepts (B, C, T) vs (B, T) natively
            ce   = self.ce(logits, targets)
            smooth = self.tmse(F.log_softmax(logits, dim=1))
            total_loss = total_loss + ce + self.lambda_smoothing * smooth
            ce_sum     = ce_sum   + ce
            tmse_sum   = tmse_sum + smooth

        n = len(stage_logits)
        return total_loss, {
            "ce":   ce_sum   / n,
            "tmse": tmse_sum / n,
        }

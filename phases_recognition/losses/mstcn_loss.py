"""MS-TCN++ loss: cross-entropy (or focal) + truncated MSE smoothness loss.

For each stage:
    L_stage = Lcls(logits, targets) + lambda_smoothing * TMSE(log_softmax(logits))

where Lcls is either standard CrossEntropyLoss or FocalLoss.

Total loss = sum over all stages (equal weight, as in the original paper).

TMSE (Truncated MSE) penalises abrupt changes between consecutive frame
predictions, but clips the squared difference at tau to avoid over-smoothing
at genuine phase boundaries.

Focal loss is useful when some phases are very short (e.g. Corneal_hydration):
it down-weights well-classified frames and focuses training on hard/rare ones.

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


class _FocalLoss(nn.Module):
    """
    Focal loss for multi-class temporal segmentation.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma:        focusing parameter. 0 = standard CE. 2 is typical.
        weight:       per-class weights (C,), same as CE weight parameter.
        ignore_index: class index to ignore (e.g. -1 for padding).
        label_smoothing: applied before computing focal weight.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = -1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma           = gamma
        self.ignore_index    = ignore_index
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, T) or (B, C)
            targets: (B, T) or (B,)  — integer class indices
        Returns:
            scalar mean focal loss
        """
        # Flatten to (N, C) and (N,)
        if logits.dim() == 3:
            B, C, T = logits.shape
            logits  = logits.permute(0, 2, 1).reshape(-1, C)   # (B*T, C)
            targets = targets.reshape(-1)                        # (B*T,)

        mask = targets != self.ignore_index
        logits  = logits[mask]
        targets = targets[mask]

        if logits.numel() == 0:
            return logits.sum() * 0.0

        # Smooth targets for log_p computation
        C = logits.size(1)
        log_p = F.log_softmax(logits, dim=1)                   # (N, C)
        p_t   = log_p.exp().gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma               # (N,)

        # Per-sample CE with optional label smoothing
        if self.label_smoothing > 0:
            smooth_loss = -log_p.mean(dim=1)                    # uniform part
            ce_loss     = -log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
            ce_loss = (1 - self.label_smoothing) * ce_loss + self.label_smoothing * smooth_loss
        else:
            ce_loss = -log_p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)

        loss = focal_weight * ce_loss                           # (N,)

        # Apply per-class weights if provided
        if self.weight is not None:
            w = self.weight.to(logits.device)[targets]         # (N,)
            loss = loss * w

        return loss.mean()


class MSTCNLoss(nn.Module):
    """
    Loss for MS-TCN++.

    Args:
        lambda_smoothing: weight of the TMSE term (0.15 in the original paper).
        tau:              clipping threshold for TMSE (4.0 in the original paper).
        label_smoothing:  label smoothing coefficient.
        class_weights:    optional (C,) tensor of per-class weights.
        focal_gamma:      if > 0, use focal loss instead of standard CE.
                          gamma=2 is a good default. 0 = standard CE.
    """

    def __init__(self, lambda_smoothing: float = 0.15, tau: float = 4.0,
                 label_smoothing: float = 0.0,
                 class_weights: torch.Tensor | None = None,
                 focal_gamma: float = 0.0):
        super().__init__()
        self.lambda_smoothing = lambda_smoothing

        if focal_gamma > 0:
            self.ce = _FocalLoss(
                gamma=focal_gamma,
                weight=class_weights,
                ignore_index=-1,
                label_smoothing=label_smoothing,
            )
        else:
            self.ce = nn.CrossEntropyLoss(
                ignore_index=-1,
                label_smoothing=label_smoothing,
                weight=class_weights,
            )

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

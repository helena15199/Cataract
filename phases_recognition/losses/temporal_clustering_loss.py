"""Temporal Clustering Loss (TCL) for open-set phase recognition.

Adds two terms to the standard classification loss:
  L_intra: penalizes distance of features to their class center (compactness)
  L_inter: penalizes proximity between class centers (separability)

Class centers are maintained as exponential moving averages (EMA) across batches
to handle the temporal autocorrelation of video features.

Reference: Gao et al., "EPGCN: Evidential Prototype-Guided Clustering Network
           for Open Set Action Recognition", ACM MM 2023.

Usage:
    tcl = TemporalClusteringLoss(num_classes=11, feature_dim=64)
    # In training loop:
    l_intra, l_inter = tcl(features, labels)
    total_loss = l_ce + beta * (l_intra + l_inter)
"""

import torch
import torch.nn as nn


class TemporalClusteringLoss(nn.Module):
    """Temporal Clustering Loss with EMA class centers.

    Args:
        num_classes: number of known phases.
        feature_dim: dimension of the feature space where clustering is applied.
        momentum: EMA momentum for center updates (0.9 = slow update, stable).
        delta: margin for inter-class term (avoids division by zero).
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        momentum: float = 0.9,
        delta: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.delta = delta

        self.register_buffer("centers", torch.zeros(num_classes, feature_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    def _update_centers(self, features: torch.Tensor, labels: torch.Tensor):
        """Update class centers with EMA. Only during training."""
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            batch_mean = features[mask].mean(dim=0)
            if not self.initialized[c]:
                self.centers[c] = batch_mean
                self.initialized[c] = True
            else:
                self.centers[c] = (self.momentum * self.centers[c]
                                   + (1 - self.momentum) * batch_mean)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute L_intra and L_inter.

        Args:
            features: (T, D) or (B*T, D) — internal features (not logits).
            labels: (T,) or (B*T,) — class indices, -1 for unknown (ignored).

        Returns:
            l_intra: intra-class compactness loss (scalar).
            l_inter: inter-class separability loss (scalar).
        """
        known_mask = labels >= 0
        if known_mask.sum() == 0:
            zero = features.new_zeros(())
            return zero, zero

        features_known = features[known_mask]
        labels_known = labels[known_mask]

        if self.training:
            self._update_centers(features_known.detach(), labels_known)

        # L_intra: mean squared distance to own class center
        assigned_centers = self.centers[labels_known]  # (N, D)
        l_intra = ((features_known - assigned_centers) ** 2).sum(dim=1).mean()

        # L_inter: inverse pairwise distance between active centers
        active = self.initialized.nonzero(as_tuple=True)[0]
        if len(active) < 2:
            l_inter = features.new_zeros(())
        else:
            active_centers = self.centers[active]  # (K, D)
            n = len(active_centers)
            # Pairwise squared distances
            diff = active_centers.unsqueeze(0) - active_centers.unsqueeze(1)  # (K, K, D)
            dist_sq = (diff ** 2).sum(dim=2)  # (K, K)
            # Upper triangle only (avoid double counting + diagonal)
            mask_triu = torch.triu(torch.ones(n, n, device=features.device, dtype=torch.bool),
                                   diagonal=1)
            l_inter = (1.0 / (dist_sq[mask_triu] + self.delta)).mean()

        return l_intra, l_inter

"""MS-TCN++ (Li et al. 2020) for temporal action segmentation.

Architecture:
  MultiStageModel
  ├── prediction_stage   : SingleStageTCN  (DilatedResidualLayer × R)
  └── refinement_stages  : RefinementStageTCN × (num_stages - 1)
                           (DualDilatedLayer × R, MS-TCN++ improvement)

Input:  (B, T, input_dim)  — one video at a time, B=1 in practice
Output: list of (B, num_classes, T), one tensor per stage

References:
  MS-TCN:   Farha & Gall, CVPR 2019
  MS-TCN++: Li et al., TPAMI 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DilatedResidualLayer(nn.Module):
    """
    Used in the prediction generation stage.
    x → conv(dilation=d) → ReLU → dropout → conv(1×1) → + x
    """

    def __init__(self, dilation: int, num_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv   = nn.Conv1d(num_channels, num_channels, kernel_size=3,
                                padding=dilation, dilation=dilation)
        self.proj   = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv(x))
        out = self.proj(self.dropout(out))
        return x + out


class DualDilatedLayer(nn.Module):
    """
    Used in refinement stages (MS-TCN++ improvement).
    Two parallel dilated convolutions (d and 2d) are summed before the
    projection and residual connection — captures two temporal scales at once.
    """

    def __init__(self, dilation: int, num_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1  = nn.Conv1d(num_channels, num_channels, kernel_size=3,
                                padding=dilation, dilation=dilation)
        self.conv2  = nn.Conv1d(num_channels, num_channels, kernel_size=3,
                                padding=2 * dilation, dilation=2 * dilation)
        self.proj   = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv1(x)) + F.relu(self.conv2(x))
        out = self.proj(self.dropout(out))
        return x + out


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

class SingleStageTCN(nn.Module):
    """
    Prediction generation stage.
    input_dim → num_f_maps via 1×1 conv, then R DilatedResidualLayers,
    then num_f_maps → num_classes via 1×1 conv.
    """

    def __init__(
        self,
        input_dim: int,
        num_f_maps: int,
        num_classes: int,
        num_layers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.input_proj  = nn.Conv1d(input_dim, num_f_maps, kernel_size=1)
        self.layers      = nn.ModuleList([
            DilatedResidualLayer(dilation=2 ** i, num_channels=num_f_maps, dropout=dropout)
            for i in range(num_layers)
        ])
        self.output_proj = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim, T) → (B, num_classes, T)"""
        out = self.input_proj(x)
        for layer in self.layers:
            out = layer(out)
        return self.output_proj(out)


class RefinementStageTCN(nn.Module):
    """
    Refinement stage (MS-TCN++).
    Takes the softmax output of the previous stage as input.
    Uses DualDilatedLayer instead of DilatedResidualLayer.
    """

    def __init__(
        self,
        num_classes: int,
        num_f_maps: int,
        num_layers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.input_proj  = nn.Conv1d(num_classes, num_f_maps, kernel_size=1)
        self.layers      = nn.ModuleList([
            DualDilatedLayer(dilation=2 ** i, num_channels=num_f_maps, dropout=dropout)
            for i in range(num_layers)
        ])
        self.output_proj = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_classes, T) → (B, num_classes, T)"""
        out = self.input_proj(x)
        for layer in self.layers:
            out = layer(out)
        return self.output_proj(out)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MSTCNPlusPlus(nn.Module):
    """
    MS-TCN++ multi-stage temporal convolutional network.

    Args:
        num_stages:  Total stages (1 prediction + num_stages-1 refinement).
        num_layers:  Dilated residual layers per stage (receptive field = 2^num_layers).
        num_f_maps:  Internal feature map channels in each stage.
        input_dim:   Feature dimension from the backbone (2048 for ResNet50).
        num_classes: Number of output phases.
        dropout:     Dropout probability inside residual layers.
    """

    def __init__(
        self,
        num_stages: int = 4,
        num_layers: int = 10,
        num_f_maps: int = 64,
        input_dim: int = 2048,
        num_classes: int = 13,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.prediction_stage = SingleStageTCN(
            input_dim=input_dim,
            num_f_maps=num_f_maps,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.refinement_stages = nn.ModuleList([
            RefinementStageTCN(
                num_classes=num_classes,
                num_f_maps=num_f_maps,
                num_layers=num_layers,
                dropout=dropout,
            )
            for _ in range(num_stages - 1)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim)  — B=1 in practice (one video per step)
        Returns:
            List of (B, num_classes, T) logits, one per stage.
            Loss should be computed on all stages; inference uses the last one.
        """
        x = x.transpose(1, 2)  # → (B, input_dim, T) for Conv1d

        outputs = []
        out = self.prediction_stage(x)          # (B, num_classes, T)
        outputs.append(out)

        for stage in self.refinement_stages:
            out = stage(F.softmax(out, dim=1))  # (B, num_classes, T)
            outputs.append(out)

        return outputs  # len = num_stages


def instantiate_mstcn(model_config: DictConfig | dict) -> MSTCNPlusPlus:
    return MSTCNPlusPlus(
        num_stages=model_config.get("num_stages", 4),
        num_layers=model_config.get("num_layers", 10),
        num_f_maps=model_config.get("num_f_maps", 64),
        input_dim=model_config.get("input_dim", 2048),
        num_classes=model_config.get("num_classes", 13),
        dropout=model_config.get("dropout", 0.5),
    )

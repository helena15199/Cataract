"""Dual-stream model: parallel LSTM + MS-TCN branches with learned gating.

Each branch specialises independently (auxiliary losses), then a per-frame
per-class gate learns to combine them end-to-end.

forward() returns a list of (1, C, T) tensors compatible with MSTCNLoss:
    [mstcn_stage_1, ..., mstcn_stage_N, lstm_out, fused]
The last element (fused) is used for metrics; all elements contribute to loss.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .lstm_tcn import _LSTMStage
from .mstcn import MSTCNPlusPlus


class DualStream(nn.Module):
    """
    Args:
        input_dim:       DINOv2 feature dimension (768)
        num_classes:     number of surgical phases
        lstm_hidden:     LSTM hidden size per direction
        lstm_layers:     number of stacked LSTM layers
        lstm_bidir:      bidirectional LSTM
        lstm_dropout:    dropout inside LSTM
        mstcn_stages:    number of MS-TCN stages
        mstcn_layers:    dilated layers per stage
        mstcn_f_maps:    feature maps per stage
        mstcn_dropout:   dropout inside MS-TCN residual layers
    """

    def __init__(
        self,
        input_dim:      int   = 768,
        num_classes:    int   = 11,
        lstm_hidden:    int   = 256,
        lstm_layers:    int   = 2,
        lstm_bidir:     bool  = True,
        lstm_dropout:   float = 0.3,
        mstcn_stages:   int   = 4,
        mstcn_layers:   int   = 10,
        mstcn_f_maps:   int   = 64,
        mstcn_dropout:  float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.lstm_branch = _LSTMStage(
            input_dim       = input_dim,
            hidden_dim      = lstm_hidden,
            num_classes     = num_classes,
            num_lstm_layers = lstm_layers,
            dropout         = lstm_dropout,
            bidirectional   = lstm_bidir,
        )

        self.mstcn_branch = MSTCNPlusPlus(
            num_stages  = mstcn_stages,
            num_layers  = mstcn_layers,
            num_f_maps  = mstcn_f_maps,
            input_dim   = input_dim,
            num_classes = num_classes,
            dropout     = mstcn_dropout,
        )

        # Gate: takes concat of both logits per frame → per-class weight in [0,1]
        # gate=1 → trust LSTM fully; gate=0 → trust MS-TCN fully
        self.gate = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, **_) -> list[torch.Tensor]:
        """
        Args:
            x: (1, T, input_dim)
        Returns:
            list of (1, C, T) — [mstcn_s1, ..., mstcn_sN, lstm_out, fused]
        """
        # MS-TCN branch: list of (1, C, T)
        mstcn_outs = self.mstcn_branch(x)       # N tensors
        mstcn_last = mstcn_outs[-1]             # (1, C, T)

        # LSTM branch: (1, C, T)
        lstm_out = self.lstm_branch(x)

        # Gating — per frame, per class
        # Transpose to (T, C) for the linear layer
        lstm_t  = lstm_out.squeeze(0).T          # (T, C)
        mstcn_t = mstcn_last.squeeze(0).T        # (T, C)
        gate_in = torch.cat([lstm_t, mstcn_t], dim=1)   # (T, 2C)
        gate    = self.gate(gate_in)             # (T, C)
        gate    = gate.unsqueeze(0).permute(0, 2, 1)     # (1, C, T)

        fused = gate * lstm_out + (1.0 - gate) * mstcn_last  # (1, C, T)

        return [*mstcn_outs, lstm_out, fused]


def instantiate_dual_stream(config: DictConfig | dict) -> DualStream:
    return DualStream(
        input_dim     = config.get("input_dim",      768),
        num_classes   = config["num_classes"],
        lstm_hidden   = config.get("lstm_hidden",    256),
        lstm_layers   = config.get("lstm_layers",    2),
        lstm_bidir    = config.get("lstm_bidir",     True),
        lstm_dropout  = config.get("lstm_dropout",   0.3),
        mstcn_stages  = config.get("mstcn_stages",   4),
        mstcn_layers  = config.get("mstcn_layers",   10),
        mstcn_f_maps  = config.get("mstcn_f_maps",   64),
        mstcn_dropout = config.get("mstcn_dropout",  0.5),
    )

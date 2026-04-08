"""TeCNO: Surgical Phase Recognition with Multi-Stage Temporal Convolutional Networks
(Czempiel et al., MICCAI 2020).

Architecture:
  - Stage 1 : feature projection + LSTM → captures short-term temporal dynamics
  - Stage 2+ : TCN refinement stages (same as MS-TCN) applied to LSTM output
               to smooth and refine predictions

Why TeCNO for surgical phases:
  - LSTM naturally models sequential state ("what happened so far")
  - TCN then cleans up over-segmentation at a global scale
  - Designed specifically for surgical workflow recognition
  - Outperforms MS-TCN on Cholec80 and Cataract-101

Differences vs MS-TCN++:
  - Stage 1 is LSTM instead of dilated TCN
  - LSTM hidden state = implicit memory of surgical progression
  - Subsequent TCN stages refine LSTM predictions

Reference: Czempiel et al., "TeCNO: Surgical Phase Recognition with Multi-Stage
           Temporal Convolutional Networks", MICCAI 2020.
           https://arxiv.org/abs/2003.10751
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .mstcn import RefinementStageTCN


# ---------------------------------------------------------------------------
# LSTM stage (Stage 1)
# ---------------------------------------------------------------------------

class _LSTMStage(nn.Module):
    """
    Projects input features → hidden_dim, runs a bidirectional LSTM,
    then projects to num_classes.

    Bidirectional: the model sees both past and future context, which is valid
    for offline surgical video analysis (the full video is available at inference).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size   = hidden_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_lstm_layers,
            batch_first  = True,
            dropout      = dropout if num_lstm_layers > 1 else 0.0,
            bidirectional= bidirectional,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_proj = nn.Linear(lstm_out_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        Returns: (B, num_classes, T)
        """
        out = self.dropout(F.relu(self.input_proj(x)))  # (B, T, hidden_dim)
        out, _ = self.lstm(out)                          # (B, T, lstm_out_dim)
        out = self.output_proj(self.dropout(out))        # (B, T, num_classes)
        return out.transpose(1, 2)                       # (B, num_classes, T)


# ---------------------------------------------------------------------------
# Full TeCNO model
# ---------------------------------------------------------------------------

class TeCNO(nn.Module):
    """
    TeCNO: LSTM stage followed by MS-TCN refinement stages.

    Args:
        num_stages:      total stages (1 LSTM + num_stages-1 TCN refinement)
        num_layers:      dilated layers per TCN refinement stage
        num_f_maps:      channels in TCN refinement stages
        hidden_dim:      LSTM hidden size
        num_lstm_layers: number of stacked LSTM layers
        input_dim:       input feature dimension (2048 for ResNet50)
        num_classes:     number of output phases
        dropout:         dropout probability
        bidirectional:   whether the LSTM is bidirectional
    """

    def __init__(
        self,
        num_stages: int = 4,
        num_layers: int = 10,
        num_f_maps: int = 32,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        input_dim: int = 2048,
        num_classes: int = 13,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm_stage = _LSTMStage(
            input_dim       = input_dim,
            hidden_dim      = hidden_dim,
            num_classes     = num_classes,
            num_lstm_layers = num_lstm_layers,
            dropout         = dropout,
            bidirectional   = bidirectional,
        )
        self.refinement_stages = nn.ModuleList([
            RefinementStageTCN(
                num_classes = num_classes,
                num_f_maps  = num_f_maps,
                num_layers  = num_layers,
                dropout     = dropout,
            )
            for _ in range(num_stages - 1)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        x: (B, T, input_dim)
        Returns: list of (B, num_classes, T), one per stage
        """
        # Stage 1: LSTM
        out = self.lstm_stage(x)  # (B, num_classes, T)
        outputs = [out]

        # Stages 2+: TCN refinement on softmax probabilities
        for stage in self.refinement_stages:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out)

        return outputs


def instantiate_tecno(model_config: DictConfig | dict) -> TeCNO:
    return TeCNO(
        num_stages      = model_config.get("num_stages",      4),
        num_layers      = model_config.get("num_layers",      10),
        num_f_maps      = model_config.get("num_f_maps",      32),
        hidden_dim      = model_config.get("hidden_dim",      256),
        num_lstm_layers = model_config.get("num_lstm_layers", 2),
        input_dim       = model_config.get("input_dim",       2048),
        num_classes     = model_config.get("num_classes",     13),
        dropout         = model_config.get("dropout",         0.3),
        bidirectional   = model_config.get("bidirectional",   True),
    )

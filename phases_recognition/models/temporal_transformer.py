"""Lightweight temporal transformer for surgical phase recognition.

Designed to work on top of frozen DINOv2 CLS-token features (or any
pre-extracted feature sequence). Follows the same interface as MS-TCN++
and ASFormer: forward() returns a list of (1, C, T) logit tensors so the
existing MSTCNLoss and TemporalTrainer work unchanged.

Architecture:
    Linear projection  input_dim → hidden_dim
    Temporal encoding  (sinusoidal on absolute position OR on ratio t/T)
    N × TransformerEncoderLayer  (Pre-LN, batch_first)
    Linear head        hidden_dim → num_classes
"""

import math

import torch
import torch.nn as nn
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Temporal encoding
# ---------------------------------------------------------------------------

def _build_sinusoidal(positions: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    positions : (T,) float  — either absolute frame indices or scaled ratios
    returns   : (T, d_model)
    """
    div = torch.exp(
        torch.arange(0, d_model, 2, device=positions.device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(len(positions), d_model, device=positions.device)
    pos = positions.unsqueeze(1)       # (T, 1)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TemporalEncoding(nn.Module):
    """
    Two modes:

    ratio (recommended for surgery):
        Each position is encoded as t/T * 10000, so the model always sees
        the same sinusoidal pattern regardless of video length.
        Frame at 20% of surgery → same encoding across all videos.

    absolute:
        Standard sinusoidal on the raw frame index.
        Fast to compute but the model must generalise across video lengths.
    """

    def __init__(self, d_model: int, mode: str = "ratio"):
        super().__init__()
        assert mode in ("ratio", "absolute"), f"Unknown mode: {mode}"
        self.mode = mode
        self.d_model = d_model

    def forward(self, x: torch.Tensor, start_pos: int = 0, total_len: int = 0) -> torch.Tensor:
        """
        x          : (T, d_model)
        start_pos  : index of first frame in the original full video (for crops)
        total_len  : length of the full video (0 = use T, i.e. full sequence)
        """
        T = x.shape[0]
        N = total_len if total_len > 0 else T
        if self.mode == "ratio":
            # positions scaled so the full video always spans [0, 10000]
            # a crop [start:start+T] maps to [start/N*10000, (start+T)/N*10000]
            t0 = start_pos / N * 10000.0
            t1 = (start_pos + T) / N * 10000.0
            positions = torch.linspace(t0, t1, T, device=x.device)
        else:
            positions = torch.arange(start_pos, start_pos + T, device=x.device).float()
        return x + _build_sinusoidal(positions, self.d_model)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DinoTemporalTransformer(nn.Module):
    """
    Lightweight temporal transformer on top of DINOv2 CLS-token features.

    Args:
        input_dim:   dimensionality of input features (768 for ViT-B, 1024 for ViT-L)
        hidden_dim:  internal dimension (64 / 128 / 256)
        num_classes: number of surgical phases
        num_layers:  number of TransformerEncoder layers (1 or 2 recommended)
        num_heads:   number of attention heads (must divide hidden_dim evenly)
        dropout:     dropout in attention and FFN
        pos_mode:    "ratio" or "absolute" — see TemporalEncoding
    """

    def __init__(
        self,
        input_dim:   int = 768,
        hidden_dim:  int = 128,
        num_classes: int = 11,
        num_layers:  int = 2,
        num_heads:   int = 4,
        dropout:     float = 0.1,
        pos_mode:    str = "ratio",
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc    = TemporalEncoding(hidden_dim, mode=pos_mode)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,   # conservative: 2× instead of 4×
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,                  # Pre-LN: more stable on small datasets
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout_out = nn.Dropout(dropout)
        self.head        = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        total_len: int = 0,
    ) -> list[torch.Tensor]:
        """
        Args:
            x          : (1, T, input_dim)
            start_pos  : index of first frame in the original full video (for crops)
            total_len  : length of the full video (0 = full sequence, no crop)
        Returns:
            list of one tensor  (1, num_classes, T)
        """
        x = x.squeeze(0)                              # (T, input_dim)
        x = self.input_proj(x)                        # (T, hidden_dim)
        x = self.pos_enc(x, start_pos, total_len)     # (T, hidden_dim)
        x = x.unsqueeze(0)         # (1, T, hidden_dim)  — batch_first
        x = self.transformer(x)    # (1, T, hidden_dim)
        x = self.dropout_out(x)
        logits = self.head(x)      # (1, T, num_classes)
        logits = logits.permute(0, 2, 1)   # (1, num_classes, T)
        return [logits]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def instantiate_dino_transformer(config: DictConfig | dict) -> DinoTemporalTransformer:
    return DinoTemporalTransformer(
        input_dim=config.get("input_dim",   768),
        hidden_dim=config.get("hidden_dim", 128),
        num_classes=config["num_classes"],
        num_layers=config.get("num_layers", 2),
        num_heads=config.get("num_heads",   4),
        dropout=config.get("dropout",       0.1),
        pos_mode=config.get("pos_mode",     "ratio"),
    )

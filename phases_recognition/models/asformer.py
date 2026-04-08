"""ASFormer: Transformer for Action Segmentation (Yi et al., BMVC 2021).

Architecture:
  - Encoder : N×  AttentionBlock (dilated causal conv + self-attention)
  - Decoder : M stages, each takes softmax of previous stage
               N× AttentionBlock with cross-attention from encoder output

Key differences vs MS-TCN++:
  - Self-attention captures global temporal context (not just local dilation window)
  - Positional encoding lets the model know WHERE in the video each frame sits
  - Cross-attention in decoder stages lets each refinement stage attend to
    the encoder's rich representations

Reference: Yi et al., "ASFormer: Transformer for Action Segmentation", BMVC 2021
           https://arxiv.org/abs/2110.08568
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to feature maps."""

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model, T)"""
        return x + self.pe[:, :x.size(2), :].transpose(1, 2)


# ---------------------------------------------------------------------------
# Attention block
# ---------------------------------------------------------------------------

class _AttentionBlock(nn.Module):
    """
    One transformer block with:
      - dilated causal conv (local context)
      - multi-head self-attention (global context)
      - optional cross-attention (for decoder stages)
      - FFN + residual + LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dilation: int,
        dropout: float = 0.1,
        with_cross_attention: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.with_cross_attention = with_cross_attention
        if with_cross_attention:
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                    dropout=dropout, batch_first=True)
            self.norm_cross = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # (B, d_model, T)
        encoder_out: torch.Tensor | None = None,  # (B, d_model, T) for cross-attn
    ) -> torch.Tensor:
        # Dilated conv (local context)
        x = x + self.dropout(F.relu(self.conv(x)))

        # Self-attention (global context) — transpose to (B, T, d_model) for MHA
        xt = x.transpose(1, 2)
        attn_out, _ = self.self_attn(xt, xt, xt)
        x = x + self.dropout(attn_out.transpose(1, 2))
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)

        # Cross-attention (decoder only)
        if self.with_cross_attention and encoder_out is not None:
            et = encoder_out.transpose(1, 2)
            xt = x.transpose(1, 2)
            ca_out, _ = self.cross_attn(xt, et, et)
            x = x + self.dropout(ca_out.transpose(1, 2))
            x = self.norm_cross(x.transpose(1, 2)).transpose(1, 2)

        # FFN
        xt = x.transpose(1, 2)
        x = x + self.dropout(self.ffn(xt)).transpose(1, 2)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)

        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class _ASFormerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_classes: int,
        num_layers: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, d_model, kernel_size=1)
        self.pos_enc    = _PositionalEncoding(d_model)
        self.layers     = nn.ModuleList([
            _AttentionBlock(d_model, n_heads, dilation=2 ** i,
                            dropout=dropout, with_cross_attention=False)
            for i in range(num_layers)
        ])
        self.output_proj = nn.Conv1d(d_model, num_classes, kernel_size=1)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, input_dim, T)
        Returns: logits (B, num_classes, T), features (B, d_model, T)
        """
        out = self.input_proj(x)
        out = self.pos_enc(out)
        for layer in self.layers:
            out = layer(out)
        return self.output_proj(out), out


# ---------------------------------------------------------------------------
# Decoder stage
# ---------------------------------------------------------------------------

class _ASFormerDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_model: int,
        num_layers: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj  = nn.Conv1d(num_classes, d_model, kernel_size=1)
        self.pos_enc     = _PositionalEncoding(d_model)
        self.layers      = nn.ModuleList([
            _AttentionBlock(d_model, n_heads, dilation=2 ** i,
                            dropout=dropout, with_cross_attention=True)
            for i in range(num_layers)
        ])
        self.output_proj = nn.Conv1d(d_model, num_classes, kernel_size=1)

    def forward(
        self,
        probs: torch.Tensor,         # (B, num_classes, T) — softmax of prev stage
        encoder_out: torch.Tensor,   # (B, d_model, T)
    ) -> torch.Tensor:
        out = self.input_proj(probs)
        out = self.pos_enc(out)
        for layer in self.layers:
            out = layer(out, encoder_out)
        return self.output_proj(out)


# ---------------------------------------------------------------------------
# Full ASFormer model
# ---------------------------------------------------------------------------

class ASFormer(nn.Module):
    """
    ASFormer: Transformer for Action Segmentation.

    Args:
        num_stages:  total stages (1 encoder + num_stages-1 decoders)
        num_layers:  attention blocks per stage
        d_model:     internal feature dimension
        n_heads:     number of attention heads (d_model must be divisible by n_heads)
        input_dim:   input feature dimension (2048 for ResNet50)
        num_classes: number of output phases
        dropout:     dropout in attention and FFN
    """

    def __init__(
        self,
        num_stages: int = 4,
        num_layers: int = 10,
        d_model: int = 64,
        n_heads: int = 8,
        input_dim: int = 2048,
        num_classes: int = 13,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = _ASFormerEncoder(
            input_dim=input_dim, d_model=d_model,
            num_classes=num_classes, num_layers=num_layers,
            n_heads=n_heads, dropout=dropout,
        )
        self.decoders = nn.ModuleList([
            _ASFormerDecoder(
                num_classes=num_classes, d_model=d_model,
                num_layers=num_layers, n_heads=n_heads, dropout=dropout,
            )
            for _ in range(num_stages - 1)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        x: (B, T, input_dim)
        Returns: list of (B, num_classes, T), one per stage
        """
        x = x.transpose(1, 2)  # → (B, input_dim, T)

        logits, enc_out = self.encoder(x)
        outputs = [logits]

        for decoder in self.decoders:
            logits = decoder(F.softmax(logits, dim=1), enc_out)
            outputs.append(logits)

        return outputs


def instantiate_asformer(model_config: DictConfig | dict) -> ASFormer:
    return ASFormer(
        num_stages  = model_config.get("num_stages",  4),
        num_layers  = model_config.get("num_layers",  10),
        d_model     = model_config.get("d_model",     64),
        n_heads     = model_config.get("n_heads",     8),
        input_dim   = model_config.get("input_dim",   2048),
        num_classes = model_config.get("num_classes", 13),
        dropout     = model_config.get("dropout",     0.1),
    )

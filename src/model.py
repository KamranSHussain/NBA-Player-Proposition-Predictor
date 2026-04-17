"""Model definitions for NBA player-prop quantile regression."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

DEFAULT_QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)


class PinballLoss(nn.Module):
    """Pinball loss for multi-quantile regression.

    The module expects predictions shaped as ``(batch_size, num_quantiles)`` and
    labels shaped as ``(batch_size, 1)`` (or broadcast-compatible).
    """

    def __init__(self, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> None:
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute average pinball loss across all configured quantiles."""
        total_loss = 0.0
        for i, q in enumerate(self.quantiles):
            pred_q = y_pred[:, i : i + 1]
            error = y_true - pred_q
            loss_q = torch.maximum(q * error, (q - 1.0) * error)
            total_loss += torch.mean(loss_q)

        return total_loss / len(self.quantiles)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence models."""

    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional information to token embeddings."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class PlayerPropTransformer(nn.Module):
    """Transformer encoder for quantile predictions from per-game history.

    Inputs:
    - ``sequence_features``: Tensor shaped ``(batch_size, seq_len, input_size)``
    - ``padding_mask``: Optional bool tensor shaped ``(batch_size, seq_len)`` where
      ``True`` means "ignore this padded timestep".
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.2,
        max_len: int = 256,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> None:
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles)

        self.input_proj = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # A learned fallback lets short-history players avoid unstable pooled states.
        self.cold_start_token = nn.Parameter(torch.zeros(1, d_model))
        self.history_scale = 8.0

        self.head = nn.Sequential(
            nn.Linear(d_model + 1, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, len(self.quantiles)),
        )

    @staticmethod
    def lengths_to_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Convert per-sample valid lengths into a transformer padding mask."""
        if lengths.dim() != 1:
            raise ValueError("lengths must be a 1D tensor of shape (batch_size,)")
        range_idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return range_idx >= lengths.unsqueeze(1)

    def _masked_mean(self, states: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Compute masked mean pool over sequence states."""
        valid_mask = (~padding_mask).unsqueeze(-1).to(states.dtype)
        valid_counts = valid_mask.sum(dim=1).clamp(min=1.0)
        pooled = (states * valid_mask).sum(dim=1) / valid_counts
        return pooled

    def forward(
        self,
        sequence_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run transformer inference for quantile predictions."""
        if sequence_features.dim() != 3:
            raise ValueError(
                "sequence_features must be shaped (batch_size, seq_len, input_size)"
            )

        batch_size, seq_len, _ = sequence_features.shape

        if padding_mask is None:
            padding_mask = torch.zeros(
                batch_size, seq_len, dtype=torch.bool, device=sequence_features.device
            )
        elif padding_mask.shape != (batch_size, seq_len):
            raise ValueError(
                "padding_mask must match shape (batch_size, seq_len)"
            )

        token_embeddings = self.input_proj(sequence_features)
        token_embeddings = self.positional_encoding(token_embeddings)
        token_embeddings = self.input_norm(token_embeddings)

        encoded = self.encoder(token_embeddings, src_key_padding_mask=padding_mask)
        pooled = self._masked_mean(encoded, padding_mask=padding_mask)

        valid_counts = (~padding_mask).sum(dim=1, keepdim=True).to(pooled.dtype)
        history_gate = torch.clamp(valid_counts / self.history_scale, min=0.0, max=1.0)

        cold_start = self.cold_start_token.expand(batch_size, -1)
        blended_state = history_gate * pooled + (1.0 - history_gate) * cold_start

        head_input = torch.cat([blended_state, torch.log1p(valid_counts)], dim=1)
        return self.head(head_input)


__all__ = [
    "DEFAULT_QUANTILES",
    "PinballLoss",
    "PositionalEncoding",
    "PlayerPropTransformer",
]
    
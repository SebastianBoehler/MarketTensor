"""Temporal convolution network."""

from __future__ import annotations

import torch
from torch import nn


class TemporalBlock(nn.Module):
    """A small residual temporal block."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)[..., : inputs.shape[-1]]


class TCNClassifier(nn.Module):
    """Compact TCN classifier."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.block = TemporalBlock(hidden_dim, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(inputs.transpose(1, 2))
        hidden = self.block(hidden)
        pooled = self.pool(hidden).squeeze(-1)
        return self.head(pooled).squeeze(-1)

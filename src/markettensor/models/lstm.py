"""LSTM benchmark model."""

from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Single-layer LSTM classifier."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        last_state = self.dropout(output[:, -1, :])
        return self.head(last_state).squeeze(-1)

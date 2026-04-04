from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """
    A compact LSTM forecaster for traffic speed regression.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        output_size: int = 1,
        horizon: int = 12,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.horizon = horizon
        effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Linear(hidden_size, horizon * output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_len, features]

        Returns:
            [batch, horizon, output_size]
        """

        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() != 3:
            raise ValueError("Expected input shape [batch, time, features].")
        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected feature size {self.input_size}, got {x.size(-1)}."
            )

        output, _ = self.lstm(x)
        last_state = output[:, -1, :]
        pred = self.head(last_state)
        return pred.view(x.size(0), self.horizon, self.output_size)


class TrafficLSTMEncoder(nn.Module):
    """
    Optional encoder variant that exposes the final hidden state.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]


TrafficLSTMRegressor = LSTMForecaster

"""LSTM branch: [B, T, F] -> [B, hidden_size]."""
import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError

"""LSTM branch: [B, T, F] -> [B, output_dim].

Two forms:
  layer_sizes null / []      : one nn.LSTM with `num_layers` stacked layers of `hidden_size` (the main model)
  layer_sizes [256,128,64,32]: Amouei's (2023) tapered stack, one nn.LSTM per layer, every layer but the last
                               returns its whole sequence (return_sequences=True in Keras); `dropout` is applied
                               between layers like nn.LSTM's inter-layer dropout. output_dim = the last size.
"""
import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float,
                 layer_sizes: list[int] | None = None):
        super().__init__()
        self.layer_sizes = [int(s) for s in (layer_sizes or [])]
        if self.layer_sizes:
            dims = [input_size] + self.layer_sizes
            self.stack = nn.ModuleList(nn.LSTM(dims[i], dims[i + 1], batch_first=True) for i in range(len(self.layer_sizes)))
            self.between = nn.Dropout(dropout)
            self.output_dim = self.layer_sizes[-1]
        else:
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
            self.output_dim = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_sizes:
            for i, layer in enumerate(self.stack):
                x, (hidden, _) = layer(x)
                if i < len(self.stack) - 1:
                    x = self.between(x)
            return hidden[-1]                        # last layer, last time step
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]                            # last layer, last time step

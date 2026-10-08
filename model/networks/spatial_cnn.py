"""CNN branch: [B, C, H, W] -> [B, embed_dim]."""
import torch
import torch.nn as nn


class SpatialCNN(nn.Module):
    def __init__(self, in_channels: int, conv_channels: list[int], kernel_size: int,
                 embed_dim: int, dropout: float):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError

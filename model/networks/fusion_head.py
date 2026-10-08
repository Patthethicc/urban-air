"""Concatenate both embeddings and regress: [B, D_s + D_t] -> [B, 1]."""
import torch
import torch.nn as nn


class FusionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float, out_dim: int = 1):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, spatial_emb: torch.Tensor, temporal_emb: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError

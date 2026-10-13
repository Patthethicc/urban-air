"""Concatenate both embeddings and regress: [B, D_s + D_t] -> [B, 1]."""
import torch
import torch.nn as nn


class FusionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float, out_dim: int = 1):
        super().__init__()
        layers, dim = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers.append(nn.Linear(dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, spatial_emb: torch.Tensor | None, temporal_emb: torch.Tensor) -> torch.Tensor:
        if spatial_emb is None:                      # temporal-only baseline
            return self.mlp(temporal_emb)
        return self.mlp(torch.cat([spatial_emb, temporal_emb], dim=1))

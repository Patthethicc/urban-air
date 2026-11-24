"""CNN branch: [B, C, H, W] -> [B, embed_dim].

feature_map = "pool"    : conv blocks -> adaptive average pool to 1x1 -> linear   (default; any patch size)
feature_map = "flatten" : conv blocks -> flatten the whole map (C*H*W) -> linear  (needs patch_px; the
                          padding keeps H = W = patch_px because kernel sizes are odd)
"""
import torch
import torch.nn as nn


class SpatialCNN(nn.Module):
    def __init__(self, in_channels: int, conv_channels: list[int], kernel_size: int,
                 embed_dim: int, dropout: float, feature_map: str = "pool", patch_px: int | None = None):
        super().__init__()
        layers, channels = [], in_channels
        for out_channels in conv_channels:
            layers += [nn.Conv2d(channels, out_channels, kernel_size, padding=kernel_size // 2), nn.ReLU()]
            channels = out_channels
        self.conv = nn.Sequential(*layers)

        self.feature_map, self.patch_px = feature_map, patch_px
        if feature_map == "pool":
            self.reduce = nn.AdaptiveAvgPool2d(1)
            flat = channels
        elif feature_map == "flatten":
            if not patch_px:
                raise ValueError("feature_map='flatten' needs patch_px (derived from the raster by config_parser)")
            self.reduce = nn.Identity()
            flat = channels * patch_px * patch_px
        else:
            raise ValueError(f"unknown feature_map '{feature_map}' (pool | flatten)")
        self.project = nn.Sequential(nn.Linear(flat, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.output_dim = embed_dim

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """conv blocks -> reduce -> flatten (before the linear layer)."""
        x = self.conv(x)
        if self.feature_map == "flatten" and x.shape[-2:] != (self.patch_px, self.patch_px):
            raise ValueError(f"patch is {tuple(x.shape[-2:])} but the model was built for "
                             f"{self.patch_px}x{self.patch_px}; rebuild with the right patch_px")
        return self.reduce(x).flatten(1)

    def forward(self, x: torch.Tensor, dedupe: bool = False) -> torch.Tensor:
        """`dedupe`: a patch is static per station, so a training batch repeats the same few patches many times. With
        dedupe each DISTINCT patch goes through the conv blocks and the linear layer once and the result is looked up
        per row; the dropout is then applied per row. Same output and same gradients as the plain path (the gradient of
        a shared patch is the sum over its rows), much less work when rows share patches."""
        if dedupe and x.shape[0] > 1:
            unique, inverse = torch.unique(x.flatten(1), dim=0, return_inverse=True)
            if unique.shape[0] < x.shape[0]:
                linear, relu, dropout = self.project
                shared = relu(linear(self._features(unique.view(-1, *x.shape[1:]))))
                return dropout(shared[inverse])
        return self.project(self._features(x))

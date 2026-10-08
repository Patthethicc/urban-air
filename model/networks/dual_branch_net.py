"""Full model: SpatialCNN + TemporalLSTM -> FusionHead."""
import torch
import torch.nn as nn


class DualBranchNet(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        self.model_cfg = model_cfg   # kept so the checkpoint can store it
        # TODO: build the three submodules from model_cfg
        raise NotImplementedError

    def forward(self, x_spatial: torch.Tensor, x_temporal: torch.Tensor) -> torch.Tensor:
        """x_spatial [B,C,H,W], x_temporal [B,T,F] -> [B,1]"""
        # TODO
        raise NotImplementedError

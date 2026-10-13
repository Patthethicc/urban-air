"""Full model: SpatialCNN + TemporalLSTM -> FusionHead."""
import torch
import torch.nn as nn

from networks.fusion_head import FusionHead
from networks.spatial_cnn import SpatialCNN
from networks.temporal_lstm import TemporalLSTM


class DualBranchNet(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        self.model_cfg = model_cfg   # kept so the checkpoint can rebuild the model without guessing
        s, t, f = model_cfg["spatial_cnn"], model_cfg["temporal_lstm"], model_cfg["fusion_head"]
        self.use_spatial = bool(s.get("enabled", True))   # False = temporal-only baseline for the R2 comparison
        self.spatial = (SpatialCNN(s["in_channels"], s["conv_channels"], s["kernel_size"], s["embed_dim"], s["dropout"],
                                   s.get("feature_map", "pool"), s.get("patch_px")) if self.use_spatial else None)
        self.temporal = TemporalLSTM(t["input_size"], t["hidden_size"], t["num_layers"], t["dropout"])
        in_dim = (self.spatial.output_dim if self.use_spatial else 0) + self.temporal.output_dim
        self.head = FusionHead(in_dim, f["hidden_dims"], f["dropout"], f["out_dim"])

    def forward(self, x_spatial: torch.Tensor, x_temporal: torch.Tensor) -> torch.Tensor:
        """x_spatial [B,C,H,W], x_temporal [B,T,F] -> [B,1]"""
        spatial = self.spatial(x_spatial) if self.use_spatial else None
        return self.head(spatial, self.temporal(x_temporal))

    def parameter_counts(self) -> dict:
        count = lambda m: sum(p.numel() for p in m.parameters())
        return {"spatial_cnn": count(self.spatial) if self.use_spatial else 0, "temporal_lstm": count(self.temporal),
                "fusion_head": count(self.head), "total": count(self)}

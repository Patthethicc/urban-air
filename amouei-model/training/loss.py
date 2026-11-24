"""Loss selection."""
import torch.nn as nn


def build_loss(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss()
    raise ValueError(f"unknown loss '{name}' (mse | huber)")

"""Loss selection."""
import torch.nn as nn


def build_loss(name: str) -> nn.Module:
    # TODO: 'mse' | 'huber'
    raise NotImplementedError

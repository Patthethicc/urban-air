"""Torch Dataset over the saved .npy files."""
import torch
from torch.utils.data import Dataset


class AirQualityDataset(Dataset):
    def __init__(self, processed_dir: str, split: str):
        """Load {X_spatial,X_temporal,y}_<split>.npy (already scaled)."""
        # TODO
        raise NotImplementedError

    def __len__(self) -> int:
        # TODO
        raise NotImplementedError

    def __getitem__(self, i: int):
        """Return (x_spatial, x_temporal, y) tensors."""
        # TODO
        raise NotImplementedError

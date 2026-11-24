"""Torch Dataset over the saved .npy files (patches looked up per station, not stored per row)."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_meta(processed_dir: str, partition: str) -> pd.DataFrame:
    return pd.read_csv(Path(processed_dir) / f"meta_{partition}.csv", parse_dates=["timestamp"])


class AirQualityDataset(Dataset):
    def __init__(self, processed_dir: str, partition: str, indices=None):
        """partition: 'train' | 'calibration' | 'test'.
        Loads patches.npy once, plus station_idx_<p>, X_temporal_<p>, y_<p>.
        `indices` selects a spatial-CV fold inside 'train' (validation comes from GroupKFold on
        stations, not from a separate file)."""
        d = Path(processed_dir)
        self.patches = np.load(d / "patches.npy")
        self.station_idx = np.load(d / f"station_idx_{partition}.npy")
        self.X_temporal = np.load(d / f"X_temporal_{partition}.npy")
        self.y = np.load(d / f"y_{partition}.npy")
        if indices is not None:
            indices = np.asarray(indices)
            self.station_idx, self.X_temporal, self.y = (
                self.station_idx[indices], self.X_temporal[indices], self.y[indices])

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        """(patches[station_idx[i]] -> [C,H,W], X_temporal[i] -> [T,F], y[i] -> [1])"""
        return (
            torch.from_numpy(self.patches[self.station_idx[i]]),
            torch.from_numpy(self.X_temporal[i]),
            torch.from_numpy(self.y[i]),
        )

"""Scalers fit on the PROPER-TRAINING stations only, then applied to calibration and test.

  target      : standard scaler on y of the training rows
  PM2.5 feats : lags, spatial lag, rolling mean/std reuse the TARGET scaler (same units); the
                rolling std is only divided (no mean shift)  -> pm25_feature_idx from config_parser
  bands       : per-band mean/std over the training stations' patches (one patch per station, so a
                station with more rows does not get more weight)
  sin/cos     : untouched
"""
import json

import numpy as np


class Scalers:
    def __init__(self):
        self.y_mean = 0.0
        self.y_std = 1.0
        self.band_mean = None
        self.band_std = None
        self.pm25_idx: list[int] = []
        self.roll_std_idx: list[int] = []

    def fit(self, patches_train, y_train, pm25_feature_idx, rolling_std_idx=()) -> "Scalers":
        """patches_train [S_train, C, H, W]; y_train [N_train, 1]."""
        y = np.asarray(y_train, dtype=np.float64).ravel()
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) or 1.0

        p = np.asarray(patches_train, dtype=np.float64)
        self.band_mean = p.mean(axis=(0, 2, 3))
        std = p.std(axis=(0, 2, 3))
        self.band_std = np.where(std > 1e-8, std, 1.0)

        self.pm25_idx = [int(i) for i in pm25_feature_idx]
        self.roll_std_idx = [int(i) for i in rolling_std_idx]
        return self

    def transform_patches(self, patches):
        p = np.asarray(patches, dtype=np.float32)
        mean = self.band_mean.astype(np.float32)[None, :, None, None]
        std = self.band_std.astype(np.float32)[None, :, None, None]
        return (p - mean) / std

    def transform_temporal(self, X_temporal):
        """[N, T, F]: scale only the pm25_feature_idx columns."""
        X = np.array(X_temporal, dtype=np.float32, copy=True)
        for i in self.pm25_idx:
            if i in self.roll_std_idx:
                X[..., i] = X[..., i] / self.y_std
            else:
                X[..., i] = (X[..., i] - self.y_mean) / self.y_std
        return X

    def transform_y(self, y):
        return ((np.asarray(y, dtype=np.float32) - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Back to ug/m3 for metrics, plots and GeoCP scores."""
        return np.asarray(y_scaled, dtype=np.float64) * self.y_std + self.y_mean

    def to_dict(self) -> dict:
        return {
            "y_mean": self.y_mean, "y_std": self.y_std,
            "band_mean": self.band_mean.tolist(), "band_std": self.band_std.tolist(),
            "pm25_idx": self.pm25_idx, "roll_std_idx": self.roll_std_idx,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Scalers":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        s = cls()
        s.y_mean, s.y_std = d["y_mean"], d["y_std"]
        s.band_mean, s.band_std = np.array(d["band_mean"]), np.array(d["band_std"])
        s.pm25_idx, s.roll_std_idx = d["pm25_idx"], d["roll_std_idx"]
        return s

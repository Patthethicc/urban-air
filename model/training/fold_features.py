"""Per-fold spatial lag.

preprocess computes the spatial lag once, with ALL training stations as neighbours. Inside
cross-validation that is a leak: a validation station's readings would be a feature of the fit
stations (and so reach the model during training). Here the lag channel of X_temporal is recomputed
for every fold so neighbours come only from that fold's FIT stations; a validation (or outer-test)
station gets its lag from fit stations too, exactly as calibration / test stations do in production.

Rows whose lag cannot be computed (no neighbour reported at one of the lag hours) are dropped from
that fold.
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.scalers import Scalers
from preprocessing.spatial_lag import gather_row_lags, lag_matrix, neighbour_table
from training.dataset import load_meta


class SpatialLagRecomputer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enabled = bool(cfg["data"]["spatial_lag"]["enabled"])
        if not self.enabled:
            return
        processed = Path(cfg["data"]["paths"]["processed_dir"])
        hourly = np.load(processed / "pm25_hourly.npz")
        self.values, self.column_ids = hourly["values"], hourly["location_ids"]
        start = pd.Timestamp(str(hourly["start"]))

        self.meta = load_meta(str(processed), "train")
        self.scalers = Scalers.load(str(processed / "scalers.json"))
        stations = pd.read_csv(processed / "stations.csv")
        self.train_ids = sorted(int(i) for i in self.meta["location_id"].unique())
        self.coords = stations[stations["location_id"].isin(self.train_ids)][["location_id", "x", "y"]]

        self.channel = cfg["derived"]["feature_names"].index("spatial_lag")
        self.lags = cfg["data"]["temporal"]["lags"]
        step = pd.Timedelta(cfg["data"]["time"]["freq"])
        self.t_idx = ((self.meta["timestamp"] - start) / step).to_numpy().astype(int)
        pos = {sid: i for i, sid in enumerate(self.train_ids)}
        self.row_pos = self.meta["location_id"].map(pos).to_numpy()

    def scaled_channel(self, reference_ids) -> np.ndarray:
        """[N_train, T] spatial-lag values (scaled like the saved channel), NaN where missing."""
        table = neighbour_table(self.coords, reference_ids, self.cfg)
        lag_mat = lag_matrix(self.values, self.column_ids, table, self.train_ids)
        raw = gather_row_lags(lag_mat, self.row_pos, self.t_idx, self.lags)
        return ((raw - self.scalers.y_mean) / self.scalers.y_std).astype(np.float32)

    def apply(self, base_ds, fit_idx, val_idx):
        """Return (dataset with the fold's lag channel, fit_idx, val_idx) with unusable rows removed.
        `base_ds` is the FULL train dataset; indices are positions in it."""
        fit_idx, val_idx = np.asarray(fit_idx), np.asarray(val_idx)
        if not self.enabled:
            return base_ds, fit_idx, val_idx
        reference = sorted(int(i) for i in self.meta.iloc[fit_idx]["location_id"].unique())
        channel = self.scaled_channel(reference)

        ds = copy.copy(base_ds)                       # shares the big patch array
        X = base_ds.X_temporal.copy()
        X[:, :, self.channel] = channel
        ds.X_temporal = X
        ok = ~np.isnan(channel).any(axis=1)
        return ds, fit_idx[ok[fit_idx]], val_idx[ok[val_idx]]

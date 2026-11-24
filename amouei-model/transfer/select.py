"""Which source stations look like Manila? A distance between PM2.5 distributions (maximum mean discrepancy), not a judgement about countries.

For every candidate station the distribution of its hourly samples [log(1 + PM2.5), sin(hour), cos(hour)] is compared with the same samples of the
Manila TRAINING stations by the squared maximum mean discrepancy with an RBF kernel (bandwidth by the median heuristic on the pooled sample).
Low = similar level, spread and daily cycle. It uses PM2.5 and the hour of day only (the model's own inputs); no weather.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _features(df: pd.DataFrame) -> np.ndarray:
    hour = pd.to_datetime(df["timestamp"]).dt.hour.to_numpy()
    return np.column_stack([np.log1p(df["pm25"].to_numpy(dtype=float)), np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)])


def _rbf(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Squared MMD (biased V-statistic): mean k(x,x) + mean k(y,y) - 2 mean k(x,y). 0 for identical samples."""
    return float(_rbf(x, x, gamma).mean() + _rbf(y, y, gamma).mean() - 2 * _rbf(x, y, gamma).mean())


def rank_stations(source: pd.DataFrame, target: pd.DataFrame, max_n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Candidate stations ordered from the most Manila-like. `source` and `target` are long tables (location_id, timestamp, pm25).
    Each side is subsampled to `max_n` rows per station (a seeded random choice), so a station with 30,000 hours does not cost 400 times a short one."""
    rng = np.random.default_rng(seed)

    def sample(df: pd.DataFrame) -> np.ndarray:
        return _features(df.iloc[rng.permutation(len(df))[:max_n]]) if len(df) else np.empty((0, 3))

    tgt = _features(target.iloc[rng.permutation(len(target))[:max_n * 4]])
    scale = np.median(np.abs(tgt - np.median(tgt, axis=0)), axis=0) + 1e-9
    gamma = 1.0 / (2.0 * np.median(scale) ** 2 * 3)                      # a bandwidth from the spread of the target's features
    rows = []
    for sid, grp in source.groupby("location_id"):
        x = sample(grp)
        if len(x) < 50:
            continue
        rows.append({"location_id": int(sid), "hours": int(len(grp)), "mean_pm25": float(grp["pm25"].mean()), "std_pm25": float(grp["pm25"].std()),
                     "mmd2": mmd2(x, tgt, gamma)})
    out = pd.DataFrame(rows, columns=["location_id", "hours", "mean_pm25", "std_pm25", "mmd2"])
    return out.sort_values("mmd2").reset_index(drop=True)

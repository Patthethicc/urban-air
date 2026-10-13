"""Temporal branch inputs per sensor: PM2.5 lags + spatial lag + optional rolling stats + cyclical hour."""
import numpy as np
import pandas as pd

from config_parser import feature_base_names


def add_cyclical_hour(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """sin_hour, cos_hour = sin/cos(2*pi*hour/24)."""
    ts = cfg["data"]["columns"]["timestamp"]
    hour = df[ts].dt.hour.to_numpy()
    df = df.copy()
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    return df


def _check_regular(df: pd.DataFrame, cfg: dict) -> None:
    """Shifting by rows equals shifting by time only if every sensor has a gap-free hourly index."""
    c = cfg["data"]["columns"]
    step = pd.Timedelta(cfg["data"]["time"]["freq"])
    diffs = df.groupby(c["group"])[c["timestamp"]].diff().dropna()
    if (diffs != step).any():
        raise ValueError("timestamps are not a strict hourly index per sensor; run clean_loader.load_clean")


def add_lags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Group by sensor, chronological. For every base feature and lag L create `<base>_t-L`
    (pm25, spatial_lag, roll_mean, roll_std, sin_hour, cos_hour as enabled). Rolling mean/std end at
    each step (window `rolling_window`), so a step at t-L only sees values up to t-L."""
    c = cfg["data"]["columns"]
    g, ts, tg = c["group"], c["timestamp"], c["target"]
    t = cfg["data"]["temporal"]

    df = df.sort_values([g, ts]).reset_index(drop=True)
    _check_regular(df, cfg)
    df["pm25"] = df[tg]
    w = t.get("rolling_window")
    if w:
        grp = df.groupby(g)[tg]
        df["roll_mean"] = grp.transform(lambda s: s.rolling(w, min_periods=w).mean())
        df["roll_std"] = grp.transform(lambda s: s.rolling(w, min_periods=w).std())

    base = feature_base_names(cfg["data"])
    missing = [b for b in base if b not in df.columns]
    if missing:
        raise KeyError(f"columns {missing} missing before add_lags (spatial lag / cyclical hour not added?)")

    new = {}
    for lag in t["lags"]:
        shifted = df.groupby(g)[base].shift(lag)
        for b in base:
            new[f"{b}_t-{lag}"] = shifted[b]
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def feature_columns(cfg: dict) -> list[list[str]]:
    """Columns for each of the T steps, oldest first, in the F-order from data.yaml:
    [pm25, (spatial_lag), (roll_mean, roll_std), (sin_hour, cos_hour)]."""
    base = feature_base_names(cfg["data"])
    return [[f"{b}_t-{lag}" for b in base] for lag in cfg["data"]["temporal"]["lags"]]


def build_windows(df: pd.DataFrame, cfg: dict):
    """One sample per (sensor, timestamp t): sequence of lag features -> target PM2.5 at t.

    Windows never cross sensors. Drop any sample whose lag hours are missing (an un-interpolated
    offline gap) or whose target is missing.

    Returns
      X_temporal : [N, T, F]   UNSCALED
      y          : [N, 1]      unscaled PM2.5 at t
      meta       : DataFrame (location_id, timestamp) per row -- aligner + partitioner use this
    """
    c = cfg["data"]["columns"]
    g, ts, tg = c["group"], c["timestamp"], c["target"]
    if cfg["data"]["temporal"].get("cyclical_hour"):
        df = add_cyclical_hour(df, cfg)
    df = add_lags(df, cfg)

    steps = feature_columns(cfg)
    flat = [col for step in steps for col in step]
    keep = df[flat + [tg]].notna().all(axis=1)
    sub = df.loc[keep]

    n, t_len, f_len = len(sub), len(steps), len(steps[0])
    X = sub[flat].to_numpy(dtype=np.float32).reshape(n, t_len, f_len)
    y = sub[[tg]].to_numpy(dtype=np.float32)
    meta = pd.DataFrame({
        "location_id": sub[g].to_numpy().astype("int64"),
        "timestamp": sub[ts].to_numpy(),
    }).reset_index(drop=True)
    return X, y, meta

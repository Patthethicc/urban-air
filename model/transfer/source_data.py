"""Source cities (data/Other Countries/<country>/<city>/, written by other-country-scrapers) -> hourly frames and model windows.

A source folder holds locations.csv (name, location_id, latitude, longitude), sensors/openaq_<id>_*.csv (the Manila layout) and source.json
(epsg = the UTM zone of the city, timezone = its local time zone). There is no raster: the windows have no patches, only the temporal input
[pm25, spatial_lag, sin_hour, cos_hour] per lag step, built by the SAME functions as for Manila (resample, gap filling, spatial lag, lags).
The spatial lag of a source station uses the other stations of the SAME city only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def list_sources(root) -> list[str]:
    """The city folders under `root` (relative names like 'vietnam/hanoi') that have a source.json and at least one sensor file."""
    root = Path(root)
    return sorted(p.parent.relative_to(root).as_posix() for p in root.rglob("source.json") if any((p.parent / "sensors").glob("*.csv")))


def source_meta(folder) -> dict:
    meta = json.loads((Path(folder) / "source.json").read_text(encoding="utf-8"))
    if not meta.get("epsg") or not meta.get("timezone"):
        raise ValueError(f"{folder}/source.json has no epsg / timezone: write the dataset again with the current other-country-scrapers/openaq_scraper.py")
    return meta


def load_city(folder, cfg: dict) -> pd.DataFrame:
    """One city -> the clean hourly frame of the main pipeline (columns data.columns: group, timestamp (naive LOCAL time), target, x, y)."""
    from pyproj import Transformer

    from preprocessing.clean_loader import find_sensor_file, interpolate_gaps, resample_hourly

    folder = Path(folder)
    meta = source_meta(folder)
    d, c = cfg["data"], cfg["data"]["columns"]
    g, ts, tg, xc, yc = c["group"], c["timestamp"], c["target"], c["x"], c["y"]
    lo, hi = d["clean"]["min_value"], d["clean"]["max_value"]
    loc = pd.read_csv(folder / "locations.csv")
    to_xy = Transformer.from_crs("EPSG:4326", f"EPSG:{int(meta['epsg'])}", always_xy=True)
    frames = []
    for _, st in loc.iterrows():
        file = find_sensor_file(str(folder / "sensors"), int(st["location_id"]))
        if file is None:
            continue
        raw = pd.read_csv(file, usecols=[ts, tg])
        stamp = pd.to_datetime(raw[ts], utc=True).dt.tz_convert(meta["timezone"]).dt.tz_localize(None)     # the city's own clock: the hour of day is local
        value = pd.to_numeric(raw[tg], errors="coerce")
        value = value.where(value.between(lo, hi))
        x, y = to_xy.transform(float(st["longitude"]), float(st["latitude"]))
        frames.append(pd.DataFrame({g: int(st["location_id"]), ts: stamp, tg: value, xc: x, yc: y}).dropna(subset=[tg]))
    if not frames:
        raise FileNotFoundError(f"{folder}: no sensor files matched locations.csv")
    return interpolate_gaps(resample_hourly(pd.concat(frames, ignore_index=True), cfg), cfg)


def city_windows(folder, cfg: dict, id_offset: int = 0, exclude_months=()):
    """(X [N,T,F] unscaled, y [N,1], meta DataFrame(location_id, timestamp)) of one city, in the F-order of the Manila model.
    `id_offset` is added to every location id so several cities never share an id; `exclude_months` drops windows whose target hour lies in those months."""
    from preprocessing.spatial_lag import add_spatial_lag
    from preprocessing.temporal_window_builder import build_windows

    df = load_city(folder, cfg)
    g = cfg["data"]["columns"]["group"]
    if "spatial_lag" in cfg["derived"]["feature_names"]:
        ids = sorted(int(i) for i in df[g].unique())
        df = add_spatial_lag(df, {"train": ids}, cfg)                   # every station of the city is a neighbour candidate (it never counts itself)
    X, y, meta = build_windows(df, cfg)
    meta["location_id"] = meta["location_id"] + int(id_offset)
    if exclude_months:
        keep = ~pd.to_datetime(meta["timestamp"]).dt.month.isin(list(exclude_months)).to_numpy()
        X, y, meta = X[keep], y[keep], meta.loc[keep].reset_index(drop=True)
    return X, y, meta


def hourly_long(folder, cfg: dict) -> pd.DataFrame:
    """The clean hourly series of a city as a long table (location_id, timestamp, pm25), for the station ranking."""
    c = cfg["data"]["columns"]
    df = load_city(folder, cfg)[[c["group"], c["timestamp"], c["target"]]].dropna()
    df.columns = ["location_id", "timestamp", "pm25"]
    return df


def manila_training_sample(cfg: dict) -> pd.DataFrame:
    """The Manila TRAINING stations' hourly PM2.5 (location_id, timestamp, pm25) from the preprocessed data: the target the sources are compared
    with. Test stations are never used, so the choice of source stations does not look at them."""
    processed = Path(cfg["data"]["paths"]["processed_dir"])
    z = np.load(processed / "pm25_hourly.npz")
    values, ids, start = z["values"], z["location_ids"], pd.Timestamp(str(z["start"]))
    train_ids = set(int(i) for i in pd.read_csv(processed / "meta_train.csv")["location_id"].unique())
    stamps = start + pd.to_timedelta(np.arange(values.shape[0]), unit="h")
    rows = [pd.DataFrame({"location_id": int(i), "timestamp": stamps, "pm25": values[:, j]}).dropna() for j, i in enumerate(ids) if int(i) in train_ids]
    return pd.concat(rows, ignore_index=True)

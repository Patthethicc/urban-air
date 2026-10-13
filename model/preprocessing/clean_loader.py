"""Per-sensor raw readings -> clean hourly frame (Preprocessing.txt: spatiotemporal alignment).

Frame columns follow data.yaml `columns`: group, timestamp (naive local time), target, x, y.
"""
import glob
import os

import numpy as np
import pandas as pd


def load_sensor_locations(cfg: dict) -> pd.DataFrame:
    """Sensors inside the Master GeoTIFF extent, with projected x/y (EPSG:32651).

    Uses the gpkg when paths.sensors_gpkg is set, otherwise the lat/lon CSV."""
    import geopandas as gpd
    import rasterio

    d = cfg["data"]
    c = d["columns"]
    if d["paths"]["sensors_gpkg"]:
        gdf = gpd.read_file(d["paths"]["sensors_gpkg"]).to_crs(d["crs"])
    else:
        raw = pd.read_csv(d["paths"]["sensor_locations_csv"])
        gdf = gpd.GeoDataFrame(
            raw[[c["group"]]],
            geometry=gpd.points_from_xy(raw[c["lon"]], raw[c["lat"]]),
            crs="EPSG:4326",
        ).to_crs(d["crs"])

    out = pd.DataFrame({
        c["group"]: gdf[c["group"]].astype("int64").to_numpy(),
        c["x"]: gdf.geometry.x.to_numpy(),
        c["y"]: gdf.geometry.y.to_numpy(),
    }).drop_duplicates(c["group"])

    with rasterio.open(d["paths"]["master_raster"]) as src:
        b = src.bounds
    inside = out[c["x"]].between(b.left, b.right) & out[c["y"]].between(b.bottom, b.top)
    return out.loc[inside].reset_index(drop=True)


def load_raw_readings(cfg: dict) -> pd.DataFrame:
    """Read every openaq_<location_id>_*.csv in paths.sensor_csv_dir for sensors inside the raster
    extent. Columns: group, timestamp, target, x, y. Out-of-range values become NaN."""
    d = cfg["data"]
    c = d["columns"]
    g, ts, tg, xc, yc = c["group"], c["timestamp"], c["target"], c["x"], c["y"]
    lo, hi = d["clean"]["min_value"], d["clean"]["max_value"]

    frames = []
    for _, sensor in load_sensor_locations(cfg).iterrows():
        sid = int(sensor[g])
        files = glob.glob(os.path.join(d["paths"]["sensor_csv_dir"], "**", f"openaq_{sid}_*.csv"), recursive=True)
        if not files:
            print(f"  warning: no CSV for sensor {sid}, skipped")
            continue
        raw = pd.read_csv(files[0], usecols=[ts, tg])
        stamp = pd.to_datetime(raw[ts], utc=True).dt.tz_convert(d["timezone"]).dt.tz_localize(None)
        value = pd.to_numeric(raw[tg], errors="coerce")
        value = value.where(value.between(lo, hi))
        frame = pd.DataFrame({g: sid, ts: stamp, tg: value, xc: sensor[xc], yc: sensor[yc]})
        frames.append(frame.dropna(subset=[tg]))
    if not frames:
        raise FileNotFoundError("no sensor CSVs matched the sensors inside the raster extent")
    return pd.concat(frames, ignore_index=True)


def resample_hourly(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Strict 1 h index per sensor (mean within the hour); x/y are constant per sensor."""
    c = cfg["data"]["columns"]
    g, ts, tg, xc, yc = c["group"], c["timestamp"], c["target"], c["x"], c["y"]
    freq = cfg["data"]["time"]["freq"]

    parts = []
    for sid, grp in df.groupby(g, sort=True):
        series = grp.set_index(ts)[tg].sort_index().resample(freq).mean()
        part = series.rename(tg).rename_axis(ts).reset_index()
        part[g], part[xc], part[yc] = sid, grp[xc].iloc[0], grp[yc].iloc[0]
        parts.append(part[[g, ts, tg, xc, yc]])
    return pd.concat(parts, ignore_index=True)


def _fill_short_gaps(s: pd.Series, max_gap: int) -> pd.Series:
    """Linear interpolation inside gaps of at most `max_gap` steps; longer gaps stay entirely NaN.
    (pandas' `limit=` would fill the first `max_gap` hours of a long gap, which fabricates data.)"""
    missing = s.isna()
    if not missing.any():
        return s
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    filled = s.interpolate(method="linear", limit_area="inside")
    return filled.mask(missing & (run_len > max_gap))


def interpolate_gaps(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Linear interpolation per sensor, capped at time.max_gap_hours (never fabricate across long downtime)."""
    c = cfg["data"]["columns"]
    g, ts, tg = c["group"], c["timestamp"], c["target"]
    max_gap = int(cfg["data"]["time"]["max_gap_hours"])
    df = df.sort_values([g, ts]).copy()
    df[tg] = df.groupby(g, group_keys=False)[tg].apply(lambda s: _fill_short_gaps(s, max_gap))
    return df.reset_index(drop=True)


def load_clean(cfg: dict) -> pd.DataFrame:
    """load_raw_readings -> resample_hourly -> interpolate_gaps. Sorted by (group, timestamp).
    Rows that remain NaN are real outages; the window builder drops any sample that touches them."""
    return interpolate_gaps(resample_hourly(load_raw_readings(cfg), cfg), cfg)

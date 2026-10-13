"""Snapshots -> spatiotemporal prediction volume, daily mean and summary numbers (Output.txt)."""
import numpy as np
import pandas as pd

from output.aqi_maps import categorize


def expand_dates(dates: list[str], hours: list[int]) -> list[pd.Timestamp]:
    """--date 2025-07-01 -> the configured snapshot hours of that day (08:00, 12:00, 17:00, 21:00)."""
    return [pd.Timestamp(d).normalize() + pd.Timedelta(hours=int(h)) for d in dates for h in hours]


def group_by_date(timestamps: list[pd.Timestamp]) -> dict:
    out: dict = {}
    for ts in sorted(set(pd.Timestamp(t) for t in timestamps)):
        out.setdefault(ts.normalize(), []).append(ts)
    return out


def write_volume(stack: np.ndarray, transform, crs, path: str, descriptions: list[str]) -> str:
    """stack [snapshots, rows, cols] -> one GeoTIFF, one band per snapshot (band description = its hour)."""
    import rasterio

    with rasterio.open(path, "w", driver="GTiff", height=stack.shape[1], width=stack.shape[2], count=stack.shape[0],
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan, compress="lzw") as dst:
        for i in range(stack.shape[0]):
            dst.write(stack[i].astype("float32"), i + 1)
        dst.descriptions = tuple(descriptions)
    return path


def daily_mean(stack: np.ndarray) -> np.ndarray:
    """Mean of the snapshots of one day. A cell missing in ANY snapshot stays missing (no partial means)."""
    return np.mean(stack, axis=0)


def summarize_date(stamps: list[pd.Timestamp], pred: np.ndarray, width: np.ndarray, cfg: dict) -> dict:
    """Numbers behind the maps: level, share of cells above the 35 and 50 ug/m3 limits, AQI category shares."""
    o = cfg["output"]
    names = [b["label"] for b in o["aqi"]["bins"]]
    snaps = []
    for ts, p, w in zip(stamps, pred, width):
        v = p[np.isfinite(p)]
        cats = categorize(p, cfg)
        snaps.append({
            "hour": ts.strftime("%H:%M"), "cells": int(v.size),
            "min": float(v.min()) if v.size else None, "mean": float(v.mean()) if v.size else None,
            "max": float(v.max()) if v.size else None,
            f"cells_above_{o['highlight_threshold']}": int((v > o["highlight_threshold"]).sum()),
            f"share_above_{o['highlight_threshold']}": float((v > o["highlight_threshold"]).mean()) if v.size else None,
            "aqi_category_share": {n: float(np.nanmean(cats == i)) if v.size else None for i, n in enumerate(names)},
            "mean_interval_width": float(np.nanmean(w)) if np.isfinite(w).any() else None,
        })
    out = {"date": stamps[0].strftime("%Y-%m-%d"), "snapshots": snaps}
    if len(stamps) > 1:
        mean = daily_mean(pred)
        v = mean[np.isfinite(mean)]
        g = o["daily_guideline"]
        out["daily_mean"] = {
            "snapshots_used": [t.strftime("%H:%M") for t in stamps], "cells": int(v.size),
            "mean": float(v.mean()) if v.size else None, "max": float(v.max()) if v.size else None,
            f"cells_above_{g}": int((v > g).sum()), f"share_above_{g}": float((v > g).mean()) if v.size else None,
            "note": "no GeoCP interval is defined for the daily mean (the intervals belong to single snapshots)",
        }
    return out

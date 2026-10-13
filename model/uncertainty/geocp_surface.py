"""Per-cell GeoCP intervals as aligned raster layers (Validation.txt: every grid cell in the domain).

Input : a point-prediction raster (one band, any model) in the projected CRS of the preprocessing.
Output: a 4-band GeoTIFF with the SAME extent, resolution and CRS: prediction, lower, upper, width.
        lower is clipped at 0 after the interval is formed; width = upper - lower.
"""
import numpy as np

from uncertainty.geocp import GeoCP

BANDS = ["prediction", "lower", "upper", "width"]


def interval_layers(pred: np.ndarray, transform, model: GeoCP, lower_bound: float = 0.0, n_grid: int = 2000):
    """(lower, upper, width) arrays for a 2-D prediction array (NaN = no prediction). Lower is clipped at
    `lower_bound` after the interval is formed; width = upper - lower. Cells with no calibration weight
    nearby get NaN."""
    from rasterio.transform import xy as cell_xy

    valid = np.isfinite(pred)
    rows, cols = np.nonzero(valid)
    xs, ys = cell_xy(transform, rows, cols, offset="center")
    q = np.full(pred.shape, np.nan)
    q[valid] = model.quantiles_grid(np.column_stack([xs, ys]), n_grid=n_grid)
    q[~np.isfinite(q)] = np.nan
    lower = np.maximum(pred - q, lower_bound)
    upper = pred + q
    return lower, upper, upper - lower


def build_interval_rasters(prediction_raster: str, model: GeoCP, out_path: str, lower_bound: float = 0.0,
                           band: int = 1, expected_crs: str | None = None, n_grid: int = 2000) -> dict:
    import rasterio
    from rasterio.transform import xy as cell_xy

    with rasterio.open(prediction_raster) as src:
        if expected_crs and str(src.crs) != str(expected_crs):
            raise ValueError(f"prediction raster is {src.crs}, expected {expected_crs} (distances must be in metres)")
        pred = src.read(band, masked=True).astype("float64")
        profile, transform = src.profile.copy(), src.transform

    valid = ~np.ma.getmaskarray(pred) & np.isfinite(pred.filled(np.nan))
    rows, cols = np.nonzero(valid)
    xs, ys = cell_xy(transform, rows, cols, offset="center")

    q = np.full(pred.shape, np.nan)
    q[valid] = model.quantiles_grid(np.column_stack([xs, ys]), n_grid=n_grid)
    q[~np.isfinite(q)] = np.nan                             # no calibration weight near the cell -> no interval

    values = pred.filled(np.nan)
    lower = np.maximum(values - q, lower_bound)
    upper = values + q
    layers = [values, lower, upper, upper - lower]

    profile.update(count=4, dtype="float32", nodata=np.nan, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, layer in enumerate(layers, start=1):
            dst.write(layer.astype("float32"), i)
        dst.descriptions = tuple(BANDS)
    ok = np.isfinite(layers[3])
    return {"cells": int(valid.sum()), "cells_with_interval": int(ok.sum()),
            "mean_width": float(layers[3][ok].mean()) if ok.any() else float("nan"), "path": str(out_path)}

"""Per-cell inference: a point prediction for EVERY grid cell of the Master GeoTIFF (Section 4.5).

A cell without a sensor has no PM2.5 history. Its history is the spatial lag of its k nearest stations
(Preprocessing.txt, "Spatial Lag"): at each lag hour the weighted mean of those stations' readings.
That stand-in fills the PM2.5 channel and the spatial-lag channel of the temporal sequence; the patch
(centred on the cell), the hour encoding and the scalers are those of training.

Caveat (also in docs/PROCESS.txt): the model was trained on sensors whose own lags are real readings,
and the GeoCP calibration scores come from those sensors, so intervals for stand-in cells describe the
model's error with real histories.
"""
import numpy as np
import pandas as pd

from preprocessing.spatial_lag import nearest_station_weights


def padded_raster(raster_path: str, patch_px: int):
    """The Master GeoTIFF as a zero-padded array [C, H + px - 1, W + px - 1] (NoData / NaN -> 0, the same
    convention as spatial_patch_extractor.extract_patch). Returns (padded, transform, crs, (H, W))."""
    import rasterio

    with rasterio.open(raster_path) as src:
        data = src.read().astype(np.float32)
        if src.nodata is not None:
            data[data == np.float32(src.nodata)] = 0.0
        transform, crs = src.transform, src.crs
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    before = patch_px // 2
    after = patch_px - before - 1
    return np.pad(data, ((0, 0), (before, after), (before, after))), transform, crs, data.shape[1:]


def gather_patches(padded: np.ndarray, rows: np.ndarray, cols: np.ndarray, patch_px: int) -> np.ndarray:
    """[n, C, px, px] patches centred on pixels (rows[i], cols[i]); identical to extract_patch for each."""
    windows = np.lib.stride_tricks.sliding_window_view(padded, (patch_px, patch_px), axis=(1, 2))
    return np.ascontiguousarray(windows[:, rows, cols].transpose(1, 0, 2, 3))


def standin_series(values: np.ndarray, column_ids, ref_ids, ref_xy: np.ndarray, cell_xy: np.ndarray,
                   hour_idx: np.ndarray, cfg: dict) -> np.ndarray:
    """Spatial lag of every cell at the given hours: [len(hour_idx), n_cells], NaN where none of the
    cell's k nearest reference stations reported. `values` [H, S] hourly PM2.5, columns = `column_ids`."""
    col = {int(c): i for i, c in enumerate(column_ids)}
    ref_cols = np.array([col[int(r)] for r in ref_ids])
    idx, w = nearest_station_weights(cell_xy, ref_xy, cfg)                       # [n, k]
    vals = values[np.ix_(hour_idx, ref_cols)][:, idx].astype(float)             # [h, n, k]
    seen = ~np.isnan(vals)
    num = (np.where(seen, vals, 0.0) * w[None]).sum(axis=-1)
    den = (seen * w[None]).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def cell_temporal_features(values, column_ids, ref_ids, ref_xy, cell_xy, t_idx: int, timestamp: pd.Timestamp,
                           cfg: dict) -> np.ndarray:
    """Raw (unscaled) X_temporal [n_cells, T, F] for the hour `timestamp` (index t_idx of the hourly matrix),
    in the F-order of data.yaml: [pm25, (spatial_lag), (roll_mean, roll_std), (sin_hour, cos_hour)]."""
    lags = cfg["data"]["temporal"]["lags"]
    names = cfg["derived"]["feature_names"]
    w = cfg["data"]["temporal"].get("rolling_window") or 1
    first = t_idx - max(lags) - (w - 1)
    if first < 0:
        raise ValueError(f"not enough history before this hour (need {max(lags) + w - 1} hours)")
    hours = np.arange(first, t_idx)
    series = standin_series(values, column_ids, ref_ids, ref_xy, cell_xy, hours, cfg)    # [h, n]

    steps = []
    for lag in lags:
        pos = (t_idx - lag) - first
        stamp = timestamp - pd.Timedelta(hours=int(lag))
        feats = []
        for name in names:
            if name in ("pm25", "spatial_lag"):
                feats.append(series[pos])                                   # the stand-in fills both channels
            elif name == "roll_mean":
                feats.append(series[pos - w + 1:pos + 1].mean(axis=0))
            elif name == "roll_std":
                feats.append(series[pos - w + 1:pos + 1].std(axis=0, ddof=1))
            elif name == "sin_hour":
                feats.append(np.full(series.shape[1], np.sin(2 * np.pi * stamp.hour / 24)))
            elif name == "cos_hour":
                feats.append(np.full(series.shape[1], np.cos(2 * np.pi * stamp.hour / 24)))
        steps.append(np.stack(feats, axis=1))                               # [n, F]
    return np.stack(steps, axis=1).astype(np.float32)                        # [n, T, F]


def predict_surface(model, scalers, padded: np.ndarray, shape, transform, hourly: dict, ref_ids, ref_xy,
                    timestamp: pd.Timestamp, cfg: dict, device: str = "cpu"):
    """Point prediction (ug/m3) for the cells of the raster. Returns (prediction [rows, cols] float32 with
    NaN where no stand-in history exists, output transform).

    hourly = {'values': [H, S], 'location_ids': [S], 'start': Timestamp}."""
    import torch
    from affine import Affine

    from affine_compat import compose
    from rasterio.transform import xy as cell_xy_of

    inf = cfg["inference"]
    stride, px = int(inf["stride_px"]), int(cfg["derived"]["patch_px"])
    height, width = shape
    r0, r1, c0, c1 = inf["window"] if inf["window"] else (0, height, 0, width)
    rows_1d = np.arange(r0 + stride // 2, r1, stride)
    cols_1d = np.arange(c0 + stride // 2, c1, stride)
    rr, cc = np.meshgrid(rows_1d, cols_1d, indexing="ij")
    rows, cols = rr.ravel(), cc.ravel()
    xs, ys = cell_xy_of(transform, rows, cols, offset="center")
    cell_xy = np.column_stack([xs, ys])

    t_idx = int(round((timestamp - hourly["start"]) / pd.Timedelta(cfg["data"]["time"]["freq"])))
    out = np.full(len(rows), np.nan, dtype=np.float32)
    model = model.to(device).eval()
    chunk = int(inf["chunk_cells"])
    with torch.no_grad():
        for a in range(0, len(rows), chunk):
            sl = slice(a, a + chunk)
            raw = cell_temporal_features(hourly["values"], hourly["location_ids"], ref_ids, ref_xy,
                                         cell_xy[sl], t_idx, timestamp, cfg)
            ok = ~np.isnan(raw).any(axis=(1, 2))
            if not ok.any():
                continue
            patches = scalers.transform_patches(gather_patches(padded, rows[sl][ok], cols[sl][ok], px))
            temporal = scalers.transform_temporal(raw[ok])
            pred = model(torch.from_numpy(patches).to(device), torch.from_numpy(temporal).to(device))
            idx = np.flatnonzero(ok) + a
            out[idx] = scalers.inverse_y(pred.cpu().numpy()).ravel().astype(np.float32)

    shape_out = (len(rows_1d), len(cols_1d))
    new_transform = compose(transform, Affine.translation(c0, r0), Affine.scale(stride))   # works on affine 2.x and 3.x
    return out.reshape(shape_out), new_transform


def write_prediction_raster(prediction: np.ndarray, transform, crs, path: str) -> str:
    import rasterio

    with rasterio.open(path, "w", driver="GTiff", height=prediction.shape[0], width=prediction.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan, compress="lzw") as dst:
        dst.write(prediction.astype("float32"), 1)
        dst.descriptions = ("pm25_prediction",)
    return path

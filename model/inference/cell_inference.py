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

from preprocessing.qc_convert import ID_BASE as QC_ID_BASE
from preprocessing.spatial_lag import nearest_station_weights
from runtime_guard import format_duration, guarded  # format_duration is also imported from here by older callers


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
                   hour_idx: np.ndarray, cfg: dict, neighbours=None) -> np.ndarray:
    """Spatial lag of every cell at the given hours: [len(hour_idx), n_cells], NaN where none of the
    cell's k nearest reference stations reported. `values` [H, S] hourly PM2.5, columns = `column_ids`.
    `neighbours` = (idx, w) from nearest_station_weights for these cells and these ref_xy, when already computed."""
    col = {int(c): i for i, c in enumerate(column_ids)}
    ref_cols = np.array([col[int(r)] for r in ref_ids])
    idx, w = neighbours if neighbours is not None else nearest_station_weights(cell_xy, ref_xy, cfg)   # [n, k]
    vals = values[np.ix_(hour_idx, ref_cols)][:, idx].astype(float)             # [h, n, k]
    seen = ~np.isnan(vals)
    num = (np.where(seen, vals, 0.0) * w[None]).sum(axis=-1)
    den = (seen * w[None]).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def openaq_only(timestamp, cfg: dict) -> bool:
    """True when the stand-in history of this hour may only use OpenAQ stations: its year is at or after
    inference.openaq_only_from_year (null = never). The year of the hour being predicted decides, not the year of its lags."""
    year = cfg["inference"].get("openaq_only_from_year")
    return year is not None and pd.Timestamp(timestamp).year >= int(year)


def reference_stations(ref_ids, ref_xy, only_openaq: bool, cfg: dict):
    """(ids, xy) of the stations whose readings may form the stand-in history: all the reference stations, or just the
    OpenAQ ones (the Quezon City network has ids from qc_convert.ID_BASE up)."""
    if not only_openaq:
        return ref_ids, ref_xy
    keep = np.array([int(r) < QC_ID_BASE for r in ref_ids])
    if not keep.any():
        raise ValueError(f"inference.openaq_only_from_year = {cfg['inference']['openaq_only_from_year']} asks for OpenAQ stations only, "
                         f"but none of the {len(ref_ids)} reference stations (inference.lag_reference = {cfg['inference']['lag_reference']}) "
                         "is an OpenAQ station; set it to null to use every reference station")
    return [r for r, k in zip(ref_ids, keep) if k], np.asarray(ref_xy)[keep]


def _first_history_hour(t_idx: int, cfg: dict) -> int:
    """Index of the earliest hourly row the features of hour `t_idx` read (lags plus the rolling window)."""
    lags = cfg["data"]["temporal"]["lags"]
    w = cfg["data"]["temporal"].get("rolling_window") or 1
    first = t_idx - max(lags) - (w - 1)
    if first < 0:
        raise ValueError(f"not enough history before this hour (need {max(lags) + w - 1} hours)")
    return first


def cell_temporal_features(values, column_ids, ref_ids, ref_xy, cell_xy, t_idx: int, timestamp: pd.Timestamp,
                           cfg: dict, neighbours=None) -> np.ndarray:
    """Raw (unscaled) X_temporal [n_cells, T, F] for the hour `timestamp` (index t_idx of the hourly matrix),
    in the F-order of data.yaml: [pm25, (spatial_lag), (roll_mean, roll_std), (sin_hour, cos_hour)].
    `neighbours`: see standin_series."""
    lags = cfg["data"]["temporal"]["lags"]
    names = cfg["derived"]["feature_names"]
    w = cfg["data"]["temporal"].get("rolling_window") or 1
    first = _first_history_hour(t_idx, cfg)
    hours = np.arange(first, t_idx)
    series = standin_series(values, column_ids, ref_ids, ref_xy, cell_xy, hours, cfg, neighbours)    # [h, n]

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


def _hour_index(timestamp: pd.Timestamp, hourly: dict, cfg: dict) -> int:
    """Row of `timestamp` in the hourly matrix, checked BEFORE any work: enough history before it, and its lag hours
    inside the data (the features read rows first .. t_idx - 1)."""
    step = pd.Timedelta(cfg["data"]["time"]["freq"])
    t_idx = int(round((timestamp - hourly["start"]) / step))
    _first_history_hour(t_idx, cfg)
    n_hours = len(hourly["values"])
    if t_idx > n_hours:
        last = hourly["start"] + (n_hours - 1) * step
        raise ValueError(f"{timestamp} is after the hourly readings (last reading {last}): no lag history for it")
    return t_idx


def _supports_reuse(model) -> bool:
    """True when the CNN embedding can be computed once and reused for every hour: a dual-branch model with its CNN
    on. A temporal-only model or anything with only forward(spatial, temporal) takes the plain path."""
    import torch

    return bool(getattr(model, "use_spatial", False)) and all(
        isinstance(getattr(model, name, None), torch.nn.Module) for name in ("spatial", "temporal", "head"))


def predict_surface(model, scalers, padded: np.ndarray, shape, transform, hourly: dict, ref_ids, ref_xy,
                    timestamps, cfg: dict, device: str = "cpu", *, reuse_embeddings: bool = True):
    """Point predictions (ug/m3) for the cells of the raster at one or several hours. Returns (prediction, output
    transform): [rows, cols] float32 for a single timestamp, [n_timestamps, rows, cols] for a sequence; NaN where no
    stand-in history exists.

    A cell's patch, and so its CNN embedding, is the same for every hour: each chunk computes the neighbours, the patches
    and the embedding once and reuses them for all hours (`reuse_embeddings`; same result as one forward pass per hour,
    which is what a temporal-only or forward-only model gets).

    From inference.openaq_only_from_year on, the hours of that year take their stand-in history from the OpenAQ reference
    stations only (`reference_stations`); earlier hours keep every reference station.

    hourly = {'values': [H, S], 'location_ids': [S], 'start': Timestamp}."""
    import time

    import torch
    from affine import Affine

    from affine_compat import cell_centers, compose

    single = isinstance(timestamps, (str, pd.Timestamp))
    stamps = [pd.Timestamp(t) for t in ([timestamps] if single else timestamps)]
    t_idx = [_hour_index(ts, hourly, cfg) for ts in stamps]                    # fail fast, before any chunk
    only = [openaq_only(ts, cfg) for ts in stamps]
    refs = {flag: reference_stations(ref_ids, ref_xy, flag, cfg) for flag in set(only)}   # flag -> (ids, xy)

    inf = cfg["inference"]
    stride, px = int(inf["stride_px"]), int(cfg["derived"]["patch_px"])
    height, width = shape
    r0, r1, c0, c1 = inf["window"] if inf["window"] else (0, height, 0, width)
    rows_1d = np.arange(r0 + stride // 2, r1, stride)
    cols_1d = np.arange(c0 + stride // 2, c1, stride)
    rr, cc = np.meshgrid(rows_1d, cols_1d, indexing="ij")
    rows, cols = rr.ravel(), cc.ravel()
    xs, ys = cell_centers(transform, rows, cols)
    cell_xy = np.column_stack([xs, ys])

    out = np.full((len(stamps), len(rows)), np.nan, dtype=np.float32)
    model = model.to(device).eval()
    reuse = reuse_embeddings and _supports_reuse(model)
    chunk = int(inf["chunk_cells"])
    total, started, next_mark = len(rows), time.time(), 0.05
    n_chunks = int(np.ceil(total / chunk))
    across = f"across {len(stamps)} snapshot{'s' if len(stamps) > 1 else ''}"
    show = bool(inf.get("progress", True)) and total >= 50_000
    with torch.no_grad(), guarded("per-cell inference") as guard:
        for i, a in enumerate(range(0, total, chunk), 1):
            guard.beat(f"chunk {i}/{n_chunks}, cells {a:,}-{min(a + chunk, total):,} of {total:,} {across}")
            sl = slice(a, a + chunk)
            neighbours = {flag: nearest_station_weights(cell_xy[sl], xy, cfg) for flag, (_, xy) in refs.items()}
            raw = []
            for ti, ts, flag in zip(t_idx, stamps, only):
                ids, xy = refs[flag]
                raw.append(cell_temporal_features(hourly["values"], hourly["location_ids"], ids, xy, cell_xy[sl], ti, ts, cfg,
                                                  neighbours[flag]))
            ok = np.stack([~np.isnan(r).any(axis=(1, 2)) for r in raw])                      # [hours, n]
            union = ok.any(axis=0)
            if union.any():
                patches = torch.from_numpy(scalers.transform_patches(
                    gather_patches(padded, rows[sl][union], cols[sl][union], px))).to(device)
                embedding = model.spatial(patches, dedupe=model.dedupe_patches) if reuse else None
                for h, ok_h in enumerate(ok):
                    if not ok_h.any():
                        continue
                    sel = torch.from_numpy(ok_h[union]).to(device)
                    temporal = torch.from_numpy(scalers.transform_temporal(raw[h][ok_h])).to(device)
                    pred = (model.head(embedding[sel], model.temporal(temporal)) if reuse
                            else model(patches[sel], temporal))
                    out[h, a + np.flatnonzero(ok_h)] = scalers.inverse_y(pred.cpu().numpy()).ravel()
            if show:
                done, elapsed = min(a + chunk, total), time.time() - started
                rate, frac = done / elapsed, done / total
                if a == 0 and total / rate > 300:
                    print(f"    {total:,} cells to predict {across}; at ~{rate:,.0f} cells/s this takes about "
                          f"{format_duration(total / rate)} (use --override inference.stride_px=5 or an "
                          "inference.window for a faster look)")
                if frac >= next_mark:
                    print(f"    {frac:5.1%}  chunk {i}/{n_chunks} {across}  elapsed {format_duration(elapsed)}  "
                          f"remaining ~{format_duration(elapsed / frac * (1 - frac))}")
                    next_mark += 0.05

    shape_out = (len(rows_1d), len(cols_1d))
    new_transform = compose(transform, Affine.translation(c0, r0), Affine.scale(stride))   # works on affine 2.x and 3.x
    out = out.reshape((len(stamps),) + shape_out)
    return (out[0] if single else out), new_transform


def write_prediction_raster(prediction: np.ndarray, transform, crs, path: str) -> str:
    import rasterio

    with rasterio.open(path, "w", driver="GTiff", height=prediction.shape[0], width=prediction.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan, compress="lzw") as dst:
        dst.write(prediction.astype("float32"), 1)
        dst.descriptions = ("pm25_prediction",)
    return path

"""Chunked inference on the FULL 10 m grid (10.6 M pixels), raster-native.

Design constraints that shaped this module
------------------------------------------
* A dense stand-in history would be 73 h x 10.6 M = 776 M floats (6 GB). It is never materialised:
  the grid is processed in row blocks, and each block carries only its own 73 h series.
* A 10.6 M-row CSV would be ~1.5 GB and no stage could hold it. The output is a 4-band float32
  GeoTIFF (prediction, lower, upper, usable) plus a small JSON summary.
* Interval bounds are computed by GeoCP at 500 m (2,147 cells, minutes) where the weight field is
  smooth in location, then resampled to 10 m. The provenance is recorded in the summary, because
  this is an interpolation of an interval field, not a per-pixel conformal computation.

Feature definition is patch-centred (preprocessing/patch_stack.py), so the predictors are the same
kind of object at training time (station patches) and inference time (every pixel).
"""
import json
import os

import numpy as np
import pandas as pd

import data
from inference.cell_inference import hourly_matrix
from preprocessing import patch_stack

CHUNK_ROWS = 32


def _stack_or_build(window_px=None):
    odd = dict(path=None)
    try:
        st = patch_stack.load()
        if window_px and st['meta']['window_px'] != int(window_px):
            raise FileNotFoundError('window changed')
        return st
    except FileNotFoundError:
        print('  patch stack missing (or window changed) -- building it now')
        st = patch_stack.build(window_px=window_px or patch_stack.WINDOW_PX)
        patch_stack.save(st)
        return st


def _hours_for(ts, n=73):
    return pd.date_range(pd.Timestamp(ts).floor('h') - pd.Timedelta(hours=n - 1),
                         pd.Timestamp(ts).floor('h'), freq='h')


def _time_and_interaction(ts, morph, feature_names):
    """Cyclical time terms and the three hour x morphology interactions the arms use."""
    t = pd.Timestamp(ts)
    hour = t.hour + t.minute / 60.0
    doy = t.dayofyear
    dow = t.dayofweek
    out = {
        'hour_sin': np.full(len(morph['bldg_dens']), np.sin(2 * np.pi * hour / 24), dtype=np.float32),
        'hour_cos': np.full(len(morph['bldg_dens']), np.cos(2 * np.pi * hour / 24), dtype=np.float32),
        'dayofweek_sin': np.full(len(morph['bldg_dens']), np.sin(2 * np.pi * dow / 7), dtype=np.float32),
        'dayofweek_cos': np.full(len(morph['bldg_dens']), np.cos(2 * np.pi * dow / 7), dtype=np.float32),
        'month_sin': np.full(len(morph['bldg_dens']), np.sin(2 * np.pi * doy / 365.25), dtype=np.float32),
        'month_cos': np.full(len(morph['bldg_dens']), np.cos(2 * np.pi * doy / 365.25), dtype=np.float32),
    }
    out['hour_x_bldg_dens'] = out['hour_sin'] * morph['bldg_dens']
    out['hour_x_ndvi'] = out['hour_sin'] * morph['ndvi_mean']
    out['hour_x_aspect_ratio'] = out['hour_sin'] * morph['aspect_ratio_mean']
    return out


def _history_from_series(series, hour_pos):
    """Flat lag/rolling features at `hour_pos` from a [n_hours, n_cells] stand-in matrix."""
    s = series
    i = hour_pos
    out = {
        'pm25_lag_1': s[i], 'pm25_lag_2': s[i - 1], 'pm25_lag_3': s[i - 2], 'pm25_lag_4': s[i - 3],
        'pm25_diff_1_2': s[i] - s[i - 1],
    }
    for win, tag in ((24, '24'), (72, '72')):
        lo = max(0, i - win + 1)
        block = s[lo:i + 1]
        out['pm25_roll%s_mean' % tag] = np.nanmean(block, axis=0)
        out['pm25_roll%s_std' % tag] = np.nanstd(block, axis=0)
    return {k: v.astype(np.float32) for k, v in out.items()}


def _standin_block(W, idx, w):
    """[n_hours, n_cells] NaN-aware stand-in, matching cell_inference.standin_series.

    A plain weighted sum would turn a pixel NaN as soon as ONE of its four neighbours missed an hour.
    The 500 m path divides by the weight that actually reported, so a pixel survives on one neighbour;
    this is the same formula, and it is why the 10 m mask looked so much sparser than the 500 m one.
    """
    vals = W[:, idx]                      # [n_hours, n_cells, k]
    seen = np.isfinite(vals)
    num = np.where(seen, vals, 0.0) * w[None]
    den = seen * w[None]
    tot = den.sum(axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(tot > 0, num.sum(axis=-1) / tot, np.nan)
    return out.astype(np.float32), (tot / w.sum(axis=-1)[None]).astype(np.float32)


def _intervals(ts, shape, transform):
    """Resample the 500 m lower/upper field onto the 10 m grid. None when unavailable."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    day = pd.Timestamp(ts).strftime('%Y%m%d')
    hh = pd.Timestamp(ts).strftime('%H%M')
    src_path = os.path.join(data.OUT_DIR, 'surfaces', day, '%sT%s.tif' % (day, hh))
    if not os.path.exists(src_path):
        return None, None, None
    lo = np.full(shape, np.nan, dtype=np.float32)
    hi = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(src_path) as src:
        for band, dest in ((2, lo), (3, hi)):
            reproject(rasterio.band(src, band), dest,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=src.crs,
                      resampling=Resampling.bilinear)
    return lo, hi, src_path



def pixel_of_xy(meta, x, y):
    """Raster row/col indices for projected coordinates (clipped to the raster)."""
    a, b, c, d, e, f = meta['transform']
    cols = np.floor((x - c) / a).astype(int)
    rows = np.floor((y - f) / e).astype(int)
    H, W = meta['shape']
    return np.clip(rows, 0, H - 1), np.clip(cols, 0, W - 1)


def fit_patch_model(df, stack, params, n_estimators, feature_names, seed=42):
    """Train the SAME feature definition used at inference: patch statistics at the station pixel.

    This is the step that makes a 10 m product legitimate. The 500 m model was trained on cell
    aggregates; feeding it patch values instead produces predictions well outside its training
    response (the smoke test measured a mean of 77 vs 19 ug/m3). Refitting on patch features at the
    station pixels removes that mismatch.
    """
    import xgboost as xgb
    from preprocessing import prep

    stations = (df[['Location', 'Grid_ID', 'x_coordinate', 'y_coordinate']]
                .drop_duplicates('Location').reset_index(drop=True))
    rows, cols = pixel_of_xy(stack['meta'], stations['x_coordinate'].to_numpy(float),
                             stations['y_coordinate'].to_numpy(float))
    patch = {k: stack['arrays'][k][rows, cols] for k in patch_stack.FEATURE_COLUMNS}
    patch['Location'] = stations['Location'].values

    import pandas as pd
    p = pd.DataFrame(patch)
    d = df.merge(p, on='Location', how='left', suffixes=('', '_patch'))
    for k in patch_stack.FEATURE_COLUMNS:
        if k + '_patch' in d.columns:
            d[k] = d[k + '_patch']
    # interactions follow from the patch values
    d['hour_x_bldg_dens'] = d['hour_sin'] * d['bldg_dens']
    d['hour_x_ndvi'] = d['hour_sin'] * d['ndvi_mean']
    d['hour_x_aspect_ratio'] = d['hour_sin'] * d['aspect_ratio_mean']

    X = d[feature_names].to_numpy(dtype=np.float32)
    y = np.log1p(d['Value'].to_numpy(float))
    model = xgb.XGBRegressor(n_estimators=int(n_estimators), random_state=seed,
                             tree_method='hist', n_jobs=-1, **params)
    model.fit(X, y)
    print('  patch model fitted on %s rows from %d stations (%d features)'
          % (f'{len(d):,}', d['Location'].nunique(), len(feature_names)))
    return model


def run(model, feature_names, dates, out_dir=None, window_px=None, chunk_rows=CHUNK_ROWS,
        row_slice=None, col_slice=None, station_df=None, verbose=True, fill_morphology=False,
        pad_to=None):
    """Predict every 10 m pixel for the four snapshot hours of each date.

    `row_slice`/`col_slice` exist so the whole path can be exercised on a small window before
    committing to a 10.6 M-pixel run.
    """
    import rasterio
    from rasterio.transform import from_origin
    from scipy.spatial import cKDTree

    from preprocessing import prep

    out_dir = out_dir or os.path.join(data.OUT_DIR, 'surfaces10m')
    os.makedirs(out_dir, exist_ok=True)

    st = _stack_or_build(window_px)
    if pad_to is not None:
        # Edge-replicate the feature stack until it covers `pad_to` = (left, bottom, right, top) in
        # metres, so cells inside the study boundary but outside the DSM strip still get a
        # prediction (morphology taken from the nearest valid pixel -- same rule as fill_morphology).
        import numpy as _np
        a_, _, c_, _, e_, f_ = st['meta']['transform'][:6]
        left, bottom, right, top = pad_to
        pad_l = max(0, int(round((c_ - left) / a_)))
        pad_r = max(0, int(round((right - (c_ + a_ * st['meta']['shape'][1])) / a_)))
        pad_t = max(0, int(round((top - f_) / abs(e_))))
        pad_b = max(0, int(round(((f_ + e_ * st['meta']['shape'][0]) - bottom) / abs(e_)))) if bottom < (f_ + e_ * st['meta']['shape'][0]) else 0
        if pad_l or pad_r or pad_t or pad_b:
            st['arrays'] = {k: _np.pad(v, ((pad_t, pad_b), (pad_l, pad_r)), mode='edge')
                            for k, v in st['arrays'].items()}
            st['valid'] = _np.pad(st['valid'], ((pad_t, pad_b), (pad_l, pad_r)), mode='edge')
            st['meta']['shape'] = st['arrays'][next(iter(st['arrays']))].shape
            st['meta']['transform'] = (a_, 0.0, c_ - pad_l * a_, 0.0, e_, f_ + pad_t * abs(e_))
            st['meta']['pad'] = (pad_t, pad_b, pad_l, pad_r)
            if verbose:
                print('    padded stack by %d/%d rows, %d/%d cols -> %s'
                      % (pad_t, pad_b, pad_l, pad_r, st['meta']['shape']))
    meta = st['meta']
    H, W = meta['shape']
    r0, r1 = row_slice if row_slice else (0, H)
    c0, c1 = col_slice if col_slice else (0, W)

    # Filling: a pixel without morphology (outside the DSM, water masked out) takes the features of
    # its NEAREST valid pixel. Declared wherever it is used: it is a smooth extrapolation of the
    # morphological surfaces, not a measurement. The `filled` band of the GeoTIFF records which
    # pixels were extrapolated, so nothing is hidden.
    fill_idx = None
    if fill_morphology:
        from scipy.ndimage import distance_transform_edt
        valid_all = st['valid']
        _, (ir, ic) = distance_transform_edt(~valid_all, return_indices=True)
        fill_idx = (ir.astype('int32'), ic.astype('int32'))

    df = prep.df_clean if station_df is None else station_df
    station_ids = sorted(df['Location'].unique())
    Wmat = hourly_matrix(df[df['Location'].isin(station_ids)], station_ids)
    station_xy = (df[df['Location'].isin(station_ids)]
                  .groupby('Location')[['x_coordinate', 'y_coordinate']].first()
                  .reindex(station_ids).values)
    tree = cKDTree(station_xy)

    written, frames = [], {}
    for d in dates:
        for hour in (8, 12, 17, 21):
            ts = pd.Timestamp(d).normalize() + pd.Timedelta(hours=hour)
            hours = _hours_for(ts)
            hour_pos = len(hours) - 1

            if ts not in Wmat.index:
                end = max(Wmat.index.max(), ts)
                Wmat = Wmat.reindex(pd.date_range(Wmat.index.min(), end, freq='h'))
            block = Wmat.loc[hours]
            W3 = block.to_numpy(dtype=np.float32)                  # [73, n_stations]

            pred = np.full((r1 - r0, c1 - c0), np.nan, dtype=np.float32)
            hist_ok = np.zeros((r1 - r0, c1 - c0), dtype=bool)
            for rs in range(r0, r1, chunk_rows):
                re = min(rs + chunk_rows, r1)
                rr, cc = np.meshgrid(np.arange(rs, re), np.arange(c0, c1), indexing='ij')
                rr, cc = rr.ravel(), cc.ravel()
                cx, cy = patch_stack.xy_of_pixels(meta, rr, cc)
                dist, idx = tree.query(np.c_[cx, cy], k=min(4, len(station_ids)))
                dist = np.atleast_2d(dist)
                idx = np.atleast_2d(idx)
                w = 1.0 / np.maximum(dist, 1.0)
                w = (w / w.sum(axis=1, keepdims=True)).astype(np.float32)

                series, coverage = _standin_block(W3, idx, w)      # [73, n_cells] + coverage
                feat = _history_from_series(series, hour_pos)
                if fill_idx is not None:
                    have = st['valid'][rr, cc]
                    rr_f = np.where(have, rr, fill_idx[0][rr, cc])
                    cc_f = np.where(have, cc, fill_idx[1][rr, cc])
                else:
                    rr_f, cc_f = rr, cc
                morph = patch_stack.features_at(st, rr_f, cc_f)
                feat.update(morph)
                feat.update(_time_and_interaction(ts, morph, feature_names))

                missing = [f for f in feature_names if f not in feat]
                if missing:
                    raise ValueError('these model features were not built: %s' % missing)
                X = np.column_stack([feat[f] for f in feature_names]).astype(np.float32)
                p = np.expm1(model.predict(X)).astype(np.float32)
                pred[rs - r0:re - r0, :] = p.reshape(re - rs, c1 - c0)
                # A pixel with no stand-in history must not be painted: XGBoost answers NaN
                # features through its default branch, which is how the 500 m surface once grew a
                # false 240 ug/m3 wedge over Manila Bay. Require the most recent hour to be finite.
                ok = np.isfinite(series[hour_pos]) & np.isfinite(series[hour_pos - 1])
                hist_ok[rs - r0:re - r0, :] = ok.reshape(re - rs, c1 - c0)

            if fill_morphology:
                # Extrapolation policy, in three steps:
                #   1. honest  = pixel has real morphology, real stand-in history and a prediction
                #   2. inpaint = any pixel still unpredicted takes the nearest predicted value
                #                (masked water, the DSM fringe, and the padded strip beyond the raster)
                #   3. clamp   = every extrapolated pixel is limited to the 0.5-99.5 percentile range
                #                of the honest pixels, so an edge artefact cannot invent a 290 ug/m3 cell
                honest = st['valid'][r0:r1, c0:c1] & hist_ok & np.isfinite(pred)
                bad = ~np.isfinite(pred)
                if bad.any() and not bad.all():
                    from scipy.ndimage import distance_transform_edt
                    _, (jr, jc) = distance_transform_edt(bad, return_indices=True)
                    pred = np.where(bad, pred[jr, jc], pred)
                if honest.any():
                    lo_ok, hi_ok = np.percentile(pred[honest], [0.5, 99.5])
                    pred = np.where(honest, pred, np.clip(pred, lo_ok, hi_ok))
                usable = np.isfinite(pred)
                filled = ~honest & usable
            else:
                usable = st['valid'][r0:r1, c0:c1] & np.isfinite(pred) & hist_ok
                filled = np.zeros_like(usable)
            pred = np.where(usable, pred, np.nan)

            transform = from_origin(meta['transform'][2] + c0 * meta['transform'][0],
                                    meta['transform'][5] + r0 * meta['transform'][4],
                                    meta['transform'][0], -meta['transform'][4])
            lo, hi, src_note = _intervals(ts, pred.shape, transform)
            lo = np.where(usable, lo, np.nan) if lo is not None else np.full_like(pred, np.nan)
            hi = np.where(usable, hi, np.nan) if hi is not None else np.full_like(pred, np.nan)

            stem = os.path.join(out_dir, '%sT%02d00' % (ts.strftime('%Y%m%d'), hour))
            path = stem + '.tif'
            with rasterio.open(path, 'w', driver='GTiff', height=pred.shape[0], width=pred.shape[1],
                               count=4, dtype='float32', crs=meta['crs'], transform=transform,
                               compress='LZW') as dst:
                dst.write(pred, 1)
                dst.write(lo, 2)
                dst.write(hi, 3)
                dst.write(usable.astype('float32'), 4)
                for i, nm in enumerate(('prediction', 'lower', 'upper', 'usable'), start=1):
                    dst.set_band_description(i, nm)
            written.append(path)
            frames[ts] = {'prediction': pred, 'lower': lo, 'upper': hi, 'usable': usable,
                          'filled': filled}
            if verbose:
                morph_ok = st['valid'][r0:r1, c0:c1]
                no_hist = int((morph_ok & ~hist_ok).sum())
                no_morph = int((~morph_ok).sum())
                if fill_morphology:
                    print('    %s: %d/%d px usable (%.1f%%) incl. %d extrapolated, mean %.2f, max %.2f'
                          % (ts, int(usable.sum()), usable.size, 100 * usable.mean(),
                             int(filled.sum()), float(np.nanmean(pred)), float(np.nanmax(pred))))
                else:
                    print('    %s: %d/%d px usable (%.1f%%), mean %.2f, max %.2f'
                          % (ts, int(usable.sum()), usable.size, 100 * usable.mean(),
                             float(np.nanmean(pred)), float(np.nanmax(pred))))
                print('        blank because: no morphology %s px (%.1f%%) | no stand-in history %s px (%.1f%%)'
                      % (f'{no_morph:,}', 100 * no_morph / usable.size,
                         f'{no_hist:,}', 100 * no_hist / usable.size))
        day = pd.Timestamp(d).strftime('%Y%m%d')
        _day = [f for t, f in frames.items() if pd.Timestamp(t).date().isoformat() == str(d)]
        _n = [int(f['usable'].sum()) for f in _day]
        summary = {
            'date': str(d),
            'hours': [str(pd.Timestamp(t)) for t in frames if pd.Timestamp(t).date().isoformat() == str(d)],
            'n_pixels': int((r1 - r0) * (c1 - c0)),
            'n_usable_min': int(min(_n)) if _n else 0,
            'n_usable_max': int(max(_n)) if _n else 0,
            'usable_fraction': round(float(min(_n)) / max(1, (r1 - r0) * (c1 - c0)), 4) if _n else 0.0,
            'window_px': meta['window_px'],
            'interval_source': (src_note or 'none') if 'src_note' in dir() else 'none',
            'note': 'interval bounds resampled bilinearly from the 500 m GeoCP field, not computed per pixel',
        }
        with open(os.path.join(out_dir, day + '_summary.json'), 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=2)
        if verbose:
            print('    summary ->', os.path.join(out_dir, day + '_summary.json'))
    return written, frames

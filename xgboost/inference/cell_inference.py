"""Step 8: point predictions for EVERY cell of the morphology grid.

The problem this solves. The model's 24 features include nine PM2.5-history columns
(`pm25_lag_1..4`, `pm25_diff_1_2`, `pm25_roll24_mean/std`, `pm25_roll72_mean/std`) built from a
sensor's OWN past readings. A raster cell with no sensor has none of them, so it cannot be scored as-is.

The stand-in, following the CNN-LSTM branch's `inference/cell_inference.py` and the thesis's
spatial-lag paragraph: a cell's PM2.5 history is the inverse-distance weighted mean of its k nearest
PROPER-TRAINING stations at each required hour. That fills the nine history columns; the six morphology
columns come from the raster grid, and the cyclical/interaction columns from the snapshot timestamp.

WHAT THIS MEANS FOR THE RESULT -- state it wherever a map is shown:

  * At an unmonitored cell the prediction is driven mainly by the interpolated history, because the
    morphology features carry ~2.5% of the model's attribution (STEP3_ABLATION.md). A city-wide map is
    therefore closer to an IDW interpolation with morphology texture than to a morphology model.
  * **The stand-in SMOOTHS the field, and the map comes out low.** Measured on the pilot, for
    2025-07-23 08:00: the sensor-side predictions (real history) average 15.28 ug/m3 with a maximum of
    89.6, while the 378-cell stand-in surface averages 9.75 with a maximum of 21.8 -- about 36% low
    with its range compressed from 4-333 ug/m3 (observed) to 4-22. The reason is that the stand-in is
    an inverse-distance MEAN of neighbouring stations and PM2.5 is strongly right-skewed, so averaging
    removes exactly the peaks the map is supposed to show. Report this as a property of the method, not
    as a finding about the city.
  * Every row carries `standin_coverage` (the FRACTION of the required 73-hour history that neighbours
    actually reported; mean ~0.63 on the pilot) and `prediction_usable`. Rows with no history at all
    carry NaN rather than a fabricated value.
  * The model was calibrated on sensors whose own lags are REAL readings, so an interval for a stand-in
    cell describes the model's error with real histories, not with interpolated ones. Treat GeoCP
    intervals on such cells as optimistic.
"""
import os

import numpy as np
import pandas as pd

from preprocessing import prep
from preprocessing import spatial_cv

HISTORY_FEATURES = ['pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3', 'pm25_lag_4', 'pm25_diff_1_2',
                    'pm25_roll24_mean', 'pm25_roll24_std', 'pm25_roll72_mean', 'pm25_roll72_std']
MORPH_FEATURES = ['bh_mean', 'bh_stdev', 'bldg_dens', 'ndvi_mean', 'aspect_ratio_mean',
                  'street_width_mean']
TIME_FEATURES = ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos']
INTERACTION_FEATURES = ['hour_x_bldg_dens', 'hour_x_ndvi', 'hour_x_aspect_ratio']


def _localize(ts, like):
    """Match a timestamp to the tz-awareness of `like` (the pipeline's DateTime column).

    `Master_DL_Ready.csv` carries '+08:00' timestamps, so pandas parses them as tz-AWARE. A bare
    `pd.Timestamp('2025-07-23 08:00')` is tz-naive, and comparing the two raises
    "Cannot compare tz-naive and tz-aware timestamps". Normalising here keeps every caller free of it.
    """
    stamp = pd.Timestamp(ts)
    if getattr(like, 'tz', None) is not None:
        return stamp.tz_localize(like.tz) if stamp.tz is None else stamp.tz_convert(like.tz)
    return stamp.tz_localize(None) if stamp.tz is not None else stamp


def grid_cells(mapping_csv_path=None):
    """One row per cell of the FULL morphology raster, with its centroid and morphology.

    Reads `full_morphology_grid.csv` (378 cells for the pilot), NOT `sensor_to_grid_map_full.csv`
    (17 rows, only the cells that contain a sensor). Reaching for the mapping file here would silently
    restrict city-wide inference to the monitored cells -- which is exactly the opposite of the point,
    and the coverage diagnostics would then read a misleading 1.00 instead of the real fraction.
    """
    import data as _d
    grid = pd.read_csv(_d.resolve(mapping_csv_path, _d.GRID_CSV))
    grid = grid.rename(columns={'id': 'Grid_ID'})
    grid['cx'] = (grid['left'] + grid['right']) / 2
    grid['cy'] = (grid['top'] + grid['bottom']) / 2
    grid['street_width_mean'] = grid['bh_mean'] / grid['aspect_ratio_mean']
    # left/bottom/right/top are kept so the maps can draw each cell as a real 500 m square in
    # projected coordinates (maps/surfaces.py `_cell_geometries`).
    cols = ['Grid_ID', 'cx', 'cy', 'left', 'bottom', 'right', 'top'] + MORPH_FEATURES
    return grid.groupby('Grid_ID', as_index=False)[cols[1:]].first()[cols]


def hourly_matrix(df, station_ids):
    """PM2.5 per hour (rows) and STATION (columns, in `station_ids` order).

    Keyed on `Location`, not `Grid_ID`. `prep.df_clean` carries Grid_ID, and a pivot_table over it
    silently produces columns named by grid cell (3-digit) instead of sensor (7-digit) -- which then
    cannot be indexed by station positions. Cell 78 holds two sensors, so the two keys differ; the
    neighbours must be stations because every other station-keyed thing here is.
    """
    wide = df.pivot_table(index='DateTime', columns='Location', values='Value', aggfunc='mean')
    wide = wide.asfreq('h')
    return wide.reindex(columns=list(station_ids))


def neighbour_weights(cell_xy, station_xy, k=4):
    """Inverse-distance weights (1/d, 1 m floor) for the k nearest stations, rows summing to 1.

    Nothing is excluded: unlike the feature-time spatial lag, a cell that happens to contain a sensor
    may use it. This mirrors the CNN-LSTM branch's `nearest_station_weights`, which is model-agnostic.
    """
    from scipy.spatial import cKDTree
    cell_xy = np.asarray(cell_xy, dtype=float).reshape(-1, 2)
    station_xy = np.asarray(station_xy, dtype=float)
    k = int(min(k, len(station_xy)))
    dist, idx = cKDTree(station_xy).query(cell_xy, k=k)
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    w = 1.0 / np.maximum(dist, 1.0)
    return idx, w / w.sum(axis=1, keepdims=True)


def standin_series(values, neighbour_idx, neighbour_w):
    """[n_hours, n_cells] stand-in PM2.5: weighted mean of the neighbours present at each hour.

    Hour-major, because `build_features` indexes it as `series[hour]` -> one value per cell.
    `values` is [n_hours, n_stations]; `neighbour_idx` indexes its station axis. Everything is
    positional, so a tz-aware DatetimeIndex never reaches numpy as an index (a real bug earlier: a float
    index array raises "only integers ... are valid indices").
    """
    vals = values[:, neighbour_idx].astype(float)                  # [h, n_cells, k]
    seen = ~np.isnan(vals)
    num = (np.where(seen, vals, 0.0) * neighbour_w[None]).sum(axis=-1)
    den = (seen * neighbour_w[None]).sum(axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)                 # [h, n_cells]


def build_features(timestamp, cells, series, hour_pos, n_neighbours=4):
    """The model's 24 features for every cell at one hour, plus stand-in coverage diagnostics.

    `series` is [n_hours, n_cells] of stand-in PM2.5 on a continuous hourly index; `hour_pos` is the
    position of `timestamp` in it. Returns (X [n_cells, 24], diagnostics DataFrame).
    """
    ts = pd.Timestamp(timestamp).floor('h')
    need = 72                                     # longest rolling window
    if hour_pos < need:
        raise ValueError(f'need at least {need} hours of history before {ts}')

    def shift(n):
        return series[hour_pos - n]

    lag1, lag2, lag3, lag4 = shift(1), shift(2), shift(3), shift(4)
    roll24 = series[hour_pos - 24:hour_pos]       # hours t-24 .. t-1, exclusive of t
    roll72 = series[hour_pos - 72:hour_pos]

    # Cells with no reading in the window (the far field, or a sensor that is not reporting) have an
    # all-NaN column. np.nanmean/nanstd then warn "Mean of empty slice" once per timestamp, which flooded
    # the infer output with RuntimeWarnings that look like errors but are the expected case -- those cells
    # are flagged by `history_present` and fall back to the interpolated stand-in. The moments below are
    # computed directly so the same NaN comes out without the noise: they match nanmean and
    # nanstd(ddof=1) exactly on any column that has data.
    def _roll_moments(win):
        valid = np.isfinite(win)
        n_valid = valid.sum(axis=0)
        filled = np.where(valid, win, 0.0)
        mean = np.where(n_valid > 0, filled.sum(axis=0) / np.maximum(n_valid, 1), np.nan)
        dev = np.where(valid, (win - mean) ** 2, 0.0)
        var = np.where(n_valid > 1, dev.sum(axis=0) / np.maximum(n_valid - 1, 1), np.nan)
        return mean, np.sqrt(var)

    roll24_mean, roll24_std = _roll_moments(roll24)
    roll72_mean, roll72_std = _roll_moments(roll72)

    hour = ts.hour
    dow = ts.dayofweek
    month = ts.month
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    n = len(cells)
    feats = pd.DataFrame({
        'bh_mean': cells['bh_mean'].values, 'bh_stdev': cells['bh_stdev'].values,
        'bldg_dens': cells['bldg_dens'].values, 'ndvi_mean': cells['ndvi_mean'].values,
        'aspect_ratio_mean': cells['aspect_ratio_mean'].values,
        'street_width_mean': cells['street_width_mean'].values,
        'hour_sin': np.full(n, hour_sin), 'hour_cos': np.full(n, hour_cos),
        'dayofweek_sin': np.full(n, np.sin(2 * np.pi * dow / 7.0)),
        'dayofweek_cos': np.full(n, np.cos(2 * np.pi * dow / 7.0)),
        'month_sin': np.full(n, np.sin(2 * np.pi * month / 12.0)),
        'month_cos': np.full(n, np.cos(2 * np.pi * month / 12.0)),
        'pm25_lag_1': lag1, 'pm25_lag_2': lag2, 'pm25_lag_3': lag3, 'pm25_lag_4': lag4,
        'pm25_diff_1_2': lag1 - lag2,
        'pm25_roll24_mean': roll24_mean, 'pm25_roll24_std': roll24_std,
        'pm25_roll72_mean': roll72_mean, 'pm25_roll72_std': roll72_std,
        'hour_x_bldg_dens': hour_sin * cells['bldg_dens'].values,
        'hour_x_ndvi': hour_sin * cells['ndvi_mean'].values,
        'hour_x_aspect_ratio': hour_sin * cells['aspect_ratio_mean'].values,
    })[list(prep.features)]

    window = series[hour_pos - 72:hour_pos + 1]
    # Coverage is the FRACTION of the required history that neighbours actually reported, not a
    # yes/no. An earlier version used ~np.isnan(window).all(axis=0), which is true if even ONE hour
    # exists and so reported 1.00 almost everywhere -- useless as a quality signal on a map.
    coverage = 1.0 - np.isnan(window).mean(axis=0)
    diag = pd.DataFrame({
        'Grid_ID': cells['Grid_ID'].values,
        'standin_coverage': coverage,
        'history_present': coverage > 0,
    })
    return feats, diag


def infer_timestamp(model, timestamp, *, station_df=None, train_cells=None, k=4,
                    mapping_csv_path=None, feature_names=None):
    """Point predictions and diagnostics for every grid cell at one timestamp."""
    import data
    cells = grid_cells(mapping_csv_path)
    df = prep.df_clean if station_df is None else station_df
    fold_doc = spatial_cv.load_fold_map()
    train_cells = train_cells or fold_doc['partitions']['train']
    station_ids = sorted(df[df['Grid_ID'].isin(train_cells)]['Location'].unique())

    wide = hourly_matrix(df[df['Location'].isin(station_ids)], station_ids)
    ts = _localize(timestamp, wide.index)
    if ts not in wide.index:
        end = max(wide.index.max(), ts)
        wide = wide.reindex(pd.date_range(wide.index.min(), end, freq='h'))
    station_xy = (df[df['Location'].isin(station_ids)]
                  .groupby('Location')[['x_coordinate', 'y_coordinate']].first()
                  .reindex(station_ids).values)
    idx, w = neighbour_weights(cells[['cx', 'cy']].values, station_xy, k=k)
    if os.environ.get('XGB_DEBUG_INFER'):
        print(f'    DEBUG station_ids={len(station_ids)} wide={wide.shape} '
              f'station_xy={station_xy.shape} idx={idx.shape} idx.max={idx.max()}')
    series = standin_series(wide.to_numpy(dtype=float), idx, w)     # [n_hours, n_cells]
    hour_pos = wide.index.get_loc(ts)
    X, diag = build_features(timestamp, cells, series, hour_pos)
    feat = list(feature_names or prep.features)
    missing = [f for f in feat if f not in X.columns]
    if missing:
        raise ValueError(f'the model expects feature(s) this pipeline did not build: {missing}')
    pred = np.expm1(model.predict(X[feat].to_numpy(dtype=float)))
    out = cells[['Grid_ID', 'cx', 'cy', 'left', 'bottom', 'right', 'top']].copy()
    out['timestamp'] = pd.Timestamp(timestamp)
    out['prediction'] = pred
    out['standin_coverage'] = diag['standin_coverage'].values
    out['history_present'] = diag['history_present'].values
    # A cell is predictable only if BOTH inputs exist: a stand-in history AND complete morphology. The
    # grid has 1,540 of 3,696 cells (42%) whose morphology is nodata -- they are water, outside the
    # building-height coverage, or between buffers. XGBoost answers anyway by routing the NaNs down its
    # default branch, and on 2025-08-30 that produced a false 35-45 ug/m3 wedge across Manila Bay.
    # Painting them would present an artefact as a hotspot, so they are marked unusable and go NaN.
    # `out` carries only geometry columns, so the morphology test must run on `cells`, which is the
    # frame the grid loader returns (same row order as `out`).
    _morph_ok = pd.Series(True, index=cells.index)
    for _c in MORPH_FEATURES:
        if _c in cells.columns:
            _morph_ok &= cells[_c].notna()
    _morph_ok = pd.Series(_morph_ok.to_numpy(), index=out.index)
    out['morphology_complete'] = _morph_ok
    out['prediction_usable'] = (out['history_present'] & _morph_ok
                                & np.isfinite(out['prediction']))
    # A cell with no history gets NaN lags, and XGBoost sends missing values down a default branch --
    # it still RETURNS a number, and on this network that number can be 170-240 ug/m3, which painted a
    # false bright band across Manila Bay. The docstring already promised "rows with no history carry
    # NaN rather than a fabricated value"; enforce it, so maps, statistics and intervals all agree.
    out.loc[~out['prediction_usable'], 'prediction'] = np.nan
    return out, X

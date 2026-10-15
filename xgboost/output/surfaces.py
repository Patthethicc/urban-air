"""Step 8 output: per-cell tables, GeoCP intervals, GeoTIFF rasters and AQI maps.

The table is the primary artifact and is always written. Rasters and PNG maps are written only when
their libraries are available (`rasterio` for GeoTIFF, `matplotlib` for maps), so the pipeline never
fails just because an optional dependency is missing -- it says so and continues.

Order of preference, and why:

  1. `<timestamp>.csv` -- 378 rows of prediction, interval and coverage diagnostics. Small, diffable,
     reviewable, and the thing to commit. A raster can always be rebuilt from it.
  2. `<timestamp>.tif` -- the same numbers as aligned float32 bands (prediction, lower, upper, width)
     in the grid's CRS, so they can be overlaid directly. Written with LZW; NaN marks cells whose
     stand-in history was too incomplete to predict.
  3. `<timestamp>_aqi.png` -- the prediction on the Philippine AQI tiers (DENR DAO 2020-14), for reading.

Interval construction uses `uncertainty.geocp.quantile_grid`, which is exact and vectorised over cells.
"""
import json
import os

import numpy as np
import pandas as pd

import data

AQI_BINS = [('Good', 25.0, '#00e400'), ('Fair', 35.0, '#ffd500'),
            ('Unhealthy for sensitive groups', 45.0, '#ff7e00'), ('Very unhealthy', 55.0, '#ff0000'),
            ('Acutely unhealthy / emergency', np.inf, '#8f3f97')]
HIGHLIGHT = 35.0


def _stamp(ts):
    return pd.Timestamp(ts).strftime('%Y%m%dT%H%M')


def _prep_df():
    from preprocessing import prep
    return prep.df_clean


def _cal_cells():
    from preprocessing import spatial_cv
    return spatial_cv.load_fold_map()['partitions']['calibration']


def geocp_intervals(model, cells, alpha=0.10, h=None, kernel='gaussian', mapping_csv_path=None):
    """Interval half-widths for the given cells, from the calibration partition of the model's run."""
    from uncertainty import geocp
    from preprocessing import spatial_cv

    cal = geocp.calibration_scores_for_model(model, _prep_df(), _cal_cells(),
                                            mapping_csv_path=mapping_csv_path)
    if h is None:
        h = geocp.resolve_bandwidth(cal, alpha, kernel, geocp.H_GRID, 'median-distance', None, True,
                                    query_xy=cells[['cx', 'cy']].values)[0]
    q = geocp.quantile_grid(cal, cells[['cx', 'cy']].values, h, kernel, 1.0 - alpha)
    out = cells.copy()
    out['q'] = q
    out['lower'] = out['prediction'] - q
    out['upper'] = out['prediction'] + q
    out['lower_display'] = np.maximum(out['lower'], 0.0)
    out['width'] = out['upper'] - out['lower']
    out['kernel'] = kernel
    out['bandwidth_m'] = float(h)
    out['alpha'] = alpha
    return out


def aqi_category(values):
    """Index into AQI_BINS per value (NaN -> -1)."""
    v = np.asarray(values, dtype=float)
    cat = np.full(v.shape, -1, dtype=int)
    for i, (_, upper, _) in enumerate(AQI_BINS):
        cat[(cat < 0) & np.isfinite(v) & (v <= upper)] = i
    return cat


def write_snapshot(cells, ts, out_dir=None):
    """Write the table, and the raster and map when their libraries exist. Returns written paths."""
    out_dir = out_dir or data.out('surfaces', _stamp(ts)[:8])
    os.makedirs(out_dir, exist_ok=True)
    stem = _stamp(ts)
    written = []

    csv_path = os.path.join(out_dir, f'{stem}.csv')
    cells.to_csv(csv_path, index=False)
    written.append(csv_path)
    print(f'    table -> {os.path.relpath(csv_path, data.ROOT)} ({len(cells)} rows)')

    raster = _try_raster(cells, ts, out_dir, stem)
    if raster:
        written.append(raster)
    png = _try_map(cells, ts, out_dir, stem)
    if png:
        written.append(png)
    written.append(_write_summary(cells, ts, out_dir, stem))
    return written


def _grid_shape():
    """(rows, cols, lookup) for the FULL morphology raster.

    Uses `full_morphology_grid.csv` (378 cells), not the sensor mapping (17). Both `row_index` and
    `col_index` are 1-based there (1..18 rows, 1..21 cols), so the origin is subtracted before sizing
    rather than assuming any particular base. `lookup` maps Grid_ID -> (0-based row, col).
    """
    import data as _d
    grid = pd.read_csv(_d.resolve(None, _d.GRID_CSV)).rename(columns={'id': 'Grid_ID'})
    r0, c0 = int(grid['row_index'].min()), int(grid['col_index'].min())
    rows = int(grid['row_index'].max()) - r0 + 1
    cols = int(grid['col_index'].max()) - c0 + 1
    lookup = {int(g): (int(r) - r0, int(c) - c0) for g, r, c in
              zip(grid['Grid_ID'], grid['row_index'], grid['col_index'])}
    return rows, cols, lookup


def _to_grid(cells, values, fill=np.nan):
    rows, cols, lookup = _grid_shape()
    grid = np.full((rows, cols), fill, dtype=float)
    for gid, val in zip(cells['Grid_ID'].values, np.asarray(values, dtype=float)):
        rc = lookup.get(int(gid))
        if rc is not None:
            grid[rc[0], rc[1]] = val
    return grid


def _try_raster(cells, ts, out_dir, stem):
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        print('    (rasterio not installed - skipping the GeoTIFF; `pip install rasterio affine`)')
        return None
    import data as _d
    grid = pd.read_csv(_d.resolve(None, _d.GRID_CSV)).rename(columns={'id': 'Grid_ID'})
    left, top = float(grid['left'].min()), float(grid['top'].max())
    res = float(grid['right'].iloc[0] - grid['left'].iloc[0])
    rows, cols, _ = _grid_shape()
    transform = from_origin(left, top, res, res)
    bands = {'prediction': 'prediction', 'lower': 'lower_display', 'upper': 'upper',
             'width': 'width'}
    path = os.path.join(out_dir, f'{stem}.tif')
    with rasterio.open(path, 'w', driver='GTiff', height=rows, width=cols,
                       count=len(bands), dtype='float32', crs='EPSG:32651', transform=transform,
                       nodata=np.nan, compress='lzw') as dst:
        for i, (name, col) in enumerate(bands.items(), start=1):
            dst.write(_to_grid(cells, cells[col].values).astype('float32'), i)
        dst.descriptions = tuple(bands)
    print(f'    raster -> {os.path.relpath(path, data.ROOT)} '
          f'({rows}x{cols}, {len(bands)} bands, LZW)')
    return path


def _try_map(cells, ts, out_dir, stem):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except ImportError:
        print('    (matplotlib not available - skipping the AQI map)')
        return None
    grid = _to_grid(cells, cells['prediction'].values)
    path = os.path.join(out_dir, f'{stem}_aqi.png')
    # Bins must be strictly increasing: taking the data max as the top edge fails when every
    # prediction sits below the highest tier (a real ValueError: "bins must be monotonically
    # increasing"). The top edge is therefore fixed at the last finite tier, and predictions above it
    # are clipped into that tier by BoundaryNorm.
    bounds = [0, 25, 35, 45, 55, 60]
    cmap = ListedColormap([c for _, _, c in AQI_BINS])
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm)
    ax.set_title(f'PM2.5 {pd.Timestamp(ts):%Y-%m-%d %H:%M} (500 m cells, {int(np.isfinite(grid).sum())} usable)')
    ax.set_xlabel('col'); ax.set_ylabel('row')
    cbar = fig.colorbar(im, ax=ax, ticks=bounds[:-1])
    cbar.ax.set_yticklabels([b[0] for b in AQI_BINS], fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'    map -> {os.path.relpath(path, data.ROOT)}')
    return path


def _write_summary(cells, ts, out_dir, stem):
    finite = cells[np.isfinite(cells['prediction'])]
    cat = aqi_category(finite['prediction'].values)
    summary = {
        'timestamp': str(pd.Timestamp(ts)),
        'cells_total': int(len(cells)),
        'cells_usable': int(len(finite)),
        'cells_missing_history': int((~cells['prediction_usable']).sum()),
        'mean_standin_coverage': float(cells['standin_coverage'].mean()),
        'prediction': {
            'mean': float(finite['prediction'].mean()) if len(finite) else None,
            'median': float(finite['prediction'].median()) if len(finite) else None,
            'min': float(finite['prediction'].min()) if len(finite) else None,
            'max': float(finite['prediction'].max()) if len(finite) else None,
        },
        'interval': {
            'mean_width': float(finite['width'].mean()) if len(finite) else None,
            'median_width': float(finite['width'].median()) if len(finite) else None,
            'bandwidth_m': float(finite['bandwidth_m'].iloc[0]) if len(finite) else None,
            'kernel': finite['kernel'].iloc[0] if len(finite) else None,
            'alpha': float(finite['alpha'].iloc[0]) if len(finite) else None,
        },
        'cells_above_35': int((finite['prediction'] > HIGHLIGHT).sum()) if len(finite) else 0,
        'aqi_category_share': {
            label: float((cat == i).mean()) for i, (label, _, _) in enumerate(AQI_BINS)
        } if len(finite) else {},
        'caveat': ('Predictions at cells without a sensor are driven by the interpolated stand-in '
                   'history; the morphology features carry ~2.5% of the model attribution. The '
                   'stand-in is an inverse-distance MEAN, and PM2.5 is right-skewed, so the surface '
                   'reads LOW and its range is compressed relative to sensor-side predictions '
                   '(measured on the pilot: 9.75 mean / 21.8 max on the map vs 15.28 / 89.6 from real '
                   'histories). See inference/cell_inference.py.'),
    }
    path = os.path.join(out_dir, f'{stem}_summary.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(f'    summary -> {os.path.relpath(path, data.ROOT)} '
          f'({summary["cells_usable"]}/{summary["cells_total"]} usable, '
          f'mean width {summary["interval"]["mean_width"]:.1f})' if len(finite) else '    summary written')
    return path

"""Step 8 presentation: per-cell tables, GeoCP intervals, GeoTIFF rasters and AQI maps.

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


def geocp_intervals(model, cells, alpha=0.10, h=None, kernel='gaussian', mapping_csv_path=None,
                    feature_names=None):
    """Interval half-widths for the given cells, from the calibration partition of the model's run."""
    from uncertainty import geocp
    from preprocessing import spatial_cv

    cal = geocp.calibration_scores_for_model(model, _prep_df(), _cal_cells(),
                                            mapping_csv_path=mapping_csv_path,
                                            feature_names=feature_names)
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


def grid_georef():
    """(transform, crs) for the morphology raster, from the grid's own cell edges.

    The pilot grid is a COMPLETE 18 x 21 rectangle of 500 m cells (378 of 378, no holes) in EPSG:32651,
    so the affine transform is exact rather than inferred: origin = (min left, max top), pixel size =
    (right - left) = 500 m, y negative.
    """
    from rasterio.transform import from_origin
    import data as _d
    grid = pd.read_csv(_d.resolve(None, _d.GRID_CSV)).rename(columns={'id': 'Grid_ID'})
    left, top = float(grid['left'].min()), float(grid['top'].max())
    res_x = float(grid['right'].iloc[0] - grid['left'].iloc[0])
    res_y = float(grid['top'].iloc[0] - grid['bottom'].iloc[0])
    rows = int(grid['row_index'].max()) - int(grid['row_index'].min()) + 1
    cols = int(grid['col_index'].max()) - int(grid['col_index'].min()) + 1
    return from_origin(left, top, res_x, res_y), 'EPSG:32651', rows, cols


def _cell_geometries(cells, values):
    """GeoDataFrame of the usable cells as 500 m squares, for plotting in real coordinates.

    Cells are buffered by a hair (1 m of a 500 m cell) because adjacent unprojected rectangles leave
    hairlines of white between them; geopandas does the same by default.
    """
    import geopandas as gpd
    from shapely.geometry import box
    r = cells[np.isfinite(values)].copy()
    if r.empty:
        return gpd.GeoDataFrame({'value': []}, geometry=[], crs='EPSG:32651')
    r['value'] = np.asarray(values)[np.isfinite(values)]
    r['geometry'] = [box(l, b, rt, t) for l, b, rt, t in
                     zip(r['left'], r['bottom'], r['right'], r['top'])]
    gdf = gpd.GeoDataFrame(r[['value', 'prediction']], geometry=r['geometry'].values, crs='EPSG:32651')
    return gdf.assign(geometry=gdf.geometry.buffer(1.0))


def _try_map(cells, ts, out_dir, stem):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.patches import Patch
    except ImportError:
        print('    (matplotlib not available - skipping the AQI map)')
        return None
    from maps import basemap

    path = os.path.join(out_dir, f'{stem}_aqi.png')
    # Bins must be strictly increasing: taking the data max as the top edge fails when every prediction
    # sits below the highest tier (a real ValueError: "bins must be monotonically increasing").
    bounds = [0, 25, 35, 45, 55, 60]
    cmap = ListedColormap([c for _, _, c in AQI_BINS])
    norm = BoundaryNorm(bounds, cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    gdf = _cell_geometries(cells, cells['prediction'].values)
    if gdf.empty:
        plt.close(fig)
        print('    (no usable cells - skipping the AQI map)')
        return None
    gdf.plot(ax=ax, column='prediction', cmap=cmap, norm=norm, edgecolor='none', zorder=2)
    xlim, ylim = (gdf.total_bounds[0], gdf.total_bounds[2]), (gdf.total_bounds[1], gdf.total_bounds[3])
    based = basemap.add_basemap(ax, gdf.crs, xlim, ylim)

    # Outline the cells above the AQI highlight threshold (thesis: 35 ug/m3).
    hot = gdf[gdf['prediction'] > HIGHLIGHT]
    if not hot.empty:
        hot.boundary.plot(ax=ax, color='#1a1a18', linewidth=1.2, zorder=4)

    # Monitoring stations, so a reader can see what the map is interpolating from.
    try:
        import data as _d
        mm = pd.read_csv(_d.resolve(None, _d.SENSOR_MAP_CSV)).drop_duplicates('id')
        ax.scatter((mm['left'] + mm['right']) / 2, (mm['top'] + mm['bottom']) / 2,
                   s=22, c='white', edgecolors='#1a1a18', linewidths=0.9, zorder=5,
                   label=f'{len(mm)} monitoring stations')
    except Exception:
        pass

    handles = [Patch(facecolor=c, edgecolor='#4a4a48', linewidth=0.5,
                     label=f'{lab} ({lo:g}-{up:g})' if np.isfinite(up) else f'{lab} (>{lo:g})')
               for (lab, up, c), lo in zip(AQI_BINS, [0, 25, 35, 45, 55])]
    if not hot.empty:
        handles.append(Patch(facecolor='none', edgecolor='#1a1a18', linewidth=1.2,
                             label=f'outline: above {HIGHLIGHT:g} ug/m3'))
    if based == 'vectors':
        handles.append(Patch(facecolor=basemap.STREET_COLOR, edgecolor='none', label='streets'))
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0], loc='lower center', ncol=2,
              fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.28))

    where = {'web': 'over web tiles', 'vectors': 'over OSM streets',
             'plain': 'no basemap (see the note above)'}[based]
    ax.set_title(f'PM2.5 {pd.Timestamp(ts):%Y-%m-%d %H:%M} — 500 m cells, {where}\n'
                 f'{len(gdf)} usable cells · mean stand-in coverage '
                 f'{cells["standin_coverage"].mean():.2f} · outline = above {HIGHLIGHT:g} µg/m³',
                 fontsize=9, loc='left')
    _km_axes(ax)
    ax.tick_params(labelsize=7)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'    map -> {os.path.relpath(path, data.ROOT)}  ({where})')

    cont = _try_continuous_map(cells, ts, out_dir, stem, based)
    return path


def suggest_limits(frames, lo=2.0, hi=98.0, floor=(0.0, 25.0)):
    """Shared colour limits for a set of snapshots, from the pooled predictions.

    One scale across the whole set is deliberate: four surfaces on four different scales cannot be
    compared. Percentiles rather than min/max so a single spike does not flatten everything else.
    """
    vals = np.concatenate([np.asarray(f['prediction'], dtype=float) for f in frames.values()])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return floor
    vmin, vmax = float(np.percentile(vals, lo)), float(np.percentile(vals, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return floor
    return max(0.0, vmin), vmax


def render_daily_mean(frames, out_dir, date, vmin=None, vmax=None, stem=None):
    """Mean of the snapshots, one cell one value, rendered on the continuous scale."""
    if not frames:
        return None
    parts = []
    for ts, f in frames.items():
        keep = ['Grid_ID', 'prediction']
        for c in ('left', 'bottom', 'right', 'top', 'cx', 'cy', 'standin_coverage'):
            if c in f.columns:
                keep.append(c)
        parts.append(f[keep].copy())
    allf = pd.concat(parts, ignore_index=True)
    agg = {'prediction': 'mean'}
    for c in ('left', 'bottom', 'right', 'top', 'cx', 'cy', 'standin_coverage'):
        if c in allf.columns:
            agg[c] = 'first'
    mean = allf.groupby('Grid_ID', as_index=False).agg(agg)
    mean = mean[np.isfinite(mean['prediction'])]
    if mean.empty:
        return None
    return _try_continuous_map(mean, pd.Timestamp(date), out_dir, stem or f'{date}_mean', True,
                               vmin=vmin, vmax=vmax, label='24-hour mean')




SURFACE_ALPHA = 0.62      # lets the basemap read through the prediction surface
_STATIONS = {}


def _stations():
    """Sensor positions for the overlay, cached. Drawn as white dots like the CNN-LSTM maps."""
    if 'df' not in _STATIONS:
        import pandas as pd
        try:
            m = pd.read_csv(data.SENSOR_MAP_CSV)
            if {'cx', 'cy'} <= set(m.columns):
                xy = m[['cx', 'cy']].copy()
            elif {'left', 'bottom'} <= set(m.columns):
                # sensor_to_grid_map carries cell bounds, not centres: derive them (500 m cells)
                half = 250.0
                xy = pd.DataFrame({'cx': m['left'] + half, 'cy': m['bottom'] + half})
            else:
                xy = None
            _STATIONS['df'] = None if xy is None else xy.dropna().drop_duplicates()
        except Exception as exc:      # never let a marker break a map
            print('  station overlay unavailable:', exc)
            _STATIONS['df'] = None
    return _STATIONS['df']


def _overlay_stations(ax):
    st = _stations()
    if st is None or st.empty:
        return
    ax.scatter(st['cx'], st['cy'], s=12, facecolor='white', edgecolor='black',
               linewidths=0.5, zorder=6)



def _km_axes(ax):
    """Label projected coordinates in kilometres.

    The raw UTM values are ~1.6e6, so matplotlib prints a "1e6" offset and five-digit ticks. Dividing
    the tick labels by 1000 removes the offset and makes the axis readable at report size.
    """
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _p: f'{v / 1000.0:.0f}')
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel('easting (km)', fontsize=7)
    ax.set_ylabel('northing (km)', fontsize=7)

def _try_continuous_map(cells, ts, out_dir, stem, based, vmin=None, vmax=None, label=None):
    """The same surface on a continuous scale.

    The AQI map is what the thesis asks for (Section 4.6.2) and is the right chart for a public-health
    reading, but its bins are coarse relative to what this pipeline actually produces: the stand-in
    surface spans roughly 4-22 ug/m3 (section 2b of the README), so EVERY cell lands in the 0-25 "Good"
    tier and the AQI map is a single flat colour. It is an honest summary and a useless one. This second
    panel uses a continuous ramp with a fixed scale, which is where the spatial pattern is legible.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    from maps import basemap
    # Data-driven by default: a fixed 0-25 scale flattened every cell into one colour because this
    # pipeline's predictions cluster near 10-15 ug/m3. Callers pass shared limits to keep snapshots
    # comparable to each other.
    if vmin is None or vmax is None:
        vals = np.asarray(cells['prediction'], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            vmin = 0.0 if vmin is None else vmin
            vmax = float(np.percentile(vals, 98)) if vmax is None else vmax
        else:
            vmin, vmax = (0.0, 25.0)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    gdf = _cell_geometries(cells, cells['prediction'].values)
    if gdf.empty:
        plt.close(fig)
        return None
    gdf.plot(ax=ax, column='prediction', cmap='viridis', vmin=vmin, vmax=vmax, alpha=SURFACE_ALPHA,
             edgecolor='none', legend=True,
             legend_kwds={'label': 'PM2.5 (µg/m³)', 'shrink': 0.7}, zorder=2)
    xlim, ylim = (gdf.total_bounds[0], gdf.total_bounds[2]), (gdf.total_bounds[1], gdf.total_bounds[3])
    if based != 'plain':
        basemap.add_basemap(ax, gdf.crs, xlim, ylim)
    _what = label or f'{pd.Timestamp(ts):%Y-%m-%d %H:%M}'
    ax.set_title(f'PM2.5 {_what} — continuous scale, {vmin:.1f}-{vmax:.1f} µg/m³\n'
                 f'{len(gdf)} cells · colour is comparable across snapshots',
                 fontsize=9, loc='left')
    _km_axes(ax)
    ax.tick_params(labelsize=7)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, f'{stem}_continuous.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'    map -> {os.path.relpath(path, data.ROOT)}  (continuous {vmin:g}-{vmax:g})')
    return path


def load_or_infer_snapshots(run=None, timestamps=None, out_dir=None, quiet=False):
    """Load the four snapshot surfaces from outputs/surfaces/, inferring them only if absent.

    Mirrors the notebook's ablation cell: fast when the results are already on disk, self-sufficient on
    a fresh clone. Returns {timestamp: DataFrame} where the frame is what `infer_timestamp` produced
    (prediction, intervals, standin_coverage, prediction_usable).
    """
    out_dir = out_dir or os.path.join(data.OUT_DIR, 'surfaces')
    frames = {}
    if os.path.isdir(out_dir):
        for day in sorted(os.listdir(out_dir)):
            day_dir = os.path.join(out_dir, day)
            if not os.path.isdir(day_dir):
                continue
            for name in sorted(os.listdir(day_dir)):
                if not (name.endswith('.csv') and name[:8].isdigit() and 'T' in name):
                    continue
                df = pd.read_csv(os.path.join(day_dir, name))
                frames[pd.Timestamp(df['timestamp'].iloc[0])] = df
    if timestamps:
        wanted = {pd.Timestamp(t).floor('h') for t in timestamps}
        frames = {k: v for k, v in frames.items() if pd.Timestamp(k).floor('h') in wanted}
    elif run or not frames:
        from run_pipeline import _fit_production_model
        from inference.cell_inference import infer_timestamp
        model, features, _ = _fit_production_model()
        hours = [8, 12, 17, 21]
        stamps = [pd.Timestamp(t) for t in timestamps] if timestamps else \
            [pd.Timestamp(run).normalize() + pd.Timedelta(hours=h) for h in hours]
        for ts in stamps:
            cells, _ = infer_timestamp(model, ts, feature_names=features)
            frames[ts] = cells
    if not quiet:
        print(f'{len(frames)} snapshot(s) from {out_dir}' if frames else 'no snapshots found')
    return dict(sorted(frames.items()))


def render_panel(frames, path, *, vmin=None, vmax=None, ncols=2, title=None):
    """A 2x2 (or ncols x n) panel of continuous-scale snapshots with ONE shared colourbar.

    A shared bar is the point: four surfaces on four different scales cannot be compared, and the whole
    reason for the continuous panel is that the AQI tiers (0-25 / 25-35 / ...) put every cell of this
    pipeline's output in the same bin, leaving the AQI map a single flat colour.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from maps import basemap

    items = [(ts, c[c['prediction_usable'].astype(bool)] if 'prediction_usable' in c.columns else c)
             for ts, c in frames.items()]
    items = [(ts, c) for ts, c in items if len(c)]
    if not items:
        raise ValueError('render_panel got no usable frames')
    if vmin is None or vmax is None:
        vmin, vmax = suggest_limits(frames)
    nrows = -(-len(items) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.6 * nrows), squeeze=False)
    norm = Normalize(vmin=vmin, vmax=vmax)
    for ax in axes.ravel()[len(items):]:
        ax.set_axis_off()
    for ax, (ts, cells) in zip(axes.ravel(), items):
        gdf = _cell_geometries(cells, cells['prediction'].values)
        if gdf.empty:
            ax.set_axis_off()
            continue
        gdf.plot(ax=ax, column='prediction', cmap='viridis', norm=norm, edgecolor='none', zorder=2,
                 alpha=SURFACE_ALPHA)
        xlim = (gdf.total_bounds[0], gdf.total_bounds[2])
        ylim = (gdf.total_bounds[1], gdf.total_bounds[3])
        based = basemap.add_basemap(ax, gdf.crs, xlim, ylim)
        ax.set_title(f'{pd.Timestamp(ts):%H:%M}   ({len(gdf)} cells, coverage '
                     f'{cells["standin_coverage"].mean():.2f})', fontsize=9, loc='left')
        _km_axes(ax)
        _overlay_stations(ax)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    fig.colorbar(ScalarMappable(norm=norm, cmap='viridis'), ax=axes.ravel().tolist(),
                 shrink=0.6, label='PM2.5 (µg/m³)', pad=0.02)
    if title:
        fig.suptitle(title, fontsize=11, x=0.02, ha='left')
    if basemap.last_basemap() == 'plain':
        fig.text(0.02, 0.005, 'no basemap: contextily not installed and no streets/boundary layer '
                              'configured (see maps/basemap.py)', fontsize=7, color='#666666')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'panel -> {os.path.relpath(path, data.ROOT)}  ({len(items)} snapshots)')
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

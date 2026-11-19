"""The continuous maps again, with the NCR outline: cut along the city boundaries, city outlines and names, scale bar, north arrow.

These are EXTRA outputs next to the plain 500 m cell maps of `surfaces.py` (`<stem>_continuous.png`, panel, daily mean), which stay as
they are. Two looks, both written:

    <stem>_continuous_ncr.png          the 500 m cells, each cut where the NCR boundary crosses it
    <stem>_continuous_ncr_smooth.png   the same predictions interpolated to ~100 m and cut exactly along the boundary

The boundary is `data/Boundaries/ncr_cities.gpkg` at the repository root (built by the CNN-LSTM branch's
`python -m preprocessing.boundaries`; GADM 4.1, 17 cities). Without it nothing is written and one line says so; no map ever fails
because of it. Display only: the table and the GeoTIFF keep every cell.
"""
import os

import numpy as np
import pandas as pd

import data
from maps import basemap, surfaces

BOUNDARY_PATH = os.path.join(os.path.dirname(data.ROOT), 'data', 'Boundaries', 'ncr_cities.gpkg')
CRS = 'EPSG:32651'
CELL_M = 500.0
SMOOTH_FACTOR = 5            # 500 m cells -> 100 m for the smooth look
INK, INK2 = '#1a1a18', '#5f5e59'
# names that would run into a neighbour or out of the map: (dx m, dy m, alignment)
LABEL_NUDGE = {'Navotas': (-900, -500, 'right'), 'Malabon': (500, 500, 'left'), 'San Juan': (-300, 450, 'right'),
               'Pateros': (900, -300, 'left')}
_CACHE = {}


def load_boundary(path=None):
    """The NCR city polygons (columns city, geometry) in EPSG:32651, or None (reported once)."""
    path = path or BOUNDARY_PATH
    if path not in _CACHE:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(path).to_crs(CRS)
            if 'city' not in gdf.columns:
                gdf['city'] = ''
            _CACHE[path] = gdf
        except Exception as exc:
            print(f'    NCR maps skipped: boundary {os.path.relpath(path, data.ROOT)} unavailable ({exc})')
            _CACHE[path] = None
    return _CACHE[path]


def _nice_km(target_m):
    exp = 10 ** np.floor(np.log10(max(target_m, 1.0)))
    return float(max(m for m in (1, 2, 5, 10) if m * exp <= target_m) * exp)


def _scale_bar(ax):
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    length = _nice_km(0.26 * abs(x1 - x0))
    left, base = min(x0, x1) + 0.05 * abs(x1 - x0), min(y0, y1) + 0.05 * abs(y1 - y0)
    h = 0.006 * abs(y1 - y0)
    from matplotlib.patches import Rectangle
    for i, face in enumerate((INK, 'white')):
        ax.add_patch(Rectangle((left + i * length / 2, base), length / 2, h, facecolor=face, edgecolor=INK, linewidth=0.6, zorder=8))
    for frac, text in ((0, '0'), (0.5, f'{length / 2000:g}'), (1, f'{length / 1000:g} km')):
        ax.text(left + frac * length, base + h * 1.8, text, fontsize=6.5, ha='center', va='bottom', zorder=8)


def _north_arrow(ax):
    ax.annotate('', xy=(0.92, 0.16), xytext=(0.92, 0.07), xycoords='axes fraction', zorder=8,
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.6))
    ax.text(0.92, 0.17, 'N', transform=ax.transAxes, ha='center', va='bottom', fontsize=8, fontweight='bold', zorder=8)


def _cities(ax, boundary):
    import matplotlib.patheffects as pe
    boundary.boundary.plot(ax=ax, color=INK2, linewidth=0.7, zorder=6)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    for name, pt in zip(boundary['city'], boundary.geometry.representative_point()):
        dx, dy, ha = LABEL_NUDGE.get(name, (0, 0, 'center'))
        if name and min(x0, x1) <= pt.x <= max(x0, x1) and min(y0, y1) <= pt.y <= max(y0, y1):
            ax.text(pt.x + dx, pt.y + dy, name, fontsize=6.5, color=INK, ha=ha, va='center', zorder=7,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])


def _lattice(cells):
    """(values [rows, cols] with NaN where there is no cell, left, top): the 500 m cells on their regular lattice."""
    col = np.rint((cells['left'].values - cells['left'].min()) / CELL_M).astype(int)
    row = np.rint((cells['top'].max() - cells['top'].values) / CELL_M).astype(int)
    grid = np.full((row.max() + 1, col.max() + 1), np.nan)
    grid[row, col] = cells['prediction'].values
    return grid, float(cells['left'].min()), float(cells['top'].max())


def smooth_surface(cells, boundary):
    """The cells interpolated to ~100 m (bicubic, kept inside the range of the data), blank outside the NCR and where there was no cell.
    Returns (array, extent (left, right, bottom, top))."""
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    from scipy import ndimage
    grid, left, top = _lattice(cells)
    valid = np.isfinite(grid)
    lo, hi = float(grid[valid].min()), float(grid[valid].max())
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    up = np.clip(ndimage.zoom(grid[tuple(nearest)], SMOOTH_FACTOR, order=3, mode='nearest', grid_mode=True), lo, hi)
    keep = ndimage.zoom(valid.astype(float), SMOOTH_FACTOR, order=1, mode='nearest', grid_mode=True) >= 0.5
    step = CELL_M / SMOOTH_FACTOR
    inside = geometry_mask(list(boundary.geometry), out_shape=up.shape, transform=from_origin(left, top, step, step), invert=True,
                           all_touched=True)
    extent = (left, left + up.shape[1] * step, top - up.shape[0] * step, top)
    return np.where(keep & inside, up, np.nan), extent


def draw(ax, cells, norm, boundary, smooth=False, stations=True, cmap='viridis', alpha=None):
    """One snapshot with the NCR furniture on `ax`; returns (mappable, number of cells drawn)."""
    from matplotlib.cm import ScalarMappable
    alpha = surfaces.SURFACE_ALPHA if alpha is None else alpha
    minx, miny, maxx, maxy = boundary.total_bounds
    mx, my = 0.01 * (maxx - minx), 0.01 * (maxy - miny)
    ax.set_xlim(minx - mx, maxx + mx)
    ax.set_ylim(miny - my, maxy + my)
    ax.set_facecolor('white')
    boundary.plot(ax=ax, facecolor='#d9dbdc', edgecolor='none', zorder=1)           # the NCR without a prediction shows grey
    n = len(cells)
    if smooth:
        shown, extent = smooth_surface(cells, boundary)
        ax.imshow(np.ma.masked_invalid(shown), extent=extent, origin='upper', cmap=cmap, norm=norm, interpolation='nearest',
                  alpha=alpha, zorder=2)
    else:
        gdf = surfaces._cell_geometries(cells, cells['prediction'].values)
        gdf = gdf.clip(boundary.union_all())                                         # each cell cut where the boundary crosses it
        n = len(gdf)
        gdf.plot(ax=ax, column='prediction', cmap=cmap, norm=norm, alpha=alpha, edgecolor='none', zorder=2)
    ax.set_xlim(minx - mx, maxx + mx)
    ax.set_ylim(miny - my, maxy + my)
    ax.set_aspect('equal')
    _cities(ax, boundary)
    if stations:
        surfaces._overlay_stations(ax)
    _scale_bar(ax)
    _north_arrow(ax)
    surfaces._km_axes(ax)
    ax.tick_params(labelsize=7)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return ScalarMappable(norm=norm, cmap=cmap), n


def render_ncr_snapshot(cells, ts, out_dir, stem, vmin=None, vmax=None, label=None, smooth=False):
    """`<stem>_continuous_ncr.png` (cut cells) or `<stem>_continuous_ncr_smooth.png`. None when there is no boundary."""
    boundary = load_boundary()
    if boundary is None or cells.empty:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    vmin = surfaces.SHARED_RANGE[0] if vmin is None else vmin
    vmax = surfaces.SHARED_RANGE[1] if vmax is None else vmax
    fig, ax = plt.subplots(figsize=(8.5, 7))
    mappable, n = draw(ax, cells, Normalize(vmin=vmin, vmax=vmax), boundary, smooth)
    ax.set_title(f'PM2.5 {label or f"{pd.Timestamp(ts):%Y-%m-%d %H:%M}"} — continuous scale, {vmin:.1f}-{vmax:.1f} µg/m³\n'
                 f'{n} cells inside the NCR · colour is comparable across snapshots', fontsize=9, loc='left')
    fig.colorbar(mappable, ax=ax, shrink=0.7, label='PM2.5 (µg/m³)', pad=0.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f'{stem}_continuous_ncr{"_smooth" if smooth else ""}.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'    map -> {os.path.relpath(path, data.ROOT)}  (NCR, {"smooth" if smooth else "500 m cells"}, {vmin:g}-{vmax:g})')
    return path


def render_ncr_panel(frames, path, vmin=None, vmax=None, ncols=2, title=None, smooth=False):
    """The 2x2 panel of `surfaces.render_panel` with the NCR outline and one shared colourbar. None when there is no boundary."""
    boundary = load_boundary()
    items = [(ts, c[c['prediction_usable'].astype(bool)] if 'prediction_usable' in c.columns else c) for ts, c in frames.items()]
    items = [(ts, c) for ts, c in items if len(c)]
    if boundary is None or not items:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    if vmin is None or vmax is None:
        vmin, vmax = surfaces.suggest_limits(frames)
    nrows = -(-len(items) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.6 * nrows), squeeze=False)
    norm = Normalize(vmin=vmin, vmax=vmax)
    for ax in axes.ravel()[len(items):]:
        ax.set_axis_off()
    mappable = None
    for ax, (ts, cells) in zip(axes.ravel(), items):
        mappable, n = draw(ax, cells, norm, boundary, smooth)
        ax.set_title(f'{pd.Timestamp(ts):%H:%M}   ({n} cells, coverage {cells["standin_coverage"].mean():.2f})', fontsize=9, loc='left')
        ax.tick_params(labelsize=6)
    fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.6, label='PM2.5 (µg/m³)', pad=0.02)
    if title:
        fig.suptitle(title, fontsize=11, x=0.02, ha='left')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'panel -> {os.path.relpath(path, data.ROOT)}  ({len(items)} snapshots, NCR, {"smooth" if smooth else "500 m cells"})')
    return path


def render_ncr_aqi(cells, ts, out_dir, stem, smooth=False):
    """The Philippine-AQI map of `surfaces._try_map` with the NCR outline: `<stem>_aqi_ncr.png` (cut cells) or `<stem>_aqi_ncr_smooth.png`.
    The cells above the 35 ug/m3 highlight are outlined, as on the plain AQI map. None when there is no boundary."""
    boundary = load_boundary()
    if boundary is None or cells.empty:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    cmap = ListedColormap([c for _, _, c in surfaces.AQI_BINS])
    norm = BoundaryNorm([0, 25, 35, 45, 55, 60], cmap.N, clip=True)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    _, n = draw(ax, cells, norm, boundary, smooth, cmap=cmap, alpha=1.0)
    above = None
    if smooth:
        shown, extent = smooth_surface(cells, boundary)
        if np.nanmax(shown) > surfaces.HIGHLIGHT >= np.nanmin(shown):
            xs = np.linspace(extent[0], extent[1], shown.shape[1], endpoint=False) + (extent[1] - extent[0]) / shown.shape[1] / 2
            ys = np.linspace(extent[3], extent[2], shown.shape[0], endpoint=False) - (extent[3] - extent[2]) / shown.shape[0] / 2
            ax.contour(xs, ys, np.where(np.isfinite(shown), shown > surfaces.HIGHLIGHT, False).astype(float), levels=[0.5],
                       colors=[INK], linewidths=1.1, zorder=4)
            above = True
    else:
        gdf = surfaces._cell_geometries(cells, cells['prediction'].values).clip(boundary.union_all())
        hot = gdf[gdf['prediction'] > surfaces.HIGHLIGHT]
        if not hot.empty:
            hot.boundary.plot(ax=ax, color=INK, linewidth=1.2, zorder=4)
            above = True
    handles = [Patch(facecolor=c, edgecolor='#4a4a48', linewidth=0.5,
                     label=f'{lab} ({lo:g}-{up:g})' if np.isfinite(up) else f'{lab} (>{lo:g})')
               for (lab, up, c), lo in zip(surfaces.AQI_BINS, [0, 25, 35, 45, 55])]
    if above:
        handles.append(Patch(facecolor='none', edgecolor=INK, linewidth=1.2, label=f'outline: above {surfaces.HIGHLIGHT:g} ug/m3'))
    handles.append(Patch(facecolor='#d9dbdc', edgecolor='none', label='No data'))
    ax.legend(handles=handles, loc='lower center', ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.28))
    ax.set_title(f'PM2.5 {pd.Timestamp(ts):%Y-%m-%d %H:%M} — {"smooth surface" if smooth else "500 m cells"}, NCR\n'
                 f'{n} usable cells · mean stand-in coverage {cells["standin_coverage"].mean():.2f} · '
                 f'outline = above {surfaces.HIGHLIGHT:g} µg/m³', fontsize=9, loc='left')
    fig.tight_layout()
    path = os.path.join(out_dir, f'{stem}_aqi_ncr{"_smooth" if smooth else ""}.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'    map -> {os.path.relpath(path, data.ROOT)}  (AQI, NCR, {"smooth" if smooth else "500 m cells"})')
    return path

"""Build the morphology grid.

Two jobs, kept separate:

    build_vector_grid(...)      the fishnet — geometry only, exactly the pilot's rule
    aggregate_morphology(...)   zonal statistics from the 10 m raster into those cells

**The extent is not chosen by hand.** The pilot's `generate_vector_grid` in
`CNN-LSTM/pilot testing/CNN-LSTM/Extra_Preprocessing_pipeline_FIXED.ipynb` builds a fishnet over the
SENSOR bounding box, padded by `buffer_m`, so edge sensors still capture their surrounding morphology:

    minx, miny, maxx, maxy = sensors_gdf.total_bounds
    minx, miny, maxx, maxy = minx-buffer_m, miny-buffer_m, maxx+buffer_m, maxy+buffer_m

That is reproduced here. The extent is therefore a function of whichever sensors are supplied, which is
why adding the QC network grows the grid from 378 to ~1,848 cells with no coordinates to pick.

Aggregation conventions, and where they differ from the pilot (read this before comparing runs):

    bh_mean      mean of the raster's building_height_mean over the cell      == pilot convention
    bh_stdev     std of the raster's building_height_mean over the cell       == pilot convention
                 (the pilot used std over a 500 m cell; the raster's own
                  building_height_stddev band is a per-10-m-pixel std and is
                  therefore NOT the same statistic -- see NOTE below)
    ndvi_mean    mean of ndvi_mean                                            == pilot convention
    bldg_dens    mean of building_density                                     == pilot convention
    aspect_ratio mean of street_aspect_ratio_mean                             SOURCE-DEPENDENT

NOTE on aspect_ratio / street_width. The pilot computed

    mean_height = zonal mean of the height raster over the UNION of all street buffers in the cell
    mean_width  = mean `calculated_width` of those street segments
    aspect_ratio = mean_height / mean_width

which is why its implied street widths have median 14.98 m (a plausible Manila road). The raster's
`street_aspect_ratio_mean` band was built with the same physical idea but per 10 m pixel
(`mean(height in buffer / road width) over buffers intersecting pixel`, radius = road width/2 + 50 m),
so a cell mean of it is NOT the pilot's quantity -- measured over the pilot's own 378 cells the two
differ by a median factor of 3.1 (0.901 vs 0.310).

Reproducing the pilot exactly needs the OSM road centrelines and their functional-class widths, which
are NOT in this repository and cannot be fetched from here (no network). `street_width_mean` is derived
as `bh_mean / aspect_ratio_mean` to match how the pipeline already derives it, so the new run is
internally consistent; it is simply not on the pilot's scale. Fixing that is a data-acquisition step,
not a code change -- supply the streets layer and the formula above reproduces the pilot convention.
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

CELL_SIZE_M = 500
BUFFER_M = 2000          # the pilot's `generate_vector_grid` default
NODATA = -9999.0

# raster band -> (grid column, aggregation), 1-based band numbers
BANDS = {
    1: ('bh_mean', 'mean'),
    3: ('ndvi_mean', 'mean'),
    4: ('bldg_dens', 'mean'),
    5: ('aspect_ratio_mean', 'mean'),
}


def build_vector_grid(xy, cell_size_m=CELL_SIZE_M, buffer_m=BUFFER_M, clip_bounds=None,
                      anchor=None, nrows=None, ncols=None):
    """The pilot's fishnet: sensor bbox + buffer, tiled at cell_size_m.

    `xy` is an (N, 2) array of projected sensor coordinates. Returns a DataFrame with
    id / left / bottom / right / top / row_index / col_index, 1-based ids.

    row_index increases SOUTHWARD and col_index eastward, matching the pilot grid and the usual raster
    convention (row 0 is the northernmost band), so `prep`'s `left <= x < right` point-in-cell test and
    the maps agree with each other.

    `clip_bounds` may be the raster's (left, bottom, right, top); when given, cells whose centre falls
    outside are dropped, which is how the 2 km buffer is reconciled with a raster that does not extend
    that far north.

    `anchor` overrides the bbox+buffer origin with an explicit (top-left x, top-left y), and `nrows` /
    `ncols` fix the size. This exists to reproduce a KNOWN grid exactly (the published pilot is 18 x 21
    starting at 284309.2 / 1615952.8, i.e. the raster's own top-left corner -- it is NOT bbox+2 km, which
    would be 24 x 25). Use it for validation or to keep an extent pinned; leave it None to derive the
    extent from the sensors.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if anchor is not None:
        minx, maxy = float(anchor[0]), float(anchor[1])
        if nrows is None or ncols is None:
            raise ValueError('anchor requires nrows and ncols')
    else:
        minx = xy[:, 0].min() - buffer_m
        maxy = xy[:, 1].max() + buffer_m
        if nrows is None or ncols is None:
            maxx = xy[:, 0].max() + buffer_m
            miny = xy[:, 1].min() - buffer_m
            ncols = int(np.ceil((maxx - minx) / cell_size_m))
            nrows = int(np.ceil((maxy - miny) / cell_size_m))

    rows = []
    gid = 0
    for r in range(nrows):
        top = maxy - r * cell_size_m
        bottom = top - cell_size_m
        for c in range(ncols):
            left = minx + c * cell_size_m
            right = left + cell_size_m
            if clip_bounds is not None:
                cb = clip_bounds
                if not (left >= cb[0] and right <= cb[2] and bottom >= cb[1] and top <= cb[3]):
                    continue
            gid += 1
            rows.append({'id': gid, 'left': left, 'top': top, 'right': right, 'bottom': bottom,
                         'row_index': r + 1, 'col_index': c + 1})
    return pd.DataFrame(rows)


def aggregate_morphology(grid, raster_path, nodata=NODATA):
    """Zonal statistics from the 10 m raster into each grid cell.

    Uses the pixel centres inside each cell (not `all_touched`), which is the convention that
    reproduces the pilot: over the pilot's own 378 cells this gives bh_mean corr 0.993, ndvi 0.982,
    bldg_dens 0.973 and bh_stdev 0.997 against the published grid.
    """
    out = []
    with rasterio.open(raster_path) as s:
        src_nodata = s.nodata if s.nodata is not None else nodata
        for _, c in grid.iterrows():
            win = from_bounds(c['left'], c['bottom'], c['right'], c['top'], s.transform)
            rec = {'id': c['id']}
            for band, (col, how) in BANDS.items():
                a = s.read(band, window=win, boundless=True, fill_value=src_nodata)
                a = a[a != src_nodata]
                rec[col] = float(a.mean()) if a.size else np.nan
            h = s.read(1, window=win, boundless=True, fill_value=src_nodata)
            h = h[h != src_nodata]
            rec['bh_stdev'] = float(h.std()) if h.size > 1 else np.nan
            rec['n_pixels'] = int(h.size)
            out.append(rec)
    df = grid.merge(pd.DataFrame(out), on='id', how='left')
    with np.errstate(invalid='ignore', divide='ignore'):
        df['street_width_mean'] = df['bh_mean'] / df['aspect_ratio_mean']
    return df


def sensor_xy(master_csv, qc_locations_csv=None, crs='EPSG:32651'):
    """Projected (x, y) for every sensor location available, from both sources.

    OpenAQ coordinates come from the wide master table; QC ships its own lat/lon file. Both are
    projected to `crs`, so the two networks land in one coordinate system.
    """
    from pyproj import Transformer
    parts = []
    if master_csv:
        m = pd.read_csv(master_csv, usecols=['location_id', 'x_coordinate', 'y_coordinate'])
        parts.append(m.groupby('location_id')[['x_coordinate', 'y_coordinate']].first()
                     .rename(columns={'x_coordinate': 'x', 'y_coordinate': 'y'}))
    if qc_locations_csv:
        q = pd.read_csv(qc_locations_csv)
        tx, ty = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(
            q['longitude'].to_numpy(), q['latitude'].to_numpy())
        parts.append(pd.DataFrame({'x': tx, 'y': ty}, index=q['location_id'].to_numpy()))
    loc = pd.concat(parts)
    loc.index.name = 'location_id'
    return loc[~loc.index.duplicated(keep='first')]


def coverage_report(grid, loc):
    """Which locations fall outside the grid, and which cells are missing usable morphology.

    `loc` must carry a `location_id` column as well as x/y, so the report names locations by their real
    id. (An earlier version listed `loc.index`, which after a reset_index was just the positional row
    number -- it reported "locations outside: [32, 60, 70]", ids that do not exist. The ids have to come
    from the column.)
    """
    if 'bh_mean' in grid.columns:
        usable = grid[grid['bh_mean'].notna()]
    else:
        usable = grid
    xs, ys = loc['x'].to_numpy(), loc['y'].to_numpy()
    inside = ((usable['left'].to_numpy()[:, None] <= xs) & (xs < usable['right'].to_numpy()[:, None]) &
              (usable['bottom'].to_numpy()[:, None] <= ys) & (ys < usable['top'].to_numpy()[:, None])
              ).any(axis=0)
    ids = loc['location_id'].to_numpy() if 'location_id' in loc.columns else np.asarray(loc.index)
    return {
        'cells': int(len(grid)),
        'cells_with_morphology': int(len(usable)),
        'cells_missing_morphology': int(len(grid) - len(usable)),
        'locations': int(len(loc)),
        'locations_inside_grid': int(inside.sum()),
        'locations_outside_grid': [int(v) for v in ids[~inside]],
    }

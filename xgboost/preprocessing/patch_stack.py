"""Patch-statistic stack at 10 m: morphology features centred on EVERY 10 m pixel.

Why a stack and not a grid CSV
------------------------------
At 500 m the grid is 3,696 rows, so a CSV is the natural container. At 10 m it is 10.6 M rows
(~1.5 GB), which no stage can comfortably hold. But the features themselves do not need cells: a
feature at a pixel is a box statistic of the raster around that pixel, and a box filter produces the
whole 10.6 M-cell field in seconds. So the "grid" becomes six float32 rasters of the raster's own
shape -- same geometry, no table.

This also makes the features well defined at any output resolution (a 32 px window centred on a 10 m
pixel is the same kind of object as one centred on a station), which is what a cell aggregate cannot
offer.

Bands used, matching grid_builder.BANDS:
    band 1 -> bh_mean (mean) and bh_stdev (std)
    band 3 -> ndvi_mean
    band 4 -> bldg_dens
    band 5 -> aspect_ratio_mean
plus street_width_mean = bh_mean / aspect_ratio_mean, as in the 500 m builder.
"""
import os

import numpy as np

import data

WINDOW_PX = 32                      # 32 px at 10 m = 320 m, matching the CNN-LSTM patch
FEATURE_COLUMNS = ['bh_mean', 'bh_stdev', 'ndvi_mean', 'bldg_dens', 'aspect_ratio_mean',
                   'street_width_mean']
STACK_PATH = os.path.join(data.DATA_DIR, 'patch_stack_10m.npz')


def _box(a, win):
    """Window-sum of `a` (already NaN-free), float32."""
    from scipy.ndimage import uniform_filter
    return uniform_filter(a.astype(np.float32), win, mode='nearest')


def _mean(std, valid_num, win):
    return (_box(std, win) / np.maximum(_box(valid_num, win), 1e-6)).astype(np.float32)


def build(raster_path=None, window_px=WINDOW_PX, verbose=True):
    """Compute the six feature rasters from the 10 m morphology GeoTIFF.

    Returns a dict of float32 arrays plus metadata:
        arrays : bh_mean, bh_stdev, ndvi_mean, bldg_dens, aspect_ratio_mean, street_width_mean
        valid  : bool array, True where the CENTRE pixel has a valid height reading
        meta   : transform, crs, shape, window_px
    """
    import rasterio

    raster_path = raster_path or data.RASTER_TIF
    with rasterio.open(raster_path) as src:
        h, w = src.height, src.width
        bands = {}
        for b in (1, 3, 4, 5):
            a = src.read(b).astype(np.float32)
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            bands[b] = a
        transform, crs = src.transform, src.crs

    bh = bands[1]
    # The centre pixel exists (not nodata). This is coverage, not water: water is finite 0.0.
    px_ok = np.isfinite(bh)
    vnum = px_ok.astype(np.float32)

    out = {}
    got = _box(np.where(px_ok, bh, 0.0), window_px)
    cnt = _box(vnum, window_px)
    out['bh_mean'] = np.where(cnt > 0.05, got / np.maximum(cnt, 1e-6), np.nan).astype(np.float32)
    out['bh_stdev'] = np.sqrt(np.clip(
        _mean(np.where(px_ok, bh * bh, 0.0), vnum, window_px) - out['bh_mean'] ** 2, 0, None)
    ).astype(np.float32)
    for b, name in ((3, 'ndvi_mean'), (4, 'bldg_dens'), (5, 'aspect_ratio_mean')):
        a = bands[b]
        ok = np.isfinite(a).astype(np.float32)
        out[name] = np.where(_box(ok, window_px) > 0.05,
                             _box(np.where(ok > 0, a, 0.0), window_px) / np.maximum(_box(ok, window_px), 1e-6),
                             np.nan).astype(np.float32)
    with np.errstate(invalid='ignore', divide='ignore'):
        out['street_width_mean'] = np.where(
            out['aspect_ratio_mean'] > 0,
            out['bh_mean'] / out['aspect_ratio_mean'], np.nan).astype(np.float32)

    # Water is a WINDOW property, not a pixel one: a road or an empty lot has height 0 and low NDVI
    # at its centre pixel, exactly like water, but its 320 m window holds buildings or trees. Only a
    # window that is entirely flat, unvegetated and unbuilt is water -- which is what the bays are.
    # ndvi < 0 (not merely low) is the discriminator: the bays read -0.01 and -0.36, while open
    # land, farmland and playing fields read positive. Using 0.02 punched blocky holes in the south.
    water = ((np.nan_to_num(out['bh_mean']) <= 0)
             & (np.nan_to_num(out['ndvi_mean']) < 0.0)
             & (np.nan_to_num(out['bldg_dens']) <= 0.01))
    valid = px_ok & ~water
    # The DSM has isolated 10 m gaps inside built-up areas (courtyards, covered walkways, informal
    # settlements). At 10 m they read as a checkerboard of holes that the 500 m aggregate averaged
    # away, so close gaps narrower than ~70 m. The bays are far wider, and sit at the raster edge,
    # so they survive the closing.
    from scipy.ndimage import binary_closing
    valid = binary_closing(valid, structure=np.ones((7, 7), dtype=bool))
    # The window test alone still paints water NEAR the shore: a pixel in Manila Bay with land inside
    # its 320 m window reads bh_mean > 0, so it survived and appeared as a speckle of points at sea.
    # A pixel must also not be water itself -- zero height and negative NDVI at its own centre.
    water_px = (np.nan_to_num(bh) <= 0) & (np.nan_to_num(bands[3]) < 0.0)
    valid = valid & ~water_px

    meta = {'transform': tuple(transform)[:6], 'crs': str(crs), 'shape': (h, w),
            'window_px': int(window_px)}
    if verbose:
        print('  patch stack: %d x %d px | window %d px (%.0f m) | %.0f MB in memory'
              % (w, h, window_px, window_px * abs(transform.a),
                 sum(v.nbytes for v in out.values()) / 1e6))
        for k in FEATURE_COLUMNS:
            a = out[k]
            print('    %-20s finite %.1f%%  mean %.3f  sd %.3f'
                  % (k, 100 * np.isfinite(a).mean(), float(np.nanmean(a)), float(np.nanstd(a))))
    return {'arrays': out, 'valid': valid, 'meta': meta}


def save(stack, path=None):
    path = path or STACK_PATH
    payload = {k: v for k, v in stack['arrays'].items()}
    payload['valid'] = stack['valid']
    for k, v in stack['meta'].items():
        payload['meta_' + k] = np.array(v)
    np.savez_compressed(path, **payload)
    print('  wrote', path, '(%.0f MB)' % (os.path.getsize(path) / 1e6))
    return path


def load(path=None):
    path = path or STACK_PATH
    if not os.path.exists(path):
        raise FileNotFoundError('%s is missing -- run `python run_pipeline.py --stage grid10`' % path)
    z = np.load(path, allow_pickle=False)
    arrays = {k: z[k] for k in FEATURE_COLUMNS if k in z}
    meta = {'transform': tuple(float(x) for x in z['meta_transform']),
            'crs': str(z['meta_crs']), 'shape': tuple(int(x) for x in z['meta_shape']),
            'window_px': int(z['meta_window_px'])}
    return {'arrays': arrays, 'valid': z['valid'], 'meta': meta}


def features_at(stack, rows, cols):
    """Feature row(s) at pixel indices, as a dict of 1-D arrays (no table needed)."""
    return {k: stack['arrays'][k][rows, cols] for k in FEATURE_COLUMNS}


def xy_of_pixels(meta, rows, cols):
    """Centroid coordinates of pixel indices, in the raster's CRS."""
    a, b, c, d, e, f = meta['transform']
    return (a * (cols + 0.5) + b * (rows + 0.5) + c,
            d * (cols + 0.5) + e * (rows + 0.5) + f)

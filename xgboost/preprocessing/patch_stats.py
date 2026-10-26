"""Phase 2: per-station statistics from the 10 m raster over the CNN-LSTM's 320 m patch.

Why this exists. The CNN-LSTM branch reads a 32x32-pixel patch at 10 m (320 m) and convolves it, so it
sees how morphology is _arranged_ inside a cell: local spread, extremes, skew. The XGBoost branch only
ever saw six zonal means per 500 m cell, so the two branches were not given the same information -- which
makes any "CNN beats XGBoost" claim a comparison of inputs, not of architectures.

This computes, for every sensor cell, the same 320 m window at 10 m resolution and summarises each band
by mean / std / min / max / p10 / p50 / p90. All from public raster statistics, no model. That lets
XGBoost compete on texture without pretending to be a convolutional network.

    python -m preprocessing.patch_stats        # -> data/patch_stats.csv (one row per sensor cell)

Columns are prefixed `tex_` and suffixed by the statistic, e.g. `tex_bh_std`, `tex_ndvi_p10`.
NaNs are ignored per band (the aspect-ratio band is nodata over ~55% of the raster), and a band with no
valid pixel in a window yields NaN for that column rather than a fabricated 0.
"""
import json
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

import data

PATCH_EXTENT_M = 320.0        # their data.yaml patch.extent_m
STATS = ('mean', 'std', 'min', 'max', 'p10', 'p50', 'p90')
BANDS = {1: 'bh', 2: 'bh_std_px', 3: 'ndvi', 4: 'dens', 5: 'aspect'}
OUT_CSV = os.path.join(data.DATA_DIR, 'patch_stats.csv')


def stats_for_window(a, nodata):
    v = a[a != nodata]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {s: np.nan for s in STATS}
    return {'mean': float(v.mean()), 'std': float(v.std()), 'min': float(v.min()),
            'max': float(v.max()), 'p10': float(np.percentile(v, 10)),
            'p50': float(np.percentile(v, 50)), 'p90': float(np.percentile(v, 90))}


def impute_texture(out, cells, k=4):
    """IDW-fill NaN texture columns for cells that sit in the split.

    Three cells (80, 169, 219) are nodata in the building-height band of the 10 m raster, so their
    patch statistics are NaN. XGBoost would tolerate that (it learns a default direction for missing
    values), but the rest of the pipeline has already imputed 500 m morphology for the same cells, so
    leaving texture NaN would make the TEXTURE arm silently decide those stations by a different rule.
    Filling them the same way keeps the imputation story single and explicit.
    """
    from scipy.spatial import cKDTree

    tex = [c for c in out.columns if c.startswith('tex_')]
    bad = out[out[tex].isna().any(axis=1)]
    good = out[out[tex].notna().all(axis=1)]
    if bad.empty or good.empty:
        return out, []

    cen = dict(zip(cells['id'], zip((cells['left'] + cells['right']) / 2,
                                    (cells['bottom'] + cells['top']) / 2)))
    tree = cKDTree(np.array([cen[i] for i in good['Grid_ID']]))
    out = out.copy()
    filled = []
    for idx, row in bad.iterrows():
        d, j = tree.query(np.array(cen[int(row['Grid_ID'])]), k=min(k, len(good)))
        d, j = np.atleast_1d(d), np.atleast_1d(j)
        w = 1.0 / np.maximum(d, 1.0)
        w = w / w.sum()
        for col in tex:
            if pd.isna(out.at[idx, col]):
                out.at[idx, col] = float(np.dot(w, good[col].to_numpy(float)[j]))
        filled.append(int(row['Grid_ID']))
    return out, filled


def main():
    mapping = pd.read_csv(data.SENSOR_MAP_CSV)
    cells = (mapping.groupby('id')
             .agg(left=('left', 'first'), bottom=('bottom', 'first'),
                  right=('right', 'first'), top=('top', 'first'))
             .reset_index())
    print('computing %s m patch statistics for %d sensor cells from %s'
          % (PATCH_EXTENT_M, len(cells), os.path.basename(data.RASTER_TIF)))

    rows = []
    with rasterio.open(data.RASTER_TIF) as s:
        nod = s.nodata if s.nodata is not None else -9999.0
        px = int(round(PATCH_EXTENT_M / s.res[0]))
        for _, c in cells.iterrows():
            cx = (c['left'] + c['right']) / 2.0
            cy = (c['bottom'] + c['top']) / 2.0
            half = PATCH_EXTENT_M / 2.0
            win = from_bounds(cx - half, cy - half, cx + half, cy + half, s.transform)
            rec = {'Grid_ID': int(c['id']), 'patch_px': px}
            for band, name in BANDS.items():
                a = s.read(band, window=win, boundless=True, fill_value=nod)
                for stat, val in stats_for_window(a, nod).items():
                    rec['tex_%s_%s' % (name, stat)] = val
            rows.append(rec)

    out = pd.DataFrame(rows)
    nan_before = sorted(out.loc[out.filter(like='tex_').isna().any(axis=1), 'Grid_ID'].astype(int))
    out, filled = impute_texture(out, cells)
    if filled:
        print('  IDW-imputed patch statistics for cells %s (nodata in the height band)' % filled)
        with open(os.path.join(data.DATA_DIR, 'imputed_texture_cells.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump({'cells': filled, 'k': 4, 'had_nan': nan_before,
                       'method': 'inverse-distance weighted mean of the k nearest complete cells',
                       'reason': 'nodata in master_raster_v2.tif building-height band'}, fh, indent=2)

    out.to_csv(OUT_CSV, index=False)
    fin = out.drop(columns=['Grid_ID', 'patch_px'])
    print('  wrote %s  (%d rows, %d features)' % (OUT_CSV, len(out), fin.shape[1]))
    print('  patch side: %d px at %.0f m = %.0f m' % (px, 10.0, px * 10.0))
    print('  NaN remaining in texture columns: %d' % int(fin.isna().sum().sum()))
    print('\n  %-18s %8s %8s %8s' % ('feature', 'min', 'median', 'max'))
    for col in ['tex_bh_mean', 'tex_bh_std', 'tex_ndvi_mean', 'tex_ndvi_p10',
                'tex_dens_mean', 'tex_dens_p90', 'tex_aspect_mean']:
        v = out[col]
        print('  %-18s %8.3f %8.3f %8.3f' % (col, v.min(), v.median(), v.max()))



if __name__ == '__main__':
    main()

"""Build the combined sensor-location table and the new morphology grid.

Run from XGBoost/. Writes:
    data/sensor_locations.csv        every location from both networks, with lat/lon and source
    data/morphology_grid_full.csv    the new fishnet + morphology
    data/morphology_grid_full.json   the same facts as a record
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import rasterio

import data
from preprocessing import grid_builder as gb

_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('--raster', default='data/master_raster_v2.tif', help='source morphology GeoTIFF')
_ap.add_argument('--openaq-locations',
                 default='../CNN-LSTM/pilot testing/CNN-LSTM/'
                         'OpenAQ Sensor Locations - OpenAQ Sensor Locations.csv')
_ap.add_argument('--qc-locations', default='data/QC Data Converted/QC Sensor Locations.csv')
_ap.add_argument('--out-dir', default=data.DATA_DIR)
_ap.add_argument('--buffer-m', type=int, default=gb.BUFFER_M,
                 help='padding around the sensor bbox (the pilot used 2000)')
_ap.add_argument('--cell-size-m', type=int, default=gb.CELL_SIZE_M)
ARGS = _ap.parse_args()

V2 = ARGS.raster
OPENAQLOC = ARGS.openaq_locations
QCLOC = ARGS.qc_locations
OUT_DIR = ARGS.out_dir
gb.BUFFER_M = ARGS.buffer_m
gb.CELL_SIZE_M = ARGS.cell_size_m

# ---------------------------------------------------------------- 1. one locations table
op = pd.read_csv(OPENAQLOC).rename(columns={'name': 'location_name'})
op['source'] = 'OpenAQ'
qc = pd.read_csv(QCLOC).rename(columns={'name': 'location_name'})
qc['source'] = 'QC'
loc = pd.concat([op[['location_id', 'location_name', 'latitude', 'longitude', 'source']],
                 qc[['location_id', 'location_name', 'latitude', 'longitude', 'source']]],
                ignore_index=True)
loc = loc.drop_duplicates(subset='location_id', keep='first').sort_values('location_id')
loc.to_csv(os.path.join(OUT_DIR, 'sensor_locations.csv'), index=False)
print('=== sensor_locations.csv ===')
print('  %d locations  (%d OpenAQ + %d QC)'
      % (len(loc), (loc.source == 'OpenAQ').sum(), (loc.source == 'QC').sum()))
print('  lat %.4f..%.4f   lon %.4f..%.4f'
      % (loc.latitude.min(), loc.latitude.max(), loc.longitude.min(), loc.longitude.max()))

# ---------------------------------------------------------------- 2. projected xy
from pyproj import Transformer
x, y = Transformer.from_crs('EPSG:4326', 'EPSG:32651', always_xy=True).transform(
    loc.longitude.to_numpy(), loc.latitude.to_numpy())
xy = np.column_stack([x, y])
print('\n=== projected (EPSG:32651) ===')
print('  x %.0f..%.0f (%.1f km)   y %.0f..%.0f (%.1f km)'
      % (x.min(), x.max(), (x.max() - x.min()) / 1000, y.min(), y.max(), (y.max() - y.min()) / 1000))

# ---------------------------------------------------------------- 3. the grid
with rasterio.open(V2) as s:
    rb = (s.bounds.left, s.bounds.bottom, s.bounds.right, s.bounds.top)
    print('\n=== raster bounds ===')
    print('  x %.0f..%.0f  y %.0f..%.0f' % (rb[0], rb[2], rb[1], rb[3]))

full = gb.build_vector_grid(xy)
clip = gb.build_vector_grid(xy, clip_bounds=rb)
print('\n=== grid ===')
print('  bbox+2km      : %d cells (%d cols x %d rows)'
      % (len(full), full.col_index.nunique(), full.row_index.nunique()))
print('  clipped to v2 : %d cells  (dropped %d beyond the raster edge)'
      % (len(clip), len(full) - len(clip)))

grid = gb.aggregate_morphology(clip, V2)
print('  merged morphology into %d cells (%d missing)'
      % (int(grid['bh_mean'].notna().sum()), int(grid['bh_mean'].isna().sum())))

loc_xy = pd.DataFrame({'location_id': loc['location_id'].to_numpy(), 'x': x, 'y': y})
rep = gb.coverage_report(grid, loc_xy)
print('  locations inside grid: %d of %d' % (rep['locations_inside_grid'], rep['locations']))
if rep['locations_outside_grid']:
    print('  OUTSIDE: %s' % rep['locations_outside_grid'])

grid = grid.sort_values(['row_index', 'col_index']).reset_index(drop=True)
grid['id'] = np.arange(1, len(grid) + 1)
# Write BOTH column vocabularies. `aggregate_morphology` names columns after the pipeline's pilot
# convention (bh_mean, bh_stdev, ndvi_mean, bldg_dens, aspect_ratio_mean, street_width_mean), which is
# what preprocessing/prep.py and spatial_cv.py expect; the raster's own band vocabulary is added as
# aliases so a reader coming from the GeoTIFF finds the names it used.
grid['building_height_mean'] = grid['bh_mean']
grid['building_height_stddev'] = grid['bh_stdev']
grid['building_density'] = grid['bldg_dens']
grid['street_aspect_ratio_mean'] = grid['aspect_ratio_mean']
cols = ['id', 'left', 'top', 'right', 'bottom', 'row_index', 'col_index',
        'bh_mean', 'bh_stdev', 'ndvi_mean', 'bldg_dens', 'aspect_ratio_mean', 'street_width_mean',
        'building_height_mean', 'building_height_stddev', 'building_density',
        'street_aspect_ratio_mean']
grid[cols].to_csv(os.path.join(OUT_DIR, 'morphology_grid_full.csv'), index=False)
print('\nwrote data/morphology_grid_full.csv  (%d cells)' % len(grid))

with open(os.path.join(OUT_DIR, 'morphology_grid_full.json'), 'w', encoding='utf-8') as fh:
    json.dump({
        'source_raster': 'data/master_raster_v2.tif',
        'raster_bounds': rb,
        'cell_size_m': gb.CELL_SIZE_M,
        'buffer_m': gb.BUFFER_M,
        'clipped_to_raster': True,
        'cells': int(len(grid)),
        'cells_with_morphology': int(grid['bh_mean'].notna().sum()),
        'columns': 'cols %d rows %d' % (grid.col_index.nunique(), grid.row_index.nunique()),
        'locations_total': int(len(loc)),
        'locations_inside_grid': rep['locations_inside_grid'],
        'locations_outside_grid': rep['locations_outside_grid'],
        'aggregation': {c: ('std of band 1 within cell' if c == 'bh_stdev' else
                            'mean of band %d' % b) for b, c in
                        [(1, 'bh_mean'), (3, 'ndvi_mean'), (4, 'bldg_dens'), (5, 'aspect_ratio_mean')]},
        'aspect_ratio_caveat': ('mean of the raster\'s street_aspect_ratio_mean band; the pilot used '
                                'mean(H over merged street buffers)/mean(road width), which differs by a '
                                'median factor of ~3.1. Needs OSM streets to reproduce exactly.'),
        'street_width_derivation': 'bh_mean / aspect_ratio_mean (same as preprocessing/prep.py)',
    }, fh, indent=2, sort_keys=True)
print('wrote data/morphology_grid_full.json')
print('\n  %-20s %9s %9s %9s' % ('column', 'min', 'median', 'max'))
for c in ['bh_mean', 'bh_stdev', 'ndvi_mean', 'bldg_dens', 'aspect_ratio_mean', 'street_width_mean']:
    v = grid[c]
    print('  %-20s %9.3f %9.3f %9.3f' % (c, v.min(), v.median(), v.max()))

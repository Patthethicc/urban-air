"""Ingest the per-sensor CSVs into the wide table the rest of the pipeline already reads.

Why this exists: `Master_DL_Ready.csv` is a WIDE table (one row per location-hour) covering only the 18
pilot sensors, while the new data is 85 per-sensor files in an 8-column LONG schema:

    location_name, location_id, sensor_id, utc_timestamp, local_timestamp, pm25_value, parameter, unit

Nothing in this branch read that, so the new network could not be used at all. This step closes that gap
and writes a table with the SAME schema as `Master_DL_Ready.csv`, so `preprocessing/prep.py` and every
stage after it work unchanged.

    python ingest_sources.py                      # both sources -> Master_DL_Ready_full.csv
    python ingest_sources.py --sources OpenAQ     # one source only (per-source comparison)
    python ingest_sources.py --sources QC --out data/Master_DL_Ready_qc.csv

Pipeline per location:
    parse local_timestamp -> drop values outside [min_value, max_value] -> strict 1 h index (mean within
    the hour) -> interpolate ONLY gaps of at most --max-gap hours -> attach morphology from the grid

The gap rule is the important one, and it is the reference branch's (`clean_loader._fill_short_gaps`).
A gap of at most `max_gap` steps is linearly interpolated; a LONGER outage stays entirely missing. This
is deliberately not pandas' `limit=`, which fills the first `max_gap` hours of any gap and so fabricates
data across a multi-day outage. Our pilot pipeline interpolated nothing at all, which is why the drop
funnel removed 79,568 of 122,841 rows; capping the interpolation is what recovers that without inventing
long stretches of readings.

Outputs (in addition to the table):
    data/ingest_report.json    per-location rows in/out, gap statistics, morphology coverage
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

import data

# from the reference branch's configs/data.yaml
MIN_VALUE, MAX_VALUE = 0.0, 1000.0
MAX_GAP_HOURS = 24

SOURCES = {
    'OpenAQ': 'data/OpenAQ/openaq_*.csv',
    'QC': 'data/QC Data Converted/qcdata_*.csv',
}


def _fill_short_gaps(s, max_gap):
    """Linear interpolation inside gaps of at most `max_gap` steps; longer gaps stay entirely NaN."""
    missing = s.isna()
    if not missing.any():
        return s
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform('sum')
    filled = s.interpolate(method='linear', limit_area='inside')
    return filled.mask(missing & (run_len > max_gap))


def read_source(pattern, loc, min_value=MIN_VALUE, max_value=MAX_VALUE):
    """Long frame for every file matching `pattern`: location_id, DateTime, pm25_value."""
    frames, per_file = [], []
    for path in sorted(glob.glob(pattern)):
        raw = pd.read_csv(path)
        if 'local_timestamp' not in raw.columns or 'pm25_value' not in raw.columns:
            print('  skip (unexpected schema): %s' % os.path.basename(path))
            continue
        lid = int(raw['location_id'].iloc[0]) if 'location_id' in raw.columns else \
            int(re.match(r'(?:openaq|qcdata)_(\d+)_', os.path.basename(path)).group(1))
        if lid not in loc.index:
            print('  skip (no coordinates): %s' % os.path.basename(path))
            continue
        ts = pd.to_datetime(raw['local_timestamp'], format='mixed', utc=True, errors='coerce')
        ts = ts.dt.tz_convert('Asia/Manila').dt.tz_localize(None)
        val = pd.to_numeric(raw['pm25_value'], errors='coerce')
        val = val.where(val.between(min_value, max_value))
        n_raw = int(val.notna().sum())
        frame = pd.DataFrame({'location_id': lid, 'DateTime': ts, 'pm25_value': val}).dropna(
            subset=['DateTime'])
        frames.append(frame)
        per_file.append({'file': os.path.basename(path), 'location_id': lid, 'readings': n_raw})
    if not frames:
        raise FileNotFoundError('no sensor CSVs matched %s' % pattern)
    return pd.concat(frames, ignore_index=True), per_file


def to_hourly(df, max_gap=MAX_GAP_HOURS):
    """Strict 1 h index per location, short gaps interpolated. Returns the frame plus gap statistics."""
    out, stats = [], []
    for lid, grp in df.groupby('location_id', sort=True):
        s = (grp.set_index('DateTime')['pm25_value'].sort_index()
                .resample('1h').mean())
        have = int(s.notna().sum())
        filled = _fill_short_gaps(s, max_gap)
        added = int(filled.notna().sum()) - have
        # measure the gaps that were left alone
        miss = s.isna()
        run = (miss != miss.shift()).cumsum()
        lens = miss.groupby(run).transform('sum')[miss]
        long_gaps = int((lens > max_gap).sum()) if len(lens) else 0
        part = filled.rename('pm25_value').rename_axis('DateTime').reset_index()
        part['location_id'] = lid
        out.append(part)
        stats.append({'location_id': int(lid), 'hours_on_index': int(len(s)),
                      'observed_hours': have, 'interpolated_hours': added,
                      'hours_in_long_gaps_left_missing': long_gaps,
                      'pct_of_span_kept': round(100.0 * int(filled.notna().sum()) / max(1, len(s)), 2)})
    return pd.concat(out, ignore_index=True), stats


def attach_morphology(df, grid_path, loc_xy, grid_csv_used):
    """Point-in-cell join, mirroring prep.build_sensor_to_grid_map's bbox test."""
    grid = pd.read_csv(grid_path)
    rows = []
    for lid, r in loc_xy.iterrows():
        cell = grid[(grid['left'] <= r['x']) & (r['x'] < grid['right']) &
                    (grid['bottom'] <= r['y']) & (r['y'] < grid['top'])]
        if len(cell) == 0:
            print('  WARNING: location %s has no containing cell in %s -- dropped'
                  % (lid, os.path.basename(grid_path)))
            continue
        c = cell.iloc[0]
        rows.append({'location_id': int(lid), 'Grid_ID': int(c['id']),
                     'left': c['left'], 'top': c['top'], 'right': c['right'], 'bottom': c['bottom'],
                     'bh_mean': c['bh_mean'], 'bh_stdev': c['bh_stdev'], 'ndvi_mean': c['ndvi_mean'],
                     'bldg_dens': c['bldg_dens'], 'aspect_ratio_mean': c['aspect_ratio_mean'],
                     'street_width_mean': c['street_width_mean']})
    mapping = pd.DataFrame(rows)
    if mapping.empty:
        raise SystemExit('no location could be placed in %s' % grid_path)
    return df.merge(mapping, on='location_id', how='inner'), mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sources', nargs='+', default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument('--out', default=os.path.join(data.DATA_DIR, 'Master_DL_Ready_full.csv'))
    ap.add_argument('--grid', default=os.path.join(data.DATA_DIR, 'morphology_grid_full.csv'))
    ap.add_argument('--locations', default=os.path.join(data.DATA_DIR, 'sensor_locations.csv'))
    ap.add_argument('--max-gap', type=int, default=MAX_GAP_HOURS)
    args = ap.parse_args()

    from pyproj import Transformer
    loc = pd.read_csv(args.locations)
    x, y = Transformer.from_crs('EPSG:4326', 'EPSG:32651', always_xy=True).transform(
        loc.longitude.to_numpy(), loc.latitude.to_numpy())
    loc_xy = pd.DataFrame({'location_id': loc.location_id.to_numpy(), 'x': x, 'y': y,
                           'latitude': loc.latitude.to_numpy(),
                           'longitude': loc.longitude.to_numpy(),
                           'source': loc.source.to_numpy()}).set_index('location_id')

    print('=== reading sources: %s ===' % ', '.join(args.sources))
    frames, per_file = [], []
    for name in args.sources:
        f, pf = read_source(SOURCES[name], loc_xy)
        f['source'] = name
        frames.append(f)
        per_file += pf
        print('  %-7s %2d files, %s readings' % (name, len(pf), format(int(f.pm25_value.notna().sum()), ',')))
    long = pd.concat(frames, ignore_index=True)

    print('\n=== strict hourly index (max gap %d h) ===' % args.max_gap)
    hourly, stats = to_hourly(long.sort_values(['location_id', 'DateTime']), args.max_gap)
    print('  %s location-hours across %d locations' % (format(len(hourly), ','), hourly.location_id.nunique()))

    print('\n=== attach morphology from %s ===' % os.path.basename(args.grid))
    # source/lat/lon are static per location, so they ride along as columns and are mapped after the
    # merge rather than being carried through the groupby (which silently dropped `source`).
    merged, mapping = attach_morphology(hourly, args.grid, loc_xy, args.grid)
    merged['source'] = merged['location_id'].map(loc_xy['source'])
    print('  %d locations placed into %d distinct grid cells'
          % (len(mapping), mapping.Grid_ID.nunique()))

    # long -> wide, matching Master_DL_Ready.csv exactly so prep.py is untouched
    # long -> wide, matching Master_DL_Ready.csv exactly so prep.py is untouched.
    # The output column is `local_timestamp` carrying a real datetime, matching the pilot file. It must
    # NOT be called `DateTime`: prep.py does `pd.to_datetime(df['DateTime'])`, and pandas treats a STRING
    # argument as a unit-mapping spec, so a column literally named `DateTime` raises
    # "cannot assemble with duplicate keys".
    wide = merged.pivot_table(index=['location_id', 'DateTime'], values='pm25_value', aggfunc='mean')
    wide = wide.reset_index().rename(columns={'DateTime': 'local_timestamp'})
    stat = pd.DataFrame(stats).set_index('location_id')
    wide['source'] = wide['location_id'].map(loc_xy['source'])
    static = mapping.set_index('location_id')
    for col in ['Grid_ID', 'left', 'top', 'right', 'bottom', 'bh_mean', 'bh_stdev', 'ndvi_mean',
                'bldg_dens', 'aspect_ratio_mean', 'street_width_mean']:
        wide[col] = wide['location_id'].map(static[col])
    for col in ['latitude', 'longitude']:
        wide[col] = wide['location_id'].map(loc_xy[col])
    wide['x_coordinate'] = wide['location_id'].map(loc_xy['x'])
    wide['y_coordinate'] = wide['location_id'].map(loc_xy['y'])

    # prep.py derives these per sensor anyway; carried so the schema matches Master_DL_Ready.csv
    wide = wide.sort_values(['location_id', 'local_timestamp'])
    g = wide.groupby('location_id')['pm25_value']
    for n in (1, 2, 3):
        wide['pm25_t-%d' % n] = g.shift(n)

    # prep.py renames {'location_id': 'Location', 'local_timestamp': 'DateTime',
    #                  'pm25_target': 'Value'}, so the value column must be named `pm25_target` to match
    # Master_DL_Ready.csv. It also KEEPS the sibling columns (merge is how='left' and the frame is not
    # narrowed), which is why `source` and the morphology columns survive into df_clean.
    # CAREFUL: do NOT carry the morphology columns here. `prep` merges them from the mapping CSV, and if
    # the master table ALSO has `bldg_dens` etc., pandas suffix-resolves the clash to `bldg_dens_x` /
    # `bldg_dens_y`, so the plain `bldg_dens` the interaction terms need disappears with KeyError.
    # The pilot's Master_DL_Ready.csv carries only ids + timestamps + target, so this matches it exactly.
    cols = ['location_id', 'local_timestamp', 'pm25_target', 'source',
            'pm25_t-1', 'pm25_t-2', 'pm25_t-3', 'latitude', 'longitude',
            'x_coordinate', 'y_coordinate']
    wide = wide.rename(columns={'pm25_value': 'pm25_target'})[cols]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    wide.to_csv(args.out, index=False)

    rows_with_value = int(wide.pm25_target.notna().sum())
    print('\n=== wrote %s ===' % args.out)
    print('  %s rows, %d locations, %s with a value'
          % (format(len(wide), ','), wide.location_id.nunique(), format(rows_with_value, ',')))
    print('  by source: %s' % wide.groupby('source').size().to_dict())

    rep = {
        'sources': args.sources,
        'grid': os.path.basename(args.grid),
        'max_gap_hours': args.max_gap,
        'locations_with_data': int(wide.location_id.nunique()),
        'distinct_cells': int(merged.Grid_ID.nunique()),
        'rows': int(len(wide)),
        'rows_with_value': rows_with_value,
        'per_location': pd.DataFrame(stats).to_dict('records'),
        'per_file': per_file,
        'morphology': {'cells_used': int(merged.Grid_ID.nunique()),
                       'grid_cells_total': int(pd.read_csv(args.grid).shape[0])},
    }
    with open(os.path.join(data.DATA_DIR, 'ingest_report.json'), 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, indent=2, sort_keys=True, default=str)
    print('  report -> data/ingest_report.json')
    print('\n  median %% of each sensor\'s span kept: %.1f%%'
          % np.median([s['pct_of_span_kept'] for s in stats]))


if __name__ == '__main__':
    main()

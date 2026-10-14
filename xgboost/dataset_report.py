"""Deliverable 1: the dataset report.

Plan Section 3.3, "Descriptive statistics (Deliverable 1): rows kept and dropped, sensors, cells, date
range, missingness, feature distributions." Written to satisfy that deliverable for the XGBoost branch,
and it also settles the open question STATUS.md has carried since Step 2 about grid cell 78.

    python dataset_report.py                 # prints the report and writes outputs/dataset_report.md

Where the numbers come from, and why each is reported:

* Rows kept and dropped are reconstructed by re-running the prep module's own pipeline stages on the
  same inputs, so the report describes the artefacts that actually exist rather than a re-derivation:
  the raw file -> rows carrying a finite pm25_target -> rows that survive the gap-aware lag cleaning
  (the drop is the price of refusing to reuse a stale reading across a timestamp gap).
* Sensors outside the raster are counted by repeating the same bounding-box join the prep module uses.
  In the pilot all 18 sensors fall inside; the count exists so that extending the raster (see
  FILE_GUIDE.md) does not silently drop sensors.
* Grid cell 78 holds TWO sensors (3370151, 6303934) reporting at the same hours. Their rows are
  duplicated cell-hours, so pooled metrics weight that cell twice. The report measures the effect and
  states it; it does not silently aggregate, because aggregating would change every published number.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import prep_n_proc_pilot as prep
import spatial_cv

OUT_DIR = 'outputs'
MAPPING_CSV = 'sensor_to_grid_map_full.csv'
MASTER_CSV = 'Master_DL_Ready.csv'
GRID_CSV = 'full_morphology_grid.csv'


def _partition_of(grid_ids, partitions):
    return {g: name for name, cells in partitions.items() for g in cells}


def dataset_report(out_dir=OUT_DIR, write=True):
    """Return the report as a dict, and optionally write outputs/dataset_report.md."""
    df = prep.df_clean
    doc = spatial_cv.load_fold_map('fold_partitions.json')
    partitions = doc['partitions']
    part_of = _partition_of(None, partitions)

    # ---------------------------------------------------------------- rows kept and dropped
    raw = pd.read_csv(MASTER_CSV)
    raw_rows = len(raw)
    finite_target = int(raw['pm25_target'].notna().sum())

    # The gap-aware lag columns are built in memory by the prep module, not present in the raw CSV, so
    # the drop is measured by re-running the same two steps (compute the lags, then drop rows whose
    # window crossed a gap) rather than by reading a column that does not exist.
    lag_cols = ['pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3', 'pm25_lag_4',
                'pm25_roll24_mean', 'pm25_roll72_mean']
    base = raw.rename(columns={'location_id': 'Location', 'pm25_target': 'Value'})
    base['DateTime'] = pd.to_datetime(base['local_timestamp'])
    lag_feats = prep._time_aware_lags_and_rolling(base[['Location', 'DateTime', 'Value']])
    merged = base[['Location', 'DateTime', 'Value']].merge(
        lag_feats, on=['Location', 'DateTime'], how='left')
    lag_ok = int(merged[lag_cols + ['Value']].notna().all(axis=1).sum())
    rows_with_value = int(merged['Value'].notna().sum())
    kept = len(df)
    dropped_by_gap = rows_with_value - kept

    # Which requirement removes the rows? The lag filter is a conjunction, so measure each stage
    # separately; a 72 h rolling window with a 54 h minimum is by far the most demanding condition and
    # most of the loss sits there rather than in the 1-4 h lags.
    funnel = []
    for label, cols in [('with a target value', ['Value']),
                        ('+ 1 h lag', ['Value', 'pm25_lag_1']),
                        ('+ 2 h lag', ['Value', 'pm25_lag_1', 'pm25_lag_2']),
                        ('+ 3 h lag', ['Value', 'pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3']),
                        ('+ 4 h lag', ['Value', 'pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3', 'pm25_lag_4']),
                        ('+ 24 h rolling mean', ['Value', 'pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3',
                                                 'pm25_lag_4', 'pm25_roll24_mean']),
                        ('+ 24 h rolling sd', ['Value', 'pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3',
                                               'pm25_lag_4', 'pm25_roll24_mean', 'pm25_roll24_std']),
                        ('+ 72 h rolling mean', ['Value', 'pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3',
                                                 'pm25_lag_4', 'pm25_roll24_mean', 'pm25_roll24_std',
                                                 'pm25_roll72_mean']),
                        ('+ 72 h rolling sd (final keep)',
                         ['Value', 'pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3', 'pm25_lag_4',
                          'pm25_roll24_mean', 'pm25_roll24_std', 'pm25_roll72_mean',
                          'pm25_roll72_std'])]:
        funnel.append({'stage': label,
                       'rows': int(merged[cols].notna().all(axis=1).sum())})
    funnel_table = pd.DataFrame(funnel)
    funnel_table['lost_vs_previous'] = (-funnel_table['rows'].diff().fillna(0)).astype(int)

    # ---------------------------------------------------------------- sensors and cells
    mapping = pd.read_csv(MAPPING_CSV)
    sensors_total = int(raw['location_id'].nunique())
    sensors_mapped = int(mapping['sensor_id'].nunique())
    cells_total = int(mapping['id'].nunique())
    grid = pd.read_csv(GRID_CSV)
    sensors_outside = max(0, sensors_total - sensors_mapped)

    # ---------------------------------------------------------------- grid cell 78 (two sensors)
    per_cell_sensors = (mapping.groupby('id')['sensor_id'].nunique().sort_values(ascending=False))
    multi = per_cell_sensors[per_cell_sensors > 1]
    dup = df[df.duplicated(['Grid_ID', 'DateTime'], keep=False)]
    dup_cells = sorted(int(c) for c in dup['Grid_ID'].unique())
    dup_keys = int(df[['Grid_ID', 'DateTime']].duplicated().sum())
    affected = int(df['Grid_ID'].isin(dup_cells).sum())
    shares = (dup.groupby('Grid_ID').size() / len(df)).to_dict()

    # ---------------------------------------------------------------- partitions
    rows = []
    for name in ('train', 'calibration', 'test'):
        cells = partitions[name]
        sub = df[df['Grid_ID'].isin(cells)]
        rows.append({
            'partition': name, 'cells': len(cells), 'rows': len(sub),
            'cell_hours': int(len(sub[['Grid_ID', 'DateTime']].drop_duplicates())),
            'sensors': int(sub['Location'].nunique()),
            'value_mean': float(sub['Value'].mean()), 'value_sd': float(sub['Value'].std()),
            'value_min': float(sub['Value'].min()), 'value_max': float(sub['Value'].max()),
        })
    part_table = pd.DataFrame(rows)

    # ---------------------------------------------------------------- time coverage
    span_by_sensor = df.groupby('Location')['DateTime'].agg(['min', 'max', 'size'])
    span_hours = (span_by_sensor['max'] - span_by_sensor['min']).dt.total_seconds() / 3600.0 + 1
    completeness = (span_by_sensor['size'] / span_hours)

    # ---------------------------------------------------------------- missingness in the raw file
    miss = []
    for col in ['pm25_target', 'pm25_t-1', 'pm25_t-2', 'pm25_t-3']:
        if col in raw.columns:
            miss.append({'column': col, 'missing': int(raw[col].isna().sum()),
                         'missing_pct': float(raw[col].isna().mean() * 100)})
    miss_table = pd.DataFrame(miss)

    # ---------------------------------------------------------------- feature distributions
    feats = []
    for col in prep.features:
        s = df[col]
        feats.append({'feature': col, 'mean': float(s.mean()), 'sd': float(s.std()),
                      'min': float(s.min()), 'p25': float(s.quantile(.25)),
                      'median': float(s.median()), 'p75': float(s.quantile(.75)),
                      'max': float(s.max()), 'missing': int(s.isna().sum())})
    feat_table = pd.DataFrame(feats)

    report = {
        'raw_rows': raw_rows, 'finite_target_rows': finite_target,
        'rows_with_value_after_merge': rows_with_value,
        'rows_passing_lag_filter': lag_ok,
        'rows_kept': kept, 'dropped_missing_target': raw_rows - finite_target,
        'dropped_by_gap_aware_lag_cleaning': dropped_by_gap,
        'sensors_total': sensors_total, 'sensors_mapped_into_raster': sensors_mapped,
        'sensors_outside_raster': sensors_outside,
        'raster_cells': int(len(grid)), 'cells_with_sensor': cells_total,
        'duplicate_cell_hours': dup_keys, 'duplicate_cells': dup_cells,
        'rows_in_duplicated_cells': affected, 'share_of_rows_in_duplicated_cells': affected / kept,
        'date_min': str(df['DateTime'].min()), 'date_max': str(df['DateTime'].max()),
        'span_days': (df['DateTime'].max() - df['DateTime'].min()).days,
        'sensor_completeness_median_pct': float(completeness.median() * 100),
        'sensor_completeness_min_pct': float(completeness.min() * 100),
        'partition_table': part_table, 'missingness_table': miss_table,
        'feature_table': feat_table, 'drop_funnel': funnel_table,
        'value_summary': {'mean': float(df['Value'].mean()), 'sd': float(df['Value'].std()),
                          'min': float(df['Value'].min()), 'median': float(df['Value'].median()),
                          'p90': float(df['Value'].quantile(.90)), 'max': float(df['Value'].max())},
    }
    if write:
        write_report(report, out_dir)
    return report


def _fmt_table(df, floatfmt='{:.4g}'):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(str(c) for c in cols) + ' |',
             '|' + '|'.join('---' for _ in cols) + '|']
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(float(v)))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def write_report(report, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    r = report
    dup_note = (f"grid cell **{r['duplicate_cells'][0]}** holds two sensors that report at the same "
                f"hours" if r['duplicate_cells'] else 'no cell holds two reporting sensors')
    md = f"""# Dataset report (Deliverable 1)

Generated by `dataset_report.py`. Plan Section 3.3: "Descriptive statistics (Deliverable 1): rows kept
and dropped, sensors, cells, date range, missingness, feature distributions."

## 1. Rows kept and dropped

| Stage | Rows | Share |
|---|---|---|
| Raw rows in `{MASTER_CSV}` | {r['raw_rows']:,} | 100% |
| With a finite `pm25_target` | {r['finite_target_rows']:,} | {100 * r['finite_target_rows'] / r['raw_rows']:.1f}% |
| **Kept** after gap-aware lag cleaning (`X`, `y`) | **{r['rows_kept']:,}** | **{100 * r['rows_kept'] / r['raw_rows']:.1f}%** |
| Dropped: no target value | {r['dropped_missing_target']:,} | {100 * r['dropped_missing_target'] / r['raw_rows']:.1f}% |
| Dropped: lag/rolling window crossed a timestamp gap | {r['dropped_by_gap_aware_lag_cleaning']:,} | {100 * r['dropped_by_gap_aware_lag_cleaning'] / r['raw_rows']:.1f}% |

The gap-aware drop is deliberate. `Master_DL_Ready.csv` carries its own `pm25_t-1/2/3` columns, which are
naive row-shifts on irregularly-timed data; during a gap they pair a reading with one from hours earlier.
This pipeline recomputes the lags on a per-sensor hourly index instead, so a lag that would cross a gap
becomes missing and the row is dropped. The cost is the row count above; the benefit is that no lag
feature is a mislabelled stale reading.

### 1.1 Which requirement removes the rows

The filter is a conjunction, so each requirement is applied in turn. Sensors report roughly 70% of the
hours in their own span, and the 72 h rolling window needs 54 of 72 consecutive hours, so most of the
loss is caused by the longest window rather than by the short lags:

{_fmt_table(r['drop_funnel'])}

This is the honest cost of the design choice. The alternative -- reusing the raw file's row-shifted lags
-- would keep far more rows but feed the model readings mislabelled by up to 29 hours. Whether to relax
the 72 h requirement (a shorter window, or an interpolation cap) is a modelling decision for the
write-up, not something this report should change.

## 2. Sensors and cells

| Quantity | Value |
|---|---|
| Sensors in the raw file | {r['sensors_total']} |
| Sensors matched into the raster | {r['sensors_mapped_into_raster']} |
| Sensors outside the raster (dropped) | {r['sensors_outside_raster']} |
| Raster cells in `{GRID_CSV}` | {r['raster_cells']} |
| Cells containing a sensor | {r['cells_with_sensor']} |

## 3. Grid cell {r['duplicate_cells'][0] if r['duplicate_cells'] else 'n/a'}: two sensors in one cell

{dup_note.capitalize()}. Consequences, measured rather than assumed:

| Quantity | Value |
|---|---|
| Duplicated cell-hours | {r['duplicate_cell_hours']:,} |
| Rows in cells with duplicates | {r['rows_in_duplicated_cells']:,} |
| Share of kept rows | {100 * r['share_of_rows_in_duplicated_cells']:.1f}% |

Every pooled metric in this branch therefore gives that cell roughly twice the weight of a
single-sensor cell. **No aggregation is applied**, because collapsing to one row per cell-hour would
change every number already reported. The spatial-lag feature does collapse the pair by mean, so a
two-sensor cell cannot double-count inside the lag. Deciding whether to aggregate belongs with the
write-up, not with this report.

## 4. Date range and temporal coverage

| Quantity | Value |
|---|---|
| First kept timestamp | {r['date_min']} |
| Last kept timestamp | {r['date_max']} |
| Span | {r['span_days']:,} days |
| Median per-sensor completeness vs its own span | {r['sensor_completeness_median_pct']:.1f}% |
| Minimum per-sensor completeness | {r['sensor_completeness_min_pct']:.1f}% |

## 5. Partitions

{_fmt_table(r['partition_table'])}

Note that the partitions are assigned by grid cell, so all rows of a cell stay together; the partition
sizes above are consequences of the 11 / 3 / 3 cell split in `fold_partitions.json`.

## 6. Missingness in the raw file

{_fmt_table(r['missingness_table'])}

The `pm25_t-1/2/3` columns are shown only to document what the pipeline replaces; they are not used as
predictors anywhere in this branch.

## 7. Target distribution (kept rows, PM2.5 ug/m3)

| mean | sd | min | median | p90 | max |
|---|---|---|---|---|---|
| {r['value_summary']['mean']:.2f} | {r['value_summary']['sd']:.2f} | {r['value_summary']['min']:.2f} | {r['value_summary']['median']:.2f} | {r['value_summary']['p90']:.2f} | {r['value_summary']['max']:.2f} |

## 8. Feature distributions (all kept rows)

{_fmt_table(r['feature_table'])}

All features are complete (no missing values) by construction: rows with a missing lag or rolling
window were dropped in step 1.

## 9. What this report does not cover

- **No ground-truth comparison.** That is Deliverable 6 (Step 6) and needs deployed-sensor data.
- **No per-sensor breakdown beyond completeness.** Available in `prep.df_clean` grouped by `Location`.
- **Feature distributions are over all kept rows**, so cells with more data dominate them. Per-partition
  distributions are in the partition table above.
"""
    path = os.path.join(out_dir, 'dataset_report.md')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(md)
    with open(os.path.join(out_dir, 'dataset_report.json'), 'w', encoding='utf-8') as fh:
        json.dump({k: v for k, v in r.items() if not isinstance(v, pd.DataFrame)}, fh,
                  indent=2, sort_keys=True, default=str)
    r['partition_table'].to_csv(os.path.join(out_dir, 'dataset_partitions.csv'), index=False)
    r['feature_table'].to_csv(os.path.join(out_dir, 'dataset_features.csv'), index=False)
    print(f'wrote {path}')
    return path


def main():
    r = dataset_report(write=True)
    print('\n=== summary ===')
    for k in ['raw_rows', 'finite_target_rows', 'rows_kept', 'dropped_by_gap_aware_lag_cleaning',
              'sensors_total', 'sensors_mapped_into_raster', 'sensors_outside_raster',
              'cells_with_sensor', 'duplicate_cell_hours', 'duplicate_cells',
              'rows_in_duplicated_cells', 'date_min', 'date_max']:
        print(f'  {k:38s} {r[k]}')
    print('\n  partitions:')
    print(r['partition_table'].to_string(index=False))


if __name__ == '__main__':
    main()

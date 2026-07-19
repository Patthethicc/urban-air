import pandas as pd
import numpy as np

# ADDITIONAL NOTES FOR FULL-DATASET RUN (18 sensors, hourly, ~2yr span)
# This mirrors prep_n_proc.py's process (same merge -> engineer -> lag -> clean -> X/y/groups
# pattern) but adapts it to Master_DL_Ready.csv, which is hourly and NOT the 4-reading/day
# cadence of the pilot subset. Two adjustments were required that the pilot script didn't need:
#
# 1) SPATIAL JOIN INSTEAD OF DIRECT LOOKUP: sensor_to_grid_map.csv only covers the 19 pilot
#    sensors. The 18 sensors here are matched to their containing cell in full_morphology_grid.csv
#    by (x_coordinate, y_coordinate) bounding-box lookup, then written out to
#    sensor_to_grid_map_full.csv in the SAME schema as sensor_to_grid_map.csv, so spatial_cv.py
#    and spatial_bias.py work completely unchanged.
#
# 2) TIME-AWARE LAGS INSTEAD OF ROW-SHIFTS: Master_DL_Ready.csv's own pm25_t-1/t-2/t-3 columns
#    are naive row-shifts on irregularly-spaced readings (~20% of rows have gaps of 2-29h, not 1h).
#    Verified directly: during a gap, pm25_t-1 silently equals whatever reading came before the
#    gap, not the reading exactly 1 hour prior -- the same mislabeling bug the team's own
#    Extra_Preprocessing_pipeline_FIXED.ipynb calls out and fixes for the CNN-LSTM branch. XGBoost's
#    pipeline had no equivalent fix. Here, lags/rolling stats are computed on a per-sensor,
#    gap-exposed hourly index (`.asfreq('h')`) so a lag legitimately becomes NaN (and later gets
#    dropped) whenever the true elapsed time doesn't match, instead of silently reusing a stale
#    reading.

MAPPING_CSV = 'sensor_to_grid_map_full.csv'


def build_sensor_to_grid_map(master_csv='Master_DL_Ready.csv',
                              grid_csv='full_morphology_grid.csv',
                              out_csv=MAPPING_CSV):
    """Spatial-join each unique sensor's (x, y) to its containing cell in full_morphology_grid.csv
    and write a sensor_to_grid_map.csv-compatible CSV (same columns, same rename targets
    sensor_id->Location / id->Grid_ID) so spatial_cv.py / spatial_bias.py need no changes."""
    master = pd.read_csv(master_csv)
    grid = pd.read_csv(grid_csv)

    sensors = master.groupby('location_id').agg(
        latitude=('latitude', 'first'), longitude=('longitude', 'first'),
        x_coordinate=('x_coordinate', 'first'), y_coordinate=('y_coordinate', 'first'),
    ).reset_index()

    rows = []
    for _, s in sensors.iterrows():
        cell = grid[(grid['left'] <= s['x_coordinate']) & (s['x_coordinate'] < grid['right']) &
                    (grid['bottom'] <= s['y_coordinate']) & (s['y_coordinate'] < grid['top'])]
        if len(cell) == 0:
            print(f"WARNING: sensor {s['location_id']} falls outside full_morphology_grid.csv -- dropped")
            continue
        cell = cell.iloc[0]
        rows.append({
            'sensor_id': int(s['location_id']), 'latitude': s['latitude'], 'longitude': s['longitude'],
            'pm25_value': None, 'source': 'master_dl_ready',
            'id': int(cell['id']), 'left': cell['left'], 'top': cell['top'],
            'right': cell['right'], 'bottom': cell['bottom'],
            'row_index': int(cell['row_index']), 'col_index': int(cell['col_index']),
            'bh_mean': cell['bh_mean'], 'bh_stdev': cell['bh_stdev'],
            'ndvi_mean': cell['ndvi_mean'], 'bldg_dens': cell['bldg_dens'],
            'aspect_ratio_mean': cell['aspect_ratio_mean'],
        })

    out = pd.DataFrame(rows)
    out.insert(0, 'fid', range(1, len(out) + 1))
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}: {len(out)} sensors -> {out['id'].nunique()} unique grid cells")
    return out


def _time_aware_lags_and_rolling(df_pm25):
    """Per-Location: expose true hourly gaps via .asfreq('h') before lagging/rolling, so a lag
    or rolling window that crosses a gap comes back as NaN (and is dropped later) instead of
    silently pairing a reading with one from several hours earlier."""
    frames = []
    for loc, g in df_pm25.groupby('Location'):
        g = g.sort_values('DateTime').drop_duplicates(subset='DateTime', keep='first')
        s = g.set_index('DateTime')['Value']
        hourly = s.asfreq('h')

        feat = pd.DataFrame({
            'pm25_lag_1': hourly.shift(1),
            'pm25_lag_2': hourly.shift(2),
            'pm25_lag_3': hourly.shift(3),
            'pm25_lag_4': hourly.shift(4),
            'pm25_roll24_mean': hourly.shift(1).rolling(24, min_periods=18).mean(),
            'pm25_roll24_std':  hourly.shift(1).rolling(24, min_periods=18).std(),
            'pm25_roll72_mean': hourly.shift(1).rolling(72, min_periods=54).mean(),
            'pm25_roll72_std':  hourly.shift(1).rolling(72, min_periods=54).std(),
        }).reset_index()
        feat['Location'] = loc
        # keep only timestamps that actually exist in the original (non-reindexed) data
        feat = feat.merge(g[['DateTime']], on='DateTime', how='inner')
        frames.append(feat)

    return pd.concat(frames, ignore_index=True)


print("Loading data...")
build_sensor_to_grid_map()

df_pm25 = pd.read_csv('Master_DL_Ready.csv')
df_pm25 = df_pm25.rename(columns={
    'location_id': 'Location', 'local_timestamp': 'DateTime', 'pm25_target': 'Value',
})
df_pm25['DateTime'] = pd.to_datetime(df_pm25['DateTime'])

df_mapping = pd.read_csv(MAPPING_CSV)
df_mapping = df_mapping.rename(columns={'sensor_id': 'Location', 'id': 'Grid_ID'})

dupe_locations = df_mapping.loc[df_mapping.duplicated('Location', keep=False), 'Location'].unique()
if len(dupe_locations):
    print(f"WARNING: duplicate Grid_ID mappings for: {list(dupe_locations)} -- keeping first occurrence only")
df_mapping = df_mapping.drop_duplicates(subset='Location', keep='first')

# Paper (Section 3.1.1) declares EIGHT input features: bh_mean, bh_stdev, aspect_ratio (H/W),
# bldg_dens, STREET WIDTH, ndvi_mean, and time (hour sin/cos) = 8 columns. street_width itself
# isn't a raw column in either mapping CSV -- only H/W (aspect_ratio_mean) and H (bh_mean) are --
# so it's derived here rather than left out, to actually match the paper's declared feature list
# instead of silently missing 1 of the 8.
df_mapping['street_width_mean'] = df_mapping['bh_mean'] / df_mapping['aspect_ratio_mean']

print("Merging morphology metrics...")
df_final = df_pm25.merge(
    df_mapping[['Location', 'Grid_ID', 'bh_mean', 'bh_stdev', 'bldg_dens',
                'ndvi_mean', 'aspect_ratio_mean', 'street_width_mean']],
    on='Location', how='left'
)

print("Engineering temporal features...")
df_final['Hour'] = df_final['DateTime'].dt.hour
df_final['hour_sin'] = np.sin(2 * np.pi * df_final['Hour'] / 24.0)
df_final['hour_cos'] = np.cos(2 * np.pi * df_final['Hour'] / 24.0)

df_final['DayOfWeek'] = df_final['DateTime'].dt.dayofweek
df_final['Month'] = df_final['DateTime'].dt.month
df_final['dayofweek_sin'] = np.sin(2 * np.pi * df_final['DayOfWeek'] / 7.0)
df_final['dayofweek_cos'] = np.cos(2 * np.pi * df_final['DayOfWeek'] / 7.0)
df_final['month_sin'] = np.sin(2 * np.pi * df_final['Month'] / 12.0)
df_final['month_cos'] = np.cos(2 * np.pi * df_final['Month'] / 12.0)

print("Computing time-aware lags & rolling stats (this replaces Master_DL_Ready.csv's own "
      "pm25_t-1/t-2/t-3, which are naive row-shifts -- see module docstring)...")
lag_feats = _time_aware_lags_and_rolling(df_final[['Location', 'DateTime', 'Value']])
df_final = df_final.merge(lag_feats, on=['Location', 'DateTime'], how='left')

df_final['pm25_diff_1_2'] = df_final['pm25_lag_1'] - df_final['pm25_lag_2']

# Morphology x time interactions (same pattern as prep_n_proc.py), plus aspect_ratio x hour:
# canyon trapping (Section 3.1.2) is a traffic-driven, i.e. diurnal, effect, so its interaction
# with hour is a natural extension of the interactions the pilot script already engineered --
# not a new raw input, just another view of two already-declared features.
df_final['hour_x_bldg_dens'] = df_final['hour_sin'] * df_final['bldg_dens']
df_final['hour_x_ndvi'] = df_final['hour_sin'] * df_final['ndvi_mean']
df_final['hour_x_aspect_ratio'] = df_final['hour_sin'] * df_final['aspect_ratio_mean']

# Intentionally NOT added: spatial-lag features (neighboring cells' morphology/recent PM).
# The paper (Section 3.1.1 / 4.2.2) scopes each reading's predictors to morphology sampled
# within that sensor's own buffer, and explicitly drops spatial identifiers to force the model
# to rely on morphology rather than memorized geography. Folding in neighbor-cell values would
# reintroduce implicit location information the paper's design deliberately excludes.

lag_cols = ['pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3', 'pm25_lag_4',
            'pm25_roll24_mean', 'pm25_roll72_mean']
df_clean = df_final.dropna(subset=lag_cols + ['Value']).reset_index(drop=True)

print("Preparing model training...")
# Log1p-transform the target: PM2.5 is right-skewed (a handful of very high readings), and this
# is a transform of y, not a new input feature, so it doesn't touch the paper's declared 8.
# Inverse with np.expm1() on predictions before computing RMSE/MAE/R2 in the original units.
y_raw = df_clean['Value'].values
y = np.log1p(y_raw)

features = [
    'bh_mean', 'bh_stdev', 'bldg_dens', 'ndvi_mean', 'aspect_ratio_mean', 'street_width_mean',
    'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
    'month_sin', 'month_cos',
    'pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3', 'pm25_lag_4',
    'pm25_diff_1_2',
    'pm25_roll24_mean', 'pm25_roll24_std', 'pm25_roll72_mean', 'pm25_roll72_std',
    'hour_x_bldg_dens', 'hour_x_ndvi', 'hour_x_aspect_ratio',
]
X = df_clean[features].values

groups = df_clean['Grid_ID'].values

print(f"Final matrix: X={X.shape}, y={y.shape}, unique groups={len(set(groups))}")

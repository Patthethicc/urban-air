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

# Intentionally NOT added here: spatial-lag features (neighboring cells' recent PM). The paper
# (Section 3.1.1 / 4.2.2) scopes each reading's predictors to morphology sampled within that
# sensor's own buffer, and explicitly drops spatial identifiers to force the model to rely on
# morphology rather than memorized geography. Folding neighbor-cell values into `features` below
# would reintroduce implicit location information the paper's design deliberately excludes, so no
# spatial lag is part of X at import time.
#
# STEP 2 ADDITION: the lag itself is now implemented in compute_spatial_lag() further down, because
# plan Section 3.1.5 requires it and Section 2.3 says it is what addresses the "held-out cells still
# use their own history" limitation. It is deliberately kept OUT of the import-time `features` list:
# it is fold-dependent (a fold may only build it from cells that fold is allowed to use), so it is
# computed per fold in the notebook and appended to X there, where the ablation can also switch it
# on and off. Run `python make_fold_partitions.py` to build the partitions and fold maps it needs.

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


# ---------------------------------------------------------------------------------------------
# STEP 2: spatial-lag feature (plan Section 3.1.5)
# ---------------------------------------------------------------------------------------------

def compute_spatial_lag(df_clean, allowed_grid_ids, target_grid_ids=None,
                        mapping_csv_path=MAPPING_CSV, k=4, log_target=True):
    """Inverse-distance weighted mean PM2.5 of the k nearest ALLOWED grid cells, per cell per hour.

    This is the neighbourhood predictor the paper requires (Section 3.1.5). It is built to be
    fold-aware rather than a fixed column:

    * `allowed_grid_ids` is the list of cells this fold is permitted to see, and `k` neighbours are
      drawn ONLY from it. The target cell's own grid cell is never one of them, because a spatial
      lag is meant to come from somewhere else (and because grid cell 78 holds two sensors, so
      "exclude self" alone would not stop a cell being served its own reading). Cells outside the
      allowed list are never used, so a held-out cell's lag comes only from training cells and never
      leaks that cell's own reading back into its features.
    * `target_grid_ids` chooses which cells to produce lag rows for, defaulting to every cell in
      `allowed_grid_ids`. It exists so a caller can lag a held-out partition from the training cells
      in one call: pass the FULL df_clean with `allowed_grid_ids=train_cells` and
      `target_grid_ids=test_cells`. Passing only the held-out subset as df_clean would leave the
      neighbour pool empty, which is why that mistake is not possible through this signature.
    * Readings are taken per (Grid_ID, DateTime), averaged within a cell first. Grid cell 78 has
      two sensors, so without that collapse it would count twice in the weighted mean.
    * The observation pool is every allowed cell present in df_clean, so a target cell is lagged
      from its neighbours' real readings. Only rows that survive the module's gap-aware cleaning
      are eligible, which keeps the lag's coverage consistent with the rows the model trains on.
    * `k` neighbours are chosen by projected centroid distance (EPSG:32651 metres) with weights
      1/d, so near neighbours dominate. Missing readings are simply absent from the weighted mean:
      the weights that remain are renormalised. Nothing is imputed and there is no fallback to the
      target cell's own history, which would defeat the point of the feature.

    Returns a DataFrame with Grid_ID, DateTime and three columns:
      pm25_spatial_lag    weighted mean, log1p-transformed to match the other pm25_* history
                          features and the log1p target (NaN when no neighbour has a reading)
      pm25_spatial_lag_w  summed inverse-distance weight actually used -> effective coverage
      pm25_spatial_lag_n  number of neighbouring cells that contributed -> coverage count

    The _w and _n columns matter because "no neighbour reported" and "neighbours reported low"
    would otherwise look identical to the model. They also let the notebook report the lag's real
    coverage instead of implying it is always available.
    """
    import spatial_cv  # local import: spatial_cv only needs pandas/numpy/sklearn

    allowed = {int(g) for g in allowed_grid_ids}
    targets = {int(g) for g in target_grid_ids} if target_grid_ids is not None else set(allowed)
    if not targets:
        raise ValueError("target_grid_ids is empty; nothing to compute a lag for")
    centroids = spatial_cv.get_grid_centroids(mapping_csv_path)
    grid_ids = [int(g) for g in centroids['Grid_ID']]
    coords = centroids[['cx', 'cy']].values

    # One PM2.5 reading per (Grid_ID, DateTime) for every candidate cell, i.e. the cells in the
    # subset that this fold is also allowed to use. The mean collapses cell 78's two sensors.
    pool = df_clean[df_clean['Grid_ID'].isin(allowed)]
    obs = pool.groupby(['Grid_ID', 'DateTime'])['Value'].mean().unstack('DateTime')
    hours = obs.columns

    rows = []
    for pos, raw_cell in enumerate(grid_ids):
        cell = int(raw_cell)  # pandas indexes these as Python ints, numpy ints do not match
        if cell not in targets:
            continue  # not a cell this call is producing lag rows for
        # NOTE: a target cell is usually NOT in obs.index, by design: a held-out partition's cells
        # are excluded from `allowed`, so their own readings are not in the observation pool. The
        # lag for such a cell is built purely from its allowed neighbours below.
        dist = np.sqrt(((coords - coords[pos]) ** 2).sum(axis=1))
        order = [j for j in np.argsort(dist) if grid_ids[j] in allowed and grid_ids[j] != cell]
        neighbours = order[:k]

        # A row exists in df_clean because THIS cell reported at that hour, so the lag is only
        # meaningful for the cell's own hours. Emitting a value for every hour in the pooled data
        # would hand back lag values for rows that do not exist and cannot be merged onto one.
        # own_hours is then clipped to hours the neighbour pool actually covers: an hour no allowed
        # cell reported at has no lag to compute, and it stays NaN once merged back.
        own_hours = pd.Index(df_clean.loc[df_clean['Grid_ID'] == cell, 'DateTime'].unique())
        own_hours = own_hours.intersection(hours)
        if len(own_hours) == 0:
            continue  # target cell has no cleaned rows overlapping the observation pool

        if not neighbours:
            rows.append(pd.DataFrame({
                'Grid_ID': cell, 'DateTime': own_hours,
                'pm25_spatial_lag': np.nan, 'pm25_spatial_lag_w': 0.0,
                'pm25_spatial_lag_n': 0,
            }))
            continue

        neighbour_weights = 1.0 / dist[neighbours]
        nb_values = obs.loc[[grid_ids[j] for j in neighbours], own_hours].values
        present = ~np.isnan(nb_values)
        # NOTE: this local MUST NOT be called `weights`. Inside the pd.DataFrame(...) call below,
        # `weights` resolves to pandas' DataFrame.weights accessor rather than this array, which
        # silently wrote 1.0 into the denominator column and made every lag value collapse to 1.
        weighted = np.where(present, nb_values, 0.0) * np.where(present, neighbour_weights[:, None],
                                                                 0.0)

        # Denominator and numerator are computed from the same masked matrix, so the weighted mean
        # is the true sum(w*x)/sum(w) over neighbours that actually reported.
        weight_sum = np.where(present, neighbour_weights[:, None], 0.0).sum(axis=0)
        value_sum = weighted.sum(axis=0)
        contributor_count = present.sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            lag = np.where(weight_sum > 0, value_sum / weight_sum, np.nan)

        rows.append(pd.DataFrame({
            'Grid_ID': cell,
            'DateTime': own_hours,
            'pm25_spatial_lag': np.log1p(lag) if log_target else lag,
            'pm25_spatial_lag_w': weight_sum,
            'pm25_spatial_lag_n': contributor_count.astype(int),
        }))

    return pd.concat(rows, ignore_index=True)


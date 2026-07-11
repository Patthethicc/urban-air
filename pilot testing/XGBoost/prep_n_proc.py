import pandas as pd
import numpy as np

# ADDITIONAL NOTES FOR PILOT TESTING SUBSET DATA
# 19 unique sensors map to only 16 unique Grid_IDs
# aspect ratio mean is the variable for building height/street width

print("Loading data...")

# time-series PM2.5 data
df_pm25 = pd.read_csv('merged_small_sensors_timeseries.csv')
df_pm25 = df_pm25.rename(columns={
    'sensor_id': 'Location',
    'timestamp': 'DateTime',
    'pm25_value': 'Value',
})
df_pm25['DateTime'] = pd.to_datetime(df_pm25['DateTime'])

# grid mapping to sensor_to_grid_map.csv for pilot testing
# (has pm2.5 ground truty, and subset of bigger data)
df_mapping = pd.read_csv('sensor_to_grid_map.csv')
df_mapping = df_mapping.rename(columns={'sensor_id': 'Location', 'id': 'Grid_ID'})

# keeps first occurrence of each location (Grid_ID is already unique),
# to avoid duplicate readings from the same sensor being counted multiple times
dupe_locations = df_mapping.loc[df_mapping.duplicated('Location', keep=False), 'Location'].unique()
if len(dupe_locations):
    print(f"WARNING: duplicate Grid_ID mappings for: {list(dupe_locations)} -- keeping first occurrence only")
df_mapping = df_mapping.drop_duplicates(subset='Location', keep='first')


# attach Grid_ID and the morphology metrics to every timestamped pm2.5 reading
df_final = df_pm25.merge(
    df_mapping[['Location', 'Grid_ID', 'bh_mean', 'bh_stdev', 'bldg_dens',
                'ndvi_mean', 'aspect_ratio_mean']],
    on='Location', how='left'
)

print("Engineering temporal features...")
# Cyclical Time Encoding
df_final['Hour'] = df_final['DateTime'].dt.hour
df_final['hour_sin'] = np.sin(2 * np.pi * df_final['Hour'] / 24.0)
df_final['hour_cos'] = np.cos(2 * np.pi * df_final['Hour'] / 24.0)

# sort strictly by location and time during lag creation
# create Lags (previous pm2.5 readings) grouped by physical sensor
df_final = df_final.sort_values(by=['Location', 'DateTime']).reset_index(drop=True)
df_final['pm25_lag_1'] = df_final.groupby('Location')['Value'].shift(1)
df_final['pm25_lag_2'] = df_final.groupby('Location')['Value'].shift(2)

# drop first few rows with NaN lag values (before lag creation)
df_clean = df_final.dropna(subset=['pm25_lag_1', 'pm25_lag_2', 'Value']).reset_index(drop=True)

print("Preparing model training...")
y = df_clean['Value'].values

#training features
features = [
    'bh_mean', 'bh_stdev', 'bldg_dens', 'ndvi_mean', 'aspect_ratio_mean',
    'hour_sin', 'hour_cos', 'pm25_lag_1', 'pm25_lag_2',
]
X = df_clean[features].values

# groups for spatial k fold use grid id since multiple sensors
# can share a grid cell
groups = df_clean['Grid_ID'].values

print(f"Final matrix: X={X.shape}, y={y.shape}, unique groups={len(set(groups))}")

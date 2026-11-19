# Exploratory Data Analysis (EDA) Report: Metro Manila PM2.5 Sensor Network

## 1. Executive Dataset Summary
- **Total Combined Hourly Records Loaded:** 1,363,448 (1,363,299 valid numeric PM2.5 readings)
- **Total Unique Sensor Stations:** 85 (40 in `QC Data Converted`, 45 in `OpenAQ`)
- **Overall Temporal Span:** `2021-07-09 00:00:00` to `2026-10-15 18:00:00` (Asia/Manila PHT)
- **Global Mean PM2.5 (All Data):** 19.10 µg/m³
- **Global Mean PM2.5 (Excluding NYE Fireworks Window):** 18.99 µg/m³
- **Peak New Year's Eve Hourly Spike:** 553.00 µg/m³

## 2. Key Data Quality Findings & Pre-Modeling Recommendations
1. **New Year's Eve Fireworks Extreme Outliers:**
   - Across both `QC Data Converted` and `QC Data` (particularly Jan 1, 2024 00:00–03:00 PHT and Jan 1, 2025 00:00–04:00 PHT), stations record extreme spikes between **300 and 529.06 µg/m³** (e.g., Pearl Drive Footbridge, Jose Rizal High School, Old Balara Elementary School).
   - *Recommendation:* Mask or winsorize `[Dec 31 21:00 – Jan 1 06:00]` before training CNN-LSTM or XGBoost models so firework anomalies do not dominate MSE/RMSE loss.
2. **Zero and Near-Zero Readings in OpenAQ:**
   - Certain OpenAQ nodes (such as `Valenzuela` location `5975180` and `Las Pinas` location `5976206`) contain stretches of exact `0.0` or `< 0.1` µg/m³ readings indicative of sensor floor/calibration dropouts.
   - *Recommendation:* Filter out `pm25_value <= 0.5` µg/m³ prior to log-transformation or feature lag generation.
3. **Staggered Sensor Deployment Cohorts (Quezon City):**
   - Cohort 1 (8 sensors, IDs `9000001`–`9000008`): Deployed July–October 2021.
   - Cohort 2 (12 sensors, IDs `9000009`–`9000020`): Deployed July–November 2022.
   - Cohort 3 (20 sensors, IDs `9000021`–`9000040`): Deployed November 2023–March 2024.
   - *Recommendation:* When constructing spatiotemporal graphs or cross-station lag features, group by deployment cohort or restrict joint multi-station training windows to **April 2024 – 2025** where all 40 QC sensors operate concurrently.
4. **Raw Wide-Format Formatting Quirks (`QC Data/`):**
   - `AQ_Sensors_Hourly_2025.csv` has a trailing comma on every line, producing an empty `Unnamed: 41` column.
   - `AQ_Sensors_Hourly_2022.csv` and `2023.csv` split Commonwealth into `Commonwealth_BH` and `Commonwealth_BHall`.
   - `QCDATA_SensorLoc.csv` has trailing spaces in column headers (`'ID '` and `'Site name '`).

## 3. Top 5 Highest Mean PM2.5 Stations
| source            | location_name                 | classification   |   mean_pm25 |   p95_pm25 |   completeness_pct |
|:------------------|:------------------------------|:-----------------|------------:|-----------:|-------------------:|
| OpenAQ            | Malabon Action Center         | OpenAQ Node      |       31.44 |      67.97 |              69.1  |
| OpenAQ            | Valenzuela                    | OpenAQ Node      |       30    |      58.8  |              76.07 |
| QC Data Converted | Cubao Arayat Market           | Roadside         |       29.45 |      65.16 |              99.97 |
| OpenAQ            | Monumento, Caloocan           | OpenAQ Node      |       27.16 |      47.8  |              78.85 |
| OpenAQ            | Valenzuela CENRO (ARCA North) | OpenAQ Node      |       26.6  |      51.9  |              68.34 |

## 4. Top 5 Longest-Running Stations by Valid Hours
| source            | location_name                      | start_date       | end_date         |   valid_hours |   completeness_pct |
|:------------------|:-----------------------------------|:-----------------|:-----------------|--------------:|-------------------:|
| QC Data Converted | Sto. Domingo Church                | 2021-07-09 00:00 | 2025-12-31 23:00 |         39260 |              99.93 |
| QC Data Converted | Quezon City General Hospital       | 2021-07-10 00:00 | 2025-12-31 23:00 |         39240 |              99.94 |
| QC Data Converted | Cubao Arayat Market                | 2021-07-17 00:00 | 2025-12-31 23:00 |         39084 |              99.97 |
| QC Data Converted | Obrero corner Calle Industria      | 2021-07-17 00:00 | 2025-12-31 23:00 |         38828 |              99.31 |
| QC Data Converted | Quirino Highway cor. Mindanao Ave. | 2021-07-17 00:00 | 2025-12-31 23:00 |         38726 |              99.05 |

## 5. Raw Wide-Format Yearly File Audit (`QC Data/`)
| file_name                  |   rows_hours |   active_station_columns |   dropped_trailing_unnamed_cols | merged_commonwealth_split   | start_timestamp     | end_timestamp       |   non_null_measurements |   overall_cell_completeness_pct |   mean_pm25 |   max_pm25 |
|:---------------------------|-------------:|-------------------------:|--------------------------------:|:----------------------------|:--------------------|:--------------------|------------------------:|--------------------------------:|------------:|-----------:|
| AQ_Sensors_Hourly_2021.csv |         4224 |                        8 |                               0 | False                       | 2021-07-09 00:00:00 | 2021-12-31 23:00:00 |                   29149 |                           86.26 |       20.28 |     171.66 |
| AQ_Sensors_Hourly_2022.csv |         8760 |                       20 |                               0 | True                        | 2022-01-01 00:00:00 | 2022-12-31 23:00:00 |                  110487 |                           63.06 |       19.83 |     712.93 |
| AQ_Sensors_Hourly_2023.csv |         8760 |                       20 |                               0 | True                        | 2023-01-01 00:00:00 | 2023-12-31 23:00:00 |                  173791 |                           99.2  |       20.4  |     283.65 |
| AQ_Sensors_Hourly_2024.csv |         8784 |                       40 |                               0 | False                       | 2024-01-01 00:00:00 | 2024-12-31 23:00:00 |                  349436 |                           99.45 |       20.11 |    1707.85 |
| AQ_Sensors_Hourly_2025.csv |         8761 |                       40 |                               1 | False                       | 2025-01-01 00:00:00 | 2025-12-31 23:00:00 |                  348391 |                           99.42 |       17.7  |    1287.98 |

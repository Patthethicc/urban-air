# Dual-branch CNN-LSTM PM2.5 model

Predicts PM2.5 (µg/m³) at monitoring stations **and at every cell of a city-wide raster**. A CNN reads a
320 m × 320 m patch of urban-morphology rasters (building height, density, street width, aspect ratio, σH,
NDVI); an LSTM reads the last hours of PM2.5 at the place plus a spatial lag from neighbouring stations. Every
prediction gets a geographically weighted conformal interval (GeoCP). Everything is evaluated on **whole
stations the model never saw**, never on random rows.

What you get from one run:

- a trained model with its full settings stored next to it, and a one-row-per-run comparison table;
- test metrics (RMSE, MAE, R²) and GeoCP interval quality (coverage, widths, urban core vs. periphery);
- a spatial-bias check of the residuals (Moran's I, LISA clusters over the morphology bands);
- for any day: four snapshot surfaces (08:00, 12:00, 17:00, 21:00) with intervals, a 24-hour mean, and maps on
  the Philippine AQI scale; Grad-CAM saliency of the CNN on the hottest and coldest cells.

> **Status.** Runs on the real data in the repo's `data/` folder: a 10 m Master GeoTIFF (EPSG:32651, 5 bands,
> 27 × 39 km) and the OpenAQ sensors (44 stations with readings, 2024-08-13 to 2026-10-15). The pipeline runs end to
> end and 313 tests pass. The numbers quoted in this file come from short smoke runs (2 epochs), so they show that the
> wiring works, not what the model can reach. Three things to know before relying on results: the raster stops short of
> the southern tip of the NCR ([2.14](#214-covering-the-whole-ncr)), a full-resolution city map takes hours on a CPU
> ([2.6](#26-make-the-maps)), and the pipeline has not been trained to convergence on this data yet.

## Contents

1. [Quick start](#1-quick-start)
2. [Tutorial: running it](#2-tutorial-running-it)
3. [How it works](#3-how-it-works)
4. [Outputs reference](#4-outputs-reference)
5. [Known limits](#5-known-limits)
6. [How it was checked](#6-how-it-was-checked)
7. [Which thesis text became which code](#7-which-thesis-text-became-which-code)

---

## 1. Quick start

```bash
cd model
pip install -r requirements.txt
python -m pytest tests -q
python run_pipeline.py --stage coverage
python run_pipeline.py --stage all --override train.epochs=2 --override train.batch_size=256
python run_pipeline.py --stage infer --date 2026-04-11 --override inference.stride_px=21
```

`coverage` checks in a second that the raster covers your sensors. The `all` line runs the data through
preprocessing, training, GeoCP, the bias check and the report in about 10 minutes on a CPU (2 epochs; the defaults
train much longer). The `infer` line maps the whole raster at a coarse 210 m grid in about 2 minutes; the full 10 m
grid takes hours (see [2.6](#26-make-the-maps)). Results are in `snapshots/<run_id>/` and the maps in
`snapshots/<run_id>/surface/20260411/maps/`.

---

## 2. Tutorial: running it

### 2.1 What you need

- **Python 3.10+** (developed and tested on 3.14.5, Windows 11, CPU-only PyTorch).
- The packages in `requirements.txt` (names only). `requirements-tested.txt` has the exact versions the code
  was tested with, if you need to reproduce results. For a GPU, install PyTorch first from pytorch.org, then
  `pip install -r requirements.txt`; the code uses CUDA automatically (`train.device: auto`).
- Optional: `contextily` for web-tile basemaps on the maps (see [2.6](#26-make-the-maps)).
- **Run every command from inside the `model/` folder.**

The data in the repo's `data/` folder is already referenced by `configs/data.yaml`, with paths relative to `model/`:

| What | Where |
|---|---|
| Hourly PM2.5 per sensor | `../data/OpenAQ/openaq_<id>_*.csv` (45 files, 44 stations have readings) |
| Sensor coordinates | `../data/OpenAQ/OpenAQ Sensor Locations - OpenAQ Sensor Locations.csv` (46 sensors) |
| Master GeoTIFF | `../data/Morphology/master_raster_v2.tif`: 10 m, EPSG:32651, 3897 × 2729 pixels, 5 bands: `building_height_mean`, `building_height_stddev` (σH), `ndvi_mean`, `building_density`, `street_aspect_ratio_mean` |
| Quezon City network (optional) | `../data/QC Data/`: one wide table per year, converted to per-sensor files by `qc_convert.py`, see [2.1b](#21b-the-quezon-city-sensors-qc-data) |

45 of the 46 sensors fall inside the raster. The Tunasan station (Muntinlupa) lies 2.3 km beyond its southern edge
and is skipped, and sensor 4797655 has no data file. `python run_pipeline.py --stage coverage` shows this at any
time ([2.14](#214-covering-the-whole-ncr)). The older 500 m pilot files in `pilot testing/` are no longer referenced.

### 2.1b The Quezon City sensors (QC data)

`data/QC Data/` holds 40 low-cost sensors in Quezon City, hourly from July 2021 (`AQ_Sensors_Hourly_<year>.csv`, one
column per sensor, local time) and their locations (`QCDATA_SensorLoc.csv`). The loader reads one file per sensor, so
convert them once:

```bash
python preprocessing/qc_convert.py
```

This writes `data/QC Data Converted/`: one `qcdata_<location_id>_<name>.csv` per site with the OpenAQ columns
(`location_name, location_id, sensor_id, utc_timestamp, local_timestamp, pm25_value, parameter, unit`) and
`QC Sensor Locations.csv` (`name, location_id, latitude, longitude`). The loader accepts `qcdata_` files exactly like
`openaq_` files. The raw files are never changed.

- **Ids:** `location_id` is `9000000` + the site number (1-40), so it cannot collide with an OpenAQ id; `sensor_id` repeats it
  (the network has no sensor ids).
- **Which column is which site:** the hourly columns are short codes (`Nova_Market`, `QCGH`, ...) and the site table has
  full names, so `configs/qc_column_map.csv` pairs them (each pairing is by name; for 18 of the 20 sites installed up to Nov 2022
  the first reading is one day after the listed installation date, which agrees; sites 3 and 4 report earlier). `Commonwealth_BH` and
  `Commonwealth_BHall` are the same sensor under two column names: they become one series, `Commonwealth_BHall` first
  and `Commonwealth_BH` filling the hours it is empty (2685 hours report in both, mean difference 6.7 µg/m³).
- **Time:** the QC timestamps have no time zone; they are treated as Asia/Manila (`--timezone` changes it).
- **Values** are not altered (only empty hours and the table's `AVERAGE` row are dropped). The network has 174 negative
  readings and 176 above 1000 µg/m³, which `data.clean` (0-1000) turns into gaps when loading.
- **Check this before training:** the 2024 sites (ids 21-40) report from 2024-01-01, up to 85 days before their listed
  installation dates. In that stretch they follow the median of the other sites at r = 0.97 (0.77 afterwards), more closely
  than independent sensors would, so those hours look backfilled and could flatter the spatial lag. `python
  preprocessing/qc_convert.py --trim-before-install` drops everything before the listed date (off by default).

To use the QC sensors, point `configs/data.yaml` at them, alone or together with OpenAQ (each of the two sensor paths may
be a list):

```yaml
sensor_locations_csv: [../data/OpenAQ/OpenAQ Sensor Locations - OpenAQ Sensor Locations.csv, ../data/QC Data Converted/QC Sensor Locations.csv]
sensor_csv_dir: [../data/OpenAQ, ../data/QC Data Converted]
```

All 40 sites lie inside the raster, so `--stage coverage` reports them as covered. They are all in Quezon City and the
stations are randomly split, so adding them makes the sensor network much denser there than elsewhere in the NCR.

### 2.2 Check the install

```bash
python -m pytest tests -q
```

Expect `313 passed` (and 4 skipped, which only apply when `contextily` is not installed) in about a minute. If something fails here, fix it before running anything else.

### 2.3 First run: the whole pipeline on the data folder

```bash
python run_pipeline.py --stage all --override train.epochs=2 --override train.batch_size=256
```

`--override section.key=value` changes one setting for this run only; here it makes training short. Without
the overrides the defaults apply (up to 50 epochs, batch 64), which takes many times longer on a CPU.

What you should see (abridged; numbers differ a little between runs):

```
Master GeoTIFF: 27.3 x 39.0 km at 10 m, EPSG:32651, lon 120.900..121.150 E, lat 14.398..14.752 N
Sensors inside the raster: 45 of 46
  OUTSIDE (skipped): Barangay Tunasan - Maharlika Highway (SM Center Muntinlupa) (4797656) ...: 2.3 km beyond the south edge
COVERAGE: INCOMPLETE: part of the study area is cropped away
stations per partition: {'train': 28, 'calibration': 7, 'test': 9}
windows: X_temporal (463848, 3, 4), y (463848, 1)
[fold 0] held-out stations RMSE 5.8  MAE 3.5
final fit on all training stations for 1 epochs
run 20261015_221114 saved to checkpoints/20261015_221114
GeoCP config: {'kernel': 'gaussian', 'bandwidth_m': inf, 'alpha': 0.1, ...}
test coverage 0.872 (target 0.90), mean width 13.36 ug/m3, median 13.67
spatial bias: Moran's I 0.220 (p 0.054, 28 cells); LISA not run
snapshot written to snapshots/20261015_221114
```

That took about 10 minutes on the machine used for this tutorial (preprocessing 10 s, training about 9 minutes, the
rest under a minute). Three folders were created, one per kind of output:

```
processed/                 the arrays the model trains on (patches, windows, scalers, partition)
checkpoints/<run_id>/      the model, the settings it used, per-fold models, learning curves
snapshots/<run_id>/        SUMMARY.md (read first), plots/, tables/, data/ (accuracy, intervals, bias analysis)
```

### 2.4 The same thing, one stage at a time

Run these to redo or inspect a single step. Each stage reads what the previous one wrote.

| Stage | Command | What it does | Main output |
|---|---|---|---|
| preprocess | `--stage preprocess` | clean hourly series, split stations into train / calibration / test, spatial lag, windows, patches, scaling | `processed/` |
| train | `--stage train` | spatial cross-validation on the training stations, then the final fit | `checkpoints/<run_id>/` |
| geocp | `--stage geocp` | predicts calibration and test stations, picks the GeoCP setting, builds intervals | `data/predictions_*.csv`, `data/geocp.json`, `data/calibration_table.csv` |
| bias | `--stage bias` | Moran's I and LISA on the out-of-fold residuals | `data/spatial_bias.json`, `plots/lisa_*.png` |
| report | `--stage report` | accuracy report and plots; prints the headline (runs `geocp` and `bias` first if missing) | `SUMMARY.md`, `metrics_summary.json`, `plots/`, `tables/` |
| infer | `--stage infer --date D` | a prediction and interval for every raster cell, four snapshots a day (runs `geocp` first if it has not been run) | `surface/<D>/` |
| explain | `--stage explain --timestamp "D HH:MM"` | Grad-CAM on the hottest and coldest cells | `explain/<D_HHMM>/` |
| tune | `--stage tune` | Bayesian hyperparameter search with nested spatial CV (slow) | `tuning/<id>/` |
| rasters | `--stage rasters` | builds the Master GeoTIFF from the source layers (one-off) | `processed/rasters/` |
| clean | `--stage clean` (alias `reset`) | **reset**: lists the generated outputs it would delete; add `--yes` to delete them. See [2.13](#213-start-over-reset) | |

Stages that act on a trained run take `--run-id <id>`; without it they use the latest run.

### 2.5 Reading the results

When `train` ends it prints a **training summary**: the data sizes, the model, and the cross-validation accuracy fold by
fold (RMSE, MAE, R², with mean ± sd). That is validation accuracy. The accuracy on the stations the model never saw comes
from `report` (also the last step of `--stage all`), which prints a short headline and writes **`snapshots/<run_id>/SUMMARY.md`**.
Open that file first; it is organised for reading, not for parsing:

| Section | Answers |
|---|---|
| 1. Headline | R², RMSE, MAE, bias, share of predictions within ±5 and ±10 µg/m³, share in the right AQI band, and the interval coverage, each with a plain-language meaning |
| 2. Is that good? | the model next to two simple rules: repeat the previous hour (persistence) and always predict the average level, with the skill against persistence |
| 3. Does it hold across the splits? | cross-validation, calibration and test side by side |
| 4. By station | every test station, largest error first, with its interval coverage (and `plots/accuracy_by_station.png`) |
| 5. Uncertainty | which GeoCP weighting was chosen, coverage against target, interval width, close to versus far from a calibration sensor |
| 6. Spatial bias | Moran's I of the residuals in one sentence |
| 7. Training | the folds and the final fit |
| 8. Read with care | automatic warnings: a short run, few test stations, intervals under target, no better than persistence, unverified AQI limits |
| 9. Files | what every file in the folder is |

How to read the accuracy numbers:

- **R², RMSE, MAE** are the three metrics of Validation.txt, on test stations never used for fitting, tuning or calibration.
  RMSE punishes large misses; MAE is the average error; R² is the share of variation explained (0 = no better than the
  average level).
- **Bias** is the mean of predicted − observed: negative means the model under-predicts on average.
- **Within ±X µg/m³** and **same AQI band** are the plain-language versions. The AQI figure must be read against the
  *guess-the-most-common-band* score printed beside it, because most hours are in the lowest band.
- **Persistence** repeats the previous hour at the same station. It is only possible where a station has history, which
  the model does not need, so it is a yardstick, not a rival.

The folder:

```
snapshots/<run_id>/
    SUMMARY.md             read this first
    metrics_summary.json   every number, machine-readable (old keys kept; the new ones are under "accuracy")
    run_info.json          settings and provenance
    plots/                 loss_curve  pred_vs_obs  residual_hist  timeseries_sample  accuracy_by_station  geocp_coverage  lisa_map ...
    tables/                accuracy_summary.csv  accuracy_by_station.csv  cv_folds.csv     (open in a spreadsheet)
    data/                  predictions_test.csv  predictions_calibration.csv  calibration_table.csv  geocp.json  spatial_bias.json ...
    surface/  explain/     from `infer` and `explain`
```

Snapshots made before this layout kept every file in the top folder; they still work (re-run `--stage report` to get
`SUMMARY.md` for them). Settings: `report.accuracy` in `configs/report.yaml` (the tolerances, the AQI-band share, the
persistence yardstick).

`metrics_summary.json` in detail:

- **`test`**: RMSE and MAE in µg/m³ and R², on stations that were never used for fitting, tuning or
  calibration. Example from a 2-epoch run: RMSE 5.75, MAE 3.43, R² 0.75 on 91,157 rows from 9 unseen stations. `supplementary` holds Willmott's IA and
  Pearson r.
- **`cv_folds`**: the same metrics for each cross-validation fold of the training stations (held-out stations
  per fold). These should be in the same range as `test`; a big gap points at a lucky or unlucky split.
- **`accuracy`**: `measures` (the extended figures per split), `yardsticks` (persistence and the average-level rule) and
  `by_station`.
- **`geocp.test`**: `coverage` should be close to `target_coverage` (0.90). In the same run it is 0.87,
  below target because only seven stations calibrate the intervals. `mean_width` / `median_width` are in µg/m³;
  `coverage_urban_core` and `coverage_peripheral` split the test stations by distance to the nearest
  calibration sensor.
- **`spatial_bias`**: `global_moran.p_sim` under 0.05 means residuals cluster in space; then
  `plots/lisa_on_bands.png` shows where, laid over each morphology band. With few stations the test has little power.
- **`parameters`** and `run_info.json` (next to the model) list every setting used, so a run can always be
  reproduced. `checkpoints/runs_index.csv` compares all runs in one table.

### 2.6 Make the maps

```bash
python run_pipeline.py --stage infer --date 2026-04-11
```

`infer` needs the GeoCP calibration of the run; if it has not been computed yet (for example right after `train`), `infer` runs `--stage geocp` first and says so. It also needs hourly readings for the three hours before each snapshot. The data covers 2024-08-13 12:00 to
2026-10-15 18:00 (44 stations). To check the range of your own data:

```bash
python -c "import numpy as np, pandas as pd; z=np.load('processed/pm25_hourly.npz'); s=pd.Timestamp(str(z['start'])); print(s, s+pd.Timedelta(hours=len(z['values'])-1))"
```

**How long a map takes.** `infer` predicts every cell of the raster: 10.6 million cells at the full 10 m grid. On a
4-thread CPU the model does roughly 300-800 cells/s (it varies with load), so one full-resolution snapshot takes
**4 to 10 hours**. A day of four snapshots costs much less than four times that: every chunk of cells computes its CNN
embedding once and reuses it for all hours of the date (measured on a 202,500-cell window of the real raster: 315 s instead
of 1,076 s for four hours, 3.4x faster, identical output), so a full-resolution day is roughly 4-5 hours on this CPU. The
run prints an estimate after the first chunk and progress lines afterwards, so you know before the long part starts. Choose the grid with `inference.stride_px` (odd values):

| `--override inference.stride_px=` | Grid | Cells | About, per snapshot |
|---|---|---|---|
| `21` | 210 m | 24 thousand | 1 minute |
| `11` | 110 m | 88 thousand | 2-5 minutes |
| `5` | 50 m | 425 thousand | 10-25 minutes |
| `1` (default) | 10 m | 10.6 million | 4-10 hours |

`inference.window: [row_start, row_stop, col_start, col_stop]` limits the map to a part of the raster. A GPU is much
faster (`train.device: auto` uses it), but the installed PyTorch is the CPU-only build, so the laptop's GPU is not used yet.
`inference.chunk_cells` (default 1024) bounds memory: the CNN needs about 1.2 GB at 1024 cells per chunk and 5.1 GB at 4096.

`--date` expands to the four hours in `configs/inference.yaml` (`snapshot_hours: [8, 12, 17, 21]`);
`--timestamp "2026-04-11 12:00"` (repeatable) gives explicit hours. Results go to
`snapshots/<run_id>/surface/20260411/`:

| File | Contents |
|---|---|
| `prediction_volume.tif` | one band per snapshot (band names are the hours) |
| `lower_volume.tif`, `upper_volume.tif`, `width_volume.tif` | the GeoCP interval, same grid |
| `prediction_daily_mean.tif` | 24-hour mean (a cell missing in any snapshot stays empty) |
| `maps/aqi_0800.png` … `daily_panels.png`, `daily_mean.png` | AQI-coloured maps (no outline; `output.outline_above_threshold: true` draws one around cells above 35 µg/m³, and 50 µg/m³ on the daily mean) |
| `output_summary.json` | share of cells per AQI category, above 35 and 50, mean interval width, which basemap was used |

**Smooth band edges.** A coarse grid (for example `stride_px` 5 or 11) used to be drawn as a staircase of squares. The PNG maps
now interpolate the grid onto a finer one before the AQI colours are assigned, so the edges between bands are smooth curves.
The values are interpolated, never the colours, and the result stays within the range of the data. It is **display only**:
the GeoTIFFs and the numbers in `output_summary.json` keep the raw per-cell values. Settings in `configs/output.yaml`:

| Setting | Default | Effect |
|---|---|---|
| `smooth_maps` | `true` | `false` draws every cell as a flat square again |
| `smooth_sigma_cells` | `0.8` | a gentle blur, in source cells, that removes cell-to-cell speckle so the edges are clean; `0` interpolates only (edges follow the speckle); larger blurs small features |
| `smooth_target_px` | `1600` | a grid with fewer cells than this along its longest side is upsampled towards it; a full 10 m grid is already finer and is drawn as it is |

Per run: `--override output.smooth_maps=false`. Small features (a single hot cell) are softened by the blur, so use `smooth_sigma_cells=0` or
`smooth_maps=false` when exact cells matter.

**Basemap.** By default the maps try web tiles first, then offline street/boundary layers, then a plain
background, and each step is skipped quietly if it is not available. No street layer is configured by default,
so the maps have a plain background (`basemap used: none`). To get a real background:

- offline: set `data.paths.streets` and `data.paths.boundary` in `configs/data.yaml` (the same OSM / boundary
  files the rasters stage uses);
- online: `pip install contextily` and, for the default CARTO tiles, a free CARTO API key in `model/.env`
  (see below). The tiles are only a picture and never touch coordinates; no Google Maps API is involved;
- on an offline machine add `--override output.basemap.web_tiles=false` to skip the network completely.

**CARTO API key (`.env`).** CARTO's raster basemaps need a **free API key**; without one CARTO still answers
but draws "API KEY REQUIRED" into every tile. So the pipeline never requests CARTO tiles without a key. To use them:

1. request a key at <https://www.carto.com/basemaps/apikey/> (free, emailed to you; the free tier is 5 million
   tile requests a month, for non-commercial use, and CARTO / OpenStreetMap attribution stays on the map, which
   the pipeline draws);
2. open `model/.env` (it already exists, with an empty `CARTO_API_KEY=`) and paste the key after the `=`;
3. `pip install contextily`, then run `--stage infer` as usual.

`.env` is git-ignored (`git check-ignore model/.env` confirms it), so the key is never committed; commit
`.env.example`, never `.env`. The key is read from the environment only: it is not written into any output file,
config snapshot or `output_summary.json`, and it is replaced by `***` in warnings and errors. Variables set in your
shell win over the file. Without a key the maps use the offline layers or a plain background (or, with
`output.basemap.strict`, the run stops). Use `output.basemap_provider: OpenStreetMap.Mapnik` for a tile source
that needs no key (mind its usage policy). The key is sent as `?key=<key>` on the tile URL, which is how the
`xyzservices` definition of the CARTO providers (used by `contextily`) writes it; if your key's instructions say
otherwise, give the whole tile URL in `output.basemap_url` with `${CARTO_API_KEY}` where the key goes.

**Grad-CAM.** `python run_pipeline.py --stage explain --timestamp "2026-04-11 12:00"` works on a snapshot made by
`infer` for that date. It needs patches of at least 3 × 3 pixels (320 m / raster resolution), which the 10 m raster
gives (32 × 32). It writes overlays for the highest- and lowest-predicted cells of the snapshot (`gradcam_hot_cells.png`,
`gradcam_cold_cells.png`). Note that with a coarse `stride_px` the hot and cold cells are chosen from that coarse map.

### 2.7 Tune hyperparameters, then train with them

Normal training uses whatever is in `configs/*.yaml`; nothing adjusts itself. To search for better values:

```bash
python run_pipeline.py --stage tune
python run_pipeline.py --stage train --tuned
```

The search (`configs/tune.yaml`) is a Gaussian-process Bayesian optimisation over learning rate, batch size,
CNN kernel size and filters, LSTM size and recurrent dropout (batch size is searched between 128 and 512, around the 256 default: on the 10 m data a batch costs the same at any size, so small batches only make every trial slower). It uses nested spatial cross-validation: inner
folds guide the search, outer folds (held-out stations) score the tuned model. **It is slow**: every trial
trains one model per inner fold. For a quick demonstration (about 3 minutes on the tutorial machine):

```bash
python run_pipeline.py --stage tune --override tune.n_calls=3 --override tune.n_initial_points=2 --override tune.outer_folds=2 --override tune.inner_folds=2 --override tune.row_stride=40 --override tune.max_epochs=4 --override tune.patience=2
```

**What it prints.** A search is hours long, so the stage reports where it is (format example, numbers illustrative):

```
tuning plan: 4 searches x 12 trials x 3 inner fits = 144 fits on every 8th row, plus 3 evaluation fits on all rows of the outer-training stations. Output: a line per finished fit and trial, and a 'still running' line every 5 min
--- outer fold 0 (search 1/4): 18 held-out stations
  [outer 0] trial 1/12 fit 1/3: 7 epochs (best 3), val loss 0.2201, 6 min
  ... [outer 0] trial 1/12 fit 2/3 epoch 4/30, batch 120/271  (running for 9 min)
[outer 0] trial 1/12  loss 0.2214  best 0.2214  took 19 min, about 3.5 h left in this search  {'learning_rate': 0.004, ...}
  at this pace one search of 12 trials takes about 3.8 h, 4 searches about 15.2 h (the evaluation fits on all rows come on top)
[outer 0] evaluation fit with the winning hyperparameters: 555,000 rows, 5 epochs
[outer 0] evaluation fit: 8,669 batches, about 19 min for this pass ...
```

`train.runtime.status_every_s` (default 300) sets how often the `...` position line appears. The evaluation fits use all rows, so they print the same epoch lines as training. The projection after the first trial multiplies that trial's duration, so it is rough (trials differ with the batch size and the epochs they need).

To fit a real budget lower `n_calls` (keep `n_initial_points` ≤ `n_calls`), raise `row_stride`, lower `outer_folds`, or set `nested_evaluation: false`. Epochs are
not searched by default (early stopping decides); set `tune.search_epochs: true` to search them as a
hyperparameter instead. `--tuned` uses the newest result; pass `--tuned <folder>` for a specific one. The run
record notes which tuning run was used.

**`--tuned` applies to that one command only.** The tuned values are never written into `configs/`, so a later plain
`python run_pipeline.py --stage train` goes back to the YAML values. (The trained model itself keeps its tuned
architecture, so `geocp`, `infer`, `explain` and `report` on that run all use the tuned model.) To make the tuned
values the default, open `tuning/<id>/best_hyperparameters.json`: its `as_overrides` list gives each value as a
ready-made setting, e.g. `train.optimizer.lr=0.0039` or `model.temporal_lstm.hidden_size=66`. Copy them into
`configs/train.yaml` / `configs/model.yaml`, or paste them after `--override` on any command.

### 2.8 Compare with a temporal-only baseline

To see how much the morphology patches add, train a model without the CNN branch and compare on the same test
rows:

```bash
python run_pipeline.py --stage train
python run_pipeline.py --stage train --override model.spatial_cnn.enabled=false
python run_pipeline.py --stage geocp --run-id <temporal-only run id>
python run_pipeline.py --stage report --run-id <full run id> --baseline <temporal-only run id>
```

`metrics_summary.json` then has `comparison_to_baseline` with `r2_gain`. The spatial lag still feeds the LSTM in
both runs; for a strictly no-spatial-information baseline also pass `--override data.spatial_lag.enabled=false`
and re-run `preprocess` first.

### 2.9 The coordinate system

Everything uses **WGS 84 / UTM zone 51N (EPSG:32651)**, for the input (sensor positions, the Master GeoTIFF) and for
the output (the prediction and interval rasters and the maps are written in the raster's CRS). The setting is
`data.crs` in `configs/data.yaml`. Three checks protect it: the CRS must be projected in metres (distances are
planar metres, so a geographic CRS in degrees is refused), the Master GeoTIFF's CRS must equal it (checked in
`preprocess` and `infer`), and a trained model refuses data prepared in another CRS. To use a different projected
CRS in metres, change `data.crs`, rebuild or reproject the raster into it, and re-run everything from `preprocess`.

### 2.10 Use your own (production) data

1. In `configs/data.yaml` set the source layers under `paths`: `buildings_csv` (Google Open Buildings, WKT
   column), `streets` (OSM centrelines), `height_raster`, `ndvi_raster`, `boundary`; and the sensor inputs
   (`sensor_locations_csv` or `sensors_gpkg`, `sensor_csv_dir`).
2. Build the Master GeoTIFF (heavy: it rasterises buildings at 10 m, in EPSG:32651):
   `python run_pipeline.py --stage rasters`.
3. Set `paths.master_raster` to the file it printed (`processed/rasters/master_morphology_stack_10m.tif`).
4. Run `--stage all`. With the 10 m raster the raster has 6 bands (σH included) and patches are 32 × 32 pixels;
   `C`, `T`, `F` and the patch size are derived automatically and written into every checkpoint.
5. Training budgets: leave `train.epochs: 50` (early stopping ends sooner), consider `--stage tune`, and use a
   GPU if you have one. The model has ~160 k parameters with `feature_map: pool`; `flatten` is far larger on 32 × 32 patches.
6. For a map of the whole city use `--stage infer`; start with `--override inference.stride_px=5` or an
   `inference.window` because a 10 m raster of the NCR has about 10⁷ cells.

The rasters stage has only been tested on synthetic data, so inspect its first real output (band values, extent,
alignment with the sensors) before training on it.

### 2.11 Changing settings

Three ways, from lightest to heaviest:

```bash
python run_pipeline.py --stage train --override train.epochs=10                  # one run only
python run_pipeline.py --stage train --override "data.temporal.lags=[4,3,2,1]"   # lists need quotes in PowerShell
```

or edit the file for good. One YAML per concern:

| File | Controls |
|---|---|
| `configs/data.yaml` | CRS version and file paths, columns, gap filling, lags, rolling statistics, spatial lag, patch size, raster building, scaling, station split |
| `configs/model.yaml` | CNN (filters, kernel, feature map, on/off), LSTM (size, layers, dropout), fusion head |
| `configs/train.yaml` | seed, device, epochs, batch size, optimiser, loss, early stopping, cross-validation, final epochs |
| `configs/tune.yaml` | search space, budget (`n_calls`, folds, `row_stride`), epoch search |
| `configs/geocp.yaml` | miscoverage `alpha`, kernels and bandwidths, normalisation, zones |
| `configs/report.yaml` | which metrics are headline, Moran's I / LISA settings, plots |
| `configs/inference.yaml` | snapshot hours, neighbour reference, grid stride / window |
| `configs/output.yaml` | AQI scale and colours, 35 / 50 µg/m³ limits, basemap, Grad-CAM size |

The switches most often wanted:

| To… | Setting | Values (default first) |
|---|---|---|
| flatten the CNN feature map instead of pooling it | `model.spatial_cnn.feature_map` | `pool`, `flatten` |
| drop the CNN (temporal-only baseline) | `model.spatial_cnn.enabled` | `true`, `false` |
| drop the spatial lag | `data.spatial_lag.enabled` | `true`, `false` |
| add rolling mean / std features | `data.temporal.rolling_window` | `null`, e.g. `3` |
| skip cross-validation in `train` | `train.cv.enabled` | `true`, `false` |
| search epochs during tuning | `tune.search_epochs` | `false`, `true` |
| skip the outer tuning loop | `tune.nested_evaluation` | `true`, `false` |
| GeoCP weights normalised to one, or with the finite-sample correction | `geocp.finite_sample_correction` | `false`, `true` |
| neighbours used for map cells | `inference.lag_reference` | `train`, `all` |
| predict a coarser grid | `inference.stride_px` (odd) | `1`, `3`, `5` … |
| basemap sources | `output.basemap.enabled`, `.web_tiles`, `.vector_layers`, `.strict` | `true`, `true`, `true`, `false` |
| run LISA even if the global test is not significant | `report.bias.local_only_if_global_significant` | `true`, `false` |

### 2.12 Troubleshooting

| You see | Cause and fix |
|---|---|
| `Master GeoTIFF is … but data.crs is …` | the raster is not in `data.crs` (EPSG:32651). Reproject it (e.g. QGIS, or `gdalwarp -t_srs EPSG:32651`) or rebuild it with `--stage rasters` |
| `no trained run found; run --stage train first` | nothing in `checkpoints/`. Train first |
| `checkpoint does not match the processed data` | `preprocess` was re-run with different settings (lags, patch size, CRS) after training. Re-train, or restore the settings; the message lists what differs |
| `KeyError: unknown config key` | typo in an `--override`. Keys are `file.key.subkey`, e.g. `train.epochs`, `data.temporal.lags` |
| `infer` gives no cells / `not enough history` | the date is outside the data, or no station reported in the three hours before a snapshot. Check the range ([2.6](#26-make-the-maps)) |
| `explain` says `prediction_volume.tif missing` | `explain` works on a snapshot made by `infer`; run `--stage infer --date …` for that date first |
| `explain: patches are 1x1 …` | the raster is too coarse for a patch of at least 3 × 3 pixels (320 m / resolution). Use a finer raster |
| `infer` prints `... this snapshot takes about N h` | the grid is large. Use `--override inference.stride_px=5` (or 11, 21) or an `inference.window`, see [2.6](#26-make-the-maps) |
| `COVERAGE: INCOMPLETE` | some sensors lie outside the Master GeoTIFF and are skipped, see [2.14](#214-covering-the-whole-ncr) |
| `contextily is not installed` warning | optional. Install it, or ignore it, or set `output.basemap.web_tiles=false` to stop looking |
| `CARTO basemap tiles need a free API key` | `CARTO_API_KEY` is empty. Put the key in `model/.env` ([2.6](#26-make-the-maps)) |
| map tiles say `API KEY REQUIRED` | the key is missing, wrong or not accepted (CARTO answers an invalid key like no key). Fix it in `.env`, then delete `.tile_cache` so the watermarked tiles are not reused |
| `ModuleNotFoundError: preprocessing` | you are not inside `model/` |
| `spatial bias: fewer than 6 grid cells` | too few training stations for Moran's I; lower `report.bias.min_cells` knowingly or use more stations |
| training prints nothing for many minutes | it is not frozen: an epoch is one long pass. The first epoch prints its length and the projected total, then a progress line every 30 s, see [2.12b](#212b-long-runs-progress-and-timeouts) |
| `TIMEOUT in stage 'train': … no progress for 600 s` | a real hang was detected (exit code 124), see [2.12b](#212b-long-runs-progress-and-timeouts) |
| training is very slow | a larger `train.batch_size` (the biggest lever, measured below), `--override train.epochs=…`, a GPU, or for tuning a smaller `n_calls` / larger `row_stride` |

### 2.12b Long runs: progress and timeouts

On a CPU one training epoch takes minutes to tens of minutes and, before, printed nothing until it ended, which looks like a
freeze. Training now prints, from the first epoch:

```
[fold 0] epoch 1/50: 8,669 batches, about 19 min for this pass at the current speed; a progress line follows every 30 s
WARNING: the whole run takes about 14.9 h if every fit stops after ~12 epochs (early stopping), at most 62 h. A larger train.batch_size shortens it ...
[fold 0] epoch 1/50:  34.0%  2,940/8,669 batches  elapsed 6 min  remaining ~12 min
```

**Why it was slow, measured on the data folder with the QC sensors** (554,785 rows per fold, 4 CPU threads, one epoch of a CV fold):

| `train.batch_size` | one epoch | note |
|---|---|---|
| 64 (the old default) | 18.6 min | a full run with 3 folds and the final fit is about 15 h |
| 256 (**the default now**) | 5.8 min | about 5 h for the same run; what the earlier runs here used |
| 1024 | 1.6 min | a batch costs almost the same at any size (each batch computes only its ~50 distinct station patches), so bigger batches are nearly free; the cost is fewer weight updates per epoch, which may need more epochs or a higher learning rate |

**Timeouts** (`runtime:` in `configs/train.yaml`, implemented in `runtime_guard.py`). Only the *absence of progress* is an error;
a slow run is not.

| Setting | Default | Meaning |
|---|---|---|
| `stall_timeout_s` | 600 | a loop (training epoch, validation, prediction, per-cell inference) that finishes **no batch** for this long is a hang. One batch takes 0.1-0.2 s and an inference chunk under 15 s, so 10 minutes never trips on healthy work. `0` = off |
| `hard_exit_grace_s` | 60 | a hang inside native code (a stuck torch operation, an unreadable page) cannot be interrupted from Python; after this much longer the process is ended with a traceback |
| `max_hours` | `null` | optional wall-clock limit per command, checked after every batch. Off by default: a legitimate CPU run is many hours, so a cap would kill good work; the first epoch tells you the projected total instead |
| `progress_every_s` | 30 | seconds between progress lines (`0` = quiet) |
| `status_every_s` | 300 | fits that print nothing per epoch (the tune stage) give one `... [outer 0] trial 3/12 fit 2/3 epoch 4/30, batch 120/271` line this often (`0` = none) |
| `warn_hours` | 2 | the first epoch prints `WARNING` when the projected run is longer |

On a hang or an exceeded budget the command prints `TIMEOUT in stage '…': …` (with the epoch and batch where it stopped), exits
with code **124** (the code GNU `timeout` uses, so scripts can test for it), and marks the unfinished run's `run_info.json` as
`"status": "timed_out"` with the reason. Finished runs say `completed`; a run that is still `running` after the process is
gone was killed from outside (closed terminal, power loss). A real Ctrl+C is not turned into a timeout. If you do raise
`train.batch_size` or `inference.chunk_cells` a lot, check that one batch or chunk still finishes well inside `stall_timeout_s`.

### 2.13 Start over (reset)

Everything the pipeline generates (trained models, prepared data, results, tuning) is kept in four folders. After
trying some data, clear them before using other data, because a model, scalers or tuned hyperparameters made from
one dataset do not belong with another:

```bash
python run_pipeline.py --stage clean
python run_pipeline.py --stage clean --yes
```

The first line is a **dry run**: it lists what would be deleted, with file counts and sizes, and deletes nothing. The
second really deletes. (`reset` is an alias for `clean`.) Options:

| Option | Effect |
|---|---|
| `--keep tuning` | leave a folder alone; repeatable. Choices: `processed`, `checkpoints`, `snapshots`, `tuning` |
| `--include-rasters` | also delete the built Master GeoTIFF in `processed/rasters/` |
| `--include-tile-cache` | also delete the cached basemap tiles (`.tile_cache`) |

What is protected: `configs/`, `.env` (and your API key), the code, and all input data (sensor CSVs, source rasters,
the Master GeoTIFF) are never touched, and any folder outside `model/` is refused. The Master GeoTIFF built by
`--stage rasters` is **kept by default**, because it is slow to rebuild. The folders stay, with their `.gitkeep`
placeholders.

To start over with different data: `--stage clean --yes`, point `data.paths` at the new files (and build a raster with
`--stage rasters` if you do not have one), then `--stage all`. If you only want to retrain with other settings, you do not need a reset:
every run is kept side by side in `checkpoints/` and compared in `runs_index.csv`.

### 2.14 Covering the whole NCR

A sensor outside the Master GeoTIFF is skipped, which silently crops the study area, so the pipeline checks and reports
it. `python run_pipeline.py --stage coverage` (read-only, a second to run) and the start of every `preprocess` print:

```
Master GeoTIFF: 27.3 x 39.0 km at 10 m, EPSG:32651, lon 120.900..121.150 E, lat 14.398..14.752 N
Sensors inside the raster: 45 of 46
  OUTSIDE (skipped): Barangay Tunasan - Maharlika Highway (SM Center Muntinlupa) (4797656) ...: 2.3 km beyond the south edge
COVERAGE: INCOMPLETE: part of the study area is cropped away
```

**What this means for the current data.** `master_raster_v2.tif` is a rectangle from 120.90 to 121.15 °E and from
14.398 to 14.752 °N. Muntinlupa is part of the NCR and its Tunasan station sits 2.3 km south of that edge, so the
raster does not reach the southern tip of the NCR. The pipeline cannot invent morphology data there: you need a raster
that extends further south (and possibly north: check the northern part of Caloocan and Valenzuela as well).

**To measure it properly,** set `data.paths.boundary` to the NCR boundary polygon (gpkg / shp / geojson). The report then
also gives the share of the NCR area the raster covers and how many km are missing on each side.
`data.coverage.strict: true` makes `preprocess` stop while coverage is incomplete.

Two other facts about the raster: it is a plain rectangle with data everywhere (no nodata outside the NCR), so city
maps also show Manila Bay and the neighbouring provinces; and far from any sensor a cell's prediction is mostly its
nearest stations' values (the spatial-lag stand-in of [3.9](#39-city-wide-inference-and-maps)).

---

## 3. How it works

### 3.1 Folder map

```
configs/        data  model  train  tune  geocp  report  inference  output   (.yaml)
preprocessing/  clean_loader  spatial_partitioner  spatial_lag  temporal_window_builder
                spatial_patch_extractor  multimodal_aligner  scalers
                vector_sanitizer  raster_harmonizer  master_raster_builder      (rasters stage)
networks/       spatial_cnn  temporal_lstm  fusion_head  dual_branch_net
training/       dataset  loss  metrics  trainer  checkpoint  predict  run_info
                fold_features (per-fold spatial lag)  tuning (Bayesian search)
uncertainty/    geocp  geocp_surface (interval rasters)
inference/      cell_inference (a prediction for every raster cell)
output/         volume (volume, daily mean, summary)  aqi_maps (maps)
explain/        gradcam  gradcam_plots
reporting/      snapshot_generator  plotters  spatial_bias (Moran's I / LISA)
tests/          pytest suite        docs/PROCESS.txt  the same process as plain text
processed/ checkpoints/ snapshots/ tuning/   generated, git-ignored
```

### 3.2 Data flow

```
sensor CSVs --> clean (hourly, gaps <= 24 h filled) --> partition stations: train / calibration / test
sensor lat/lon --> projected x, y --------------------^         |
                                                                 v
                          spatial lag (training stations only) --> windows [N, T, F] and targets [N, 1]
Master GeoTIFF --> one 320 m patch per station --> patches.npy   |
                                                                 v
                                   scale with TRAIN statistics --> processed/<version>/
processed --> train (spatial CV + final fit) --> checkpoint ---------------------------+
          --> tune (optional) --> best_hyperparameters --> train --tuned               |
checkpoint --> geocp (calibration + test) --> bias --> report
checkpoint --> infer (every raster cell, 4 snapshots/day) --> volumes + AQI maps ; explain (Grad-CAM)
```

### 3.3 Preprocessing

1. Per-sensor CSVs are read for the sensors inside the raster extent and put on a strict hourly index;
   timestamps are converted to local time. Readings outside `clean.min_value` … `max_value` become missing.
2. Gaps of up to 24 hours are linearly interpolated; longer gaps stay entirely empty. (pandas' `limit=` would
   fill the first 24 hours of a long gap and fabricate data, so gaps are measured first and then masked.)
3. **Stations are split by location** into train / calibration / test (seeded, 60 / 20 / 20 by default). Every
   row, lag window and patch of a station stays in one partition; the test stations are not touched until the
   end.
4. **Spatial lag**: for every station and hour, the inverse-distance mean of the PM2.5 of its k = 5 nearest
   *training* stations, never itself. A station without a reading in an hour is dropped from that mean.
5. **Windows**: for each (station, hour *t*) the sequence of the lag hours `[3, 2, 1]`, each step carrying
   `[PM2.5, spatial lag, sin(hour), cos(hour)]`, with PM2.5 at *t* as target. A sample is dropped if any lag
   hour is missing. Optional rolling mean / std can be added.
6. **Patches**: one patch per station from the Master GeoTIFF, centred on the raster cell that contains the
   sensor, 320 m / raster resolution pixels wide (32 at the 10 m raster), zero-padded at the edge.
   Stored once per station and looked up per sample, so the data does not grow with the number of hours.
7. **Scalers** are fit on the training partition only. The target and every PM2.5-derived column share one
   standard scaler; raster bands are scaled per band over the training stations' patches.
8. The hourly PM2.5 matrix is saved too (`pm25_hourly.npz`) so cross-validation and inference can recompute the
   spatial lag later.

The Master GeoTIFF bands are `height, ndvi, building_density, street_width, aspect_ratio, height_std`
(σH, the standard deviation of building height in a 500 m window around each pixel, the same footprint the
XGBoost side uses per grid cell). The cyclical time encoding is an LSTM input, not a raster band.

### 3.4 The model and training

A CNN encodes the patch into one vector, an LSTM encodes the lag sequence into its last hidden state, the two are
concatenated (late fusion), and a dense head outputs the PM2.5 value. There is one loss; the gradient splits at
the concatenation point and flows into both branches, and Adam updates everything end to end (the gradient flow
into both branches is tested).

The CNN's feature map is a switch: `pool` (default) average-pools it to 1×1 and applies a linear layer;
`flatten` flattens the whole map first (the literal reading of the thesis text). `flatten` is much wider on
32×32 patches (128 × 32 × 32 inputs to the linear layer).

`train` does two things:

1. **Spatial k-fold cross-validation on the training stations** (`split.cv_folds`, default 3). Each fold holds
   out whole stations, recomputes the spatial lag from the fit stations (3.6) and trains with early stopping.
   Fold metrics go into `run_info.json`, out-of-fold predictions into `oof_predictions.csv`.
2. **The final fit on all training stations** for the mean best epoch of the folds (or the tuned epochs with
   `--tuned`). This model is `best_dual_branch_model.pt`.

`epochs` in `train.yaml` is only an upper limit; early stopping ends training once validation loss stops improving.

**Speed.** A station's patch is the same for every hour, so a batch repeats the same few patches. The CNN therefore
processes each *distinct* patch of a batch once and the result is looked up per row (dropout is still applied per
row). The outputs and gradients are identical to the plain computation (tested), and on the 10 m data it is much
faster: with batch 256 an epoch over the 300 thousand training rows takes about 2 minutes instead of 15 on a 4-thread
CPU (about 7 minutes instead of 15 at the default batch 64). Larger batches gain the most.

### 3.5 Hyperparameter tuning

`--stage tune` follows the thesis text:

- **Space**, in the three groups: global (learning rate, batch size), spatial (CNN kernel size, filters),
  temporal (LSTM state size, recurrent dropout). Optional: the CNN feature map.
- **Method**: a Gaussian-process surrogate (`skopt.gp_minimize`) with an acquisition function (EI); the first
  `n_initial_points` trials are random.
- **One objective for the whole network**: the mean, over inner folds, of the best validation loss after the
  dense head on held-out stations, so CNN and LSTM hyperparameters are tuned together.
- **Nested spatial CV, inside the training stations only**: for each outer fold (a group of held-out stations)
  the inner folds drive the search, the winner is refit on the outer-training stations and scored on the outer
  held-out stations. A final search over all training stations writes `best_hyperparameters.json`. Calibration
  and test stations are never used.
- **Epochs**: by default not searched (each trial early-stops, so it costs only the epochs it needs, and the
  winner's mean best epoch becomes the final epoch count). With `search_epochs: true` they are searched and each
  trial trains exactly that many epochs, scored by the loss after the last one.
- For speed the search uses every `row_stride`-th row (consecutive hours are near-duplicates); refits and
  evaluation use all rows.

### 3.6 The spatial lag inside cross-validation

Preprocessing computes the spatial lag with all training stations as neighbours. Inside cross-validation that
would leak: a validation station's readings would be a *feature* of the fit stations and reach the model during
training, making validation scores optimistic. So for every fold (and every tuning fold) the lag is recomputed
from that fold's fit stations only, and validation stations get theirs from fit stations too, as calibration and
test stations do in production. Rows whose lag is not computable are dropped from that fold. The final fit,
calibration and test keep the preprocessing lag. Tests show that changing a held-out station's readings changes
nobody's lag, and that on the pilot data the recomputed lag with the full training set equals the saved one.

### 3.7 Validation

- **Metrics**: RMSE, MAE and R² are the headline numbers, computed on stations withheld from fitting. IA and
  Pearson r are kept as supplementary.
- **Spatial bias** (`--stage bias`), with residual = actual − predicted. Out-of-fold residuals of the spatial CV
  (training stations only; an assertion refuses any calibration or test station) are averaged per grid cell,
  KNN spatial weights are built (k = 4), and Global Moran's I is tested (999 permutations, two-sided). Only if
  significant is Anselin's Local Moran's I run, giving LISA clusters: **High-High = under-prediction hotspots**,
  Low-Low = over-prediction coldspots, High-Low / Low-High = outliers. Clusters are laid over every raster
  band and a per-cluster band profile is written, to show which morphology the model keeps missing.
- **Temporal-only baseline** for the R² comparison: see [2.8](#28-compare-with-a-temporal-only-baseline).
- **Order of access**: nested CV and tuning (training stations) → final fit (training stations) → calibration
  predictions and GeoCP configuration → test predictions, intervals and Table 4.2 metrics. The test stations are
  touched once, last.

### 3.8 GeoCP: intervals around every prediction

The calibration stations never fit the model. After training, their absolute errors |y − ŷ| (in µg/m³, after
undoing the target scaling) are the nonconformity scores, saved with projected cell-centre coordinates in
`calibration_table.csv`, in a format that any model (e.g. XGBoost) could produce. For a query location each
score is weighted by a kernel of the planar distance (metres) between its station's cell centre and the query,
the weights are normalised to sum to one, and the (1 − α) quantile of that weighted distribution is the
half-width, applied symmetrically around the prediction. The lower end is then clipped at 0; coverage is
checked on the **unclipped** interval, the published width is upper − clipped lower.

- **Kernel and bandwidth** are chosen by leave-one-station-out on the calibration stations: the narrowest
  intervals whose coverage stays within `coverage_tolerance` of 1 − α. Bandwidth `.inf` is plain split
  conformal. Only then are the test stations scored.
- **Normalisation**: `finite_sample_correction: false` (default, as in the text) or `true` (adds the query's own
  weight at +∞; wider when calibration data is small). Every run scores both options on the calibration
  stations (`geocp.json → correction_comparison`), so choosing between them never needs the test stations.
- **Table 4.2 metrics** on the test stations: coverage, mean and median width, width-to-prediction ratio, and
  coverage in the urban core vs. peripheral zones (split by distance to the nearest calibration sensor).
- Before predicting, the checkpoint's patch size, band count, lags, feature order, CRS and scaler hash are
  compared with the data on disk and a mismatch stops the run.

### 3.9 City-wide inference and maps

`--stage infer` predicts the centre cell of a patch for every cell of the raster (patches overlap), at the hours
in `inference.snapshot_hours`.

A cell without a sensor has no PM2.5 history. The thesis (Section 4.5) does not say how such a cell is fed to
the model; the only description is the spatial-lag paragraph of the preprocessing text ("areas without a
reading rely on neighbouring cells' values"), which is what this applies: at each lag hour the spatial lag of
the cell's k nearest stations fills **both** the PM2.5 and the spatial-lag channels; the patch, hour encoding
and scalers are exactly those of training. `inference.lag_reference: train` uses the training stations as
neighbours (what the model and the calibration saw).

Per date it writes the prediction volume with aligned lower / upper / width volumes, the 24-hour mean, AQI maps
and `output_summary.json` (see [2.6](#26-make-the-maps)). The AQI scale is in `configs/output.yaml`: **Good
0–25, Fair 25.1–35, Unhealthy for sensitive groups 35.1–45** are confirmed; "Very unhealthy 45.1–55" comes from
a single secondary source and the tiers above 55 are not verified, so check them against DENR DAO 2020-14 and
edit `aqi.bins`. No interval is defined for the daily mean (intervals belong to single snapshots).

`--stage surface --prediction-raster pm25.tif` builds the same interval layers for any prediction raster.

### 3.10 Explainability: Grad-CAM

On the last convolutional layer of the CNN: channel weights are the mean gradient of the scalar prediction, the
map is the ReLU of the weighted sum of activations, rescaled to [0, 1] per patch. It is the gradient of the
full prediction, so it shows what in the patch mattered *given the temporal history*. For the highest- and
lowest-predicted cells of a snapshot it writes overlay figures (every morphology band with the attended region
outlined), `gradcam_attention.csv`, and a summary giving for each band how many patch standard deviations higher
the attended pixels are than the rest of the patch (positive values for building density, aspect ratio, height
or street width point at dense blocks, canyons and arterials). The overlays are still to be inspected by eye.
SHAP belongs to the XGBoost branch and is not part of this pipeline.

### 3.11 The coordinate system

One CRS runs through the whole pipeline: **WGS 84 / UTM zone 51N (EPSG:32651)**, chosen with the team. Sensor
positions are projected into it, the Master GeoTIFF is built in it, patches, KNN and GeoCP distances are planar
metres in it, and the rasters and maps `infer` writes inherit the raster's CRS, so input and output always agree.
(The thesis' output text names PRS92; the project does not use it.) Safeguards: `data.crs` must be projected in
metres (a geographic CRS in degrees is refused); the raster's CRS is checked in `preprocess` and `infer`; a
checkpoint refuses data prepared in another CRS.

### 3.12 Tracking every run

Hyperparameters never change by themselves: whatever the YAML files plus `--override` say is used.
`checkpoints/<run_id>/run_info.json` (also copied to the snapshot folder) records the hyperparameters, the
derived shapes (T, F, bands, patch size), parameter counts per branch, the full layer listing, the git commit,
partition sizes, per-fold CV results, the final epochs and where they came from, `tuned_from`, test and
calibration metrics, the GeoCP choice and the spatial-bias summary. `checkpoints/runs_index.csv` has one row
per run for side-by-side comparison. The checkpoint also stores the model settings, so
`training.checkpoint.load_model(path)` rebuilds the network without guessing.

---

## 4. Outputs reference

```
processed/
    patches.npy  patch_index.json  stations.csv  partition.json  scalers.json  pm25_hourly.npz
    X_temporal_{train,calibration,test}.npy  y_*.npy  station_idx_*.npy  meta_*.csv  preprocess_info.json
checkpoints/
    runs_index.csv
    <run_id>/  best_dual_branch_model.pt  fold_*.pt  history.json  oof_predictions.csv  run_info.json
tuning/<id>/
    best_hyperparameters.json  tune_info.json  search_outer_*.json  search_final.json  convergence_final.png
snapshots/<run_id>/
    SUMMARY.md  metrics_summary.json  run_info.json
    plots/    loss_curve.png  pred_vs_obs.png  residual_hist.png  timeseries_sample.png  accuracy_by_station.png
              geocp_coverage.png  lisa_map.png  lisa_on_bands.png
    tables/   accuracy_summary.csv  accuracy_by_station.csv  cv_folds.csv
    data/     predictions_test.csv  predictions_calibration.csv  calibration_table.csv  geocp.json
              spatial_bias.json  residuals_by_cell.csv
    surface/<YYYYMMDD>/   prediction_volume.tif  lower_volume.tif  upper_volume.tif  width_volume.tif
                          prediction_daily_mean.tif  output_summary.json  maps/*.png
    explain/<YYYYMMDD_HHMM>/   gradcam_hot_cells.png  gradcam_cold_cells.png  gradcam_attention.csv  gradcam_summary.json
```

### Visualizations at a glance

| Picture | Made by | Shows |
|---|---|---|
| `loss_curve.png` | `report` | training and validation loss per epoch for every CV fold, the final fit dashed, best epochs as dots |
| `pred_vs_obs.png` | `report` | predicted vs observed PM2.5 on the test stations (density hexbins, 1:1 line; n, RMSE and r in the title) |
| `residual_hist.png` | `report` | histogram of prediction − observation with the mean marked |
| `timeseries_sample.png` | `report` | the last 7 days at the best-covered test station: observed, predicted and the GeoCP band |
| `accuracy_by_station.png` | `report` | RMSE of every test station (bars) with its MAE (dots) against the overall RMSE |
| `geocp_coverage.png` | `report` | per test station, the share of readings inside the interval against the target |
| `lisa_map.png` | `bias` | grid cells coloured by LISA cluster (or by mean residual if LISA was not run); Moran's I in the title |
| `lisa_on_bands.png` | `bias` | the LISA clusters laid over every morphology band |
| `convergence_final.png` | `tune` | validation loss of every Bayesian-search trial and the best so far |
| `maps/aqi_HHMM.png` | `infer` | one map per snapshot on the Philippine AQI scale, stations marked (optional outline above 35 µg/m³, off by default) |
| `maps/daily_panels.png` | `infer` | the day's snapshots side by side on one scale |
| `maps/daily_mean.png` | `infer` | the 24-hour mean on the same scale (optional outline at the 50 µg/m³ guideline, off by default) |
| `gradcam_hot_cells.png`, `gradcam_cold_cells.png` | `explain` | for the highest / lowest predicted cells: every morphology band with the attended region outlined, and the Grad-CAM map |

Not drawn as pictures (yet): the GeoCP **interval-width surface** exists only as a GeoTIFF (`width_volume.tif`), although
the thesis asks to inspect it visually; coverage by zone (urban core vs periphery) is in the numbers only; there is no
map of which stations are train / calibration / test; SHAP belongs to the XGBoost branch; and the comparison with
earlier local studies needs external data.

---

## 5. Known limits

State these when reporting results.

- **The split is spatial only.** Every partition covers the same hours, so a regional weather episode appears in
  train and test. Scores measure generalisation to *new locations*, not forecasting.
- Inside a fold the spatial lag has fewer reference stations, so a held-out station's lag is slightly less
  informative than in production.
- **The raster does not cover the whole NCR.** It stops at 14.398 °N, and the Tunasan station in Muntinlupa lies 2.3 km
  south of it ([2.14](#214-covering-the-whole-ncr)). That station is skipped and nothing is predicted for that part.
  Without an NCR boundary file the missing area cannot be measured exactly.
- The raster is a rectangle, so maps also cover Manila Bay and the neighbouring provinces, and far from any sensor a
  prediction is mostly the nearest stations' values.
- **The numbers in this file are from 2-epoch smoke runs** on 44 stations; the model has not been trained to
  convergence or tuned on this data yet.
- The Quezon City sensors (`data/QC Data`) are converted by `preprocessing/qc_convert.py` ([2.1b](#21b-the-quezon-city-sensors-qc-data)) but `configs/data.yaml` still reads OpenAQ only until you point it at them.
- A full-resolution city map takes hours on a CPU (see [2.6](#26-make-the-maps)).
- With few calibration stations GeoCP coverage can fall below target on the test stations (0.87 vs 0.90 in the 2-epoch
  run, with 7 calibration stations).
- **The rasters stage has only been tested on synthetic data**, not on the real building / street / height / NDVI files.
- **City-wide inference is an interpretation** of the thesis (see 3.9). The model was trained on sensors whose
  own lags are real readings, and the GeoCP calibration scores also come from real histories, so intervals on
  stand-in cells may be too narrow and the surface smoother than reality. A stricter alternative is to compute
  the calibration predictions with the same stand-in.
- Inference at 10 m over the whole NCR is very heavy on a CPU; use `stride_px` / `window` to look first.
- The AQI tiers above 55 µg/m³ are not verified. Web-tile basemaps have only been tested with stand-ins.
- Moran's I on out-of-fold residuals describes the cross-validation stations only, not the whole city.
- The thesis compares the daily mean with coarser interpolations from earlier local studies; that needs
  external data and is not built.
- The example runs use a few epochs and tiny search budgets: they check that things work, not that they are good.

---

## 6. How it was checked

`python -m pytest tests -q` runs 317 tests, 313 of which apply when `contextily` is installed (about 1 min), covering: configuration and the CRS check; the reset command; the infer / geocp workflow; the coverage check; the QC data conversion and the `qcdata_` loader; progress, hang detection and timeouts (also for the tune stage); the accuracy report and the snapshot layout, the map outline switch and the smooth band edges; shared-patch training; shapes and
gradient flow into both branches (pool and flatten); trainer and checkpoint round trips; metrics against
hand-computed values; gap capping; patch/window alignment checked by hand; partitioning; the spatial lag
(hand-computed, leakage, per-fold); scalers; GeoCP (normalisation, clipping, Table 4.2 metrics, zones, grid
vs. exact quantiles, interval rasters); tuning (search, nested folds, epoch switch); Moran's I / LISA (sign
convention, significance); the temporal-only baseline; checkpoint compatibility; per-cell inference (vectorised
patches equal the training extraction, stand-in hand-computed); output volumes, AQI categories and maps
(including pixel comparisons for the basemap switches); Grad-CAM (localisation, determinism, hooks removed).

End-to-end runs on the pilot data covered preprocess → tune → train `--tuned` → geocp → bias → report → infer
→ explain, a temporal-only run compared with `--baseline`, and (on the data folder) preprocess → train → report → infer
→ explain, including Grad-CAM on real 32 × 32 patches.

---

## 7. Which thesis text became which code

| Thesis text | Code |
|---|---|
| Preprocessing: sanitation, street widths, CRS | `preprocessing/vector_sanitizer.py` |
| Preprocessing: raster harmonisation, Master GeoTIFF | `raster_harmonizer.py`, `master_raster_builder.py` |
| Preprocessing: alignment, cyclical time, lags, 24 h gap cap | `clean_loader.py`, `temporal_window_builder.py` |
| Preprocessing: spatial partitioning | `spatial_partitioner.py` |
| Preprocessing: spatial lag | `spatial_lag.py`, `training/fold_features.py` |
| Preprocessing: patches, dual-branch tensors, scaling | `spatial_patch_extractor.py`, `multimodal_aligner.py`, `scalers.py` |
| Training: architecture | `networks/` |
| Training: hyperparameter groups, Bayesian optimisation, nested CV | `training/tuning.py`, `configs/tune.yaml` |
| Validation: RMSE / MAE / R², spatial bias | `training/metrics.py`, `reporting/snapshot_generator.py`, `reporting/spatial_bias.py` |
| Section 4.5 / Validation: GeoCP | `uncertainty/geocp.py`, `uncertainty/geocp_surface.py` |
| Output: inference, volume, AQI maps | `inference/`, `output/` |
| Output: Grad-CAM | `explain/` |

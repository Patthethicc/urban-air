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

> **Status.** Developed against the 500 m *pilot* raster and 18 pilot stations. The pipeline runs end to end
> and 204 tests pass, but the pilot numbers in this file only show that the wiring works: the CNN sees 1×1-pixel
> patches there, and the example runs train for a few epochs. The real 10 m raster, real training budgets and
> the rasters stage on real source files have not been run (see [Known limits](#5-known-limits)).

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
python run_pipeline.py --stage all --override train.epochs=3 --override train.batch_size=256
python run_pipeline.py --stage infer --date 2026-04-11
```

The `all` line runs the pilot data through preprocessing, training, GeoCP, the bias check and the report in
about a minute on a CPU (3 epochs; the defaults train much longer). Results are in
`snapshots/<run_id>/` and the maps in `snapshots/<run_id>/surface/20260411/maps/`.

---

## 2. Tutorial: running it

### 2.1 What you need

- **Python 3.10+** (developed and tested on 3.14.5, Windows 11, CPU-only PyTorch).
- The packages in `requirements.txt` (names only). `requirements-tested.txt` has the exact versions the code
  was tested with, if you need to reproduce results. For a GPU, install PyTorch first from pytorch.org, then
  `pip install -r requirements.txt`; the code uses CUDA automatically (`train.device: auto`).
- Optional: `contextily` for web-tile basemaps on the maps (see [2.6](#26-make-the-maps)).
- **Run every command from inside the `model/` folder.**

The pilot data is already referenced by `configs/data.yaml`, with paths relative to `model/`:

| What | Where |
|---|---|
| Hourly PM2.5 per sensor | `../pilot testing/CNN-LSTM/Air Data/openaq_<id>_*.csv` (45 sensors) |
| Sensor coordinates | `../pilot testing/CNN-LSTM/OpenAQ Sensor Locations - OpenAQ Sensor Locations.csv` |
| Master GeoTIFF (500 m pilot, UTM 51N) | `../pilot testing/CNN-LSTM/master_morphology_stack_500m.tif` |

Only 18 of the 45 sensors fall inside the pilot raster; the rest are skipped automatically.

### 2.2 Check the install

```bash
python -m pytest tests -q
```

Expect `204 passed` (and 4 skipped, which only apply when `contextily` is not installed) in about a minute. If something fails here, fix it before running anything else.

### 2.3 First run: the whole pipeline on the pilot data

```bash
python run_pipeline.py --stage all --override train.epochs=3 --override train.batch_size=256
```

`--override section.key=value` changes one setting for this run only; here it makes training short. Without
the overrides the defaults apply (up to 50 epochs, batch 64), which takes many times longer on a CPU.

What you should see (abridged; numbers will differ a little between runs):

```
stations per partition: {'train': 11, 'calibration': 3, 'test': 4}
windows: X_temporal (157333, 3, 4), y (157333, 1)
[fold 0] held-out stations RMSE 5.2  MAE 3.0  ...
final fit on all training stations for 3 epochs
run 20261013_122820 saved to checkpoints/20261013_122820
GeoCP config: {'kernel': 'gaussian', 'bandwidth_m': inf, 'alpha': 0.1, ...}
test coverage 0.829 (target 0.90), mean width 13.09 ug/m3, median 13.15
spatial bias: Moran's I -0.080 (p 0.784, 11 cells); LISA not run
snapshot written to snapshots/20261013_122820
```

That took about 70 seconds on the machine used for this tutorial. Three folders were created, one per kind of
output:

```
processed/                 the arrays the model trains on (patches, windows, scalers, partition)
checkpoints/<run_id>/      the model, the settings it used, per-fold models, learning curves
snapshots/<run_id>/        metrics, intervals, plots, bias analysis
```

### 2.4 The same thing, one stage at a time

Run these to redo or inspect a single step. Each stage reads what the previous one wrote.

| Stage | Command | What it does | Main output |
|---|---|---|---|
| preprocess | `--stage preprocess` | clean hourly series, split stations into train / calibration / test, spatial lag, windows, patches, scaling | `processed/` |
| train | `--stage train` | spatial cross-validation on the training stations, then the final fit | `checkpoints/<run_id>/` |
| geocp | `--stage geocp` | predicts calibration and test stations, picks the GeoCP setting, builds intervals | `predictions_*.csv`, `geocp.json`, `calibration_table.csv` |
| bias | `--stage bias` | Moran's I and LISA on the out-of-fold residuals | `spatial_bias.json`, `lisa_*.png` |
| report | `--stage report` | metrics summary and plots (runs `geocp` and `bias` first if missing) | `metrics_summary.json`, `*.png` |
| infer | `--stage infer --date D` | a prediction and interval for every raster cell, four snapshots a day (runs `geocp` first if it has not been run) | `surface/<D>/` |
| explain | `--stage explain --timestamp "D HH:MM"` | Grad-CAM on the hottest and coldest cells | `explain/<D_HHMM>/` |
| tune | `--stage tune` | Bayesian hyperparameter search with nested spatial CV (slow) | `tuning/<id>/` |
| rasters | `--stage rasters` | builds the Master GeoTIFF from the source layers (one-off) | `processed/rasters/` |
| clean | `--stage clean` (alias `reset`) | **reset**: lists the generated outputs it would delete; add `--yes` to delete them. See [2.13](#213-start-over-reset) | |

Stages that act on a trained run take `--run-id <id>`; without it they use the latest run.

### 2.5 Reading the results

Open `snapshots/<run_id>/metrics_summary.json`:

- **`test`**: RMSE and MAE in µg/m³ and R², on stations that were never used for fitting, tuning or
  calibration. Example from the pilot: RMSE 6.89, MAE 4.12, R² 0.67. `supplementary` holds Willmott's IA and
  Pearson r.
- **`cv_folds`**: the same metrics for each cross-validation fold of the training stations (held-out stations
  per fold). These should be in the same range as `test`; a big gap points at a lucky or unlucky split.
- **`geocp.test`**: `coverage` should be close to `target_coverage` (0.90). On the pilot it is 0.83, which is
  below target because only three stations calibrate the intervals. `mean_width` / `median_width` are in µg/m³;
  `coverage_urban_core` and `coverage_peripheral` split the test stations by distance to the nearest
  calibration sensor.
- **`spatial_bias`**: `global_moran.p_sim` under 0.05 means residuals cluster in space; then
  `lisa_on_bands.png` shows where, laid over each morphology band. With few stations the test has little power.
- **`parameters`** and `run_info.json` (next to the model) list every setting used, so a run can always be
  reproduced. `checkpoints/runs_index.csv` compares all runs in one table.

Plots in the same folder: `loss_curve.png`, `pred_vs_obs.png`, `residual_hist.png`, `timeseries_sample.png`,
`geocp_coverage.png`, `lisa_map.png`.

### 2.6 Make the maps

```bash
python run_pipeline.py --stage infer --date 2026-04-11
```

`infer` needs the GeoCP calibration of the run; if it has not been computed yet (for example right after `train`), `infer` runs `--stage geocp` first and says so. It also needs hourly readings for the three hours before each snapshot. The pilot data covers 2024-08-13 12:00 to
2026-07-03 22:00 (a median of 17 stations report in any hour). To check the range of your own data:

```bash
python -c "import numpy as np, pandas as pd; z=np.load('processed/pm25_hourly.npz'); s=pd.Timestamp(str(z['start'])); print(s, s+pd.Timedelta(hours=len(z['values'])-1))"
```

`--date` expands to the four hours in `configs/inference.yaml` (`snapshot_hours: [8, 12, 17, 21]`);
`--timestamp "2026-04-11 12:00"` (repeatable) gives explicit hours. Results go to
`snapshots/<run_id>/surface/20260411/`:

| File | Contents |
|---|---|
| `prediction_volume.tif` | one band per snapshot (band names are the hours) |
| `lower_volume.tif`, `upper_volume.tif`, `width_volume.tif` | the GeoCP interval, same grid |
| `prediction_daily_mean.tif` | 24-hour mean (a cell missing in any snapshot stays empty) |
| `maps/aqi_0800.png` … `daily_panels.png`, `daily_mean.png` | AQI-coloured maps; cells above 35 µg/m³ outlined, the daily mean outlined at the 50 µg/m³ guideline |
| `output_summary.json` | share of cells per AQI category, above 35 and 50, mean interval width, which basemap was used |

**Basemap.** By default the maps try web tiles first, then offline street/boundary layers, then a plain
background, and each step is skipped quietly if it is not available. The pilot has no street layer configured,
so its maps have a plain background (`basemap used: none`). To get a real background:

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

**Grad-CAM.** `python run_pipeline.py --stage explain --timestamp "2026-04-11 12:00"` needs patches of at least
3 × 3 pixels. The 500 m pilot raster gives 1 × 1, so on the pilot it prints a message and skips. To try it
anyway, widen the patch and pass the **same override to every stage** (the model and the data must agree):

```bash
python run_pipeline.py --stage preprocess --override data.patch.extent_m=2500
python run_pipeline.py --stage train --override data.patch.extent_m=2500 --override train.epochs=3 --override train.batch_size=256
python run_pipeline.py --stage geocp --override data.patch.extent_m=2500
python run_pipeline.py --stage infer --override data.patch.extent_m=2500 --date 2026-04-11
python run_pipeline.py --stage explain --override data.patch.extent_m=2500 --timestamp "2026-04-11 12:00"
```

### 2.7 Tune hyperparameters, then train with them

Normal training uses whatever is in `configs/*.yaml`; nothing adjusts itself. To search for better values:

```bash
python run_pipeline.py --stage tune
python run_pipeline.py --stage train --tuned
```

The search (`configs/tune.yaml`) is a Gaussian-process Bayesian optimisation over learning rate, batch size,
CNN kernel size and filters, LSTM size and recurrent dropout. It uses nested spatial cross-validation: inner
folds guide the search, outer folds (held-out stations) score the tuned model. **It is slow**: every trial
trains one model per inner fold. For a quick demonstration (about 3 minutes on the tutorial machine):

```bash
python run_pipeline.py --stage tune --override tune.n_calls=3 --override tune.n_initial_points=2 --override tune.outer_folds=2 --override tune.inner_folds=2 --override tune.row_stride=40 --override tune.max_epochs=4 --override tune.patience=2
```

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
and re-run `preprocess` first. (On the pilot the baseline is not worse, which is expected with 1×1 patches.)

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
| `explain: patches are 1x1 …` | 500 m pilot raster. Use the 10 m raster or the 5 × 5 trial in [2.6](#26-make-the-maps) |
| `contextily is not installed` warning | optional. Install it, or ignore it, or set `output.basemap.web_tiles=false` to stop looking |
| `CARTO basemap tiles need a free API key` | `CARTO_API_KEY` is empty. Put the key in `model/.env` ([2.6](#26-make-the-maps)) |
| map tiles say `API KEY REQUIRED` | the key is missing, wrong or not accepted (CARTO answers an invalid key like no key). Fix it in `.env`, then delete `.tile_cache` so the watermarked tiles are not reused |
| `ModuleNotFoundError: preprocessing` | you are not inside `model/` |
| `spatial bias: fewer than 6 grid cells` | too few training stations for Moran's I; lower `report.bias.min_cells` knowingly or use more stations |
| training is very slow | use `--override train.epochs=…`, a larger `train.batch_size`, a GPU, or for tuning a smaller `n_calls` / larger `row_stride` |

### 2.13 Start over (reset)

Everything the pipeline generates (trained models, prepared data, results, tuning) is kept in four folders. After
trying the pilot data, clear them before using the real data, because a model, scalers or tuned hyperparameters from
pilot data do not belong with the real one:

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
the pilot raster) are never touched, and any folder outside `model/` is refused. The Master GeoTIFF built by
`--stage rasters` is **kept by default**, because it is slow to rebuild. The folders stay, with their `.gitkeep`
placeholders.

To switch from the pilot to real data: `--stage clean --yes`, point `data.paths` at your real files, build the raster
(`--stage rasters`), then `--stage all`. If you only want to retrain with other settings, you do not need a reset:
every run is kept side by side in `checkpoints/` and compared in `runs_index.csv`.

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
   sensor, 320 m / raster resolution pixels wide (32 at 10 m, 1 on the 500 m pilot), zero-padded at the edge.
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
    metrics_summary.json  geocp.json  calibration_table.csv  predictions_calibration.csv  predictions_test.csv
    spatial_bias.json  residuals_by_cell.csv  lisa_map.png  lisa_on_bands.png  run_info.json
    loss_curve.png  pred_vs_obs.png  residual_hist.png  timeseries_sample.png  geocp_coverage.png
    surface/<YYYYMMDD>/   prediction_volume.tif  lower_volume.tif  upper_volume.tif  width_volume.tif
                          prediction_daily_mean.tif  output_summary.json  maps/*.png
    explain/<YYYYMMDD_HHMM>/   gradcam_hot_cells.png  gradcam_cold_cells.png  gradcam_attention.csv  gradcam_summary.json
```

---

## 5. Known limits

State these when reporting results.

- **The split is spatial only.** Every partition covers the same hours, so a regional weather episode appears in
  train and test. Scores measure generalisation to *new locations*, not forecasting.
- Inside a fold the spatial lag has fewer reference stations, so a held-out station's lag is slightly less
  informative than in production.
- **The pilot raster is 500 m**, so patches are 1×1 pixel and the CNN is nearly an MLP; only 18 of 45 pilot
  sensors fall inside it. Pilot results show the wiring, not the science.
- With few calibration stations GeoCP coverage can fall below target on the test stations (0.83 vs 0.90 on the pilot).
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

`python -m pytest tests -q` runs 208 tests, 204 of which apply when `contextily` is installed (about 1 min), covering: configuration and the CRS check; the reset command; the infer / geocp workflow; shapes and
gradient flow into both branches (pool and flatten); trainer and checkpoint round trips; metrics against
hand-computed values; gap capping; patch/window alignment checked by hand; partitioning; the spatial lag
(hand-computed, leakage, per-fold); scalers; GeoCP (normalisation, clipping, Table 4.2 metrics, zones, grid
vs. exact quantiles, interval rasters); tuning (search, nested folds, epoch switch); Moran's I / LISA (sign
convention, significance); the temporal-only baseline; checkpoint compatibility; per-cell inference (vectorised
patches equal the training extraction, stand-in hand-computed); output volumes, AQI categories and maps
(including pixel comparisons for the basemap switches); Grad-CAM (localisation, determinism, hooks removed).

End-to-end runs on the pilot data covered preprocess → tune → train `--tuned` → geocp → bias → report → infer
→ explain, a temporal-only run compared with `--baseline`, and both CRS versions.

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

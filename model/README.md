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
- for any day: 24 hourly snapshot surfaces (the whole day; `inference.snapshot_hours`) with intervals, a 24-hour mean, an animation and a slideshow, and maps on
  the Philippine AQI scale; Grad-CAM saliency of the CNN on the hottest and coldest cells.

> **Status.** Runs on the real data in the repo's `data/` folder: a 10 m Master GeoTIFF (EPSG:32651, 5 bands,
> 27 × 39 km) and the OpenAQ sensors (44 stations with readings, 2024-08-13 to 2026-10-15). The pipeline runs end to
> end and 500 tests pass. The numbers quoted in this file come from short smoke runs (2 epochs), so they show that the
> wiring works, not what the model can reach. Three things to know before relying on results: the raster stops short of
> the southern tip of the NCR ([2.14](#214-covering-the-whole-ncr)), a full-resolution city map takes hours on a CPU
> ([2.6](#26-make-the-maps)), and the pipeline has not been trained to convergence on this data yet.

## Contents

0. [The study in order: from data collection to results](#the-study-in-order-from-data-collection-to-results) (read this first for the story; sections 1 to 8 are the manual and the reference)
1. [Quick start](#1-quick-start)
2. [Tutorial: running it](#2-tutorial-running-it)
3. [How it works](#3-how-it-works)
4. [Outputs reference](#4-outputs-reference)
5. [Known limits](#5-known-limits)
6. [How it was checked](#6-how-it-was-checked)
7. [Which thesis text became which code](#7-which-thesis-text-became-which-code)
8. [Findings so far and the transfer-learning plan](#8-findings-so-far-and-the-transfer-learning-plan)

---

## The study in order: from data collection to results

This section follows the work in the order it happens: **collect → clean and align → choose stations → train and compare → transfer learning → results**. Each step says what was done, why,
the numbers, and where the detail lives. The rest of the file (sections 1 to 8) is the manual and the reference; read this part first when you need the story for the thesis or the panel.

| Step | What happens | Where |
|---|---|---|
| A | Collect the Manila sensor data and the other cities' data | A.1 to A.3 |
| B | Clean it and put it on a strict one-hour grid | B.1 to B.3 |
| C | Choose stations: Manila's fixed test stations, the source stations | C.1, C.2 |
| D | Train and compare the four Manila models, the learning curve | D |
| E | Transfer learning: the pools and how the winner is chosen | E |
| F | Results so far, and what is still to run | F |
| G | Which manuscript sections each step feeds | G |

### A. Data collection

**A.1 Manila (target).** Two networks, combined into one dataset (`preprocessing/combine_dataset.py`, details in [2.1b](#21b-the-quezon-city-sensors-qc-data) and
[2.1c](#21c-the-combined-cleaned-dataset-both-networks-three-year-ranges)): OpenAQ stations and the Quezon City (QC) network. The dataset used for every result so far is
`from_2023_with_preinstall`: 85 stations, 1,216,537 readings, 2023-01-01 to 2026-10-15 for OpenAQ. **The QC files end on 2025-12-31**, so Quezon City stations have no readings in 2026. To bring the Manila data
up to a later date run `update_manila_data.ipynb` (backs up the old folder, rebuilds the combined dataset, checks that the stations are the same); retrain every model afterwards, because
results on a longer range are a different experiment from the earlier ones.

**A.2 Other countries (sources for transfer learning).** `other-country-scrapers/scrapers.ipynb`, OpenAQ API v3 only. It looks at the **capital region** of each country (the counterpart of Manila's NCR): Hanoi,
Bangkok and its metropolitan ring, Jakarta and Greater Jakarta, Kuala Lumpur and the Klang Valley, plus Ho Chi Minh City as the largest Vietnamese city. Range 2023-01-01 to 2026-11-19 (`END` is exclusive: it
includes all of 18 November). The API key comes from `.env` (never committed). Things that made the download robust:
- each sensor is fetched in **60-day windows**, each saved on its own, so a stop or an error resumes at that window; a window that the server answers with HTTP 408 (timeout) is halved and retried;
- moving `END` later fetches only the missing days and appends them;
- one failing sensor is skipped and listed, not fatal; the download cell is run again to retry it.

**A.3 What was not used.** The Zenodo "HCMC / Bangkok hourly" files are CAMS model output, not measurements. Details and the other source-by-source critique: [2.21](#221-the-transfer-learning-reference-transferlearnpdf-what-holds-and-what-to-change).

### B. Cleaning and the one-hour grid

**B.1 Cleaning rules (before any alignment).** A reading is removed when it is not a number, at most 0.5 µg/m³ or negative (dropout), above 1,000, in the New Year's Eve fireworks hours, a QC reading from
before the site's installation date (the kept-or-removed variants are the `*_with_preinstall` folders), or an isolated single-sensor spike. Each removed reading is listed with its reason in
`excluded_readings.csv` (149 not a number, 5,352 at or below the floor, 176 above 1,000, 1,519 fireworks, 20,708 before installation, 451 spikes).

**B.2 Putting every station on a strict one-hour grid** (`preprocessing/clean_loader.py`, settings `data.time` in `configs/data.yaml`). The data is not exactly hourly: sensors miss hours or report
more than once in an hour. What the pipeline does, in order:
1. **Hourly from the source.** The OpenAQ `/hours` endpoint already gives one value per hour (with a coverage percentage). The scraper labels an hour by its **start** (`time_label = "from"`, an assumption
   not yet checked against the OpenAQ portal).
2. **A complete hourly index per sensor** (`resample`, 1 h). If an hour has several readings the **mean** is taken; if it has none it is missing.
3. **Short gaps are filled by linear interpolation: gaps of 24 hours or less** (`data.time.max_gap_hours: 24`).
4. **Long gaps stay missing** (more than 24 hours): nothing is made up across a long outage.
5. **A sample that touches a missing hour is dropped** (B.3).
6. The hour of day is the city's own **local** hour (each source city is converted from UTC to its time zone).

**How much was interpolated** (computed on `from_2023_with_preinstall`, 85 stations, from each station's first to its last reading):

| | Station-hours | Share of the grid |
|---|---|---|
| Hourly slots on the grid | 1,404,039 | 100 % |
| Measured | 1,240,019 | 88.3 % |
| **Interpolated** (gap of 24 h or less) | 154,853 | **11.0 %** |
| Left missing (gap of more than 24 h) | 9,167 | 0.7 % |

There are 70,738 gaps: 70,647 are 24 hours or less and 91 are longer; the median gap is 1 hour and half of all gaps (37,286) are a single hour. The share varies by station (median 10 %, worst 36 %; 43 of 85 stations
above 5 %). Interpolated hours are also in the targets, so a similar share of what is scored on the test stations is interpolated, not measured. A sensitivity check with a shorter cap (for example 3 hours)
has **not been run**; do it if a panelist asks whether the results depend on the interpolation.

**B.3 Why a sample with a missing hour is dropped, not filled.** The model reads the last three hours of PM2.5 (and the spatial lag, the inverse-distance-weighted mean of the five nearest stations at those hours) to
predict the next hour. If the target, one of those three hours, or the neighbours' values at those hours are missing, there is no complete sample, and we leave it out rather than guess. A model trained on made-up
values over long outages would learn our fill instead of PM2.5. Example: a sensor that is down for 3 days loses every sample whose target or whose three previous hours fall in those 72 hours. Counts on the same
dataset: 1,394,354 samples have a complete own history (0.7 % of station-hours lost to missing hours); the final number is **1,360,277** (1,060,985 train, 299,292 test), 34,077 fewer (2.4 %) because the
neighbouring stations had no data at those hours (the spatial lag). In total 3.1 % of the station-hours do not become a sample.

### C. Choosing stations

**C.1 Manila: the same test stations for every result.** The split is by station, never by row or time. The partition behind all results so far is 54 training, 13 calibration and 17 test stations
(`model/configs/pinned_partition_2026-11.json`; 16 of the 17 are scored, as in every earlier result). `PIN_PARTITION` in `training_cells.ipynb` and `transfer_learning.ipynb` passes those stations to every stage,
so a longer or updated dataset is scored on the same stations. If a pinned station is missing from new data, preprocessing stops and names it.

**C.2 The source stations, and why not all of them.** Written out in [2.20, "Which source stations are used, and why not all of them"](#which-source-stations-are-used-and-why-not-all-of-them): capital regions only; the
quality rules (at least 6,000 hours, 12 calendar months, 50 % of the hours present); the requirement that neighbours report in the same hours (Kuala Lumpur's two stations never overlap, so that city gives no
window); and the four pools that are *tested* rather than assumed.

### D. Training and comparing the Manila models

Four models, one dataset (CNN-LSTM and LSTM-only, with and without GeoCP): `training_cells.ipynb` ([2.15](#215-four-models-on-kaggle-training_cellsipynb)), findings in [8.1](#81-what-the-four-models-showed).
Then the learning curve ([2.16](#216-learning-curve-learning_curveipynb), result in [8.5](#85-result-of-step-1-the-learning-curve-run-locally-10-to-12-november-2026)). Both say the same: **the CNN branch adds no
measurable accuracy over the temporal model** at 20 to 67 training stations. State it as "did not improve accuracy at these station counts", not "morphology has no effect on PM2.5". The no-lag diagnostic
([2.19](#219-the-no-pm25-history-diagnostic-does-the-morphology-carry-signal-on-its-own)) is built and tested and **not run yet**.

### E. Transfer learning

Because the skill is in the temporal branch, an LSTM-only model is **pretrained on the other cities' PM2.5** and then fine-tuned on Manila (`transfer_learning.ipynb`, reference [2.20](#220-transfer-learning-of-the-temporal-side-other-cities--manila)).
Checkpoints: `manila_only` (baseline), `single_<country>`, `mmd_top_10`, `all_sources`; each used as `finetune`, `head_only` or `zero_shot`. With `USE_CV = True` (the default) the pool is **chosen by spatial
cross-validation on the Manila training stations** and its test R² is read once afterwards, so the test stations play no part in the choice. A failing job is listed and skipped, finished jobs are skipped on a re-run, and
a new Kaggle session resumes from the saved output (the run cell prints `RESUMING: x of y jobs are already finished`).

### F. Results so far

**Manila models (test R², same 16 test stations).** CNN-LSTM 0.707 to 0.721, LSTM-only 0.718 to 0.724 across the four variants; previous-hour baseline 0.646 ([8.1](#81-what-the-four-models-showed), [8.5](#85-result-of-step-1-the-learning-curve-run-locally-10-to-12-november-2026)).

**Transfer learning, first Kaggle run (20 November 2026), pool `single_indonesia` only** (5 Jakarta stations). Cross-validation is the 3-fold spatial CV on the Manila training stations.

| checkpoint | variant | CV R² | test R² | test RMSE | test MAE | folds won vs `manila_only` |
|---|---|---|---|---|---|---|
| `manila_only` | from scratch | 0.7174 | 0.7267 | 5.349 | 3.329 | n/a |
| `single_indonesia` | `finetune` | 0.7173 | 0.7258 | 5.358 | 3.334 | 1 of 3 |
| `single_indonesia` | `head_only` | 0.7062 | 0.7102 | 5.508 | 3.409 | 0 of 3 |
| `single_indonesia` | `zero_shot` | not cross-validated | 0.6465 | 6.083 | 4.036 | n/a |

Reading: pretraining on Jakarta **does not improve** on Manila-only (a difference of −0.0001 in CV R² and −0.0009 in test R² is nothing); freezing the pretrained LSTM and training only the head is worse; with no Manila
training at all the model reaches R² 0.646, which is **no better than repeating the previous hour** (RMSE 6.08 against 6.08). This is one pool and one pass (Jakarta has 5 stations); do not generalise from it.
The Malaysian pool failed because Kuala Lumpur gives no windows (B.3, C.2). **Still to run:** `single_thailand`, `single_vietnam`, `mmd_top_10`, `all_sources`; the no-lag diagnostic; a 24-hour map and Grad-CAM on real data;
the sensitivity check of B.2. If every pool matches `manila_only`, the finding is "transfer from other Southeast Asian capitals did not help, in this setup", which is a result.

### G. Which manuscript sections each step feeds

A is 4.1.4 (source-domain data acquisition) and the data description of chapter 4; B is 4.2 (cleaning, harmonisation); C is 4.2.6 and 4.3.4; D is 4.3.5, 4.4.4 and 4.4.5; E is 4.3.4 and 4.4.7; F is the results chapter.
The full list with search keywords is in [8.7](#87-manuscript-sections-to-add-to-the-table-of-contents).

---

## 1. Quick start

```bash
cd model
pip install -r requirements.txt
python -m pytest tests -q
python run_pipeline.py --stage coverage
python run_pipeline.py --stage all --override train.epochs=2 --override train.batch_size=256
python run_pipeline.py --stage infer --date 2025-07-01 --override inference.stride_px=21
```

`coverage` checks in a second that the raster covers your sensors. The `all` line runs the data through
preprocessing, training, GeoCP, the bias check and the report in about 10 minutes on a CPU (2 epochs; the defaults
train much longer). The `infer` line maps the whole raster at a coarse 210 m grid in about 2 minutes; the full 10 m
grid takes hours (see [2.6](#26-make-the-maps)). Results are in `snapshots/<run_id>/` and the maps in
`snapshots/<run_id>/surface/20250701/maps/`.

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

### 2.1c The combined, cleaned dataset (both networks, three year ranges)

`python -m preprocessing.combine_dataset` (from `model/`) merges OpenAQ and QC into one cleaned table, using the findings of
`eda/eda_results/EDA_Report.md` and an audit of the files, and writes `data/Combined Dataset/` (details and counts in the `README.md`
generated there). Readings are removed, and listed with their reason in `excluded_readings.csv`, when they are: empty; at or below
0.5 µg/m³ (dropouts, negatives); above 1000; inside the New Year fireworks window (Dec 31 21:00 to Jan 1 06:00); a QC reading before
the site's installation date (it looks backfilled); or a lone spike (above 150 and more than 5 times the mean of the other stations
that hour, which is what 85 % of the remaining readings above 150 are). Nothing is interpolated or rescaled.

| Folder | Starts | Meaning |
|---|---|---|
| `full` | 2021-07-09 | everything that passes the rules |
| `from_2023` | 2023-01-01 | drops 2021-2022, when only 8 to 20 stations existed and few reported together |
| `from_2024` | 2024-01-01 | drops 2021-2023; from here 40 QC stations run together |
| `full_with_preinstall`, `from_2023_with_preinstall`, `from_2024_with_preinstall` | as above | the same ranges but KEEPING the QC readings before each site's installation date |

The `*_with_preinstall` twins exist so that rule can be tested as an experiment (it removes 20,708 readings, 18,596 of them in Jan-Mar
2024, so the year ranges alone do not remove them). Train once on a range and once on its twin with the same settings; the stations, and
so the spatial split, are identical. Compare on the SAME rows: `--stage report --run-id <twin run> --baseline <clean run>` scores only
the rows both have. Scoring the twin on its own rows would flatter it, because the kept pre-installation readings follow the network
median closely and are easy to predict. `python -m preprocessing.combine_dataset --only-clean` skips the twins.

**Training on Kaggle.** `docs/KAGGLE_GUIDE.md` is the cell-by-cell guide (upload the data, clone the code, run the stages, split the work across
sessions); every command in it was run once in a local stand-in for Kaggle.

**Comparing the datasets.** `docs/EXPERIMENT_CHECKLIST.md` is the step-by-step plan (train on each, pick by out-of-fold scores, then tune).
Two pieces support it. `data.split.cv_assignment: by_station` (default `balanced`) gives every station its CV fold from a seeded shuffle of the
station ids, so every dataset validates the same station groups (the default balances folds by rows, which moves stations between folds
whenever the data changes). `python -m reporting.compare_runs --runs-root runs` scores all runs on the (station, hour) rows they share, with the
mean and sd over seeds, the gap to the best dataset with a station-bootstrap confidence interval, and a check that all runs used the same folds.
It uses out-of-fold predictions only, never the test partition.

Each folder has `combined_hourly.csv` (station, coordinates, `network`, cleaned PM2.5, hour encoding) and the per-sensor files plus
`locations.csv` that the pipeline reads. To train on a range, set `sensor_locations_csv` to its `locations.csv` and `sensor_csv_dir`
to its `sensors/` folder in `configs/data.yaml`, then re-run `preprocess`. The two networks read differently where both operate
(median hourly network mean 13.2 against 17.2 µg/m³) and cannot be calibrated against each other, so the values are left as measured.

### 2.2 Check the install

```bash
python -m pytest tests -q
```

Expect `416 passed` (and 4 skipped, which only apply when `contextily` is not installed) in about a minute. If something fails here, fix it before running anything else.

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
| dataset | `--stage dataset` | the dataset report: sensors and coverage, rows kept and dropped, missingness, temporal availability, the PM2.5 distribution, the distributions of the five morphology bands | `dataset_report/` next to `processed/` |
| train | `--stage train` | spatial cross-validation on the training stations, then the final fit | `checkpoints/<run_id>/` |
| geocp | `--stage geocp` | predicts calibration and test stations, picks the GeoCP setting, builds intervals | `data/predictions_*.csv`, `data/geocp.json`, `data/calibration_table.csv` |
| bias | `--stage bias` | Moran's I and LISA on the out-of-fold residuals | `data/spatial_bias.json`, `plots/lisa_*.png` |
| report | `--stage report` | accuracy report and plots; prints the headline (runs `geocp` and `bias` first if missing) | `SUMMARY.md`, `metrics_summary.json`, `plots/`, `tables/` |
| infer | `--stage infer --date D` | a prediction and interval for every raster cell, 24 snapshots a day by default (runs `geocp` first if it has not been run) | `surface/<D>/` |
| explain | `--stage explain --date D` (or `--timestamp "D HH:MM"`, repeatable) | Grad-CAM on the hottest and coldest cells of every snapshot hour, and (default) their aggregate | `explain/<D_HHMM>/`, `explain/<D>_all/` |
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
| 5. By year | the test accuracy per calendar year, with how many OpenAQ and QC stations were scored that year (`tables/accuracy_by_year.csv`); read a year with the stations column in view, because the network mix changes (QC has no readings from 2026) |
| 6. Uncertainty | which GeoCP weighting was chosen, coverage against target, interval width, close to versus far from a calibration sensor |
| 7. Spatial bias | Moran's I of the residuals in one sentence |
| 8. Training | the folds and the final fit |
| 9. Read with care | automatic warnings: a short run, few test stations, intervals under target, no better than persistence, unverified AQI limits |
| 10. Files | what every file in the folder is |

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
    tables/                accuracy_summary.csv  accuracy_by_station.csv  accuracy_by_year.csv  cv_folds.csv     (open in a spreadsheet)
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
python run_pipeline.py --stage infer --date 2025-07-01
```

`infer` needs the GeoCP calibration of the run; if it has not been computed yet (for example right after `train`), `infer` runs `--stage geocp` first and says so. It also needs hourly readings for the three hours before each snapshot. The data covers 2024-08-13 12:00 to
2026-10-15 18:00 (44 stations). To check the range of your own data:

```bash
python -c "import numpy as np, pandas as pd; z=np.load('processed/pm25_hourly.npz'); s=pd.Timestamp(str(z['start'])); print(s, s+pd.Timedelta(hours=len(z['values'])-1))"
```

**How long a map takes.** `infer` predicts every cell of the raster: 10.6 million cells at the full 10 m grid. On a
4-thread CPU the model does roughly 300-800 cells/s (it varies with load), so one full-resolution snapshot takes
**4 to 10 hours**. A day of snapshots costs much less than the number of hours times that: every chunk of cells computes its CNN
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

**Dates from 2026 (OpenAQ-only history).** The Quezon City sensors have no readings from 2026, and a cell's history comes from its five
nearest training stations. Where all five are Quezon City sensors there would be no history and no prediction (about 12 % of the cells,
around Quezon City, on 2026-04-11). `inference.openaq_only_from_year: 2026` (the default) therefore builds the history of every hour of
2026 or later from the nearest **OpenAQ** training stations only, so every cell is predicted. Hours before 2026 are unchanged. `infer` says
which rule it used (a line when it starts, and `lag_stations` in `output_summary.json`), and `explain` follows the same rule. To go back to the
old behaviour use `--override inference.openaq_only_from_year=null`; to move the switch use another year. Read these maps with two cautions:
the five stations are farther away than in 2025 (OpenAQ is the sparser network: 32 of the 54 training stations in the `from_2024` dataset), and the model never saw a history made of
OpenAQ stations alone. The by-year table in `SUMMARY.md` (section 5) shows how the test stations fared in 2026 (OpenAQ stations only).
The option affects maps and Grad-CAM only: it does not change training, preprocessing or the test rows.

`--date` expands to the hours in `configs/inference.yaml` (`snapshot_hours`, **all 24 by default**; `[8, 12, 17, 21]` gives the four peak hours, six times faster);
`--timestamp "2025-07-01 12:00"` (repeatable) gives explicit hours. Results go to
`snapshots/<run_id>/surface/20250701/`:

| File | Contents |
|---|---|
| `prediction_volume.tif` | one band per snapshot (band names are the hours) |
| `lower_volume.tif`, `upper_volume.tif`, `width_volume.tif` | the GeoCP interval, same grid |
| `prediction_daily_mean.tif` | 24-hour mean (a cell missing in any snapshot stays empty) |
| `maps/aqi_0800.png` … `daily_panels.png`, `daily_mean.png` | maps on the Philippine AQI bands (no outline; `output.outline_above_threshold: true` draws one around cells above 35 µg/m³, and 50 µg/m³ on the daily mean) |
| `maps/continuous_0800.png` … `continuous_panels.png`, `continuous_daily_mean.png` | the same predictions on a continuous viridis scale with a colourbar; one scale for all hours of the day |
| `output_summary.json` | share of cells per AQI category, above 35 and 50, mean interval width, which basemap was used, cells and coverage of every map, which map styles were drawn |

**How the maps are drawn.** Both styles are clipped to the National Capital Region and carry what a thesis figure needs: the 17 city outlines and names, a scale bar, a north arrow, easting and northing in km, the stations, a legend with value ranges and a grey **No data** swatch, panel titles of the form `12:00 (12,322 cells, coverage 1.00)`, a few lines saying what the numbers mean, and a caption (date, run, cell size, boundary source).

- **cells** = grid cells inside the NCR that have a prediction; **coverage** = their share of all cells inside the NCR. A hole (for example the Quezon City cells without history on a 2026 date, if the OpenAQ-only rule is switched off) shows up as coverage below 1.
- **The clip is display only**: the GeoTIFFs and the other numbers in `output_summary.json` keep every cell of the raster.
- **The boundary** is `data/Boundaries/ncr_cities.gpkg`, made once with `python -m preprocessing.boundaries --src ../data/Boundaries/raw/gadm41_PHL_2.json --out ../data/Boundaries/ncr_cities.gpkg` from the free GADM 4.1 level-2 file (<https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_PHL_2.json.zip>, academic and non-commercial use, not redistributable, so `data/` stays out of git). It must contain exactly the 17 NCR cities. Without the file the maps still work, unclipped and without city names, with one warning.
- **Settings** (`output.maps` in `configs/output.yaml`, each also as `--override`): `styles` (`[aqi, continuous]`, or one of them), `clip_to_boundary`, `city_labels`, `scale_bar`, `north_arrow`, `caption`, `metric_notes`, `continuous_cmap`, `continuous_range` (default `[0, 25]` µg/m³: ONE fixed scale shared with the XGBoost maps, so both models' colours compare directly; `auto` = the 2nd to 98th percentile of the day inside the NCR), `look` (default `xgboost`: the AQI and continuous maps are drawn like the XGBoost branch's: raw cells instead of a smoothed surface, see-through continuous surface, XGBoost's two-line titles, colourbar numbers every 5, the outline above 35 on the AQI maps; `default` = the older smoothed CNN-LSTM look). The XGBoost branch writes, next to its plain 500 m cell maps, `<stem>_continuous_ncr(_smooth).png`, `<stem>_aqi_ncr(_smooth).png` and `panel_<day>_ncr(_smooth).png`: the same maps cut along the NCR boundary with city outlines and names (`xgboost/maps/ncr_maps.py`).

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

- offline: set `data.paths.streets` in `configs/data.yaml` (an OSM street file; the NCR city outlines come from `data.paths.boundary`, which is set by default);
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

**Grad-CAM.** `python run_pipeline.py --stage explain --date 2025-07-01` explains every snapshot hour of the day (the ones
`infer --date` made: 08:00, 12:00, 17:00, 21:00) and, by default (`output.explain.aggregate`), pools them into
`explain/20250701_all/`: the band profile, SHAP-style swarm, importance and ablation figures of all hours together,
`gradcam_band_by_hour.png` (mean attention per band for each hour and for all hours), and the hot and cold cells of the
*average* of the day's snapshots with the Grad-CAM averaged over the hours. A finished hour is reused
(`output.explain.redo: true` redoes it). `--timestamp "2025-07-01 12:00"` (repeatable) explains single hours the same way, and
works on a snapshot made by `infer` for that date. It needs patches of at least 3 × 3 pixels (320 m / raster resolution), which the 10 m raster
gives (32 × 32). It writes four figures into `explain/<D_HHMM>/`:

| File | Shows |
|---|---|
| `gradcam_map.png` | the predicted PM2.5 of that hour on the NCR map, with the explained **hot** cells (red circles) and **cold** cells (blue squares), numbered |
| `gradcam_hot_cells.png`, `gradcam_cold_cells.png` | one row per numbered cell: every morphology band (with units) with the attended region outlined in orange, a 100 m scale, and the Grad-CAM map with its colourbar |
| `gradcam_band_profile.png` | hot against cold: for each band, how much higher or lower the attended pixels are than the rest of the patch, averaged over the explained cells |
| `morphology_sheet.png` | the five raster bands mapped over the NCR in quantile classes with range legends (what the CNN branch gets to see), with the explained cells marked |
| `gradcam_band_swarm.png`, `gradcam_band_importance.png`, `gradcam_band_swarm.csv` | the SHAP-style summary of the Grad-CAM band scores over a sample of NCR cells (below): a beeswarm and a bar chart |
| `feature_groups/` | the hot and cold cells of the **high** and the **low** end of each band over the whole NCR (below): `map_<band>_<level>.png`, `gradcam_<band>_<level>_<high\|low>.png`, `feature_group_comparison.png`, `feature_group_summary.csv`, `feature_group_cells.csv`, `feature_group_attention.csv`, `feature_group_cities.csv` |

**Hot and cold** are the highest- and lowest-predicted cells of that hour (`output.explain.top_n` / `bottom_n`, six each). They are ranked inside the NCR and no two may be closer than `output.explain.min_separation_m` (1,000 m), so they are different places and not one hotspot's neighbours. **Grad-CAM** shows which pixels of a cell's 320 m patch drove its prediction; the orange line encloses the top 20 % of that attention (`output.explain.attention_quantile`). Attention shows where the model looked, not why PM2.5 is high: read it next to the morphology sheet. Note that with a coarse `stride_px` the hot and cold cells are chosen from that coarse map.

**Band swarm** (`output.explain.band_swarm`, on by default: `n_cells: 300`, `seed`, `chunk`). The Grad-CAM counterpart of a SHAP summary plot, over many cells instead of the hot and cold ones. `explain` samples `n_cells` cells across the NCR, runs Grad-CAM on each patch and keeps, per band, the attention score of the band profile ((band on the attended pixels − patch average) / patch standard deviation). `gradcam_band_swarm.png` is drawn exactly like the XGBoost `shap_beeswarm.png` (same size, fonts, dotted rows, red-blue colours, Low/High colourbar; no notes on the figure, put the explanation in the caption) and has one row per band, ordered by mean |score|, and one dot per cell: x = the signed score (right = the CNN looked at the higher values of the band in that patch), colour = the band's value in the patch (blue low, red high, SHAP's colours). `gradcam_band_importance.png` is the bar version, the mean |score| with a 95 % bootstrap interval; `gradcam_band_swarm.csv` holds every cell's scores and patch means. It shows where the CNN looks and what lies under it, relative to each patch; it is not an effect in µg/m³ like a SHAP value and not a cause.

**Feature groups** (`output.explain.feature_groups`, on by default, written to `explain/<hour>/feature_groups/`). Besides the plain hot and cold cells, `explain` asks the same question for the high and the low end of every morphology band. A cell is in the *high* group of a band when the **raster** value of its 320 m patch (the patch mean, NoData ignored) is at or above an NCR percentile, in the *low* group when at or below the low percentile; only inside a group does the prediction pick the **hot** cells (highest predicted PM2.5 of that hour) and **cold** cells (lowest), spread by `min_separation_m`. Two levels are on by default: `zone` (top and bottom 10 %) and `extreme` (top and bottom 0.1 %); add more under `levels`. Per band and level you get an NCR map of the two ends with the numbered picks (circles = high end, diamonds = low end), a Grad-CAM close-up sheet for each end, and across bands `feature_group_comparison.png` (mean predicted PM2.5 in the high against the low end) and `feature_group_cities.csv` (per city: share of its cells in each end and the mean prediction there). This is an **association**: it shows what the model predicts and attends to where a feature is high or low, not what would happen if the feature were changed. When many cells share the limit value (e.g. a band that is 0 over water) a group is larger than its percentile says; `feature_group_summary.csv` gives the true size.

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
| no GeoCP at all (no calibration stations, no intervals in `all` / `report` / `infer`; `--stage geocp` and `--stage surface` still run it when called) | `geocp.enabled` | `true`, `false` |
| continue a stopped training run from its last finished CV fold | `train.resume` | `false`, `true` |
| train on only n of the training stations (learning curve; the test stations never change; see `learning_curve.ipynb`) | `data.split.train_subset.n_stations` (+ `.seed`) | `null`, `20`, `40` … |
| Grad-CAM of the hours pooled into `explain/<D>_all/` | `output.explain.aggregate` | `true`, `false` |
| neighbours used for map cells | `inference.lag_reference` | `train`, `all` |
| OpenAQ-only history from a year on | `inference.openaq_only_from_year` | `2026`, `null` (off) |
| predict a coarser grid | `inference.stride_px` (odd) | `1`, `3`, `5` … |
| which hours of a date get a map (and the animation) | `inference.snapshot_hours` | all 24 (default), `[8, 12, 17, 21]` |
| which hours Grad-CAM explains (slow) | `inference.explain_hours` | `[]` (default) = every snapshot hour, `[8, 12, 17, 21]` |
| the day animation / slideshow | `output.animation.enabled` / `.fps` / `.formats` | `true`, `2`, `[gif, mp4, html]` |
| train without any PM2.5 history (the morphology diagnostic, [2.19](#219-the-no-pm25-history-diagnostic-does-the-morphology-carry-signal-on-its-own)) | `data.temporal.use_pm25_lags` | `true` (default), `false` |
| start training from pretrained weights / keep parts fixed (transfer learning, [2.20](#220-transfer-learning-of-the-temporal-side-other-cities--manila)) | `train.init_from`, `train.freeze` | `null` / `[]` (default), a `.pt` path / `[temporal]` |
| pretrain on other cities (`--stage pretrain`) | `transfer.sources`, `.top_k_stations`, `.exclude_months`, `.epochs` | see `configs/transfer.yaml` |
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

**To measure it properly,** `data.paths.boundary` points at the NCR city polygons (`data/Boundaries/ncr_cities.gpkg`, built as in [2.6](#26-make-the-maps)). The report then
also gives the share of the NCR area the raster covers and how many km are missing on each side (with the GADM 4.1 boundary: 93.7 % covered, 5.5 km missing in the south and 3.3 km in the north). A missing file is only a note, not an error.
`data.coverage.strict: true` makes `preprocess` stop while coverage is incomplete.

Two other facts about the raster: it is a plain rectangle with data everywhere (no nodata outside the NCR), so city
maps also show Manila Bay and the neighbouring provinces; and far from any sensor a cell's prediction is mostly its
nearest stations' values (the spatial-lag stand-in of [3.9](#39-city-wide-inference-and-maps)).

### 2.15 Four models on Kaggle: `training_cells.ipynb`

The notebook at the repository root trains and compares four models of one dataset, in one go and resumably:

| name | CNN branch | GeoCP | training stations |
|---|---|---|---|
| `cnn_nogeocp` | yes | no | 67 (no calibration set) |
| `nocnn_nogeocp` | no (LSTM only) | no | 67 |
| `cnn_geocp` | yes | yes | 54 + 13 calibration |
| `nocnn_geocp` | no (LSTM only) | yes | 54 + 13 calibration |

Cell 1 also sets the hours: `INFER_HOURS` (default all 24: the maps, the animation and the slideshow of each date) and `EXPLAIN_HOURS` (default all 24: the hours Grad-CAM explains, among `INFER_HOURS`; `SHOW_HOURS` picks which of them the figures cell displays).
A date mapped earlier with other hours is mapped again. An optional diagnostic pair, `cnn_nolags` and `nocnn_nolags` (no PM2.5 history, see [2.19](#219-the-no-pm25-history-diagnostic-does-the-morphology-carry-signal-on-its-own)), can be added to `RUN_VARIANTS`.

`RUN_VARIANTS` (cell 1) lists the models to run, in order (the most important first); `MAPS_FOR` lists the ones that also get maps and
Grad-CAM (default `cnn_nogeocp`; they need the CNN branch). Comparing a row of the table says what the morphology adds, comparing a column
says what holding the calibration stations out costs. Each model has its own `checkpoints_<name>/` and `snapshots_<name>/` under
`runs/<dataset>/`; preprocessing is shared by the two models with the same GeoCP setting (`processed_geocp/`, `processed_nogeocp/`). A
GeoCP-trained run left by an older version of the notebook (`runs/<dataset>/{processed,checkpoints,snapshots}`) is reused as `cnn_geocp`.
After the runs, cell 5 writes `model_comparison.csv`: CV R², test R², RMSE, MAE, and the R² the CNN adds within each GeoCP setting.

**Turning GeoCP off** is `geocp.enabled: false` (`configs/geocp.yaml`; the notebook sets it per model). Then no calibration stations are
picked (the stations that would have been calibration stations train the model; the test stations are the same), the `geocp` stage inside
`all` only writes the test predictions that `report` needs, `infer` writes `prediction_volume.tif` without the lower / upper / width volumes,
and `SUMMARY.md` says GeoCP was not run. `--stage geocp` and `--stage surface` still run GeoCP when you call them yourself. The test rows
differ slightly between the two settings (287 k with GeoCP, 295 k without) because a larger training set leaves fewer windows without a usable
lag, so compare the CNN effect within a GeoCP setting.

**The 12-hour limit and resuming.** `TIME_BUDGET_H` (default 11) makes the notebook stop by itself: every stage gets
`train.runtime.max_hours` for the time left, and a stage that runs out stops cleanly (exit code 124), after which the notebook prints where it
stopped. Training saves every finished cross-validation fold (`fold_<k>_result.json`, `fold_<k>_oof.csv`), and `train.resume: true` (set by the
notebook) continues the newest run that was stopped (status `running` / `timed_out`) from its first unfinished fold; the fold that was running
starts again. To continue in another session: *Save Version*, then *Add Input* → this notebook's output (and the data), and run all: finished
steps are skipped. Changing a model or training setting between sessions makes a finished run count as stale and be retrained
(`RETRAIN_IF_CONFIG_CHANGED`); do not change them while a run is half done. To redo something on purpose put its name in `REDO`.

**Grad-CAM over the explained hours.** `python run_pipeline.py --stage explain --date 2025-07-23` explains the hours in `inference.explain_hours` (default `[]` = every snapshot hour, so **all 24** of the day; see [2.18](#218-a-whole-day-of-maps-an-animation-and-a-slideshow)) (`explain/20250723_0800/` … `_2100/`) and, by default (`output.explain.aggregate`), pools them into `explain/20250723_all/`:

| File | Shows |
|---|---|
| `gradcam_band_profile.png`, `gradcam_attention.csv` | hot against cold cells, every hour's cells pooled |
| `gradcam_band_swarm.png`, `gradcam_band_importance.png`, `gradcam_band_swarm.csv` | the SHAP-style swarm and bars, the sampled NCR cells of all hours pooled (cell-hours) |
| `band_ablation.png`, `band_ablation_swarm.png`, `band_ablation.csv` | the impact counterpart (swap a band, see the prediction move), pooled |
| `gradcam_band_by_hour.png`, `.csv` | mean \|attention\| per band for each hour and for all hours: are the hours alike? |
| `gradcam_hot_cells.png`, `gradcam_cold_cells.png`, `gradcam_*_row.png`, `gradcam_map.png` | the hot and cold cells of the **average of the explained hours**, with the Grad-CAM **averaged over those hours** |

A finished hour is reused (`output.explain.redo: true` redoes it); `output.explain.aggregate: false` gives the hours only. `--timestamp` (repeatable)
explains single hours the same way. Nothing is recomputed for the pooled figures: they read the tables the hours already wrote.

### 2.16 Learning curve: `learning_curve.ipynb`

The question: **does the CNN branch pay off as the number of training stations grows?** It is the first step of the transfer-learning plan
([8](#8-findings-so-far-and-the-transfer-learning-plan)). The notebook trains the CNN-LSTM and the LSTM-only model on 20, 40, 54 and all 67 training stations
(`SIZES`) and scores both on the **same test stations**. Subsets come from `data.split.train_subset` (`n_stations`, `seed`): the first n stations of a
seeded shuffle of the training stations, so the subsets of one seed are nested (20 ⊂ 40 ⊂ 54). The stations left out belong to no partition: they are not
trained on, are not lag neighbours and are not tested on. GeoCP is off. By default cross-validation is off and the epochs are fixed per model (`EPOCHS`: 7 for the
CNN-LSTM, 15 for the LSTM-only, what cross-validation chose on the full data), so a point costs one fit; `USE_CV = True` brings cross-validation back (about four times the time).
The notebook has the same time budget and resume as 2.15. It writes `learning_curve.csv`, `learning_curve_gap.csv` and `learning_curve.png`.

**What a "draw" is, and how to use it.** A draw is one random subset of the training stations: a replicate of the station sample. At 20 stations it matters which 20
you pick, so a single subset mixes "fewer stations" with "lucky or unlucky stations". With `DRAWS` replicates per size:

- each dot is one draw; the width of the dots at a size is the station-sampling variability; the line is the mean over draws;
- both models of a draw see the identical stations, so the **paired gap** (CNN-LSTM minus LSTM-only) cancels station luck. The gap is the number the decision rests on;
- the test stations are fixed, so the curve describes these 16 test stations. Test-set uncertainty is a separate matter (a bootstrap over the test stations);
- draws overlap more at large sizes (two draws of 54 from 67 share at least 41 stations), so the spread there understates the sampling variability, and the full-size point has
  no subset variability at all (add training seeds if you need it);
- three draws is a minimum; with n = 3 report min and max, not a standard deviation. Small sizes are cheap, so more draws cost mostly time there.

A rule worth writing down before the run: *the CNN branch is judged to benefit from more stations if the mean gap rises across sizes and the gap at the
largest size lies above the draw-to-draw spread of the gap at the smallest.* Suggested wording for the thesis: "We repeated each training-set size over R random station subsets,
nested within a draw, and trained both models on the same subset. The accuracy of a size is the mean over draws, the CNN effect is the paired difference between models within a draw,
and the spread across draws shows how much of the curve is due to which stations were sampled. The 16 test stations are the same at every point."

### 2.17 The dataset report

`python run_pipeline.py --stage dataset` (add the same `--override` data paths as for the other stages) writes `dataset_report/` next to the processed folder; start with
`DATASET_REPORT.md`. It reads the raw sensor files with the same cleaning as `preprocessing/clean_loader.py` and the processed folder for the partitions and model rows, and trains nothing:

| Part | What it reports |
|---|---|
| sensors and coverage | every sensor of the locations table, inside the raster or not, its network (OpenAQ / QC) and partition; the NCR share the raster covers |
| rows | readings in the files, not numeric, outside the allowed range, usable, hours with a reading, hours filled by interpolation, hours left missing, model rows, and the hours that have no model row; sensors with readings but no model rows are named |
| missingness | per sensor: the share of its active span with no reading and the longest gap (`dataset_sensors.csv`) |
| temporal availability | per year: sensors per network, station-hours, sensors reporting at the same hour; `fig_availability.png` is a sensor × month heatmap |
| PM2.5 distribution | mean, spread, percentiles, share above 35 and 50 µg/m³, per network (`fig_pm25_distribution.png`) |
| morphology | each of the five bands over the NCR (10 m pixels and 320 m patch means) and at the sensors' own patches, and how much of the NCR's range the sensors span (`fig_morphology.png`, `dataset_morphology_*.csv`) |

On the data of the four-model runs: 84 sensors inside the raster (44 OpenAQ from 2024-08, 40 QC to 2025-12-31), 1.21 M usable readings, 1.35 M hours with a value once short gaps are filled,
1.31 M model rows; median missing share 0.3 % for QC and 31 % for OpenAQ; the sensors span 73 to 91 % of the NCR patches' 5th–95th percentile range of each band (92 to 98 % of the full range).

---

### 2.18 A whole day of maps, an animation and a slideshow

`--stage infer --date D` now predicts **all 24 hours** of the day by default (`inference.snapshot_hours`, one band per hour in `prediction_volume.tif`, a true 24-hour mean in
`prediction_daily_mean.tif`). Beside the hourly maps, `maps/` gets, for each style in `output.maps.styles` (`continuous` and `aqi` by default):

| File | What it is |
|---|---|
| `<style>_panels.png` | the 24 hours on one colour scale, 4 rows × 6 columns (up to 6 hours keep the old 2-column layout) |
| `<style>_animation.gif` | the day as a looping animation, one frame per hour, one colour scale for all frames (`output.animation.fps`, default 2: a day in 12 s) |
| `<style>_animation.mp4` | the same, smaller, for slides; only written when `pip install imageio imageio-ffmpeg` is available (otherwise a one-line notice) |
| `<style>_slideshow.html` | one page with play / pause and a slider over the hours; keep it next to the PNGs |

The animation only re-assembles the hourly PNGs that `render_day` already drew, so it costs nothing extra to predict. Settings: `output.animation` (`enabled`, `fps`, `formats`, `hold_last_s`).
**The cost is the 24 predictions:** a day takes six times as long as four hours (every chunk of cells serves all hours of a date, so a little less than six). Use
`--override "inference.snapshot_hours=[8,12,17,21]"` for the old four-hour day.

**Grad-CAM covers the whole day too.** `--stage explain --date D` explains `inference.explain_hours` (default `[]` = every snapshot hour, so all 24 hours and their aggregate). Each explained hour is slow,
so a full day costs six times the four-hour one; `--override "inference.explain_hours=[8,12,17,21]"` gives the four peak hours. The hours must be among the day's `infer` hours. The hot and cold cells of the pooled figures come from the mean of **exactly the explained hours** (read from the day's volume), so a cell's
mean prediction and its averaged Grad-CAM describe the same hours.

The XGBoost branch does the same: `python run_pipeline.py --stage infer --date D` makes all 24 hours (`XGB_SNAPSHOT_HOURS=8,12,17,21` for the old set), a 4 × 6 panel and
`animation_<day>.gif` + `slideshow_<day>.html` in the day's folder, drawn on one shared scale. The full 10 m grid (`infer10`) stays at four hours (one 10 m snapshot is ~10.6 million cells).

### 2.19 The no-PM2.5-history diagnostic: does the morphology carry signal on its own?

The learning curve and the four models show that the CNN branch adds nothing **on top of** the PM2.5 lags and the spatial lag. They do not say whether morphology carries signal at all,
or is simply redundant with those lags (a station's own recent readings already reflect what its surroundings do). `data.temporal.use_pm25_lags: false` removes the PM2.5 lags and the spatial lag:
the temporal branch then sees only the hour of day (`sin_hour`, `cos_hour`), so the **only** way to tell stations apart is the morphology. Train the CNN-LSTM and the LSTM-only model under
this setting and compare their test R²:

| result | reading |
|---|---|
| CNN-LSTM clearly above LSTM-only | the morphology carries signal on its own; it is redundant once the lags are in (the usual reading). Other cities' morphology would help only a model without history |
| the two about equal | the morphology carries little signal here, even with no other information: building the five bands for other cities is not worth the weeks |

`training_cells.ipynb` runs it: add `"cnn_nolags"` and `"nocnn_nolags"` to `RUN_VARIANTS` (GeoCP off, no maps; both share `processed_nolags/`). Without lags the model needs no history, so every
hour with a reading is a row (about 1.35 million rows at 20 stations against 295 thousand test rows with the lags): compare the two diagnostic models with **each other**, not row for row with the others.
These models are far less accurate than the lag models by design; the diagnostic is the **gap**, not the level.

### 2.20 Transfer learning of the temporal side (other cities → Manila)

The CNN branch adds no accuracy, so transfer goes to the temporal side (LSTM on PM2.5 history). The group decided that weather and forecasting ahead in time are out of scope, so the model
transfers **PM2.5 history only**.

1. **Get the source data** (`other-country-scrapers/`). `scrapers.ipynb` (VS Code or Jupyter) downloads hourly PM2.5 of stations in Ho Chi Minh City and Hanoi (Vietnam), Bangkok (Thailand), Jakarta
   (Indonesia) and Kuala Lumpur (Malaysia) from **one site, the OpenAQ API v3** (`/locations` to find the stations, `/sensors/{id}/hours` for the values), and writes
   `data/Other Countries/<country>/<city>/{sensors/openaq_<id>_<name>.csv, locations.csv, station_quality.csv, source.json}` in the Manila layout. You need a **free OpenAQ key**
   (<https://explore.openaq.org/register>) in the environment variable `OPENAQ_API_KEY`; free keys allow 60 requests a minute and 2,000 an hour, the scraper stays under both and resumes where it stopped
   (`DRY_RUN = True` first prints the number of stations and the estimated time). A station is kept with at least 6,000 hours, readings in 12 different months and 50 % of the hours between its first and
   last reading present; `station_quality.csv` lists every station with the reason it was kept or dropped. Hours are labelled by their start (`TIME_LABEL = "from"`): compare one value with a Manila
   file before mixing the networks. Tests of the scraper (a fake API, no key): `python -m pytest other-country-scrapers/tests -q`.
2. **Pretrain** (`--stage pretrain`, `configs/transfer.yaml`, `model/transfer/`). The windows of the source cities are built by the same functions as Manila's (lags, spatial lag among the stations
   of the same city, local hour of day), scaled with **Manila's scalers** (run `--stage preprocess` first), and an LSTM-only model is trained on them with early stopping on held-out source stations.
   `python run_pipeline.py --stage pretrain --name vn_th --source vietnam/hanoi --source thailand/bangkok` writes `<data.paths.pretrained_dir>/vn_th/pretrained_lstm.pt`, `summary.json` and
   `station_ranking.csv` (every source station by its distance to the Manila training stations: maximum mean discrepancy on log PM2.5 and the hour of day; `transfer.top_k_stations` keeps the closest k).
   `transfer.exclude_months` leaves calendar months out (for example 2 to 4, the regional burning season): an option to test, not a default.
3. **Fine-tune** on Manila: `--stage train --override model.spatial_cnn.enabled=false --override train.init_from=<pretrained_lstm.pt>`. `train.init_from` copies every layer whose name and shape
   match (an LSTM-only model loads all of it; a CNN-LSTM loads the temporal branch and leaves the CNN and fusion head fresh) and **stops** if the temporal branch does not match completely.
   `train.freeze: [temporal]` keeps parts fixed (any of `spatial`, `temporal`, `head`). The default (`null`, `[]`) changes nothing. `run_info.json` records what was loaded and frozen.
4. **Compare** (`transfer_learning.ipynb`, Kaggle or local). It runs the checkpoints `manila_only`, `single_<country>` (one per country), `mmd_top_<k>` and `all_sources`, each as `finetune`,
   `head_only` (the pretrained LSTM frozen) and `zero_shot` (one pass at learning rate 0), all scored on the **same 16 Manila test stations**, and writes `transfer_comparison.csv` and a figure.
   Judge a checkpoint by `finetune` against `manila_only`; `all_sources` against the ranked subset shows negative transfer if there is any.

**XGBoost does the same** with a booster instead of a network (`xgboost/training/transfer.py`, commented for review): `python run_pipeline.py --stage pretrain --pretrain-arm A --source ...` builds the
temporal features (the lags, rolling statistics and calendar encodings of `prep.py`) from the source cities, fills the morphology columns with NaN (XGBoost handles missing values), trains a booster and
saves `outputs/pretrained/<name>/pretrained_booster.json`; with `XGB_INIT_FROM=<that file>` the `test` stage and the production fit **add** the Manila trees on top of the pretrained ones. Use the
same arm on both sides (the trees read the columns by position); arm A (time + PM2.5 history, no morphology) is the temporal-only choice.

**Choosing the pool without the test stations.** `transfer_learning.ipynb` has `USE_CV` (default `True`). Every checkpoint is then fitted with the same spatial cross-validation as the
main notebook (3 folds over the Manila training stations, the pretrained weights loaded afresh in each fold; the pretraining never sees Manila, so nothing leaks). The pool is ranked by its
**CV R²** and by in how many folds it beats `manila_only`; its **test R² is read once, after the choice**. Cost: about four times a plain fit (minutes for an LSTM-only model); the
`zero_shot` variant is never cross-validated. `USE_CV = False` compares on the test stations directly (quick, but the winner's test score is then a little optimistic: say so if you report it).

#### Which source stations are used, and why not all of them

The question a panel will ask: *"OpenAQ has hundreds of stations in these cities. Why not train on all of them?"* The answer is a chain of filters, each with a reason, and a **test** at the
end that lets the data (not our opinion) decide whether using more stations helps.

**1. Where we look (the scraper).** Only the **capital region** of each country (the counterpart of Manila's National Capital Region, about 640 km²): Hanoi, Bangkok with its metropolitan ring,
Jakarta with Greater Jakarta, Kuala Lumpur with the Klang Valley, plus Ho Chi Minh City as the largest Vietnamese city. Only **fixed stations that report PM2.5** (mobile units have no place, other
pollutants are not our target). All from OpenAQ, the same network as the Manila `openaq_*` files, so values are defined the same way. The Zenodo "HCMC / Bangkok" files are **not** used: they are CAMS model output, not measurements.

**2. Is the station good enough (quality rules, `scrapers.ipynb`).** A station is kept only with at least **6,000 hourly values** (about 8 months), values in **12 or more different calendar
months** (so it sees a whole seasonal cycle, not just the dry season), and **50 % or more of the hours** present between its first and last reading. Every station, kept or not, is listed with its
reason in `station_quality.csv` (`kept`, `reason`), together with `is_monitor` (reference-grade monitor or low-cost sensor), so the choice can be audited. Result of the run of 19 November 2026
(local `data/Other Countries`):

| City | Stations found | Kept | Most common reasons for dropping |
|---|---|---|---|
| Bangkok | 272 | 125 | too few hours, fewer than 12 months, not downloaded (rate limit) |
| Hanoi | 38 | 11 | too few hours, no values in the range |
| Jakarta | 28 | 5 | too few hours, no values in the range |
| Kuala Lumpur | 22 | 2 | too few hours |
| Ho Chi Minh City | 20 | 1 | too few hours, no values in the range |

**3. Can the model use it (`transfer/source_data.py`).** The model's input is the station's own last three hours **and the spatial lag** (the inverse-distance-weighted mean of the five nearest stations of the **same city** at those hours). A
window exists only if both are present, so a station needs neighbours that report **at the same time**. Kuala Lumpur's two kept stations never overlap (one stops in October 2024, the other
starts in December 2024), so that city gives **no window** and is left out with a message. Source values are scaled with **Manila's** scalers, and the hour of day is the city's own local hour.

**4. Which of the usable stations (the pools, `transfer_learning.ipynb`).** We do **not** pick by hand. Four kinds of pool are trained and compared with a Manila-only model on the same data:

| Pool | Stations used | What it tests |
|---|---|---|
| `manila_only` | none (baseline) | does any pretraining help at all |
| `single_<country>` | all kept stations of one country | which country transfers, the cleanest control |
| `mmd_top_10` | the 10 stations whose PM2.5 distribution is closest to the Manila **training** stations | does a smaller, more similar pool beat a big one |
| `all_sources` | every kept station | the "why not all" case, tested rather than assumed |

*Closest* is the squared maximum mean discrepancy (RBF kernel) between a station's samples of `[log(1 + PM2.5), sin(hour), cos(hour)]` and the same samples of the Manila training stations
(`transfer/select.py`): level, spread and daily cycle. It uses PM2.5 and the hour of day only and **never the Manila test stations**. The pool that is **chosen** is the one with the best
**cross-validated R²** on the Manila training stations (and the most folds won); its test R² is read once afterwards.

**Why not simply use all stations, in short**
- **More is not always better in transfer learning.** Stations from another city can differ in level, in daily cycle, in sensor type and in what burns or traffics around them; a model that
  learns them can do worse on Manila than one that never saw them (*negative transfer*). Whether this happens here is exactly what `all_sources` against `manila_only` measures.
- **"All" is not "balanced".** Bangkok contributes 125 of the 144 kept stations (87 %), so `all_sources` is mostly Bangkok, not a regional average. `single_<country>` and `mmd_top_10` show
  whether that matters.
- **Seasonal shifts.** Regional biomass-burning haze (roughly January to April in Thailand and northern Vietnam) is a source Manila does not share; `transfer.exclude_months` can remove those months from
  the source series to test it.
- **Weak stations add noise, not information.** A station with a few weeks of data or half its hours missing teaches the model gaps; hence the three quality rules.
- **No peeking.** The choice of stations never looks at the Manila test stations (similarity uses the training stations; the pool is chosen by cross-validation).

**Limits to state honestly.** Similarity of the distribution is not proof of benefit (a close station can still transfer badly), which is why it is tested rather than trusted. The thresholds (6,000
hours, 12 months, 50 %) are our choices, not a published standard; for a stricter rule see the US EPA data-completeness rule (75 % of samples per quarter, 40 CFR 50 Appendix N) and the EU
Directive 2008/50/EC (90 % data capture), and report a sensitivity check if a panelist asks. Low-cost sensors are kept and flagged (`is_monitor`), not corrected for humidity.

**What was checked.** Unit tests for each piece (warm start, freezing, source windows, station ranking, the pretrain stage, the scraper, the XGBoost module; `tests/test_transfer.py`,
`checks/test_transfer.py`), and a full chain on real Manila data with a synthetic source city (preprocess, pretrain, `finetune` / `head_only` / `zero_shot`, report). **Not run:** the real source cities
(it needs your OpenAQ key and hours of downloading), so there is no transfer result yet.

### 2.21 The transfer-learning reference (`transferlearn.pdf`): what holds and what to change

The file is a chat with an AI assistant, not a source. What was checked, and what it means for the thesis:

- **Holds:** transferring the temporal side; OpenAQ as the data source; a single-country baseline before pooling; a ranking of stations by distribution distance instead of choosing countries by hand;
  a completeness floor over a full year; the four-checkpoint ablation (Manila-only, one country, ranked subset, all stations) to show negative transfer; excluding the burning season **as an option to test**.
- **Wrong or unverifiable:** the Zenodo "Ho Chi Minh City" and "Bangkok" hourly datasets are **CAMS model output served by Open-Meteo**, not sensor measurements (a 53 kB file for HCMC): do not use them as ground truth.
  The Nanjing paper (Kokkonen et al. 2021, *Environ. Sci.: Atmos.* 1(7):481) is quoted as showing that morphology has little effect; it found that morphology explained **up to 73 %** of the variance of PM2.5
  in street canyons (13 % in open areas); R² 0.076 belongs to distance to major roads alone. "Indriyati et al. 2025" is Musa, Adhi and Mulsandi (*Jurnal Meteorologi dan Geofisika* 26(1):37-54), and the page-48 quotation
  was not verified. Aziim et al. (2026, *Front. Environ. Sci.* 14:2000424) and Lung et al. (2022, *IJERPH* 19(3):1522) exist; Rakholia et al. (2022) is *Urban Climate* 46:101315 (its title in the file is wrong).
  "k = 8 to 15 stations" has no source. The data of this thesis are OpenAQ and Quezon City stations, not Raspberry Pi or mobile nodes.
- **Out of scope here:** weather (humidity correction, ERA5, boundary-layer height) and any forecasting ahead in time, by group decision. Where the file leans on them, only the PM2.5-history part is used.
- **Applies to both models:** the CNN-LSTM transfers its temporal branch (the morphology branch is off, there are no rasters abroad); XGBoost continues boosting from a booster trained on the temporal features.
  The morphological transfer the file starts from (Taiwan and dense street canyons first) is **not** supported by our results (the CNN branch adds no accuracy); the file's own second answer drops it.

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
neighbours (what the model and the calibration saw). From `inference.openaq_only_from_year` (2026) on, only the OpenAQ stations among them
are used, because the Quezon City network has no readings then ([2.6](#26-make-the-maps)).

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
outlined), a map of where those cells are, a hot-against-cold band chart, a morphology sheet, `gradcam_attention.csv`, and a summary giving for each band how many patch standard deviations higher
the attended pixels are than the rest of the patch (positive values for building density, aspect ratio, height
or street width point at dense blocks, canyons and arterials; [2.6](#26-make-the-maps) lists the figures). The overlays are still to be inspected by eye.
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
    tables/   accuracy_summary.csv  accuracy_by_station.csv  accuracy_by_year.csv  cv_folds.csv
    data/     predictions_test.csv  predictions_calibration.csv  calibration_table.csv  geocp.json
              spatial_bias.json  residuals_by_cell.csv
    surface/<YYYYMMDD>/   prediction_volume.tif  lower_volume.tif  upper_volume.tif  width_volume.tif
                          prediction_daily_mean.tif  output_summary.json  maps/*.png
    explain/<YYYYMMDD_HHMM>/   gradcam_map.png  gradcam_hot_cells.png  gradcam_cold_cells.png  gradcam_band_profile.png  morphology_sheet.png
                               gradcam_attention.csv  gradcam_summary.json  feature_groups/  (maps, close-ups, comparison chart, csv)
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
| `maps/aqi_HHMM.png` | `infer` | one map per snapshot on the Philippine AQI scale, clipped to the NCR with city names, scale bar and north arrow, stations marked (optional outline above 35 µg/m³, off by default) |
| `maps/daily_panels.png` | `infer` | the day's snapshots side by side on one scale |
| `maps/continuous_HHMM.png`, `continuous_panels.png`, `continuous_daily_mean.png` | `infer` | the same predictions on a continuous viridis scale with a colourbar (the 2×2 panel figure) |
| `gradcam_map.png` | `explain` | the explained hot and cold cells, numbered, on the predicted PM2.5 map |
| `gradcam_band_profile.png` | `explain` | per band, what the attended pixels look like for hot against cold cells |
| `morphology_sheet.png` | `explain` | the five morphology bands mapped over the NCR, with the explained cells marked |
| `feature_groups/*` | `explain` | hot and cold cells of the high and low end of each band over the NCR: maps, Grad-CAM close-ups, a high-against-low chart, per-city table |
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
- **The Quezon City sensors stop on 2025-12-31, and the spatial lag does not skip stations that are silent.** A cell's lag comes from its five
  nearest training stations; where all five are Quezon City sensors there is no lag in 2026 (88.3 % of cells predicted on 2026-04-11, 100 % on
  2025-07-01, at stride 21). Maps and Grad-CAM for 2026 or later therefore use the nearest OpenAQ training stations instead (`inference.openaq_only_from_year`,
  [2.6](#26-make-the-maps)): every cell is predicted, from farther stations and with a history the model never saw in training. The model rows
  are built differently and are not helped: OpenAQ station 6340868 (reports only from 2026-05, test partition) yields no model rows, so the test set
  is effectively 16 stations, not 17. Choosing the nearest stations that actually report, in training too, would fix that; it is a method change and has not been made.
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

`python -m pytest tests -q` runs 420 tests, 416 of which apply when `contextily` is installed (about 1 min), covering: configuration and the CRS check; the reset command; the infer / geocp workflow; the coverage check; the QC data conversion and the `qcdata_` loader; progress, hang detection and timeouts (also for the tune stage); the accuracy report (including the by-year table) and the snapshot layout, the OpenAQ-only history for 2026 dates, the NCR-clipped maps and explain figures, the map outline switch and the smooth band edges, the combined datasets, pinned CV folds and the run-comparison tool; shared-patch training; shapes and
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

---

## 8. Findings so far and the transfer-learning plan

### 8.1 What the four models showed

Runs of 4 November 2026 (`training_cells.ipynb`, dataset `from_2023_with_preinstall`, raster `master_raster_v3`, tuned hyperparameters, seed 42, 3 spatial CV folds,
16 test stations with readings). Scores are on stations the models never saw.

| model | CV R² | test R² | test RMSE | test MAE | parameters |
|---|---|---|---|---|---|
| CNN-LSTM, no GeoCP | 0.708 | 0.721 | 5.42 | 3.39 | 1,058,929 |
| LSTM-only, no GeoCP | 0.715 | 0.724 | 5.39 | 3.36 | 169,073 |
| CNN-LSTM, GeoCP | 0.692 | 0.707 | 5.50 | 3.44 | 1,058,929 |
| LSTM-only, GeoCP | 0.699 | 0.718 | 5.39 | 3.38 | 169,073 |

- **The CNN branch does not improve accuracy.** Paired over the 16 test stations (bootstrap, 2,000 resamples): without GeoCP the test R² gap (CNN-LSTM minus LSTM-only) is −0.003, 95 % range
  −0.008 to +0.002 (no difference; the CNN is better at 8 of 16 stations); with GeoCP it is −0.011, range −0.016 to −0.008 (the CNN-LSTM is worse; better at 1 of 16). The two
  models' errors correlate at 0.97 to 0.98 and a 50/50 average gives R² 0.726. Cross-validation agrees with the test.
- **Most of the skill is the previous hour.** Repeating the previous hour gives R² 0.646; the LSTM adds 0.078, the CNN branch −0.003 on top of that. The LSTM-only model's skill against
  the previous hour is 0.22 (1 − MSE ratio).
- **Peaks dominate the error.** The 9 % of test hours above 30 µg/m³ produce 51 % of the squared error, the 1.3 % above 50 µg/m³ produce 29 %. The CNN is slightly better below 30 and
  slightly worse above, so the two cancel.
- **Training.** The CNN-LSTM's best validation epoch comes after 1 to 10 epochs (final fits of 7 and 3 epochs), the LSTM-only's after 4 to 19 (final fits of 15 and 9). 885,760 of the
  CNN-LSTM's 1.06 M parameters are in the CNN, fitted on one 320 m patch per station, 54 to 67 distinct images in total. Overfitting to those few patches is the likely reading;
  the runs show early stopping and the parameter counts, not the mechanism.
- **Grad-CAM and band ablation (`cnn_nogeocp`, 3 dates × 4 hours).** Attention ranks NDVI first on every date and in every hour (the hours look alike); swapping a band moves the
  prediction most for the building-height bands (0.4 to 1.2 µg/m³ on average, against a test RMSE of 5.4). The model reads the bands, but attention and impact disagree and the
  impact does not buy accuracy. Both are model sensitivities, not causal effects.
- **More training stations helped, the CNN most.** Removing GeoCP returned the 13 calibration stations to training (54 → 67): test R² 0.707 → 0.721 for the CNN-LSTM (+0.014) and
  0.718 → 0.724 for the LSTM-only (+0.006). This is one split and the test rows differ slightly (287 k and 295 k), so it is a hint, not a result.
- No spatial clustering is left in the residuals (global Moran's I not significant in all four runs).

### 8.2 What this means for transfer learning

- **Capacity is not the limit.** A vision transformer or a deeper or pretrained backbone has even more capacity for the same 54 to 67 distinct patches, so it should not be expected to
  help (ImageNet backbones also do not fit 5-band, 32 × 32 morphology). Run it as an ablation and report it even if it loses.
- **Data is the lever.** Transfer learning pays only if it adds distinct stations or conditions. Other Southeast Asian cities would; the risks are different sensors, sources and
  climate, and that each city needs the same five morphology bands built first.
- **The temporal side needs no rasters** to transfer: pretrain the LSTM and the lag model on other cities' hourly PM2.5 and fine-tune on Manila.
- **The model sees only PM2.5 history** (three lags, the spatial lag and the hour), and repeating the previous hour already gives R² 0.646. Weather inputs and forecasting ahead in time are
  out of scope for this thesis (group decision, November 2026), so the model stays a PM2.5-only estimator of present conditions.

### 8.3 The plan, in order of cost against information

| step | what | cost | it tells you |
|---|---|---|---|
| 1 | learning curve: both models on 20, 40, 54, 67 Manila stations (`learning_curve.ipynb`, [2.16](#216-learning-curve-learning_curveipynb)) | hours, no new data | whether more stations make the CNN pay off, and whether the LSTM-only gains from more stations |
| 2 | pretrain the LSTM on other cities' hourly PM2.5, fine-tune on Manila | no rasters needed | whether the temporal part transfers |
| 3 | pretrain the CNN-LSTM on other Southeast Asian cities, fine-tune on Manila | five morphology bands per city: weeks | the adviser's main idea |
| (ablation) | a ViT or another backbone | a configuration change | expected negative; report it |

### 8.4 Is step 1 redundant?

Partly, and it depends on what it is for.

- **It gates only step 3.** Step 2 does not depend on it: run it regardless. Step 1 decides whether the expensive morphological transfer (weeks of raster building) is worth it.
- **It is cheap, needs no new data, and gives two curves.** The gap (CNN-LSTM minus LSTM-only) answers the morphological question. The LSTM-only curve alone answers the temporal one: if it keeps rising
  with the number of stations, more stations or cities should help step 2 too; if it is flat, expect little from transfer there.
- **A flat or negative gap is a strong reason not to build step 3**, and a positive trend is a reason to; either result is a thesis result and answers "why transfer?" at the defence.
- **Its limits.** A trend over 20 to 67 stations is an extrapolation to hundreds, and it cannot test domain shift (other sensors, sources, climate). A positive result is necessary
  evidence, not sufficient.
- **When it is redundant.** If you will do step 3 whatever step 1 says, step 1 has no decision value, because step 3 contains its own comparison with and without morphology. Then
  run it only as context, with fewer draws (for example sizes 20, 40 and all, 2 draws).

### 8.5 Result of step 1: the learning curve (run locally, 10 to 12 November 2026)

Both models trained on 20, 40, 54 and all 67 training stations (`data.split.train_subset`, seed 0, one draw per size, nested subsets), GeoCP off, no cross-validation, fixed epochs
(CNN-LSTM 7, LSTM-only 15), scored on the same 16 test stations (about 295 k rows). Run with the same code and settings as `learning_curve.ipynb`, outputs in `Outputs/learning_curve/`
(`learning_curve.csv`, `learning_curve_gap.csv`, `learning_curve.png`, per-size runs under `lc/`). The CNN-LSTM fits took about 1 to 6 h each, the LSTM-only fits 3 to 6 min.

| training stations | CNN-LSTM test R² | LSTM-only test R² | gap (CNN-LSTM − LSTM-only) | CNN-LSTM RMSE | LSTM-only RMSE |
|---|---|---|---|---|---|
| 20 | 0.7089 | 0.7164 | −0.0075 | 5.530 | 5.458 |
| 40 | 0.7123 | 0.7180 | −0.0058 | 5.501 | 5.446 |
| 54 | 0.7106 | 0.7239 | −0.0133 | 5.515 | 5.387 |
| 67 (all) | 0.7173 | 0.7240 | −0.0067 | 5.451 | 5.386 |

- **The CNN branch does not pay off as stations grow.** The gap is negative at every size (−0.006 to −0.013) and does not rise across sizes, so the rule written in 2.16 ("the mean gap rises
  across sizes") is not met. This agrees with the four-model comparison (8.1: gap −0.003 without GeoCP).
- **Both models improve slowly with stations**, by about the same amount: +0.008 R² from 20 to 67 stations for each (CNN-LSTM 0.709 → 0.717, LSTM-only 0.716 → 0.724). More Manila stations
  therefore do not close the gap; the LSTM-only model stays ahead.
- **The curves are nearly flat.** The change from 20 to 67 stations (+0.008) is small next to the previous-hour baseline (R² 0.646) and to the difference between the LSTM and that baseline (+0.078).
- **The 67-station points agree with the earlier full-data runs** (CNN-LSTM 0.717 here against 0.721, LSTM-only 0.724 against 0.724), so the pipeline reproduces.
- **Limits.** One draw per size, so station-sampling variability is not measured; differences of about 0.005 are within what another draw could change. The 16 test stations are fixed, the full-size point has
  no subset variability, and 20 to 67 stations says little about hundreds. Read it as consistent evidence, not as a proven absence of any effect.
- **Consequence for the plan (8.3):** the morphological transfer (step 3) is the weakest bet, because neither the four-model comparison nor the learning curve shows the CNN branch adding accuracy,
  and more stations alone do not change that. Step 2 does not depend on this result.

### 8.6 Is transfer learning on the temporal side alone enough?

Yes, as the main transfer-learning contribution, on the evidence so far; with three conditions.

- **Why it is enough.** The accuracy comes from the temporal side: the previous hour gives R² 0.646, the LSTM adds 0.078, the CNN branch adds nothing (−0.003 to −0.011 in the four-model comparison, negative at
  every size in 8.5). Transfer should go where the skill is.
- **It is cheaper.** It needs only other cities' hourly PM2.5 series, no morphology rasters (weeks per city). The LSTM-only model has 169 k parameters and trains in minutes, so pretraining and
  fine-tuning runs can be repeated.
- **Condition 1: expect a modest gain.** The LSTM-only curve rises by only +0.008 R² from 20 to 67 stations, so more stations of the same kind buy little. A gain from other cities comes from new conditions
  (sources, seasons, extremes) rather than from the count; the peaks above 30 µg/m³ give 51 % of the squared error (8.1), and that is where other cities could help.
- **Condition 2: test against the right baselines**, on the same 16 test stations: Manila-only LSTM (0.724), pretrained then fine-tuned, pretrained and not fine-tuned (zero-shot), and the previous-hour baseline.
  Report negative transfer if it occurs (other sensors, sources and climate can hurt).
- **Condition 3: keep the story consistent.** The thesis finding is then "morphology bands did not add accuracy over the temporal model at 20 to 67 stations; the accuracy is temporal; transfer
  targets the temporal side". The CNN-LSTM, Grad-CAM and band ablation stay as the analysis of what the morphology branch learned, not as the accuracy claim. If the adviser wants the morphological
  transfer, step 3 remains possible, but this evidence does not support its cost.

### 8.7 Manuscript: sections to add to the table of contents

Written against the table of contents of the manuscript (chapters 1 to 4 as pasted). Numbers are suggestions that follow the existing numbering. Weather inputs and
forecasting ahead in time are out of scope (group decision, November 2026), so there are no sections for them. Search strings are for Google Scholar, Scopus or OpenAlex; add `PM2.5` and `"Southeast Asia"` where it narrows the field. Verify every paper yourself before citing it.

**Review of Related Literature (chapter 2)**

| # | Section | What to write (from our work) | Search keywords |
|---|---|---|---|
| 2.3.3 | Morphology against temporal drivers of PM2.5 | The mixed evidence: morphology explains between-site variation (Kokkonen et al. 2021, Nanjing: up to 73 % in street canyons, 13 % in open areas) but hourly prediction is dominated by recent readings (Deep-AIR, Zhang et al. 2022, found historical PM2.5 the best predictor). Sets up why we test the branch rather than assume it helps. | `urban morphology PM2.5 spatial variation street canyon`, `built environment land use regression PM2.5 explained variance`, `temporal versus spatial predictors PM2.5 hourly prediction`, `feature importance lagged PM2.5 deep learning` |
| 2.4.5 | Data quantity, learning curves and model capacity | Why a learning curve answers "does the CNN need more stations?"; power-law shape, ill-behaved curves; small training sets inflate validation optimism for CNNs. Backs the learning-curve test (4.4.5). | `learning curve training set size generalization`, `shape of learning curves review`, `scaling laws deep learning data size`, `sample size convolutional neural network small dataset remote sensing`, `spatial autocorrelation inflates CNN performance` |
| 2.4.6 | Transfer learning in air-quality modelling | Pretraining on data-rich cities, fine-tuning on a target city; negative transfer; domain shift (sensors, climate, sources); partial transfer (temporal side only). | `transfer learning air quality prediction cross-city`, `PM2.5 forecasting transfer learning data-scarce city`, `domain adaptation air pollution low-cost sensors`, `negative transfer spatiotemporal`, `pretrain fine-tune LSTM PM2.5`, `Maximum Mean Discrepancy source selection transfer` |
| 2.4.7 | Benchmarks and significance testing | Persistence (previous hour) as the benchmark, skill score, paired bootstrap over stations; why a model that merely matches persistence has no added value. | `skill score persistence baseline air quality`, `paired bootstrap spatial cross-validation stations`, `Diebold-Mariano test model comparison`, `persistence benchmark PM2.5 machine learning` |
| 2.4.8 | Explainable deep learning for environmental maps | Grad-CAM and SHAP on spatial inputs; saliency is a sensitivity, not a cause; disagreement between attention and perturbation. | `Grad-CAM remote sensing regression`, `SHAP spatial deep learning environmental`, `saliency maps reliability sanity checks`, `feature attribution perturbation versus gradient disagreement` |
| 2.5 | Synthesis (extend) | State the gap: morphology-aware CNN-LSTMs are reported to help, but few studies test the CNN branch against a temporal-only model on unseen stations, with a learning curve. | `ablation study CNN-LSTM air quality unseen stations`, `spatial leave-station-out PM2.5 deep learning` |

**Theoretical Framework (chapter 3)**

| # | Section | Content |
|---|---|---|
| 3.3.4 | Transfer learning | Source and target domains, pretraining, fine-tuning, freezing, warm-start boosting for XGBoost, domain shift, negative transfer, station selection by distribution distance. |
| 3.4.4 | Baselines, skill scores and paired comparison | Persistence baseline (previous hour), skill = 1 − MSE ratio, paired bootstrap over test stations, error correlation between models, simple average of two models. |
| 3.4.5 | Learning curves and branch ablation | Definition, nested training subsets, draws as station-sample replicates, paired gap between the two models, a decision rule written before the run; the 2 × 2 design (CNN branch yes/no × GeoCP yes/no). |
| 3.6 | Model interpretability (new, now only in 4.6.4) | Grad-CAM, band swarm, band ablation, what each does and does not show. |

**Methodology (chapter 4)**

| # | Section | Content |
|---|---|---|
| 4.1.4 | Source-domain data acquisition | OpenAQ v3 stations in Vietnam, Thailand, Indonesia and Malaysia; completeness rules; licences. Note that the Zenodo HCMC and Bangkok sets are CAMS model output, not measurements. |
| 4.2.6 | Harmonisation and source-station selection | Units, time zones, completeness floor, optional exclusion of biomass-burning months, distance to the Manila distribution. |
| 4.3.4 | Transfer-learning procedure | Pretraining the temporal branch, fine-tuning, morphology branch off or frozen; XGBoost warm start; the four checkpoints (Manila-only, single country, ranked subset, all sources). |
| 4.3.5 | CNN-branch ablation (the four-model design) | `training_cells.ipynb`: CNN-LSTM and LSTM-only, with and without GeoCP; resume rules. |
| 4.4.4 | Baselines and paired comparison | Previous-hour baseline, bootstrap of 2,000 resamples over the 16 test stations, error correlation, 50/50 average, error concentration above 30 and 50 µg/m³, residual Moran's I. |
| **4.4.5** | **Learning-curve experiment** | **Sizes 20, 40, 54 and 67 training stations; nested seeded subsets; both models on the same subset; same 16 test stations; fixed epochs (7 and 15); the rule for "the CNN benefits from more stations"; result: negative gap at every size (−0.006 to −0.013 R²); limits (one draw per size).** |
| 4.4.6 | Morphology diagnostic (no PM2.5 lags) | Both models without the lag channels, to separate "morphology carries no signal" from "morphology is redundant with the lags" ([2.19](#219-the-no-pm25-history-diagnostic-does-the-morphology-carry-signal-on-its-own)). Built and tested, not run on the full data yet. |
| 4.4.7 | Transfer-learning evaluation | Same test stations; baselines; zero-shot against fine-tuned; negative-transfer check. |
| 4.6.2 | Snapshot series (extend) | All 24 hours by default; Grad-CAM keeps its own hours (`inference.explain_hours`). |
| 4.6.4 | Interpretability (extend) | Add the band ablation and the pooling of Grad-CAM over hours. |
| 4.6.5 | Day animation and slideshow | How the 24 hourly maps become a GIF, MP4 and slideshow on one colour scale. |

**Results chapter (not in the pasted contents): the tests we ran that need a place.** Four-model comparison (CV and test R², RMSE, MAE); paired bootstrap; previous-hour baseline
and skill; error concentration; Moran's I of the residuals; Grad-CAM and band ablation (attention and impact disagree); the learning curve (8.5); the dataset report (availability,
coverage, morphology representativeness); training behaviour (best epoch, parameter counts, the overfitting reading). The learning curve and the four-model comparison are the two
experiments that support one claim: **the morphology branch adds no measurable accuracy over the temporal model**. Word it so: "the CNN branch did not improve accuracy at 20 to 67
training stations", not "morphology has no effect on PM2.5", which these tests do not show (8.1, and the no-lag diagnostic in 4.4.6 is what would sharpen it).

**Checked: tests in this work that belong in chapter 2 (as literature to justify them), not only in the methods:** the learning curve (2.4.5), the persistence baseline and skill score
(2.4.7), the paired bootstrap (2.4.7), spatial validation without leakage (already 2.3 to 3.4, add the CNN-inflation evidence), Grad-CAM and perturbation (2.4.8), the ablation design (2.5),
residual Moran's I (already in 3.4.3 and 4.4.3).


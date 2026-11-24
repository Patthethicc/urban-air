# amouei-model: which parts of the Amouei (2023) framework improve the PM2.5 model?

This folder is a copy of `model/` (the dual-branch CNN-LSTM; its manual is now [README_MAIN_MODEL.md](README_MAIN_MODEL.md)). The
elements of Amouei's framework are added to it as **switches**. All switches are **off** by default, so with no switch on this folder reproduces the main model exactly. Each
experiment turns on one switch and is scored against the main model on the **same 16 pinned test stations**.

> Amouei, M. (2023). *Classification of urban morphology and its relationship with air pollution using deep learning.* MSc thesis, ITC, University of Twente.
> Two stages: a CNN classifies Sentinel-2 patches into Local Climate Zones (LCZ). An LSTM (256-128-64-32, dense 128, dropout 0.5, 24-hour window) then predicts
> PM2.5 from the LCZ class and its probabilities, weather (temperature, humidity, wind as a vector), a road-density traffic proxy, and sin/cos of the time of day and year.
> Test R² 0.75 on the last two months of the **same** sensors.

## 1. Answer in short

| Amouei element | Effect on test R² (local screen, 2 seeds) | Add to `model/`? |
|---|---|---|
| **24-hour input window** (instead of 3 h) | **+0.013** (+0.012 and +0.013; every interval above 0) | **Yes**: a config change |
| **Weather** incl. the target hour (temperature, humidity, wind vector) | **+0.006** (+0.005 and +0.007) | **Yes**: code below |
| Weather at the lag hours only | +0.004 (+0.003 and +0.006) | the target-hour version is better |
| 4 stacked LSTM layers 256-128-64-32 + dense 128, dropout 0.5 | +0.001 at 3 h; at 24 h **worse than the 24 h window alone** (+0.007 vs +0.013), unstable between seeds, 3× the parameters | **No** |
| Time of year (sin/cos) | +0.0005 (noise) | No (harmless) |
| Traffic proxy (street share of the 320 m patch) | −0.001 (noise) | No |
| LCZ-like class per station (one-hot) | −0.001 (noise) | No |
| Morphology band means as LSTM columns | −0.003 (noise) | No |
| Everything Amouei used, together | +0.013, the same as the 24 h window alone | the gain comes from the window and the weather |
| **24 h window + weather incl. target hour + time of year** (`recommended`) | **+0.018** (+0.019 and +0.017; R² 0.710 → 0.728) | **Yes: this is the combination to adopt** |
| 12 h window | +0.005, unstable between seeds (+0.002 / +0.008) | 24 h is better |

The baseline in this screen scores R² 0.710 and repeating the previous hour scores 0.647. **Answer for the adviser:** making the model look like Amouei's adds about **0.018 R²**,
and nearly all of it comes from **two inputs**: a longer history (24 h) and weather. Together they score higher than Amouei's full setup (+0.018 against +0.013), which also carries the deep network and the urban-form inputs. Amouei's deeper network does not help, and neither do the urban-form inputs (LCZ, traffic, morphology).
The urban-form result agrees with Amouei's own finding (removing LCZ cost only 0.012 R²) and with ours (the CNN branch adds about 0). These are **quick-screen** numbers
(section 4). Confirm them with the Kaggle notebook before quoting them in the thesis.

## 2. What was added (all in this folder; `model/` is unchanged)

| Switch (`configs/*.yaml`) | Amouei element | Code |
|---|---|---|
| `data.temporal.lags: [24, ..., 1]` | 24-hour window (`input_width` 24) | existing; no change needed |
| `data.temporal.weather.enabled`, `.variables`, `.lead_hours` | weather of the nearest station | `preprocessing/amouei_features.py` (`add_weather`), `preprocessing/fetch_weather.py` |
| `data.temporal.cyclical_year` | "time of year" sin/cos | `amouei_features.add_cyclical_year` |
| `data.temporal.static_features: [road_density]` | road-density KDE (300 m) traffic proxy | `amouei_features.static_station_features` |
| `data.temporal.static_features: [lcz_proxy]` | LCZ class as an input | `amouei_features.lcz_proxy_class` |
| `data.temporal.static_features: [morphology]` | urban form as tabular LSTM inputs | `amouei_features.static_station_features` |
| `model.temporal_lstm.layer_sizes: [256,128,64,32]` + `model.fusion_head` `[128]`, dropout 0.5 | Amouei's LSTM network | `networks/temporal_lstm.py` |
| `train.row_stride` | (screening only: every n-th training row) | `run_pipeline.py` |

Supporting changes:
- `config_parser.py` adds the new columns to the feature list.
- `preprocessing/scalers.py` z-scores weather and static columns on the training rows only (`fit_extra`).
- `run_pipeline.py` joins the new columns in `preprocess`.
- `inference/cell_inference.py` builds weather and time of year for map cells.
- `tests/test_amouei_features.py` has 16 tests.
- The suite has 516 tests and all pass.

**Weather data.** `python -m preprocessing.fetch_weather` downloads hourly ERA5 reanalysis from the Open-Meteo archive. The archive is free and needs no key. It writes 9 points over the NCR
(7 distinct grid cells; Open-Meteo snaps two pairs of points to the same cell), 2023-01-01 to 2026-11-18, local time, to `../data/Weather/`. Each station takes the nearest point, like Amouei's nearest station.
`lead_hours: 1` lets the last input step carry the weather of the **target hour**. This fits a model that estimates the present (the main model's scope, and what the maps need).
It does not fit forecasting, because the target hour's weather would then have to be a forecast.

**What is not copied, and why.**
- *Amouei's LCZ stage (Sentinel-2 + ResNet-50).* It needs labelled LCZ training areas that this project does not have. `lcz_proxy` assigns a Stewart & Oke style class from
  building density, height and NDVI (thresholds in `configs/data.yaml`). It tests the idea of discrete urban-form classes as inputs. 73 of the 85 stations fall in two
  classes (compact low-rise 34, open low-rise 39), so the classes tell the stations apart less well than the continuous bands.
- *The weighted road KDE.* The repository has no road vectors. `road_density` is the share of street pixels in the 320 m patch, close to Amouei's 300 m kernel but without
  road-class weights.
- *Amouei's test split* (the last two months of the same sensors). It is not used. It would raise R², but because the test becomes easier, not because the model is better.
  Every score here is on stations the model never saw.

## 3. How to run

```
cd amouei-model
python -m pytest tests -q                                      # 516 tests
python -m preprocessing.fetch_weather --end 2026-11-18         # once: data/Weather/
python -m experiments.run_ablation --list                      # the experiments
python -m experiments.run_ablation --quick --seeds 42 43       # the local screen (section 4), about 2 h on a 4-core CPU
python -m experiments.run_ablation --seeds 42 43               # the full protocol (CV, all rows, early stopping): use Kaggle
python -m experiments.run_ablation --compare-only --quick --seeds 42 43
```

**Kaggle:** [amouei_ablation.ipynb](amouei_ablation.ipynb). Commit and push `amouei-model/` first, because the notebook clones the repository. If your data dataset
has no `Weather/` folder, turn Internet on so the notebook can download it. Run with the GPU on. The notebook runs the tests, then the experiments one by one (finished experiments are skipped,
so it resumes across sessions), then draws the comparison table and a chart of the gains with their intervals and prints a verdict per element.

Each experiment writes its predictions to `runs/<ARM>/ablation/snapshots_<experiment>/` and its log to `experiments/logs/`. The tables go to `experiments/results/`:
`ablation[_quick].csv` has one row per experiment and seed, and `ablation[_quick]_by_experiment.csv` has the mean over seeds and the verdict.

## 4. How the comparison is made

- **Same data:** `from_2023_with_preinstall` (1,360,277 samples). Pinned partition `configs/pinned_partition_2026-11.json`: 68 training stations and 16 scored test stations.
  No GeoCP, so the calibration stations train the model, as in the best main-model result.
- **LSTM-only.** The CNN branch added nothing in the main model and is the slow part. `cnn_baseline` and `cnn_weather_lead1` exist to check this on Kaggle.
- **Same rows.** A 24 h window drops 609 more test rows than a 3 h window. Every R² is therefore computed on the 298,683 test rows that **all** experiments kept.
- **Gain and interval.** The R² gain is measured against the baseline trained with the **same seed**. The 95 % interval comes from a bootstrap over the 16 test **stations** (2,000 resamples).
  An element "helps" only if the interval is above 0 for every seed.
- **Quick screen (the numbers above):**
  - no cross-validation;
  - every 4th training row;
  - 6 epochs;
  - the main model's tuned learning rate and batch size for every experiment.

  The baseline reaches R² 0.710 here, against 0.724 with the full protocol, so the absolute values are low and only the **differences** are meaningful.
  The bootstrap interval covers the choice of stations, not training randomness. Training randomness is why there are two seeds: the baseline moves by 0.0025 between seeds.

### Local screen, all runs

The mean over seeds 42 and 43, from `experiments/results/ablation_quick_by_experiment.csv`:

| experiment | test R² | gain | gain per seed (42 / 43) | verdict | parameters | RMSE above 30 µg/m³ |
|---|---|---|---|---|---|---|
| `recommended` | 0.7278 | +0.0179 | +0.0186 / +0.0172 | helps | 171,809 | 12.18 |
| `window24` | 0.7226 | +0.0127 | +0.0122 / +0.0132 | helps | 169,073 | 12.27 |
| `amouei_full` | 0.7225 | +0.0126 | +0.0133 / +0.0119 | helps | 548,865 | 12.40 |
| `amouei_lstm_w24` | 0.7165 | +0.0066 | +0.0114 / +0.0019 | no clear effect | 532,481 | 12.46 |
| `weather_lead1` | 0.7155 | +0.0056 | +0.0046 / +0.0066 | helps | 170,897 | 12.22 |
| `window12` | 0.7147 | +0.0048 | +0.0018 / +0.0077 | no clear effect | 169,073 | 12.79 |
| `weather` | 0.7141 | +0.0042 | +0.0030 / +0.0055 | helps | 170,897 | 12.27 |
| `amouei_lstm` | 0.7110 | +0.0011 | +0.0021 / +0.0001 | no clear effect | 532,481 | 12.31 |
| `year` | 0.7104 | +0.0005 | +0.0007 / +0.0003 | no clear effect | 169,985 | 12.28 |
| `baseline` | 0.7099 | 0 | n/a |  | 169,073 | 12.25 |
| `road_density` | 0.7091 | -0.0008 | -0.0019 / +0.0004 | no clear effect | 169,529 | 12.41 |
| `lcz_proxy` | 0.7089 | -0.0010 | -0.0019 / -0.0001 | no clear effect | 173,177 | 12.38 |
| `morphology_tabular` | 0.7072 | -0.0027 | -0.0044 / -0.0009 | no clear effect | 171,353 | 12.72 |

## 5. What to change in `model/` (if the Kaggle run confirms it)

1. **24 h window:** set `data.temporal.lags` to 24 down to 1 in `model/configs/data.yaml`. Maps already work with any lags. Each map hour then needs 24 hours of readings before it.
   Re-tune afterwards, because the tuned settings were found for 3 steps.
2. **Weather (with `lead_hours: 1`):** copy from this folder:
   - `preprocessing/fetch_weather.py`;
   - the weather parts of `preprocessing/amouei_features.py`;
   - the `config_parser`, `scalers`, `run_pipeline.preprocess` and `cell_inference` changes listed in section 2;
   - the `data.temporal.weather` and `data.paths.weather_csv` settings.
3. Time of year is optional. It is in `recommended` because it costs nothing, but on its own it did not help.
4. Leave out the 4-layer network, LCZ, traffic and the morphology columns: they did not help.

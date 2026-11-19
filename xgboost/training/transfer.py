"""Transfer learning for the XGBoost model: pretrain on other cities' PM2.5 history, then CONTINUE boosting on Manila.

The idea in four lines
    1. other-country-scrapers/scrapers.ipynb wrote hourly PM2.5 of other cities to data/Other Countries/<country>/<city>/ (the Manila sensor layout).
    2. `pretrain_booster` builds the same TEMPORAL features the Manila model uses (lags, rolling statistics, hour / weekday / month as sin and cos)
       from those series, and trains a booster on them. The other cities have no morphology, so every morphology column (and the interactions that
       need it) is NaN; XGBoost handles missing values natively, so those columns simply never help the pretrained trees.
    3. `fit_regressor(..., init_from=<that booster>)` then adds the Manila trees on top of the pretrained ones (xgboost's `xgb_model=`): the final model
       = pretrained trees + new trees fitted to the Manila training rows.
    4. Compare it with the Manila-only model on the SAME test cells (the `test` stage does both when XGB_INIT_FROM is set).
No weather, no forecasts: PM2.5 history and the clock only.

Not applied to the source series: the Manila pipeline drops the New Year fireworks hours (prep.py); do the same here if a source city shows the spike.

Everything here is plain numpy / pandas / xgboost and takes the feature names as an argument, so it can be read and checked without loading the
Manila data. It mirrors preprocessing/prep.py `_time_aware_lags_and_rolling` (the same lags and windows); keep the two in step.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

# the history features and their definitions, identical to preprocessing/prep.py
LAGS = (1, 2, 3, 4)                       # pm25_lag_1 .. pm25_lag_4: the reading 1 .. 4 hours before
ROLL_WINDOWS = {24: 18, 72: 54}           # window length in hours -> minimum number of readings required (as in prep.py)


def list_sources(root):
    """City folders under `root` (relative names such as 'vietnam/hanoi') that hold a source.json and at least one sensor file."""
    root = os.fspath(root)
    out = []
    for meta in glob.glob(os.path.join(root, '**', 'source.json'), recursive=True):
        folder = os.path.dirname(meta)
        if glob.glob(os.path.join(folder, 'sensors', '*.csv')):
            out.append(os.path.relpath(folder, root).replace(os.sep, '/'))
    return sorted(out)


def read_city(folder):
    """One scraped city -> long table (Location, DateTime, Value). DateTime is the city's own LOCAL clock (naive), so 'hour of day' means the same
    thing it means in Manila (rush hour at the same clock time), not the UTC hour. Location ids are those of OpenAQ."""
    with open(os.path.join(folder, 'source.json'), encoding='utf-8') as fh:
        tz = json.load(fh).get('timezone')
    if not tz:
        raise ValueError(f'{folder}/source.json has no timezone: write the dataset again with the current other-country-scrapers')
    frames = []
    for path in sorted(glob.glob(os.path.join(folder, 'sensors', '*.csv'))):
        raw = pd.read_csv(path, usecols=['location_id', 'utc_timestamp', 'pm25_value'])
        t = pd.to_datetime(raw['utc_timestamp'], utc=True).dt.tz_convert(tz).dt.tz_localize(None)
        v = pd.to_numeric(raw['pm25_value'], errors='coerce')
        v = v.where(v.between(0.0, 1000.0))            # the same plausible range the Manila data is cleaned to
        frames.append(pd.DataFrame({'Location': raw['location_id'].astype('int64'), 'DateTime': t, 'Value': v}).dropna(subset=['Value']))
    if not frames:
        raise FileNotFoundError(f'{folder}: no sensor files')
    return pd.concat(frames, ignore_index=True)


def temporal_features(df):
    """Per station: lags, rolling mean / std of the PREVIOUS hours, and the calendar features. Returns one row per original reading.

    The series is put on a gap-free hourly index first (`asfreq('h')`), so a lag or a window that crosses a gap is NaN (and the row is dropped
    later) instead of silently pairing a reading with one from several hours earlier."""
    frames = []
    for loc, g in df.groupby('Location'):
        g = g.sort_values('DateTime').drop_duplicates(subset='DateTime', keep='first')
        hourly = g.set_index('DateTime')['Value'].asfreq('h')
        feat = {f'pm25_lag_{k}': hourly.shift(k) for k in LAGS}                      # reading k hours earlier
        prev = hourly.shift(1)                                                        # windows end at t-1: they never contain the value being predicted
        for w, need in ROLL_WINDOWS.items():
            feat[f'pm25_roll{w}_mean'] = prev.rolling(w, min_periods=need).mean()
            feat[f'pm25_roll{w}_std'] = prev.rolling(w, min_periods=need).std()
        out = pd.DataFrame(feat)
        out['Value'] = hourly
        out = out.reset_index()
        out['Location'] = loc
        frames.append(out.merge(g[['DateTime']], on='DateTime', how='inner'))         # keep only hours that really had a reading
    f = pd.concat(frames, ignore_index=True)
    f['pm25_diff_1_2'] = f['pm25_lag_1'] - f['pm25_lag_2']
    # the same encodings as prep.py: hour 0-23, weekday 0-6 (Monday = 0), month 1-12 (NOT 0-11: the Manila model uses the calendar number)
    hour, dow, month = f['DateTime'].dt.hour, f['DateTime'].dt.day_of_week, f['DateTime'].dt.month
    for name, value, period in (('hour', hour, 24), ('dayofweek', dow, 7), ('month', month, 12)):
        f[f'{name}_sin'] = np.sin(2 * np.pi * value / period)                         # a cycle: 23:00 is next to 00:00
        f[f'{name}_cos'] = np.cos(2 * np.pi * value / period)
    return f


def design_matrix(feat, feature_names):
    """The columns of `feature_names`, in that order. A name the source cities cannot have (morphology, interactions with it) becomes NaN."""
    cols = [feat[n].to_numpy(dtype=float) if n in feat.columns else np.full(len(feat), np.nan) for n in feature_names]
    return np.column_stack(cols)


def source_xy(feat, feature_names, exclude_months=()):
    """(X, y, stations) for training: rows with every lag and rolling mean present, y = log1p(PM2.5) (as the Manila model), months excluded if asked."""
    needed = [f'pm25_lag_{k}' for k in LAGS] + [f'pm25_roll{w}_mean' for w in ROLL_WINDOWS]
    ok = feat[needed + ['Value']].notna().all(axis=1)
    if exclude_months:
        ok &= ~feat['DateTime'].dt.month.isin(list(exclude_months))       # calendar months 1-12, e.g. the regional burning season: an option to test
    feat = feat.loc[ok].reset_index(drop=True)
    return design_matrix(feat, feature_names), np.log1p(feat['Value'].to_numpy(dtype=float)), feat['Location'].to_numpy()


def station_split(stations, fraction, seed=42):
    """Whole stations held out to stop the pretraining (the Manila model is also validated on stations it did not train on). -> boolean mask 'is validation'."""
    ids = np.array(sorted(set(int(s) for s in stations)))
    if len(ids) < 2:
        return np.zeros(len(stations), dtype=bool)
    n_val = int(min(len(ids) - 1, max(1, round(len(ids) * fraction))))
    val_ids = set(int(i) for i in np.random.default_rng(seed).permutation(ids)[:n_val])
    return np.array([int(s) in val_ids for s in stations])


def pretrain_booster(X, y, stations, params, out_dir, n_estimators=600, early_stopping_rounds=30, val_fraction=0.15, seed=42, extra=None):
    """Train on the source stations and save <out_dir>/pretrained_booster.json (+ summary.json). `params` are XGBRegressor parameters
    (max_depth, learning_rate, ...); the validation stations decide when to stop. Returns the summary dict."""
    import xgboost as xgb

    os.makedirs(out_dir, exist_ok=True)
    is_val = station_split(stations, val_fraction, seed)
    model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=seed, tree_method='hist', n_jobs=-1,
                             early_stopping_rounds=early_stopping_rounds if is_val.any() else None, **params)
    if is_val.any():
        model.fit(X[~is_val], y[~is_val], eval_set=[(X[is_val], y[is_val])], verbose=False)
    else:
        model.fit(X, y, verbose=False)
    path = os.path.join(out_dir, 'pretrained_booster.json')
    model.get_booster().save_model(path)
    best = int(getattr(model, 'best_iteration', n_estimators - 1)) + 1 if is_val.any() else int(n_estimators)
    summary = {'booster': path, 'n_features': int(X.shape[1]), 'trees': int(model.get_booster().num_boosted_rounds()), 'best_iteration': best,
               'rows_train': int((~is_val).sum()), 'rows_val': int(is_val.sum()), 'stations': int(len(set(stations.tolist()))),
               'params': params, 'nan_share_by_feature': [round(float(np.isnan(X[:, j]).mean()), 3) for j in range(X.shape[1])], **(extra or {})}
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


def fit_regressor(params, n_estimators, X, y, seed=42, init_from=None):
    """The Manila model. With `init_from` (a booster saved by pretrain_booster) the new `n_estimators` trees are ADDED to the pretrained ones.
    Without it this is the ordinary fit (nothing changes when the transfer is not asked for)."""
    import xgboost as xgb

    model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=seed, tree_method='hist', n_jobs=-1, **params)
    if init_from:
        booster = xgb.Booster()
        booster.load_model(init_from)
        if booster.num_features() != X.shape[1]:                                          # another feature set: the trees would read the wrong columns
            raise ValueError(f'{init_from} was trained on {booster.num_features()} features, this model has {X.shape[1]}: pretrain with the same arm')
        model.fit(X, y, xgb_model=booster)
    else:
        model.fit(X, y)
    return model


def init_from_env():
    """The pretrained booster the stages should continue from: the environment variable XGB_INIT_FROM (a path), or None."""
    path = os.environ.get('XGB_INIT_FROM', '').strip()
    if path and not os.path.isfile(path):
        raise FileNotFoundError(f'XGB_INIT_FROM={path} does not exist')
    return path or None

"""Checks for training/transfer.py (pretrain on other cities, continue boosting on Manila). Run from xgboost/:  python -m pytest checks/test_transfer.py -q
These tests do not need the Manila data: the module takes the feature names as an argument."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
xgb = pytest.importorskip('xgboost')
from training import transfer  # noqa: E402

FEATURES = ['bh_mean', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos',
            'pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3', 'pm25_lag_4', 'pm25_diff_1_2',
            'pm25_roll24_mean', 'pm25_roll24_std', 'pm25_roll72_mean', 'pm25_roll72_std', 'hour_x_bldg_dens']


def _long(n_hours=400, stations=(1, 2, 3), seed=0, start='2024-01-01'):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n_hours, freq='h')
    rows = []
    for s in stations:
        v = np.clip(20 + 6 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) + np.cumsum(rng.normal(0, 0.6, n_hours)) * 0.3, 2, 150)
        rows.append(pd.DataFrame({'Location': s, 'DateTime': t, 'Value': v}))
    return pd.concat(rows, ignore_index=True)


def test_lags_and_windows_look_only_backwards():
    df = _long(stations=(1,))
    f = transfer.temporal_features(df).set_index('DateTime')
    v = df.set_index('DateTime')['Value']
    t = v.index[100]
    assert f.loc[t, 'pm25_lag_1'] == v.iloc[99] and f.loc[t, 'pm25_lag_4'] == v.iloc[96]
    assert f.loc[t, 'pm25_roll24_mean'] == pytest.approx(v.iloc[76:100].mean())        # the 24 hours BEFORE t: t itself is not in the window
    assert f.loc[t, 'pm25_diff_1_2'] == pytest.approx(v.iloc[99] - v.iloc[98])


def test_a_gap_makes_the_lag_nan_instead_of_pairing_old_values():
    df = _long(stations=(1,)).drop(index=range(50, 53))                                  # three missing hours
    f = transfer.temporal_features(df).set_index('DateTime')
    after = f.loc[pd.Timestamp('2024-01-01') + pd.Timedelta(hours=53)]
    assert np.isnan(after['pm25_lag_3'])                                                 # lag 3 reaches into the gap -> NaN, not the reading before it


def test_calendar_features_are_cycles():
    f = transfer.temporal_features(_long(stations=(1,)))
    assert (f['hour_sin'] ** 2 + f['hour_cos'] ** 2).round(6).eq(1).all()
    first = f.loc[f['DateTime'].dt.hour == 0].iloc[0]
    assert first['hour_sin'] == pytest.approx(0.0) and first['hour_cos'] == pytest.approx(1.0)


def test_the_calendar_encoding_matches_prep_py_month_is_1_to_12():
    f = transfer.temporal_features(_long(stations=(1,), start='2024-03-04'))            # a Monday in March
    row = f.iloc[0]
    assert row['month_sin'] == pytest.approx(np.sin(2 * np.pi * 3 / 12)) and row['dayofweek_sin'] == pytest.approx(0.0)


def test_columns_the_sources_cannot_have_are_nan_and_the_order_is_kept():
    f = transfer.temporal_features(_long())
    X = transfer.design_matrix(f, FEATURES)
    assert X.shape[1] == len(FEATURES)
    assert np.isnan(X[:, FEATURES.index('bh_mean')]).all() and np.isnan(X[:, FEATURES.index('hour_x_bldg_dens')]).all()
    assert np.isfinite(X[:, FEATURES.index('hour_sin')]).all()


def test_source_rows_need_the_history_and_months_can_be_left_out():
    f = transfer.temporal_features(_long(n_hours=24 * 80, start='2024-01-15'))           # runs into March
    X, y, st = transfer.source_xy(f, FEATURES)
    assert len(y) == len(st) == X.shape[0] and len(y) < len(f)                           # the first hours have no 24 h / 72 h history
    assert np.allclose(y, np.log1p(f.loc[f[['pm25_lag_4', 'pm25_roll24_mean', 'pm25_roll72_mean']].notna().all(axis=1), 'Value']))
    X2, y2, _ = transfer.source_xy(f, FEATURES, exclude_months=[2])
    assert len(y2) < len(y)


def test_the_station_split_holds_out_whole_stations():
    st = np.repeat(np.arange(20), 3)
    val = transfer.station_split(st, 0.2, seed=1)
    assert len(set(st[val])) == 4 and not (set(st[val]) & set(st[~val]))
    assert not transfer.station_split(np.array([5, 5, 5]), 0.2).any()


def _fit_inputs():
    f = transfer.temporal_features(_long(n_hours=700, stations=tuple(range(1, 7))))
    return transfer.source_xy(f, FEATURES)


PARAMS = {'max_depth': 3, 'learning_rate': 0.2, 'subsample': 1.0, 'colsample_bytree': 1.0, 'min_child_weight': 1.0, 'reg_alpha': 0.0, 'reg_lambda': 1.0, 'gamma': 0.0}


def test_pretrain_saves_a_booster_and_continuing_adds_trees(tmp_path):
    X, y, st = _fit_inputs()
    summary = transfer.pretrain_booster(X, y, st, PARAMS, str(tmp_path), n_estimators=40, early_stopping_rounds=5)
    assert os.path.isfile(summary['booster']) and summary['n_features'] == len(FEATURES) and summary['rows_val'] > 0
    assert summary['nan_share_by_feature'][FEATURES.index('bh_mean')] == 1.0
    pre_trees = summary['trees']
    # "Manila": other rows, same features; the new trees are added on top of the pretrained ones
    X2, y2, _ = transfer.source_xy(transfer.temporal_features(_long(n_hours=500, stations=(7, 8), seed=5)), FEATURES)
    model = transfer.fit_regressor(PARAMS, 10, X2, y2, init_from=summary['booster'])
    assert model.get_booster().num_boosted_rounds() == pre_trees + 10
    scratch = transfer.fit_regressor(PARAMS, 10, X2, y2)
    assert scratch.get_booster().num_boosted_rounds() == 10                              # without init_from nothing changes
    assert np.isfinite(model.predict(X2)).all()


def test_a_booster_of_another_feature_count_is_refused(tmp_path):
    X, y, st = _fit_inputs()
    path = transfer.pretrain_booster(X, y, st, PARAMS, str(tmp_path), n_estimators=5, early_stopping_rounds=2)['booster']
    with pytest.raises(ValueError, match='features'):
        transfer.fit_regressor(PARAMS, 5, X[:, :-2], y, init_from=path)


def test_the_environment_variable_names_the_booster(monkeypatch, tmp_path):
    monkeypatch.delenv('XGB_INIT_FROM', raising=False)
    assert transfer.init_from_env() is None
    f = tmp_path / 'b.json'
    f.write_text('{}')
    monkeypatch.setenv('XGB_INIT_FROM', str(f))
    assert transfer.init_from_env() == str(f)
    monkeypatch.setenv('XGB_INIT_FROM', str(tmp_path / 'missing.json'))
    with pytest.raises(FileNotFoundError):
        transfer.init_from_env()


def test_a_scraped_city_is_read_on_its_own_clock(tmp_path):
    city = tmp_path / 'thailand' / 'bangkok'
    (city / 'sensors').mkdir(parents=True)
    stamps = pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC')
    pd.DataFrame({'location_name': 'A', 'location_id': 7, 'sensor_id': 7, 'utc_timestamp': stamps.strftime('%Y-%m-%dT%H:%M:%SZ'),
                  'local_timestamp': '', 'pm25_value': [10.0, 20.0, -5.0, 30.0, 5000.0], 'parameter': 'PM2.5', 'unit': 'x'}).to_csv(city / 'sensors' / 'openaq_7_A.csv', index=False)
    (city / 'source.json').write_text(json.dumps({'timezone': 'Asia/Bangkok', 'epsg': 32647}))
    assert transfer.list_sources(tmp_path) == ['thailand/bangkok']
    df = transfer.read_city(str(city))
    assert df['DateTime'].min() == pd.Timestamp('2024-01-01 07:00')                      # 00:00 UTC = 07:00 in Bangkok
    assert list(df['Value']) == [10.0, 20.0, 30.0]                                       # -5 and 5000 are outside 0-1000 and dropped

"""data.temporal.use_pm25_lags: false = the diagnostic without any PM2.5 history (the temporal branch sees only the hour of day)."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from config_parser import derive, feature_base_names
from preprocessing.temporal_window_builder import build_windows


def _readings(n=12, stations=(1, 2)):
    rows = []
    for s in stations:
        rows += [{"location_id": s, "local_timestamp": pd.Timestamp("2025-07-01") + pd.Timedelta(hours=h), "pm25_value": 10.0 * s + h}
                 for h in range(n)]
    return pd.DataFrame(rows)


def test_the_default_keeps_the_lags_and_the_spatial_lag(base_cfg):
    names = feature_base_names(base_cfg["data"])
    assert names[0] == "pm25" and "spatial_lag" in names and names[-2:] == ["sin_hour", "cos_hour"]


def test_without_lags_only_the_hour_is_left(base_cfg):
    cfg = deepcopy(base_cfg)
    cfg["data"]["temporal"]["use_pm25_lags"] = False
    derive(cfg)
    assert cfg["derived"]["feature_names"] == ["sin_hour", "cos_hour"]
    assert cfg["derived"]["F"] == 2 and cfg["model"]["temporal_lstm"]["input_size"] == 2
    assert cfg["derived"]["pm25_feature_idx"] == []                       # nothing in the temporal input is a PM2.5 value to scale


def test_without_lags_and_without_the_hour_there_is_no_input_and_validation_says_so(base_cfg):
    from config_parser import validate

    cfg = deepcopy(base_cfg)
    cfg["data"]["temporal"].update(use_pm25_lags=False, cyclical_hour=False)
    derive(cfg)
    with pytest.raises(ValueError, match="no input at all"):
        validate(cfg)


def test_windows_carry_no_pm25_when_lags_are_off(small_cfg):
    small_cfg["data"]["temporal"].update(lags=[2, 1], cyclical_hour=True, use_pm25_lags=False)
    derive(small_cfg)
    df = _readings()
    X, y, meta = build_windows(df, small_cfg)
    assert X.shape[1:] == (2, 2)                                            # [T = 2 lags, F = sin, cos]
    assert np.abs(X).max() <= 1.0 + 1e-6                                    # only sin / cos values: no PM2.5 (10-30) anywhere in the input
    on = deepcopy(small_cfg)
    on["data"]["temporal"]["use_pm25_lags"] = True
    derive(on)
    X_on, y_on, _ = build_windows(df, on)
    assert X_on.shape[2] == 3 and X_on[..., 0].max() > 1                  # with the lags: [pm25, sin, cos], PM2.5 present
    assert len(y) >= len(y_on) and set(y_on.ravel()) <= set(y.ravel())     # the same targets (the lag rows are never fewer without the lags)


def test_the_spatial_lag_recomputer_is_off_without_lags(small_cfg):
    from training.fold_features import SpatialLagRecomputer

    small_cfg["data"]["spatial_lag"]["enabled"] = True
    small_cfg["data"]["temporal"].update(lags=[2, 1], cyclical_hour=True, use_pm25_lags=False)
    derive(small_cfg)
    assert SpatialLagRecomputer(small_cfg).enabled is False                  # it would otherwise look for a spatial_lag channel that is not there

"""The Amouei (2023) switches: off = the main model, on = the documented columns, scaled from training rows only."""
import numpy as np
import pandas as pd
import pytest
import torch

from config_parser import derive
from networks.dual_branch_net import DualBranchNet
from networks.temporal_lstm import TemporalLSTM
from preprocessing import amouei_features as am
from preprocessing.scalers import Scalers
from preprocessing.temporal_window_builder import build_windows


def _all_on(cfg):
    t = cfg["data"]["temporal"]
    t["cyclical_year"] = True
    t["weather"]["enabled"] = True
    t["static_features"] = ["morphology", "road_density", "lcz_proxy"]
    return derive(cfg)


def test_defaults_are_the_main_model(base_cfg):
    assert base_cfg["derived"]["feature_names"] == ["pm25", "spatial_lag", "sin_hour", "cos_hour"]
    assert base_cfg["derived"]["standardize_idx"] == []
    assert base_cfg["model"]["temporal_lstm"]["layer_sizes"] is None
    assert not am.any_enabled(base_cfg["data"])


def test_feature_order_and_scaled_columns(cfg):
    d = _all_on(cfg)["derived"]
    names = d["feature_names"]
    assert names[:6] == ["pm25", "spatial_lag", "sin_hour", "cos_hour", "sin_doy", "cos_doy"]
    assert names[6:10] == ["temperature_2m", "relative_humidity_2m", "wind_u", "wind_v"]
    assert "road_density" in names and sum(n.startswith("morph_") for n in names) == 5
    assert sum(n.startswith("lcz_") for n in names) == len(am.LCZ_CLASSES)
    scaled = {names[i] for i in d["standardize_idx"]}
    assert "temperature_2m" in scaled and "road_density" in scaled and "morph_ndvi_mean" in scaled
    assert not any(n.startswith(("lcz_", "sin_", "cos_", "pm25", "spatial")) for n in scaled)   # one-hot, cyclic, PM2.5: other scalers
    assert cfg["model"]["temporal_lstm"]["input_size"] == len(names)


def test_unknown_static_feature_is_refused(cfg):
    cfg["data"]["temporal"]["static_features"] = ["traffic_counts"]
    with pytest.raises(ValueError):
        derive(cfg)


def test_cyclical_year():
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01 00:00", "2024-07-02 00:00"])})
    out = am.add_cyclical_year(df, "ts")
    assert out["sin_doy"].iloc[0] == pytest.approx(0.0) and out["cos_doy"].iloc[0] == pytest.approx(1.0)
    assert out["cos_doy"].iloc[1] < -0.99                                       # mid-year is the opposite side


def test_wind_becomes_a_vector(tmp_path):
    p = tmp_path / "w.csv"
    pd.DataFrame({"point_id": [0, 0], "latitude": 14.5, "longitude": 121.0,
                  "local_timestamp": ["2024-01-01 00:00", "2024-01-01 01:00"],
                  "wind_speed_10m": [2.0, 3.0], "wind_direction_10m": [90.0, 0.0]}).to_csv(p, index=False)
    w = am.load_weather(str(p))
    assert w["wind_u"].tolist() == pytest.approx([0.0, 3.0], abs=1e-9)
    assert w["wind_v"].tolist() == pytest.approx([2.0, 0.0], abs=1e-9)


def _weather_setup(cfg, tmp_path, lead=0):
    from pyproj import Transformer
    hours = pd.date_range("2024-01-01", periods=4, freq="h")
    rows = []
    for pid, (lat, lon, base) in enumerate([(14.45, 120.95, 10.0), (14.70, 121.10, 100.0)]):
        for k, h in enumerate(hours):
            rows.append({"point_id": pid, "latitude": lat, "longitude": lon, "local_timestamp": h,
                         "temperature_2m": base + k, "relative_humidity_2m": 50.0,
                         "wind_speed_10m": 1.0, "wind_direction_10m": 0.0})
    p = tmp_path / "w.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    cfg["data"]["paths"]["weather_csv"] = str(p)
    cfg["data"]["temporal"]["weather"].update(enabled=True, lead_hours=lead)
    derive(cfg)
    x, y = Transformer.from_crs("EPSG:4326", cfg["data"]["crs"], always_xy=True).transform([120.951, 121.099], [14.451, 14.699])
    stations = pd.DataFrame({"location_id": [1, 2], "x": x, "y": y})
    c = cfg["data"]["columns"]
    df = pd.DataFrame({c["group"]: np.repeat([1, 2], 4), c["timestamp"]: np.tile(hours, 2), c["target"]: 1.0})
    return df, stations, c


def test_weather_from_the_nearest_point(cfg, tmp_path):
    df, stations, c = _weather_setup(cfg, tmp_path)
    out = am.add_weather(df, stations, cfg).sort_values([c["group"], c["timestamp"]])
    assert out[out[c["group"]] == 1]["temperature_2m"].tolist() == [10, 11, 12, 13]
    assert out[out[c["group"]] == 2]["temperature_2m"].tolist() == [100, 101, 102, 103]


def test_weather_lead_moves_the_next_hour_back(cfg, tmp_path):
    df, stations, c = _weather_setup(cfg, tmp_path, lead=1)
    out = am.add_weather(df, stations, cfg).sort_values([c["group"], c["timestamp"]])
    s1 = out[out[c["group"]] == 1]["temperature_2m"].tolist()
    assert s1[:3] == [11, 12, 13] and np.isnan(s1[3])                       # the last hour has no next hour: its sample drops


def test_lcz_proxy_rules():
    assert am.lcz_proxy_class(0.02, 0.0, 0.7) == "natural_trees"
    assert am.lcz_proxy_class(0.02, 0.0, 0.3) == "natural_low_plants"
    assert am.lcz_proxy_class(np.nan, np.nan, -0.1) == "natural_bare_water"
    assert am.lcz_proxy_class(0.5, 6.0, 0.1) == "compact_low"
    assert am.lcz_proxy_class(0.2, 15.0, 0.1) == "open_mid"
    assert am.lcz_proxy_class(0.45, 40.0, 0.1) == "compact_high"
    assert am.lcz_proxy_class(0.2, 15.0, 0.1, {"high_m": 12.0}) == "open_high"


def test_static_station_features(cfg):
    cfg["data"]["temporal"]["static_features"] = ["morphology", "road_density", "lcz_proxy"]
    derive(cfg)
    bands = cfg["data"]["temporal"]["morphology_bands"]
    p = np.zeros((len(bands), 4, 4))
    p[bands.index("building_density")] = 0.5
    p[bands.index("building_height_mean")] = 8.0
    p[bands.index("street_aspect_ratio_mean"), :2, :] = 0.7                    # half the pixels are street
    out = am.static_station_features({7: p}, bands, cfg)
    row = out.iloc[0]
    assert row["road_density"] == pytest.approx(0.5)
    assert row["morph_building_density"] == pytest.approx(0.5)
    assert row["lcz_compact_low"] == 1.0
    assert sum(row[f"lcz_{c}"] for c in am.LCZ_CLASSES) == 1.0


def test_windows_carry_the_new_columns(cfg):
    cfg["data"]["temporal"].update(lags=[2, 1], cyclical_year=True, static_features=["road_density"])
    cfg["data"]["spatial_lag"]["enabled"] = False
    derive(cfg)
    c = cfg["data"]["columns"]
    hours = pd.date_range("2024-03-01", periods=5, freq="h")
    df = pd.DataFrame({c["group"]: 1, c["timestamp"]: hours, c["target"]: np.arange(5.0), "road_density": 0.3})
    X, y, _ = build_windows(df, cfg)
    names = cfg["derived"]["feature_names"]
    assert X.shape == (3, 2, len(names)) and names[-1] == "road_density"
    assert np.allclose(X[..., -1], 0.3)
    assert y.ravel().tolist() == [2.0, 3.0, 4.0]


def test_extra_scaler_round_trip(tmp_path):
    X = np.zeros((10, 3, 4), dtype=np.float32)
    X[..., 2] = np.arange(30).reshape(10, 3)
    s = Scalers().fit(np.zeros((2, 1, 2, 2)), np.arange(10.0)[:, None], [0]).fit_extra(X, [2])
    Z = s.transform_temporal(X)
    assert Z[..., 2].mean() == pytest.approx(0.0, abs=1e-5) and Z[..., 2].std() == pytest.approx(1.0, abs=1e-4)
    s.save(str(tmp_path / "s.json"))
    assert np.allclose(Scalers.load(str(tmp_path / "s.json")).transform_temporal(X), Z)


def test_old_scaler_file_still_loads(tmp_path):
    import json
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"y_mean": 1.0, "y_std": 2.0, "band_mean": [0.0], "band_std": [1.0], "pm25_idx": [0], "roll_std_idx": []}))
    assert Scalers.load(str(p)).extra_idx == []


def test_tapered_lstm_stack():
    m = TemporalLSTM(10, 999, 1, 0.5, layer_sizes=[256, 128, 64, 32])
    assert m.output_dim == 32 and len(m.stack) == 4
    out = m(torch.randn(5, 24, 10))
    assert out.shape == (5, 32)
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_amouei_network_in_the_model(cfg):
    cfg["model"]["temporal_lstm"]["layer_sizes"] = [256, 128, 64, 32]
    cfg["model"]["fusion_head"].update(hidden_dims=[128], dropout=0.5)
    cfg["model"]["spatial_cnn"]["enabled"] = False
    m = DualBranchNet(cfg["model"]).eval()
    d = cfg["derived"]
    assert m(torch.zeros(1), torch.randn(3, d["T"], d["F"])).shape == (3, 1)
    assert m.head.mlp[0].in_features == 32


def test_maps_refuse_only_static_inputs(cfg):
    import run_pipeline
    _all_on(cfg)
    with pytest.raises(SystemExit):
        run_pipeline._no_amouei_maps(cfg, "infer")
    cfg["data"]["temporal"]["static_features"] = []
    derive(cfg)
    run_pipeline._no_amouei_maps(cfg, "infer")               # window, weather and time of year are mapped


def test_map_cells_get_year_and_weather(cfg, tmp_path):
    """The raster-cell inputs follow the training order, with the nearest point's weather (lead applied)."""
    from pyproj import Transformer
    from inference.cell_inference import cell_temporal_features
    am._WEATHER_CACHE.clear()
    df, stations, c = _weather_setup(cfg, tmp_path, lead=1)
    cfg["data"]["temporal"]["cyclical_year"] = True
    cfg["data"]["spatial_lag"]["k"] = 2
    derive(cfg)
    ts = pd.Timestamp("2024-01-01 03:00")                    # lags 3, 2, 1 -> hours 0, 1, 2; lead 1 -> weather of 1, 2, 3
    x, y = Transformer.from_crs("EPSG:4326", cfg["data"]["crs"], always_xy=True).transform([120.951], [14.451])
    hourly = np.column_stack([np.arange(4.0), np.arange(4.0)])
    X = cell_temporal_features(hourly, [1, 2], [1, 2], stations[["x", "y"]].to_numpy(), np.c_[x, y], t_idx=3,
                               timestamp=ts, cfg=cfg)
    names = cfg["derived"]["feature_names"]
    assert X.shape == (1, 3, len(names))
    assert X[0, :, names.index("temperature_2m")].tolist() == pytest.approx([11.0, 12.0, 13.0])     # point 0, base 10
    assert X[0, 0, names.index("cos_doy")] == pytest.approx(np.cos(2 * np.pi * 0 / 365.25))

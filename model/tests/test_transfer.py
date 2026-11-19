"""Transfer learning of the temporal side: warm start / freeze, source cities, station ranking, the pretrain stage."""
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from config_parser import derive
from networks.dual_branch_net import DualBranchNet
from training.transfer import build_model, freeze_parts, warm_start


def _mc(base_cfg, cnn):
    mc = deepcopy(base_cfg["model"])
    mc["spatial_cnn"]["enabled"] = cnn
    mc["spatial_cnn"]["in_channels"] = 5
    mc["spatial_cnn"]["patch_px"] = 4
    mc["temporal_lstm"].update(input_size=4, hidden_size=8, num_layers=1)
    mc["spatial_cnn"].update(conv_channels=[4], embed_dim=6, feature_map="pool")
    mc["fusion_head"].update(hidden_dims=[8])
    return mc


def _save(model, path):
    torch.save({"model_state": model.state_dict(), "model_cfg": model.model_cfg, "epoch": 3, "val_loss": 0.5}, path)


def test_an_lstm_only_checkpoint_fills_the_temporal_branch_of_a_cnn_lstm(base_cfg, tmp_path):
    donor = DualBranchNet(_mc(base_cfg, False))
    _save(donor, tmp_path / "p.pt")
    target = DualBranchNet(_mc(base_cfg, True))
    report = warm_start(target, str(tmp_path / "p.pt"))
    assert "temporal" in report["parts_loaded"] and "spatial" not in report["parts_loaded"]
    for (k, v) in donor.temporal.state_dict().items():
        assert torch.equal(target.temporal.state_dict()[k], v)
    assert any(k.startswith("head.") for k in report["layers_skipped"])         # the fusion head has another input size: left fresh


def test_an_lstm_only_checkpoint_loads_completely_into_an_lstm_only_model(base_cfg, tmp_path):
    donor = DualBranchNet(_mc(base_cfg, False))
    _save(donor, tmp_path / "p.pt")
    target = DualBranchNet(_mc(base_cfg, False))
    report = warm_start(target, str(tmp_path / "p.pt"))
    assert report["layers_skipped"] == [] and all(torch.equal(v, target.state_dict()[k]) for k, v in donor.state_dict().items())


def test_a_temporal_branch_that_does_not_match_stops_the_run(base_cfg, tmp_path):
    donor = DualBranchNet(_mc(base_cfg, False))
    _save(donor, tmp_path / "p.pt")
    other = _mc(base_cfg, False)
    other["temporal_lstm"]["hidden_size"] = 16
    with pytest.raises(ValueError, match="temporal branch does not match"):
        warm_start(DualBranchNet(other), str(tmp_path / "p.pt"))


def test_freezing_keeps_parts_out_of_training_and_refuses_to_freeze_everything(base_cfg):
    model = DualBranchNet(_mc(base_cfg, True))
    out = freeze_parts(model, ["spatial", "temporal"])
    assert out["frozen"] == ["spatial", "temporal"] and not any(p.requires_grad for p in model.temporal.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    with pytest.raises(ValueError, match="nothing to train"):
        freeze_parts(model, ["head"])
    with pytest.raises(ValueError, match="unknown part"):
        freeze_parts(DualBranchNet(_mc(base_cfg, True)), ["lstm"])
    assert freeze_parts(DualBranchNet(_mc(base_cfg, False)), ["spatial"])["frozen"] == []        # no CNN: nothing to freeze, no error


def test_the_optimiser_gets_only_the_trainable_parameters(base_cfg, tmp_path):
    from training.trainer import Trainer

    mc = _mc(base_cfg, True)
    tc = {**base_cfg["train"], "freeze": ["spatial"], "init_from": None}
    model = build_model(mc, tc)
    trainer = Trainer(model, tc, mc, str(tmp_path))
    n_opt = sum(len(g["params"]) for g in trainer.optimizer.param_groups)
    assert n_opt == sum(1 for p in model.parameters() if p.requires_grad) < sum(1 for _ in model.parameters())


def test_by_default_nothing_is_loaded_or_frozen(base_cfg):
    tc = {**base_cfg["train"]}
    assert tc["init_from"] is None and tc["freeze"] == []
    model = build_model(_mc(base_cfg, True), tc)
    assert all(p.requires_grad for p in model.parameters())


# ---------------------------------------------------------------- source cities
def _city(tmp_path, n_stations=4, hours=500, seed=0):
    """A fake scraped city near Bangkok: locations.csv, sensors/openaq_<id>_*.csv in the Manila layout, source.json."""
    rng = np.random.default_rng(seed)
    city = tmp_path / "thailand" / "bangkok"
    (city / "sensors").mkdir(parents=True)
    locs = []
    stamps = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    for i in range(n_stations):
        sid = 100 + i
        locs.append({"name": f"S{i}", "location_id": sid, "latitude": 13.70 + 0.01 * i, "longitude": 100.50 + 0.01 * i})
        pm = np.clip(25 + np.cumsum(rng.normal(0, 1.5, hours)) * 0.3 + 5 * np.sin(np.arange(hours) * 2 * np.pi / 24), 3, 120)
        pd.DataFrame({"location_name": f"S{i}", "location_id": sid, "sensor_id": sid, "utc_timestamp": stamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "local_timestamp": stamps.tz_convert("Asia/Bangkok").map(lambda t: t.isoformat()), "pm25_value": pm, "parameter": "PM2.5",
                      "unit": "µg/m³"}).to_csv(city / "sensors" / f"openaq_{sid}_S{i}.csv", index=False)
    pd.DataFrame(locs).to_csv(city / "locations.csv", index=False)
    (city / "source.json").write_text(json.dumps({"epsg": 32647, "timezone": "Asia/Bangkok", "city": "thailand/bangkok"}), encoding="utf-8")
    return city


def test_a_source_city_becomes_windows_in_the_manila_feature_order(base_cfg, tmp_path):
    from transfer.source_data import city_windows, list_sources

    cfg = deepcopy(base_cfg)
    cfg["data"]["spatial_lag"].update(k=2)
    city = _city(tmp_path)
    assert list_sources(tmp_path) == ["thailand/bangkok"]
    X, y, meta = city_windows(city, cfg, id_offset=10 ** 9)
    assert X.shape[1:] == (len(cfg["data"]["temporal"]["lags"]), cfg["derived"]["F"]) and len(y) == len(meta) > 1000
    assert meta["location_id"].min() >= 10 ** 9                                        # ids are offset so cities cannot collide
    sl = cfg["derived"]["feature_names"].index("spatial_lag")
    assert np.isfinite(X[:, :, sl]).all()                                              # the lag comes from the other stations of the same city
    Xm, ym, mm = city_windows(city, cfg, exclude_months=[1])
    assert len(ym) == 0                                                                # every hour of the fake city is in January: all excluded


def test_local_hours_follow_the_city_clock(base_cfg, tmp_path):
    from transfer.source_data import load_city

    df = load_city(_city(tmp_path, n_stations=2, hours=48), base_cfg)
    ts = base_cfg["data"]["columns"]["timestamp"]
    assert df[ts].min() == pd.Timestamp("2024-01-01 07:00")                           # 00:00 UTC is 07:00 in Bangkok (UTC+7)


def test_a_folder_without_epsg_asks_for_a_rewrite(tmp_path):
    from transfer.source_data import source_meta

    (tmp_path / "source.json").write_text(json.dumps({"city": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="epsg / timezone"):
        source_meta(tmp_path)


def test_the_ranking_puts_the_station_like_manila_first():
    from transfer.select import rank_stations

    rng = np.random.default_rng(0)
    stamps = pd.date_range("2024-01-01", periods=1500, freq="h")

    def series(sid, mean):
        return pd.DataFrame({"location_id": sid, "timestamp": stamps, "pm25": np.clip(rng.normal(mean, 4, len(stamps)), 1, None)})

    target = pd.concat([series(1, 15), series(2, 15)])
    source = pd.concat([series(10, 15), series(11, 60)])                                # 10 looks like Manila, 11 is far dirtier
    out = rank_stations(source, target, max_n=800)
    assert list(out["location_id"]) == [10, 11] and out["mmd2"].iloc[0] < out["mmd2"].iloc[1]


def test_split_keeps_whole_stations_apart():
    from transfer.pretrain import split_stations

    ids = np.repeat(np.arange(20), 5)
    tr, va = split_stations(ids, 0.2, 1)
    assert len(va) == 4 and not set(tr) & set(va) and len(tr) + len(va) == 20
    assert len(split_stations(np.array([7, 7]), 0.2, 1)[1]) == 0                        # one station: nothing to hold out


def test_the_pretrain_stage_writes_a_checkpoint_the_lstm_only_model_loads(base_cfg, tmp_path):
    """Manila side faked (scalers, hourly matrix, meta_train); two source stations; two epochs."""
    from preprocessing.scalers import Scalers
    from transfer.pretrain import pretrain

    cfg = deepcopy(base_cfg)
    cfg["data"]["spatial_lag"].update(k=2)
    cfg["train"].update(batch_size=128, epochs=2)
    cfg["transfer"].update(source_root=str(tmp_path / "src"), epochs=2, val_station_fraction=0.25)
    _city(tmp_path / "src", n_stations=4, hours=300)
    processed = tmp_path / "processed"
    processed.mkdir()
    cfg["data"]["paths"].update(processed_dir=str(processed), pretrained_dir=str(tmp_path / "pre"))
    dv = cfg["derived"]
    Scalers().fit(np.random.rand(3, dv["C"], 2, 2), np.array([[10.0], [20.0], [30.0]]), dv["pm25_feature_idx"], dv["rolling_std_idx"]).save(str(processed / "scalers.json"))
    np.savez_compressed(processed / "pm25_hourly.npz", values=np.random.rand(300, 2).astype("float32") * 30, location_ids=np.array([1, 2]), start="2024-01-01 00:00:00")
    pd.DataFrame({"location_id": [1, 2]}).to_csv(processed / "meta_train.csv", index=False)

    ckpt = pretrain(cfg, "t", None)
    assert ckpt.name == "pretrained_lstm.pt" and (ckpt.parent / "summary.json").is_file() and (ckpt.parent / "station_ranking.csv").is_file()
    summary = json.loads((ckpt.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["sources"] == ["thailand/bangkok"] and summary["stations_val"] == 1 and summary["feature_names"] == dv["feature_names"]
    fresh = _mc(base_cfg, False)
    fresh["temporal_lstm"].update(input_size=dv["F"], hidden_size=cfg["model"]["temporal_lstm"]["hidden_size"], num_layers=cfg["model"]["temporal_lstm"]["num_layers"])
    fresh.update({k: cfg["model"][k] for k in ("fusion_head",)})
    mc = deepcopy(cfg["model"])
    mc["spatial_cnn"]["enabled"] = False
    report = warm_start(DualBranchNet(mc), str(ckpt))
    assert report["layers_skipped"] == []

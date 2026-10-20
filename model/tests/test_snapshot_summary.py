import json

import numpy as np
import pandas as pd
import pytest

from networks.dual_branch_net import DualBranchNet
from output.aqi_maps import categorize
from reporting import accuracy
from reporting.layout import describe, snap_file
from reporting.snapshot_generator import generate_snapshot
from reporting.summary import _cautions, build_markdown, headline_text, training_summary_text
from training import metrics as M
from training.run_info import new_run_info, save_run_info

RUN = "20260101_000000"
UPPERS = [25, 35, 45, 55]


# ------------------------------------------------------------------------------ the new measures
def test_bias_is_predicted_minus_observed_so_under_prediction_is_negative():
    assert M.bias([10, 20], [8, 19]) == pytest.approx(-1.5)
    assert M.bias([10, 20], [12, 23]) == pytest.approx(2.5)


def test_share_within_a_tolerance_counts_the_boundary_as_inside():
    assert M.fraction_within([10, 10, 10, 10], [10, 15, 16, 4], 5) == 0.5          # errors 0, 5, 6, 6


def test_band_index_uses_the_same_rule_as_the_aqi_maps(cfg):
    values = np.array([0, 24.9, 25, 25.1, 35, 35.1, 45, 55, 55.1, 400])
    assert M.band_index(values, accuracy.aqi_uppers(cfg)).tolist() == categorize(values, cfg).astype(int).tolist()
    assert M.band_index([25, 35], UPPERS).tolist() == [0, 1]                         # an upper bound belongs to the lower band


def test_band_agreement_reports_exact_within_one_and_the_trivial_baseline():
    obs = [10] * 6 + [30] * 3 + [50]                                                  # bands 0 x6, 1 x3, 3 x1
    pred = [10] * 6 + [30, 40, 60] + [50]                                             # ... 1, 2, 4 for the 30s
    a = M.band_agreement(obs, pred, UPPERS)
    assert a["exact"] == pytest.approx(0.8)                                           # 6 + 1 (30 -> band 1) + 1 (50 -> band 3)
    assert a["within_one"] == pytest.approx(0.9)                                      # 40 is one band off, 60 is two
    assert a["majority_baseline"] == pytest.approx(0.6) and a["majority_band"] == 0


def test_measures_carry_the_extras_named_in_the_config(cfg):
    df = pd.DataFrame({"location_id": [1] * 4, "y_true": [10.0, 20.0, 30.0, 40.0], "y_pred": [11.0, 18.0, 31.0, 50.0]})
    m = accuracy.measures(df, cfg)
    assert m["rows"] == 4 and m["stations"] == 1 and {"rmse", "mae", "r2", "bias", "within_5", "within_10"} <= set(m)
    assert m["within_5"] == 0.75 and m["within_10"] == 1.0
    assert m["aqi_majority_band"] == "Good" and 0 <= m["aqi_exact"] <= 1


def test_extras_can_be_switched_off_in_report_yaml(cfg):
    cfg["report"]["accuracy"] = {"tolerances_ugm3": [3], "aqi_bands": False, "persistence": False}
    m = accuracy.measures(pd.DataFrame({"location_id": [1, 1], "y_true": [1.0, 2.0], "y_pred": [1.0, 2.5]}), cfg)
    assert "within_3" in m and "within_5" not in m and "aqi_exact" not in m


# ------------------------------------------------------------------------------ the yardsticks
def _hourly(n=20, drop=None):
    t = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame({"location_id": 7, "timestamp": t, "y_true": np.arange(n, dtype=float) + 10})
    df["y_pred"] = df["y_true"] + 0.5
    return df.drop(index=drop) if drop is not None else df


def test_persistence_scores_the_previous_hour_on_the_same_rows_as_the_model(cfg):
    y = accuracy.persistence_reference(_hourly(), cfg)
    assert y["rows"] == 19                                                            # the first hour has no previous hour
    assert y["persistence"]["rmse"] == pytest.approx(1.0) and y["model"]["rmse"] == pytest.approx(0.5)
    assert y["skill_vs_persistence"] == pytest.approx(0.75)                           # 1 - 0.25 / 1
    assert y["mean_predictor"]["r2"] == 0.0 and y["mean_predictor"]["rmse"] > 4


def test_a_missing_hour_is_not_bridged_by_persistence(cfg):
    y = accuracy.persistence_reference(_hourly(drop=10), cfg)
    assert y["rows"] == 17                                                            # 19 minus the row after the gap


def test_persistence_can_be_switched_off_or_missing_timestamps(cfg):
    assert accuracy.persistence_reference(_hourly().drop(columns="timestamp"), cfg) is None
    cfg["report"]["accuracy"] = {"persistence": False}
    assert accuracy.persistence_reference(_hourly(), cfg) is None


def test_stations_are_listed_worst_first_with_their_network(cfg):
    a = pd.DataFrame({"location_id": 5, "y_true": [10.0, 20.0], "y_pred": [10.5, 20.5]})
    b = pd.DataFrame({"location_id": 9_000_001, "y_true": [10.0, 20.0], "y_pred": [14.0, 24.0]})
    rows = accuracy.by_station(pd.concat([a, b]), cfg, [{"location_id": 5, "coverage": 0.9, "distance_to_nearest_calibration_m": 100.0}])
    assert [r["location_id"] for r in rows] == [9_000_001, 5] and rows[0]["network"] == "QC" and rows[1]["network"] == "OpenAQ"
    assert rows[1]["geocp_coverage"] == 0.9 and "geocp_coverage" not in rows[0]


# ------------------------------------------------------------------------------ the folder layout
def test_files_are_written_into_subfolders_and_read_from_old_flat_snapshots(tmp_path):
    new = snap_file(tmp_path, "predictions_test.csv", write=True)
    assert new == tmp_path / "data" / "predictions_test.csv" and new.parent.is_dir()
    assert snap_file(tmp_path, "metrics_summary.json", write=True) == tmp_path / "metrics_summary.json"   # stays at the top

    (tmp_path / "geocp.json").write_text("{}")                                         # an old snapshot
    assert snap_file(tmp_path, "geocp.json") == tmp_path / "geocp.json"
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "geocp.json").write_text("{}")                                # now both: the new place wins
    assert snap_file(tmp_path, "geocp.json") == tmp_path / "data" / "geocp.json"
    assert snap_file(tmp_path, "calibration_table.csv") == tmp_path / "data" / "calibration_table.csv"   # neither: new place


def test_every_file_the_pipeline_writes_has_a_description_for_the_report():
    for name in ("loss_curve.png", "lisa_on_bands.png", "geocp.json", "accuracy_by_station.csv", "predictions_test.csv"):
        assert describe(name) and describe(f"{__import__('reporting.layout', fromlist=['SUBDIR']).SUBDIR[name]}/{name}")
    assert describe("SUMMARY.md") and describe("nothing.txt") == ""


# ------------------------------------------------------------------------------ the text
def _info(cfg, with_cv=True):
    info = new_run_info(RUN, cfg, DualBranchNet(cfg["model"]), {
        "rows_per_partition": {"train": 1000, "calibration": 300, "test": 400},
        "stations_per_partition": {"train": 6, "calibration": 2, "test": 3}})
    info["created"] = "2026-01-01T00:00:00"
    if with_cv:
        info["cv"] = [{"fold": k, "val_stations": [1, 2], "fit_rows": 600, "val_rows": 400, "best_epoch": 3, "epochs_run": 7,
                       "best_val_loss": 0.2, "metrics": {"rmse": 5.0 + k, "mae": 3.0 + k / 2, "r2": 0.8 - 0.1 * k, "ia": 0.9, "pearson_r": 0.9}}
                      for k in range(3)]
    info["final_fit"] = {"epochs": 4, "epochs_source": "mean best epoch of CV folds", "last_train_loss": 0.31}
    return info


def test_the_training_summary_has_the_fold_table_the_spread_and_the_next_step(cfg):
    text = training_summary_text(_info(cfg))
    assert "TRAINING SUMMARY" in text and RUN in text
    assert "1,000 train rows (6 stations)" in text and "dual-branch CNN-LSTM" in text
    import re
    assert re.search(r"^\s+0\s+2\s+5\.00\s+3\.00\s+0\.800\s+3 of 7$", text, re.M)       # one aligned row per fold
    assert re.search(r"^\s+2\s+2\s+7\.00\s+4\.00\s+0\.600\s+3 of 7$", text, re.M) and "|" not in text
    assert "mean ± sd" in text and "RMSE 6.000 ± 0.816" in text
    assert "fitted on all training stations for 4 epoch" in text and "--stage report" in text


def _summary(coverage=0.87, stations=17, skill=0.11, epochs_override=False):
    test = {"rows": 1000, "stations": stations, "rmse": 5.7, "mae": 3.4, "r2": 0.75, "bias": -1.0, "within_5": 0.8, "within_10": 0.95,
            "aqi_exact": 0.85, "aqi_within_one": 0.98, "aqi_majority_baseline": 0.76, "aqi_majority_band": "Good",
            "ia": 0.92, "pearson_r": 0.87}
    return {"run_id": RUN, "accuracy": {
        "measures": {"test": test, "calibration": {**test, "stations": 5}, "cross_validation": {**test, "rmse": 6.0, "stations": 40}},
        "yardsticks": {"rows": 900, "share_of_test_rows": 0.9, "model": {"rmse": 5.7, "mae": 3.4, "r2": 0.75},
                       "persistence": {"rmse": 6.1, "mae": 3.5, "r2": 0.72}, "mean_predictor": {"rmse": 11.0, "mae": 8.0, "r2": 0.0},
                       "skill_vs_persistence": skill},
        "by_station": [{"location_id": 1, "name": "Alpha", "network": "OpenAQ", "rows": 500, "rmse": 6.0, "mae": 3.5, "r2": 0.7,
                        "bias": -1.2, "geocp_coverage": coverage}]},
        "geocp": {"selected": {"kernel": "gaussian", "bandwidth_m": 5000.0, "loso_coverage": 0.91},
                  "test": {"coverage": coverage, "target_coverage": 0.9, "mean_width": 13.0, "median_width": 12.5,
                           "mean_width_to_prediction_ratio": 1.1, "coverage_urban_core": 0.9, "coverage_peripheral": 0.84}},
        "spatial_bias": {"cells": 30, "global_moran": {"I": 0.1, "p_sim": 0.2, "significant": False}}}


def test_the_headline_gives_accuracy_the_yardstick_and_the_interval_in_a_few_lines(tmp_path):
    text = headline_text(_summary(), tmp_path)
    assert "ACCURACY on 17 test stations" in text and "R² 0.750" in text and "RMSE 5.70" in text and "bias -1.00" in text
    assert "within ±5 µg/m³: 80.0 %" in text and "same AQI band: 85.0 %" in text and "'Good' would give 76.0 %" in text
    assert "RMSE 5.70 against 6.10" in text and "skill +11%" in text
    assert "contain the observation 87.0 %" in text and "SUMMARY.md" in text


def test_the_report_has_every_section_and_explains_each_number(cfg):
    md = build_markdown(_summary(), _info(cfg), ["SUMMARY.md", "plots/loss_curve.png"])
    for heading in ("## 1. Headline", "## 2. Is that good?", "## 3. Does the accuracy hold", "## 4. Accuracy by test station",
                    "## 5. Uncertainty", "## 6. Are the errors spatially clustered", "## 7. Training", "## 8. Read with care",
                    "## 9. Files in this folder"):
        assert heading in md
    assert "| R² | 0.750 |" in md and "Guessing 'Good' every time scores 76.0 %" in md
    assert "Repeat the previous hour (persistence)" in md and "Skill against persistence: **+11%**" in md
    assert "5 km gaussian kernel" in md and "| Alpha | OpenAQ |" in md and "`plots/loss_curve.png` | training and validation loss" in md


def test_the_cautions_name_what_a_reader_should_not_over_trust(cfg):
    info = _info(cfg)
    notes = " ".join(_cautions(_summary(coverage=0.80, stations=9, skill=-0.1), {**info, "overrides": ["train.epochs=2"]}))
    assert "train.epochs=2" in notes and "Only 9 test stations" in notes
    assert "NOT more accurate than repeating the previous hour" in notes and "under-cover" in notes and "AQI bands use" in notes
    quiet = " ".join(_cautions(_summary(coverage=0.89, stations=30, skill=0.2), {**info, "overrides": []}))
    assert "Only" not in quiet and "under-cover" not in quiet and "NOT more accurate" not in quiet


def test_a_test_error_far_from_cross_validation_is_called_out_in_the_right_direction(cfg):
    worse = _summary()
    worse["accuracy"]["measures"]["cross_validation"]["rmse"] = 4.0                  # test 5.7 is 42 % above
    assert "ABOVE the cross-validation RMSE" in build_markdown(worse, _info(cfg), ["SUMMARY.md"])
    better = _summary()
    better["accuracy"]["measures"]["cross_validation"]["rmse"] = 9.0                 # test is 37 % below
    md = build_markdown(better, _info(cfg), ["SUMMARY.md"])
    assert "below the cross-validation RMSE" in md and "ABOVE" not in md


# ------------------------------------------------------------------------------ the whole snapshot
def _make_run(cfg, tmp_path, flat: bool):
    cfg["data"]["paths"].update(checkpoints_dir=str(tmp_path / "checkpoints"), snapshots_dir=str(tmp_path / "snapshots"))
    ckpt, snap = tmp_path / "checkpoints" / RUN, tmp_path / "snapshots" / RUN
    ckpt.mkdir(parents=True)
    (snap if flat else snap / "data").mkdir(parents=True)
    rng = np.random.default_rng(0)
    t = pd.date_range("2025-03-01", periods=240, freq="h")

    def frame(ids):
        rows = []
        for sid in ids:
            truth = 22 + 8 * np.sin(np.arange(len(t)) / 9.0) + rng.normal(0, 1, len(t))
            rows.append(pd.DataFrame({"location_id": sid, "timestamp": t, "y_true": truth,
                                      "y_pred": truth + rng.normal(-1.0, 3.0, len(t))}))
        return pd.concat(rows, ignore_index=True)

    test, cal = frame([11, 22, 33]), frame([44, 55])
    test["q"], test["lo_raw"], test["hi"] = 6.0, test["y_pred"] - 6.0, test["y_pred"] + 6.0
    test["lo"], test["width"] = test["lo_raw"].clip(lower=0), test["hi"] - test["lo_raw"].clip(lower=0)
    test["covered"] = (test["y_true"] >= test["lo_raw"]) & (test["y_true"] <= test["hi"])
    test["zone"] = "urban_core"
    out = snap if flat else snap / "data"
    test.to_csv(out / "predictions_test.csv", index=False)
    cal.to_csv(out / "predictions_calibration.csv", index=False)
    per_station = [{"location_id": s, "rows": 240, "coverage": float(g["covered"].mean()), "half_width": 6.0,
                    "distance_to_nearest_calibration_m": 1500.0} for s, g in test.groupby("location_id")]
    (out / "geocp.json").write_text(json.dumps({
        "selected": {"kernel": "gaussian", "bandwidth_m": 5000.0, "alpha": 0.1, "finite_sample_correction": False,
                     "loso_coverage": 0.9, "loso_mean_width": 12.0, "met_coverage_target": True},
        "test": {"target_coverage": 0.9, "coverage": float(test["covered"].mean()), "mean_width": 12.0, "median_width": 12.0,
                 "mean_width_to_prediction_ratio": 0.5, "coverage_urban_core": 0.9, "coverage_peripheral": 0.9,
                 "zone_threshold_m": 1500.0, "per_station": per_station}}))

    info = _info(cfg)
    save_run_info(ckpt, info)
    hist = {"train_loss": [0.5, 0.3, 0.2], "val_loss": [0.6, 0.4, 0.35], "best_epoch": 3, "best_val_loss": 0.35, "epochs_run": 3}
    (ckpt / "history.json").write_text(json.dumps({"folds": [hist] * 3, "final": hist}))
    oof = frame([1, 2, 3, 4, 5, 6])
    oof["fold"] = oof["location_id"] % 3
    oof.to_csv(ckpt / "oof_predictions.csv", index=False)
    return snap


@pytest.mark.parametrize("flat", [False, True], ids=["new layout", "old flat layout"])
def test_a_snapshot_is_written_into_subfolders_with_a_report_whichever_layout_it_reads(cfg, tmp_path, flat):
    snap = _make_run(cfg, tmp_path, flat)
    generate_snapshot(RUN, cfg)

    for rel in ("SUMMARY.md", "metrics_summary.json", "run_info.json", "plots/pred_vs_obs.png", "plots/loss_curve.png",
                "plots/residual_hist.png", "plots/timeseries_sample.png", "plots/geocp_coverage.png",
                "plots/accuracy_by_station.png", "tables/accuracy_summary.csv", "tables/accuracy_by_station.csv", "tables/cv_folds.csv"):
        assert (snap / rel).is_file() and (snap / rel).stat().st_size > 50, rel
    assert not (snap / "pred_vs_obs.png").exists() and not (snap / "loss_curve.png").exists()     # plots are no longer in the top folder

    summary = json.loads((snap / "metrics_summary.json").read_text(encoding="utf-8"))
    assert summary["test"]["rmse"] > 0 and "supplementary" in summary["test"]                   # the old keys are still there
    t = summary["accuracy"]["measures"]["test"]
    assert t["bias"] == pytest.approx(-1.0, abs=0.4) and 0.5 < t["within_5"] < 1.0 and t["stations"] == 3
    assert set(summary["accuracy"]["measures"]) == {"cross_validation", "calibration", "test"}
    assert summary["accuracy"]["yardsticks"]["rows"] > 600 and len(summary["accuracy"]["by_station"]) == 3

    table = pd.read_csv(snap / "tables" / "accuracy_summary.csv")
    assert table["split"].tolist() == ["cross-validation (out-of-fold)", "calibration", "test"]
    md = (snap / "SUMMARY.md").read_text(encoding="utf-8")
    assert "## 1. Headline" in md and "plots/loss_curve.png" in md and "`tables/accuracy_summary.csv`" in md
    assert (tmp_path / "checkpoints" / "runs_index.csv").is_file()


def test_report_prints_the_headline_and_where_the_report_is(cfg, tmp_path, capsys, monkeypatch):
    import run_pipeline
    _make_run(cfg, tmp_path, flat=False)
    monkeypatch.setattr(run_pipeline, "_ensure_geocp", lambda *a, **k: None)
    monkeypatch.setattr(run_pipeline, "bias", lambda *a, **k: None)
    run_pipeline.report(cfg, RUN)
    out = capsys.readouterr().out
    assert "ACCURACY on 3 test stations" in out and "UNCERTAINTY" in out and "SUMMARY.md" in out and "snapshot written to" in out

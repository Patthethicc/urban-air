import json
from pathlib import Path

import pytest

import run_pipeline
from config_parser import apply_overrides
from training.tuning import apply_hyperparameters, hp_to_overrides

HP = {"learning_rate": 0.005, "batch_size": 32, "epochs": 7, "cnn_kernel_size": 5, "cnn_filters": 16,
      "lstm_hidden_size": 48, "recurrent_dropout": 0.25}


# ------------------------------------------------------------------------------ infer needs the GeoCP calibration
def _cfg_with_snapshots(cfg, tmp_path):
    cfg["data"]["paths"]["snapshots_dir"] = str(tmp_path / "snapshots")
    (tmp_path / "snapshots" / "RUN").mkdir(parents=True)
    return cfg


def test_a_missing_geocp_calibration_is_created_instead_of_failing(cfg, tmp_path, monkeypatch, capsys):
    cfg = _cfg_with_snapshots(cfg, tmp_path)
    calls = []
    monkeypatch.setattr(run_pipeline, "geocp", lambda c, rid: calls.append(rid))
    run_pipeline._ensure_geocp(cfg, "RUN")
    assert calls == ["RUN"] and "running --stage geocp first" in capsys.readouterr().out


def test_an_existing_calibration_is_not_recomputed(cfg, tmp_path, monkeypatch):
    cfg = _cfg_with_snapshots(cfg, tmp_path)
    for name in ("geocp.json", "calibration_table.csv"):
        (tmp_path / "snapshots" / "RUN" / name).write_text("x")
    calls = []
    monkeypatch.setattr(run_pipeline, "geocp", lambda c, rid: calls.append(rid))
    run_pipeline._ensure_geocp(cfg, "RUN")
    assert calls == []


def test_one_missing_calibration_file_is_enough_to_rerun_geocp(cfg, tmp_path, monkeypatch):
    cfg = _cfg_with_snapshots(cfg, tmp_path)
    (tmp_path / "snapshots" / "RUN" / "geocp.json").write_text("x")            # the table is missing
    calls = []
    monkeypatch.setattr(run_pipeline, "geocp", lambda c, rid: calls.append(rid))
    run_pipeline._ensure_geocp(cfg, "RUN")
    assert calls == ["RUN"]


# ------------------------------------------------------------------------------ geocp.enabled: false
def test_geocp_is_on_unless_switched_off(cfg):
    assert run_pipeline._geocp_on(cfg)
    cfg["geocp"]["enabled"] = False
    assert not run_pipeline._geocp_on(cfg)


def test_with_geocp_off_only_the_predictions_are_ensured(cfg, tmp_path, monkeypatch):
    cfg = _cfg_with_snapshots(cfg, tmp_path)
    cfg["geocp"]["enabled"] = False
    calls = []
    monkeypatch.setattr(run_pipeline, "geocp", lambda c, rid: calls.append(rid))
    run_pipeline._ensure_geocp(cfg, "RUN")                                      # no predictions yet: the stage writes them
    assert calls == ["RUN"]
    for name in ("predictions_test.csv", "predictions_calibration.csv"):
        (tmp_path / "snapshots" / "RUN" / name).write_text("x")
    run_pipeline._ensure_geocp(cfg, "RUN")                                      # no calibration table is asked for
    assert calls == ["RUN"]


def test_surface_and_the_geocp_stage_still_run_geocp_when_switched_off(cfg, tmp_path, monkeypatch):
    cfg["geocp"]["enabled"] = False
    forced = run_pipeline._geocp_forced(cfg)
    assert run_pipeline._geocp_on(forced) and not run_pipeline._geocp_on(cfg)      # the original is not changed
    seen = []
    monkeypatch.setattr(run_pipeline, "_ensure_geocp", lambda c, rid: seen.append(run_pipeline._geocp_on(c)))
    monkeypatch.setattr(run_pipeline, "_load_geocp", lambda c, rid: object())
    import uncertainty.geocp_surface as gs
    monkeypatch.setattr(gs, "build_interval_rasters", lambda *a, **k: {})
    cfg = _cfg_with_snapshots(cfg, tmp_path)
    run_pipeline.surface(cfg, "RUN", "pm25.tif")
    assert seen == [True]


# ------------------------------------------------------------------------------ tuned values become overrides
def test_the_tuned_overrides_rebuild_exactly_the_tuned_configuration(cfg):
    """Pasting `as_overrides` into a normal run gives the same settings as --tuned."""
    via_tuned = apply_hyperparameters(cfg, HP)
    via_overrides = apply_overrides(json.loads(json.dumps(cfg)), hp_to_overrides(HP, cfg))
    assert via_overrides["train"]["optimizer"]["lr"] == via_tuned["train"]["optimizer"]["lr"] == 0.005
    assert via_overrides["train"]["batch_size"] == via_tuned["train"]["batch_size"] == 32
    assert via_overrides["train"]["epochs"] == via_tuned["train"]["epochs"] == 7 and via_overrides["train"]["final_epochs"] == 7
    assert via_overrides["model"] == via_tuned["model"]


def test_overrides_cover_every_searched_name(cfg):
    names = {o.split("=")[0] for o in hp_to_overrides({**HP, "cnn_feature_map": "flatten"}, cfg)}
    assert names == {"train.optimizer.lr", "train.batch_size", "train.epochs", "train.final_epochs",
                     "model.spatial_cnn.kernel_size", "model.spatial_cnn.conv_channels", "model.spatial_cnn.feature_map",
                     "model.temporal_lstm.hidden_size", "model.temporal_lstm.dropout"}


# ------------------------------------------------------------------------------ the clean command
def _clean_cfg(root):
    paths = {k: str(root / v) for k, v in (("processed_dir", "processed"), ("checkpoints_dir", "checkpoints"),
                                           ("snapshots_dir", "snapshots"), ("tuning_dir", "tuning"),
                                           ("raster_work_dir", "processed/rasters"))}
    return {"data": {"paths": paths}, "output": {"basemap_cache_dir": str(root / ".tile_cache")}}


@pytest.fixture
def populated(tmp_path):
    root = tmp_path / "model"
    for folder in ("processed", "checkpoints", "snapshots", "tuning"):
        (root / folder).mkdir(parents=True)
        (root / folder / ".gitkeep").touch()
    (root / "processed" / "patches.npy").write_text("x" * 2048)
    (root / "processed" / "rasters").mkdir()
    (root / "processed" / "rasters" / "master.tif").write_text("r")
    (root / "checkpoints" / "run1").mkdir()
    (root / "checkpoints" / "run1" / "model.pt").write_text("m" * 1024)
    return root


def test_clean_is_a_dry_run_unless_yes_is_given(populated, capsys):
    run_pipeline.clean(_clean_cfg(populated), execute=False, root=populated)
    out = capsys.readouterr().out
    assert "Would delete (dry run)" in out and "Dry run: nothing was deleted" in out and "--yes" in out
    assert (populated / "processed" / "patches.npy").exists() and (populated / "checkpoints" / "run1").exists()


def test_clean_yes_deletes_and_reports_what_it_freed(populated, capsys):
    run_pipeline.clean(_clean_cfg(populated), execute=True, root=populated)
    out = capsys.readouterr().out
    assert "Deleted 2 files" in out and "--stage preprocess" in out
    assert not (populated / "processed" / "patches.npy").exists() and not (populated / "checkpoints" / "run1").exists()
    assert (populated / "processed" / ".gitkeep").exists()
    assert (populated / "processed" / "rasters" / "master.tif").exists()                    # protected by default


def test_clean_says_when_there_is_nothing_to_do(tmp_path, capsys):
    root = tmp_path / "model"
    root.mkdir()
    run_pipeline.clean(_clean_cfg(root), execute=True, root=root)
    assert "Nothing to clean" in capsys.readouterr().out


def test_clean_keep_and_include_rasters_work_through_the_command(populated, capsys):
    run_pipeline.clean(_clean_cfg(populated), execute=True, keep=("checkpoints",), include_rasters=True, root=populated)
    assert (populated / "checkpoints" / "run1" / "model.pt").exists()
    assert not (populated / "processed" / "rasters").exists()


def test_clean_and_its_reset_alias_are_real_stages():
    import re
    src = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    choices = set(re.findall(r'"(\w+)"', re.search(r'choices=\[(.*?)\]\)', src, re.S).group(1)))
    assert {"clean", "reset"} <= choices


def test_no_geocp_function_runs_in_the_geocp_stage_when_it_is_off(cfg, tmp_path, monkeypatch):
    """Every GeoCP entry point raises: with geocp.enabled false the stage must only predict the test stations and write the files."""
    import pandas as pd
    import training.checkpoint, training.predict, uncertainty.geocp as g

    cfg = _cfg_with_snapshots(cfg, tmp_path)
    cfg["geocp"]["enabled"] = False
    cfg["data"]["paths"]["checkpoints_dir"] = str(tmp_path / "ckpt")
    cfg["data"]["paths"]["processed_dir"] = str(tmp_path / "processed")
    (tmp_path / "ckpt" / "RUN").mkdir(parents=True)
    (tmp_path / "processed").mkdir()
    pd.DataFrame({"location_id": [1], "x": [0.0], "y": [0.0], "cell_x": [0.0], "cell_y": [0.0]}).to_csv(tmp_path / "processed" / "stations.csv", index=False)

    def boom(*a, **k):
        raise AssertionError("a GeoCP function ran with geocp.enabled false")
    for name in ("GeoCP", "select_config", "apply_intervals", "calibration_table", "scores_from_predictions"):
        monkeypatch.setattr(g, name, boom)
    asked = []
    monkeypatch.setattr(training.checkpoint, "load_model", lambda path: (object(), {}))
    monkeypatch.setattr(training.checkpoint, "check_compatibility", lambda *a, **k: None)
    monkeypatch.setattr(training.predict, "predict_partition", lambda m, c, part: asked.append(part) or pd.DataFrame({"location_id": [1], "y_true": [1.0], "y_pred": [1.0]}))
    monkeypatch.setattr(run_pipeline, "_cell_stations", lambda s: s)
    run_pipeline.geocp(cfg, "RUN")
    snap = tmp_path / "snapshots" / "RUN"
    assert asked == ["test"]                                                      # no calibration partition is predicted
    assert (snap / "data" / "predictions_test.csv").exists() or (snap / "predictions_test.csv").exists()
    assert not (snap / "data" / "calibration_table.csv").exists() and not (snap / "calibration_table.csv").exists()


# ------------------------------------------------------------------------------ train.resume
def test_a_stopped_run_is_found_and_a_finished_one_is_not(cfg, tmp_path):
    import json
    cfg["data"]["paths"]["checkpoints_dir"] = str(tmp_path)
    for name, status, model in (("20260101_000000", "completed", True), ("20260102_000000", "timed_out", False), ("20260103_000000", "running", False)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "run_info.json").write_text(json.dumps({"status": status}))
        if model:
            (tmp_path / name / run_pipeline.MODEL_FILE).write_text("x")
    assert run_pipeline._unfinished_run(cfg) == "20260103_000000"                 # the newest stopped one
    (tmp_path / "20260103_000000" / run_pipeline.MODEL_FILE).write_text("x")      # finished after all
    assert run_pipeline._unfinished_run(cfg) == "20260102_000000"


def test_a_finished_fold_is_saved_and_loaded_back(tmp_path):
    import pandas as pd
    oof = pd.DataFrame({"location_id": [1, 2], "timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00"]), "fold": 0,
                        "y_true": [1.0, 2.0], "y_pred": [1.5, 2.5]})
    run_pipeline._save_fold(tmp_path, 0, {"best_epoch": 3, "epochs_run": 5}, {"fold": 0, "metrics": {"rmse": 4.0, "r2": 0.7}}, oof)
    hist, entry, back = run_pipeline._load_fold(tmp_path, 0)
    assert hist["best_epoch"] == 3 and entry["metrics"]["r2"] == 0.7 and back["y_pred"].tolist() == [1.5, 2.5]
    assert run_pipeline._load_fold(tmp_path, 1) is None                            # a fold that never finished

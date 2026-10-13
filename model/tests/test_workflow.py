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

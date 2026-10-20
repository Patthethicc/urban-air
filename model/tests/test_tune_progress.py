import json
import time

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

import runtime_guard as rg
from preprocessing.scalers import Scalers
from runtime_guard import configure, guarded
from training import tuning
from training.tuning import gp_search, make_objective, nested_tuning

LR_SPACE = {"learning_rate": {"type": "real", "low": 1e-4, "high": 1e-2, "prior": "log-uniform"}}


@pytest.fixture(autouse=True)
def _fresh_limits():
    configure({})
    tuning._PACE_PENDING[0] = True
    yield
    configure({})


def _tune(**kw):
    return {"n_calls": 3, "n_initial_points": 2, "acq_func": "EI", "random_state": 0, **kw}


# ------------------------------------------------------------------------------ the search lines
def test_a_label_aware_objective_is_told_which_trial_it_runs_and_a_plain_one_still_works():
    seen = []

    def aware(hp, label=""):
        seen.append(label)
        return hp["learning_rate"], {}

    gp_search(aware, LR_SPACE, _tune(), tag="[outer 0]")
    assert seen == ["[outer 0] trial 1/3", "[outer 0] trial 2/3", "[outer 0] trial 3/3"]
    assert len(gp_search(lambda hp: (hp["learning_rate"], {}), LR_SPACE, _tune())["trials"]) == 3     # no label argument


def test_every_trial_line_gives_its_duration_and_the_time_left_and_the_last_has_none_left(capsys):
    def slow(hp):
        time.sleep(0.05)
        return hp["learning_rate"], {}

    gp_search(slow, LR_SPACE, _tune(), tag="[final]")
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("[final] trial")]
    assert len(lines) == 3 and all("took" in x for x in lines)
    assert "left in this search" in lines[0] and "left in this search" in lines[1] and "left in this search" not in lines[2]


def test_the_pace_projection_is_printed_once_with_the_number_of_searches(capsys):
    objective = lambda hp: (hp["learning_rate"], {})
    gp_search(objective, LR_SPACE, _tune(), tag="[outer 0]", searches=4)
    gp_search(objective, LR_SPACE, _tune(), tag="[outer 1]", searches=4)
    out = capsys.readouterr().out
    assert out.count("at this pace one search of 3 trials takes about") == 1 and "4 searches about" in out
    assert "evaluation fits on all rows come on top" in out


# ------------------------------------------------------------------------------ the fits inside a trial
def _toy_objective(cfg):
    d = cfg["derived"]
    g = torch.Generator().manual_seed(0)
    spatial, temporal = torch.randn(64, d["C"], 2, 2, generator=g), torch.randn(64, d["T"], d["F"], generator=g)
    ds = TensorDataset(spatial, temporal, temporal[:, -1, :1].clone())
    cfg["tune"].update(search_epochs=False, max_epochs=3, patience=1)
    cfg["train"]["device"] = "cpu"
    folds = [(np.arange(0, 32), np.arange(32, 64)), (np.arange(32, 64), np.arange(0, 32))]

    class NoRecompute:
        def apply(self, ds, fit, val):
            return ds, fit, val
    return make_objective(cfg, ds, folds, stride=1, seed=0, recomputer=NoRecompute())


def test_each_finished_fit_prints_a_line_naming_its_trial_and_fold(cfg, capsys):
    _toy_objective(cfg)({"learning_rate": 0.01, "batch_size": 16}, label="[final] trial 2/12")
    out = capsys.readouterr().out
    assert "[final] trial 2/12 fit 1/2: " in out and "[final] trial 2/12 fit 2/2: " in out
    assert "epochs (best" in out and "val loss" in out


def test_a_quiet_fit_gives_a_position_line_every_interval_and_names_trial_fit_and_epoch(cfg, capsys):
    configure({"status_every_s": 0.001})
    _toy_objective(cfg)({"learning_rate": 0.01, "batch_size": 16}, label="[outer 0] trial 3/12")
    out = capsys.readouterr().out
    status = [x for x in out.splitlines() if x.startswith("  ... ")]
    assert status and all("running for" in x for x in status)
    assert any("[outer 0] trial 3/12 fit 1/2 epoch 1/3, batch" in x for x in status)


# ------------------------------------------------------------------------------ the guard's status lines
def test_quiet_loops_print_a_rate_limited_position_line_and_noisy_ones_print_none(capsys):
    configure({"status_every_s": 0.25})
    with guarded("quiet loop", quiet=True) as g:
        for i in range(12):
            time.sleep(0.1)
            g.beat(f"batch {i}")
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("  ... ")]
    assert 3 <= len(lines) <= 5                                       # 1.2 s at one line per 0.25 s, not one per batch
    with guarded("noisy loop") as g:
        for i in range(5):
            time.sleep(0.1)
            g.beat(f"batch {i}")
    assert capsys.readouterr().out == ""


def test_position_lines_can_be_switched_off(capsys):
    configure({"status_every_s": 0})
    with guarded("quiet loop", quiet=True) as g:
        time.sleep(0.2)
        g.beat("batch 1")
    assert capsys.readouterr().out == ""


def test_normal_training_prints_epoch_lines_but_no_extra_position_lines(cfg, tmp_path, capsys):
    from torch.utils.data import DataLoader
    from networks.dual_branch_net import DualBranchNet
    from training.trainer import Trainer
    configure({"status_every_s": 0.001})
    d = cfg["derived"]
    g = torch.Generator().manual_seed(0)
    ds = TensorDataset(torch.randn(32, d["C"], 2, 2, generator=g), torch.randn(32, d["T"], d["F"], generator=g), torch.randn(32, 1, generator=g))
    trainer = Trainer(DualBranchNet(cfg["model"]), {**cfg["train"], "device": "cpu"}, cfg["model"], str(tmp_path))
    trainer.fit(DataLoader(ds, batch_size=8), None, epochs=2, verbose=True, tag="[final]")
    out = capsys.readouterr().out
    assert "[final] epoch   1/2" in out and "running for" not in out


# ------------------------------------------------------------------------------ the whole tune stage
def _tiny_processed(cfg, folder, stations=8, rows=24):
    d = cfg["derived"]
    rng = np.random.default_rng(0)
    patches = rng.normal(size=(stations, d["C"], d["patch_px"], d["patch_px"])).astype("float32")
    X = rng.normal(size=(stations * rows, d["T"], d["F"])).astype("float32")
    y = X[:, -1, :1].copy()
    np.save(folder / "patches.npy", patches)
    np.save(folder / "station_idx_train.npy", np.repeat(np.arange(stations), rows))
    np.save(folder / "X_temporal_train.npy", X)
    np.save(folder / "y_train.npy", y)
    pd.DataFrame({"location_id": np.repeat(np.arange(1, stations + 1) * 10, rows),
                  "timestamp": np.tile(pd.date_range("2025-01-01", periods=rows, freq="h"), stations)}).to_csv(
        folder / "meta_train.csv", index=False)
    Scalers().fit(patches, y, [0], []).save(str(folder / "scalers.json"))


def test_the_whole_tune_stage_reports_its_plan_every_search_every_trial_and_the_evaluation_fits(small_cfg, tmp_path, capsys):
    cfg = small_cfg
    processed, tuning_dir = tmp_path / "processed", tmp_path / "tuning"
    processed.mkdir()
    _tiny_processed(cfg, processed)
    cfg["data"]["paths"].update(processed_dir=str(processed), tuning_dir=str(tuning_dir))
    cfg["train"]["device"] = "cpu"
    cfg["tune"].update(outer_folds=2, inner_folds=2, n_calls=2, n_initial_points=2, row_stride=1, max_epochs=2, patience=1)
    configure({"status_every_s": 0.001})

    folder = nested_tuning(cfg)
    out = capsys.readouterr().out

    assert "tuning plan: 3 searches x 2 trials x 2 inner fits = 12 fits on all rows, plus 2 evaluation fits" in out
    assert "a 'still running' line every" in out
    for header in ("--- outer fold 0 (search 1/3)", "--- outer fold 1 (search 2/3)", "(search 3/3)"):
        assert header in out
    for line in ("[outer 0] trial 1/2", "[outer 1] trial 2/2", "[final] trial 2/2", "[outer 0] trial 1/2 fit 1/2: "):
        assert line in out
    assert out.count("at this pace one search of 2 trials") == 1 and "3 searches about" in out
    assert "[outer 0] evaluation fit with the winning hyperparameters" in out
    import re
    assert re.search(r"^\[outer 0\] evaluation fit epoch +\d+/\d+  train \d", out, re.M)       # the real epoch line, not a status line
    assert "tuned model on held-out stations: RMSE" in out
    assert any(x.startswith("  ... [") for x in out.splitlines())                 # a position line from a quiet fit
    best = json.loads((folder / "best_hyperparameters.json").read_text(encoding="utf-8"))
    assert "hyperparameters" in best and best["as_overrides"]


def test_the_shipped_status_interval_is_five_minutes(base_cfg):
    assert base_cfg["train"]["runtime"]["status_every_s"] == 300 and rg.status_every_s() >= 0

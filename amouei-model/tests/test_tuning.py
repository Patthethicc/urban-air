import numpy as np
import pandas as pd

from preprocessing.spatial_partitioner import nested_spatial_folds
import torch
from torch.utils.data import TensorDataset

from training.tuning import apply_hyperparameters, build_space, effective_space, gp_search, make_objective

HP = {"learning_rate": 0.005, "batch_size": 32, "epochs": 7, "cnn_kernel_size": 5, "cnn_filters": 16,
      "lstm_hidden_size": 48, "recurrent_dropout": 0.25}


def test_every_searched_name_lands_in_the_config(cfg):
    out = apply_hyperparameters(cfg, HP)
    assert out["train"]["optimizer"]["lr"] == 0.005 and out["train"]["batch_size"] == 32 and out["train"]["epochs"] == 7
    assert out["model"]["spatial_cnn"]["kernel_size"] == 5
    assert out["model"]["spatial_cnn"]["conv_channels"] == [16, 32, 64]           # doubles per layer
    assert out["model"]["temporal_lstm"]["hidden_size"] == 48 and out["model"]["temporal_lstm"]["dropout"] == 0.25
    assert cfg["train"]["epochs"] != 7                                              # original untouched


def test_search_space_matches_the_three_hyperparameter_groups(cfg):
    names = {d.name for d in build_space(cfg["tune"]["space"])}
    assert {"learning_rate", "batch_size"} <= names                                 # global
    assert "epochs" not in names and cfg["tune"]["max_epochs"] > 0                  # early stopping decides
    assert {"cnn_kernel_size", "cnn_filters"} <= names                              # spatial
    assert {"lstm_hidden_size", "recurrent_dropout"} <= names                       # temporal


def test_recurrent_dropout_is_not_inert(cfg):
    """nn.LSTM ignores dropout with a single layer, so the search is only meaningful with >= 2."""
    assert cfg["model"]["temporal_lstm"]["num_layers"] >= 2


def test_nested_folds_keep_stations_apart():
    meta = pd.DataFrame({"location_id": np.repeat(range(1, 13), 5)})
    for fit, test, inner in nested_spatial_folds(meta, 3, 3):
        outer_test = set(meta.iloc[test]["location_id"])
        assert not outer_test & set(meta.iloc[fit]["location_id"])
        for inner_fit, inner_val in inner:
            assert set(inner_fit) <= set(fit) and set(inner_val) <= set(fit)        # never reaches the outer test set
            assert not set(meta.iloc[inner_fit]["location_id"]) & set(meta.iloc[inner_val]["location_id"])


def test_gp_search_improves_on_its_first_point_and_returns_native_types(cfg):
    space = {"learning_rate": {"type": "real", "low": 1e-4, "high": 1e-2, "prior": "log-uniform"},
             "epochs": {"type": "integer", "low": 5, "high": 25}}
    tune = {"n_calls": 14, "n_initial_points": 5, "acq_func": "EI", "random_state": 0}
    objective = lambda hp: ((np.log10(hp["learning_rate"]) + 3) ** 2 + abs(hp["epochs"] - 15) / 10, {})
    res = gp_search(objective, space, tune)
    assert len(res["trials"]) == 14
    assert res["best_loss"] <= res["trials"][0]["loss"]
    assert type(res["best_hyperparameters"]["epochs"]) is int


def test_epoch_search_is_off_by_default(cfg):
    assert cfg["tune"]["search_epochs"] is False
    assert "epochs" not in effective_space(cfg["tune"])


def test_enabling_the_switch_adds_epochs_to_the_search(cfg):
    cfg["tune"]["search_epochs"] = True
    names = {d.name for d in build_space(effective_space(cfg["tune"]))}
    assert "epochs" in names and "learning_rate" in names


def test_bad_epoch_range_is_rejected(cfg):
    import pytest
    from config_parser import apply_overrides, validate
    cfg = apply_overrides(cfg, ["tune.search_epochs=true", "tune.epochs_space.low=30"])
    with pytest.raises(ValueError):
        validate(cfg)


class _NoRecompute:
    def apply(self, ds, fit, val):
        return ds, fit, val


def _toy_objective(cfg, searched):
    d = cfg["derived"]
    g = torch.Generator().manual_seed(0)
    spatial, temporal = torch.randn(64, d["C"], 2, 2, generator=g), torch.randn(64, d["T"], d["F"], generator=g)
    ds = TensorDataset(spatial, temporal, temporal[:, -1, :1].clone())
    cfg["tune"].update(search_epochs=searched, max_epochs=6, patience=1)
    cfg["train"]["device"] = "cpu"
    folds = [(np.arange(0, 32), np.arange(32, 64)), (np.arange(32, 64), np.arange(0, 32))]
    return make_objective(cfg, ds, folds, stride=1, seed=0, recomputer=_NoRecompute())


def test_fixed_epoch_mode_trains_exactly_the_searched_epochs(cfg):
    obj = _toy_objective(cfg, searched=True)
    loss, details = obj({"learning_rate": 0.01, "batch_size": 16, "epochs": 3})
    assert details["epochs_run"] == [3, 3] and "suggested_epochs" not in details
    assert loss == sum(details["fold_losses"]) / 2


def test_early_stopping_mode_suggests_epochs_and_respects_the_cap(cfg):
    obj = _toy_objective(cfg, searched=False)
    loss, details = obj({"learning_rate": 0.01, "batch_size": 16})
    assert all(n <= 6 for n in details["epochs_run"])
    assert 1 <= details["suggested_epochs"] <= 6

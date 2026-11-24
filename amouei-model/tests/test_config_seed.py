import numpy as np
import torch

from config_parser import apply_overrides, load_config
from seed import set_seed


def test_config_loads_all_sections():
    cfg = load_config()
    assert {"data", "model", "train", "report", "geocp", "derived"} <= set(cfg)


def test_derived_values_follow_the_config(cfg):
    d = cfg["derived"]
    assert d["T"] == len(cfg["data"]["temporal"]["lags"])
    assert d["F"] == len(d["feature_names"])
    assert cfg["model"]["temporal_lstm"]["input_size"] == d["F"]
    assert cfg["model"]["spatial_cnn"]["in_channels"] == d["C"]


def test_rolling_and_cyclical_change_F(cfg):
    base = cfg["derived"]["F"]
    cfg = apply_overrides(cfg, ["temporal.rolling_window=3".replace("temporal", "data.temporal")])
    assert cfg["derived"]["F"] == base + 2
    assert cfg["derived"]["rolling_std_idx"]


def test_unknown_override_key_raises(cfg):
    try:
        apply_overrides(cfg, ["train.nonexistent=1"])
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_seed_repeatable():
    set_seed(0)
    a = (np.random.rand(3), torch.rand(3))
    set_seed(0)
    b = (np.random.rand(3), torch.rand(3))
    assert np.allclose(a[0], b[0]) and torch.equal(a[1], b[1])

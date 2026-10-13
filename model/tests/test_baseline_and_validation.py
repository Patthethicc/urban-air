import json
import numpy as np
import pandas as pd
import pytest
import torch

from networks.dual_branch_net import DualBranchNet
from reporting.snapshot_generator import split_metrics
from training.checkpoint import check_compatibility, file_sha1, save_checkpoint
from training.metrics import compute_all


def _temporal_only(cfg):
    cfg["model"]["spatial_cnn"]["enabled"] = False
    for block in cfg["model"].values():
        block["dropout"] = 0.0
    return DualBranchNet(cfg["model"])


def test_temporal_only_model_ignores_the_patch_and_has_no_cnn_parameters(cfg):
    model = _temporal_only(cfg).eval()
    d = cfg["derived"]
    temporal = torch.randn(4, d["T"], d["F"])
    a = model(torch.randn(4, d["C"], 2, 2), temporal)
    b = model(torch.randn(4, d["C"], 2, 2) * 100, temporal)
    assert torch.allclose(a, b) and a.shape == (4, 1)
    assert model.parameter_counts()["spatial_cnn"] == 0 and model.spatial is None


def test_temporal_only_gradients_reach_the_lstm_and_head(cfg):
    model = _temporal_only(cfg)
    d = cfg["derived"]
    out = model(torch.randn(6, d["C"], 2, 2), torch.randn(6, d["T"], d["F"]))
    torch.nn.functional.mse_loss(out, torch.randn(6, 1)).backward()
    for module in (model.temporal, model.head):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters()) > 0


def test_full_model_is_the_default(base_cfg):
    assert base_cfg["model"]["spatial_cnn"]["enabled"] is True


def test_searching_cnn_params_without_a_cnn_is_refused(cfg):
    from config_parser import apply_overrides, validate, validate_for_tuning
    cfg = apply_overrides(cfg, ["model.spatial_cnn.enabled=false"])
    validate(cfg)                                    # fine for every stage except tuning
    with pytest.raises(ValueError):
        validate_for_tuning(cfg)


def test_headline_metrics_are_rmse_mae_r2(base_cfg):
    out = split_metrics(compute_all([1, 2, 3, 4], [1.5, 2.5, 2.5, 4.5]), base_cfg)
    assert base_cfg["report"]["metrics"] == ["rmse", "mae", "r2"]
    assert {"rmse", "mae", "r2", "supplementary"} == set(out) and set(out["supplementary"]) == {"ia", "pearson_r"}


def _processed(tmp_path, derived, lags, scalers=b"{}"):
    (tmp_path / "scalers.json").write_bytes(scalers)
    info = {**{k: derived[k] for k in ("T", "F", "C", "patch_px", "feature_names")}, "temporal": {"lags": lags}}
    (tmp_path / "preprocess_info.json").write_text(json.dumps(info))
    return str(tmp_path)


def _ckpt(cfg, tmp_path, lags, sha):
    path = str(tmp_path / "m.pt")
    save_checkpoint(path, DualBranchNet(cfg["model"]), cfg["model"], cfg["train"], 1, 0.1,
                    {"derived": cfg["derived"], "lags": lags, "scalers_sha1": sha})
    return torch.load(path, weights_only=False)


def test_compatible_checkpoint_passes(cfg, tmp_path):
    processed = _processed(tmp_path, cfg["derived"], [3, 2, 1])
    check_compatibility(_ckpt(cfg, tmp_path, [3, 2, 1], file_sha1(tmp_path / "scalers.json")), processed)


def test_changed_lags_patch_or_scalers_are_caught(cfg, tmp_path):
    processed = _processed(tmp_path, cfg["derived"], [4, 2, 1])
    ckpt = _ckpt(cfg, tmp_path, [3, 2, 1], file_sha1(tmp_path / "scalers.json"))
    with pytest.raises(ValueError, match="lags"):
        check_compatibility(ckpt, processed)

    processed = _processed(tmp_path, {**cfg["derived"], "patch_px": 32}, [3, 2, 1])
    with pytest.raises(ValueError, match="patch_px"):
        check_compatibility(ckpt, processed)

    processed = _processed(tmp_path, cfg["derived"], [3, 2, 1], scalers=b'{"y_mean": 99}')
    with pytest.raises(ValueError, match="scalers"):
        check_compatibility(ckpt, processed)

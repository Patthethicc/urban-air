import copy

import numpy as np
import pytest
import torch

from networks.dual_branch_net import DualBranchNet


def _batch(cfg, rows=24, stations=3, px=8, seed=0):
    """`rows` samples that share `stations` distinct patches, like a training batch."""
    g = torch.Generator().manual_seed(seed)
    d = cfg["derived"]
    patches = torch.randn(stations, d["C"], px, px, generator=g)
    which = torch.randint(0, stations, (rows,), generator=g)
    return patches[which], torch.randn(rows, d["T"], d["F"], generator=g), which


def _models(cfg, **spatial):
    cfg["model"]["spatial_cnn"].update(spatial)
    for block in cfg["model"].values():
        block["dropout"] = 0.0
    torch.manual_seed(0)
    plain = DualBranchNet(cfg["model"])
    shared = copy.deepcopy(plain)
    shared.dedupe_patches = True
    return plain, shared


@pytest.mark.parametrize("spatial", [{}, {"feature_map": "flatten", "patch_px": 8}])
def test_shared_patches_give_the_same_prediction_and_the_same_gradients(cfg, spatial):
    plain, shared = _models(cfg, **spatial)
    x, t, _ = _batch(cfg)
    target = torch.randn(len(x), 1)
    outs = []
    for model in (plain, shared):
        out = model(x, t)
        torch.nn.functional.mse_loss(out, target).backward()
        outs.append(out)
    assert torch.allclose(outs[0], outs[1], atol=1e-5)
    for (name, a), (_, b) in zip(plain.named_parameters(), shared.named_parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-5, rtol=1e-4), name


def test_eval_mode_is_identical_too(cfg):
    plain, shared = _models(cfg)
    x, t, _ = _batch(cfg, seed=1)
    with torch.no_grad():
        assert torch.allclose(plain.eval()(x, t), shared.eval()(x, t), atol=1e-5)


def test_the_cnn_really_runs_once_per_distinct_patch(cfg):
    _, shared = _models(cfg)
    seen = []
    shared.spatial.conv.register_forward_hook(lambda m, i, o: seen.append(o.shape[0]))
    x, t, which = _batch(cfg, rows=24, stations=3)
    shared(x, t)
    assert seen == [len(set(which.tolist()))]                      # 3 distinct patches, not 24 rows


def test_a_batch_without_repeats_takes_the_normal_path(cfg):
    _, shared = _models(cfg)
    seen = []
    shared.spatial.conv.register_forward_hook(lambda m, i, o: seen.append(o.shape[0]))
    x = torch.randn(5, cfg["derived"]["C"], 8, 8)
    shared(x, torch.randn(5, cfg["derived"]["T"], cfg["derived"]["F"]))
    assert seen == [5]


def test_dropout_is_still_applied_per_row_after_the_lookup(cfg):
    """Two rows with the same patch must get DIFFERENT dropout masks (as without sharing), not one shared mask."""
    cfg["model"]["spatial_cnn"].update(dropout=0.5)
    torch.manual_seed(0)
    model = DualBranchNet(cfg["model"]).train()
    model.dedupe_patches = True
    x = torch.randn(1, cfg["derived"]["C"], 8, 8).repeat(64, 1, 1, 1)
    emb = model.spatial(x, dedupe=True)
    assert len({tuple((row > 0).tolist()) for row in emb}) > 8       # many different masks across identical rows


def test_the_checkpoint_layout_is_unchanged(cfg):
    keys = list(DualBranchNet(cfg["model"]).state_dict())
    assert "spatial.project.0.weight" in keys and "spatial.conv.0.weight" in keys and not any("dedupe" in k for k in keys)


def test_trainer_and_prediction_switch_it_on(cfg, tmp_path):
    from training.trainer import Trainer
    model = DualBranchNet(cfg["model"])
    assert model.dedupe_patches is False
    Trainer(model, {**cfg["train"], "device": "cpu"}, cfg["model"], str(tmp_path))
    assert model.dedupe_patches is True

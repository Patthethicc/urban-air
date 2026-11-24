import torch

from networks.dual_branch_net import DualBranchNet


def test_forward_shape(cfg):
    model = DualBranchNet(cfg["model"]).eval()
    d = cfg["derived"]
    out = model(torch.randn(6, d["C"], d["patch_px"], d["patch_px"]), torch.randn(6, d["T"], d["F"]))
    assert out.shape == (6, 1)
    assert torch.isfinite(out).all()


def test_accepts_the_production_patch_size(cfg):
    model = DualBranchNet(cfg["model"]).eval()
    d = cfg["derived"]
    out = model(torch.randn(2, d["C"], 32, 32), torch.randn(2, d["T"], d["F"]))
    assert out.shape == (2, 1)


def test_flatten_option_matches_the_patch_size(cfg):
    cfg["model"]["spatial_cnn"].update(feature_map="flatten", patch_px=4)
    model = DualBranchNet(cfg["model"]).eval()
    d = cfg["derived"]
    out = model(torch.randn(3, d["C"], 4, 4), torch.randn(3, d["T"], d["F"]))
    assert out.shape == (3, 1) and torch.isfinite(out).all()
    assert model.spatial.project[0].in_features == cfg["model"]["spatial_cnn"]["conv_channels"][-1] * 4 * 4     # the whole map is kept


def test_flatten_rejects_a_different_patch_size(cfg):
    import pytest
    cfg["model"]["spatial_cnn"].update(feature_map="flatten", patch_px=4)
    model = DualBranchNet(cfg["model"]).eval()
    d = cfg["derived"]
    with pytest.raises(ValueError):
        model(torch.randn(2, d["C"], 6, 6), torch.randn(2, d["T"], d["F"]))


def test_default_is_pooling(base_cfg):
    assert base_cfg["model"]["spatial_cnn"]["feature_map"] == "pool"

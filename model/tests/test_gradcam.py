import numpy as np
import pytest
import torch
import torch.nn as nn

from explain.gradcam import attention_profile, grad_cam
from explain.gradcam_plots import plot_cam_overlays
from networks.dual_branch_net import DualBranchNet


class _ToySpatial(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 1, 1, bias=False), nn.ReLU())
        self.conv[0].weight.data.fill_(1.0)


class _Toy(nn.Module):
    """Output = mean of the last conv activation: every pixel matters equally, so Grad-CAM follows the activation."""
    def __init__(self):
        super().__init__()
        self.spatial = _ToySpatial()

    def forward(self, spatial, temporal):
        return self.spatial.conv(spatial).mean(dim=(1, 2, 3)).unsqueeze(1) + 0.0 * temporal[:, 0, :1]


def test_cam_peaks_where_the_feature_is_active():
    patch = np.ones((1, 1, 5, 5), dtype=np.float32)
    patch[0, 0, 2, 3] = 5.0
    cam, pred = grad_cam(_Toy(), patch, np.zeros((1, 3, 4), dtype=np.float32))
    assert cam.shape == (1, 5, 5) and np.unravel_index(cam[0].argmax(), (5, 5)) == (2, 3)
    assert cam[0, 2, 3] == pytest.approx(1.0) and cam[0, 0, 0] == pytest.approx(0.2)         # 1 / 5


def test_inactive_pixels_get_no_attention():
    patch = -np.ones((1, 1, 4, 4), dtype=np.float32)
    patch[0, 0, 1, 1] = 3.0
    cam, _ = grad_cam(_Toy(), patch, np.zeros((1, 3, 4), dtype=np.float32))
    assert cam[0, 1, 1] == pytest.approx(1.0) and cam.sum() == pytest.approx(1.0)


def test_real_model_cam_is_normalised_deterministic_and_leaves_the_model_untouched(cfg):
    model = DualBranchNet(cfg["model"])
    d = cfg["derived"]
    rng = np.random.default_rng(0)
    patches = rng.normal(size=(4, d["C"], 6, 6)).astype(np.float32)
    temporal = rng.normal(size=(4, d["T"], d["F"])).astype(np.float32)
    before = [p.detach().clone() for p in model.parameters()]
    cam1, pred1 = grad_cam(model, patches, temporal)
    cam2, pred2 = grad_cam(model, patches, temporal)
    assert cam1.shape == (4, 6, 6) and cam1.min() >= 0 and cam1.max() <= 1 + 1e-6
    assert np.allclose(cam1, cam2) and np.allclose(pred1, pred2)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert len(model.spatial.conv[-1]._forward_hooks) == 0                                   # hook removed


def test_the_prediction_returned_is_the_plain_forward_pass(cfg):
    model = DualBranchNet(cfg["model"]).eval()
    d = cfg["derived"]
    patches = np.random.default_rng(1).normal(size=(3, d["C"], 4, 4)).astype(np.float32)
    temporal = np.random.default_rng(2).normal(size=(3, d["T"], d["F"])).astype(np.float32)
    _, pred = grad_cam(model, patches, temporal)
    with torch.no_grad():
        assert np.allclose(pred, model(torch.from_numpy(patches), torch.from_numpy(temporal)).numpy().ravel(), atol=1e-6)


def test_temporal_only_model_cannot_be_explained(cfg):
    cfg["model"]["spatial_cnn"]["enabled"] = False
    model = DualBranchNet(cfg["model"])
    d = cfg["derived"]
    with pytest.raises(ValueError, match="CNN"):
        grad_cam(model, np.zeros((1, d["C"], 3, 3), np.float32), np.zeros((1, d["T"], d["F"]), np.float32))


def test_attention_profile_reports_which_band_values_the_model_looks_at():
    cam = np.zeros((1, 5, 5), dtype=np.float32)
    cam[0, 1:3, 1:3] = 1.0
    raw = np.zeros((1, 2, 5, 5), dtype=np.float32)
    raw[0, 0] = cam[0] * 10                                  # band 0 is high exactly where the model looks
    raw[0, 1] = 7.0                                          # band 1 is constant
    rows = {r["band"]: r for r in attention_profile(raw, cam, ["building_density", "ndvi"], quantile=0.8)}
    assert rows["building_density"]["std_units"] > 1.0 and rows["building_density"]["attended_mean"] == 10.0
    assert rows["ndvi"]["std_units"] == 0.0 and rows["building_density"]["attended_pixels"] == 4


def test_overlay_figure_is_written(tmp_path):
    cam = np.random.default_rng(3).random((2, 8, 8)).astype(np.float32)
    raw = np.random.default_rng(4).random((2, 3, 8, 8)).astype(np.float32)
    path = plot_cam_overlays(raw, cam, ["a", "b", "c"], ["cell 1", "cell 2"], str(tmp_path / "g.png"))
    assert (tmp_path / "g.png").stat().st_size > 5000 and path.endswith("g.png")

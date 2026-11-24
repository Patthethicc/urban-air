"""The SHAP-style summary of the Grad-CAM band scores (explain/band_swarm.py, band_swarm_plots.py)."""
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin

from explain import band_swarm as bs
from explain.band_swarm_plots import plot_band_importance, plot_band_swarm, swarm_offsets

TRANSFORM = from_origin(0, 400, 20, 20)                       # a 20 x 20 grid of 20 m cells
NAMES = ["building_height_mean", "ndvi_mean", "building_density"]


def test_the_same_seed_samples_the_same_cells_and_only_from_the_allowed_ones():
    ok = np.zeros((20, 20), bool)
    ok[5:15, 5:15] = True
    a, b, c = bs.sample_cells(ok, 30, 0), bs.sample_cells(ok, 30, 0), bs.sample_cells(ok, 30, 1)
    assert a.tolist() == b.tolist() and a.tolist() != c.tolist() and len(set(a)) == 30 == len(a)
    assert ok.ravel()[a].all() and a.tolist() == sorted(a.tolist())
    assert bs.sample_cells(ok, 1000).tolist() == np.flatnonzero(ok.ravel()).tolist()        # fewer cells than asked: all of them


def test_the_score_is_how_far_the_attended_pixels_are_from_the_patch_average_in_patch_std():
    raw = np.zeros((2, 3, 6, 6), dtype="float32")
    cam = np.zeros((2, 6, 6), dtype="float32")
    for i in range(2):
        cam[i, 1:3, 1:3] = 1.0                                 # the attended pixels: a 2 x 2 corner
    raw[:, 0] = np.arange(36).reshape(6, 6)                    # band 0 is low where the model looks (the top-left corner)
    raw[:, 1][:, 1:3, 1:3] = 10.0                              # band 1 is high exactly where it looks
    raw[:, 2] = 5.0                                            # band 2 is constant
    scores, values = bs.attention_scores(raw, cam, NAMES, 0.8)
    assert scores[0, 0] < -0.5 and scores[0, 1] > 1.0 and scores[0, 2] == 0.0
    assert values[0, 2] == 5.0 and values[0, 1] == pytest.approx(raw[0, 1].mean())          # the colour value is the patch mean
    flat = np.zeros((1, 3, 6, 6), dtype="float32")             # no attention at all: no score, not a zero
    s0, _ = bs.attention_scores(flat, np.zeros((1, 6, 6), dtype="float32"), NAMES, 0.8)
    assert np.isnan(s0).all()


def test_rows_run_from_the_largest_mean_absolute_score_down_and_colours_are_percentile_clipped():
    scores = np.array([[0.1, -0.9, 0.3], [-0.1, 0.8, -0.3], [0.1, np.nan, 0.3]])
    assert bs.row_order(scores).tolist() == [1, 2, 0]
    v = np.column_stack([np.arange(101.0), np.full(101, 4.0), np.r_[np.nan, np.arange(100.0)]])
    lv = bs.colour_levels(v)
    assert lv[5, 0] == pytest.approx(0.0) and lv[95, 0] == pytest.approx(1.0) and lv[0, 0] == 0.0 and lv[100, 0] == 1.0     # clipped at p5 and p95
    assert (lv[:, 1] == 0.5).all() and np.isnan(lv[0, 2])


def test_dots_with_the_same_x_are_spread_apart_inside_the_row():
    x = np.r_[np.zeros(20), np.linspace(1, 2, 5)]
    y = swarm_offsets(x)
    assert np.abs(y).max() <= 0.4 + 1e-9 and len(set(np.round(y[:20], 6))) == 20           # twenty equal values are twenty different heights
    assert (swarm_offsets(x) == y).all()                                                    # and the layout is repeatable
    assert swarm_offsets(np.array([])).size == 0 and np.abs(swarm_offsets(np.full(3, 2.0))).max() > 0     # empty and all-equal inputs are fine


def test_the_importance_table_has_the_mean_abs_score_inside_its_interval():
    rng = np.random.default_rng(0)
    scores = np.column_stack([rng.normal(0.4, 0.3, 200), rng.normal(0, 0.1, 200), np.full(200, np.nan)])
    t = bs.importance(scores)
    assert (t["ci_low"][:2] <= t["mean_abs_score"][:2]).all() and (t["mean_abs_score"][:2] <= t["ci_high"][:2]).all()
    assert t["mean_abs_score"][0] > t["mean_abs_score"][1] and t["cells"][2] == 0 and np.isnan(t["mean_abs_score"][2])


def _explain_factory(calls):
    def explain(xs, ys):
        calls.append(len(xs))
        n = len(xs)
        keep = np.ones(n, bool)
        keep[::7] = False                                      # some cells have no history
        k = int(keep.sum())
        rng = np.random.default_rng(len(calls))
        raw = rng.random((k, 3, 8, 8)).astype("float32")
        cam = rng.random((k, 8, 8)).astype("float32")
        return raw, cam, np.linspace(5, 30, k), keep
    return explain


def test_the_pass_writes_both_figures_and_a_table_with_one_row_per_explained_cell(cfg, tmp_path):
    cfg["output"]["explain"]["band_swarm"] = {"n_cells": 300, "chunk": 128, "seed": 0}
    ok = np.ones((20, 20), bool)
    calls = []
    out = bs.run(cfg, pred_map=np.ones((20, 20)), transform=TRANSFORM, ok=ok, names=NAMES, explain_cells=_explain_factory(calls),
                 out_dir=tmp_path, ts=pd.Timestamp("2025-07-23 12:00"), caption="c", dpi=60)
    assert calls == [128, 128, 44]                                                            # 300 of the 400 cells, 128 per batch
    table = pd.read_csv(tmp_path / "gradcam_band_swarm.csv")
    assert len(table) == out["cells_explained"] and out["cells_explained"] < out["cells_sampled"]            # the cells without history are left out
    assert {f"score_{b}" for b in NAMES} <= set(table.columns) and {f"patch_mean_{b}" for b in NAMES} <= set(table.columns)
    assert sorted(out["order"]) == sorted(NAMES) and out["reading"].startswith("x = attended")
    for f in ("gradcam_band_swarm.png", "gradcam_band_importance.png"):
        assert (tmp_path / f).stat().st_size > 8_000, f


def test_nothing_to_explain_is_reported_not_a_crash(cfg, tmp_path):
    out = bs.run(cfg, pred_map=np.ones((20, 20)), transform=TRANSFORM, ok=np.ones((20, 20), bool), names=NAMES,
                 explain_cells=lambda xs, ys: (np.empty((0, 3, 8, 8), "float32"), np.empty((0, 8, 8), "float32"), np.empty(0), np.zeros(len(xs), bool)),
                 out_dir=tmp_path, ts=pd.Timestamp("2025-07-23 12:00"), caption=None, dpi=60)
    assert "skipped" in out and not (tmp_path / "gradcam_band_swarm.png").exists()


def test_the_figures_work_with_missing_scores_and_without_a_config(tmp_path):
    rng = np.random.default_rng(1)
    scores = rng.normal(0, 0.4, (60, 3))
    scores[::5, 1] = np.nan                                    # a band with no attended pixels in some cells
    levels = rng.random((60, 3))
    plot_band_swarm(scores, levels, NAMES, str(tmp_path / "s.png"), 60)
    plot_band_importance(bs.importance(scores).assign(band=NAMES), NAMES, bs.row_order(scores), str(tmp_path / "i.png"), 60)
    assert (tmp_path / "s.png").stat().st_size > 8_000 and (tmp_path / "i.png").stat().st_size > 5_000


def test_the_setting_is_on_by_default_with_300_cells(cfg):
    s = bs.settings(cfg)
    assert s["enabled"] and s["n_cells"] == 300 and s["chunk"] == 128

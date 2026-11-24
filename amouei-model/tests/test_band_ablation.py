"""The band ablation: swap one band with another cell's, see how far the prediction moves (explain/band_ablation.py)."""
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin

from explain import band_ablation as ba

TRANSFORM = from_origin(0, 400, 20, 20)
NAMES = ["building_height_mean", "ndvi_mean", "building_density"]


def _toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 3, 4, 4)).astype("float32")
    feats = rng.random((n, 2, 3)).astype("float32")
    return raw, feats


def toy_predict(raw, feats):
    """The model only reads band 1 (ndvi): twice its patch mean; the other bands and the history do nothing."""
    return 2.0 * raw[:, 1].mean(axis=(1, 2)).astype("float64")


def test_every_cell_gets_the_band_of_another_cell():
    rng = np.random.default_rng(0)
    for n in (2, 3, 10, 57):
        p = ba.swap_partner(n, rng)
        assert sorted(p.tolist()) == list(range(n)) and (p != np.arange(n)).all()          # a permutation with nobody keeping their own


def test_only_the_swapped_band_changes():
    raw, _ = _toy()
    partner = ba.swap_partner(len(raw), np.random.default_rng(1))
    out = ba.permute_band(raw, 1, partner)
    assert (out[:, 1] == raw[partner, 1]).all() and (out[:, [0, 2]] == raw[:, [0, 2]]).all()
    assert out is not raw and (raw == _toy()[0]).all()                                      # the input is not touched


def test_the_band_the_model_reads_moves_the_prediction_and_the_others_do_not():
    raw, feats = _toy()
    base, signed, absolute = ba.band_effects(toy_predict, raw, feats, repeats=4, seed=0)
    assert (base == toy_predict(raw, feats)).all()
    assert (absolute[:, [0, 2]] == 0).all() and (signed[:, [0, 2]] == 0).all()               # an unused band: exactly no effect
    assert absolute[:, 1].mean() > 0.05 and (absolute[:, 1] >= np.abs(signed[:, 1]) - 1e-12).all()   # |mean change| <= mean |change|
    imp = ba.importance(absolute, signed)
    assert imp["mean_abs_score"].idxmax() == 1 and (imp["ci_low"] <= imp["mean_abs_score"]).all() and (imp["mean_abs_score"] <= imp["ci_high"]).all()


def test_the_same_seed_gives_the_same_effects_and_chunking_changes_nothing():
    raw, feats = _toy()
    a = ba.band_effects(toy_predict, raw, feats, 3, seed=5, chunk=7)
    b = ba.band_effects(toy_predict, raw, feats, 3, seed=5, chunk=1000)
    assert all(np.allclose(x, y) for x, y in zip(a, b))
    c = ba.band_effects(toy_predict, raw, feats, 3, seed=6)
    assert not np.allclose(a[1][:, 1], c[1][:, 1])


def test_one_cell_cannot_be_swapped_with_anyone():
    raw, feats = _toy(1)
    base, signed, absolute = ba.band_effects(toy_predict, raw, feats, 3)
    assert len(base) == 1 and (signed == 0).all() and (absolute == 0).all()


def _inputs_factory(raw, feats, seen):
    def cell_inputs(xs, ys):
        seen.append(len(xs))
        keep = np.ones(len(xs), bool)
        keep[::9] = False                                       # some cells have no history
        return raw[:len(xs)][keep], feats[:len(xs)][keep], keep
    return cell_inputs


def test_the_pass_writes_the_bar_the_swarm_and_the_table(cfg, tmp_path):
    raw, feats = _toy(400)
    cfg["output"]["explain"]["band_swarm"] = {"n_cells": 150, "seed": 0}
    cfg["output"]["explain"]["band_ablation"] = {"repeats": 2, "chunk": 64}
    seen = []
    out = ba.run(cfg, ok=np.ones((20, 20), bool), transform=TRANSFORM, names=NAMES, cell_inputs=_inputs_factory(raw, feats, seen),
                 predict=toy_predict, out_dir=tmp_path, ts=pd.Timestamp("2025-07-23 12:00"), dpi=60)
    assert seen == [150] and out["order"][0] == "ndvi_mean" and out["cells_explained"] < out["cells_sampled"] == 150     # the swarm's cells
    t = pd.read_csv(tmp_path / "band_ablation.csv")
    assert len(t) == out["cells_explained"] and (t["abs_delta_building_density"] == 0).all() and t["abs_delta_ndvi_mean"].mean() > 0
    for f in ("band_ablation.png", "band_ablation_swarm.png"):
        assert (tmp_path / f).stat().st_size > 8_000, f
    assert out["reading"].startswith("change in the predicted PM2.5")


def test_too_few_cells_is_reported_not_a_crash(cfg, tmp_path):
    out = ba.run(cfg, ok=np.ones((20, 20), bool), transform=TRANSFORM, names=NAMES,
                 cell_inputs=lambda xs, ys: (np.empty((0, 3, 4, 4), "float32"), np.empty((0, 2, 3), "float32"), np.zeros(len(xs), bool)),
                 predict=toy_predict, out_dir=tmp_path, ts=pd.Timestamp("2025-07-23 12:00"))
    assert "skipped" in out and not (tmp_path / "band_ablation.png").exists()


def test_the_setting_is_on_by_default_with_five_swaps(cfg):
    s = ba.settings(cfg)
    assert s["enabled"] and s["repeats"] == 5 and s["n_cells"] is None

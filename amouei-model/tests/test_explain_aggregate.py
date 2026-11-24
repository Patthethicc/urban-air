"""Grad-CAM pooled over the snapshot hours of a day (explain/aggregate.py) and the hot / cold pick shared by the hourly and the daily explain."""
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin

import run_pipeline
from explain import aggregate

NAMES = ["building_height_mean", "ndvi_mean", "building_density"]
HOURS = ["08:00", "12:00", "17:00", "21:00"]


def _swarm(n, rng, shift=0.0):
    table = pd.DataFrame({"x": rng.random(n), "y": rng.random(n), "predicted": rng.random(n) * 30})
    for b in NAMES:
        table[f"score_{b}"] = rng.normal(shift if b == "ndvi_mean" else 0.0, 1.0, n)
        table[f"patch_mean_{b}"] = rng.random(n)
    return table


def _ablation(n, rng):
    table = pd.DataFrame({"x": rng.random(n), "y": rng.random(n), "predicted": rng.random(n) * 30})
    for b in NAMES:
        d = rng.normal(0, 2.0, n)
        table[f"delta_{b}"], table[f"abs_delta_{b}"], table[f"patch_mean_{b}"] = d, np.abs(d), rng.random(n)
    return table


def _attention(rng):
    rows = []
    for group in ("hot", "cold"):
        for cell in range(1, 4):
            for b in NAMES:
                rows.append({"group": group, "cell_number": cell, "band": b, "std_units": rng.normal(), "patch_mean": rng.random()})
    return pd.DataFrame(rows)


@pytest.fixture
def hour_folders(tmp_path):
    rng = np.random.default_rng(0)
    folders = {}
    for hour in HOURS:
        folder = tmp_path / hour.replace(":", "")
        folder.mkdir()
        _swarm(40, rng, shift=-0.3).to_csv(folder / "gradcam_band_swarm.csv", index=False)
        _ablation(40, rng).to_csv(folder / "band_ablation.csv", index=False)
        _attention(rng).to_csv(folder / "gradcam_attention.csv", index=False)
        folders[hour] = folder
    return folders


def test_the_tables_of_the_hours_are_stacked_with_an_hour_column(hour_folders):
    swarm = aggregate.read_hours(hour_folders, "gradcam_band_swarm.csv")
    assert len(swarm) == 4 * 40 and swarm["hour"].unique().tolist() == HOURS
    assert aggregate.bands_of(swarm, "score_") == NAMES
    assert aggregate.read_hours(hour_folders, "nothing_here.csv").empty                # a missing file leaves the hour out


def test_an_hour_without_a_table_is_just_left_out(hour_folders):
    (hour_folders["12:00"] / "band_ablation.csv").unlink()
    ablation = aggregate.read_hours(hour_folders, "band_ablation.csv")
    assert sorted(ablation["hour"].unique()) == ["08:00", "17:00", "21:00"]


def test_the_by_hour_table_has_one_column_per_hour_and_one_for_all(hour_folders):
    swarm = aggregate.read_hours(hour_folders, "gradcam_band_swarm.csv")
    table = aggregate.by_hour_table(swarm, NAMES)
    assert table.columns.tolist() == ["band", *HOURS, "all hours"] and table["band"].tolist() == NAMES
    mean_of_hours = table[HOURS].mean(axis=1)                                          # equal sample sizes: the pool is the mean of the hours
    np.testing.assert_allclose(mean_of_hours, table["all hours"], rtol=1e-9)


def test_the_pooled_figures_and_tables_are_written(cfg, hour_folders, tmp_path):
    out = tmp_path / "all"
    summary = aggregate.run(cfg, folders=hour_folders, out_dir=out, names=NAMES, caption="test", dpi=60, date="2025-07-23")
    for name in ("gradcam_band_profile.png", "gradcam_band_swarm.png", "gradcam_band_importance.png", "gradcam_band_by_hour.png",
                 "band_ablation.png", "band_ablation_swarm.png", "gradcam_band_swarm.csv", "gradcam_band_by_hour.csv",
                 "band_ablation.csv", "gradcam_attention.csv"):
        assert (out / name).is_file() and (out / name).stat().st_size > 100, name
    assert summary["hours"] == HOURS and summary["band_swarm"]["cell_hours"] == 160 and summary["band_ablation"]["cell_hours"] == 160
    assert len(pd.read_csv(out / "gradcam_band_swarm.csv")) == 160
    assert len(summary["band_swarm"]["by_hour"]) == len(NAMES)


def test_ndvi_leaning_negative_in_every_hour_stays_negative_when_pooled(cfg, hour_folders, tmp_path):
    summary = aggregate.run(cfg, folders=hour_folders, out_dir=tmp_path / "all", names=NAMES, caption=None, dpi=60, date="d")
    ndvi = next(b for b in summary["band_swarm"]["bands"] if b["band"] == "ndvi_mean")
    assert ndvi["mean_score"] < 0


def test_hot_and_cold_cells_come_from_the_ends_of_the_grid_and_do_not_overlap(cfg):
    cfg["output"]["explain"].update(top_n=3, bottom_n=3, min_separation_m=0)
    pred = np.arange(100, dtype=float).reshape(10, 10)                                 # rises to the bottom-right: the top cells are the last ones
    ok = np.ones_like(pred, bool)
    ok[0, 0] = False                                                                   # the lowest value is outside the NCR
    groups = run_pipeline._hot_cold_cells(cfg, pred, from_origin(0, 100, 10, 10), ok)
    assert sorted(groups["hot"].tolist()) == [97, 98, 99] and sorted(groups["cold"].tolist()) == [1, 2, 3]
    assert not set(groups["hot"]) & set(groups["cold"])

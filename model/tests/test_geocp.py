import numpy as np
import pandas as pd
import pytest

from uncertainty.geocp import GeoCP, apply_intervals, kernel_weights, scores_from_predictions, select_config

XY = pd.DataFrame({"location_id": [1, 2, 3], "x": [0.0, 1000.0, 50000.0], "y": [0.0, 0.0, 0.0]})


def _cal(scores_by_station: dict) -> pd.DataFrame:
    return pd.DataFrame([{"location_id": s, "score": v} for s, vals in scores_by_station.items() for v in vals])


def test_unweighted_matches_split_conformal():
    """9 scores, alpha 0.2: the ceil(0.8 * 10) = 8th smallest."""
    cal = _cal({1: list(range(1, 10))})
    q = GeoCP(0.2, "gaussian", np.inf).fit(cal, XY).quantile((0.0, 0.0))
    assert q == 8.0


def test_default_normalises_weights_to_one_so_a_small_calibration_set_still_gives_an_interval():
    cal = _cal({1: [1.0, 2.0]})
    assert GeoCP(0.1, "gaussian", np.inf).fit(cal, XY).quantile((0.0, 0.0)) == 2.0


def test_finite_sample_correction_gives_an_infinite_interval_when_calibration_is_too_small():
    cal = _cal({1: [1.0, 2.0]})
    assert np.isinf(GeoCP(0.1, "gaussian", np.inf, finite_sample_correction=True).fit(cal, XY).quantile((0.0, 0.0)))


def test_no_calibration_weight_nearby_means_no_interval():
    cal = _cal({3: [5.0] * 20})                         # only the 50 km station, tiny bandwidth
    assert np.isinf(GeoCP(0.1, "gaussian", 100.0).fit(cal, XY).quantile((0.0, 0.0)))


def test_nearby_stations_dominate_the_quantile():
    cal = _cal({1: [1.0] * 50, 3: [100.0] * 50})                  # station 3 is 50 km away and much worse
    near = GeoCP(0.1, "gaussian", 2000.0).fit(cal, XY)
    flat = GeoCP(0.1, "gaussian", np.inf).fit(cal, XY)
    assert near.quantile((0.0, 0.0)) == 1.0
    assert flat.quantile((0.0, 0.0)) == 100.0


def test_loso_excludes_the_held_out_station():
    cal = _cal({1: [1.0] * 40, 2: [2.0] * 40, 3: [3.0] * 40})
    q = GeoCP(0.1, "gaussian", np.inf).fit(cal, XY).quantile((0.0, 0.0), exclude=3)
    assert q == 2.0


def test_kernel_weights_are_one_at_zero_distance():
    for k in ("gaussian", "exponential"):
        assert kernel_weights(np.array([0.0]), k, 1000.0)[0] == 1.0
    assert (kernel_weights(np.array([5.0, 50.0]), "gaussian", np.inf) == 1).all()


def test_coverage_on_exchangeable_data_is_close_to_target():
    rng = np.random.default_rng(1)
    cal_pred = pd.DataFrame({"location_id": rng.choice([1, 2], 4000), "y_true": rng.normal(20, 4, 4000)})
    cal_pred["y_pred"] = 20.0
    test = pd.DataFrame({"location_id": rng.choice([1, 2], 4000), "y_true": rng.normal(20, 4, 4000)})
    test["y_pred"] = 20.0
    gm = GeoCP(0.1, "gaussian", np.inf).fit(scores_from_predictions(cal_pred), XY)
    _, summary = apply_intervals(test, gm, XY)
    assert summary["coverage"] == pytest.approx(0.9, abs=0.03)


def test_coverage_uses_the_unclipped_interval_and_widths_use_the_clipped_one():
    pred = pd.DataFrame({"location_id": [1, 1], "y_true": [0.5, 9.0], "y_pred": [1.0, 1.0]})
    gm = GeoCP(0.1, "gaussian", np.inf).fit(_cal({1: [2.0] * 40}), XY)
    out, summary = apply_intervals(pred, gm, XY, lower_bound=0.0)
    assert out["lo_raw"].tolist() == [-1.0, -1.0] and out["lo"].tolist() == [0.0, 0.0]     # clipped for display
    assert out["covered"].tolist() == [True, False]                                           # checked unclipped
    assert out["width"].tolist() == [3.0, 3.0] and summary["mean_width_unclipped"] == 4.0


def test_table_4_2_metrics_and_zones():
    rng = np.random.default_rng(3)
    cal = _cal({1: np.abs(rng.normal(0, 2, 200)).tolist(), 2: np.abs(rng.normal(0, 2, 200)).tolist()})
    xy = pd.DataFrame({"location_id": [1, 2, 7, 8], "x": [0.0, 1000.0, 1500.0, 30000.0], "y": [0.0] * 4})
    test = pd.DataFrame({"location_id": np.repeat([7, 8], 50), "y_true": 10.0, "y_pred": 10.0 + rng.normal(0, 2, 100)})
    gm = GeoCP(0.1, "gaussian", np.inf).fit(cal, xy)
    out, s = apply_intervals(test, gm, xy)
    assert {"coverage", "mean_width", "median_width", "mean_width_to_prediction_ratio",
            "coverage_urban_core", "coverage_peripheral"} <= set(s)
    zones = {r["location_id"]: r["zone"] for r in s["per_station"]}
    assert zones == {7: "urban_core", 8: "peripheral"}                 # 500 m vs 29 km from the nearest calibration sensor
    assert s["mean_width_to_prediction_ratio"] > 0 and s["median_width"] > 0


def test_explicit_zone_threshold_is_respected():
    cal = _cal({1: [1.0] * 30})
    xy = pd.DataFrame({"location_id": [1, 7], "x": [0.0, 3000.0], "y": [0.0, 0.0]})
    test = pd.DataFrame({"location_id": [7] * 10, "y_true": 5.0, "y_pred": 5.0})
    gm = GeoCP(0.1, "gaussian", np.inf).fit(cal, xy)
    _, near = apply_intervals(test, gm, xy, zone_threshold_m=5000.0)
    _, far = apply_intervals(test, gm, xy, zone_threshold_m=1000.0)
    assert near["coverage_urban_core"] is not None and near["coverage_peripheral"] is None
    assert far["coverage_peripheral"] is not None and far["coverage_urban_core"] is None


def test_grid_quantiles_equal_the_exact_ones():
    rng = np.random.default_rng(5)
    cal = pd.DataFrame({"location_id": rng.choice([1, 2, 3], 600), "score": np.round(np.abs(rng.normal(0, 3, 600)), 2)})
    for correction in (False, True):
        gm = GeoCP(0.1, "gaussian", 5000.0, correction).fit(cal, XY)
        pts = np.array([[0.0, 0.0], [800.0, 0.0], [20000.0, 0.0], [4000.0, 3000.0]])
        exact = np.array([gm.quantile(p) for p in pts])
        assert np.allclose(gm.quantiles_grid(pts, n_grid=5000), exact, equal_nan=False)


def test_calibration_table_is_georeferenced_with_scores():
    from uncertainty.geocp import calibration_table
    pred = pd.DataFrame({"location_id": [1, 2], "timestamp": pd.to_datetime(["2025-01-01"] * 2),
                         "y_true": [10.0, 20.0], "y_pred": [8.0, 25.0]})
    t = calibration_table(pred, XY)
    assert list(t.columns) == ["location_id", "x", "y", "timestamp", "y_true", "y_pred", "score"]
    assert t["score"].tolist() == [2.0, 5.0] and t["x"].tolist() == [0.0, 1000.0]


def test_select_config_returns_a_valid_candidate():
    rng = np.random.default_rng(2)
    cal = pd.DataFrame({"location_id": rng.choice([1, 2, 3], 900), "score": np.abs(rng.normal(0, 3, 900))})
    gcfg = {"alpha": 0.1, "kernels": ["gaussian"], "bandwidths_m": [2000, np.inf], "coverage_tolerance": 0.05}
    best, table = select_config(cal, XY, gcfg)
    assert {"kernel", "bandwidth_m", "loso_coverage", "loso_mean_width"} <= set(best)
    assert len(table) == 2


def test_lower_bound_is_clipped():
    pred = pd.DataFrame({"location_id": [1] * 20, "y_true": [1.0] * 20, "y_pred": [1.0] * 20})
    gm = GeoCP(0.1, "gaussian", np.inf).fit(_cal({1: [5.0] * 40}), XY)
    out, _ = apply_intervals(pred, gm, XY, lower_bound=0.0)
    assert (out["lo"] == 0.0).all() and (out["hi"] == 6.0).all()


def test_correction_option_never_makes_an_interval_narrower_and_the_default_is_off():
    rng = np.random.default_rng(11)
    cal = pd.DataFrame({"location_id": rng.choice([1, 2, 3], 60), "score": np.round(np.abs(rng.normal(0, 3, 60)), 2)})
    plain = GeoCP(0.1, "gaussian", 5000.0).fit(cal, XY)
    corrected = GeoCP(0.1, "gaussian", 5000.0, finite_sample_correction=True).fit(cal, XY)
    assert plain.correction is False                                   # normalised sum to one is the default
    for p in ([0.0, 0.0], [900.0, 0.0], [30000.0, 0.0]):
        assert corrected.quantile(p) >= plain.quantile(p)


def test_select_config_scores_either_option(base_cfg):
    rng = np.random.default_rng(12)
    cal = pd.DataFrame({"location_id": rng.choice([1, 2, 3], 300), "score": np.round(np.abs(rng.normal(0, 3, 300)), 2)})
    for flag in (False, True):
        g = {**base_cfg["geocp"], "finite_sample_correction": flag, "bandwidths_m": [2000, np.inf]}
        best, _ = select_config(cal, XY, g)
        assert best["finite_sample_correction"] is flag

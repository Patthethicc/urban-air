"""The dataset report (reporting/dataset_report.py): the counting of one sensor and the morphology statistics."""
import numpy as np
import pandas as pd
import pytest

from reporting import dataset_report as dr


def _sensor(cfg, stamps, values):
    return dr.station_hours(pd.Series(pd.to_datetime(stamps)), pd.Series(values), cfg)


def test_a_short_gap_is_filled_and_a_long_one_stays_missing(cfg):
    cfg["data"]["time"]["max_gap_hours"] = 3
    hours = pd.date_range("2025-01-01", periods=12, freq="h")
    keep = [0, 1, 5, 6, 11]                                  # gaps of 3 hours (filled) and 4 hours (not filled)
    r = _sensor(cfg, hours[keep], [10.0, 12.0, 20.0, 22.0, 30.0])
    assert r["span_hours"] == 12 and r["observed_hours"] == 5
    assert r["interpolated_hours"] == 3 and r["missing_hours"] == 4 and r["longest_gap_hours"] == 4


def test_values_outside_the_allowed_range_are_counted_and_dropped(cfg):
    lo, hi = cfg["data"]["clean"]["min_value"], cfg["data"]["clean"]["max_value"]
    hours = pd.date_range("2025-01-01", periods=4, freq="h")
    r = _sensor(cfg, hours, [10.0, hi + 1, "bad", lo - 1])
    assert r["file_rows"] == 4 and r["bad_rows"] == 1 and r["out_of_range_rows"] == 2 and r["usable_rows"] == 1


def test_readings_inside_one_hour_are_averaged(cfg):
    stamps = pd.to_datetime(["2025-01-01 00:10", "2025-01-01 00:40", "2025-01-01 01:05"])
    r = _sensor(cfg, stamps, [10.0, 20.0, 30.0])
    assert r["usable_rows"] == 3 and r["observed_hours"] == 2 and r["observed"].iloc[0] == 15.0


def test_a_sensor_without_readings_gives_zero_counts(cfg):
    r = _sensor(cfg, [], [])
    assert r["observed_hours"] == 0 and r["span_hours"] == 0 and pd.isna(r["first_hour"])


def test_the_network_comes_from_the_file_name():
    assert dr.network_of("x/qcdata_9000001_a.csv") == "QC" and dr.network_of("x/openaq_123_b.csv") == "OpenAQ" and dr.network_of(None) == ""


def test_block_means_cut_the_raster_into_patches_and_drop_the_edge():
    arr = np.arange(2 * 5 * 5, dtype=float).reshape(2, 5, 5)
    out = dr.block_means(arr, 2)
    assert out.shape == (2, 2, 2) and out[0, 0, 0] == arr[0, :2, :2].mean() and out[1, 1, 1] == arr[1, 2:4, 2:4].mean()


def test_summarize_and_span_coverage():
    s = dr.summarize(np.array([0.0, 0.0, 1.0, 2.0, np.nan]))
    assert s["n"] == 4 and s["zero_share"] == 0.5 and s["min"] == 0.0 and s["max"] == 2.0
    ref = np.linspace(0, 10, 101)
    sample = np.array([2.0, 4.0, 6.0, 8.0])
    c = dr.span_coverage(ref, sample)
    assert c["inside_minmax"] == pytest.approx(0.61, abs=0.02) and c["inside_p5_p95"] < c["inside_minmax"]
    assert np.isnan(dr.span_coverage(ref, np.array([]))["inside_minmax"])

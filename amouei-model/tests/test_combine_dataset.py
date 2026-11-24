import numpy as np
import pandas as pd
import pytest

from preprocessing.clean_loader import load_raw_readings
from preprocessing.combine_dataset import (COLUMNS, REASONS, RULES, VARIANTS, apply_rules, build, new_year_window)

SPIKY = {**RULES, "spike_min_others": 2}                     # the fixture has 3 stations, not the >= 5 of real data
HOURS = pd.date_range("2022-12-31 00:00", "2024-01-02 23:00", freq="h")
INSIDE = {"Alpha": (14.58, 121.01), "Gamma": (14.65, 121.04)}                      # inside the raster
OUTSIDE = (14.30, 121.01)                                                          # south of it


def _file(path, name, lid, values):
    rows = pd.DataFrame({"location_name": name, "location_id": lid, "sensor_id": lid * 10 + 1,
                         "utc_timestamp": "x", "local_timestamp": [t.strftime("%Y-%m-%dT%H:%M:%S+08:00") for t in HOURS],
                         "pm25_value": values, "parameter": "PM2.5", "unit": "µg/m³"})
    rows.to_csv(path, index=False, encoding="utf-8")


@pytest.fixture
def fixture(tmp_path):
    """Alpha + Beta (OpenAQ, Beta outside the raster) and Gamma (QC, installed 2023-06-01). Planted problems:
    Alpha: a 0.2 at 2023-03-01 10:00 and a NaN; Beta: 1500 at 2023-03-02 10:00; Alpha: a spike of 400 at 2023-07-03 12:00 while the
    others read ~20; Gamma: readings before its installation date; every station: the New Year window."""
    oa, qc, raw = tmp_path / "OpenAQ", tmp_path / "QC Data Converted", tmp_path / "QC Data"
    for d in (oa, qc, raw):
        d.mkdir()
    base = 20.0 + np.sin(np.arange(len(HOURS)) / 7.0)
    alpha, beta, gamma = base.copy(), base + 1, base + 2
    at = lambda s: int(np.flatnonzero(HOURS == pd.Timestamp(s))[0])
    alpha[at("2023-03-01 10:00")] = 0.2
    alpha[at("2023-03-04 10:00")] = np.nan
    beta[at("2023-03-02 10:00")] = 1500.0
    alpha[at("2023-07-03 12:00")] = 400.0
    nye = at("2023-01-01 02:00")
    alpha[nye] = 300.0
    _file(oa / "openaq_3000001_Alpha.csv", "Alpha", 3000001, alpha)
    _file(oa / "openaq_3000002_Beta.csv", "Beta", 3000002, beta)
    _file(qc / "qcdata_9000001_Gamma.csv", "Gamma", 9000001, gamma)
    pd.DataFrame({"name": ["Alpha", "Beta"], "location_id": [3000001, 3000002],
                  "latitude": [INSIDE["Alpha"][0], OUTSIDE[0]], "longitude": [INSIDE["Alpha"][1], OUTSIDE[1]]}).to_csv(
        oa / "OpenAQ Sensor Locations.csv", index=False)
    pd.DataFrame({"name": ["Gamma"], "location_id": [9000001], "latitude": [INSIDE["Gamma"][0]],
                  "longitude": [INSIDE["Gamma"][1]]}).to_csv(qc / "QC Sensor Locations.csv", index=False)
    (raw / "QCDATA_SensorLoc.csv").write_text(
        'ID , Site name , District, Classification, Installation Date, Coordinates\n'
        f'1,Gamma ,V,Roadside,6/1/2023,"{INSIDE["Gamma"][0]},{INSIDE["Gamma"][1]}"\n', encoding="utf-8")
    return oa, qc, raw, tmp_path / "out"


def _build(cfg, fx, **kw):
    oa, qc, raw, out = fx
    return build(oa, qc, raw, out, cfg, **{"with_preinstall": False, **kw}), out


# ------------------------------------------------------------------------------ the rules, one by one
def test_the_new_year_window_is_dec_31_21h_to_jan_1_06h():
    ts = pd.Series(pd.to_datetime(["2023-12-31 20:00", "2023-12-31 21:00", "2024-01-01 00:00", "2024-01-01 06:00", "2024-01-01 07:00", "2024-01-02 03:00"]))
    assert new_year_window(ts).tolist() == [False, True, True, True, False, False]


def test_every_planted_problem_is_removed_for_its_own_reason(cfg, fixture):
    summary, out = _build(cfg, fixture, rules=SPIKY)
    ex = pd.read_csv(out / "excluded_readings.csv").set_index(["location_id", "local_timestamp"])["reason"]
    assert ex[(3000001, "2023-03-01T10:00:00+08:00")] == "floor_or_negative"
    assert ex[(3000001, "2023-03-04T10:00:00+08:00")] == "no_value"
    assert ex[(3000002, "2023-03-02T10:00:00+08:00")] == "above_1000"
    assert ex[(3000001, "2023-07-03T12:00:00+08:00")] == "isolated_spike"
    assert ex[(3000001, "2023-01-01T02:00:00+08:00")] == "new_year_fireworks"
    # two New Year windows in the range (10 hours each) x 3 stations; Gamma is installed 2023-06-01: 152 days = 3,648 earlier
    # hours, of which the 10 New Year hours of the first window are labelled by that rule first
    assert summary["excluded"] == {"no_value": 1, "floor_or_negative": 1, "above_1000": 1, "new_year_fireworks": 60,
                                   "before_installation": 3638, "isolated_spike": 1}
    assert set(ex) <= set(REASONS)


def test_each_reading_gets_the_first_rule_it_breaks(cfg):
    ts = pd.to_datetime(["2023-01-01 02:00", "2023-06-01 12:00"])
    readings = pd.DataFrame({"location_id": [1, 1], "sensor_id": 1, "ts": ts, "pm25_raw": [0.1, 2000.0]})
    st = pd.DataFrame({"location_id": [1], "installed": [pd.NaT]})
    kept, excluded = apply_rules(readings, st)
    assert kept.empty and excluded["reason"].tolist() == ["floor_or_negative", "above_1000"]       # not the fireworks window


def test_a_spike_is_judged_against_the_other_stations_in_that_hour_only(cfg):
    ts = pd.Timestamp("2023-05-05 10:00")
    others = pd.DataFrame({"location_id": range(2, 8), "sensor_id": 1, "ts": ts, "pm25_raw": 22.0})
    st = pd.DataFrame({"location_id": range(1, 8), "installed": pd.NaT})
    solo = pd.concat([pd.DataFrame({"location_id": [1], "sensor_id": 1, "ts": ts, "pm25_raw": [400.0]}), others], ignore_index=True)
    assert apply_rules(solo, st)[1]["reason"].tolist() == ["isolated_spike"]                       # others at 22: a lone spike
    event = solo.assign(pm25_raw=lambda d: np.where(d["location_id"] == 1, 400.0, 380.0))
    assert apply_rules(event, st)[1].empty                                                         # everybody high: a real event
    few = solo[solo["location_id"] <= 3]
    assert apply_rules(few, st)[1].empty                                                           # too few stations to judge


def test_readings_before_installation_can_be_kept_with_the_switch(cfg):
    readings = pd.DataFrame({"location_id": [1, 1], "sensor_id": 1, "ts": pd.to_datetime(["2024-02-01 10:00", "2024-04-01 10:00"]),
                             "pm25_raw": [20.0, 21.0]})
    st = pd.DataFrame({"location_id": [1], "installed": pd.to_datetime(["2024-03-01"])})
    assert apply_rules(readings, st)[1]["reason"].tolist() == ["before_installation"]
    assert apply_rules(readings, st, {**RULES, "trim_before_install": False})[1].empty


# ------------------------------------------------------------------------------ the dataset
def test_nothing_is_lost_or_invented_and_the_table_has_the_documented_columns(cfg, fixture):
    summary, out = _build(cfg, fixture, rules=SPIKY)
    full = pd.read_csv(out / "full" / "combined_hourly.csv")
    assert list(full.columns) == COLUMNS and len(full) == summary["kept_rows"]
    assert summary["kept_rows"] + sum(summary["excluded"].values()) == summary["raw_rows"] == 3 * len(HOURS)
    assert full["pm25_value"].between(0.5, 1000, inclusive="neither").all() and full["pm25_value"].notna().all()
    assert not full.duplicated(["location_id", "local_timestamp"]).any()
    noon = full[full["local_timestamp"].str.contains("T12:00:00")]
    assert noon["hour_sin"].abs().max() < 1e-6 and (noon["hour_cos"] == -1.0).all()               # the encoding of 12:00
    assert set(full["network"]) == {"OpenAQ", "QC"} and set(full.loc[full["network"] == "QC", "location_id"]) == {9000001}


def test_the_year_ranges_are_exact_subsets_of_the_full_table(cfg, fixture):
    summary, out = _build(cfg, fixture, rules=SPIKY)
    full = pd.read_csv(out / "full" / "combined_hourly.csv")
    for name, start in VARIANTS.items():
        part = pd.read_csv(out / name / "combined_hourly.csv")
        expect = full if start is None else full[full["local_timestamp"] >= start]
        assert part.equals(expect.reset_index(drop=True)), name
    assert pd.read_csv(out / "from_2023" / "combined_hourly.csv")["local_timestamp"].str[:4].min() == "2023"
    assert pd.read_csv(out / "from_2024" / "combined_hourly.csv")["local_timestamp"].str[:4].unique().tolist() == ["2024"]
    assert summary["variants"]["full"]["rows"] > summary["variants"]["from_2023"]["rows"] > summary["variants"]["from_2024"]["rows"] > 0


def test_stations_know_whether_the_raster_covers_them(cfg, fixture):
    _, out = _build(cfg, fixture, rules=SPIKY)
    st = pd.read_csv(out / "stations.csv").set_index("location_id")
    assert st.loc[3000001, "inside_raster"] and not st.loc[3000002, "inside_raster"] and st.loc[9000001, "inside_raster"]
    assert st.loc[9000001, "classification"] == "Roadside" and st.loc[3000001, "network"] == "OpenAQ"
    assert st["readings_full"].sum() > st["readings_from_2023"].sum() > st["readings_from_2024"].sum()


def test_the_pipeline_loader_reads_every_range_and_skips_the_station_outside_the_raster(cfg, fixture):
    _, out = _build(cfg, fixture, rules=SPIKY)
    for name in VARIANTS:
        cfg["data"]["paths"].update(sensor_locations_csv=str(out / name / "locations.csv"), sensor_csv_dir=str(out / name / "sensors"))
        cfg["data"]["clean"].update(min_value=0.0, max_value=1000.0)
        got = load_raw_readings(cfg)
        table = pd.read_csv(out / name / "combined_hourly.csv")
        assert set(got["location_id"]) == {3000001, 9000001}                                       # Beta lies outside the raster
        assert len(got) == int(table["inside_raster"].sum())


def test_a_rerun_replaces_the_old_sensor_files_and_writes_the_readme(cfg, fixture):
    oa, qc, raw, out = fixture
    _build(cfg, fixture, rules=SPIKY)
    stale = out / "from_2024" / "sensors" / "openaq_1_stale.csv"
    stale.write_text("x")
    build(oa, qc, raw, out, cfg, SPIKY, with_preinstall=False)
    assert not stale.exists() and len(list((out / "from_2024" / "sensors").glob("*.csv"))) == 3
    text = (out / "README.md").read_text(encoding="utf-8")
    for needle in ("from_2023", "drops 2021-2022", "excluded_readings.csv", "isolated_spike", "Stations reporting per year", "Read with care"):
        assert needle in text


def test_a_station_without_a_data_file_is_reported_not_fatal(cfg, fixture):
    oa, qc, raw, out = fixture
    (oa / "openaq_3000002_Beta.csv").unlink()
    summary, _ = _build(cfg, fixture, rules=SPIKY)
    assert summary["stations_without_file"] == [3000002]


# ------------------------------------------------------------------------------ boundaries
def test_a_reading_of_exactly_the_floor_is_removed_and_one_just_above_is_kept():
    readings = pd.DataFrame({"location_id": 1, "sensor_id": 1, "ts": pd.date_range("2023-05-05 10:00", periods=4, freq="h"),
                             "pm25_raw": [0.5, 0.51, 1000.0, 1000.01]})
    kept, excluded = apply_rules(readings, pd.DataFrame({"location_id": [1], "installed": pd.NaT}))
    assert kept["pm25_raw"].tolist() == [0.51, 1000.0] and excluded["reason"].tolist() == ["floor_or_negative", "above_1000"]


def test_a_range_includes_its_own_first_hour(cfg, fixture, monkeypatch):
    from preprocessing import combine_dataset
    monkeypatch.setattr(combine_dataset, "VARIANTS", {"full": None, "from_march": "2023-03-01 00:00"})
    _, out = _build(cfg, fixture, rules=SPIKY)
    part = pd.read_csv(out / "from_march" / "combined_hourly.csv")
    assert part["local_timestamp"].min() == "2023-03-01T00:00:00+08:00"


# ------------------------------------------------------------------------------ the twin set that keeps pre-installation readings
def test_the_twin_folders_differ_from_the_clean_ones_only_by_the_pre_installation_readings(cfg, fixture):
    from preprocessing.combine_dataset import WITH_PREINSTALL
    summary, out = _build(cfg, fixture, rules=SPIKY, with_preinstall=True)
    assert set(summary["variants"]) == {n + s for n in VARIANTS for s in ("", WITH_PREINSTALL)}
    assert summary["variants"]["from_2023"]["pre_installation"] == "removed" and summary["variants"]["from_2023" + WITH_PREINSTALL]["pre_installation"] == "kept"
    for name, start in VARIANTS.items():
        clean = pd.read_csv(out / name / "combined_hourly.csv")
        twin = pd.read_csv(out / (name + WITH_PREINSTALL) / "combined_hourly.csv")
        extra = twin.merge(clean[["location_id", "local_timestamp"]], on=["location_id", "local_timestamp"], how="left", indicator=True)
        extra = extra[extra["_merge"] == "left_only"]
        assert set(clean["local_timestamp"] + clean["location_id"].astype(str)) <= set(twin["local_timestamp"] + twin["location_id"].astype(str))
        assert set(extra["location_id"]) <= {9000001} and (extra["local_timestamp"] < "2023-06-01").all()      # only QC, only before its date
        assert len(twin) - len(clean) == len(extra)
        # Gamma is installed 2023-06-01: 3,648 earlier hours, minus the 10 New Year hours (removed as fireworks in both sets) = 3,638 in
        # `full`; from 2023 the 21 other hours of 2022-12-31 are gone too (3,617); from 2024 all of them are after installation (0)
        assert len(extra) == {None: 3638, "2023-01-01": 3617, "2024-01-01": 0}[start]
        assert set(clean["location_id"]) == set(twin["location_id"])                                       # same stations, same spatial split


def test_the_twin_set_does_not_change_the_main_set_or_its_audit(cfg, fixture):
    only, out_a = _build(cfg, fixture, rules=SPIKY)
    both, out_b = _build(cfg, fixture, rules=SPIKY, with_preinstall=True)
    assert only["excluded"] == both["excluded"] and only["kept_rows"] == both["kept_rows"]
    assert pd.read_csv(out_a / "from_2023" / "combined_hourly.csv").equals(pd.read_csv(out_b / "from_2023" / "combined_hourly.csv"))


def test_no_twin_is_made_when_the_rule_is_already_off_or_when_asked_not_to(cfg, fixture):
    summary, _ = _build(cfg, fixture, rules={**SPIKY, "trim_before_install": False}, with_preinstall=True)
    assert set(summary["variants"]) == set(VARIANTS) and summary["variants"]["full"]["pre_installation"] == "removed"
    summary, out = _build(cfg, fixture, rules=SPIKY, with_preinstall=False)
    assert set(summary["variants"]) == set(VARIANTS) and not any(out.glob("*_with_preinstall"))


def test_the_pipeline_loader_reads_the_twin_folder_too(cfg, fixture):
    from preprocessing.combine_dataset import WITH_PREINSTALL
    _, out = _build(cfg, fixture, rules=SPIKY, with_preinstall=True)
    cfg["data"]["paths"].update(sensor_locations_csv=str(out / ("from_2023" + WITH_PREINSTALL) / "locations.csv"),
                                sensor_csv_dir=str(out / ("from_2023" + WITH_PREINSTALL) / "sensors"))
    clean = pd.read_csv(out / ("from_2023" + WITH_PREINSTALL) / "combined_hourly.csv")
    assert len(load_raw_readings(cfg)) == int(clean["inside_raster"].sum())
    assert "_with_preinstall" in (out / "README.md").read_text(encoding="utf-8") and "SAME rows" in (out / "README.md").read_text(encoding="utf-8")


def test_pre_installation_readings_never_change_how_other_readings_are_judged(cfg):
    """Seven stations at 22 and a spike of 400, plus a station whose reading of 600 predates its installation: the spike must be
    flagged whether or not that backfilled reading is kept, and the backfilled one is not itself called a spike."""
    ts = pd.Timestamp("2023-05-05 10:00")
    readings = pd.DataFrame({"location_id": range(1, 9), "sensor_id": 1, "ts": ts, "pm25_raw": [400.0] + [22.0] * 6 + [600.0]})
    st = pd.DataFrame({"location_id": range(1, 9), "installed": [pd.NaT] * 7 + [pd.Timestamp("2023-06-01")]})
    kept, excluded = apply_rules(readings, st)
    assert dict(zip(excluded["location_id"], excluded["reason"])) == {1: "isolated_spike", 8: "before_installation"}
    kept_twin, excluded_twin = apply_rules(readings, st, {**RULES, "trim_before_install": False})
    assert dict(zip(excluded_twin["location_id"], excluded_twin["reason"])) == {1: "isolated_spike"} and 8 in set(kept_twin["location_id"])

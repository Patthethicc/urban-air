import numpy as np
import pandas as pd
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from config_parser import _resolve_paths
from preprocessing.clean_loader import find_sensor_file, load_raw_readings
from preprocessing.qc_convert import HEADER, convert, describe, read_column_map, safe_name

ORIGIN_X, TOP_Y = 280_000.0, 1_610_000.0
TO_LL = Transformer.from_crs(32651, 4326, always_xy=True)
OPENAQ_HEADER = "location_name,location_id,sensor_id,utc_timestamp,local_timestamp,pm25_value,parameter,unit"


def _lonlat(dx, dy):
    return TO_LL.transform(ORIGIN_X + dx, TOP_Y - dy)


@pytest.fixture
def qc(tmp_path):
    """A tiny network: three sites, one of them (id 3) reported under two column names."""
    src = tmp_path / "QC Data"
    src.mkdir()
    (src / "QCDATA_SensorLoc.csv").write_text(
        "ID , Site name , District, Classification, Installation Date, Coordinates\n"
        + "\n".join(f'{i},{name} ,V,Roadside,{inst},"{lat},{lon}"' for i, name, inst, (lon, lat) in [
            (1, "Alpha  Market", "1/1/2024", _lonlat(300, 300)),
            (2, "Beta St. & Co", "1/3/2024", _lonlat(600, 400)),
            (3, "Gamma Hall", "1/1/2024", _lonlat(500, 800))]) + "\n", encoding="utf-8")
    stamps = pd.date_range("2024-01-01 00:00", periods=96, freq="h")
    a, b = np.arange(96) + 10.0, np.arange(96) + 50.0
    c_old, c_new = np.full(96, np.nan), np.full(96, np.nan)
    c_old[:48], c_new[24:] = 30.0, 31.0                       # the sensor was renamed on day 2: both report on day 2
    b[10:14] = np.nan
    wide = pd.DataFrame({"Date (mm/dd/yy) and Time (hh:mm)": [f"{t.month}/{t.day}/{t:%y} {t.hour}:00" for t in stamps],
                         "Alpha": a, "Beta": b, "Gamma_old": c_old, "Gamma_new": c_new})
    wide["Unnamed: 5"] = np.nan
    wide.loc[len(wide)] = ["AVERAGE", 1, 1, 1, 1, np.nan]
    wide.to_csv(src / "AQ_Sensors_Hourly_2024.csv", index=False)
    cmap = tmp_path / "map.csv"
    cmap.write_text("hourly_column,qc_id,note\nAlpha,1,\nBeta,2,\nGamma_new,3,primary\nGamma_old,3,fills gaps\n", encoding="utf-8")
    return src, cmap, tmp_path / "out"


def test_each_site_gets_an_openaq_formatted_file(qc):
    src, cmap, out = qc
    r = convert(src, out, cmap)
    names = sorted(p.name for p in out.glob("qcdata_*.csv"))
    assert names == ["qcdata_9000001_Alpha_Market.csv", "qcdata_9000002_Beta_St____Co.csv", "qcdata_9000003_Gamma_Hall.csv"]
    raw = (out / names[0]).read_bytes()
    assert raw.split(b"\r\n")[0].decode() == OPENAQ_HEADER and HEADER == OPENAQ_HEADER.split(",")   # same columns, CRLF as OpenAQ
    assert not raw.startswith(b"\xef\xbb\xbf") and "µg/m³".encode() in raw
    assert len(r["sites"]) == 3 and r["dropped_timestamps"] == ["AVERAGE"]


def test_timestamps_are_manila_local_with_offset_and_utc(qc):
    src, cmap, out = qc
    convert(src, out, cmap)
    d = pd.read_csv(out / "qcdata_9000001_Alpha_Market.csv")
    assert d.loc[0, "local_timestamp"] == "2024-01-01T00:00:00+08:00" and d.loc[0, "utc_timestamp"] == "2023-12-31T16:00:00Z"
    assert d.loc[0, "pm25_value"] == 10.0 and d["parameter"].eq("PM2.5").all() and d["unit"].eq("µg/m³").all()
    assert d["location_id"].eq(9000001).all() and d["location_name"].eq("Alpha Market").all()    # cleaned site name


def test_empty_hours_are_dropped_and_the_average_row_never_appears(qc):
    src, cmap, out = qc
    convert(src, out, cmap)
    d = pd.read_csv(out / "qcdata_9000002_Beta_St____Co.csv")
    assert len(d) == 96 - 4 and d["pm25_value"].notna().all() and not d["local_timestamp"].str.contains("AVERAGE").any()


def test_two_columns_for_one_site_merge_first_listed_wins_and_the_overlap_is_reported(qc):
    src, cmap, out = qc
    r = convert(src, out, cmap)
    d = pd.read_csv(out / "qcdata_9000003_Gamma_Hall.csv")
    assert len(d) == 96 and d["local_timestamp"].is_unique
    assert d.loc[:23, "pm25_value"].eq(30.0).all() and d.loc[24:, "pm25_value"].eq(31.0).all()   # new column wins on the overlap
    site = next(s for s in r["sites"] if s["qc_id"] == 3)
    assert site["columns"] == ["Gamma_new", "Gamma_old"] and "24 overlapping hours" in site["notes"][0]
    assert "mean |diff|" in describe(r)


def test_the_locations_file_has_the_openaq_columns_and_coordinates(qc):
    src, cmap, out = qc
    loc = pd.read_csv(convert(src, out, cmap)["locations_csv"])
    assert list(loc.columns) == ["name", "location_id", "latitude", "longitude"] and loc["location_id"].tolist() == [9000001, 9000002, 9000003]
    lon, lat = _lonlat(300, 300)
    assert loc.loc[0, "latitude"] == pytest.approx(lat, abs=1e-5) and loc.loc[0, "longitude"] == pytest.approx(lon, abs=1e-5)


def test_trim_before_install_drops_readings_before_the_listed_date_and_is_off_by_default(qc):
    src, cmap, out = qc
    convert(src, out, cmap)
    assert len(pd.read_csv(next(out.glob("qcdata_9000002_*.csv")))) == 92          # default: nothing trimmed (96 h - 4 empty)
    r = convert(src, out, cmap, trim_before_install=True)               # site 2 is installed 1/3: its first 48 h go
    d = pd.read_csv(next(out.glob("qcdata_9000002_*.csv")))
    assert d["local_timestamp"].iloc[0] == "2024-01-03T00:00:00+08:00" and len(d) == 48
    assert next(x for x in r["sites"] if x["qc_id"] == 2)["trimmed"] == 44         # 48 h minus the 4 that were empty anyway
    assert next(x for x in r["sites"] if x["qc_id"] == 1)["trimmed"] == 0          # installed on the first day
    assert "trimmed 44 h before installation" in describe(r)


def test_a_rerun_replaces_the_old_files(qc):
    src, cmap, out = qc
    out.mkdir()
    (out / "qcdata_1_stale.csv").write_text("x")
    convert(src, out, cmap)
    assert not (out / "qcdata_1_stale.csv").exists() and len(list(out.glob("qcdata_*.csv"))) == 3


def test_a_column_missing_from_the_map_is_an_error_not_a_silent_skip(qc):
    src, cmap, out = qc
    cmap.write_text("hourly_column,qc_id,note\nAlpha,1,\nBeta,2,\nGamma_new,3,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Gamma_old"):
        convert(src, out, cmap)


def test_an_unknown_site_id_and_a_repeated_column_are_errors(qc):
    src, cmap, out = qc
    cmap.write_text("hourly_column,qc_id,note\nAlpha,1,\nBeta,2,\nGamma_new,99,\nGamma_old,3,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="99"):
        convert(src, out, cmap)
    cmap.write_text("hourly_column,qc_id,note\nAlpha,1,\nAlpha,2,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        read_column_map(cmap)


def test_a_blank_site_id_skips_the_column(qc):
    src, cmap, out = qc
    cmap.write_text("hourly_column,qc_id,note\nAlpha,1,\nBeta,2,\nGamma_new,3,\nGamma_old,,skip\n", encoding="utf-8")
    r = convert(src, out, cmap)
    assert r["skipped_columns"] == ["Gamma_old"] and (pd.read_csv(next(out.glob("qcdata_9000003_*.csv"))).shape[0] == 72)


def test_the_shipped_column_map_covers_every_column_and_each_site_once_except_commonwealth():
    from pathlib import Path
    m = read_column_map(Path(__file__).resolve().parents[1] / "configs" / "qc_column_map.csv")
    ids = [i for i in m.values() if i is not None]
    assert sorted(set(ids)) == list(range(1, 41)) and len(m) == 41
    assert [c for c in m if m[c] == 8] == ["Commonwealth_BHall", "Commonwealth_BH"]       # the renamed sensor; BHall is primary


def test_safe_names_follow_the_openaq_rule():
    assert safe_name("Gen. Luna St. corner Gov. Pascual Ave.") == "Gen__Luna_St__corner_Gov__Pascual_Ave_"


# ------------------------------------------------------------------------------ the loader accepts qcdata_ files
def test_find_sensor_file_accepts_both_prefixes_and_a_list_of_folders(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "sub").mkdir(parents=True)
    b.mkdir()
    (a / "sub" / "openaq_111_x.csv").write_text("")
    (b / "qcdata_9000001_y.csv").write_text("")
    assert find_sensor_file(str(a), 111).endswith("openaq_111_x.csv")
    assert find_sensor_file(str(b), 9000001).endswith("qcdata_9000001_y.csv")
    assert find_sensor_file([str(a), str(b)], 9000001).endswith("qcdata_9000001_y.csv")
    assert find_sensor_file(str(a), 9000001) is None


def test_list_valued_paths_resolve_to_absolute_paths():
    cfg = {"paths": {"one": "x", "many": ["a", "b"], "none": None}}
    _resolve_paths(cfg)
    from pathlib import Path
    assert Path(cfg["paths"]["one"]).is_absolute() and Path(cfg["paths"]["one"]).name == "x"
    assert isinstance(cfg["paths"]["many"], list) and [Path(q).name for q in cfg["paths"]["many"]] == ["a", "b"]
    assert all(Path(q).is_absolute() for q in cfg["paths"]["many"]) and cfg["paths"]["none"] is None


def test_converted_files_load_through_the_real_loader_alone_and_next_to_openaq(qc, cfg, tmp_path):
    src, cmap, out = qc
    convert(src, out, cmap)
    raster = tmp_path / "r.tif"
    with rasterio.open(raster, "w", driver="GTiff", height=200, width=200, count=1, dtype="float32", crs="EPSG:32651",
                       transform=from_origin(ORIGIN_X, TOP_Y, 10, 10)) as dst:
        dst.write(np.ones((1, 200, 200), "float32"))
    oa_dir = tmp_path / "oa"
    oa_dir.mkdir()
    lon, lat = _lonlat(1000, 1000)
    pd.DataFrame({"name": ["OA"], "location_id": [3015285], "latitude": [lat], "longitude": [lon]}).to_csv(oa_dir / "loc.csv", index=False)
    pd.DataFrame({"local_timestamp": ["2024-01-01T00:00:00+08:00", "2024-01-01T01:00:00+08:00"], "pm25_value": [5.0, 6.0]}).to_csv(
        oa_dir / "openaq_3015285_OA.csv", index=False)

    cfg["data"]["paths"].update(master_raster=str(raster), sensors_gpkg=None, boundary=None,
                                sensor_locations_csv=str(out / "QC Sensor Locations.csv"), sensor_csv_dir=str(out))
    only_qc = load_raw_readings(cfg)
    assert sorted(only_qc["location_id"].unique()) == [9000001, 9000002, 9000003]
    first = only_qc[only_qc["location_id"] == 9000001].iloc[0]
    assert first["local_timestamp"] == pd.Timestamp("2024-01-01 00:00") and first["pm25_value"] == 10.0     # naive Manila time, as for OpenAQ

    cfg["data"]["paths"].update(sensor_locations_csv=[str(oa_dir / "loc.csv"), str(out / "QC Sensor Locations.csv")],
                                sensor_csv_dir=[str(oa_dir), str(out)])
    both = load_raw_readings(cfg)
    assert sorted(both["location_id"].unique()) == [3015285, 9000001, 9000002, 9000003]

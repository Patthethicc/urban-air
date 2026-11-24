"""One cleaned dataset from both sensor networks (OpenAQ + Quezon City), in three year ranges.

  python -m preprocessing.combine_dataset        # (run from model/) reads ../data/OpenAQ, ../data/QC Data Converted; writes ../data/Combined Dataset

Cleaning follows eda/eda_results/EDA_Report.md plus what an audit of the files added. A reading is removed, and listed with its
reason in excluded_readings.csv, when it is:
  no_value             not a number
  floor_or_negative    <= 0.5 ug/m3 (zero and negative readings are sensor dropouts; EDA recommendation 2)
  above_1000           beyond any physical reading (the pipeline's data.clean maximum)
  new_year_fireworks   Dec 31 21:00 - Jan 1 06:00 local (mean 116 against 19 ug/m3 outside; EDA recommendation 1)
  before_installation  a QC reading earlier than the site's listed installation date: it follows the network median at
                       r = 0.97 (0.85 afterwards), so it looks backfilled
  isolated_spike       > 150 ug/m3 and more than 5 times the mean of the other stations that hour (>= 5 reporting): a
                       single-sensor spike; the other stations are normal in those hours
Rules run in that order on the whole record, then the year ranges are cut, so every range is an exact subset of `full`.
Each range is written twice: without the pre-installation readings (full, from_2023, from_2024) and with them
(*_with_preinstall), identical in every other respect, so the one rule can be tested as an experiment. Both sets share the
same stations, hence the same spatial split; but score them on the SAME rows (see README, `--baseline`), because the kept
pre-installation rows are the backfilled ones and are easy to predict.
Nothing is interpolated or rescaled; the pipeline still fills gaps up to 24 h itself. The two networks read differently
(median hourly network mean 13.2 against 17.2 ug/m3 where both operate) and no pair of stations is close enough to calibrate
them against each other, so `network` is a column and the values are left as measured.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.qc_convert import ID_BASE, read_locations, safe_name, to_rows

VARIANTS = {"full": None, "from_2023": "2023-01-01", "from_2024": "2024-01-01"}      # from_X drops every year before X
WITH_PREINSTALL = "_with_preinstall"      # folder suffix of the copies that keep the QC readings before the installation date
PREFIX = {"OpenAQ": "openaq", "QC": "qcdata"}
RULES = {"floor": 0.5, "ceiling": 1000.0, "spike_min": 150.0, "spike_factor": 5.0, "spike_min_others": 5,
         "trim_before_install": True}
REASONS = ["no_value", "floor_or_negative", "above_1000", "new_year_fireworks", "before_installation", "isolated_spike"]
COLUMNS = ["location_id", "location_name", "network", "sensor_id", "latitude", "longitude", "x_utm51n", "y_utm51n",
           "inside_raster", "classification", "local_timestamp", "pm25_value", "hour_sin", "hour_cos"]


def read_stations(openaq_dir: Path, qc_dir: Path, qc_raw_dir: Path, crs: str, raster_path: str) -> pd.DataFrame:
    """Both networks' stations with names from the location tables, UTM coordinates and whether the raster covers them."""
    import rasterio
    from pyproj import Transformer

    oa = pd.read_csv(next(openaq_dir.glob("*Locations*.csv")))
    qc = pd.read_csv(qc_dir / "QC Sensor Locations.csv")
    sites = read_locations(qc_raw_dir / "QCDATA_SensorLoc.csv").assign(location_id=lambda d: ID_BASE + d["qc_id"])
    qc = qc.merge(sites[["location_id", "district", "classification", "installed"]], on="location_id", how="left")
    st = pd.concat([oa.assign(network="OpenAQ", classification="OpenAQ node"), qc.assign(network="QC")], ignore_index=True)
    st = st.rename(columns={"name": "location_name"})
    st["location_name"] = st["location_name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    st["x_utm51n"], st["y_utm51n"] = Transformer.from_crs(4326, crs, always_xy=True).transform(
        st["longitude"].to_numpy(), st["latitude"].to_numpy())
    with rasterio.open(raster_path) as r:
        b = r.bounds
    st["inside_raster"] = st["x_utm51n"].between(b.left, b.right) & st["y_utm51n"].between(b.bottom, b.top)
    return st[["location_id", "location_name", "network", "classification", "district", "installed", "latitude", "longitude",
               "x_utm51n", "y_utm51n", "inside_raster"]]


def read_readings(folders: dict, stations: pd.DataFrame, timezone: str) -> tuple:
    """Every station's file -> (location_id, sensor_id, ts [naive local time], pm25_raw), and the stations without a file."""
    frames, missing = [], []
    for st in stations.itertuples():
        hits = sorted(folders[st.network].glob(f"{PREFIX[st.network]}_{st.location_id}_*.csv"))
        if not hits:
            missing.append(int(st.location_id))
            continue
        d = pd.read_csv(hits[0], usecols=["sensor_id", "local_timestamp", "pm25_value"])
        frames.append(pd.DataFrame({
            "location_id": st.location_id, "sensor_id": d["sensor_id"],
            "ts": pd.to_datetime(d["local_timestamp"], utc=True).dt.tz_convert(timezone).dt.tz_localize(None),
            "pm25_raw": pd.to_numeric(d["pm25_value"], errors="coerce")}))
    return pd.concat(frames, ignore_index=True), missing


def new_year_window(ts: pd.Series) -> pd.Series:
    m, d, h = ts.dt.month, ts.dt.day, ts.dt.hour
    return ((m == 1) & (d == 1) & (h <= 6)) | ((m == 12) & (d == 31) & (h >= 21))


def apply_rules(readings: pd.DataFrame, stations: pd.DataFrame, rules: dict = RULES) -> tuple:
    """(kept readings, excluded readings with their reason). Each reading gets the FIRST rule it breaks."""
    v = readings["pm25_raw"]
    reason = pd.Series("", index=readings.index, dtype=object)

    def mark(mask, label):
        reason[mask & (reason == "")] = label

    mark(v.isna(), "no_value")
    mark(v <= rules["floor"], "floor_or_negative")
    mark(v > rules["ceiling"], "above_1000")
    mark(new_year_window(readings["ts"]), "new_year_fireworks")
    pre = readings["ts"] < readings["location_id"].map(stations.set_index("location_id")["installed"])     # NaT: never
    if rules["trim_before_install"]:
        mark(pre, "before_installation")

    # The spike rule compares a reading with the other stations of that hour. Removed values do not count as "others", and neither
    # do pre-installation ones (kept or not), nor are they judged as spikes: so the sets with and without them differ by exactly them.
    alive = readings[(reason == "") & ~pre]
    hour = alive.groupby("ts")["pm25_raw"]
    others = hour.transform("count") - 1
    others_mean = (hour.transform("sum") - alive["pm25_raw"]) / others.where(others > 0)
    spike = ((alive["pm25_raw"] > rules["spike_min"]) & (others >= rules["spike_min_others"])
             & (alive["pm25_raw"] > rules["spike_factor"] * others_mean))
    reason[spike[spike].index] = "isolated_spike"

    return readings[reason == ""], readings.assign(reason=reason)[reason != ""]


def assemble(kept: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """The model-input table: the cleaned reading, its station and the hour-of-day encoding the model uses
    (sin and cos of 2 pi hour / 24)."""
    df = kept.merge(stations.drop(columns=["district", "installed"]), on="location_id", how="left")
    hour = df["ts"].dt.hour
    df["local_timestamp"] = df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")
    df["pm25_value"] = df["pm25_raw"]
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24).round(6), np.cos(2 * np.pi * hour / 24).round(6)
    return df.sort_values(["location_id", "ts"]).reset_index(drop=True)[COLUMNS + ["ts"]]


def write_pipeline_folder(df: pd.DataFrame, folder: Path, timezone: str) -> int:
    """sensors/<prefix>_<id>_<name>.csv in the OpenAQ layout + locations.csv: point data.yaml at them to train on this range."""
    out = folder / "sensors"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.csv"):
        old.unlink()
    for (sid, name, network), g in df.groupby(["location_id", "location_name", "network"], sort=False):
        rows = to_rows({"name": name, "location_id": int(sid)}, pd.Series(g["pm25_value"].to_numpy(), index=g["ts"]), timezone)
        rows["sensor_id"] = g["sensor_id"].to_numpy()
        rows.to_csv(out / f"{PREFIX[network]}_{int(sid)}_{safe_name(name)}.csv", index=False, encoding="utf-8")
    loc = df.drop_duplicates("location_id")[["location_name", "location_id", "latitude", "longitude"]].rename(columns={"location_name": "name"})
    loc.to_csv(folder / "locations.csv", index=False, encoding="utf-8")
    return int(len(loc))


def build(openaq_dir, qc_dir, qc_raw_dir, out_dir, cfg: dict, rules: dict = RULES, sensor_folders: bool = True,
          with_preinstall: bool = True) -> dict:
    """Write the cleaned ranges, and (with_preinstall) a twin set that keeps the QC readings before installation."""
    openaq_dir, qc_dir, out_dir = Path(openaq_dir), Path(qc_dir), Path(out_dir)
    tz, crs = cfg["data"]["timezone"], cfg["data"]["crs"]
    stations = read_stations(openaq_dir, qc_dir, Path(qc_raw_dir), crs, cfg["data"]["paths"]["master_raster"])
    readings, no_file = read_readings({"OpenAQ": openaq_dir, "QC": qc_dir}, stations, tz)
    sets = {"": rules}
    if with_preinstall and rules["trim_before_install"]:
        sets[WITH_PREINSTALL] = {**rules, "trim_before_install": False}

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"raw_rows": int(len(readings)), "stations_without_file": no_file, "variants": {}}
    counts = {}
    for suffix, set_rules in sets.items():
        kept, excluded = apply_rules(readings, stations, set_rules)
        table = assemble(kept, stations)
        if not suffix:                                                   # the audit and the README describe the main set
            main = table
            excl = excluded.assign(local_timestamp=excluded["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S+08:00"))
            excl[["location_id", "local_timestamp", "pm25_raw", "reason"]].sort_values(["location_id", "local_timestamp"]).to_csv(
                out_dir / "excluded_readings.csv", index=False)
            summary.update(kept_rows=int(len(table)), excluded={r: int((excluded["reason"] == r).sum()) for r in REASONS})
        for name, start in VARIANTS.items():
            part = table if start is None else table[table["ts"] >= pd.Timestamp(start)]
            folder = out_dir / (name + suffix)
            folder.mkdir(exist_ok=True)
            part.drop(columns="ts").to_csv(folder / "combined_hourly.csv", index=False, encoding="utf-8")
            n_files = write_pipeline_folder(part, folder, tz) if sensor_folders else 0
            counts[name + suffix] = part.groupby("location_id").size()
            summary["variants"][name + suffix] = {
                "from": start or "2021-07-09", "rows": int(len(part)), "stations": int(part["location_id"].nunique()),
                "stations_inside_raster": int(part.loc[part["inside_raster"], "location_id"].nunique()), "sensor_files": n_files,
                "pre_installation": "kept" if suffix else "removed"}
    table = main
    st = stations.assign(**{f"readings_{n}": stations["location_id"].map(c).fillna(0).astype(int) for n, c in counts.items()})
    span = table.groupby("location_id")["ts"].agg(first_reading="min", last_reading="max")
    st = st.merge(span, on="location_id", how="left")
    st["installed"] = st["installed"].dt.strftime("%Y-%m-%d")
    st.to_csv(out_dir / "stations.csv", index=False, encoding="utf-8")
    write_readme(out_dir, table, summary)
    return summary


def write_readme(out_dir: Path, table: pd.DataFrame, summary: dict) -> None:
    """A short data dictionary next to the data, with the counts of this build."""
    year = table.assign(year=table["ts"].dt.year).groupby(["year", "network"])["location_id"].nunique().unstack(fill_value=0)
    rows = [f"| {y} | {int(r.get('OpenAQ', 0))} | {int(r.get('QC', 0))} |" for y, r in year.iterrows()]
    variants = [f"| `{k}` | {v['from']} | {v['rows']:,} | {v['stations']} ({v['stations_inside_raster']} inside the raster) | {v['pre_installation']} |"
                for k, v in summary["variants"].items()]
    removed = [f"| `{r}` | {n:,} |" for r, n in summary["excluded"].items()]
    text = f"""# Combined PM2.5 dataset: OpenAQ + Quezon City

Built by `python -m preprocessing.combine_dataset` (from `model/`). {summary['raw_rows']:,} readings in, {summary['kept_rows']:,} kept.

| Folder | Starts | Rows | Stations | QC pre-installation readings |
|---|---|---|---|---|
{chr(10).join(variants)}

`from_2023` drops 2021-2022 and `from_2024` drops 2021-2023, cut from `full`, so each is an exact subset of it. The `*_with_preinstall`
folders are the same ranges with the QC readings before the sites' installation dates KEPT (they look backfilled, see below), so the
effect of that one rule can be tested: train once on each, with the same settings. The stations, and so the spatial split, are the same
in both. Compare the two runs on the SAME rows (`--stage report --run-id <with> --baseline <without>` scores only the rows both have),
because the kept pre-installation rows are the ones that track the network median and are easy to predict, which would flatter a
model scored on them. Every folder holds
`combined_hourly.csv` (one table, columns below) and what the pipeline reads: `sensors/` (one file per station, OpenAQ layout) and
`locations.csv`. To train on a range, set in `configs/data.yaml`:
`sensor_locations_csv: ../data/Combined Dataset/<folder>/locations.csv` and `sensor_csv_dir: ../data/Combined Dataset/<folder>/sensors`.

## Columns of `combined_hourly.csv`
`location_id`, `location_name`, `network` (OpenAQ or QC), `sensor_id`, `latitude`, `longitude`, `x_utm51n`, `y_utm51n` (EPSG:32651),
`inside_raster` (the Master GeoTIFF covers the station), `classification` (roadside / non-roadside for QC), `local_timestamp`
(Asia/Manila, +08:00), `pm25_value` (ug/m3, cleaned), `hour_sin`, `hour_cos` (the hour-of-day encoding the model uses).
`stations.csv` lists every station with its reading counts per folder. Lags and the spatial lag are NOT stored: they depend on which
stations train, so the pipeline builds them per fold.

## Readings removed (`excluded_readings.csv` lists each one)
| Reason | Readings |
|---|---|
{chr(10).join(removed)}

Rules, in this order, are explained in `preprocessing/combine_dataset.py`. Nothing is interpolated or rescaled.

## Stations reporting per year
| Year | OpenAQ | QC |
|---|---|---|
{chr(10).join(rows)}

## Read with care
- The networks read differently where both operate (median hourly network mean 13.2 against 17.2 ug/m3) and no two stations are
  close enough to calibrate them against each other. Values are left as measured; use `network` if it matters.
- QC has no readings after 2025-12-31, so 2026 is OpenAQ only; 2021-2022 have few stations reporting at once.
- Station 4797655 is in the OpenAQ location table but has no data file; Tunasan (4797656) lies south of the raster.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def describe(summary: dict) -> str:
    lines = [f"{summary['raw_rows']:,} readings -> {summary['kept_rows']:,} kept"]
    lines += [f"  removed {n:>7,}  {r}" for r, n in summary["excluded"].items()]
    if summary["stations_without_file"]:
        lines.append(f"  stations with a location but no data file: {summary['stations_without_file']}")
    lines += [f"{k:>25}: from {v['from']}  {v['rows']:>9,} rows  {v['stations']} stations ({v['stations_inside_raster']} inside the raster), "
              f"pre-installation {v['pre_installation']}" for k, v in summary["variants"].items()]
    return "\n".join(lines)


def main():
    from config_parser import load_config

    root = Path(__file__).resolve().parents[2] / "data"
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=str(root / "Combined Dataset"))
    ap.add_argument("--only-clean", action="store_true", help="skip the *_with_preinstall twin folders")
    ap.add_argument("--no-sensor-folders", action="store_true", help="write only the combined tables, not the per-sensor files")
    a = ap.parse_args()
    summary = build(root / "OpenAQ", root / "QC Data Converted", root / "QC Data", a.out_dir, load_config(),
                    RULES, not a.no_sensor_folders, not a.only_clean)
    print(describe(summary))


if __name__ == "__main__":
    main()

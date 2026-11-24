"""Convert the Quezon City sensor network (data/QC Data) to the per-sensor CSV layout the loader reads.

The raw data are one wide table per year (one column per sensor, local time) plus a site table. The output follows the
OpenAQ files: one `qcdata_<location_id>_<name>.csv` per site with the OpenAQ columns, and a locations CSV with
`name,location_id,latitude,longitude`. The loader accepts `qcdata_` files exactly like `openaq_` files.

  python preprocessing/qc_convert.py                         # reads ../data/QC Data, writes ../data/QC Data Converted

Which hourly column belongs to which site is NOT guessable from the files (the columns are short codes, the site table
has full names), so it lives in configs/qc_column_map.csv. Two columns can share a site (renamed sensor): the first one
listed is used and the later ones only fill the hours it is empty.

Measurements are not altered: only empty hours and the table's AVERAGE row are dropped. Out-of-range values (the
network has negatives and values above 1000) are left for the pipeline's `data.clean` range to remove.

Caution: the 2024 sites (ids 21-40) report from 2024-01-01 although the site table lists later installation dates
(up to 85 days). In that period they follow the median of the other sites at r~0.97 (r~0.77 once installed), which
independent sensors would not, so those hours look backfilled. `--trim-before-install` drops readings before the
listed installation date; it is off by default so the conversion stays faithful to the source.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

HEADER = ["location_name", "location_id", "sensor_id", "utc_timestamp", "local_timestamp", "pm25_value", "parameter", "unit"]
ID_BASE = 9_000_000                  # location_id = ID_BASE + the site's ID (1-40): cannot collide with OpenAQ's 7-digit ids
PREFIX = "qcdata"
TIME_FORMAT = "%m/%d/%y %H:%M"
LOCATIONS_NAME = "QC Sensor Locations.csv"


def read_locations(path) -> pd.DataFrame:
    """Site table -> qc_id, name, district, classification, installed, latitude, longitude."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    coords = raw["Coordinates"].astype(str).str.split(",", n=1, expand=True)
    return pd.DataFrame({
        "qc_id": raw["ID"].astype(int),
        "name": raw["Site name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
        "district": raw["District"].astype(str).str.strip(),
        "classification": raw["Classification"].astype(str).str.strip(),
        "installed": pd.to_datetime(raw["Installation Date"].astype(str).str.strip(), format="mixed", errors="coerce"),
        "latitude": pd.to_numeric(coords[0].str.strip()),
        "longitude": pd.to_numeric(coords[1].str.strip())})


def read_column_map(path) -> dict:
    """hourly column -> qc_id, in file order (order = priority when two columns share a site). Blank qc_id = skipped."""
    raw = pd.read_csv(path)
    out = {}
    for col, qc_id in zip(raw["hourly_column"].astype(str).str.strip(), raw["qc_id"]):
        if col in out:
            raise ValueError(f"column map lists {col!r} twice")
        out[col] = None if pd.isna(qc_id) else int(qc_id)
    return out


def read_hourly(files) -> tuple:
    """Yearly wide tables -> (one hourly frame indexed by local time, rows dropped for an unreadable timestamp)."""
    frames, dropped = [], []
    for f in sorted(files):
        raw = pd.read_csv(f)
        raw.columns = [str(c).strip() for c in raw.columns]
        stamp = pd.to_datetime(raw.iloc[:, 0], format=TIME_FORMAT, errors="coerce")
        dropped += raw.loc[stamp.isna(), raw.columns[0]].astype(str).tolist()          # e.g. the AVERAGE row
        raw = raw.loc[stamp.notna()].drop(columns=raw.columns[0]).set_axis(stamp[stamp.notna()], axis=0)
        frames.append(raw.loc[:, ~raw.columns.str.startswith("Unnamed")])
    wide = pd.concat(frames)
    if wide.index.duplicated().any():
        raise ValueError(f"{int(wide.index.duplicated().sum())} duplicated timestamps across the yearly files")
    return wide.apply(pd.to_numeric, errors="coerce").sort_index(), dropped


def safe_name(name: str) -> str:
    """OpenAQ's file-name rule: every character that is not a letter or digit becomes '_'."""
    return re.sub(r"[^A-Za-z0-9]", "_", name)


def _site_series(wide: pd.DataFrame, columns: list) -> tuple:
    """One series for a site: the first column, gaps filled from the others. Returns it with a note on any overlap."""
    series = wide[columns[0]]
    notes = []
    for other in columns[1:]:
        both = series.notna() & wide[other].notna()
        if both.any():
            notes.append(f"{other}: {int(both.sum())} overlapping hours, mean |diff| {(series[both] - wide[other][both]).abs().mean():.2f}")
        series = series.combine_first(wide[other])
    return series, notes


def to_rows(site: dict, series: pd.Series, timezone: str) -> pd.DataFrame:
    """Local naive hourly series -> OpenAQ rows (ISO local time with offset, and UTC)."""
    series = series.dropna()
    local = series.index.to_series().dt.tz_localize(timezone)
    offset = local.dt.strftime("%z")
    return pd.DataFrame({
        "location_name": site["name"], "location_id": site["location_id"], "sensor_id": site["location_id"],
        "utc_timestamp": local.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
        "local_timestamp": (local.dt.strftime("%Y-%m-%dT%H:%M:%S") + offset.str[:3] + ":" + offset.str[3:]).to_numpy(),
        "pm25_value": series.to_numpy(), "parameter": "PM2.5", "unit": "µg/m³"})[HEADER]


def convert(qc_dir, out_dir, column_map, timezone="Asia/Manila", trim_before_install=False) -> dict:
    """Write one qcdata_<id>_<name>.csv per site and the locations CSV. Returns a summary (also per site)."""
    qc_dir, out_dir = Path(qc_dir), Path(out_dir)
    sites = read_locations(qc_dir / "QCDATA_SensorLoc.csv").set_index("qc_id")
    cmap = read_column_map(column_map)
    wide, dropped = read_hourly(qc_dir.glob("AQ_Sensors_Hourly_*.csv"))

    unmapped = [c for c in wide.columns if c not in cmap]
    if unmapped:
        raise ValueError(f"hourly columns missing from the column map: {unmapped}")
    bad = sorted({i for i in cmap.values() if i is not None and i not in sites.index})
    if bad:
        raise ValueError(f"column map names site ids not in the site table: {bad}")

    by_site = {}
    for col, qc_id in cmap.items():
        if qc_id is not None and col in wide.columns:
            by_site.setdefault(qc_id, []).append(col)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{PREFIX}_*.csv"):                                         # a re-run replaces, never mixes
        old.unlink()
    summary, locations = [], []
    for qc_id, columns in sorted(by_site.items()):
        site = {"name": sites.loc[qc_id, "name"], "location_id": ID_BASE + qc_id}
        series, notes = _site_series(wide, columns)
        trimmed = 0
        if trim_before_install and pd.notna(sites.loc[qc_id, "installed"]):
            before = series.index < sites.loc[qc_id, "installed"]
            trimmed = int(series[before].notna().sum())
            series = series[~before]
        rows = to_rows(site, series, timezone)
        path = out_dir / f"{PREFIX}_{site['location_id']}_{safe_name(site['name'])}.csv"
        rows.to_csv(path, index=False, lineterminator="\r\n", encoding="utf-8")
        locations.append({"name": site["name"], "location_id": site["location_id"],
                          "latitude": sites.loc[qc_id, "latitude"], "longitude": sites.loc[qc_id, "longitude"]})
        first = series.first_valid_index()
        summary.append({"qc_id": qc_id, "location_id": site["location_id"], "name": site["name"], "columns": columns,
                        "rows": len(rows), "first": first, "installed": sites.loc[qc_id, "installed"], "notes": notes,
                        "trimmed": trimmed})
    pd.DataFrame(locations).to_csv(out_dir / LOCATIONS_NAME, index=False, lineterminator="\r\n", encoding="utf-8")

    return {"sites": summary, "out_dir": out_dir, "locations_csv": out_dir / LOCATIONS_NAME,
            "dropped_timestamps": dropped, "skipped_columns": [c for c, i in cmap.items() if i is None],
            "sites_without_data": sorted(set(sites.index) - set(by_site))}


def describe(result: dict) -> str:
    lines = [f"Wrote {len(result['sites'])} sites -> {result['out_dir']}", f"Locations: {result['locations_csv']}"]
    for s in result["sites"]:
        early = (isinstance(s["first"], pd.Timestamp) and pd.notna(s["installed"]) and s["first"] < s["installed"] - pd.Timedelta(days=1))
        lines.append(f"  {s['location_id']}  {s['name'][:44]:<44} {s['rows']:>6} rows  first {str(s['first'])[:10]}"
                     f"{'  (!! before its installation date ' + str(s['installed'])[:10] + ')' if early else ''}"
                     f"{'  [' + ' + '.join(s['columns']) + ']' if len(s['columns']) > 1 else ''}"
                     f"{'  (trimmed ' + str(s['trimmed']) + ' h before installation)' if s['trimmed'] else ''}")
        lines += [f"      {n}" for n in s["notes"]]
    unknown = [s["location_id"] for s in result["sites"] if pd.isna(s["installed"])]
    if unknown:
        lines.append(f"Installation date unreadable for {len(unknown)} sites, so their first reading was not checked against it")
    if result["dropped_timestamps"]:
        lines.append(f"Dropped rows with an unreadable timestamp: {sorted(set(result['dropped_timestamps']))}")
    if result["skipped_columns"]:
        lines.append(f"Columns skipped by the map: {result['skipped_columns']}")
    if result["sites_without_data"]:
        lines.append(f"Sites in the site table with no column: {result['sites_without_data']}")
    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qc-dir", default=str(root.parent / "data" / "QC Data"))
    ap.add_argument("--out-dir", default=str(root.parent / "data" / "QC Data Converted"))
    ap.add_argument("--column-map", default=str(root / "configs" / "qc_column_map.csv"))
    ap.add_argument("--timezone", default="Asia/Manila", help="time zone of the (naive) QC timestamps")
    ap.add_argument("--trim-before-install", action="store_true",
                    help="drop readings before each site's listed installation date (the 2024 sites look backfilled)")
    a = ap.parse_args()
    print(describe(convert(a.qc_dir, a.out_dir, a.column_map, a.timezone, a.trim_before_install)))


if __name__ == "__main__":
    main()

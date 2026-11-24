"""Download hourly weather for the NCR from the Open-Meteo historical archive (ERA5 reanalysis, free, no key).

Amouei (2023) used temperature, relative humidity and wind (as a vector) from the nearest meteorological station.
Manila has no open hourly station network covering 2023-2026, so a small grid of reanalysis points over the NCR
stands in for the stations; every PM2.5 station takes the values of its nearest point (preprocessing/weather.py),
like Amouei's "distance to nearest hub".

    python -m preprocessing.fetch_weather                       # from amouei-model/, writes ../data/Weather/
    python -m preprocessing.fetch_weather --start 2023-01-01 --end 2026-11-19

Only the grid coordinates and the date range are sent. Times are requested in Asia/Manila, the same local time
the sensor files use (local_timestamp). Re-running skips the download when the file already covers the range.
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "precipitation",
             "surface_pressure"]
# 3 x 3 points over the NCR (lat 14.40-14.75 N, lon 120.90-121.15 E), about 13 km apart: ERA5 is a 0.25 degree grid,
# so finer points would repeat the same cells.
LATS = [14.45, 14.575, 14.70]
LONS = [120.95, 121.025, 121.10]
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "Weather"


def fetch(lats, lons, start: str, end: str, retries: int = 4) -> list[dict]:
    pts = [(la, lo) for la in lats for lo in lons]
    q = {"latitude": ",".join(str(p[0]) for p in pts), "longitude": ",".join(str(p[1]) for p in pts),
         "start_date": start, "end_date": end, "hourly": ",".join(VARIABLES), "timezone": "Asia/Manila",
         "wind_speed_unit": "ms"}
    req = urllib.request.Request(f"{URL}?{urllib.parse.urlencode(q)}", headers={"User-Agent": "urban-air-research/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                body = json.loads(r.read().decode("utf-8"))
            return body if isinstance(body, list) else [body]
        except Exception as e:                              # rate limit / timeout: wait and retry
            if attempt == retries - 1:
                raise
            wait = 20 * (attempt + 1)
            print(f"  {type(e).__name__}: {e}; retry in {wait} s")
            time.sleep(wait)


def to_frame(points: list[dict]) -> pd.DataFrame:
    parts = []
    for i, p in enumerate(points):
        h = p["hourly"]
        df = pd.DataFrame({k: h[k] for k in VARIABLES})
        df.insert(0, "local_timestamp", pd.to_datetime(h["time"]))
        df.insert(0, "longitude", p["longitude"])
        df.insert(0, "latitude", p["latitude"])
        df.insert(0, "point_id", i)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=None, help="last day (default: 6 days ago, the archive's delay)")
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    end = a.end or str(date.today() - timedelta(days=6))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "open_meteo_ncr_hourly.csv"
    if target.is_file():
        old = pd.read_csv(target, usecols=["local_timestamp"], parse_dates=["local_timestamp"])
        if old["local_timestamp"].min() <= pd.Timestamp(a.start) and old["local_timestamp"].max() >= pd.Timestamp(end):
            print(f"{target} already covers {a.start} .. {end}: nothing to do")
            return
    # one request per year keeps each answer small
    frames = []
    for y0 in range(int(a.start[:4]), int(end[:4]) + 1):
        s = max(a.start, f"{y0}-01-01")
        e = min(end, f"{y0}-12-31")
        print(f"fetching {s} .. {e} for {len(LATS) * len(LONS)} points")
        frames.append(to_frame(fetch(LATS, LONS, s, e)))
    df = pd.concat(frames, ignore_index=True).sort_values(["point_id", "local_timestamp"])
    df.to_csv(target, index=False)
    df.groupby("point_id")[["latitude", "longitude"]].first().reset_index().to_csv(out / "points.csv", index=False)
    print(f"wrote {target}: {len(df):,} rows, {df['local_timestamp'].min()} .. {df['local_timestamp'].max()}, "
          f"missing values: {int(df[VARIABLES].isna().sum().sum())}")


if __name__ == "__main__":
    main()

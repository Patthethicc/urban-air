"""Hourly PM2.5 of other Southeast Asian cities from the OpenAQ v3 API, written in the layout of the Manila data.

What it does (one function each, so every step can be run and checked on its own):
    discover_stations   the PM2.5 stations inside a city's bounding box (/v3/locations)
    plan                how many requests and how long the download will take, BEFORE it starts
    download_sensor     one sensor's hourly values for a date range (/v3/sensors/{id}/hours), saved to raw/ so a stopped run resumes
    station_quality     hours, span and completeness of one station: the rules that keep or drop it
    write_dataset       the kept stations as <out>/<country>/sensors/openaq_<location_id>_<name>.csv + locations.csv, the same columns as
                        data/Combined Dataset/<arm>/sensors (location_name, location_id, sensor_id, utc_timestamp, local_timestamp,
                        pm25_value, parameter, unit)

Needs an OpenAQ API key (free: https://explore.openaq.org/register), read from the environment variable OPENAQ_API_KEY, never from a file.
Free keys are limited to 60 requests per minute and 2,000 per hour; `RateLimiter` stays under both and waits when the API answers 429.
Nothing here uses weather, forecasts or anything but PM2.5 readings.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

API = "https://api.openaq.org/v3"
PM25_PARAMETER_ID = 2                      # OpenAQ's id of PM2.5; sensors are still checked by parameter NAME below
PAGE = 1000                                # results per request (the API default is 100; fewer, bigger pages = fewer requests)
COLUMNS = ["location_name", "location_id", "sensor_id", "utc_timestamp", "local_timestamp", "pm25_value", "parameter", "unit"]

# The cities. bbox = (min lon, min lat, max lon, max lat) in WGS 84; epsg = the UTM zone the city lies in (distances in metres for the spatial
# lag); tz = the local time zone of the readings. Add a city by adding a line.
# Manila's own area is the National Capital Region (NCR, about 640 km2), so the sources are the capital regions: Hanoi (the capital of
# Vietnam), Bangkok with its metropolitan ring, Jakarta with its Greater Jakarta neighbours, Kuala Lumpur with the Klang Valley. Ho Chi Minh
# City is not a capital but is the largest city of Vietnam and is fetched as well, so Vietnam has two sources.
CITIES = {
    "vietnam/ho_chi_minh_city": {"bbox": (106.35, 10.35, 107.05, 11.15), "epsg": 32648, "tz": "Asia/Ho_Chi_Minh"},
    "vietnam/hanoi":            {"bbox": (105.60, 20.80, 106.10, 21.30), "epsg": 32648, "tz": "Asia/Ho_Chi_Minh"},
    "thailand/bangkok":         {"bbox": (100.30, 13.50, 100.95, 14.00), "epsg": 32647, "tz": "Asia/Bangkok"},
    "indonesia/jakarta":        {"bbox": (106.60, -6.45, 107.05, -6.05), "epsg": 32748, "tz": "Asia/Jakarta"},
    "malaysia/kuala_lumpur":    {"bbox": (101.55, 2.95, 101.85, 3.30), "epsg": 32647, "tz": "Asia/Kuala_Lumpur"},
}


class RateLimiter:
    """Keeps a client under `per_minute` and `per_hour` requests (the free-key limits are 60 and 2,000; a margin is kept)."""

    def __init__(self, per_minute: int = 55, per_hour: int = 1900, clock=time.monotonic, sleep=time.sleep):
        self.per_minute, self.per_hour, self.clock, self.sleep = per_minute, per_hour, clock, sleep
        self.stamps: deque = deque()                       # the times of the requests of the last hour

    def wait(self) -> None:
        """Block until one more request is allowed, then count it."""
        while True:
            now = self.clock()
            while self.stamps and now - self.stamps[0] > 3600:
                self.stamps.popleft()
            last_minute = [t for t in self.stamps if now - t <= 60]
            if len(self.stamps) >= self.per_hour:
                self.sleep(max(1.0, 3600 - (now - self.stamps[0])))
            elif len(last_minute) >= self.per_minute:
                self.sleep(max(0.5, 60 - (now - last_minute[0])))
            else:
                self.stamps.append(now)
                return


class OpenAQTimeout(RuntimeError):
    """The server gave up on the query (HTTP 408 or 504). The remedy is a smaller query, so the caller splits the date window and tries again."""


class OpenAQ:
    """A small client: the API key from the environment, the rate limiter, retries on 429 and on server errors."""

    def __init__(self, api_key: str | None = None, session=None, limiter: RateLimiter | None = None, sleep=time.sleep, retries: int = 6):
        self.key = api_key or os.environ.get("OPENAQ_API_KEY", "")
        if not self.key:
            raise RuntimeError("OPENAQ_API_KEY is not set. Register for a free key at https://explore.openaq.org/register and set it in the "
                               "environment (PowerShell: $env:OPENAQ_API_KEY='...'); it is never read from a file or written anywhere.")
        if session is None:
            import requests
            session = requests.Session()
        self.session, self.limiter, self.sleep, self.retries = session, limiter or RateLimiter(), sleep, retries
        self.requests_made = 0

    def get(self, path: str, **params) -> dict:
        """GET /v3<path>; returns the parsed JSON. Waits and retries on 429 / 5xx; any other error stops with the API's message."""
        params = {k: v for k, v in params.items() if v is not None}
        last = ""
        for attempt in range(self.retries):
            self.limiter.wait()
            self.requests_made += 1
            r = self.session.get(f"{API}{path}", params=params, headers={"X-API-Key": self.key}, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (408, 504):                    # "Connection timed out: try a smaller time frame": the query was too big, waiting alone rarely helps
                if attempt >= 1:
                    raise OpenAQTimeout(f"OpenAQ {r.status_code} for {path}: {r.text[:200]}")
                self.sleep(5)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                hd = getattr(r, "headers", {}) or {}
                last = f"last status {r.status_code}, x-ratelimit-remaining={hd.get('x-ratelimit-remaining')}, reset in {hd.get('x-ratelimit-reset')} s"
                reset = r.headers.get("x-ratelimit-reset") if hasattr(r, "headers") else None
                self.sleep(min(float(reset), 3600) if reset and str(reset).replace(".", "", 1).isdigit() else min(2 ** attempt * 5, 300))
                continue
            raise RuntimeError(f"OpenAQ {r.status_code} for {path}: {r.text[:300]}")
        raise RuntimeError(f"OpenAQ still refusing {path} after {self.retries} tries ({last}). 429 = the rate limit (2,000 requests an hour on a free key: wait, "
                           "it clears by itself); 5xx = the server is struggling. A ban would be a 403, not this.")

    def pages(self, path: str, **params):
        """Every result of a paged endpoint, page after page, until the last page is short."""
        page = 1
        while True:
            body = self.get(path, limit=PAGE, page=page, **params)
            results = body.get("results", [])
            yield from results
            if len(results) < PAGE:
                return
            page += 1


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_") or "station"


def discover_stations(client: OpenAQ, city: str) -> pd.DataFrame:
    """PM2.5 stations of a city: one row per (location, PM2.5 sensor). Mobile units are left out (they have no fixed place)."""
    c = CITIES[city]
    bbox = ",".join(str(v) for v in c["bbox"])
    rows = []
    for loc in client.pages("/locations", bbox=bbox, parameters_id=PM25_PARAMETER_ID):
        if loc.get("isMobile"):
            continue
        for s in loc.get("sensors", []):
            p = s.get("parameter") or {}
            if str(p.get("name", "")).lower().replace(".", "") not in ("pm25",):      # 'pm25' (OpenAQ's name); a PM10 sensor of the same station is skipped
                continue
            first, last = (loc.get("datetimeFirst") or {}).get("utc"), (loc.get("datetimeLast") or {}).get("utc")
            rows.append({"city": city, "location_id": loc["id"], "location_name": loc.get("name") or f"location_{loc['id']}",
                         "latitude": loc["coordinates"]["latitude"], "longitude": loc["coordinates"]["longitude"], "sensor_id": s["id"],
                         "provider": (loc.get("provider") or {}).get("name"), "is_monitor": loc.get("isMonitor"),
                         "first_utc": first, "last_utc": last})
    return pd.DataFrame(rows, columns=["city", "location_id", "location_name", "latitude", "longitude", "sensor_id", "provider", "is_monitor",
                                       "first_utc", "last_utc"])


def plan(stations: pd.DataFrame, start: str, end: str, chunk_days: int = 60) -> dict:
    """What the download will cost: requests (at least one per window of `chunk_days` days per sensor, pages of PAGE hours) and the time at the free limit.
    A station whose own first/last readings do not overlap start..end costs nothing."""
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    requests_needed = 0
    for _, r in stations.iterrows():
        a = max(s, pd.Timestamp(r["first_utc"])) if r["first_utc"] else s
        b = min(e, pd.Timestamp(r["last_utc"])) if r["last_utc"] else e
        if b <= a:
            continue
        hours = (b - a).total_seconds() / 3600
        requests_needed += max(1, int(-(-hours // PAGE))) + int((b - a).days // chunk_days)        # pages + chunk boundaries
    hours_at_limit = requests_needed / 1900
    return {"stations": int(len(stations)), "requests": int(requests_needed), "hours_at_free_limit": round(hours_at_limit, 1),
            "note": "free keys: 60 requests a minute and 2,000 an hour; the download resumes where it stopped"}


def _windows(start: str, end: str, days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """start..end cut into consecutive windows of at most `days` days (UTC)."""
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    out, cur = [], s
    while cur < e:
        nxt = min(cur + pd.Timedelta(days=days), e)
        out.append((cur, nxt))
        cur = nxt
    return out


def _fetch_window(client: OpenAQ, sensor_id: int, a: pd.Timestamp, b: pd.Timestamp, parts: Path, min_days: float = 1.0) -> list[pd.DataFrame]:
    """One window of one sensor, cached as parts/<from>_<to>.csv (an existing part is not fetched again). If the server times out, the window is
    halved and each half is tried again, down to `min_days`; only a window that cannot be made smaller raises."""
    fmt = "%Y%m%dT%H%M"
    cache = parts / f"{a.strftime(fmt)}_{b.strftime(fmt)}.csv"
    if cache.exists():
        return [pd.read_csv(cache)]
    rows = []
    try:
        for item in client.pages(f"/sensors/{sensor_id}/hours", datetime_from=a.strftime("%Y-%m-%dT%H:%M:%SZ"), datetime_to=b.strftime("%Y-%m-%dT%H:%M:%SZ")):
            if item.get("value") is None:
                continue
            period = item.get("period") or {}
            frm, to = period.get("datetimeFrom") or {}, period.get("datetimeTo") or {}
            rows.append({"utc_from": frm.get("utc"), "utc_to": to.get("utc"), "local_from": frm.get("local"), "value": item["value"],
                         "unit": (item.get("parameter") or {}).get("units", "µg/m³"),
                         "percent_coverage": (item.get("coverage") or {}).get("percentCoverage")})
    except OpenAQTimeout:
        if (b - a) <= pd.Timedelta(days=min_days):
            raise
        mid = a + (b - a) / 2
        mid = mid.floor("h")
        return _fetch_window(client, sensor_id, a, mid, parts, min_days) + _fetch_window(client, sensor_id, mid, b, parts, min_days)
    df = pd.DataFrame(rows, columns=["utc_from", "utc_to", "local_from", "value", "unit", "percent_coverage"])
    tmp = cache.with_suffix(".part")
    df.to_csv(tmp, index=False)
    tmp.replace(cache)                                     # a window is written whole or not at all
    return [df]


def download_sensor(client: OpenAQ, sensor_id: int, start: str, end: str, raw_dir: Path, force: bool = False, window_days: int = 60,
                    first: str | None = None, last: str | None = None) -> Path:
    """One sensor's hourly values for start..end into raw_dir/<sensor_id>.csv (utc_from, utc_to, local_from, value, unit, percent_coverage).
    The range is fetched in windows of `window_days` days (a long query makes the server time out with HTTP 408), every window is saved on its own
    under raw_dir/<sensor_id>.parts/, and a window that still times out is halved. So a stopped or failed run resumes at the window it was on, not
    at the start of the sensor. `first`/`last` (the station's own first and last reading) trim the range so nothing outside them is requested.
    A finished <sensor_id>.csv is kept; force=True fetches everything again."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{sensor_id}.csv"
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    if first is not None and pd.notna(first):
        s = max(s, pd.Timestamp(first).floor("D"))
    if last is not None and pd.notna(last):
        e = min(e, pd.Timestamp(last).ceil("D"))
    old = None
    if path.exists() and not force:
        # A finished file is kept. If END was moved later since it was written and the station has readings after the file's last hour, only
        # that gap is fetched and appended (this is how a download is extended to a newer date without starting over).
        if last is None:
            return path
        old = pd.read_csv(path)
        have = pd.to_datetime(old["utc_from"], utc=True).max() if len(old) else None
        if have is None or have + pd.Timedelta(days=1) >= e:
            return path
        s = max(s, have.floor("D"))
    parts = raw_dir / f"{sensor_id}.parts"
    if force and parts.exists():
        for f in parts.glob("*.csv"):
            f.unlink()
    parts.mkdir(exist_ok=True)
    frames = []
    for a, b in _windows(s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"), window_days) if s < e else []:
        frames += _fetch_window(client, sensor_id, a, b, parts)
    cols = ["utc_from", "utc_to", "local_from", "value", "unit", "percent_coverage"]
    frames = [f for f in frames if len(f)]
    if old is not None:
        frames = [old] + frames
    merged = pd.concat(frames, ignore_index=True).drop_duplicates("utc_from") if frames else pd.DataFrame(columns=cols)
    tmp = path.with_suffix(".part")
    merged.to_csv(tmp, index=False)
    tmp.replace(path)                                      # written whole or not at all: a stopped run never leaves a half file that looks finished
    for f in parts.glob("*"):
        f.unlink()
    parts.rmdir()
    return path


def station_quality(df: pd.DataFrame, start: str, end: str) -> dict:
    """hours kept, first and last reading, the share of the hours between them that have a value, and the months with data."""
    if df.empty:
        return {"hours": 0, "first": None, "last": None, "completeness": 0.0, "months_with_data": 0, "span_days": 0}
    t = pd.to_datetime(df["utc_timestamp"], utc=True)
    span_hours = max(1, int((t.max() - t.min()).total_seconds() // 3600) + 1)
    return {"hours": int(t.nunique()), "first": str(t.min()), "last": str(t.max()), "completeness": round(t.nunique() / span_hours, 3),
            "months_with_data": int(t.dt.strftime("%Y-%m").nunique()), "span_days": int((t.max() - t.min()).days)}


def to_manila_layout(raw: pd.DataFrame, station: pd.Series, tz: str, time_label: str = "from") -> pd.DataFrame:
    """OpenAQ rows -> the columns of the Manila sensor files. time_label 'from' labels an hour by its START (the default, matching what the
    Manila files appear to do: check one value against the OpenAQ portal before mixing the two), 'to' by its END."""
    col = "utc_from" if time_label == "from" else "utc_to"
    utc = pd.to_datetime(raw[col], utc=True)
    local = utc.dt.tz_convert(ZoneInfo(tz))
    out = pd.DataFrame({
        "location_name": station["location_name"], "location_id": station["location_id"], "sensor_id": station["sensor_id"],
        "utc_timestamp": utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "local_timestamp": local.apply(lambda x: x.isoformat()),
        "pm25_value": raw["value"], "parameter": "PM2.5", "unit": raw["unit"].fillna("µg/m³")})
    return out.drop_duplicates("utc_timestamp").sort_values("utc_timestamp")[COLUMNS]


def write_dataset(stations: pd.DataFrame, raw_dir: Path, out_dir: Path, start: str, end: str, min_hours: int = 6000, min_months: int = 12,
                  min_completeness: float = 0.5, time_label: str = "from") -> pd.DataFrame:
    """Write the kept stations of ONE city under out_dir (= <data>/Other Countries/<country>/<city>): sensors/*.csv, locations.csv and
    station_quality.csv (every station, kept or not, with the reason). Rules: at least `min_hours` hours, `min_months` different calendar
    months with data, and `min_completeness` of the hours between its first and last reading present. Returns the quality table."""
    out_dir, raw_dir = Path(out_dir), Path(raw_dir)
    (out_dir / "sensors").mkdir(parents=True, exist_ok=True)
    quality, kept = [], []
    for _, st in stations.drop_duplicates("sensor_id").iterrows():
        f = raw_dir / f"{st['sensor_id']}.csv"
        if not f.exists():
            quality.append({**st.to_dict(), "kept": False, "reason": "not downloaded"})
            continue
        raw = pd.read_csv(f)
        if raw.empty:
            quality.append({**st.to_dict(), **station_quality(pd.DataFrame(), start, end), "kept": False, "reason": "no values in the range"})
            continue
        df = to_manila_layout(raw, st, CITIES[st["city"]]["tz"], time_label)
        q = station_quality(df, start, end)
        reason = ("hours %d < %d" % (q["hours"], min_hours) if q["hours"] < min_hours else
                  "months %d < %d" % (q["months_with_data"], min_months) if q["months_with_data"] < min_months else
                  "completeness %.2f < %.2f" % (q["completeness"], min_completeness) if q["completeness"] < min_completeness else "")
        quality.append({**st.to_dict(), **q, "kept": not reason, "reason": reason or "kept"})
        if not reason:
            df.to_csv(out_dir / "sensors" / f"openaq_{st['location_id']}_{_slug(st['location_name'])}.csv", index=False, encoding="utf-8")
            kept.append(st)
    table = pd.DataFrame(quality)
    table.to_csv(out_dir / "station_quality.csv", index=False)
    if kept:
        loc = pd.DataFrame(kept).drop_duplicates("location_id")[["location_name", "location_id", "latitude", "longitude"]]
        loc.columns = ["name", "location_id", "latitude", "longitude"]
        loc.to_csv(out_dir / "locations.csv", index=False)
    (out_dir / "source.json").write_text(json.dumps({
        "source": "OpenAQ API v3 (https://api.openaq.org/v3), /locations and /sensors/{id}/hours", "city": str(stations["city"].iloc[0]) if len(stations) else None,
        "range": [start, end], "time_label": time_label, "epsg": CITIES[str(stations["city"].iloc[0])]["epsg"] if len(stations) else None,
        "timezone": CITIES[str(stations["city"].iloc[0])]["tz"] if len(stations) else None, "rules": {"min_hours": min_hours, "min_months": min_months, "min_completeness": min_completeness},
        "stations_kept": len(kept), "note": "values are the providers' data as served by OpenAQ; check each provider's licence before redistributing"},
        indent=2), encoding="utf-8")
    return table

"""The OpenAQ scraper against a fake API (no network, no key needed). Run: python -m pytest other-country-scrapers/tests -q"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openaq_scraper as oa  # noqa: E402


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._body, self.headers, self.text = status, body or {}, headers or {}, str(body)[:100]

    def json(self):
        return self._body


class FakeSession:
    """Answers from a list of canned responses, in order, and remembers the requests."""

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return self.responses.pop(0)


def _client(responses, **kw):
    return oa.OpenAQ(api_key="test-key", session=FakeSession(responses), limiter=oa.RateLimiter(per_minute=10_000, per_hour=10_000),
                     sleep=lambda s: None, **kw)


def _loc(i, name="Station", mobile=False, params=("pm25",)):
    return {"id": i, "name": name, "isMobile": mobile, "coordinates": {"latitude": 10.8, "longitude": 106.7},
            "provider": {"name": "P"}, "isMonitor": True, "datetimeFirst": {"utc": "2023-01-01T00:00:00Z"}, "datetimeLast": {"utc": "2026-01-01T00:00:00Z"},
            "sensors": [{"id": i * 10 + k, "name": p, "parameter": {"id": 2 if p == "pm25" else 1, "name": p}} for k, p in enumerate(params)]}


def test_a_missing_key_is_refused_with_a_message_and_the_key_goes_in_a_header_only(monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAQ_API_KEY"):
        oa.OpenAQ(session=FakeSession([]))
    c = _client([Resp(body={"results": []})])
    c.get("/locations", bbox="1,2,3,4")
    url, params, headers = c.session.calls[0]
    assert headers == {"X-API-Key": "test-key"} and "key" not in " ".join(map(str, params.values())).lower()


def test_discovery_keeps_pm25_sensors_of_fixed_stations_only():
    body = {"results": [_loc(1), _loc(2, mobile=True), _loc(3, params=("pm10", "pm25")), _loc(4, params=("o3",))]}
    st = oa.discover_stations(_client([Resp(body=body)]), "vietnam/ho_chi_minh_city")
    assert sorted(st["location_id"]) == [1, 3]                              # the mobile unit and the ozone-only station are out
    assert st.loc[st.location_id == 3, "sensor_id"].iloc[0] == 31           # the PM2.5 sensor, not the PM10 one of the same station


def test_paging_goes_on_until_a_short_page(monkeypatch):
    monkeypatch.setattr(oa, "PAGE", 2)
    pages = [Resp(body={"results": [{"a": 1}, {"a": 2}]}), Resp(body={"results": [{"a": 3}]})]
    c = _client(pages)
    assert [r["a"] for r in c.pages("/x")] == [1, 2, 3]
    assert [call[1]["page"] for call in c.session.calls] == [1, 2]


def test_a_429_is_waited_out_and_retried_and_other_errors_stop():
    slept = []
    c = oa.OpenAQ(api_key="k", session=FakeSession([Resp(429, headers={"x-ratelimit-reset": "7"}), Resp(body={"results": [1]})]),
                  limiter=oa.RateLimiter(per_minute=100, per_hour=100), sleep=slept.append)
    assert c.get("/x")["results"] == [1] and slept == [7.0]
    with pytest.raises(RuntimeError, match="401"):
        _client([Resp(401, body={"detail": "bad key"})]).get("/x")


def test_the_rate_limiter_stays_under_the_minute_limit():
    t, slept = [0.0], []
    lim = oa.RateLimiter(per_minute=3, per_hour=1000, clock=lambda: t[0], sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s)))
    for _ in range(4):
        lim.wait()
    assert slept and slept[0] >= 0.5                                         # the 4th request in a minute had to wait


def _hours(values, start="2024-01-01T00:00:00Z"):
    ts = pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    return [{"value": v, "parameter": {"units": "µg/m³"}, "coverage": {"percentCoverage": 100},
             "period": {"datetimeFrom": {"utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "local": t.tz_convert("Asia/Ho_Chi_Minh").isoformat()},
                        "datetimeTo": {"utc": (t + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")}}} for t, v in zip(ts, values)]


def test_a_sensor_is_downloaded_once_and_nulls_are_dropped(tmp_path):
    c = _client([Resp(body={"results": _hours([10.0, None, 12.0])})])
    path = oa.download_sensor(c, 31, "2024-01-01", "2024-02-01", tmp_path)
    df = pd.read_csv(path)
    assert list(df["value"]) == [10.0, 12.0] and not list(tmp_path.glob("*.part"))
    again = oa.download_sensor(_client([]), 31, "2024-01-01", "2024-02-01", tmp_path)      # an existing file: no request at all
    assert again == path


def test_the_manila_layout_and_the_local_time():
    raw = pd.DataFrame({"utc_from": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"], "utc_to": ["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"],
                        "value": [10.0, 20.0], "unit": ["µg/m³", "µg/m³"]})
    st = pd.Series({"location_name": "A b", "location_id": 5, "sensor_id": 50})
    out = oa.to_manila_layout(raw, st, "Asia/Ho_Chi_Minh")
    assert list(out.columns) == oa.COLUMNS
    assert out["utc_timestamp"].iloc[0] == "2024-01-01T00:00:00Z" and out["local_timestamp"].iloc[0] == "2024-01-01T07:00:00+07:00"
    assert oa.to_manila_layout(raw, st, "Asia/Ho_Chi_Minh", "to")["utc_timestamp"].iloc[0] == "2024-01-01T01:00:00Z"


def test_the_quality_rules_keep_a_full_station_and_say_why_others_are_dropped(tmp_path):
    raw_dir, out = tmp_path / "raw", tmp_path / "out"
    raw_dir.mkdir()
    full = pd.DataFrame(_hours([10.0] * 24 * 400))                                   # 400 days, no gaps
    short = pd.DataFrame(_hours([10.0] * 100))
    for sid, rows in ((11, full), (12, short)):
        pd.DataFrame([{"utc_from": r["period"]["datetimeFrom"]["utc"], "utc_to": r["period"]["datetimeTo"]["utc"],
                       "local_from": r["period"]["datetimeFrom"]["local"], "value": r["value"], "unit": "µg/m³", "percent_coverage": 100}
                      for r in rows.to_dict("records")]).to_csv(raw_dir / f"{sid}.csv", index=False)
    stations = pd.DataFrame([{"city": "vietnam/hanoi", "location_id": i, "location_name": f"S {i}", "latitude": 21.0, "longitude": 105.8,
                              "sensor_id": i, "provider": "P", "is_monitor": True, "first_utc": None, "last_utc": None} for i in (1, 2)]
                            ).assign(sensor_id=[11, 12])
    table = oa.write_dataset(stations, raw_dir, out, "2024-01-01", "2025-06-01")
    assert dict(zip(table["sensor_id"], table["kept"])) == {11: True, 12: False}
    assert "hours 100" in table.loc[table.sensor_id == 12, "reason"].iloc[0]
    assert (out / "sensors" / "openaq_1_S_1.csv").exists() and not (out / "sensors" / "openaq_2_S_2.csv").exists()
    loc = pd.read_csv(out / "locations.csv")
    assert list(loc.columns) == ["name", "location_id", "latitude", "longitude"] and list(loc["location_id"]) == [1]
    assert list(pd.read_csv(out / "sensors" / "openaq_1_S_1.csv").columns) == oa.COLUMNS


def test_the_plan_counts_requests_and_ignores_stations_outside_the_range():
    st = pd.DataFrame([{"first_utc": "2023-01-01T00:00:00Z", "last_utc": "2026-01-01T00:00:00Z"},
                       {"first_utc": "2019-01-01T00:00:00Z", "last_utc": "2020-01-01T00:00:00Z"}])
    p = oa.plan(st, "2023-01-01", "2026-01-01")
    assert p["stations"] == 2 and 20 < p["requests"] < 60 and p["hours_at_free_limit"] < 1


def test_a_408_splits_the_window_and_a_stopped_run_resumes_at_its_window(tmp_path):
    # window 1 (60 days) times out twice -> halved; the two halves answer; then the second 60-day window answers
    ok = lambda start: Resp(body={"results": _hours([1.0, 2.0], start)})          # noqa: E731
    timeout = Resp(status=408, body={"detail": "Connection timed out"})
    c = _client([timeout, timeout, ok("2024-01-01"), ok("2024-01-31"), ok("2024-03-01")])
    path = oa.download_sensor(c, 7, "2024-01-01", "2024-04-30", tmp_path, window_days=60)
    assert len(pd.read_csv(path)) == 6 and not (tmp_path / "7.parts").exists()
    # a failure mid-way leaves the finished windows on disk, and the next run fetches only what is missing
    bad = [ok("2024-01-01"), Resp(status=408), Resp(status=408)] + [Resp(status=408)] * 40
    c2 = _client(bad)
    with pytest.raises(oa.OpenAQTimeout):
        oa.download_sensor(c2, 8, "2024-01-01", "2024-04-30", tmp_path, window_days=60)
    assert len(list((tmp_path / "8.parts").glob("*.csv"))) == 1
    c3 = _client([ok("2024-03-01")])
    oa.download_sensor(c3, 8, "2024-01-01", "2024-04-30", tmp_path, window_days=60)
    assert len(c3.session.calls) == 1 and (tmp_path / "8.csv").exists()


def test_moving_end_later_fetches_only_the_gap_and_appends(tmp_path):
    first, last = "2024-01-01T00:00:00Z", "2024-03-01T00:00:00Z"
    c = _client([Resp(body={"results": _hours([1.0] * 24 * 10, "2024-01-01")})])
    path = oa.download_sensor(c, 9, "2024-01-01", "2024-01-11", tmp_path, first=first, last=last)
    assert len(pd.read_csv(path)) == 240
    c2 = _client([Resp(body={"results": _hours([2.0] * 24 * 5, "2024-01-10")})])          # END moved to 2024-03-01: only from the last day on
    oa.download_sensor(c2, 9, "2024-01-01", "2024-03-01", tmp_path, first=first, last=last)
    df = pd.read_csv(path)
    assert len(c2.session.calls) >= 1 and "2024-01-10" in c2.session.calls[0][1]["datetime_from"] and len(df) == 240 + 5 * 24 - 24
    c3 = _client([])                                                                       # already up to date: no request
    oa.download_sensor(c3, 9, "2024-01-01", "2024-01-11", tmp_path, first=first, last="2024-01-11T00:00:00Z")
    assert not c3.session.calls

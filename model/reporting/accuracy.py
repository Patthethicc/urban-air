"""Accuracy of the point predictions, organised so the numbers can be read (Validation.txt: RMSE, MAE, R2 on the
spatially withheld stations). Adds what makes those numbers interpretable: bias, shares within a tolerance, AQI-band
agreement, and yardsticks (repeat the previous hour; always predict the average level).
"""
import numpy as np
import pandas as pd

from training import metrics as M

DEFAULTS = {"tolerances_ugm3": [5, 10], "aqi_bands": True, "persistence": True}
QC_ID_BASE = 9_000_000                     # location_id of the Quezon City sites (preprocessing/qc_convert.py)


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg["report"].get("accuracy") or {})}


def aqi_uppers(cfg: dict) -> list:
    bins = cfg["output"]["aqi"]["bins"]
    return [b["upper"] for b in bins[:-1]]


def tolerance_key(tol) -> str:
    return f"within_{float(tol):g}"


def extras(y_true, y_pred, cfg: dict) -> dict:
    """Bias, shares within each tolerance and AQI-band agreement."""
    s = settings(cfg)
    out = {"bias": M.bias(y_true, y_pred)}
    for tol in s["tolerances_ugm3"]:
        out[tolerance_key(tol)] = M.fraction_within(y_true, y_pred, tol)
    if s["aqi_bands"]:
        a = M.band_agreement(y_true, y_pred, aqi_uppers(cfg))
        out.update(aqi_exact=a["exact"], aqi_within_one=a["within_one"], aqi_majority_baseline=a["majority_baseline"],
                   aqi_majority_band=cfg["output"]["aqi"]["bins"][a["majority_band"]]["label"])
    return out


def measures(df: pd.DataFrame, cfg: dict) -> dict:
    """Everything reported for one set of predictions (columns y_true, y_pred, location_id)."""
    m = M.compute_all(df["y_true"], df["y_pred"])
    keep = list(cfg["report"]["metrics"]) + [k for k in cfg["report"].get("extra_metrics", []) if k in m]
    return {"rows": int(len(df)), "stations": int(df["location_id"].nunique()),
            **{k: m[k] for k in keep}, **extras(df["y_true"], df["y_pred"], cfg)}


def persistence_reference(df: pd.DataFrame, cfg: dict) -> dict | None:
    """Yardstick: predict each hour with the previous hour's observation at the same station. Scored on the rows
    where that previous hour is in the table, together with the model on exactly the same rows."""
    if not settings(cfg)["persistence"] or not {"location_id", "timestamp"} <= set(df.columns):
        return None
    step = pd.Timedelta(cfg["data"]["time"]["freq"])
    d = df.sort_values(["location_id", "timestamp"]).copy()
    d["previous"] = d.groupby("location_id")["y_true"].shift(1)
    d["gap"] = d.groupby("location_id")["timestamp"].diff()
    d = d[(d["gap"] == step) & d["previous"].notna()]
    if len(d) < 10:
        return None
    model = M.compute_all(d["y_true"], d["y_pred"])
    naive = M.compute_all(d["y_true"], d["previous"])
    return {"rows": int(len(d)), "share_of_test_rows": float(len(d) / len(df)),
            "model": {k: model[k] for k in ("rmse", "mae", "r2")},
            "persistence": {k: naive[k] for k in ("rmse", "mae", "r2")},
            "mean_predictor": {"rmse": float(np.std(d["y_true"])), "mae": float(np.mean(np.abs(d["y_true"] - d["y_true"].mean()))),
                               "r2": 0.0},
            "skill_vs_persistence": float(1.0 - model["rmse"] ** 2 / naive["rmse"] ** 2)}


def station_names(cfg: dict) -> dict:
    """location_id -> name from the configured sensor location tables (best effort)."""
    try:
        from preprocessing.coverage import all_sensor_locations
        t = all_sensor_locations(cfg)
        return {int(i): str(n) for i, n in zip(t[cfg["data"]["columns"]["group"]], t["name"])}
    except Exception:
        return {}


def by_station(df: pd.DataFrame, cfg: dict, geocp_stations: list[dict] | None = None) -> list[dict]:
    """One row per station: accuracy and, when GeoCP ran, the interval coverage. Worst RMSE first."""
    names = station_names(cfg)
    cov = {int(r["location_id"]): r for r in (geocp_stations or [])}
    rows = []
    for sid, g in df.groupby("location_id"):
        sid = int(sid)
        row = {"location_id": sid, "name": names.get(sid, ""), "network": "QC" if sid >= QC_ID_BASE else "OpenAQ",
               **measures(g, cfg)}
        row.pop("stations")
        if sid in cov:
            row["geocp_coverage"] = float(cov[sid]["coverage"])
            row["distance_to_nearest_calibration_m"] = float(cov[sid].get("distance_to_nearest_calibration_m", np.nan))
        rows.append(row)
    return sorted(rows, key=lambda r: -r["rmse"])

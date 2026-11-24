"""Which parts of the Amouei (2023) framework improve the model? One experiment per switch, same data, same test stations.

    cd amouei-model
    python -m experiments.run_ablation --list
    python -m experiments.run_ablation --quick                          # local screen: no CV, every 4th row, 6 epochs
    python -m experiments.run_ablation --exp baseline weather year      # a subset
    python -m experiments.run_ablation --seeds 42 43                     # repeat with other seeds (noise level)
    python -m experiments.run_ablation --compare-only                    # rebuild the table from finished runs

Every experiment runs `preprocess` (shared by experiments with the same data settings) and `train` through
run_pipeline.py, then experiments/predict_test.py (the test predictions, no GeoCP), in its own folders under
runs/<ARM>/ablation/. The comparison (experiments/results/) is computed on the test (station, hour) rows that EVERY
experiment kept, because a 24-hour window or a weather lead drops different samples than the baseline; R2 on a
different set of rows would not be comparable. The gain over the baseline gets a 95 % interval from a bootstrap over
the test STATIONS (rows of one station are not independent).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent                                   # amouei-model/
REPO = MODEL.parent

AMOUEI_LSTM = {"model.temporal_lstm.layer_sizes": [256, 128, 64, 32], "model.temporal_lstm.dropout": 0.0,
               "model.fusion_head.hidden_dims": [128], "model.fusion_head.dropout": 0.5}
WINDOW24 = {"data.temporal.lags": list(range(24, 0, -1))}
WEATHER = {"data.temporal.weather.enabled": True}

# name -> (what it tests, overrides). Every experiment is LSTM-only unless "cnn" is in the overrides (the CNN branch added
# nothing in the main model's runs, and it is the slow part); "model.spatial_cnn.enabled": True switches it on.
EXPERIMENTS = {
    "baseline":           ("the main model (3 h window, LSTM 2 x 114, PM2.5 + spatial lag + hour)", {}),
    "year":               ("+ time of year (sin/cos day of year)", {"data.temporal.cyclical_year": True}),
    "weather":            ("+ weather at the lag hours (temperature, humidity, wind vector)", WEATHER),
    "weather_lead1":      ("+ weather incl. the target hour (each step carries the next hour's weather)",
                           {**WEATHER, "data.temporal.weather.lead_hours": 1}),
    "road_density":       ("+ traffic proxy (street share of the 320 m patch)", {"data.temporal.static_features": ["road_density"]}),
    "lcz_proxy":          ("+ LCZ-like class of the station (one-hot)", {"data.temporal.static_features": ["lcz_proxy"]}),
    "morphology_tabular": ("+ morphology band means as LSTM inputs (instead of a CNN)", {"data.temporal.static_features": ["morphology"]}),
    "window24":           ("24 h input window instead of 3 h", WINDOW24),
    "amouei_lstm":        ("Amouei network: LSTM 256-128-64-32 + dense 128, dropout 0.5 (3 h window)", AMOUEI_LSTM),
    "amouei_lstm_w24":    ("Amouei network with the 24 h window", {**AMOUEI_LSTM, **WINDOW24}),
    "amouei_full":        ("everything of Amouei: network, 24 h, year, weather, traffic, LCZ",
                           {**AMOUEI_LSTM, **WINDOW24, **WEATHER, "data.temporal.cyclical_year": True,
                            "data.temporal.static_features": ["road_density", "lcz_proxy"]}),
    "window12":           ("12 h input window (between the main model's 3 h and Amouei's 24 h)", {"data.temporal.lags": list(range(12, 0, -1))}),
    "window48":           ("48 h input window (longer than Amouei's)", {"data.temporal.lags": list(range(48, 0, -1))}),
    "recommended":        ("the elements that helped, together: 24 h window + weather incl. the target hour + time of year",
                           {**WINDOW24, **WEATHER, "data.temporal.weather.lead_hours": 1, "data.temporal.cyclical_year": True}),
    "cnn_baseline":       ("the main CNN-LSTM (for reference)", {"model.spatial_cnn.enabled": True}),
    "cnn_weather_lead1":  ("CNN-LSTM + weather incl. the target hour", {"model.spatial_cnn.enabled": True, **WEATHER,
                                                                         "data.temporal.weather.lead_hours": 1}),
}
DEFAULT = ["baseline", "year", "weather", "weather_lead1", "road_density", "lcz_proxy", "morphology_tabular",
           "window24", "amouei_lstm", "amouei_lstm_w24", "amouei_full", "recommended"]
QUICK = {"train.cv.enabled": False, "train.final_epochs": 6, "train.row_stride": 4}


def ov(d: dict) -> list[str]:
    return [x for k, v in d.items() for x in ("--override", f"{k}={json.dumps(v) if not isinstance(v, str) else v}")]


def data_key(overrides: dict) -> str:
    """Experiments with the same data.* settings share one preprocessing."""
    data = {k: v for k, v in sorted(overrides.items()) if k.startswith("data.")}
    if not data:
        return "processed_base"
    import hashlib
    return "processed_" + hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:10]


def common(args, name: str, overrides: dict, seed: int) -> dict:
    root = f"runs/{args.arm}/ablation"
    tag = name if seed == 42 else f"{name}_s{seed}"
    out = {
        "data.paths.master_raster": f"../data/Morphology/{args.raster}",
        "data.paths.sensor_locations_csv": f"../data/Combined Dataset/{args.arm}/locations.csv",
        "data.paths.sensor_csv_dir": f"../data/Combined Dataset/{args.arm}/sensors",
        "data.split.cv_assignment": "by_station", "geocp.enabled": False, "model.spatial_cnn.enabled": False,
        "train.seed": seed,
        "data.paths.processed_dir": f"{root}/{data_key(overrides)}",
        "data.paths.checkpoints_dir": f"{root}/checkpoints_{tag}",
        "data.paths.snapshots_dir": f"{root}/snapshots_{tag}",
    }
    if args.pin:
        part = json.loads((MODEL / args.pin).read_text())
        out["data.split.test_stations"] = sorted(part["test"])
        out["data.split.calibration_stations"] = sorted(part["calibration"])
    if args.quick:
        out.update(QUICK)
    out.update(overrides)
    for kv in args.override or []:
        k, _, v = kv.partition("=")
        out[k] = v
    return out


def run_stage(stage: str, settings: dict, log) -> None:
    cmd = ([sys.executable, "-u", "-m", "experiments.predict_test", *ov(settings)] if stage == "predict"
           else [sys.executable, "-u", "run_pipeline.py", "--stage", stage, *ov(settings)])
    log.write("$ " + " ".join(cmd) + "\n")
    log.flush()
    t = time.time()
    code = subprocess.run(cmd, cwd=MODEL, stdout=log, stderr=subprocess.STDOUT).returncode
    if code != 0:
        raise RuntimeError(f"{stage} failed (exit {code}); see {log.name}")
    print(f"    {stage}: {time.time() - t:.0f} s", flush=True)


def finished_run(snap_dir: Path) -> Path | None:
    runs = sorted(p for p in snap_dir.glob("*/data/predictions_test.csv")) if snap_dir.is_dir() else []
    return runs[-1] if runs else None


def run_experiment(args, name: str, seed: int) -> None:
    desc, overrides = EXPERIMENTS[name]
    s = common(args, name, overrides, seed)
    if finished_run(MODEL / s["data.paths.snapshots_dir"]) and not args.redo:
        print(f"[{name} seed {seed}] finished before, skipped")
        return
    print(f"[{name} seed {seed}] {desc}", flush=True)
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    with open(logs / f"{name}_s{seed}.log", "w", encoding="utf-8") as log:
        if not (MODEL / s["data.paths.processed_dir"] / "scalers.json").is_file() or args.redo_preprocess:
            run_stage("preprocess", s, log)
        run_stage("train", s, log)
        run_stage("predict", s, log)


# ------------------------------------------------------------------------------------------------------------ comparison
def _metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    out = {"r2": 1 - ((y - p) ** 2).sum() / ss, "rmse": float(np.sqrt(((y - p) ** 2).mean())), "mae": float(np.abs(y - p).mean())}
    hi = y > 30
    out["rmse_above_30"] = float(np.sqrt(((y[hi] - p[hi]) ** 2).mean())) if hi.any() else np.nan
    return out


def _station_sums(df: pd.DataFrame) -> pd.DataFrame:
    """Per test station: n, sum y, sum y^2, sum of squared errors (pooled R2 of any station sample follows from these)."""
    e2 = (df["y_true"] - df["y_pred"]) ** 2
    return pd.DataFrame({"n": 1.0, "sy": df["y_true"], "syy": df["y_true"] ** 2, "sse": e2,
                         "location_id": df["location_id"]}).groupby("location_id").sum()


def _station_bootstrap(base: pd.DataFrame, other: pd.DataFrame, n: int = 2000, seed: int = 0):
    """95 % interval of R2(other) - R2(base) over resampled test stations (same rows in both)."""
    b, o = _station_sums(base), _station_sums(other).loc[_station_sums(base).index]
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(b), np.full(len(b), 1 / len(b)), size=n).astype(float)      # [n, stations] resample counts
    N, SY, SYY = w @ b["n"].to_numpy(), w @ b["sy"].to_numpy(), w @ b["syy"].to_numpy()
    sst = SYY - SY ** 2 / N
    diffs = (w @ b["sse"].to_numpy() - w @ o["sse"].to_numpy()) / sst
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def compare(args) -> pd.DataFrame | None:
    root = MODEL / "runs" / args.arm / "ablation"
    preds = {}
    for name in EXPERIMENTS:
        for seed in args.seeds:
            tag = name if seed == 42 else f"{name}_s{seed}"
            f = finished_run(root / f"snapshots_{tag}")
            if f is not None:
                preds[(name, seed)] = pd.read_csv(f, parse_dates=["timestamp"])
    if not any(n == "baseline" for n, _ in preds):
        print("no finished baseline yet: nothing to compare")
        return None
    keys = None
    for df in preds.values():
        k = set(zip(df["location_id"], df["timestamp"]))
        keys = k if keys is None else keys & k
    keyidx = pd.MultiIndex.from_tuples(sorted(keys), names=["location_id", "timestamp"])

    def on_common(df):
        return df.set_index(["location_id", "timestamp"]).loc[keyidx].reset_index()

    bases = {s: on_common(preds[("baseline", s)]) for s in args.seeds if ("baseline", s) in preds}
    base = next(iter(bases.values()))
    rows = []
    for (name, seed), df in preds.items():
        if seed not in bases:                      # gains are measured against the baseline of the SAME seed
            continue
        b = bases[seed]
        c = on_common(df)
        m = _metrics(c["y_true"], c["y_pred"])
        own = _metrics(df["y_true"], df["y_pred"])
        run_info = sorted((root / f"checkpoints_{name if seed == 42 else f'{name}_s{seed}'}").glob("*/run_info.json"))
        params = json.loads(run_info[-1].read_text(encoding="utf-8")).get("parameters", {}) if run_info else {}
        lo, hi = (np.nan, np.nan) if name == "baseline" else _station_bootstrap(b, c, n=args.bootstrap)
        rows.append({"experiment": name, "seed": seed, "what": EXPERIMENTS[name][0],
                     "test_r2": m["r2"], "r2_gain": m["r2"] - _metrics(b["y_true"], b["y_pred"])["r2"],
                     "gain_ci_low": lo, "gain_ci_high": hi, "test_rmse": m["rmse"], "test_mae": m["mae"],
                     "rmse_above_30": m["rmse_above_30"], "own_rows": len(df), "own_r2": own["r2"],
                     "parameters": params.get("total") if isinstance(params, dict) else None})
    table = pd.DataFrame(rows).sort_values(["seed", "test_r2"], ascending=[True, False])
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stem = "ablation_quick" if args.quick else "ablation"
    table.round(4).to_csv(out / f"{stem}.csv", index=False)
    summary = {"common_test_rows": len(keyidx), "test_stations": int(base["location_id"].nunique()),
               "quick": bool(args.quick), "seeds": args.seeds, "persistence_r2": _persistence(base)}
    (out / f"{stem}_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    agg = (table.groupby("experiment").agg(seeds=("seed", "count"), test_r2=("test_r2", "mean"), r2_gain=("r2_gain", "mean"),
                                           gain_min=("r2_gain", "min"), gain_max=("r2_gain", "max"),
                                           ci_low_min=("gain_ci_low", "min"), ci_high_max=("gain_ci_high", "max"),
                                           ci_low_max=("gain_ci_low", "max"), rmse_above_30=("rmse_above_30", "mean"),
                                           parameters=("parameters", "first"))
           .sort_values("r2_gain", ascending=False))
    agg["verdict"] = np.where(agg.index == "baseline", "",
                     np.where(agg["ci_low_min"] > 0, "helps (every seed's interval above 0)",
                     np.where(agg["ci_high_max"] < 0, "hurts", "no clear effect")))
    agg.round(4).to_csv(out / f"{stem}_by_experiment.csv")
    print(agg.round(4).to_string())
    pd.set_option("display.width", 220)
    print(table.drop(columns=["what"]).round(4).to_string(index=False))
    print(json.dumps(summary))
    return table


def _persistence(base: pd.DataFrame) -> float | None:
    """R2 of repeating the previous hour, on the common rows that have a measured previous hour."""
    b = base.sort_values(["location_id", "timestamp"]).copy()
    prev = b.groupby("location_id")["y_true"].shift(1)
    gap = b.groupby("location_id")["timestamp"].diff() == pd.Timedelta("1h")
    ok = prev.notna() & gap
    return float(_metrics(b.loc[ok, "y_true"], prev[ok])["r2"]) if ok.any() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", default=DEFAULT, help=f"experiments (default {DEFAULT})")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--arm", default="from_2023_with_preinstall")
    ap.add_argument("--raster", default="master_raster_v3.tif")
    ap.add_argument("--pin", default="configs/pinned_partition_2026-11.json", help="'' = let the seed pick the stations")
    ap.add_argument("--quick", action="store_true", help=f"screening settings {QUICK}")
    ap.add_argument("--override", nargs="*", help="extra section.key=value for every experiment")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--redo-preprocess", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, (d, o) in EXPERIMENTS.items():
            print(f"{k:20s} {d}")
        return
    unknown = [e for e in a.exp if e not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiments {unknown}; --list shows them")
    if not a.compare_only:
        order = ["baseline"] + [e for e in a.exp if e != "baseline"]
        for seed in a.seeds:
            for name in order:
                try:
                    run_experiment(a, name, seed)
                except RuntimeError as e:
                    print(f"[{name} seed {seed}] FAILED: {e}", flush=True)
                compare(a)
    else:
        compare(a)


if __name__ == "__main__":
    main()

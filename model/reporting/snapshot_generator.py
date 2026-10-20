"""Write the results of a run into snapshots/<run_id>/ (layout: reporting/layout.py).

Reads what the geocp stage wrote (predictions, GeoCP summary), computes the accuracy, and writes SUMMARY.md (the page
to read first), metrics_summary.json, the small CSV tables and the plots.
"""
import json
import shutil
from pathlib import Path

import pandas as pd

from reporting import accuracy, plotters
from reporting.layout import DESCRIPTIONS, snap_file
from reporting.summary import build_markdown
from training.metrics import compute_all
from training.run_info import load_run_info, update_run_info, update_runs_index, write_json


def split_metrics(m: dict, cfg: dict) -> dict:
    """The three metrics of Validation.txt at the top level, the rest under 'supplementary'."""
    primary = {k: m[k] for k in cfg["report"]["metrics"]}
    extra = {k: m[k] for k in cfg["report"].get("extra_metrics", []) if k in m}
    return {**primary, "supplementary": extra}


def station_metrics(pred: pd.DataFrame, cfg: dict) -> list[dict]:
    rows = []
    for sid, grp in pred.groupby("location_id"):
        rows.append({"location_id": int(sid), "rows": int(len(grp)),
                     **split_metrics(compute_all(grp["y_true"], grp["y_pred"]), cfg)})
    return rows


def compare_to_baseline(test: pd.DataFrame, baseline_id: str, cfg: dict) -> dict:
    """R2 / RMSE / MAE of this run versus a baseline run (normally the temporal-only model, trained with
    --override model.spatial_cnn.enabled=false) on the SAME test rows: how much the morphology adds."""
    paths = cfg["data"]["paths"]
    bpath = snap_file(Path(paths["snapshots_dir"]) / baseline_id, "predictions_test.csv")
    if not bpath.exists():
        raise FileNotFoundError(f"{bpath} missing: run --stage geocp --run-id {baseline_id} first")
    base = pd.read_csv(bpath, parse_dates=["timestamp"])
    both = test.merge(base[["location_id", "timestamp", "y_pred"]], on=["location_id", "timestamp"],
                      suffixes=("", "_baseline"))
    m, b = compute_all(both["y_true"], both["y_pred"]), compute_all(both["y_true"], both["y_pred_baseline"])
    binfo = load_run_info(Path(paths["checkpoints_dir"]) / baseline_id)
    temporal_only = not binfo["hyperparameters"]["model"]["spatial_cnn"].get("enabled", True)
    return {
        "baseline_run_id": baseline_id, "baseline_is_temporal_only": temporal_only,
        "rows_compared": int(len(both)),
        "this_run": split_metrics(m, cfg), "baseline": split_metrics(b, cfg),
        "r2_gain": m["r2"] - b["r2"], "rmse_change": m["rmse"] - b["rmse"], "mae_change": m["mae"] - b["mae"],
        "reading": ("r2_gain = variance in PM2.5 explained by adding the morphology patches to the temporal "
                    "inputs" if temporal_only else "baseline is NOT a temporal-only run; r2_gain is a model "
                    "comparison, not a morphology effect"),
    }


def _read(snap: Path, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(snap_file(snap, name), **kw)


def _tables(snap: Path, measures: dict, stations: list[dict], years: list[dict], info: dict) -> None:
    """The small spreadsheet-friendly CSVs."""
    labels = {"cross_validation": "cross-validation (out-of-fold)", "calibration": "calibration", "test": "test"}
    rows = [{"split": labels[k], **v} for k, v in measures.items()]
    pd.DataFrame(rows).to_csv(snap_file(snap, "accuracy_summary.csv", write=True), index=False)
    if stations:
        pd.DataFrame(stations).to_csv(snap_file(snap, "accuracy_by_station.csv", write=True), index=False)
    if years:
        pd.DataFrame(years).to_csv(snap_file(snap, "accuracy_by_year.csv", write=True), index=False)
    if info.get("cv"):
        pd.DataFrame([{"fold": c["fold"], "held_out_stations": len(c.get("val_stations") or []), "fit_rows": c.get("fit_rows"),
                       "val_rows": c.get("val_rows"), **{k: c["metrics"][k] for k in ("rmse", "mae", "r2")},
                       "best_epoch": c["best_epoch"], "epochs_run": c["epochs_run"]} for c in info["cv"]]
                     ).to_csv(snap_file(snap, "cv_folds.csv", write=True), index=False)


def present_files(snap: Path) -> list[str]:
    """Relative names of the files that exist, in the order of layout.DESCRIPTIONS."""
    out = []
    for rel in DESCRIPTIONS:
        path = snap / rel
        legacy = snap / Path(rel).name
        if path.exists() or (not rel.endswith("/") and legacy.exists() and not path.exists()):
            out.append(rel if path.exists() else Path(rel).name)
    return out


def generate_snapshot(run_id: str, cfg: dict, baseline: str | None = None) -> Path:
    """Read the predictions written by the geocp stage, compute the accuracy, write SUMMARY.md,
    metrics_summary.json, the tables and the plots, and add the metrics to run_info.json / runs_index.csv."""
    run_dir = Path(cfg["data"]["paths"]["checkpoints_dir"]) / run_id
    snap = Path(cfg["data"]["paths"]["snapshots_dir"]) / run_id
    test = _read(snap, "predictions_test.csv", parse_dates=["timestamp"])
    cal = _read(snap, "predictions_calibration.csv", parse_dates=["timestamp"])
    geocp = json.loads(snap_file(snap, "geocp.json").read_text(encoding="utf-8"))
    info = load_run_info(run_dir)

    test_metrics = compute_all(test["y_true"], test["y_pred"])
    cal_metrics = compute_all(cal["y_true"], cal["y_pred"])

    measures = {}
    oof_path = run_dir / "oof_predictions.csv"
    if oof_path.exists():
        measures["cross_validation"] = accuracy.measures(pd.read_csv(oof_path, parse_dates=["timestamp"]), cfg)
    measures["calibration"] = accuracy.measures(cal, cfg)
    measures["test"] = accuracy.measures(test, cfg)
    by_station = accuracy.by_station(test, cfg, (geocp.get("test") or {}).get("per_station"))
    by_year = accuracy.by_year(test, cfg)

    summary = {
        "run_id": run_id,
        "primary_metrics": cfg["report"]["metrics"],
        "note": "RMSE and MAE in ug/m3; R2 = share of PM2.5 variance explained. Computed on stations withheld "
                "from model fitting.",
        "test": {"rows": int(len(test)), "stations": int(test["location_id"].nunique()),
                 **split_metrics(test_metrics, cfg)},
        "calibration": {"rows": int(len(cal)), "stations": int(cal["location_id"].nunique()),
                        **split_metrics(cal_metrics, cfg)},
        "cv_folds": [{"fold": f["fold"], "val_stations": f.get("val_stations"),
                      **split_metrics(f["metrics"], cfg)} for f in info.get("cv", [])],
        "accuracy": {"measures": measures, "yardsticks": accuracy.persistence_reference(test, cfg), "by_station": by_station,
                     "by_year": by_year},
        "geocp": {k: geocp[k] for k in ("selected", "test") if k in geocp},
        "test_by_station": station_metrics(test, cfg),
        "parameters": info["parameters"],
        "spatial_branch": info["hyperparameters"]["model"]["spatial_cnn"].get("enabled", True),
        "hyperparameters_file": "run_info.json",
    }
    if baseline:
        summary["comparison_to_baseline"] = compare_to_baseline(test, baseline, cfg)
    bias_path = snap_file(snap, "spatial_bias.json")
    if bias_path.exists():
        bias = json.loads(bias_path.read_text(encoding="utf-8"))
        summary["spatial_bias"] = {k: bias[k] for k in ("cells", "global_moran", "cluster_counts", "skipped", "note")
                                   if k in bias}
    write_json(snap / "metrics_summary.json", summary)
    _tables(snap, measures, by_station, by_year, info)

    wanted = set(cfg["report"]["plots"])
    history_path = run_dir / "history.json"
    if "loss_curve" in wanted and history_path.exists():
        plotters.plot_loss_curve(json.loads(history_path.read_text(encoding="utf-8")),
                                 str(snap_file(snap, "loss_curve.png", write=True)))
    if "pred_vs_obs" in wanted:
        plotters.plot_pred_vs_obs(test["y_true"], test["y_pred"], str(snap_file(snap, "pred_vs_obs.png", write=True)),
                                  test_metrics)
    if "residual_hist" in wanted:
        plotters.plot_residual_hist(test["y_true"], test["y_pred"], str(snap_file(snap, "residual_hist.png", write=True)))
    if "timeseries_sample" in wanted:
        plotters.plot_timeseries_sample(test, str(snap_file(snap, "timeseries_sample.png", write=True)),
                                        cfg["report"]["timeseries_days"])
    if "geocp_coverage" in wanted and geocp.get("test"):
        plotters.plot_geocp_coverage(geocp["test"]["per_station"], geocp["test"]["target_coverage"],
                                     str(snap_file(snap, "geocp_coverage.png", write=True)))
    if by_station and "accuracy_by_station" in wanted:
        plotters.plot_accuracy_by_station(by_station, measures["test"]["rmse"],
                                          str(snap_file(snap, "accuracy_by_station.png", write=True)))

    sections = {"test_metrics": test_metrics, "calibration_metrics": cal_metrics,
                "accuracy": {k: {m: v for m, v in d.items() if not isinstance(v, str)} for k, d in measures.items()}}
    if baseline:
        sections["comparison_to_baseline"] = summary["comparison_to_baseline"]
    info = update_run_info(run_dir, **sections)
    update_runs_index(cfg["data"]["paths"]["checkpoints_dir"], info)
    shutil.copy(run_dir / "run_info.json", snap / "run_info.json")

    (snap / "SUMMARY.md").write_text(build_markdown(summary, info, ["SUMMARY.md"] + present_files(snap)), encoding="utf-8")
    return snap

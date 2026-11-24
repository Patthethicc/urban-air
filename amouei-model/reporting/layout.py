"""Where each file of a snapshot lives.

    snapshots/<run_id>/
        SUMMARY.md              read this first: accuracy, uncertainty, spatial bias, what every file is
        metrics_summary.json    every number of the summary, machine-readable
        run_info.json           hyperparameters and provenance of the run
        plots/                  PNG figures
        tables/                 small CSVs you can open in a spreadsheet
        data/                   row-level predictions and the GeoCP calibration (read by later stages)
        surface/ explain/       city-wide maps and Grad-CAM (written by `infer` / `explain`)

Snapshots made before this layout kept everything in the top folder. `snap_file` reads either, so old runs still work.
"""
from pathlib import Path

SUBDIR = {
    "loss_curve.png": "plots", "pred_vs_obs.png": "plots", "residual_hist.png": "plots",
    "timeseries_sample.png": "plots", "accuracy_by_station.png": "plots", "geocp_coverage.png": "plots",
    "lisa_map.png": "plots", "lisa_on_bands.png": "plots",
    "accuracy_summary.csv": "tables", "accuracy_by_station.csv": "tables", "accuracy_by_year.csv": "tables", "cv_folds.csv": "tables",
    "predictions_test.csv": "data", "predictions_calibration.csv": "data", "calibration_table.csv": "data",
    "residuals_by_cell.csv": "data", "geocp.json": "data", "spatial_bias.json": "data",
}

DESCRIPTIONS = {
    "SUMMARY.md": "this report: start here",
    "metrics_summary.json": "every number of the report, machine-readable",
    "run_info.json": "hyperparameters, data sizes and provenance of the run",
    "plots/loss_curve.png": "training and validation loss per epoch, per fold",
    "plots/pred_vs_obs.png": "predicted against observed PM2.5 at the test stations",
    "plots/residual_hist.png": "distribution of the errors (predicted - observed)",
    "plots/timeseries_sample.png": "a week of predictions against observations",
    "plots/accuracy_by_station.png": "error (RMSE) of every test station",
    "plots/geocp_coverage.png": "share of readings inside the 90% interval, per test station",
    "plots/lisa_map.png": "mean residual per grid cell; the residual clusters (LISA) are marked when Moran's I is significant",
    "plots/lisa_on_bands.png": "the residual clusters over the morphology bands (only when LISA ran)",
    "tables/accuracy_summary.csv": "accuracy per data partition, one row each",
    "tables/accuracy_by_station.csv": "accuracy and interval coverage of every test station",
    "tables/accuracy_by_year.csv": "test accuracy per calendar year, with the number of stations of each network",
    "tables/cv_folds.csv": "accuracy of every cross-validation fold",
    "data/predictions_test.csv": "every test prediction with its interval",
    "data/predictions_calibration.csv": "every calibration prediction",
    "data/calibration_table.csv": "georeferenced calibration errors (the input of GeoCP, also for other models)",
    "data/geocp.json": "the GeoCP configuration search and its test summary",
    "data/spatial_bias.json": "Moran's I / LISA result",
    "data/residuals_by_cell.csv": "residuals aggregated by grid cell",
    "surface/": "city-wide prediction and interval maps (`--stage infer`)",
    "explain/": "Grad-CAM of hot and cold cells (`--stage explain`)",
}


def describe(name: str) -> str:
    """One-line description of a snapshot file, given its path in the new layout or its old top-folder name."""
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    if name in SUBDIR:                                   # an old snapshot kept the file in the top folder
        return DESCRIPTIONS.get(f"{SUBDIR[name]}/{name}", "")
    return ""


def snap_file(snap, name: str, write: bool = False) -> Path:
    """Path of a snapshot file. For writing: the new location (its folder is created). For reading: the new
    location, or the old top-folder location when only that exists."""
    snap = Path(snap)
    new = snap / SUBDIR[name] / name if name in SUBDIR else snap / name
    if write:
        new.parent.mkdir(parents=True, exist_ok=True)
        return new
    legacy = snap / name
    return legacy if not new.exists() and legacy.exists() else new

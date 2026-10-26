"""CLI entry point.

    python run_pipeline.py --stage rasters       # one-off: build the Master GeoTIFF (slow)

    python run_pipeline.py --stage preprocess
    python run_pipeline.py --stage tune          # Bayesian search, nested spatial CV (slow)
    python run_pipeline.py --stage train [--tuned [latest|<tuning folder>]]
    python run_pipeline.py --stage geocp   [--run-id 20260101_120000]
    python run_pipeline.py --stage bias    [--run-id 20260101_120000]   # Moran's I / LISA on out-of-fold residuals
    python run_pipeline.py --stage report  [--run-id ...] [--baseline <temporal-only run id>]
    python run_pipeline.py --stage coverage      # does the Master GeoTIFF cover all sensors / the whole NCR? (read-only)
    python run_pipeline.py --stage clean         # RESET: dry run, lists the generated outputs it would delete
    python run_pipeline.py --stage clean --yes   #        ... and deletes them (also: --keep tuning, --include-rasters)
    python run_pipeline.py --stage surface --prediction-raster pm25.tif [--run-id ...]   # GeoCP rasters for any prediction raster
    python run_pipeline.py --stage infer --date 2025-07-01 [--run-id ...]      # 08:00 12:00 17:00 21:00 volume + maps
    python run_pipeline.py --stage infer --timestamp "2025-07-01 12:00" ...    # explicit hours
    python run_pipeline.py --stage explain --timestamp "2025-07-01 12:00"      # Grad-CAM of hot / cold cells
    python run_pipeline.py --stage all           # preprocess -> train -> geocp -> report

    --tuned   train with the hyperparameters found by --stage tune instead of the YAML values
    --override train.epochs=5 --override "data.temporal.lags=[2,1]"   (section.key=value, YAML values)

Long stages print progress and stop with exit code 124 if they hang or pass train.runtime.max_hours (runtime_guard.py).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import runtime_guard
from config_parser import load_config
from reporting.layout import snap_file
from reporting.summary import headline_text, training_summary_text
from runtime_guard import StageTimeout
from seed import set_seed

MODEL_FILE = "best_dual_branch_model.pt"


# ------------------------------------------------------------------ helpers
def _paths(cfg: dict) -> dict:
    return cfg["data"]["paths"]


def _cell_stations(stations: pd.DataFrame) -> pd.DataFrame:
    """GeoCP measures distances between raster CELL CENTRES (Section 4.5.2): x, y become the cell-centre
    coordinates; the sensor's own coordinates are kept as sensor_x / sensor_y. Old processed folders
    without cell columns fall back to the sensor coordinates."""
    if not {"cell_x", "cell_y"} <= set(stations.columns):
        return stations
    return stations.assign(sensor_x=stations["x"], sensor_y=stations["y"], x=stations["cell_x"], y=stations["cell_y"])


def _load_geocp(cfg: dict, run_id: str):
    """The GeoCP engine of a finished geocp stage, rebuilt from calibration_table.csv + geocp.json."""
    from uncertainty.geocp import GeoCP

    snap = Path(_paths(cfg)["snapshots_dir"]) / run_id
    selected = json.loads(snap_file(snap, "geocp.json").read_text(encoding="utf-8"))["selected"]
    table = pd.read_csv(snap_file(snap, "calibration_table.csv"))
    stations = _cell_stations(pd.read_csv(Path(_paths(cfg)["processed_dir"]) / "stations.csv"))
    return GeoCP(selected["alpha"], selected["kernel"], float(selected["bandwidth_m"]),
                 selected["finite_sample_correction"]).fit(table[["location_id", "score"]], stations)


def _ensure_geocp(cfg: dict, run_id: str) -> None:
    """The interval rasters need the GeoCP calibration of the run (calibration_table.csv + geocp.json), which only the
    geocp stage writes. Run it now when it is missing, as `report` does, instead of failing on a missing file."""
    snap = Path(_paths(cfg)["snapshots_dir"]) / run_id
    if not (snap_file(snap, "geocp.json").exists() and snap_file(snap, "calibration_table.csv").exists()):
        print(f"no GeoCP calibration found for run {run_id}: running --stage geocp first "
              "(it calibrates the prediction intervals)")
        geocp(cfg, run_id)


def latest_run_id(cfg: dict) -> str:
    runs = sorted(p.parent.name for p in Path(_paths(cfg)["checkpoints_dir"]).glob(f"*/{MODEL_FILE}"))
    if not runs:
        raise FileNotFoundError("no trained run found; run --stage train first")
    return runs[-1]


def _save_hourly(df: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    """Hourly PM2.5 matrix [hours, stations] on a gap-free time axis. Cross-validation needs it to
    recompute the spatial lag from each fold's own fit stations (training/fold_features.py)."""
    c = cfg["data"]["columns"]
    wide = df.pivot(index=c["timestamp"], columns=c["group"], values=c["target"])
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq=cfg["data"]["time"]["freq"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "pm25_hourly.npz", values=wide.to_numpy(dtype=np.float32),
                        location_ids=wide.columns.to_numpy(dtype=np.int64), start=str(wide.index[0]))


# ------------------------------------------------------------------ stages
def build_rasters(cfg: dict) -> None:
    """vector_sanitizer -> raster_harmonizer -> master_raster_builder -> Master GeoTIFF.

    Writes processed/rasters/master_morphology_stack_<res>m.tif; point paths.master_raster at it
    afterwards (the pilot 500 m raster is never overwritten)."""
    import geopandas as gpd
    import rasterio

    from preprocessing import master_raster_builder as mrb
    from preprocessing import raster_harmonizer as rh
    from preprocessing.vector_sanitizer import ingest_and_sanitize

    d, p = cfg["data"], _paths(cfg)
    needed = ["buildings_csv", "streets", "height_raster", "ndvi_raster", "boundary"]
    missing = [k for k in needed if not p[k]]
    if missing:
        raise ValueError(f"set these paths in configs/data.yaml first: {missing}")

    work = Path(p["raster_work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    res, crs = d["raster"]["resolution_m"], d["crs"]

    print("1/5 sanitising vectors")
    buildings, streets = ingest_and_sanitize(cfg)
    boundary = gpd.read_file(p["boundary"]).to_crs(crs)

    print("2/5 harmonising height + NDVI to one grid")
    # one bilinear step to exact res x res pixels (4 m height -> 10 m; NDVI is already 10 m)
    height_10 = rh.reproject_raster(p["height_raster"], str(work / "height_10m.tif"), crs, resolution=res)
    height_clip = rh.clip_and_mask(height_10, boundary, str(work / "height_10m_clip.tif"))
    ndvi_10 = rh.reproject_raster(p["ndvi_raster"], str(work / "ndvi_10m.tif"), crs, resolution=res)
    ndvi_aligned = rh.align_to_reference(ndvi_10, height_clip, str(work / "ndvi_10m_aligned.tif"))

    with rasterio.open(height_clip) as ref:
        profile = ref.profile.copy()
        height = np.nan_to_num(ref.read(1, masked=True).filled(0.0))
    with rasterio.open(ndvi_aligned) as src:
        ndvi = np.nan_to_num(src.read(1, masked=True).filled(0.0))

    print("3/5 rasterising building density + street width")
    density = mrb.rasterize_building_density(buildings, profile, d["raster"]["density_supersample"])
    width = mrb.rasterize_street_width(streets, profile)
    print("4/5 aspect ratio")
    aspect = mrb.compute_pixel_aspect_ratio(height, width, cfg)
    height_std = mrb.compute_height_std(height, cfg)

    print("5/5 stacking Master GeoTIFF")
    bands = {"height": height, "ndvi": ndvi, "building_density": density,
             "street_width": width, "aspect_ratio": aspect, "height_std": height_std}
    out = work / f"master_morphology_stack_{int(res)}m.tif"
    mrb.build_master_geotiff(bands, profile, str(out), d["raster"]["band_order"])
    print(f"done ({d['crs']}): {out}\nSet paths.master_raster to this file in configs/data.yaml.")


def coverage(cfg: dict) -> dict:
    """Read-only: which sensors fall outside the Master GeoTIFF (they would be skipped = a silent crop of the study
    area) and, if data.paths.boundary is set, how much of the NCR boundary the raster covers."""
    from config_parser import check_raster_crs
    from preprocessing.coverage import check_coverage

    check_raster_crs(cfg)
    return check_coverage(cfg)


def preprocess(cfg: dict) -> None:
    """Order matters (the spatial lag needs the partition):
      1. clean_loader.load_clean                       hourly, interpolated (cap 24 h)
      2. spatial_partitioner.partition_locations       train / calibration / test by location
      3. spatial_lag.add_spatial_lag                   reference = train stations only
      4. temporal_window_builder.build_windows         X_temporal [N,T,F], y [N,1], meta
      5. spatial_patch_extractor.extract_patches       one patch per station
      6. multimodal_aligner.stack_patches/station_index
      7. Scalers.fit on TRAIN rows / stations only, transform all partitions
      8. multimodal_aligner.save_splits                patches.npy + 9 per-partition files + scalers.json
    """
    from config_parser import check_raster_crs
    from preprocessing.clean_loader import load_clean
    from preprocessing.multimodal_aligner import save_splits, stack_patches, station_index
    from preprocessing.scalers import Scalers
    from preprocessing.spatial_lag import add_spatial_lag, station_coordinates
    from preprocessing.spatial_partitioner import partition_indices, partition_locations
    from preprocessing.spatial_patch_extractor import add_cell_columns, extract_patches
    from preprocessing.temporal_window_builder import build_windows

    d, dv = cfg["data"], cfg["derived"]
    out_dir = Path(_paths(cfg)["processed_dir"])
    check_raster_crs(cfg)
    from preprocessing.coverage import check_coverage
    check_coverage(cfg)                              # lists sensors outside the raster; stops if data.coverage.strict

    print("loading + cleaning sensor series")
    df = load_clean(cfg)
    stations = add_cell_columns(station_coordinates(df, cfg), _paths(cfg)["master_raster"])
    partition = partition_locations(stations["location_id"], cfg)
    print("stations per partition:", {k: len(v) for k, v in partition.items()})

    _save_hourly(df, cfg, out_dir)
    if d["spatial_lag"]["enabled"]:
        df = add_spatial_lag(df, partition, cfg)
    X, y, meta = build_windows(df, cfg)
    print(f"windows: X_temporal {X.shape}, y {y.shape}")

    patches = extract_patches(_paths(cfg)["master_raster"], stations, cfg)
    patch_arr, patch_index = stack_patches(patches)
    rows = partition_indices(meta, partition)
    empty = [k for k, v in rows.items() if len(v) == 0]
    if empty:
        raise ValueError(f"partitions with no windows: {empty}")

    train_patch_rows = [patch_index[i] for i in partition["train"]]
    scalers = Scalers().fit(patch_arr[train_patch_rows], y[rows["train"]],
                            dv["pm25_feature_idx"], dv["rolling_std_idx"])
    patch_scaled = scalers.transform_patches(patch_arr)

    splits = {}
    for name, idx in rows.items():
        m = meta.iloc[idx].reset_index(drop=True)
        splits[name] = {
            "station_idx": station_index(m, patch_index),
            "X_temporal": scalers.transform_temporal(X[idx]),
            "y": scalers.transform_y(y[idx]),
            "meta": m,
        }
    save_splits({"patches": patch_scaled, "patch_index": patch_index, "partition": partition,
                 "stations": stations, "splits": splits}, str(out_dir))
    scalers.save(str(out_dir / "scalers.json"))

    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "shapes": {"patches": list(patch_scaled.shape),
                   **{n: {"X_temporal": list(s["X_temporal"].shape), "y": list(s["y"].shape)} for n, s in splits.items()}},
        "rows_per_partition": {k: int(len(v)) for k, v in rows.items()},
        "stations_per_partition": {k: len(v) for k, v in partition.items()},
        "partition": partition,
        "feature_names": dv["feature_names"], "crs": d["crs"],
        "T": dv["T"], "F": dv["F"], "C": dv["C"], "patch_px": dv["patch_px"],
        "scalers": scalers.to_dict(),
        "temporal": d["temporal"], "spatial_lag": d["spatial_lag"], "time": d["time"],
    }
    (out_dir / "preprocess_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"saved to {out_dir}")


def tune(cfg: dict) -> str:
    """Bayesian optimisation (GP surrogate + acquisition) with nested spatial CV on the training
    stations; writes tuning/<id>/best_hyperparameters.json (+ search logs, tune_info.json, plot)."""
    from training.tuning import nested_tuning

    return str(nested_tuning(cfg))


def train(cfg: dict) -> str:
    """Spatial CV on the training stations (fold best epochs set the final epoch count), then one final
    fit on ALL training stations. Saves best_dual_branch_model.pt, history.json and run_info.json.
    Returns run_id (timestamp)."""
    from networks.dual_branch_net import DualBranchNet
    from preprocessing.scalers import Scalers
    from preprocessing.spatial_partitioner import fold_settings, spatial_cv_folds
    from torch.utils.data import Subset

    from training.dataset import AirQualityDataset, load_meta
    from training.fold_features import SpatialLagRecomputer
    from training.metrics import compute_all
    from training.run_info import new_run_info, save_run_info, update_runs_index, write_json
    from training.trainer import Trainer, make_loader

    tc, mc = cfg["train"], cfg["model"]
    processed = _paths(cfg)["processed_dir"]
    if not (Path(processed) / "scalers.json").is_file():
        raise SystemExit(f"nothing to train on: {processed} has no preprocessed data. Run `--stage preprocess` first, "
                         "with the same --override settings (data paths, split, cv_assignment) as this command.")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(_paths(cfg)["checkpoints_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    scalers = Scalers.load(str(Path(processed) / "scalers.json"))
    data_summary = json.loads((Path(processed) / "preprocess_info.json").read_text(encoding="utf-8"))
    data_summary.pop("scalers", None)
    meta_train = load_meta(processed, "train")
    from training.checkpoint import file_sha1
    extra = {"derived": cfg["derived"], "lags": cfg["data"]["temporal"]["lags"], "run_id": run_id,
             "crs": cfg["data"]["crs"],
             "scalers_sha1": file_sha1(Path(processed) / "scalers.json")}

    set_seed(tc["seed"])
    info = new_run_info(run_id, cfg, DualBranchNet(mc), data_summary)
    info["status"] = "running"                    # completed / timed_out replace it; still "running" = killed from outside
    save_run_info(run_dir, info)
    history = {"folds": [], "final": None}
    folds = int(cfg["data"]["split"]["cv_folds"])
    runtime_guard.set_plan(passes=(folds + folds / (folds - 1)) if tc["cv"]["enabled"] and folds > 1 else 1,
                           max_epochs=int(tc["epochs"]))       # K folds on (K-1)/K of the stations + the final fit on all

    train_ds = AirQualityDataset(processed, "train")
    recomputer = SpatialLagRecomputer(cfg)
    best_epochs, oof_parts = [], []
    if tc["cv"]["enabled"]:
        for k, (fit_idx, val_idx) in enumerate(spatial_cv_folds(meta_train, cfg["data"]["split"]["cv_folds"], **fold_settings(cfg))):
            set_seed(tc["seed"] + k + 1)
            fold_ds, fit_idx, val_idx = recomputer.apply(train_ds, fit_idx, val_idx)   # lag from fit stations only
            fit_ds, val_ds = Subset(fold_ds, fit_idx), Subset(fold_ds, val_idx)
            trainer = Trainer(DualBranchNet(mc), tc, mc, str(run_dir), extra)
            hist = trainer.fit(make_loader(fit_ds, tc, True), make_loader(val_ds, tc, False),
                               ckpt_path=str(run_dir / f"fold_{k}.pt"), tag=f"[fold {k}]")
            _, y_true, y_pred = trainer.evaluate(make_loader(val_ds, tc, False))
            metrics = compute_all(scalers.inverse_y(y_true), scalers.inverse_y(y_pred))
            oof_parts.append(meta_train.iloc[val_idx][["location_id", "timestamp"]].reset_index(drop=True).assign(
                fold=k, y_true=scalers.inverse_y(y_true).ravel(), y_pred=scalers.inverse_y(y_pred).ravel()))
            best_epochs.append(hist["best_epoch"])
            history["folds"].append(hist)
            info["cv"].append({
                "fold": k,
                "val_stations": sorted(int(i) for i in meta_train.iloc[val_idx]["location_id"].unique()),
                "fit_rows": int(len(fit_idx)), "val_rows": int(len(val_idx)),
                "best_epoch": hist["best_epoch"], "epochs_run": hist["epochs_run"],
                "best_val_loss": hist["best_val_loss"], "metrics": metrics,
            })
            print(f"[fold {k}] held-out stations RMSE {metrics['rmse']:.3f}  MAE {metrics['mae']:.3f}  R2 {metrics['r2']:.3f}")

    if oof_parts:                                  # out-of-fold predictions: the spatial-bias analysis reads these
        pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)

    final_epochs = tc["final_epochs"] or (int(round(np.mean(best_epochs))) if best_epochs else tc["epochs"])
    final_epochs = max(1, int(final_epochs))
    print(f"final fit on all training stations for {final_epochs} epochs")
    set_seed(tc["seed"])
    trainer = Trainer(DualBranchNet(mc), tc, mc, str(run_dir), extra)
    hist = trainer.fit(make_loader(train_ds, tc, True), None, epochs=final_epochs,
                       ckpt_path=str(run_dir / MODEL_FILE), tag="[final]")
    history["final"] = hist
    info["status"] = "completed"
    info["final_fit"] = {"epochs": final_epochs, "epochs_source": ("tuned epochs (mean best epoch of the winning trial)" if cfg.get("tuned_from")
                                           else "final_epochs override") if tc["final_epochs"]
                         else ("mean best epoch of CV folds" if best_epochs else "train.epochs"),
                         "last_train_loss": hist["train_loss"][-1], "checkpoint": MODEL_FILE}

    write_json(run_dir / "history.json", history)
    save_run_info(run_dir, info)
    update_runs_index(_paths(cfg)["checkpoints_dir"], info)
    print(f"run {run_id} saved to {run_dir}")
    print(training_summary_text(info))
    return run_id


def geocp(cfg: dict, run_id: str | None) -> str:
    """Predict calibration + test, pick the GeoCP configuration on the calibration stations only,
    then attach intervals to the test predictions (Validation.txt).

    Writes snapshots/<run_id>/data/: calibration_table.csv (georeferenced scores), predictions_*.csv, geocp.json."""
    from training.checkpoint import check_compatibility, load_model
    from training.predict import predict_partition
    from training.run_info import update_run_info, update_runs_index, write_json
    from uncertainty.geocp import (GeoCP, apply_intervals, calibration_table, scores_from_predictions,
                                   select_config)

    run_id = run_id or latest_run_id(cfg)
    run_dir = Path(_paths(cfg)["checkpoints_dir"]) / run_id
    snap = Path(_paths(cfg)["snapshots_dir"]) / run_id
    snap.mkdir(parents=True, exist_ok=True)
    gcfg = cfg["geocp"]

    model, ckpt = load_model(str(run_dir / MODEL_FILE))
    check_compatibility(ckpt, _paths(cfg)["processed_dir"])            # same patches, bands, lags, scalers as training
    stations = _cell_stations(pd.read_csv(Path(_paths(cfg)["processed_dir"]) / "stations.csv"))

    cal = predict_partition(model, cfg, "calibration")
    scores = scores_from_predictions(cal)
    best, table = select_config(scores, stations, gcfg)               # calibration stations only
    print(f"GeoCP config: {best}")
    # The other value of finite_sample_correction, scored the same way (leave-one-station-out on the
    # CALIBRATION stations). This is how to decide between the two options without using the test set.
    other, _ = select_config(scores, stations, {**gcfg, "finite_sample_correction": not gcfg["finite_sample_correction"]})
    comparison = {"note": "leave-one-station-out on the calibration stations only; the test stations are not used",
                  "normalised_sum_to_one": other if gcfg["finite_sample_correction"] else best,
                  "finite_sample_correction": best if gcfg["finite_sample_correction"] else other,
                  "in_use": "finite_sample_correction" if gcfg["finite_sample_correction"] else "normalised_sum_to_one"}
    for name in ("normalised_sum_to_one", "finite_sample_correction"):
        c = comparison[name]
        print(f"  {name:25s} LOSO coverage {c['loso_coverage']:.3f}  mean width {c['loso_mean_width']:.2f}  "
              f"({c['kernel']}, bandwidth {c['bandwidth_m']})")
    gm = GeoCP(gcfg["alpha"], best["kernel"], best["bandwidth_m"], gcfg["finite_sample_correction"]).fit(scores, stations)

    test = predict_partition(model, cfg, "test")                       # first touch of the test stations
    test, test_summary = apply_intervals(test, gm, stations, gcfg["min_lower_bound"],
                                         gcfg["zone_threshold_m"], gcfg["width_ratio_floor"])
    print(f"test coverage {test_summary['coverage']:.3f} (target {test_summary['target_coverage']:.2f}), "
          f"mean width {test_summary['mean_width']:.2f} ug/m3, median {test_summary['median_width']:.2f}")
    print(f"  urban core {test_summary['coverage_urban_core']}, peripheral {test_summary['coverage_peripheral']}")

    calibration_table(cal, stations).to_csv(snap_file(snap, "calibration_table.csv", write=True), index=False)
    cal.to_csv(snap_file(snap, "predictions_calibration.csv", write=True), index=False)
    test.to_csv(snap_file(snap, "predictions_test.csv", write=True), index=False)
    write_json(snap_file(snap, "geocp.json", write=True), {"selected": best, "candidates": table.to_dict(orient="records"),
                                     "correction_comparison": comparison, "test": test_summary})

    info = update_run_info(run_dir, geocp={"selected": best, "correction_comparison": comparison,
                                           "test": {k: v for k, v in test_summary.items() if k != "per_station"}})
    update_runs_index(_paths(cfg)["checkpoints_dir"], info)
    return run_id


def bias(cfg: dict, run_id: str | None) -> str:
    """Moran's I / LISA on the run's out-of-fold residuals (training stations only)."""
    from reporting.spatial_bias import run_spatial_bias

    run_id = run_id or latest_run_id(cfg)
    summary = run_spatial_bias(run_id, cfg)
    if "skipped" in summary:
        print(f"spatial bias: {summary['skipped']}")
    else:
        g = summary["global_moran"]
        print(f"spatial bias: Moran's I {g['I']:.3f} (p {g['p_sim']:.3f}, {summary['cells']} cells); "
              + (f"clusters {summary['cluster_counts']}" if "cluster_counts" in summary else "LISA not run"))
    return run_id


def surface(cfg: dict, run_id: str | None, prediction_raster: str) -> str:
    """Per-cell GeoCP interval rasters (prediction, lower, upper, width) for ANY point-prediction raster."""
    from uncertainty.geocp_surface import build_interval_rasters

    run_id = run_id or latest_run_id(cfg)
    _ensure_geocp(cfg, run_id)
    snap = Path(_paths(cfg)["snapshots_dir"]) / run_id
    out = snap / "geocp_surface.tif"
    stats = build_interval_rasters(prediction_raster, _load_geocp(cfg, run_id), str(out),
                                   cfg["geocp"]["min_lower_bound"], expected_crs=cfg["data"]["crs"])
    print(f"interval surface: {stats}")
    return str(out)


def _inference_context(cfg: dict, run_id: str) -> dict:
    """Everything the per-cell stages share: model, scalers, hourly readings, reference stations and the
    zero-padded Master GeoTIFF. Runs the checkpoint-compatibility and raster-CRS checks first."""
    from config_parser import check_raster_crs
    from inference.cell_inference import padded_raster
    from preprocessing.scalers import Scalers
    from training.checkpoint import check_compatibility, load_model
    from training.trainer import resolve_device

    check_raster_crs(cfg)
    run_dir = Path(_paths(cfg)["checkpoints_dir"]) / run_id
    processed = Path(_paths(cfg)["processed_dir"])
    model, ckpt = load_model(str(run_dir / MODEL_FILE))
    check_compatibility(ckpt, str(processed))

    z = np.load(processed / "pm25_hourly.npz")
    stations = pd.read_csv(processed / "stations.csv")
    partition = json.loads((processed / "partition.json").read_text(encoding="utf-8"))
    ref_ids = partition["train"] if cfg["inference"]["lag_reference"] == "train" else sorted(stations["location_id"])
    padded, transform, crs, shape = padded_raster(_paths(cfg)["master_raster"], cfg["derived"]["patch_px"])
    return {
        "model": model, "scalers": Scalers.load(str(processed / "scalers.json")),
        "hourly": {"values": z["values"], "location_ids": z["location_ids"], "start": pd.Timestamp(str(z["start"]))},
        "stations": stations, "ref_ids": ref_ids,
        "ref_xy": stations.set_index("location_id").loc[ref_ids, ["x", "y"]].to_numpy(),
        "padded": padded, "transform": transform, "crs": crs, "shape": shape,
        "device": resolve_device(cfg["train"]["device"]),
    }


def infer(cfg: dict, run_id: str | None, timestamps: list[str], dates: list[str]) -> list[str]:
    """City-wide inference and map generation (Output.txt, Section 4.5).

    For every date: one prediction surface per snapshot hour (inference.snapshot_hours: 08, 12, 17, 21),
    stacked into a prediction volume (one band per snapshot) with the aligned GeoCP lower / upper / width
    volumes, a 24-hour mean surface, AQI-coloured maps (optional outline above 35 ug/m3, output.outline_above_threshold; the mean
    at the 50 ug/m3 guideline) and output_summary.json, in snapshots/<run_id>/surface/<YYYYMMDD>/.

    Cells without a sensor take the spatial lag of their k nearest stations as PM2.5 history
    (configs/inference.yaml)."""
    from inference.cell_inference import _hour_index, openaq_only, predict_surface, reference_stations
    from output import cartography
    from output.aqi_maps import (STYLES, basemap_settings, continuous_range, render_panels, render_snapshot,
                                 reset_basemap_state)
    from output.volume import daily_mean, expand_dates, group_by_date, summarize_date, write_volume
    from training.run_info import write_json
    from uncertainty.geocp_surface import interval_layers

    stamps = [pd.Timestamp(t) for t in timestamps] + expand_dates(dates, cfg["inference"]["snapshot_hours"])
    if not stamps:
        raise SystemExit("--stage infer needs --date 2025-07-01 or at least one --timestamp '2025-07-01 12:00'")
    run_id = run_id or latest_run_id(cfg)
    _ensure_geocp(cfg, run_id)
    ctx = _inference_context(cfg, run_id)
    for ts in stamps:                                         # a date without readings behind it stops here, before any folder is made
        _hour_index(ts, ctx["hourly"], cfg)
    geocp_engine = _load_geocp(cfg, run_id)
    o = cfg["output"]
    reset_basemap_state()

    written = []
    for date, group in group_by_date(stamps).items():
        out_dir = Path(_paths(cfg)["snapshots_dir"]) / run_id / "surface" / date.strftime("%Y%m%d")
        (out_dir / "maps").mkdir(parents=True, exist_ok=True)
        labels = [t.strftime("%H:%M") for t in group]
        only = openaq_only(group[0], cfg)                      # every hour of a date is in the same year
        if only:
            print(f"{date:%Y-%m-%d}: stand-in history from OpenAQ stations only (inference.openaq_only_from_year = "
                  f"{cfg['inference']['openaq_only_from_year']}), {len(reference_stations(ctx['ref_ids'], ctx['ref_xy'], True, cfg)[0])} "
                  "reference stations")

        preds, tr = predict_surface(ctx["model"], ctx["scalers"], ctx["padded"], ctx["shape"], ctx["transform"],
                                    ctx["hourly"], ctx["ref_ids"], ctx["ref_xy"], group, cfg, ctx["device"])
        # one pass over the raster for all hours; the interval layers are written in place (no per-hour lists + copies)
        stack = {"prediction": preds, **{k: np.empty_like(preds) for k in ("lower", "upper", "width")}}
        for i, (ts, pred) in enumerate(zip(group, preds)):
            lo, hi, w = interval_layers(pred, tr, geocp_engine, cfg["geocp"]["min_lower_bound"])
            stack["lower"][i], stack["upper"][i], stack["width"][i] = lo, hi, w
            print(f"{ts}: {int(np.isfinite(pred).sum())} cells, mean {np.nanmean(pred):.1f} ug/m3, "
                  f"mean interval width {np.nanmean(w):.1f}")

        for key, arr in stack.items():
            write_volume(arr, tr, ctx["crs"], str(out_dir / f"{key}_volume.tif"), labels)
        mean = daily_mean(stack["prediction"]) if len(group) > 1 else None
        if mean is not None:
            write_volume(mean[None], tr, ctx["crs"], str(out_dir / "prediction_daily_mean.tif"), ["24-hour mean"])

        sensors = ctx["stations"]
        styles = cartography.settings(cfg)["styles"]
        bad = [x for x in styles if x not in STYLES]
        if bad:
            raise SystemExit(f"output.maps.styles has unknown style(s) {bad}: use any of {list(STYLES)}")
        cell_m = abs(tr.a)
        caption = (f"{date:%Y-%m-%d}, run {run_id}; {cell_m:g} m prediction cells (inference.stride_px {cfg['inference']['stride_px']}), "
                   f"{ctx['crs']}; city boundaries: GADM 4.1." + (" Every cell's history comes from OpenAQ stations only." if only else ""))
        shared = continuous_range(stack["prediction"], tr, cfg) if "continuous" in styles else None     # one colour scale for all hours of the day
        files = {"aqi": ("aqi_{h}", "daily_panels", "daily_mean"),                       # file stems per style (the AQI names are the old ones)
                 "continuous": ("continuous_{h}", "continuous_panels", "continuous_daily_mean")}
        for style in styles:
            hourly_stem, panels_stem, mean_stem = files[style]
            vr = shared if style == "continuous" else None
            for i, label in enumerate(labels):
                render_snapshot(stack["prediction"][i], tr, ctx["crs"], str(out_dir / "maps" / f"{hourly_stem.format(h=label.replace(':', ''))}.png"),
                                f"PM2.5 prediction, {date:%Y-%m-%d} {label}", cfg, o["highlight_threshold"], sensors, style=style,
                                vrange=vr, caption=caption)
            if len(group) > 1:
                render_panels(stack["prediction"], labels, tr, ctx["crs"], str(out_dir / "maps" / f"{panels_stem}.png"),
                              f"PM2.5 prediction, {date:%Y-%m-%d}", cfg, o["highlight_threshold"], sensors, style=style, vrange=vr,
                              caption=caption)
                render_snapshot(mean, tr, ctx["crs"], str(out_dir / "maps" / f"{mean_stem}.png"),
                                f"24-hour mean of {len(group)} snapshots, {date:%Y-%m-%d}", cfg, o["daily_guideline"], sensors,
                                style=style, caption=caption)
        from output.aqi_maps import last_basemap
        summary = summarize_date(group, stack["prediction"], stack["width"], cfg)
        summary["lag_stations"] = "OpenAQ only" if only else "all reference stations"
        for snap, pred in zip(summary["snapshots"], stack["prediction"]):          # cells and coverage as printed in the map titles
            st = cartography.grid_stats(pred, tr, cfg)
            snap["map_cells"], snap["map_coverage"] = st["cells"], st["coverage"]
        summary["map_styles"] = styles
        summary["clipped_to_ncr"] = cartography.active_boundary(cfg) is not None
        summary["basemap"] = last_basemap() or "none"            # web | vectors | none (plain background)
        summary["basemap_settings"] = basemap_settings(cfg)
        write_json(out_dir / "output_summary.json", summary)
        print(f"basemap used: {summary['basemap']}")
        print(f"{date:%Y-%m-%d}: volume + maps written to {out_dir}")
        written.append(str(out_dir))
    return written


def explain(cfg: dict, run_id: str | None, timestamp: str | None) -> str:
    """Grad-CAM (Output.txt): for the highest- and lowest-predicted cells of one inferred snapshot, which pixels
    of the morphology patch drove the CNN branch. Writes overlays, gradcam_attention.csv and a summary to
    snapshots/<run_id>/explain/<YYYYMMDD_HHMM>/. Run --stage infer for that date first."""
    import rasterio
    from rasterio.transform import rowcol, xy as cell_xy_of

    from explain.gradcam import attention_profile, grad_cam, spread_selection
    from explain.gradcam_map import plot_gradcam_map
    from explain.gradcam_plots import plot_band_profile, plot_cam_overlays
    from explain.morphology_sheet import plot_morphology_sheet
    from output import cartography
    from inference.cell_inference import cell_temporal_features, gather_patches, openaq_only, reference_stations
    from reporting.spatial_bias import band_names_for
    from training.run_info import write_json

    if not timestamp:
        raise SystemExit("--stage explain needs --timestamp 'YYYY-MM-DD HH:MM' (a snapshot made by --stage infer)")
    run_id = run_id or latest_run_id(cfg)
    ts = pd.Timestamp(timestamp)
    volume = Path(_paths(cfg)["snapshots_dir"]) / run_id / "surface" / ts.strftime("%Y%m%d") / "prediction_volume.tif"
    if not volume.exists():
        raise FileNotFoundError(f"{volume} missing: run --stage infer for {ts:%Y-%m-%d} first")
    out_dir = Path(_paths(cfg)["snapshots_dir"]) / run_id / "explain" / ts.strftime("%Y%m%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)

    px = cfg["derived"]["patch_px"]
    if px < 3:
        msg = (f"patches are {px}x{px} pixel(s) (the {cfg['derived']['raster_resolution_m']:g} m pilot raster): "
               "a saliency map needs a larger patch; use the 10 m raster")
        print(f"explain: {msg}")
        write_json(out_dir / "gradcam_summary.json", {"skipped": msg})
        return str(out_dir)

    with rasterio.open(volume) as src:
        label = ts.strftime("%H:%M")
        if label not in src.descriptions:
            raise ValueError(f"{label} is not a snapshot of {volume.name}: {list(src.descriptions)}")
        pred_map, vol_transform = src.read(list(src.descriptions).index(label) + 1), src.transform

    ctx = _inference_context(cfg, run_id)
    ex = cfg["output"]["explain"]
    boundary = cartography.active_boundary(cfg)
    ok = np.isfinite(pred_map)
    if boundary is not None:                                  # hot and cold are ranked inside the NCR, not in the sea or the next provinces
        ok &= cartography.inside_mask(vol_transform, pred_map.shape, boundary)
    valid = np.flatnonzero(ok.ravel())
    order = valid[np.argsort(pred_map.ravel()[valid])]

    def pick(best_first, n):                                  # the n best cells that are at least min_separation_m apart
        flat = best_first[:20000]
        px_, py_ = cell_xy_of(vol_transform, *np.unravel_index(flat, pred_map.shape), offset="center")
        return flat[spread_selection(np.asarray(px_), np.asarray(py_), n, float(ex.get("min_separation_m", 0)))]

    groups = {"hot": pick(order[::-1], int(ex["top_n"])), "cold": pick(order, int(ex["bottom_n"]))}
    names = band_names_for(_paths(cfg)["master_raster"], cfg)
    t_idx = int(round((ts - ctx["hourly"]["start"]) / pd.Timedelta(cfg["data"]["time"]["freq"])))

    ref_ids, ref_xy = reference_stations(ctx["ref_ids"], ctx["ref_xy"], openaq_only(ts, cfg), cfg)    # the same rule as `infer`

    res = cfg["derived"]["raster_resolution_m"]
    caption = (f"{ts:%Y-%m-%d %H:%M}, run {run_id}; prediction cells of {abs(vol_transform.a):g} m, morphology patches of {px * res:g} m; "
               "Grad-CAM on the last convolutional layer of the CNN branch; city boundaries: GADM 4.1."
               + (" The history of every cell comes from OpenAQ stations only." if openaq_only(ts, cfg) else ""))
    cells = []
    rows, summary = [], {"timestamp": str(ts), "patch_px": px, "attention_quantile": ex["attention_quantile"], "groups": {},
                         "ranked_inside_ncr": boundary is not None}
    for group, flat in groups.items():
        r_out, c_out = np.unravel_index(flat, pred_map.shape)
        xs, ys = cell_xy_of(vol_transform, r_out, c_out, offset="center")
        r_pix, c_pix = (np.asarray(v) for v in rowcol(ctx["transform"], xs, ys))
        raw = gather_patches(ctx["padded"], r_pix, c_pix, px)
        feats = cell_temporal_features(ctx["hourly"]["values"], ctx["hourly"]["location_ids"], ref_ids, ref_xy,
                                       np.column_stack([xs, ys]), t_idx, ts, cfg)
        keep = ~np.isnan(feats).any(axis=(1, 2))
        raw, feats, xs, ys = raw[keep], feats[keep], np.asarray(xs)[keep], np.asarray(ys)[keep]
        cam, p_scaled = grad_cam(ctx["model"], ctx["scalers"].transform_patches(raw),
                                 ctx["scalers"].transform_temporal(feats), ctx["device"])
        pred = ctx["scalers"].inverse_y(p_scaled).ravel()
        prof = attention_profile(raw, cam, names, ex["attention_quantile"])
        for r in prof:
            rows.append({"group": group, "cell_number": r["patch"] + 1, "x": float(xs[r["patch"]]), "y": float(ys[r["patch"]]),
                         "predicted": float(pred[r["patch"]]), **{k: v for k, v in r.items() if k != "patch"}})
        cells += [{"group": group, "number": i + 1, "x": float(xs[i]), "y": float(ys[i]), "predicted": float(pred[i])}
                  for i in range(len(cam))]
        titles = [f"{group.upper()} #{i + 1}\n{pred[i]:.1f} \u00b5g/m\u00b3\n({xs[i]:.0f}, {ys[i]:.0f})" for i in range(len(cam))]
        plot_cam_overlays(raw, cam, names, titles, str(out_dir / f"gradcam_{group}_cells.png"), ex["attention_quantile"],
                          cfg["output"]["dpi"], cfg=cfg, px_m=res, caption=caption,
                          heading=f"Grad-CAM, {group} cells of {ts:%Y-%m-%d %H:%M}")
        by_band = pd.DataFrame([r for r in prof]).groupby("band")["std_units"].mean() if prof else pd.Series(dtype=float)
        summary["groups"][group] = {"cells": int(len(cam)), "mean_prediction": float(pred.mean()),
                                    "attention_std_units_by_band": {k: float(v) for k, v in by_band.items()}}

    pd.DataFrame(rows).to_csv(out_dir / "gradcam_attention.csv", index=False)
    dpi = cfg["output"]["dpi"]
    if rows:
        plot_band_profile(rows, names, str(out_dir / "gradcam_band_profile.png"), dpi, cfg=cfg, caption=caption,
                          heading=f"What the CNN attends to, hot against cold cells ({ts:%Y-%m-%d %H:%M})")
    plot_gradcam_map(pred_map, vol_transform, ctx["crs"], cells, cfg, str(out_dir / "gradcam_map.png"),
                     f"PM2.5 prediction, {ts:%Y-%m-%d %H:%M}", stations=ctx["stations"], caption=caption, dpi=dpi)
    plot_morphology_sheet(_paths(cfg)["master_raster"], cfg, cells, str(out_dir / "morphology_sheet.png"), names,
                          stations=ctx["stations"], caption=caption, dpi=dpi)
    summary["hot_cold"] = ("hot = the highest predicted PM2.5 cells of this hour, cold = the lowest "
                           "(inside the NCR when a boundary is set); Grad-CAM shows which pixels of each cell's patch drove its prediction")
    summary["reading"] = ("std_units > 0: the attended pixels carry higher values of that band than the rest of the "
                          "patch (e.g. dense blocks, deep canyons, wide arterials); inspect the overlays to confirm "
                          "they are aerodynamically plausible structures")
    write_json(out_dir / "gradcam_summary.json", summary)
    print(f"Grad-CAM written to {out_dir}")
    return str(out_dir)


def report(cfg: dict, run_id: str | None, baseline: str | None = None) -> None:
    """generate_snapshot for run_id (default: latest run); runs the geocp and bias stages first if needed."""
    from reporting.snapshot_generator import generate_snapshot

    run_id = run_id or latest_run_id(cfg)
    snap_dir = Path(_paths(cfg)["snapshots_dir"]) / run_id
    _ensure_geocp(cfg, run_id)
    if not snap_file(snap_dir, "spatial_bias.json").exists() and (Path(_paths(cfg)["checkpoints_dir"]) / run_id / "oof_predictions.csv").exists():
        bias(cfg, run_id)
    snap = generate_snapshot(run_id, cfg, baseline)
    print(headline_text(json.loads((snap / "metrics_summary.json").read_text(encoding="utf-8")), snap))
    print(f"snapshot written to {snap}")


def clean(cfg: dict, execute: bool = False, keep=(), include_rasters: bool = False,
          include_tile_cache: bool = False, root=None) -> None:
    """RESET: delete the generated outputs (processed/, checkpoints/, snapshots/, tuning/) so the next run starts from
    nothing, e.g. after trying the pilot data. A dry run unless `execute` (--yes). Never touches configs, .env, code
    or input data; keeps the built Master GeoTIFF (processed/rasters/) and the tile cache unless asked."""
    from cleanup import MODEL_DIR, describe, execute_cleanup, human, plan_cleanup

    root = root or MODEL_DIR
    plan = plan_cleanup(cfg, keep, include_rasters, include_tile_cache, root)
    files, nbytes = sum(t.files for t in plan), sum(t.bytes for t in plan)
    if files == 0:
        print("Nothing to clean: no generated outputs found.")
        return
    print("Deleting:" if execute else "Would delete (dry run):")
    print(describe(plan, root))
    kept = [k for t in plan for k in t.kept]
    if kept:
        print("Kept (pass --include-rasters to delete): the built Master GeoTIFF in processed/rasters/")
    print("Never touched: configs/, .env, the code, and your input data (sensor CSVs, source rasters).")
    if not execute:
        print(f"\nDry run: nothing was deleted ({files} files, {human(nbytes)} would go). "
              "Run the same command with --yes to delete.")
        return
    done_files, done_bytes = execute_cleanup(plan)
    print(f"\nDeleted {done_files} files, freed {human(done_bytes)}. Start again with:  python run_pipeline.py --stage preprocess")


# ------------------------------------------------------------------ CLI
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["rasters", "preprocess", "tune", "train", "geocp", "bias", "surface", "infer", "explain", "report", "all",
                                                       "clean", "reset", "coverage"])
    p.add_argument("--yes", action="store_true", help="clean: really delete (without it clean is a dry run)")
    p.add_argument("--keep", action="append", default=[], choices=["processed", "checkpoints", "snapshots", "tuning"],
                   help="clean: leave this folder alone (repeatable)")
    p.add_argument("--include-rasters", action="store_true", help="clean: also delete the built Master GeoTIFF (processed/rasters)")
    p.add_argument("--include-tile-cache", action="store_true", help="clean: also delete the cached basemap tiles")
    p.add_argument("--tuned", nargs="?", const="latest", default=None,
                   help="train with tune output: latest (default) or a tuning folder")
    p.add_argument("--run-id", default=None)
    p.add_argument("--baseline", default=None, help="report: run id of a temporal-only run to compare R2 with")
    p.add_argument("--prediction-raster", default=None, help="surface: point-prediction GeoTIFF")
    p.add_argument("--timestamp", action="append", default=[], help="infer / explain: local hour, e.g. '2025-07-01 12:00' (repeatable for infer)")
    p.add_argument("--date", action="append", default=[], help="infer: a day, expanded to inference.snapshot_hours (repeatable)")
    p.add_argument("--override", action="append", default=[], help="section.key=value (YAML value)")
    args = p.parse_args()

    cfg = load_config(args.override)
    if args.tuned:
        from training.tuning import apply_tuned
        cfg = apply_tuned(cfg, args.tuned)
        print(f"using tuned hyperparameters from {cfg['tuned_from']}")
    set_seed(cfg["train"]["seed"])
    runtime_guard.configure(cfg["train"].get("runtime"), on_hard_exit=lambda: _record_timeout(cfg, args.stage, "hang (forced exit)"))
    try:
        run(cfg, args)
    except StageTimeout as err:
        _record_timeout(cfg, args.stage, str(err))
        print(f"\nTIMEOUT in stage '{args.stage}': {err}", file=sys.stderr)
        sys.exit(runtime_guard.EXIT_CODE)


def _record_timeout(cfg: dict, stage: str, reason: str) -> None:
    """Best effort: mark the unfinished training run as timed out, so its folder says why it has no model."""
    try:
        if stage not in ("train", "all"):
            return
        runs = sorted(p for p in Path(_paths(cfg)["checkpoints_dir"]).glob("*/run_info.json")
                      if not (p.parent / MODEL_FILE).exists())
        if runs:
            info = json.loads(runs[-1].read_text(encoding="utf-8"))
            info.update(status="timed_out", timeout=reason)
            runs[-1].write_text(json.dumps(info, indent=2), encoding="utf-8")
    except Exception:
        pass                                      # never hide the timeout behind a bookkeeping error


def run(cfg: dict, args) -> None:
    stage = args.stage
    if stage == "coverage":
        coverage(cfg)
        return
    if stage in ("clean", "reset"):
        clean(cfg, args.yes, args.keep, args.include_rasters, args.include_tile_cache)
        return
    if stage == "rasters":
        build_rasters(cfg)
    if stage in ("preprocess", "all"):
        preprocess(cfg)
    if stage == "tune":
        tune(cfg)
    run_id = args.run_id
    if stage in ("train", "all"):
        run_id = train(cfg)
    if stage in ("geocp", "all"):
        geocp(cfg, run_id)
    if stage == "bias":
        bias(cfg, run_id)
    if stage == "surface":
        if not args.prediction_raster:
            raise SystemExit("--stage surface needs --prediction-raster")
        surface(cfg, run_id, args.prediction_raster)
    if stage == "infer":
        infer(cfg, run_id, args.timestamp, args.date)
    if stage == "explain":
        explain(cfg, run_id, args.timestamp[0] if args.timestamp else None)
    if stage in ("report", "all"):
        report(cfg, run_id, args.baseline)


if __name__ == "__main__":
    main()

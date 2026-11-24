"""Per-run metadata: one run_info.json per run plus a runs_index.csv row, so runs are easy to compare.

Hyperparameters do NOT adjust themselves during a run: they are whatever configs/*.yaml (plus any
--override) said. run_info.json records exactly what was used, and each later stage (geocp, report)
adds its results to the same file.
"""
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _clean(obj):
    """JSON-safe: inf/nan -> strings, numpy scalars -> python."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            obj = obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return str(obj)
    return obj


def write_json(path, data) -> None:
    Path(path).write_text(json.dumps(_clean(data), indent=2), encoding="utf-8")


def new_run_info(run_id: str, cfg: dict, model, data_summary: dict) -> dict:
    return {
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "overrides": cfg.get("overrides", []),
        "tuned_from": cfg.get("tuned_from"),
        "tuned_hyperparameters": cfg.get("tuned_hyperparameters"),
        "hyperparameters": {
            "model": cfg["model"],
            "train": cfg["train"],
            "temporal": cfg["data"]["temporal"],
            "spatial_lag": cfg["data"]["spatial_lag"],
            "patch": cfg["data"]["patch"],
            "split": cfg["data"]["split"],
            "geocp": cfg["geocp"],
        },
        "derived": cfg["derived"],
        "parameters": model.parameter_counts(),
        "architecture": str(model),
        "data": data_summary,
        "cv": [],
        "final_fit": {},
        "test_metrics": None,
        "calibration_metrics": None,
        "geocp": None,
        "spatial_bias": None,
    }


def load_run_info(run_dir) -> dict:
    return json.loads((Path(run_dir) / "run_info.json").read_text(encoding="utf-8"))


def save_run_info(run_dir, info: dict) -> None:
    write_json(Path(run_dir) / "run_info.json", info)


def update_run_info(run_dir, **sections) -> dict:
    info = load_run_info(run_dir)
    info.update(sections)
    save_run_info(run_dir, info)
    return info


def update_runs_index(checkpoints_dir, info: dict) -> None:
    """One row per run in checkpoints/runs_index.csv (rewritten if the run already has a row)."""
    hp = info["hyperparameters"]
    cv = [f.get("metrics", {}).get("rmse") for f in info.get("cv", [])]
    cv = [v for v in cv if v is not None]
    gc = info.get("geocp") or {}
    row = {
        "run_id": info["run_id"],
        "created": info["created"],
        "git_commit": info.get("git_commit"),
        "overrides": ";".join(info.get("overrides", [])),
        "tuned": bool(info.get("tuned_from")),
        "params": info["parameters"]["total"],
        "lstm_hidden": hp["model"]["temporal_lstm"]["hidden_size"],
        "cnn_channels": "x".join(str(c) for c in hp["model"]["spatial_cnn"]["conv_channels"]),
        "feature_map": hp["model"]["spatial_cnn"].get("feature_map", "pool"),
        "spatial_branch": hp["model"]["spatial_cnn"].get("enabled", True),
        "lr": hp["train"]["optimizer"]["lr"],
        "batch_size": hp["train"]["batch_size"],
        "lags": "-".join(str(l) for l in hp["temporal"]["lags"]),
        "final_epochs": info.get("final_fit", {}).get("epochs"),
        "cv_rmse_mean": round(sum(cv) / len(cv), 4) if cv else None,
        "test_rmse": (info.get("test_metrics") or {}).get("rmse"),
        "test_mae": (info.get("test_metrics") or {}).get("mae"),
        "test_r2": (info.get("test_metrics") or {}).get("r2"),
        "geocp_coverage": (gc.get("test") or {}).get("coverage"),
        "geocp_width": (gc.get("test") or {}).get("mean_width"),
    }
    path = Path(checkpoints_dir) / "runs_index.csv"
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["run_id"] != row["run_id"]]
    rows.append({k: ("" if v is None else v) for k, v in row.items()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerows(rows)

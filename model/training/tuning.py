"""Bayesian hyperparameter optimisation with nested spatial cross-validation (Training.txt).

    outer fold k  = a group of training stations held out as a test set
      inner folds = spatial GroupKFold over the OTHER training stations; they drive the search
      search      = Gaussian-process surrogate (skopt.gp_minimize) + acquisition function
                    objective(hp) = mean over inner folds of the validation loss of the whole
                    dual-branch network. tune.search_epochs false (default): best loss with early
                    stopping, so a trial costs only the epochs it needs. true: `epochs` is searched,
                    trained exactly, loss after the last epoch.
      refit       = train on all outer-training stations with the winning hyperparameters for the
                    winning trial's mean best epoch, score on the outer test stations
                    -> honest estimate of the tuned model
    final search  = the same search over ALL training stations -> best_hyperparameters.json

Inside every fold the spatial lag is recomputed from that fold's fit stations only
(training/fold_features.py). Calibration and test stations are never touched.
`train --tuned` then trains with the result.
"""
import inspect
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from networks.dual_branch_net import DualBranchNet
from preprocessing.scalers import Scalers
from preprocessing.spatial_partitioner import nested_spatial_folds, spatial_cv_folds
import runtime_guard
from runtime_guard import format_duration
from seed import set_seed
from training.dataset import AirQualityDataset, load_meta
from training.fold_features import SpatialLagRecomputer
from training.metrics import compute_all
from training.run_info import write_json
from training.trainer import Trainer, make_loader


def to_native(v):
    return v.item() if hasattr(v, "item") else v


def build_space(space_cfg: dict) -> list:
    from skopt.space import Categorical, Integer, Real

    dims = []
    for name, s in space_cfg.items():
        if s["type"] == "real":
            dims.append(Real(float(s["low"]), float(s["high"]), prior=s.get("prior", "uniform"), name=name))
        elif s["type"] == "integer":
            dims.append(Integer(int(s["low"]), int(s["high"]), name=name))
        elif s["type"] == "categorical":
            dims.append(Categorical(list(s["values"]), name=name))
        else:
            raise ValueError(f"unknown search dimension type '{s['type']}' for {name}")
    return dims


def effective_space(tune_cfg: dict) -> dict:
    """tune.space, plus `epochs` (tune.epochs_space) when tune.search_epochs is on."""
    space = dict(tune_cfg["space"])
    if tune_cfg.get("search_epochs"):
        space["epochs"] = tune_cfg["epochs_space"]
    return space


def apply_hyperparameters(cfg: dict, hp: dict) -> dict:
    """Copy of cfg with the searched values written into train / model. Names follow tune.yaml."""
    cfg = deepcopy(cfg)
    tc, mc = cfg["train"], cfg["model"]
    if "learning_rate" in hp:
        tc["optimizer"]["lr"] = float(hp["learning_rate"])
    if "batch_size" in hp:
        tc["batch_size"] = int(hp["batch_size"])
    if "epochs" in hp:
        tc["epochs"] = int(hp["epochs"])
    if "cnn_kernel_size" in hp:
        mc["spatial_cnn"]["kernel_size"] = int(hp["cnn_kernel_size"])
    if "cnn_filters" in hp:
        layers = len(mc["spatial_cnn"]["conv_channels"])
        mc["spatial_cnn"]["conv_channels"] = [int(hp["cnn_filters"]) * 2 ** i for i in range(layers)]
    if "cnn_feature_map" in hp:
        mc["spatial_cnn"]["feature_map"] = str(hp["cnn_feature_map"])
    if "lstm_hidden_size" in hp:
        mc["temporal_lstm"]["hidden_size"] = int(hp["lstm_hidden_size"])
    if "recurrent_dropout" in hp:
        mc["temporal_lstm"]["dropout"] = float(hp["recurrent_dropout"])
    cfg["tuned_hyperparameters"] = {k: to_native(v) for k, v in hp.items()}
    return cfg


def hp_to_overrides(hp: dict, cfg: dict) -> list[str]:
    """The tuned values as `--override` strings, so they can be pasted into any command (or copied into the YAML
    files) to make them the settings of a normal `train`. Applying them gives the same configuration as `--tuned`."""
    out = []
    layers = len(cfg["model"]["spatial_cnn"]["conv_channels"])
    if "learning_rate" in hp:
        out.append(f"train.optimizer.lr={float(hp['learning_rate'])!r}")
    if "batch_size" in hp:
        out.append(f"train.batch_size={int(hp['batch_size'])}")
    if "epochs" in hp:
        out += [f"train.epochs={int(hp['epochs'])}", f"train.final_epochs={int(hp['epochs'])}"]
    if "cnn_kernel_size" in hp:
        out.append(f"model.spatial_cnn.kernel_size={int(hp['cnn_kernel_size'])}")
    if "cnn_filters" in hp:
        out.append("model.spatial_cnn.conv_channels=[" + ", ".join(str(int(hp["cnn_filters"]) * 2 ** i) for i in range(layers)) + "]")
    if "cnn_feature_map" in hp:
        out.append(f"model.spatial_cnn.feature_map={hp['cnn_feature_map']}")
    if "lstm_hidden_size" in hp:
        out.append(f"model.temporal_lstm.hidden_size={int(hp['lstm_hidden_size'])}")
    if "recurrent_dropout" in hp:
        out.append(f"model.temporal_lstm.dropout={float(hp['recurrent_dropout'])!r}")
    return out


def apply_tuned(cfg: dict, ref: str) -> dict:
    """Use the output of a tune run for training. `ref` = a tuning folder, a best_hyperparameters.json,
    or 'latest'. The tuned `epochs` (mean best epoch of the winning trial) is the final-fit epoch count."""
    root = Path(cfg["data"]["paths"]["tuning_dir"])
    if ref == "latest":
        runs = sorted(p.parent for p in root.glob("*/best_hyperparameters.json"))
        if not runs:
            raise FileNotFoundError("no tuning run found; run --stage tune first")
        folder = runs[-1]
    else:
        folder = Path(ref)
        folder = folder.parent if folder.suffix == ".json" else folder
        if not folder.is_absolute() and not folder.exists():
            folder = root / ref
    hp = json.loads((folder / "best_hyperparameters.json").read_text(encoding="utf-8"))["hyperparameters"]
    out = apply_hyperparameters(cfg, hp)
    if "epochs" in hp:
        out["train"]["final_epochs"] = int(hp["epochs"])
    out["tuned_from"] = str(folder)
    return out


# ------------------------------------------------------------------ the search
_PACE_PENDING = [True]                   # the "at this pace ..." projection is printed once per tuning run


def gp_search(objective, space_cfg: dict, tune_cfg: dict, log_path: Path | None = None, tag: str = "",
              searches: int = 1) -> dict:
    """Run skopt.gp_minimize on objective(hp) -> (loss, details). Every trial is logged (and written to
    log_path as it happens, so an interrupted search keeps what it learned).

    An objective that takes a `label` argument is told which trial it runs ("[outer 0] trial 3/12"), so its progress
    lines can say where they are. `searches` = how many searches the whole tuning run has (for the one-off projection
    printed after the very first trial)."""
    from skopt import gp_minimize
    from skopt.utils import use_named_args

    dims = build_space(space_cfg)
    n_calls = int(tune_cfg["n_calls"])
    takes_label = "label" in inspect.signature(objective).parameters
    trials: list[dict] = []

    @use_named_args(dims)
    def wrapped(**kw):
        hp = {k: to_native(v) for k, v in kw.items()}
        label = f"{tag} trial {len(trials) + 1}/{n_calls}".strip()
        t0 = time.time()
        loss, details = objective(hp, label=label) if takes_label else objective(hp)
        loss = float(loss) if np.isfinite(loss) else 1e3          # a diverged model must not break the GP
        took = time.time() - t0
        trials.append({"trial": len(trials) + 1, "loss": loss, "hyperparameters": hp,
                       "seconds": round(took, 1), **details})
        best = min(t["loss"] for t in trials)
        left = (n_calls - len(trials)) * float(np.mean([t["seconds"] for t in trials]))
        remaining = f", about {format_duration(left)} left in this search" if left > 0 else ""
        print(f"{label}  loss {loss:.4f}  best {best:.4f}  took {format_duration(took)}{remaining}  {hp}", flush=True)
        if _PACE_PENDING[0]:
            _PACE_PENDING[0] = False
            one_search = n_calls * took
            print(f"  at this pace one search of {n_calls} trials takes about {format_duration(one_search)}, "
                  f"{searches} search{'es' if searches > 1 else ''} about {format_duration(searches * one_search)} "
                  "(the evaluation fits on all rows come on top)", flush=True)
        if log_path is not None:
            write_json(log_path, {"tag": tag, "trials": trials})
        return loss

    gp_minimize(wrapped, dims, n_calls=n_calls,
                n_initial_points=min(int(tune_cfg["n_initial_points"]), n_calls),
                acq_func=tune_cfg["acq_func"], random_state=int(tune_cfg["random_state"]))
    best = min(trials, key=lambda t: t["loss"])
    hp = dict(best["hyperparameters"])
    if "suggested_epochs" in best:
        hp["epochs"] = int(best["suggested_epochs"])
    return {"best_hyperparameters": hp, "best_loss": best["loss"], "trials": trials}


def make_objective(cfg: dict, dataset, folds, stride: int, seed: int, recomputer: SpatialLagRecomputer):
    """hp -> (mean validation loss over the inner folds, details). Each fold recomputes the spatial
    lag from the fold's fit stations first.

    tune.search_epochs false: early stopping (cap tune.max_epochs, patience tune.patience); the loss is
        the best validation loss reached and `suggested_epochs` the mean epoch where it was reached.
    tune.search_epochs true: hp["epochs"] is trained exactly (no early stopping); the loss is the
        validation loss after the last epoch."""
    t = cfg["tune"]
    searched = bool(t.get("search_epochs"))

    def objective(hp: dict, label: str = ""):
        c = apply_hyperparameters(cfg, hp)
        if searched:
            tc = {**c["train"], "early_stopping": False}               # epochs comes from hp
        else:
            tc = {**c["train"], "early_stopping": True, "early_stopping_patience": int(t["patience"]),
                  "epochs": int(t["max_epochs"])}
        losses, best_epochs, ran = [], [], []
        for k, (fit, val) in enumerate(folds):
            set_seed(seed + k)
            ds, fit, val = recomputer.apply(dataset, fit, val)
            trainer = Trainer(DualBranchNet(c["model"]), tc, c["model"], "")
            fit_tag = f"{label} fit {k + 1}/{len(folds)}".strip()
            t_fit = time.time()
            hist = trainer.fit(make_loader(Subset(ds, fit[::stride]), tc, True),
                               make_loader(Subset(ds, val[::stride]), tc, False),
                               epochs=tc["epochs"], verbose=False, tag=fit_tag)
            losses.append(hist["val_loss"][-1] if searched else hist["best_val_loss"])
            best_epochs.append(hist["best_epoch"])
            ran.append(hist["epochs_run"])
            print(f"  {fit_tag}: {hist['epochs_run']} epochs (best {hist['best_epoch']}), val loss {losses[-1]:.4f}, "
                  f"{format_duration(time.time() - t_fit)}", flush=True)
        details = {"fold_losses": [float(x) for x in losses], "best_epochs": best_epochs, "epochs_run": ran}
        if not searched:
            details["suggested_epochs"] = max(1, int(round(float(np.mean(best_epochs)))))
        return float(np.mean(losses)), details
    return objective


def nested_tuning(cfg: dict) -> Path:
    """Outer loop (optional) + final search. Returns the tuning folder."""
    from config_parser import validate_for_tuning
    validate_for_tuning(cfg)
    t, processed = cfg["tune"], cfg["data"]["paths"]["processed_dir"]
    tune_dir = Path(cfg["data"]["paths"]["tuning_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    tune_dir.mkdir(parents=True, exist_ok=True)

    dataset = AirQualityDataset(processed, "train")
    meta = load_meta(processed, "train")
    scalers = Scalers.load(str(Path(processed) / "scalers.json"))
    recomputer = SpatialLagRecomputer(cfg)
    stride, seed = int(t["row_stride"]), int(t["random_state"])
    nested = bool(t["nested_evaluation"])
    searches = (int(t["outer_folds"]) if nested else 0) + 1
    fits = searches * int(t["n_calls"]) * int(t["inner_folds"])
    _PACE_PENDING[0] = True
    status = runtime_guard.status_every_s()
    print(f"tuning plan: {searches} search{'es' if searches > 1 else ''} x {t['n_calls']} trials x {t['inner_folds']} inner fits"
          f" = {fits} fits on {'all rows' if stride == 1 else f'every {stride}th row'}"
          + (f", plus {t['outer_folds']} evaluation fits on all rows of the outer-training stations" if nested else "")
          + ". Output: a line per finished fit and trial"
          + (f", and a 'still running' line every {format_duration(status)}" if status else ""), flush=True)

    outer_results = []
    if nested:
        for k, (fit, test, inner) in enumerate(nested_spatial_folds(meta, t["outer_folds"], t["inner_folds"])):
            print(f"--- outer fold {k} (search {k + 1}/{searches}): "
                  f"{len(set(meta.iloc[test]['location_id']))} held-out stations", flush=True)
            res = gp_search(make_objective(cfg, dataset, inner, stride, seed + 100 * k, recomputer),
                            effective_space(t), t, tune_dir / f"search_outer_{k}.json", tag=f"[outer {k}]",
                            searches=searches)
            hp = res["best_hyperparameters"]
            c = apply_hyperparameters(cfg, hp)
            tc = {**c["train"], "early_stopping": False}          # fixed epoch count: no validation set here
            set_seed(seed + 100 * k)
            ds, fit_ok, test_ok = recomputer.apply(dataset, fit, test)   # lag from the outer-fit stations only
            trainer = Trainer(DualBranchNet(c["model"]), tc, c["model"], "")
            print(f"[outer {k}] evaluation fit with the winning hyperparameters: {len(fit_ok):,} rows, "
                  f"{tc['epochs']} epochs", flush=True)
            trainer.fit(make_loader(Subset(ds, fit_ok), tc, True), None, epochs=tc["epochs"], verbose=True,
                        tag=f"[outer {k}] evaluation fit")           # all rows: the same progress lines as training
            _, y_true, y_pred = trainer.evaluate(make_loader(Subset(ds, test_ok), tc, False))
            metrics = compute_all(scalers.inverse_y(y_true), scalers.inverse_y(y_pred))
            outer_results.append({
                "outer_fold": k,
                "fit_stations": sorted(int(i) for i in meta.iloc[fit]["location_id"].unique()),
                "held_out_stations": sorted(int(i) for i in meta.iloc[test]["location_id"].unique()),
                "fit_rows": int(len(fit_ok)), "held_out_rows": int(len(test_ok)),
                "best_hyperparameters": hp, "inner_loss": res["best_loss"], "outer_test_metrics": metrics,
            })
            print(f"[outer {k}] tuned model on held-out stations: RMSE {metrics['rmse']:.3f}  MAE {metrics['mae']:.3f}")

    print(f"--- final search over all training stations (search {searches}/{searches})", flush=True)
    final = gp_search(make_objective(cfg, dataset, spatial_cv_folds(meta, t["inner_folds"]), stride, seed, recomputer),
                      effective_space(t), t, tune_dir / "search_final.json", tag="[final]", searches=searches)
    write_json(tune_dir / "best_hyperparameters.json",
               {"hyperparameters": final["best_hyperparameters"], "inner_cv_loss": final["best_loss"],
                "epochs_mode": "searched" if t["search_epochs"] else "early_stopping",
                "epochs_note": ("epochs = searched hyperparameter, every trial trained exactly that many"
                                if t["search_epochs"] else
                                "epochs = mean best epoch of the winning trial's inner folds (early stopping)"),
                "use_with": "python run_pipeline.py --stage train --tuned",
                "as_overrides": hp_to_overrides(final["best_hyperparameters"], cfg),
                "note": "--tuned applies these to ONE command; a plain `train` uses configs/*.yaml. To make them the "
                        "default, copy them into configs/train.yaml and configs/model.yaml, or pass them as --override."})

    summary = {}
    if outer_results:
        for key in ("rmse", "mae", "r2"):
            v = [r["outer_test_metrics"][key] for r in outer_results]
            summary[key] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    write_json(tune_dir / "tune_info.json", {
        "created": datetime.now().isoformat(timespec="seconds"),
        "tune_config": t, "overrides": cfg.get("overrides", []),
        "base_hyperparameters": {"model": cfg["model"], "train": cfg["train"]},
        "outer_results": outer_results, "outer_summary": summary,
        "final": {"best_hyperparameters": final["best_hyperparameters"], "best_inner_loss": final["best_loss"]},
        "objective": ("mean inner-fold validation loss after exactly `epochs` epochs (epochs searched), "
                      if t["search_epochs"] else
                      "mean inner-fold best validation loss with early stopping (epochs not searched), ")
                     + "scaled units, spatial lag recomputed per fold",
    })

    from reporting.plotters import plot_search_convergence
    plot_search_convergence(final["trials"], str(tune_dir / "convergence_final.png"))
    print(f"tuning saved to {tune_dir}")
    print("Train with them:  python run_pipeline.py --stage train --tuned     (applies to that command only)")
    print("Make them the default: copy best_hyperparameters.json -> as_overrides into configs/train.yaml and model.yaml")
    return tune_dir

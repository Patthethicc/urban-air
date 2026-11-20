"""`--stage pretrain`: train the temporal branch (an LSTM-only DualBranchNet) on other cities' hourly PM2.5.

Result: <data.paths.pretrained_dir>/<name>/pretrained_lstm.pt (a normal checkpoint of the LSTM-only model), summary.json and station_ranking.csv.
Fine-tune on Manila with   --stage train --override train.init_from=<that file>   (an LSTM-only model, model.spatial_cnn.enabled=false, loads all of it;
a CNN-LSTM loads the temporal branch and leaves the CNN and the fusion head fresh). Nothing here reads a raster, weather or morphology.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

CITY_ID_STEP = 10 ** 9          # city i gets location ids + i * this, so ids of different cities never collide


class WindowDataset(Dataset):
    """(dummy patch, X_temporal [T,F], y [1]): the shape the Trainer expects; the dummy is ignored by a model without the CNN."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y
        self._dummy = torch.zeros(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self._dummy, torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])


def split_stations(location_ids: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Whole stations held out for early stopping: (train_ids, validation_ids). At least one of each when there are two or more stations."""
    ids = np.array(sorted(set(int(i) for i in location_ids)))
    if len(ids) < 2:
        return ids, ids[:0]
    n_val = int(min(len(ids) - 1, max(1, round(len(ids) * fraction))))
    order = np.random.default_rng(seed).permutation(len(ids))
    return np.sort(ids[order[n_val:]]), np.sort(ids[order[:n_val]])


def pretrain(cfg: dict, name: str = "default", sources: list[str] | None = None) -> Path:
    """Pretrain on the source cities and return the checkpoint path. `cfg` is the normal Manila config with the Manila preprocessing done."""
    from preprocessing.scalers import Scalers
    from seed import set_seed
    from training.checkpoint import file_sha1
    from training.trainer import Trainer
    from transfer.select import rank_stations
    from transfer.source_data import city_windows, hourly_long, list_sources, manila_training_sample

    tr = cfg["transfer"]
    root = Path(tr["source_root"]) if Path(tr["source_root"]).is_absolute() else (Path(__file__).resolve().parents[1] / tr["source_root"]).resolve()
    names = sources or tr["sources"] or list_sources(root)
    if not names:
        raise SystemExit(f"no source cities under {root}: run other-country-scrapers/scrapers.ipynb first (it writes data/Other Countries/<country>/<city>/)")
    processed = Path(cfg["data"]["paths"]["processed_dir"])
    if not (processed / "scalers.json").is_file():
        raise SystemExit(f"{processed} has no scalers.json: run `--stage preprocess` for Manila first, the source series are scaled with its scalers")
    out_dir = Path(cfg["data"]["paths"]["pretrained_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- the source windows of every city, with ids that cannot collide
    parts, longs = [], []
    for i, city in enumerate(names):
        X, y, meta = city_windows(root / city, cfg, id_offset=(i + 1) * CITY_ID_STEP, exclude_months=tr["exclude_months"])
        print(f"  {city}: {meta['location_id'].nunique()} stations, {len(y):,} windows")
        if not len(y):                    # e.g. a city whose few stations never report in the same hours: no spatial lag, so every window is dropped
            print(f"    -> {city} gives no training window and is left out: a window needs the station's own last hours AND the same hours of its "
                  f"neighbouring stations (spatial lag). With stations that never report at the same time, or a single station, none exists.")
            continue
        parts.append((X, y, meta))
        long = hourly_long(root / city, cfg)
        long["location_id"] += (i + 1) * CITY_ID_STEP
        longs.append(long)
    if not parts:
        raise SystemExit(f"no training window in any of the source cities {names}: see the lines above for why (stations that never overlap in time, "
                         f"too few stations, or exclude_months). Choose other cities for this checkpoint.")

    # ---- which stations look like Manila (always written; used to keep only the top k when transfer.top_k_stations is set)
    has_windows = set(int(i) for p in parts for i in p[2]["location_id"].unique())          # a station without any window cannot be pretrained on: do not rank it
    all_long = pd.concat(longs, ignore_index=True)
    ranking = rank_stations(all_long[all_long["location_id"].isin(has_windows)], manila_training_sample(cfg), seed=tr["seed"])
    ranking["city"] = [names[int(i // CITY_ID_STEP) - 1] for i in ranking["location_id"]]
    ranking.to_csv(out_dir / "station_ranking.csv", index=False)
    keep_ids = set(ranking["location_id"].head(int(tr["top_k_stations"]))) if tr["top_k_stations"] else None
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    meta = pd.concat([p[2] for p in parts], ignore_index=True)
    if keep_ids is not None:
        mask = meta["location_id"].isin(keep_ids).to_numpy()
        X, y, meta = X[mask], y[mask], meta.loc[mask].reset_index(drop=True)
    if tr["max_rows"] and len(y) > int(tr["max_rows"]):
        pick = np.sort(np.random.default_rng(tr["seed"]).permutation(len(y))[:int(tr["max_rows"])])
        X, y, meta = X[pick], y[pick], meta.iloc[pick].reset_index(drop=True)
    if not len(y):
        raise SystemExit("no source windows left: check transfer.exclude_months / top_k_stations / the source folders")

    # ---- Manila's scalers: the pretrained weights see the scale the fine-tuning will use
    scalers = Scalers.load(str(processed / "scalers.json"))
    Xs, ys = scalers.transform_temporal(X), scalers.transform_y(y)
    train_ids, val_ids = split_stations(meta["location_id"].to_numpy(), tr["val_station_fraction"], tr["seed"])
    is_val = meta["location_id"].isin(val_ids).to_numpy()
    tc = {**cfg["train"], "epochs": int(tr["epochs"])}
    mc = deepcopy(cfg["model"])
    mc["spatial_cnn"]["enabled"] = False                                   # the temporal branch only: the source cities have no morphology
    set_seed(tc["seed"])
    from networks.dual_branch_net import DualBranchNet

    model = DualBranchNet(mc)
    extra = {"derived": cfg["derived"], "lags": cfg["data"]["temporal"]["lags"], "pretrained_on": names, "scalers_sha1": file_sha1(processed / "scalers.json")}
    trainer = Trainer(model, tc, mc, str(out_dir), extra)
    loader = lambda mask, shuffle: DataLoader(WindowDataset(Xs[mask], ys[mask]), batch_size=int(tc["batch_size"]), shuffle=shuffle, num_workers=0)
    train_loader = loader(~is_val, True)
    val_loader = loader(is_val, False) if is_val.any() else None
    print(f"pretraining on {int((~is_val).sum()):,} windows of {len(train_ids)} stations, validating on {int(is_val.sum()):,} of {len(val_ids)}")
    hist = trainer.fit(train_loader, val_loader, epochs=int(tr["epochs"]), ckpt_path=str(out_dir / "pretrained_lstm.pt"), tag="[pretrain]")

    summary = {"name": name, "sources": names, "stations_train": int(len(train_ids)), "stations_val": int(len(val_ids)), "windows_train": int((~is_val).sum()),
               "windows_val": int(is_val.sum()), "top_k_stations": tr["top_k_stations"], "exclude_months": tr["exclude_months"], "best_epoch": hist["best_epoch"],
               "best_val_loss": hist["best_val_loss"], "epochs_run": hist["epochs_run"], "feature_names": cfg["derived"]["feature_names"],
               "scaled_with": str(processed / "scalers.json"), "checkpoint": "pretrained_lstm.pt",
               "next": "python run_pipeline.py --stage train --override model.spatial_cnn.enabled=false --override train.init_from=" + str(out_dir / "pretrained_lstm.pt")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"pretrained temporal branch saved to {out_dir / 'pretrained_lstm.pt'}")
    return out_dir / "pretrained_lstm.pt"

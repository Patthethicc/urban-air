"""Pair each temporal window with its station's patch and write the arrays.

Patches are stored once per station, not once per row:
    patches.npy              [S, C, H, W]   one patch per station (all stations, scaled)
    patch_index.json         {location_id: row in patches.npy}
    station_idx_<p>.npy      [N_p]          which patch each sample uses
    X_temporal_<p>.npy       [N_p, T, F]
    y_<p>.npy                [N_p, 1]
for p in train / calibration / test (nine per-partition files), plus meta_<p>.csv, partition.json
and stations.csv (location_id, x, y, partition, patch_row).
A sample's spatial input is patches[station_idx[i]], i.e. X_spatial[N, C, H, W] is only built per batch.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def stack_patches(patches: dict) -> tuple[np.ndarray, dict]:
    """{location_id: [C,H,W]} -> (array [S,C,H,W], {location_id: row})."""
    ids = sorted(patches)
    index = {int(i): row for row, i in enumerate(ids)}
    return np.stack([patches[i] for i in ids]).astype(np.float32), index


def station_index(meta: pd.DataFrame, patch_index: dict) -> np.ndarray:
    """[N] int array: meta.location_id[i] -> row in patches.npy. Raise if a station has no patch."""
    ids = meta["location_id"].to_numpy()
    missing = sorted(set(int(i) for i in ids) - set(patch_index))
    if missing:
        raise KeyError(f"no patch for stations {missing}")
    return np.array([patch_index[int(i)] for i in ids], dtype=np.int64)


def save_splits(arrays: dict, out_dir: str) -> None:
    """Write the files listed in the module docstring. Everything saved is already scaled.

    arrays = {
      'patches': [S,C,H,W], 'patch_index': {id: row}, 'partition': {name: [ids]},
      'stations': DataFrame(location_id, x, y),
      'splits': {name: {'station_idx', 'X_temporal', 'y', 'meta'}},
    }"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "patches.npy", arrays["patches"].astype(np.float32))
    (out / "patch_index.json").write_text(json.dumps({str(k): v for k, v in arrays["patch_index"].items()}, indent=2))
    (out / "partition.json").write_text(json.dumps(arrays["partition"], indent=2))

    stations = arrays["stations"].copy()
    owner = {i: name for name, ids in arrays["partition"].items() for i in ids}
    stations["partition"] = stations["location_id"].map(owner)
    stations["patch_row"] = stations["location_id"].map(arrays["patch_index"])
    stations.to_csv(out / "stations.csv", index=False)

    for name, split in arrays["splits"].items():
        np.save(out / f"station_idx_{name}.npy", split["station_idx"].astype(np.int64))
        np.save(out / f"X_temporal_{name}.npy", split["X_temporal"].astype(np.float32))
        np.save(out / f"y_{name}.npy", split["y"].astype(np.float32))
        split["meta"].to_csv(out / f"meta_{name}.csv", index=False)

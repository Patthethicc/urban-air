"""Join spatial patches and temporal windows, split chronologically, save .npy files."""
import numpy as np


def align(X_temporal, y, meta, patches: dict):
    """Return X_spatial [N, C, H, W] so row i matches X_temporal[i] / y[i]
    (same station, same timestamp)."""
    # TODO
    raise NotImplementedError


def chronological_split(meta, cfg: dict) -> dict[str, np.ndarray]:
    """Return {'train': idx, 'val': idx, 'test': idx}.

    Split by time (and by station for held-out stations), NOT randomly:
    overlapping windows would leak across a random split.
    Drop/purge windows that straddle a split boundary.
    """
    # TODO
    raise NotImplementedError


def save_splits(arrays: dict, out_dir: str) -> None:
    """Write the nine files: {X_spatial,X_temporal,y}_{train,val,test}.npy"""
    # TODO
    raise NotImplementedError

"""Predict a whole partition and return a DataFrame in ug/m3."""
from pathlib import Path

import pandas as pd

from preprocessing.scalers import Scalers
from training.dataset import AirQualityDataset, load_meta
from training.trainer import make_loader, resolve_device


def predict_partition(model, cfg: dict, partition: str, batch_size: int = 1024) -> pd.DataFrame:
    """Columns: location_id, timestamp, y_true, y_pred (original units, scalers loaded from processed_dir)."""
    import numpy as np
    import torch

    processed = cfg["data"]["paths"]["processed_dir"]
    scalers = Scalers.load(str(Path(processed) / "scalers.json"))
    device = resolve_device(cfg["train"]["device"])
    model = model.to(device).eval()

    loader = make_loader(AirQualityDataset(processed, partition), cfg["train"], shuffle=False, batch_size=batch_size)
    preds, trues = [], []
    with torch.no_grad():
        for spatial, temporal, target in loader:
            preds.append(model(spatial.to(device), temporal.to(device)).cpu().numpy())
            trues.append(target.numpy())
    out = load_meta(processed, partition)
    out["y_true"] = scalers.inverse_y(np.concatenate(trues)).ravel()
    out["y_pred"] = scalers.inverse_y(np.concatenate(preds)).ravel()
    return out

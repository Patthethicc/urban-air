"""Evaluation metrics (in original units; inverse-scale y first)."""
import numpy as np


def _arrays(y_true, y_pred):
    return np.asarray(y_true, dtype=np.float64).ravel(), np.asarray(y_pred, dtype=np.float64).ravel()


def rmse(y_true, y_pred) -> float:
    o, p = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((p - o) ** 2)))


def mae(y_true, y_pred) -> float:
    o, p = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(p - o)))


def index_of_agreement(y_true, y_pred) -> float:
    """Willmott's IA (1 = perfect)."""
    o, p = _arrays(y_true, y_pred)
    denom = np.sum((np.abs(p - o.mean()) + np.abs(o - o.mean())) ** 2)
    return float(1.0 - np.sum((p - o) ** 2) / denom) if denom > 0 else float("nan")


def pearson_r(y_true, y_pred) -> float:
    o, p = _arrays(y_true, y_pred)
    if o.std() == 0 or p.std() == 0:
        return float("nan")
    return float(np.corrcoef(o, p)[0, 1])


def r2(y_true, y_pred) -> float:
    o, p = _arrays(y_true, y_pred)
    ss_tot = np.sum((o - o.mean()) ** 2)
    return float(1.0 - np.sum((o - p) ** 2) / ss_tot) if ss_tot > 0 else float("nan")


def compute_all(y_true, y_pred) -> dict[str, float]:
    """{'rmse','mae','ia','pearson_r','r2'} -- keys match metrics_summary.json."""
    return {
        "rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred),
        "ia": index_of_agreement(y_true, y_pred), "pearson_r": pearson_r(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }

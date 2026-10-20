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


def bias(y_true, y_pred) -> float:
    """Mean error, predicted - observed: positive = over-prediction, negative = under-prediction."""
    o, p = _arrays(y_true, y_pred)
    return float(np.mean(p - o))


def fraction_within(y_true, y_pred, tolerance: float) -> float:
    """Share of predictions that are within +/- tolerance of the observation (same unit as the data)."""
    o, p = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(p - o) <= tolerance))


def band_index(values, uppers) -> np.ndarray:
    """Category of every value for ascending upper bounds (a value equal to a bound belongs to the lower band),
    the same rule as the AQI maps."""
    return np.digitize(np.asarray(values, dtype=np.float64).ravel(), list(uppers), right=True)


def band_agreement(y_true, y_pred, uppers) -> dict:
    """Classification-style accuracy of the predicted band: exactly the observed band, within one band, and the
    score of the trivial rule 'always say the most common observed band' (the number to beat)."""
    o, p = band_index(y_true, uppers), band_index(y_pred, uppers)
    counts = np.bincount(o, minlength=len(list(uppers)) + 1)
    return {"exact": float(np.mean(o == p)), "within_one": float(np.mean(np.abs(o - p) <= 1)),
            "majority_baseline": float(counts.max() / len(o)), "majority_band": int(counts.argmax())}


def compute_all(y_true, y_pred) -> dict[str, float]:
    """{'rmse','mae','ia','pearson_r','r2'} -- keys match metrics_summary.json."""
    return {
        "rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred),
        "ia": index_of_agreement(y_true, y_pred), "pearson_r": pearson_r(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }

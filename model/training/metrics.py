"""Evaluation metrics (in original units; inverse-scale y first)."""
import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError


def index_of_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Willmott's IA."""
    raise NotImplementedError


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """{'rmse','mae','ia','pearson_r'} -- keys match metrics_summary.json."""
    # TODO
    raise NotImplementedError

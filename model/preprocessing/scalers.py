"""Three scalers (spatial, temporal, target). Fit on TRAIN only, apply to val/test."""
import numpy as np


class Scalers:
    def fit(self, X_spatial_train, X_temporal_train, y_train) -> "Scalers":
        # TODO: per-channel for spatial, per-feature for temporal, scalar for y
        raise NotImplementedError

    def transform(self, X_spatial, X_temporal, y):
        # TODO
        raise NotImplementedError

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Back to original units for metrics and plots."""
        # TODO
        raise NotImplementedError

    def save(self, path: str) -> None:
        # TODO
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "Scalers":
        # TODO
        raise NotImplementedError

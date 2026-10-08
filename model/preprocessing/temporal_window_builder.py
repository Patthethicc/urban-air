"""Sliding windows over each station's time series."""
import numpy as np
import pandas as pd


def build_windows(df: pd.DataFrame, cfg: dict):
    """Windows must never cross station boundaries.

    Returns
      X_temporal : [N, T, F]
      y          : [N, 1]
      meta       : DataFrame with station_id and target timestamp per row
                   (the aligner uses this to fetch the matching spatial patch)
    """
    # TODO
    raise NotImplementedError

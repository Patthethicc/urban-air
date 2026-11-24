import numpy as np
import pandas as pd

from preprocessing.clean_loader import _fill_short_gaps


def test_short_gaps_are_interpolated_and_long_gaps_stay_empty():
    values = [1.0] + [np.nan] * 3 + [5.0] + [np.nan] * 30 + [40.0]
    out = _fill_short_gaps(pd.Series(values), max_gap=24)
    assert out.iloc[1:4].tolist() == [2.0, 3.0, 4.0]          # 3-hour gap bridged linearly
    assert out.iloc[5:35].isna().all()                          # 30-hour gap: NOT partly filled
    assert out.iloc[0] == 1.0 and out.iloc[-1] == 40.0


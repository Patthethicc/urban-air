import numpy as np
import pytest

from training import metrics as m


def test_known_values():
    o, p = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])
    assert m.rmse(o, p) == pytest.approx(1.0)
    assert m.mae(o, p) == pytest.approx(1.0)
    assert m.pearson_r(o, p) == pytest.approx(1.0)
    assert m.index_of_agreement(o, p) == pytest.approx(1 - 3 / 11)   # worked by hand
    assert m.r2(o, p) == pytest.approx(1 - 3 / 2)


def test_perfect_prediction():
    o = np.array([4.0, 8.0, 15.0, 16.0])
    out = m.compute_all(o, o)
    assert out["rmse"] == 0 and out["ia"] == pytest.approx(1.0) and out["r2"] == pytest.approx(1.0)


def test_compute_all_keys_match_the_summary():
    assert set(m.compute_all([1, 2, 3], [1, 2, 4])) == {"rmse", "mae", "ia", "pearson_r", "r2"}

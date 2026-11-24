import numpy as np
import pytest

from preprocessing.scalers import Scalers


def _fit(patches, y, pm25=(0, 1), roll_std=()):
    return Scalers().fit(patches, y, pm25, roll_std)


def test_scalers_fit_on_train_only():
    rng = np.random.default_rng(0)
    y_train, y_all = rng.normal(20, 5, (500, 1)), np.concatenate([rng.normal(20, 5, (500, 1)), rng.normal(80, 5, (500, 1))])
    s = _fit(rng.normal(size=(4, 2, 3, 3)), y_train)
    assert s.y_mean == pytest.approx(y_train.mean()) and s.y_mean != pytest.approx(y_all.mean(), abs=1)


def test_band_scaler_uses_one_patch_per_station():
    """Rows are irrelevant to the band statistics: only the stations' patches count."""
    patches = np.stack([np.full((2, 3, 3), v) for v in (1.0, 3.0)])
    s = _fit(patches, np.zeros((1000, 1)) + np.arange(1000)[:, None])
    assert s.band_mean.tolist() == [2.0, 2.0]
    assert np.allclose(s.transform_patches(patches).mean(axis=(0, 2, 3)), 0)


def test_only_pm25_columns_are_scaled():
    s = _fit(np.ones((2, 1, 1, 1)), np.array([[10.0], [30.0]]), pm25=(0,))
    X = np.array([[[20.0, 0.5], [30.0, -0.5]]], dtype=np.float32)    # [N=1, T=2, F=2]; F1 = sin_hour
    out = s.transform_temporal(X)
    assert out[0, :, 1].tolist() == [0.5, -0.5]
    assert out[0, 0, 0] == pytest.approx(0.0) and out[0, 1, 0] == pytest.approx(1.0)


def test_rolling_std_is_divided_not_shifted():
    s = _fit(np.ones((2, 1, 1, 1)), np.array([[10.0], [30.0]]), pm25=(0,), roll_std=(0,))
    out = s.transform_temporal(np.array([[[20.0]]], dtype=np.float32))
    assert out[0, 0, 0] == pytest.approx(20.0 / 10.0)


def test_inverse_y_roundtrip_and_save_load(tmp_path):
    s = _fit(np.random.rand(3, 2, 2, 2), np.random.rand(50, 1) * 40)
    y = np.array([[5.0], [17.5]])
    assert np.allclose(s.inverse_y(s.transform_y(y)), y, atol=1e-4)
    s.save(str(tmp_path / "s.json"))
    assert np.allclose(Scalers.load(str(tmp_path / "s.json")).inverse_y(s.transform_y(y)), y, atol=1e-4)

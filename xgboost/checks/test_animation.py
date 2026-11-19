"""Checks for the XGBoost day animation and the 24-hour snapshot setting. Run: python -m pytest checks/test_animation.py"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_default_snapshot_hours_are_the_whole_day():
    import data
    assert data.SNAPSHOT_HOURS == list(range(24))


def test_snapshot_hours_can_be_set_and_a_typo_is_refused(monkeypatch):
    import importlib

    import data
    monkeypatch.setenv('XGB_SNAPSHOT_HOURS', '8,12,17,21')
    assert importlib.reload(data).SNAPSHOT_HOURS == [8, 12, 17, 21]
    monkeypatch.setenv('XGB_SNAPSHOT_HOURS', '8,25')
    with pytest.raises(ValueError):
        importlib.reload(data)
    monkeypatch.delenv('XGB_SNAPSHOT_HOURS')
    importlib.reload(data)


def test_gif_and_slideshow_from_frames(tmp_path):
    from PIL import Image

    from maps import animation
    pngs = []
    for i in range(3):
        p = tmp_path / f'frame_{i}.png'
        Image.fromarray(np.full((40 + i, 60, 3), 50 * i, 'uint8')).save(p)     # sizes differ by a pixel, as tight boxes do
        pngs.append(str(p))
    gif = animation.write_gif(pngs, str(tmp_path / 'a.gif'))
    img = Image.open(gif)
    assert img.n_frames == 3 and img.info['loop'] == 0
    page = animation.write_slideshow([('08:00', pngs[0]), ('09:00', pngs[1])], str(tmp_path / 's.html'), 'title')
    html = open(page, encoding='utf-8').read()
    assert 'frame_0.png' in html and '09:00' in html


def test_one_hour_is_not_an_animation(tmp_path):
    from maps import animation
    frames = {pd.Timestamp('2026-04-11 08:00'): pd.DataFrame({'prediction': [1.0]})}
    assert animation.render_day_animation(frames, str(tmp_path), '20260411', 0, 25) == []

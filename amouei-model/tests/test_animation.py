"""The day animation: hourly PNGs -> GIF, MP4 (optional) and a slideshow page."""
import numpy as np
import pytest
from PIL import Image

from output.animation import frame_paths, render_animation, settings


def _frames(tmp_path, labels, size=(60, 40)):
    for i, lab in enumerate(labels):
        arr = np.full((size[1], size[0], 3), i * 10, dtype="uint8")
        Image.fromarray(arr).save(tmp_path / f"continuous_{lab.replace(':', '')}.png")


def test_settings_fall_back_to_the_defaults_for_an_old_config():
    s = settings({"output": {}})
    assert s["enabled"] is True and s["fps"] == 2 and "gif" in s["formats"]
    assert settings({"output": {"animation": {"fps": 5}}})["fps"] == 5


def test_frames_are_found_in_hour_order_and_missing_ones_are_skipped(tmp_path):
    labels = ["00:00", "01:00", "02:00"]
    _frames(tmp_path, labels[:2])
    got = frame_paths(tmp_path, "continuous_{h}", labels)
    assert [p.name for p in got] == ["continuous_0000.png", "continuous_0100.png"]


def test_a_day_becomes_a_looping_gif_and_a_slideshow(tmp_path):
    labels = [f"{h:02d}:00" for h in range(5)]
    _frames(tmp_path, labels)
    cfg = {"output": {"animation": {"formats": ["gif", "html"]}}}
    written = render_animation(tmp_path, "continuous_{h}", "continuous", labels, "PM2.5 prediction, 2025-07-01", cfg)
    names = sorted(p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in written)
    assert names == ["continuous_animation.gif", "continuous_slideshow.html"]
    gif = Image.open(tmp_path / "continuous_animation.gif")
    assert gif.n_frames == 5 and gif.info["loop"] == 0
    html = (tmp_path / "continuous_slideshow.html").read_text(encoding="utf-8")
    assert "continuous_0000.png" in html and "continuous_0400.png" in html and "00:00" in html


def test_one_frame_or_a_disabled_setting_writes_nothing(tmp_path):
    _frames(tmp_path, ["08:00"])
    assert render_animation(tmp_path, "continuous_{h}", "continuous", ["08:00"], "t", {"output": {}}) == []
    _frames(tmp_path, ["08:00", "09:00"])
    off = {"output": {"animation": {"enabled": False}}}
    assert render_animation(tmp_path, "continuous_{h}", "continuous", ["08:00", "09:00"], "t", off) == []


def test_mp4_is_skipped_with_a_notice_when_imageio_is_missing(tmp_path, monkeypatch, capsys):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("imageio"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    labels = ["08:00", "09:00"]
    _frames(tmp_path, labels)
    written = render_animation(tmp_path, "continuous_{h}", "continuous", labels, "t", {"output": {"animation": {"formats": ["mp4"]}}})
    assert written == [] and "no .mp4" in capsys.readouterr().out


def test_frames_of_different_sizes_still_make_a_gif(tmp_path):
    Image.fromarray(np.zeros((40, 60, 3), "uint8")).save(tmp_path / "continuous_0800.png")
    Image.fromarray(np.full((41, 61, 3), 200, "uint8")).save(tmp_path / "continuous_0900.png")    # differs: identical frames are merged by GIF
    written = render_animation(tmp_path, "continuous_{h}", "continuous", ["08:00", "09:00"], "t", {"output": {"animation": {"formats": ["gif"]}}})
    assert len(written) == 1 and Image.open(written[0]).n_frames == 2

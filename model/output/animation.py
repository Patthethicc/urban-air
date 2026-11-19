"""A day of hourly maps as an animation and a slideshow (Output.txt: how the air quality moves over a day).

The hourly PNGs are already drawn by output.aqi_maps.render_day (one colour scale for the whole day), so the animation is only a
re-assembly of those frames: nothing is predicted or drawn again. Outputs, next to the hourly maps:
    <style>_animation.gif    looping animation, always (Pillow is a matplotlib dependency)
    <style>_animation.mp4    only when `imageio` + `imageio-ffmpeg` are installed (smaller, plays in slides); skipped with a notice otherwise
    <style>_slideshow.html   one self-contained page: play / pause, a slider and the hour, frames loaded from the PNGs beside it
Settings: output.animation (enabled, fps, formats).
"""
from __future__ import annotations

from pathlib import Path

DEFAULTS = {"enabled": True, "fps": 2, "formats": ["gif", "mp4", "html"], "hold_last_s": 1.5}


def settings(cfg: dict) -> dict:
    """output.animation merged over the defaults, so an old config without the block still works."""
    return {**DEFAULTS, **(cfg.get("output", {}).get("animation") or {})}


def frame_paths(maps_dir: Path, hourly_stem: str, labels: list[str]) -> list[Path]:
    """The hourly PNGs of one style in hour order (hourly_stem looks like 'continuous_{h}', h = HHMM without the colon)."""
    paths = [Path(maps_dir) / f"{hourly_stem.format(h=label.replace(':', ''))}.png" for label in labels]
    return [p for p in paths if p.exists()]


def write_gif(frames: list[Path], path: Path, fps: float, hold_last_s: float) -> Path:
    """Looping GIF. The last frame is held a little longer so the loop restart is visible."""
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in frames]
    size = images[0].size
    images = [im if im.size == size else im.resize(size) for im in images]       # every frame the same size, or the GIF is refused
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=256) for im in images]
    ms = int(round(1000 / max(fps, 0.1)))
    durations = [ms] * (len(pal) - 1) + [ms + int(hold_last_s * 1000)]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=False)
    return path


def write_mp4(frames: list[Path], path: Path, fps: float) -> Path | None:
    """MP4 through imageio-ffmpeg. Returns None (and prints why) when the libraries are missing."""
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401  (only to fail early when the ffmpeg binary package is missing)
    except ImportError:
        print("animation: no .mp4 (pip install imageio imageio-ffmpeg to get it); the GIF and the slideshow are written")
        return None
    import numpy as np
    from PIL import Image

    first = Image.open(frames[0]).convert("RGB")
    w, h = (first.width // 2 * 2, first.height // 2 * 2)                          # H.264 needs even sizes
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for p in frames:
            writer.append_data(np.asarray(Image.open(p).convert("RGB").resize((w, h))))
    return path


def write_slideshow(frames: list[Path], labels: list[str], path: Path, title: str, fps: float) -> Path:
    """A single HTML page that plays the frames (relative links: keep the page in the maps folder)."""
    names = [p.name for p in frames]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>body{{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#17212B}}
img{{max-width:100%;height:auto;display:block;margin:8px auto}}.bar{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
input[type=range]{{flex:1;min-width:160px}}button{{padding:6px 14px}}</style></head><body><h1 style="font-size:1.1rem">{title}</h1>
<div class="bar"><button id="play">Pause</button><input id="slider" type="range" min="0" max="{len(names) - 1}" value="0"><b id="hour"></b></div>
<img id="frame" alt="map"><script>
const files={names!r}, labels={labels[:len(names)]!r}, delay=Math.round(1000/{fps});let i=0,timer=null;
const img=document.getElementById('frame'),slider=document.getElementById('slider'),hour=document.getElementById('hour'),btn=document.getElementById('play');
function show(k){{i=(k+files.length)%files.length;img.src=files[i];slider.value=i;hour.textContent=labels[i];}}
function play(){{timer=setInterval(()=>show(i+1),delay);btn.textContent='Pause';}}
function stop(){{clearInterval(timer);timer=null;btn.textContent='Play';}}
btn.onclick=()=>timer?stop():play();slider.oninput=()=>{{stop();show(+slider.value);}};show(0);play();
</script></body></html>"""
    path.write_text(page, encoding="utf-8")
    return path


def render_animation(maps_dir, hourly_stem: str, out_stem: str, labels: list[str], title: str, cfg: dict) -> list[str]:
    """Everything configured for one style of one day. Needs at least two hourly frames; returns the files written."""
    s = settings(cfg)
    if not s["enabled"]:
        return []
    maps_dir = Path(maps_dir)
    frames = frame_paths(maps_dir, hourly_stem, labels)
    if len(frames) < 2:
        return []
    kept = [lab for lab, p in zip(labels, [maps_dir / f"{hourly_stem.format(h=x.replace(':', ''))}.png" for x in labels]) if p.exists()]
    fps, written = float(s["fps"]), []
    if "gif" in s["formats"]:
        written.append(write_gif(frames, maps_dir / f"{out_stem}_animation.gif", fps, float(s["hold_last_s"])))
    if "mp4" in s["formats"]:
        mp4 = write_mp4(frames, maps_dir / f"{out_stem}_animation.mp4", fps)
        if mp4:
            written.append(mp4)
    if "html" in s["formats"]:
        written.append(write_slideshow(frames, kept, maps_dir / f"{out_stem}_slideshow.html", title, fps))
    return [str(p) for p in written]

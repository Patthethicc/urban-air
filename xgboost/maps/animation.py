"""A day of hourly XGBoost maps as an animation (GIF) and a slideshow page.

Why frames are drawn again here: the per-hour maps written while the day is being predicted each use their own colour limits (the
limits of the whole day are only known at the end). Colours in an animation must mean the same thing in every frame, so each hour is
drawn once more on the shared limits `vmin`, `vmax`, into <day folder>/frames/, and those frames are assembled.
Nothing is predicted again: the frames come from the cells already in memory.

Outputs (in the day folder):
    animation_<day>.gif       looping animation (Pillow only)
    slideshow_<day>.html      play / pause + slider page that loads frames/*.png (keep it next to the frames folder)
"""
import os

import pandas as pd

from maps import surfaces


def render_frames(frames, frames_dir, vmin, vmax):
    """Draw every hour on one shared scale. `frames` = {timestamp: cells DataFrame}. Returns [(label, png path)] in hour order."""
    os.makedirs(frames_dir, exist_ok=True)
    out = []
    for ts in sorted(frames):                                   # hour order = animation order
        cells = frames[ts]
        if 'prediction_usable' in cells.columns:
            cells = cells[cells['prediction_usable'].astype(bool)]    # only cells that have a usable prediction
        if cells.empty:
            continue
        stem = f'frame_{pd.Timestamp(ts):%H%M}'
        # _try_continuous_map writes <stem>_continuous.png and returns its path
        path = surfaces._try_continuous_map(cells, ts, frames_dir, stem, True, vmin=vmin, vmax=vmax)
        if path:
            out.append((f'{pd.Timestamp(ts):%H:%M}', path))
    return out


def write_gif(png_paths, gif_path, fps=2, hold_last_s=1.5):
    """Looping GIF; the last frame is held a little longer so the restart is visible; all frames are forced to one size."""
    from PIL import Image
    images = [Image.open(p).convert('RGB') for p in png_paths]
    size = images[0].size
    images = [im if im.size == size else im.resize(size) for im in images]   # tight bounding boxes can differ by a pixel; GIF needs one size
    pal = [im.convert('P', palette=Image.ADAPTIVE, colors=256) for im in images]
    ms = int(round(1000 / max(fps, 0.1)))                                    # milliseconds per frame
    durations = [ms] * (len(pal) - 1) + [ms + int(hold_last_s * 1000)]
    pal[0].save(gif_path, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=False)
    return gif_path


def write_slideshow(items, html_path, title, fps=2):
    """One HTML page: play / pause, a slider and the hour. `items` = [(label, png path)]; links are relative to the page."""
    base = os.path.dirname(html_path)
    files = [os.path.relpath(p, base).replace(os.sep, '/') for _, p in items]
    labels = [lab for lab, _ in items]
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{title}</title>'
        '<style>body{font-family:system-ui,sans-serif;margin:16px}img{max-width:100%;height:auto;display:block;margin:8px auto}'
        '.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}input[type=range]{flex:1;min-width:160px}</style></head><body>'
        f'<h1 style="font-size:1.1rem">{title}</h1>'
        f'<div class="bar"><button id="play">Pause</button><input id="slider" type="range" min="0" max="{len(files) - 1}" value="0">'
        '<b id="hour"></b></div><img id="frame" alt="map"><script>'
        f'const files={files!r},labels={labels!r},delay=Math.round(1000/{fps});let i=0,timer=null;'
        "const img=document.getElementById('frame'),slider=document.getElementById('slider'),"
        "hour=document.getElementById('hour'),btn=document.getElementById('play');"
        'function show(k){i=(k+files.length)%files.length;img.src=files[i];slider.value=i;hour.textContent=labels[i];}'
        "function play(){timer=setInterval(()=>show(i+1),delay);btn.textContent='Pause';}"
        "function stop(){clearInterval(timer);timer=null;btn.textContent='Play';}"
        'btn.onclick=()=>timer?stop():play();slider.oninput=()=>{stop();show(+slider.value);};show(0);play();'
        '</script></body></html>')
    with open(html_path, 'w', encoding='utf-8') as fh:
        fh.write(page)
    return html_path


def render_day_animation(frames, day_dir, day, vmin, vmax, fps=2):
    """Frames on one scale -> GIF + slideshow. Needs two or more hours (one hour is not an animation). Returns the files written."""
    if len(frames) < 2:
        return []
    items = render_frames(frames, os.path.join(day_dir, 'frames'), vmin, vmax)
    if len(items) < 2:
        return []
    gif = write_gif([p for _, p in items], os.path.join(day_dir, f'animation_{day}.gif'), fps=fps)
    page = write_slideshow(items, os.path.join(day_dir, f'slideshow_{day}.html'),
                           f'PM2.5 prediction, {day[:4]}-{day[4:6]}-{day[6:]} (continuous scale {vmin:.1f}-{vmax:.1f} ug/m3)', fps=fps)
    print(f'  animation -> {os.path.relpath(gif, data_root())}  ({len(items)} frames)')
    return [gif, page]


def data_root():
    """The xgboost folder, for tidy relative paths in the printout."""
    import data
    return data.ROOT

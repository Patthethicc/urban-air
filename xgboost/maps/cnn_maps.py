"""CNN-LSTM-style prediction maps for the XGBoost rasters.

The CNN-LSTM figures are produced by model/output/aqi_maps.py + model/output/cartography.py:
a light CARTO basemap, a #d9d9d9 "no data" fill clipped to the NCR, city outlines and names,
stations as white dots, kilometre axes, a two-tone scale bar, a north arrow, a viridis colourbar
and a footer with the attribution. This module reproduces that furniture for our GeoTIFFs.

Display only: the GeoTIFFs keep every cell, nothing here changes a number.

    python maps/cnn_maps.py --days 20250830                # 2x2 panel per date
    python maps/cnn_maps.py --days 20250830 --single       # one map per hour

Tiles are cached under data/basemap_cache/, so a later run works offline.
"""
import argparse
import io
import math
import os
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from pyproj import Transformer  # noqa: E402

INK, INK2, SURFACE, NO_DATA = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d9d9"
HOURS = ("0800", "1200", "1700", "2100")
CLOCK = {"0800": "08:00", "1200": "12:00", "1700": "17:00", "2100": "21:00"}
LABEL_NUDGE = {"Navotas": (-900, -500, "right"), "Malabon": (500, 500, "left"),
               "San Juan": (-300, 450, "right"), "Pateros": (900, -300, "left")}
# OSM Mapnik tiles, greyed and blended toward white below: the same visual effect as CartoDB.Positron,
# with no API key. Set XGB_TILE_URL to use another provider.
TILE_URL = os.environ.get("XGB_TILE_URL",
                        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/"
                        "MapServer/tile/{z}/{y}/{x}")   # light, roads and labels, no API key
TILE_LIGHTEN = float(os.environ.get("XGB_TILE_LIGHTEN", "0.15"))   # 0 = raw tiles, 1 = pure white
ATTRIB = "Basemap: Esri, HERE, Garmin, OpenStreetMap contributors.  Boundaries: GADM 4.1."
CACHE = os.path.join("data", "basemap_cache")
# the view the CNN-LSTM figures use (their boundary file stops at ~1594 km northing)
CNN_VIEW = (277000.0, 1594000.0, 298500.0, 1632500.0)


def _merc(lon, lat):
    x = (lon + 180.0) / 360.0 * 2 ** 20
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * 2 ** 20
    return x, y


def _tile(z, x, y):
    p = os.path.join(CACHE, "esri_%d_%d_%d.png" % (z, x, y))
    if os.path.exists(p):
        return open(p, "rb").read()
    os.makedirs(CACHE, exist_ok=True)
    req = urllib.request.Request(TILE_URL.format(z=z, x=x, y=y), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        b = r.read()
    open(p, "wb").write(b)
    return b


def basemap(bounds, z=13):
    """Basemap RGB + extent in EPSG:32651 for `bounds` = (left, bottom, right, top) in UTM metres."""
    from PIL import Image
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    to_ll = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
    lon0, lat1 = to_ll.transform(bounds[0], bounds[3])
    lon1, lat0 = to_ll.transform(bounds[2], bounds[1])
    x0, y0 = _merc(lon0, lat1)
    x1, y1 = _merc(lon1, lat0)
    scale = 2 ** (20 - z)                     # _merc() returns world pixels at zoom 20
    tx0, tx1 = int(x0 // scale), int(x1 // scale)
    ty0, ty1 = int(y0 // scale), int(y1 // scale)
    w, h = (tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256
    mosaic = Image.new("RGB", (w, h), (255, 255, 255))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            try:
                im = Image.open(io.BytesIO(_tile(z, tx, ty))).convert("RGB")
            except Exception:
                continue
            mosaic.paste(im, ((tx - tx0) * 256, (ty - ty0) * 256))
    left = (tx0 / 2 ** z) * 360.0 - 180.0
    right = ((tx1 + 1) / 2 ** z) * 360.0 - 180.0
    top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty0 / 2 ** z))))
    bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty1 + 1) / 2 ** z))))
    arr = np.asarray(mosaic).astype("float32")
    grey = arr.mean(axis=2, keepdims=True)                       # desaturate
    arr = 255.0 - (255.0 - grey) * (1.0 - TILE_LIGHTEN)          # blend toward white
    src = np.clip(np.repeat(arr, 3, axis=2), 0, 255).astype("uint8")
    src_transform = from_bounds(left, bottom, right, top, src.shape[1], src.shape[0])
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    ulx, uly = to_utm.transform(left, top)
    lrx, lry = to_utm.transform(right, bottom)
    dw = max(1, int((lrx - ulx) / 10))
    dh = max(1, int((uly - lry) / 10))
    dst_transform = from_bounds(ulx, lry, lrx, uly, dw, dh)
    dst = np.zeros((3, dh, dw), dtype="uint8")
    for b in range(3):
        reproject(src[:, :, b], dst[b], src_transform=src_transform, src_crs="EPSG:4326",
                  dst_transform=dst_transform, dst_crs="EPSG:32651", resampling=Resampling.bilinear)
    extent = (dst_transform.c, dst_transform.c + dst_transform.a * dw,
              dst_transform.f + dst_transform.e * dh, dst_transform.f)
    return np.transpose(dst, (1, 2, 0)), extent


def nice_length(target_m):
    if target_m <= 0:
        return 1.0
    exp = 10 ** np.floor(np.log10(target_m))
    return float(max(m * exp for m in (1, 2, 5) if m * exp <= target_m))


def scale_bar(ax, zorder=8):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    length = nice_length(abs(x1 - x0) * 0.25)
    left = min(x0, x1) + abs(x1 - x0) * 0.05
    base = min(y0, y1) + abs(y1 - y0) * 0.10
    h = abs(y1 - y0) * 0.008
    for i, face in enumerate((INK, SURFACE)):
        ax.add_patch(Rectangle((left + i * length / 2, base), length / 2, h, facecolor=face,
                               edgecolor=INK, linewidth=0.7, zorder=zorder))
    unit, div = ("km", 1000.0) if length >= 1000 else ("m", 1.0)
    halo = [pe.withStroke(linewidth=2, foreground=SURFACE)]
    for frac, text in ((0, "0"), (0.5, f"{length / 2 / div:g}"), (1, f"{length / div:g} {unit}")):
        ax.text(left + frac * length, base + h * 1.8, text, fontsize=6.5, ha="center", va="bottom",
                color=INK, zorder=zorder, path_effects=halo)


def north_arrow(ax, zorder=8):
    ax.annotate("", xy=(0.92, 0.16), xytext=(0.92, 0.07), xycoords="axes fraction", zorder=zorder,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, shrinkA=0, shrinkB=0))
    ax.text(0.92, 0.17, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
            fontweight="bold", color=INK, zorder=zorder,
            path_effects=[pe.withStroke(linewidth=2, foreground=SURFACE)])


def km_axes(ax):
    to_km = FuncFormatter(lambda v, _: f"{v / 1000:g}")
    ax.xaxis.set_major_formatter(to_km)
    ax.yaxis.set_major_formatter(to_km)
    ax.set_xlabel("easting (km)", fontsize=7, color=INK2)
    ax.set_ylabel("northing (km)", fontsize=7, color=INK2)
    ax.tick_params(colors=INK2, labelsize=7)


def draw_cities(ax, boundary, zorder=6, labels=True):
    if boundary is None:
        return
    boundary.boundary.plot(ax=ax, color=INK2, linewidth=0.7, zorder=zorder)
    if not labels:
        return
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    for name, pt in zip(boundary["city"], boundary.geometry.representative_point()):
        dx, dy, ha = LABEL_NUDGE.get(name, (0, 0, "center"))
        if name and min(x0, x1) <= pt.x <= max(x0, x1) and min(y0, y1) <= pt.y <= max(y0, y1):
            ax.text(pt.x + dx, pt.y + dy, name, fontsize=6.5, color=INK, ha=ha, va="center",
                    zorder=zorder + 1, path_effects=[pe.withStroke(linewidth=2, foreground=SURFACE)])


def load_boundary(path):
    if not path:
        return None
    import geopandas as gpd
    gdf = gpd.read_file(path).to_crs("EPSG:32651")
    if "city" not in gdf.columns:
        gdf["city"] = ""
    return gdf


def station_xy(csv_path):
    import csv as _csv
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    xs, ys = [], []
    with open(csv_path, encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            if row.get("latitude") and row.get("longitude"):
                x, y = tr.transform(float(row["longitude"]), float(row["latitude"]))
                xs.append(x)
                ys.append(y)
    return np.array(xs), np.array(ys)


def read_band1(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.transform


def clip_to(vals, transform, boundary):
    if boundary is None:
        return vals
    from rasterio.features import geometry_mask
    inside = geometry_mask(list(boundary.geometry), out_shape=vals.shape, transform=transform,
                           invert=True, all_touched=True)
    return np.where(inside, vals, np.nan)


def draw_panel(ax, tif, vmin, vmax, pts=None, boundary=None, base=True, alpha=0.72,
               label_cities=True, view_window=None):
    vals, transform = read_band1(tif)
    vals = clip_to(vals, transform, boundary)
    extent = (transform.c, transform.c + transform.a * vals.shape[1],
              transform.f + transform.e * vals.shape[0], transform.f)
    if view_window is not None:
        view = view_window
    elif boundary is not None:
        view = boundary.total_bounds
    else:
        view = (extent[0], extent[2], extent[1], extent[3])
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])
    ax.set_facecolor(SURFACE)
    if base:
        try:
            img, ext = basemap((view[0], view[1], view[2], view[3]))
            ax.imshow(img, extent=ext, zorder=0, interpolation="bilinear")
        except Exception as exc:                       # offline and no cache: plain paper
            print("  basemap unavailable (%s)" % exc)
            base = False
    if boundary is not None:
        boundary.plot(ax=ax, facecolor=NO_DATA, edgecolor="none", zorder=1)
    im = ax.imshow(np.ma.masked_invalid(vals), extent=extent, origin="upper", cmap="viridis",
                   vmin=vmin, vmax=vmax, interpolation="nearest", zorder=2, alpha=alpha if base else 1.0)
    if pts is not None:
        ax.scatter(pts[0], pts[1], s=14, c=SURFACE, edgecolors=INK, linewidths=0.8, zorder=5)
    draw_cities(ax, boundary, labels=label_cities)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    km_axes(ax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    scale_bar(ax)
    north_arrow(ax)
    return im


def render_snapshot(tif, path, title, vmin, vmax, pts=None, boundary=None, base=True, view_window=None):
    fig, ax = plt.subplots(figsize=(6.6, 7.4), facecolor=SURFACE)
    im = draw_panel(ax, tif, vmin, vmax, pts, boundary, base, view_window=view_window)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("PM$_{2.5}$ (µg/m$^3$)", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    fig.legend(handles=[Patch(facecolor=NO_DATA, edgecolor=INK2, linewidth=0.5, label="No data")],
               loc="lower right", bbox_to_anchor=(0.98, 0.012), ncol=1, fontsize=7, frameon=False,
               labelcolor=INK2)
    fig.text(0.012, 0.012, ATTRIB, fontsize=6, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def render_panels(day, folder, path, vmin, vmax, hours=HOURS, pts=None, boundary=None, base=True,
                  view_window=None):
    tifs = [os.path.join(folder, "%sT%s.tif" % (day, h)) for h in hours]
    missing = [p for p in tifs if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing rasters: %s" % ", ".join(missing))
    n = len(tifs)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.9 * ncols, 6.9 * nrows),
                             facecolor=SURFACE, squeeze=False, layout="constrained")
    im = None
    for ax, tif, hour in zip(axes.ravel(), tifs, hours):
        im = draw_panel(ax, tif, vmin, vmax, pts, boundary, base, view_window=view_window)
        ax.set_title(CLOCK[hour], fontsize=10, color=INK)
    for ax in axes.ravel()[len(tifs):]:
        ax.axis("off")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cb.set_label("PM$_{2.5}$ (µg/m$^3$)", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    fig.legend(handles=[Patch(facecolor=NO_DATA, edgecolor=INK2, linewidth=0.5, label="No data")],
               loc="lower right", bbox_to_anchor=(0.99, 0.005), ncol=1, fontsize=7, frameon=False,
               labelcolor=INK2)
    fig.text(0.006, 0.004, ATTRIB, fontsize=5.5, color=INK2, ha="left", va="bottom")
    try:
        fig.get_layout_engine().set(rect=(0, 0.028, 1, 1))
    except Exception:
        pass
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", default="outputs/surfaces10m")
    ap.add_argument("--days", nargs="+", default=["20250830"])
    ap.add_argument("--hours", nargs="+", default=list(HOURS))
    ap.add_argument("--stations", default="data/sensor_locations.csv")
    ap.add_argument("--boundary", default="data/ncr_cities.gpkg")
    ap.add_argument("--out-dir", default="outputs/figures/exports")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--single", action="store_true", help="one map per hour instead of a panel per day")
    ap.add_argument("--no-basemap", action="store_true")
    ap.add_argument("--no-stations", action="store_true")
    ap.add_argument("--view", choices=["cnn", "boundary"], default="cnn",
                    help="cnn = the window the CNN-LSTM figures use; boundary = the full NCR bbox")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    pts = None if args.no_stations else station_xy(args.stations)
    boundary = load_boundary(args.boundary)
    view_window = CNN_VIEW if args.view == 'cnn' else None
    print("  boundary:", "none" if boundary is None else "%d cities" % len(boundary))
    for day in args.days:
        if args.single:
            for h in args.hours:
                tif = os.path.join(args.folder, "%sT%s.tif" % (day, h))
                out = os.path.join(args.out_dir, "xgb_%s_%s.png" % (day, h))
                render_snapshot(tif, out,
                                "PM2.5 prediction, %s-%s-%s %s" % (day[:4], day[4:6], day[6:], CLOCK[h]),
                                args.vmin, args.vmax, pts, boundary, not args.no_basemap, view_window)
                print("  ->", out)
        else:
            out = os.path.join(args.out_dir, "xgb_panels_%s.png" % day)
            render_panels(day, args.folder, out, args.vmin, args.vmax, args.hours, pts, boundary,
                          not args.no_basemap, view_window)
            print("  ->", out)


if __name__ == "__main__":
    main()

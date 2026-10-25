"""Render 10 m raster surfaces with matplotlib's imshow.

At 10.6 M pixels the polygon renderer in `maps.surfaces` cannot be used: it builds one matplotlib
patch per cell. This module plots the array directly and downsamples for display only -- the
GeoTIFF keeps full resolution.
"""
import os

import numpy as np


def _block_mean(a, factor):
    """Display-only downsample: mean over factor x factor blocks, NaN-aware."""
    if factor <= 1:
        return a
    h, w = a.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    b = a[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor)
    with np.errstate(invalid='ignore'):
        return np.nanmean(b, axis=(1, 3))


def render(arr, out_png, extent, vmin, vmax, title, stations_xy=None, factor=8,
           cmap='viridis', basemap=True, alpha=0.75):
    """One 10 m field as a PNG. `extent` is (xmin, xmax, ymin, ymax) in the raster CRS."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from maps import basemap as BM

    show = _block_mean(arr, factor)
    fig, ax = plt.subplots(figsize=(7.6, 8.4))
    im = ax.imshow(np.ma.masked_invalid(show), extent=extent, origin='upper',
                   cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax),
                   interpolation='nearest', alpha=alpha, zorder=2)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    if basemap:
        # signature is add_basemap(ax, crs, xlim, ylim); passing the extent four times raised
        # TypeError, which the old try/except swallowed -- the tiles were never drawn.
        BM.add_basemap(ax, 'EPSG:32651', (extent[0], extent[1]), (extent[2], extent[3]))
    if stations_xy is not None and len(stations_xy):
        ax.scatter(stations_xy[:, 0], stations_xy[:, 1], s=9, c='white',
                   edgecolors='#222222', linewidths=0.5, zorder=6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('easting (km)'); ax.set_ylabel('northing (km)')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: '%.0f' % (v / 1000)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: '%.0f' % (v / 1000)))
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label('PM2.5 (ug/m3)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_png


def panel(frames, out_png, extent, vmin, vmax, day, stations_xy=None, factor=8):
    """2x2 panel of the four snapshots on one shared scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from maps import basemap as BM

    ts_list = sorted(frames)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 13.0), sharex=True, sharey=True)
    last = None
    for ax, ts in zip(axes.ravel(), ts_list):
        arr = frames[ts]['prediction']
        im = ax.imshow(np.ma.masked_invalid(_block_mean(arr, factor)), extent=extent,
                       origin='upper', cmap='viridis', norm=Normalize(vmin=vmin, vmax=vmax),
                       interpolation='nearest', alpha=0.75, zorder=2)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        BM.add_basemap(ax, 'EPSG:32651', (extent[0], extent[1]), (extent[2], extent[3]))
        if stations_xy is not None and len(stations_xy):
            ax.scatter(stations_xy[:, 0], stations_xy[:, 1], s=8, c='white',
                       edgecolors='#222222', linewidths=0.4, zorder=6)
        usable = int(frames[ts]['usable'].sum())
        ax.set_title('%s  (%.1f%% of 10 m px usable)'
                     % (str(ts)[11:16], 100 * usable / frames[ts]['usable'].size), fontsize=9)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: '%.0f' % (v / 1000)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: '%.0f' % (v / 1000)))
        last = im
    fig.suptitle('PM2.5 at 10 m, %s (shared %.1f-%.1f ug/m3)' % (day, vmin, vmax), fontsize=11)
    cb = fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.6, pad=0.015)
    cb.set_label('PM2.5 (ug/m3)')
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out_png

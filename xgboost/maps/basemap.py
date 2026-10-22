"""Basemap layers for the AQI maps, mirroring the CNN-LSTM branch's `output/aqi_maps.py`.

Their overlay technique, in one line: the raster is drawn in REAL projected coordinates
(`ax.imshow(values, extent=(left, right, bottom, top))` + `ax.set_aspect('equal')`), and a basemap is
drawn underneath at `zorder=0`. Anything that shares the CRS therefore lines up with no manual
registration -- that is the whole trick.

This module reproduces that with the same fallback chain, so the overlay turns on by itself:

    1. web tiles      via contextily (needs the library AND internet; CartoDB.* providers need a key)
    2. offline layers via geopandas, from `streets` / `boundary` files
    3. plain          a message saying why, and no basemap

`basemap.enabled: true, web_tiles: true, vector_layers: true, strict: false` is the default: every
available source is tried, an unavailable one is noted once and skipped, and a failure never breaks the
map. `strict: true` turns an unavailable enabled source into an error, for CI.
"""
import os
import warnings

import numpy as np

CONFIG = {
    'enabled': True,
    'web_tiles': True,
    'vector_layers': True,
    'strict': False,
    'provider': 'CartoDB.Positron',   # any xyzservices name; OpenStreetMap.Mapnik needs no key
    'url': None,                      # explicit tile URL, overrides `provider` (${VAR} read from env)
    'key_env': 'CARTO_API_KEY',
    'cache_dir': '.tile_cache',
    'boundary': None,                 # one GeoJSON holding the study-area polygon
    'streets': None,                  # GeoJSON / gpkg of road centrelines
}

_STATE = {'last': None, 'noted': set()}
STREET_COLOR = '#5f5e59'
BOUNDARY_COLOR = '#4a4a48'


def configure(**kwargs):
    """Override any CONFIG key, e.g. configure(web_tiles=False) for an offline run."""
    unknown = set(kwargs) - set(CONFIG)
    if unknown:
        raise KeyError(f'unknown basemap option(s): {sorted(unknown)}; known: {sorted(CONFIG)}')
    CONFIG.update(kwargs)
    _STATE['last'] = None
    _STATE['noted'].clear()
    return dict(CONFIG)


def last_basemap():
    """'web' | 'vectors' | 'plain' | None -- what the most recent ax actually got."""
    return _STATE['last']


def _note(message):
    """Warn once per reason, so a 4-panel figure does not repeat itself."""
    if message not in _STATE['noted']:
        _STATE['noted'].add(message)
        if CONFIG['strict']:
            raise RuntimeError(message)
        warnings.warn(message)


def _read_layer(path, crs):
    import geopandas as gpd
    if not path or not os.path.exists(path):
        return None
    layer = gpd.read_file(path)
    if layer.empty:
        return None
    return layer if layer.crs is None else layer.to_crs(crs)


def _draw_vectors(ax, crs, xlim, ylim):
    """OSM streets + study-area boundary, clipped to the map window."""
    drew = False
    for key, style in (('streets', {'color': STREET_COLOR, 'linewidth': 0.5, 'alpha': 0.85}),
                       ('boundary', {'color': BOUNDARY_COLOR, 'linewidth': 1.1,
                                     'linestyle': (0, (5, 3))})):
        layer = _read_layer(CONFIG[key], crs)
        if layer is None:
            continue
        layer = layer.cx[xlim[0]:xlim[1], ylim[0]:ylim[1]]
        if layer.empty:
            continue
        if key == 'boundary' and layer.geom_type.str.contains('Polygon').all():
            layer = layer.boundary
        layer.plot(ax=ax, zorder=1, **style)
        drew = True
    if not drew:
        _note('basemap vector_layers is on but no streets/boundary layer was available in the window; '
              'set basemap.boundary / basemap.streets, or web_tiles=True for tiles instead.')
    return drew


def _draw_web(ax, crs):
    """Web tiles through contextily. Returns False (and notes why) when unavailable."""
    try:
        import contextily as cx
    except ImportError:
        # No contextily: fetch the tiles directly. This is the normal path in this environment.
        try:
            if _add_tiles_manual(ax, crs):
                return True
        except Exception:
            pass
        _note('web tiles unavailable: contextily not installed and the tile fetch failed '
              '(offline?). Set basemap.web_tiles=False to silence this.')
        return False
    try:
        return _add_tiles(ax, crs, cx)
    except Exception as exc:                       # offline, bad provider, key rejected, ...
        _note(f'web tiles unavailable ({type(exc).__name__}: {exc}); continuing without them.')
        return False


# Keyless light basemap. CartoDB's light_all tiles are public, need no key, and give the pale grey style
# the CNN-LSTM figures use. Swap for tile.openstreetmap.org if you prefer the standard OSM palette.
TILE_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
# Standard OSM tiles are more saturated than the pale Positron style the CNN-LSTM figures use, so
# blend them toward white; 1.0 = untouched, 0.0 = plain white.
TILE_LIGHTEN = 0.72


# Tile fetching is best-effort and must never stall a run. Three guards, all added after a Kaggle run
# spent over an hour here:
#   * 5 s timeout, not 20. OSM throttles datacenter IPs, so every request otherwise sat out the timeout;
#   * STICKY failure: one failed request disables tiles for the rest of the process. Without it, nine map
#     renders x 9 tiles x 20 s repeated the same doomed request 81 times (~27 min for nothing);
#   * the cache is checked FIRST, so a shipped .tile_cache renders instantly with no network at all.
# A pre-seeded cache is the reliable way to get tiles on a machine that cannot reach the tile server.
_TILE_STATE = {'failed': False, 'fetched': 0}
TILE_TIMEOUT_S = 5


# Accepted spellings of the cache folder, in order of preference. '.tile_cache' is what this module
# writes; the others exist because a hand-built data bundle may name it differently, and losing the
# basemap over a missing underscore is a silly failure.
_CACHE_NAMES = ('.tile_cache', '.tilecache', 'tile_cache', 'tilecache')


def _cache_dirs():
    """Where to look for a cached tile, in order.

    1. beside the code (the working directory) -- what a normal run writes;
    2. inside `data/` -- so a DATA bundle can carry the tiles. On Kaggle the repository is cloned into a
       fresh directory and the data is symlinked in, so a cache shipped with the data would otherwise
       never be found and every map would lose its basemap.

    Four spellings are accepted per location, because the folder may be created by hand.
    """
    roots = ['.']
    try:
        import data as _data
        roots.append(_data.DATA_DIR)
    except Exception:
        pass

    dirs = []
    shipped = CONFIG['cache_dir']
    if shipped not in dirs:
        dirs.append(shipped)                     # honour an explicit override first
    for root in roots:
        for name in _CACHE_NAMES:
            cand = os.path.join(root, name)
            if cand not in dirs:
                dirs.append(cand)
    return dirs


def _find_tile(z, x, y):
    """(path, dir_to_write_into) for an existing tile, else (None, default_dir)."""
    dirs = _cache_dirs()
    for root in dirs:
        p = os.path.join(root, str(z), str(x), '%d.png' % y)
        if os.path.exists(p):
            return p, root
    return None, dirs[0]


def _tile_png(z, x, y):
    """One tile as an HxWx3 float array, from cache or the network. None when unavailable."""
    import urllib.request
    import matplotlib.image as mpimg

    path, root = _find_tile(z, x, y)
    cache = os.path.join(root, str(z), str(x))

    if path is None:
        if os.environ.get('XGB_NO_TILES', '').strip() not in ('', '0'):
            return None
        if _TILE_STATE['failed']:
            return None
        os.makedirs(cache, exist_ok=True)
        try:
            req = urllib.request.Request(TILE_URL.format(z=z, x=x, y=y),
                                         headers={'User-Agent': 'thesis-pm25-mapper/1.0'})
            with urllib.request.urlopen(req, timeout=TILE_TIMEOUT_S) as resp:
                data = resp.read()
            with open(os.path.join(cache, '%d.png' % y), 'wb') as fh:
                fh.write(data)
            path = os.path.join(cache, '%d.png' % y)
            _TILE_STATE['fetched'] += 1
        except Exception:
            _TILE_STATE['failed'] = True      # give up for the rest of the process
            return None
    try:
        img = mpimg.imread(path)
        return img[:, :, :3]
    except Exception:
        return None


def _add_tiles_manual(ax, crs):
    """Fetch and stitch tiles without contextily. Returns False when offline."""
    import numpy as np

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    try:
        from pyproj import Transformer
        to_merc = Transformer.from_crs(str(crs), 'EPSG:3857', always_xy=True)
        back = Transformer.from_crs('EPSG:3857', str(crs), always_xy=True)
        mx0, my0 = to_merc.transform(xlim[0], ylim[0])
        mx1, my1 = to_merc.transform(xlim[1], ylim[1])
    except Exception:
        return False

    merc_half = 20037508.342789244
    span = max(mx1 - mx0, my1 - my0)
    zoom = int(np.clip(np.floor(np.log2(156543.03392 * 700.0 / max(span, 1.0))), 8, 17))
    n = 2 ** zoom
    tile_span = 2 * merc_half / n

    def to_tile(mx, my):
        return (int(np.floor((mx + merc_half) / tile_span)),
                int(np.floor((merc_half - my) / tile_span)))

    tx0, ty1 = to_tile(mx0, my0)
    tx1, ty0 = to_tile(mx1, my1)
    if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > 64:      # refuse to hammer the server
        return False

    rows, missing = [], 0
    for ty in range(ty0, ty1 + 1):
        row = []
        for tx in range(tx0, tx1 + 1):
            tile = _tile_png(zoom, tx, ty)
            if tile is None:
                missing += 1
                tile = np.ones((256, 256, 3), dtype=float)
            row.append(tile)
        rows.append(np.concatenate(row, axis=1))
    if missing == len(rows) * (tx1 - tx0 + 1):
        return False                                 # nothing downloaded: likely offline
    mosaic = np.concatenate(rows, axis=0)
    mosaic = np.clip(TILE_LIGHTEN * mosaic + (1.0 - TILE_LIGHTEN), 0.0, 1.0)

    # mosaic extent back in the axes' CRS. Over ~25 km the Mercator->UTM scale change is ~0.1%, i.e.
    # far below a display pixel, so transforming the corners and drawing the mosaic as an image is
    # accurate enough for a basemap and avoids a full reprojection.
    wx0 = -merc_half + tx0 * tile_span
    wy1 = merc_half - ty0 * tile_span
    wx1 = -merc_half + (tx1 + 1) * tile_span
    wy0 = merc_half - (ty1 + 1) * tile_span
    cx0, cy0 = back.transform(wx0, wy0)
    cx1, cy1 = back.transform(wx1, wy1)

    ax.imshow(mosaic, extent=[cx0, cx1, cy0, cy1], origin='upper', zorder=0,
              interpolation='bilinear', aspect='auto')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return True


def _add_tiles(ax, crs, cx):
    url = CONFIG['url']
    if url:
        for name, value in list(os.environ.items()):
            url = url.replace('${%s}' % name, value)
        ax.set_axis_off()
        cx.add_basemap(ax, crs=str(crs), source=url, zorder=0)
        return True
    provider = CONFIG['provider']
    key = os.environ.get(CONFIG['key_env'], '')
    if provider.startswith('CartoDB') and not key:
        _note(f'{provider} needs {CONFIG["key_env"]} in the environment; no tiles requested. '
              f'Use provider="OpenStreetMap.Mapnik" for a keyless source.')
        return False
    os.makedirs(CONFIG['cache_dir'], exist_ok=True)
    cx.add_basemap(ax, crs=str(crs), source=provider, zorder=0,
                   cache=cx.cache.FileCache(CONFIG['cache_dir']))
    return True


def add_basemap(ax, crs, xlim, ylim, quiet=False):
    """Try tiles, then vectors. Sets the axis limits so tiles know which window to fetch."""
    _STATE['last'] = 'plain'
    if not CONFIG['enabled']:
        return 'plain'
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if CONFIG['web_tiles'] and _draw_web(ax, crs):
        _STATE['last'] = 'web'
        return 'web'
    if CONFIG['vector_layers'] and _draw_vectors(ax, crs, xlim, ylim):
        _STATE['last'] = 'vectors'
        return 'vectors'
    return 'plain'

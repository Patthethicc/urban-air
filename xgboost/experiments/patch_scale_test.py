"""Phase 0 gate: would a 10 m output grid carry information, or just resolution?

With a 32 px (320 m) window, two adjacent 10 m pixels share 31/32 of their window, so their patch
statistics differ only slightly. If "slightly" is small compared with the total spatial spread, then a
10.6 M-cell product is 10.6 M near-duplicates of ~3,700 independent values -- finer looking, not finer.

This script measures that directly, using box filters (O(N) regardless of window size) rather than
cutting 10 M patches:

  1. read a square block of the raster (default 1024 x 1024 px = 10.24 km, inside the study area);
  2. compute the 32 px patch mean of each band for EVERY 10 m pixel in the block;
  3. report how much the field changes per 10 m step, how correlated neighbours are, and the
     distance over which the field decorrelates;
  4. compare against the station spacing -- the scale at which the product can be validated.

Usage
-----
    python experiments/patch_scale_test.py                       # default block, all bands
    python experiments/patch_scale_test.py --block 2048 --window 32
    python experiments/patch_scale_test.py --window 11           # 110 m windows, for comparison
"""
import argparse
import os
import sys

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def patch_mean(a, win):
    """win x win box mean for every pixel, NaN-aware via a coverage mask."""
    valid = np.isfinite(a).astype(np.float32)
    num = uniform_filter(np.where(valid > 0, a, 0.0).astype(np.float32), win, mode="nearest")
    den = uniform_filter(valid, win, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.05, num / den, np.nan)


def decorrelation_length(field, step_m, max_lag=80):
    """Distance at which the lag-k autocorrelation first drops below 0.5 (along rows)."""
    f = np.nan_to_num(field - np.nanmean(field))
    var = float(np.mean(f * f))
    if var <= 0:
        return float("nan")
    for lag in range(1, max_lag):
        a, b = f[:, :-lag], f[:, lag:]
        if float(np.mean(a * b)) / var < 0.5:
            return lag * step_m
    return float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raster", default=os.path.join("data", "master_raster_v2.tif"))
    ap.add_argument("--block", type=int, default=1024, help="side of the test block, in 10 m pixels")
    ap.add_argument("--window", type=int, default=32, help="patch side in pixels (32 = 320 m)")
    ap.add_argument("--row0", type=int, default=None)
    ap.add_argument("--col0", type=int, default=None)
    args = ap.parse_args(argv)

    with rasterio.open(args.raster) as src:
        n = min(args.block, src.height, src.width)
        r0 = args.row0 if args.row0 is not None else max(0, (src.height - n) // 2)
        c0 = args.col0 if args.col0 is not None else max(0, (src.width - n) // 2)
        win = rasterio.windows.Window(c0, r0, n, n)
        arr = src.read(window=win).astype(np.float32)
        nodata = src.nodata
        step = float(src.res[0])
        names = [src.descriptions[i] or "band%d" % (i + 1) for i in range(src.count)]

    if nodata is not None:
        arr[arr == nodata] = np.nan

    print("raster      : %dx%d px at %.0f m" % (arr.shape[2], arr.shape[1], step))
    print("test block  : %d x %d px = %.2f x %.2f km  (row0=%d, col0=%d)"
          % (n, n, n * step / 1000, n * step / 1000, r0, c0))
    print("patch window: %d px = %.0f m\n" % (args.window, args.window * step))

    print("  %-10s %9s %9s %9s %9s %9s" % ("band", "mean", "sd(field)", "mean|d10m|",
                                           "d10m/sd", "decorr(m)"))
    for b in range(arr.shape[0]):
        field = patch_mean(arr[b], args.window)
        sd = float(np.nanstd(field))
        if not np.isfinite(sd) or sd == 0:
            continue
        step_diff = float(np.nanmean(np.abs(np.diff(field, axis=1))))
        ratio = step_diff / sd
        L = decorrelation_length(field, step)
        print("  %-10s %9.4f %9.4f %9.5f %9.4f %9.0f"
              % (names[b][:10], float(np.nanmean(field)), sd, step_diff, ratio, L))

    print("\n  how to read d10m/sd  (share of the field's spread moved by one 10 m step)")
    print("    >= 0.30  the 10 m grid is genuinely resolved")
    print("    ~ 0.05   decorative: neighbours within ~%d px are near-duplicates" % args.window)
    print("    <= 0.01  only the %d m window's own scale survives; publish at 320-500 m"
          % (args.window * int(step)))
    print("\n  station spacing for reference: median nearest neighbour 1,268 m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

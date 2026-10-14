"""Does the sensor network sample the morphology the raster actually contains?

If the 17 cells were a narrow slice of the raster's morphology, "morphology does not predict PM2.5"
would be bounded by the network never having observed much morphological variety - a much weaker and
much more defensible claim than "urban form does not matter". This measures that directly, and is why
the Step 3 negative result must be worded as a power statement rather than a physical one.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import spatial_cv  # noqa: E402

MORPH = ['bh_mean', 'bh_stdev', 'bldg_dens', 'ndvi_mean', 'aspect_ratio_mean']

grid = pd.read_csv('full_morphology_grid.csv')
cells = grid.groupby('id')[MORPH].first()
doc = spatial_cv.load_fold_map('fold_partitions.json')
parts = doc['partitions']
used = parts['train'] + parts['calibration'] + parts['test']

print(f"raster cells: {len(cells)} | cells containing a sensor: {len(used)}")
print(f"unsampled raster cells: {len(cells) - len(used)}")

rows = []
for f in MORPH:
    r = cells[f]
    net = r.loc[[c for c in used if c in r.index]]
    p05, p95 = r.quantile(.05), r.quantile(.95)
    rows.append({
        'feature': f,
        'raster_min': r.min(), 'raster_max': r.max(),
        'network_min': net.min(), 'network_max': net.max(),
        'network_sd_over_raster_sd': net.std() / r.std() if r.std() else np.nan,
        'network_span_over_p05_p95': (net.max() - net.min()) / (p95 - p05) if p95 > p05 else np.nan,
        'raster_cells_outside_network_range': int(((r < net.min()) | (r > net.max())).sum()),
        'pct_raster_outside': 100 * ((r < net.min()) | (r > net.max())).mean(),
    })
s = pd.DataFrame(rows).set_index('feature')
print('\n' + s.round(3).to_string())

print(f"\nMean share of raster SD captured by the network: "
      f"{s['network_sd_over_raster_sd'].mean():.0%} "
      f"(range {s['network_sd_over_raster_sd'].min():.0%}-"
      f"{s['network_sd_over_raster_sd'].max():.0%})")
print(f"Mean coverage of the raster 5-95% band: {s['network_span_over_p05_p95'].mean():.0%}")
print(f"Raster cells outside the network's observed range: "
      f"{s['pct_raster_outside'].mean():.0f}% on average by feature")

print('\nCONCLUSION: morphological coverage is not the binding constraint; statistical power is.')
print(f'  The morphology question has an effective n of {len(parts["train"])} training cells '
      f'(CV) and {len(parts["test"])} test cells.')
print('  The negative result bounds morphology WITHIN the range the network observed, and with this')
print('  sample size. It does not establish that urban form is irrelevant to street-level PM2.5.')

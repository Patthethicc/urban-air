"""Cohort/window sweep: does the five-arm ordering survive a change of time window?

The CNN-LSTM branch swept seven data compositions (raw, full, from_2023, from_2024, …) and found
`from_2023_with_preinstall` the best; this branch has only a single `cohort_window.py` sensitivity.
The point of this script is NOT to find a higher R2 -- a shorter window drops rows, so its R2 is not
comparable. The point is to test whether the paper's claim ("no detectable morphology signal")
survives every window, which is what makes it a finding rather than an artefact of one date range.

Each window runs in its own subprocess (data.py reads XGB_WINDOW_START at import, so a fresh process
is the only honest way to switch it) and writes into its own folder, so the main results in
outputs/ are never touched.

Usage
-----
    python experiments/window_sweep.py --dry-run                 # print the commands, run nothing
    python experiments/window_sweep.py --arms A C C+ --n-trials 20
    python experiments/window_sweep.py --windows 2021-07-11 2023-01-01 2024-01-01 2025-01-01

Cost, measured on this machine with --row-stride 4 --n-trials 60:
    one window x 5 arms  ~ 1.9 h        four windows  ~ 7.5 h
That is why the notebook keeps the sweep OFF by default: Kaggle CPU sessions stop at 12 h.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The start of the record is 2021-07-11, so that entry means "no window" (the current result).
WINDOWS = ['2021-07-11', '2023-01-01', '2024-01-01', '2025-01-01']
ARMS = ['A', 'B', 'C', 'C+', 'TEXTURE']


def plan(windows, arms, n_trials, row_stride, out_root):
    """The commands that would run, one per window. Kept separate so --dry-run is honest."""
    jobs = []
    for w in windows:
        out = os.path.join(out_root, w)
        cmd = [sys.executable, 'run_pipeline.py', '--stage', 'ablation',
               '--arms', *arms, '--n-trials', str(n_trials), '--row-stride', str(row_stride),
               '--ablation-out-dir', out]
        env = dict(os.environ)
        # 2021-07-11 IS the first timestamp, so leave the switch unset for the full record
        if w != WINDOWS[0]:
            env['XGB_WINDOW_START'] = w
        else:
            env.pop('XGB_WINDOW_START', None)
        jobs.append((w, cmd, env, out))
    return jobs


def collect(out_root, windows):
    """One row per (window, arm) from each window's ablation_summary.csv."""
    rows = []
    for w in windows:
        path = os.path.join(out_root, w, 'ablation_summary.csv')
        if not os.path.exists(path):
            continue
        import pandas as pd
        d = pd.read_csv(path)
        d.insert(0, 'window', w)
        rows.append(d[['window', 'arm', 'n_features', 'pooled_r2', 'r2_mean', 'r2_sd',
                       'pooled_rmse', 'pooled_mae', 'oof_rows']])
    if not rows:
        return None
    import pandas as pd
    return pd.concat(rows, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--windows', nargs='+', default=WINDOWS)
    ap.add_argument('--arms', nargs='+', default=ARMS)
    ap.add_argument('--n-trials', type=int, default=60)
    ap.add_argument('--row-stride', type=int, default=4)
    ap.add_argument('--out-root', default=os.path.join('outputs', 'sweep'))
    ap.add_argument('--dry-run', action='store_true', help='print the commands and stop')
    args = ap.parse_args(argv)

    jobs = plan(args.windows, args.arms, args.n_trials, args.row_stride, args.out_root)
    print('window sweep: %d window(s) x %d arm(s), %d trials, stride %d'
          % (len(jobs), len(args.arms), args.n_trials, args.row_stride))
    for w, cmd, env, out in jobs:
        print('\n[%s] -> %s' % (w, out))
        print('  XGB_WINDOW_START=%s' % env.get('XGB_WINDOW_START', '(unset: full record)'))
        print('  ' + ' '.join(cmd[1:]))
    if args.dry_run:
        print('\n--dry-run: nothing executed.')
        return 0

    for w, cmd, env, out in jobs:
        print('\n' + '=' * 74)
        print('window %s' % w, flush=True)
        print('=' * 74, flush=True)
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    table = collect(args.out_root, args.windows)
    if table is None:
        print('\nno summaries found under', args.out_root)
        return 1
    dest = os.path.join(args.out_root, 'window_sweep.csv')
    os.makedirs(args.out_root, exist_ok=True)
    table.to_csv(dest, index=False)
    print('\n=== window sweep ===')
    print(table.round(3).to_string(index=False))
    print('\nwrote', dest)

    # the claim this experiment is here to test
    pivot = table.pivot_table(index='arm', columns='window', values='pooled_r2')
    print('\npooled R2 by window and arm:')
    print(pivot.round(3).to_string())
    if 'C' in pivot.index and 'A' in pivot.index:
        gap = (pivot.loc['C'] - pivot.loc['A']).round(3)
        print('\nC - A (morphology on top of time+history), per window:')
        print(gap.to_string())
        print('\nif every entry stays near zero, the null survives the window change;'
              ' if one window flips it positive by more than ~0.02, that window needs discussion.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

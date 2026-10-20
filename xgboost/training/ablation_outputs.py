import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

"""Shared guard for the results the ablation writes into outputs/.

`oof_predictions_*.csv` and `ablation_params.json` are deliberately NOT committed: they are fully
regenerable from one command and together weigh 8.4 MB. The verification scripts that consume them call
require_ablation_outputs() first so a missing file produces one actionable message instead of a
traceback from deep inside pandas.
"""
import os

REGENERATE = 'python ablation.py --arms A B C --n-trials 20'


def require_ablation_outputs(out_dir='outputs', need_oof=True, need_params=False,
                             arms=('A', 'B', 'C')):
    """Raise SystemExit with a clear message if the ablation outputs are missing."""
    missing = []
    if not os.path.exists(os.path.join(out_dir, 'ablation_results.csv')):
        missing.append('ablation_results.csv')
    if not os.path.exists(os.path.join(out_dir, 'ablation_summary.csv')):
        missing.append('ablation_summary.csv')
    if need_oof:
        for arm in arms:
            f = f'oof_predictions_{arm}.csv'
            if not os.path.exists(os.path.join(out_dir, f)):
                missing.append(f)
    if need_params and not os.path.exists(os.path.join(out_dir, 'ablation_params.json')):
        missing.append('ablation_params.json')
    if missing:
        raise SystemExit(
            f"missing ablation output(s) in {out_dir}/: {', '.join(missing)}\n"
            f"The large per-row files are not committed because they are regenerable. Run:\n"
            f"    {REGENERATE}\n"
            f"then re-run this check."
        )

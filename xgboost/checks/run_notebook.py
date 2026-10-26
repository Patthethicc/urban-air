"""Execute a notebook end to end, like Jupyter does, and report the first failure with its cell.

Uses nbclient (a real kernel), so `!python ...`, `%matplotlib inline` and `display()` all behave as they
do in the browser. A cell that raises is reported with its index and the executed source, which is what
"make sure each cell works" requires -- static parsing cannot catch a NameError from a cell whose
definition was deleted.

    python checks/run_notebook.py xgboost_full.ipynb
    python checks/run_notebook.py xgboost_full.ipynb --timeout 5400
"""
import argparse
import os
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('notebook')
    ap.add_argument('--timeout', type=int, default=7200,
                    help='per-cell timeout in seconds (default 7200)')
    ap.add_argument('--save', default=None, help='write the executed notebook here')
    a = ap.parse_args()

    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(a.notebook, as_version=4)
    n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
    print('executing %s (%d cells, %d code) with a 2 h cell cap' % (a.notebook, len(nb.cells), n_code))

    client = NotebookClient(nb, timeout=a.timeout, kernel_name='python3',
                            allow_errors=True, resources={'metadata': {'path': os.getcwd()}})
    t0 = time.time()
    client.execute()

    failed = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        errs = [o for o in cell.get('outputs', []) if o.get('output_type') == 'error']
        if errs:
            failed.append((i, errs[0].get('ename'), errs[0].get('evalue'),
                           ''.join(cell['source'])))

    print('\n%s  (%.1f min)' % ('=' * 78, (time.time() - t0) / 60))
    if failed:
        print('%d CELL(S) FAILED' % len(failed))
        for i, ename, evalue, src in failed:
            print('\n---- cell %d: %s: %s' % (i, ename, evalue))
            for line in src.strip().splitlines()[:12]:
                print('     ' + line)
    else:
        print('ALL CELLS EXECUTED WITHOUT ERROR')

    if a.save:
        import nbformat as _nbf
        _nbf.write(nb, a.save)
        print('executed notebook written to %s' % a.save)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

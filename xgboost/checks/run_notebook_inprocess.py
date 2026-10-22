"""Execute a notebook's code cells in-process, in order, reporting any cell that raises.

Why not nbclient: launching a kernel needs `jupyter_core` to secure its connection file, which this
sandbox denies (`pywintypes.error: (5, 'SetFileSecurity', 'Access is denied.')`). Running the cells here
avoids a kernel and still exercises every cell, in order, against one shared namespace -- which is what
catches a NameError caused by a definition that was accidentally deleted.

Two translation rules, both learned the hard way:

* a line-magic starts with `%` IMMEDIATELY followed by a letter (`%matplotlib inline`). A wrapped
  continuation such as `      % (f'{len(df_clean):,}', ...)` also begins with `%` once stripped, and
  treating that as a magic replaced a real operand with `pass`, producing "SyntaxError: unmatched ')'".
  The regex `^%[A-Za-z]` distinguishes them.
* `{NAME}` in a `!` line must expand when the line RUNS, not when the cell is translated: IPython
  expands per line, and a cell may define the variable above the `!` line. The generated code embeds an
  f-string so the expansion is deferred to exec time.

    python checks/run_notebook_inprocess.py xgboost_full.ipynb
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback

os.environ.setdefault('MPLBACKEND', 'Agg')

MAGIC = re.compile(r'^%[A-Za-z]')


def _shell_line(indent, cmd):
    """Build a subprocess call whose {NAME} placeholders expand at exec time."""
    if cmd.startswith('python '):
        cmd = '"%s" %s' % (sys.executable, cmd[7:])
    # an f-string literal keeps {NAME} deferred; backslashes and triple quotes cannot appear in a
    # literal f-string, so escape them for this narrow use (commands are simple CLI invocations).
    body = cmd.replace('\\', '\\\\')
    return indent + 'subprocess.run(f"""' + body + '""", shell=True, check=True)'


def transform(src):
    out = []
    for line in src.splitlines():
        stripped = line.strip()
        indent = line[:len(line) - len(line.lstrip())]
        if stripped.startswith('!'):
            out.append(_shell_line(indent, stripped[1:].strip()))
        elif MAGIC.match(stripped):
            out.append(indent + 'pass  # ipython magic: ' + stripped)
        else:
            out.append(line)
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('notebook')
    a = ap.parse_args()

    nb = json.load(open(a.notebook, encoding='utf-8'))
    root = os.path.dirname(os.path.abspath(a.notebook))
    os.chdir(root)
    sys.path.insert(0, root)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.show = lambda *x, **k: None          # never block on a figure
    except Exception:
        pass

    g = {'__name__': '__main__', 'subprocess': subprocess, 'os': os, 'sys': sys}
    try:
        from IPython.display import display, Image, Markdown   # noqa: F401
    except Exception:
        g['display'] = lambda *x, **k: None

    t_all = time.time()
    failures = []
    code = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
    print('executing %s: %d cells (%d code)' % (a.notebook, len(nb['cells']), len(code)))

    for i in code:
        src = ''.join(nb['cells'][i]['source'])
        first = next((ln for ln in src.splitlines() if ln.strip()), '')[:64]
        t0 = time.time()
        try:
            exec(compile(transform(src), '<cell %d>' % i, 'exec'), g)
        except Exception as exc:
            failures.append((i, exc, traceback.format_exc()))
            print('  cell %2d  FAIL  %-64s %6.0fs  %s: %s'
                  % (i, first, time.time() - t0, type(exc).__name__, exc))
            continue
        print('  cell %2d  ok    %-64s %6.0fs' % (i, first, time.time() - t0))

    print('\n' + '=' * 100)
    print('total %.1f min' % ((time.time() - t_all) / 60))
    if failures:
        print('%d CELL(S) FAILED' % len(failures))
        for i, exc, tb in failures:
            print('\n---- cell %d: %s: %s' % (i, type(exc).__name__, exc))
            print('\n'.join(tb.strip().splitlines()[-5:]))
        return 1
    print('ALL CELLS EXECUTED WITHOUT ERROR')
    return 0


if __name__ == '__main__':
    sys.exit(main())

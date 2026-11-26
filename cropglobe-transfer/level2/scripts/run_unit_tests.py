"""Runs the unit / data tests that are embedded in the notebook cells (77 tests) outside Colab.  Usage: python scripts/run_unit_tests.py"""
import io, os, sys, contextlib, tempfile
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); CELLS = os.path.join(ROOT, 'src', 'cells')
tmp = tempfile.gettempdir(); ns = {'__name__': '__main__', 'OUT_DIR': tmp, 'MAIN_OUT_DIR': tmp}
for f in ['c02_config.py', 'c04_core.py', 'c07_tests.py', 'c09_data.py', 'c10_tests_data.py']:
    with contextlib.redirect_stdout(io.StringIO()): exec(compile(open(os.path.join(CELLS, f)).read(), f, 'exec'), ns)
R = ns['TEST_REPORT']; bad = [r for r in R if r[1] != 'PASS']
for n, st, m in bad: print(st, n[:100], '->', m[:400])
print(f'{len(R)} tests, {len(R) - len(bad)} passed'); sys.exit(1 if bad else 0)

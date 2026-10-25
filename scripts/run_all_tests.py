"""Runs the embedded unit/data tests of Level 1 (v3.8 cells) and Level 2 (v3 cells) outside Colab, plus the synthetic ordering/closing tests.  Usage: python scripts/run_all_tests.py"""
import io, os, sys, subprocess, contextlib, tempfile
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); tmp = tempfile.gettempdir(); fails = 0
def unit(label, cells, files):
    global fails
    ns = {'__name__': '__main__', 'OUT_DIR': tmp, 'MAIN_OUT_DIR': tmp}
    for f in files:
        with contextlib.redirect_stdout(io.StringIO()): exec(compile(open(os.path.join(cells, f)).read(), f, 'exec'), ns)
    R = ns['TEST_REPORT']; bad = [r for r in R if r[1] != 'PASS']; fails += len(bad)
    for n, st, m in bad: print('  FAIL', n[:90], '->', m[:300])
    print(f'{label}: {len(R)} tests, {len(R) - len(bad)} passed')
unit('Level 1 v3.8', os.path.join(ROOT, 'level1', 'src', 'cells'), ['c02_config.py', 'c04_core.py', 'c07_tests.py', 'c09_data.py', 'c10_tests_data.py'])
unit('Level 2 v3', os.path.join(ROOT, 'level2', 'src', 'cells'), ['c02_config.py', 'c04_core.py', 'c07_tests.py', 'c09_data.py', 'c10_tests_data.py'])
for t in [('level2', 'tests', 'test_usa_order.py'), ('level1', 'tests', 't_probe.py')]:
    r = subprocess.run([sys.executable, os.path.join(ROOT, *t)], capture_output=True, text=True); print(os.path.join(*t), '->', (r.stdout.strip().splitlines() or ['(no output)'])[-1]); fails += r.returncode != 0
sys.exit(1 if fails else 0)

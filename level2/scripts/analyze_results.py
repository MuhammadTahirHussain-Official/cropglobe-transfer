"""Re-computes every number of the final analysis from the finished results files (no training).
Usage:  python scripts/analyze_results.py MAIN_results.csv [FOLLOWUP_results.csv]
MAIN_results.csv = results of the main Level 2 notebook (seeds 0-4);  FOLLOWUP_results.csv = results of the follow-up notebook (seeds 5-9).
Prints the pre-registered claims first, then the exploratory tables (which never change a verdict)."""
import io, os, sys, contextlib, warnings, tempfile
import pandas as pd
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); CELLS = os.path.join(ROOT, 'src', 'cells')
if len(sys.argv) < 2: sys.exit(__doc__)
tmp = tempfile.gettempdir(); ns = {'__name__': '__main__', 'OUT_DIR': tmp, 'MAIN_OUT_DIR': tmp}
for f in ['c02_config.py', 'c04_core.py', 'c24_level2_analysis.py', 'c02f_followup.py', 'c24f_followup_analysis.py']:
    with contextlib.redirect_stdout(io.StringIO()): exec(compile(open(os.path.join(CELLS, f)).read(), f, 'exec'), ns)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
df = ns['load_final'](sys.argv[1]); print(f'MAIN: {len(df)} runs | per seed {df.groupby("seed").size().to_dict()} | metric {df.groupby("target").main_metric.first().to_dict()}')
print('\n=== PRE-REGISTERED CLAIMS L2-C1 / L2-C2 ==='); print(ns['replication_table'](df).round(2).to_string(index=False))
print('\n=== L2-C3 augmentation vs its steps-matched copy ==='); print(ns['c3_aug_contrast'](df).round(2).to_string(index=False))
print('\n=== EXPLORATORY: every mix vs its own single regions (same budget) ==='); print(ns['mix_vs_own_table'](df).round(3).to_string(index=False))
print('\n=== EXPLORATORY: budget effect with the pool held fixed ==='); print(pd.concat([ns['fixed_pool_budget_effect'](df, t) for t in ('ARG', 'USA')]).round(2).to_string(index=False))
print('\n=== EXPLORATORY: Argentina per-doubling slope with / without regions 75, 76 ==='); print(ns['per_doubling_subset'](df, 'ARG').round(3).to_string(index=False))
if len(sys.argv) > 2:
    fu = ns['load_final'](sys.argv[2]); r = ns['followup_claims'](fu); print(f'\nFOLLOW-UP: {len(fu)} runs | per seed {fu.groupby("seed").size().to_dict()}')
    for k in ('FC1', 'FC2'): print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in r[k].items()})
    print(r['descriptive_vs_best'].round(3).to_string(index=False))

"""Checks that the code cells of the Level 1 v3.8 notebooks are byte-identical to the files in level1/src/cells (so the readable sources and the notebooks cannot drift apart).
Usage: python level1/tools/check_notebook_matches_cells.py"""
import json, os, sys
L1 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); CELLS = os.path.join(L1, 'src', 'cells')
COLAB = ['c01_env', 'c02_config', 'c04_core', 'c07_tests', 'c09_data', 'c10_tests_data', 'c11_acquire', 'c12_testsB', 'c13_load', 'c15_plan', 'c17_torch', 'c19_run', 'c21_summary', 'c22_reconcile', 'c23_export']
KAGGLE = {**{n: n for n in COLAB}, 'c01_env': 'c01_env_kaggle', 'c11_acquire': 'c11_acquire_kaggle'}
bad = 0
for nbname, names in [('CropGlobe_Level1_v3_8_COLAB.ipynb', {n: n for n in COLAB}), ('CropGlobe_Level1_v3_8_KAGGLE.ipynb', {n: KAGGLE[n] for n in COLAB if n != 'c23_export'})]:
    codes = [''.join(c['source']).rstrip('\n') for c in json.load(open(os.path.join(L1, 'notebooks', nbname)))['cells'] if c['cell_type'] == 'code']
    for code, (_, fname) in zip(codes, names.items()):
        same = open(os.path.join(CELLS, fname + '.py')).read().rstrip('\n') == code
        if not same: bad += 1; print('DIFFERENT:', nbname, fname)
    print(nbname, 'code cells:', len(codes), '| all identical to src/cells' if not bad else '| MISMATCHES')
sys.exit(1 if bad else 0)

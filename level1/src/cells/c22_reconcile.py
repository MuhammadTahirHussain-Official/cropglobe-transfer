# ======================================================================================
# OPTIONAL: reconcile two results files (e.g. the review's 5,004-row and 5,590-row files).  Read-only; nothing is merged or overwritten.
# The two files were NOT supplied with this update, so this cell is unconfigured by default.
# ======================================================================================
RECONCILE_A = None      # e.g. '/content/drive/MyDrive/.../results_5004.csv'
RECONCILE_B = None      # e.g. '/content/drive/MyDrive/.../results_5590.csv'
if RECONCILE_A and RECONCILE_B:
    _rep = reconcile_results(pd.read_csv(RECONCILE_A), pd.read_csv(RECONCILE_B), names=('A', 'B'))
    print(json.dumps({k: v for k, v in _rep.items() if k != 'conflict_examples'}, indent=1, default=str))
    for _m, _ex in _rep.get('conflict_examples', {}).items(): print(f'  conflicts in {_m} (first rows):', _ex[:5])
else:
    print('reconciliation not configured: set RECONCILE_A / RECONCILE_B to the two results CSVs (they were not supplied, so no reconciliation was performed or verified here).')
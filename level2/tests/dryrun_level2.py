import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(ROOT, 'src', 'cells'))      # the cell files are read by relative name
# synthetic END-TO-END dry run of the Level 2 notebook logic (no torch: a logistic-regression stand-in replaces the network). It tests the PLUMBING, not the science.
import tempfile, os, io, contextlib, sys, json, numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else 'plan'          # 'plan' = real region/crop counts, real config, plan only | 'run' = reduced budgets, all 5 seeds, gate + run + closing
ns = {'__name__': '__main__'}
def X(f, src=None): exec(compile(src or open(f).read(), f, 'exec'), ns)
X('c02_config.py'); X('c04_core.py')
buf = io.StringIO()
with contextlib.redirect_stdout(buf): X('c07_tests.py'); X('c09_data.py')
assert ns['TESTS_A_PASSED']
td = tempfile.mkdtemp(); ns.update(OUT_DIR=os.path.join(td, 'outputs', 'phase3_level2_fra'), GIT_COMMIT=ns['EXPECTED_CROPNET_COMMIT']); os.makedirs(ns['OUT_DIR'])
src = open('c11_acquire.py').read(); X('ns_guard', src[src.index('# ---- namespace guard'):src.index('check_namespace(write=False)\nprint(')])
# ---- synthetic French source with the REAL (pre-dedup) region x crop counts; targets small
rc = pd.read_csv(os.path.join(ROOT, 'data', 'fra_region_crop.csv')).iloc[:13]
rc['wheat'] = rc['springwheat'] + rc['winterwheat']; rc['region_code'] = rc['region_code'].astype(int).astype(str)
rows = []
for r in rc.itertuples():
    for c in ('corn', 'soybeans', 'wheat', 'rice', 'other'):
        rows += [(r.region_code, c)] * int(getattr(r, c))
st = np.array([a for a, b in rows]); cr = np.array([b for a, b in rows]); rng = np.random.default_rng(0); perm = rng.permutation(len(st)); st, cr = st[perm], cr[perm]
yid = np.array([ns['CROP_ID'][c] for c in cr]); mu = rng.normal(0, 1, (7, 730)).astype(np.float32)
feats = lambda y: (mu[y] + rng.normal(0, 1.5, (len(y), 730))).astype(np.float32).reshape(len(y), 1, 10, 73)
meta = pd.DataFrame({'npz_row': np.arange(len(st)) * 2, 'crop_id': yid, 'state': st, 'crop': cr}); INFO = {}; FEAT = {ns['SOURCE']: feats(yid)}; LBL = {ns['SOURCE']: yid}
TG = ['ARG', 'USA', 'CHN']
for t in TG:
    cls = ns['DEFINED_CLASSES'][t]                                   # the label set each target really has (config)
    y = np.array([ns['CROP_ID'][c] for c in cls for _ in range(120)]); FEAT[t] = feats(y); LBL[t] = y; INFO[t] = dict(keep=np.arange(len(y)) * 3)
if MODE == 'run': ns.update(BUDGETS=[1000], SEEDS=[0, 1, 2, 3, 4], TARGETS_TO_RUN=['ARG', 'USA', 'CHN'], AUG_BUDGETS=[1000], AUG_COMBOS=2)
ns.update(meta=meta, INFO=INFO, FEATURES=FEAT, NPZ_LBL=LBL, TESTS_PASSED=True, SMOKE_OK=True, IN_NOTEBOOK=False, GEOGRAPHY_OBSERVED={}, FRA_META_REPORT=dict(assigned=len(st), rows_per_region={}), PLAN_NOTE_ARG='x', AUTO_MODE=True, AUTO_MAX_HOURS=None)
ns['APPROVED']['plan'] = True
with contextlib.redirect_stdout(buf): X('c15_plan.py')
cells = ns['PLAN_CELLS_RUN']; g = ns['PLAN_GRID']
print('MODE', MODE, '| planned cells', len(cells), '| runs per seed', {s: len(ns['plan_for_seed'](s)) for s in ns['SEEDS']}, '| total', sum(len(ns['plan_for_seed'](s)) for s in ns['SEEDS']))
pd.set_option('display.width', 250)
print(g[['target', 'budget', 'k', 'feasible', 'pool_size', 'n_combos', 'n_total_combos']].to_string(index=False) if MODE == 'plan' else '')
if MODE == 'plan':
    h = ns['design_hash'](); print('design hash (synthetic table, indicative only):', h[:16]); sys.exit()
from sklearn.linear_model import LogisticRegression
ns['EVAL_X'] = {t: FEAT[t] for t in TG}; ns['EVAL_Y'] = LBL
def fit(Xtr, ytr, Xva, yva, seed, ep=None, aug='off'):
    m = LogisticRegression(max_iter=40, C=0.05).fit(Xtr.reshape(len(Xtr), -1), ytr); v = float(ns['f1_score'](yva, m.predict(Xva.reshape(len(Xva), -1)), average='macro'))
    return dict(model=m, info=dict(val_f1=v, best_epoch=7, epochs_run=22, stop_reason='early_stop_patience', restore_delta=0.0, history=[v]))
ns['FIT_FN'] = fit; ns['PREDICT_FN'] = lambda m, X_: m.predict(X_.reshape(len(X_), -1))
# 1) the run cell must REFUSE before approval
try:
    with contextlib.redirect_stdout(buf): X('c19_run.py')
    print('GATE FAIL: ran without approval'); sys.exit(1)
except Exception as e: print('1) run refused before approval ->', type(e).__name__, str(e)[:90])
assert not os.path.exists(os.path.join(ns['OUT_DIR'], 'results.csv')), 'results written before approval'
# 2) wrong hash refused
ns.update(INSTRUCTOR_APPROVED=True, APPROVED_DESIGN_SHA256='0' * 64)
try:
    with contextlib.redirect_stdout(buf): X('c19_run.py')
    print('GATE FAIL: ran with wrong hash'); sys.exit(1)
except Exception as e: print('2) run refused with a wrong hash ->', type(e).__name__, str(e)[:90])
# 3) approved hash -> runs
ns['APPROVED_DESIGN_SHA256'] = ns['design_hash']()
with contextlib.redirect_stdout(buf): X('c19_run.py')
csv = os.path.join(ns['OUT_DIR'], 'results.csv'); d = ns['load_results'](csv); d['seed'] = pd.to_numeric(d.seed)
print('3) approved run ->', ns.get('AUTO_RESULT'), '| ok', int((d.status == 'ok').sum()), '| per seed', d.groupby('seed').size().to_dict(), '| expected', {s: len(ns['plan_for_seed'](s)) for s in ns['SEEDS']})
print('   tracks:', d.groupby(['target', 'aug']).size().to_dict())
print('   metric by target:', d.groupby('target').main_metric.first().to_dict() if 'main_metric' in d else '')
assert ns['plan_consistency_check']()['ok']
X('c24_level2_analysis.py')
out = ns['closing_report_l2'](csv, os.path.join(ns['OUT_DIR'], 'closing'))
print('4) closing report ->', None if out is None else list(out)); print('   frozen:', os.path.exists(os.path.join(ns['OUT_DIR'], 'closing', 'FREEZE_level2.json')))
rep = out['replication']; c2a = rep[(rep.target == 'ARG') & rep.claim.str.startswith('C2') & (rep.group == 'confirmatory')]
print('5) ARG C2 verdict where no interval exists ->', c2a.verdict.tolist()); assert all(v.startswith('NOT TESTABLE') for v in c2a.verdict), c2a
c2u = rep[(rep.target == 'USA') & rep.claim.str.startswith('C2') & (rep.group == 'confirmatory')]; print('   USA C2 ->', c2u.verdict.tolist())
print('DRY RUN OK')
u_ = d[d.target == 'USA']; print('6) USA rows scored with the fixed class set:', u_.main_metric.value_counts().to_dict(), '| fixed4_clean_f1 == main_f1:', bool(np.allclose(pd.to_numeric(u_.fixed4_clean_f1), pd.to_numeric(u_.main_f1))))
assert (u_.main_metric == 'fixed4_clean_macro_f1').all()
# 7) a results file written under the OLD rule (coverage-conditional main score) is refused by the run cell until re-scored from the SAVED predictions; the re-score must give the same fixed-class scores the run produced
ref = d[d.target == 'USA'].set_index('run_id').fixed4_clean_f1.astype(float)
o_ = d.copy(); m_ = o_.target == 'USA'; o_.loc[m_, 'main_f1'] = pd.to_numeric(o_.loc[m_, 'cov_cond_primary_f1']); o_.loc[m_, 'main_metric'] = 'coverage_conditional_macro_f1'; o_.loc[m_, 'fixed4_f1'] = np.nan; o_.loc[m_, 'fixed4_clean_f1'] = np.nan; o_.loc[m_, 'main_f1_before_fixed4'] = np.nan; o_.loc[m_, 'main_f1_before_clean'] = np.nan
o_.to_csv(csv, index=False)
try: ns['scoring_consistency_check'](csv); print('GATE FAIL: mixed file accepted'); sys.exit(1)
except AssertionError as e: print('7) old-rule USA rows refused ->', str(e)[:60])
with contextlib.redirect_stdout(io.StringIO()): rr = ns['rescore_usa_fixed4'](ns['OUT_DIR'])
new = pd.read_csv(csv).set_index('run_id'); chk = (new.loc[ref.index, 'main_f1'].astype(float) - ref).abs().max()
print('8) re-scored', len(rr), 'USA rows from saved predictions; max |difference| to the clean score computed at run time =', float(chk)); assert len(rr) == len(ref) and chk < 1e-9
assert ns['scoring_consistency_check'](csv) is True; print('DRY RUN 2 OK')

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); L1 = os.path.abspath(os.path.join(HERE, '..')); CELLS = os.path.join(L1, 'src', 'cells'); CLOSING = os.path.join(L1, 'closing')
os.chdir(CELLS)      # the cell files are read by relative name
import tempfile, os, io, contextlib, numpy as np, pandas as pd
ns={'__name__':'__main__'}
def X(f, src=None): exec(compile(src or open(f).read(), f, 'exec'), ns)
X('c02_config.py'); X('c04_core.py'); X('c07_tests.py'); X('c09_data.py')
assert ns['TESTS_A_PASSED']
td=tempfile.mkdtemp(); ns.update(OUT_DIR=os.path.join(td,'outputs','phase3_level1_v3_fixedmix'), GIT_COMMIT=ns['EXPECTED_CROPNET_COMMIT']); os.makedirs(ns['OUT_DIR'])
src=open('c11_acquire.py').read(); X('ns_guard', src[src.index('# ---- namespace guard'):src.index('check_namespace(write=False)\nprint(')])
ns.update(BUDGETS=[1000], SEEDS=[0,1,2], MIN_STATE_ROWS=300, TARGETS_TO_RUN=['ARG'], AUS_SPEC_CONFIRMED=False)
st,cr=ns['_synthetic_usa'](14,2500); yid=np.array([ns['CROP_ID'][c] for c in cr]); rng=np.random.default_rng(0); mu=rng.normal(0,1,(7,730)).astype(np.float32)
feats=lambda y:(mu[y]+rng.normal(0,1.5,(len(y),730))).astype(np.float32).reshape(len(y),1,10,73)
meta=pd.DataFrame({'npz_row':np.arange(len(st))*2,'crop_id':yid,'state':st,'crop':cr}); INFO={}; ns['FEATURES']={'USA':feats(yid)}; ns['NPZ_LBL']={'USA':yid}
for t in ['ARG','FRA','CHN','AUS']:
    y=np.array([ns['CROP_ID'][c] for c in ns['DEFINED_CLASSES'][t] for _ in range(80)]); ns['FEATURES'][t]=feats(y); ns['NPZ_LBL'][t]=y; INFO[t]=dict(keep=np.arange(len(y))*3)
ns.update(meta=meta, INFO=INFO, TESTS_PASSED=True, SMOKE_OK=True, IN_NOTEBOOK=False, GEOGRAPHY_OBSERVED={}, PLAN_NOTE_ARG='paper sec.4.4 says no alignment for ARG; the released code maps wheat->other. Both are scored; no choice is made.')
ns['APPROVED']['plan']=True
# synthetic world has sugarcane/rice in letter-named states -> use permissive geography only for this synthetic run (strictness is tested separately)
buf=io.StringIO()
ns['APPROVED']['aug_on']=True; ns.update(KC_COMBOS=2, AUG_COMBOS=2)
with contextlib.redirect_stdout(buf): X('c15_plan.py')
pl=buf.getvalue(); print('PLAN CELLS RUN', len(ns['PLAN_CELLS_RUN']), 'seeds', ns['SEEDS'])
from sklearn.linear_model import LogisticRegression
ns['EVAL_X']={t:ns['FEATURES'][t] for t in ['ARG','FRA','CHN','AUS']}; ns['EVAL_Y']=ns['NPZ_LBL']
def fit(Xtr,ytr,Xva,yva,seed,ep=None,aug='off'):
    m=LogisticRegression(max_iter=60,C=0.05).fit(Xtr.reshape(len(Xtr),-1),ytr); v=float(ns['f1_score'](yva,m.predict(Xva.reshape(len(Xva),-1)),average='macro'))
    return dict(model=m,info=dict(val_f1=v,best_epoch=7,epochs_run=22,stop_reason='early_stop_patience',restore_delta=0.0,history=[v]))
ns['FIT_FN']=fit; ns['PREDICT_FN']=lambda m,X: m.predict(X.reshape(len(X),-1))

ns['AUTO_MODE']=True; ns['AUTO_MAX_HOURS']=None; ns['APPROVED']['plan']=True
with contextlib.redirect_stdout(buf): X('c19_run.py')
csv=os.path.join(ns['OUT_DIR'],'results.csv'); d=ns['load_results'](csv); d['seed']=pd.to_numeric(d.seed)
print('A) run ->',ns['AUTO_RESULT'],'| ok',int((d.status=='ok').sum()),'| per-seed',d.groupby('seed').size().to_dict(),'| expected',[len(ns['plan_for_seed'](s)) for s in ns['SEEDS']])
print('B) kcenter rows per seed',d[d.sample_sel=='kcenter'].groupby('seed').size().to_dict(),'| aug_copy per seed',d[d.aug=='copy'].groupby('seed').size().to_dict())
assert ns['plan_consistency_check']()['ok']
print('C) manifest amendments:',[a.get('id') for a in __import__('json').load(open(os.path.join(ns['OUT_DIR'],'manifest.json'))).get('amendments',[])])
print('D) env files:',sorted(f for f in os.listdir(ns['OUT_DIR']) if f.startswith('run_environment')))
# canary on identical code -> OK
buf2=io.StringIO()
with contextlib.redirect_stdout(buf2): cc=ns['canary_check']()
print('E) canary same code ->',cc['ok'],cc['note'],[round(x['abs_diff'],9) for x in cc['rows']])
# design drift: the fit function changes -> canary must fail (same device)
def fit2(Xtr,ytr,Xva,yva,seed,ep=None,aug='off'):
    m=LogisticRegression(max_iter=60,C=5.0).fit(Xtr.reshape(len(Xtr),-1)[:, ::2].repeat(2,axis=1),ytr); 
    v=float(ns['f1_score'](yva,m.predict(Xva.reshape(len(Xva),-1)[:, ::2].repeat(2,axis=1)),average='macro'))
    return dict(model=m,info=dict(val_f1=v,best_epoch=7,epochs_run=19,stop_reason='early_stop_patience',restore_delta=0.0,history=[v]))
ns['FIT_FN']=fit2; ns['PREDICT_FN']=lambda m,X: m.predict(X.reshape(len(X),-1)[:, ::2].repeat(2,axis=1))
cc2=ns['canary_check']()
print('F) canary after the fit changed ->',cc2['ok'],cc2['note'])
with contextlib.redirect_stdout(buf2): g=ns['run_guards']()
print('G) run_guards ->',g)
# plan drift: budget list changed
ns['FIT_FN']=fit; ns['PREDICT_FN']=lambda m,X: m.predict(X.reshape(len(X),-1))
cell=ns['PLAN_CELLS_RUN'][0]; saved=ns['PLAN_CELLS_RUN']; ns['PLAN_CELLS_RUN']=[dict(c,budget=c['budget']+1) if i==0 else c for i,c in enumerate(saved)]
with contextlib.redirect_stdout(buf2): g2=ns['plan_consistency_check']()
print('H) plan drift ->',g2['ok'],g2['msg'][:120]); ns['PLAN_CELLS_RUN']=saved

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); L1 = os.path.abspath(os.path.join(HERE, '..')); CELLS = os.path.join(L1, 'src', 'cells'); CLOSING = os.path.join(L1, 'closing')
os.chdir(CELLS)      # the cell files are read by relative name
import tempfile, os, io, contextlib, numpy as np, pandas as pd
ns={'__name__':'__main__'}
def X(f, src=None): exec(compile(src or open(f).read(), f, 'exec'), ns)
X('c02_config.py', open('c02_config.py').read().replace("PROFILE = 'main'  ","PROFILE = 'extension'  ",1)); X('c04_core.py'); X('c07_tests.py'); X('c09_data.py')
assert ns['TESTS_A_PASSED']
td=tempfile.mkdtemp(); ns.update(MAIN_OUT_DIR=os.path.join(td,'outputs','phase3_level1_v3_fixedmix'), GIT_COMMIT=ns['EXPECTED_CROPNET_COMMIT']); ns['OUT_DIR']=os.path.join(ns['MAIN_OUT_DIR'],'extension'); os.makedirs(ns['OUT_DIR'])
src=open('c11_acquire.py').read(); X('ns_guard', src[src.index('# ---- namespace guard'):src.index('check_namespace(write=False)\nprint(')])
ns.update(SEEDS=[0,1], MIN_STATE_ROWS=300, AUS_SPEC_CONFIRMED=False); print('PROFILE',ns['PROFILE'],ns['BUDGETS'],ns['TARGETS_TO_RUN'],ns['ALLOC_RULE'])
st,cr=ns['_synthetic_usa'](14,2500); yid=np.array([ns['CROP_ID'][c] for c in cr]); rng=np.random.default_rng(0); mu=rng.normal(0,1,(7,730)).astype(np.float32)
feats=lambda y:(mu[y]+rng.normal(0,1.5,(len(y),730))).astype(np.float32).reshape(len(y),1,10,73)
meta=pd.DataFrame({'npz_row':np.arange(len(st))*2,'crop_id':yid,'state':st,'crop':cr}); INFO={}; ns['FEATURES']={'USA':feats(yid)}; ns['NPZ_LBL']={'USA':yid}
for t in ['FRA','CHN']:
    y=np.array([ns['CROP_ID'][c] for c in ns['DEFINED_CLASSES'][t] for _ in range(80)]); ns['FEATURES'][t]=feats(y); ns['NPZ_LBL'][t]=y; INFO[t]=dict(keep=np.arange(len(y))*3)
ns.update(meta=meta, INFO=INFO, TESTS_PASSED=True, SMOKE_OK=True, IN_NOTEBOOK=False, GEOGRAPHY_OBSERVED={}, PLAN_NOTE_ARG='paper sec.4.4 says no alignment for ARG; the released code maps wheat->other. Both are scored; no choice is made.')
ns['APPROVED']['plan']=True
# synthetic world has sugarcane/rice in letter-named states -> use permissive geography only for this synthetic run (strictness is tested separately)
buf=io.StringIO()
with contextlib.redirect_stdout(buf): X('c15_plan.py')
pl=buf.getvalue(); print(pl[pl.index('AUTHORS'):][:1900])
from sklearn.linear_model import LogisticRegression
ns['EVAL_X']={t:ns['FEATURES'][t] for t in ['FRA','CHN']}; ns['EVAL_Y']=ns['NPZ_LBL']
def fit(Xtr,ytr,Xva,yva,seed,ep=None,aug='off'):
    m=LogisticRegression(max_iter=60,C=0.05).fit(Xtr.reshape(len(Xtr),-1),ytr); v=float(ns['f1_score'](yva,m.predict(Xva.reshape(len(Xva),-1)),average='macro'))
    return dict(model=m,info=dict(val_f1=v,best_epoch=7,epochs_run=22,stop_reason='early_stop_patience',restore_delta=0.0,history=[v]))
ns['FIT_FN']=fit; ns['PREDICT_FN']=lambda m,X: m.predict(X.reshape(len(X),-1))
ns['RUN_SEED']=1
try: X('c19_run.py'); print('!! seed1 ran unapproved')
except AssertionError as e: print('approval gate OK:',str(e)[:60])
ns['RUN_SEED']=0
with contextlib.redirect_stdout(buf): X('c19_run.py')
print('GATE REPORT (excerpt):'); rep=open(os.path.join(ns['OUT_DIR'],'gate_reports','seed0.txt')).read(); print(rep[rep.index('-- TRAINING'):][:3300])
ns['APPROVED']['seed0_gate_report']=True; ns['RUN_SEED']=1
with contextlib.redirect_stdout(buf): X('c19_run.py')
out=io.StringIO()
with contextlib.redirect_stdout(out): X('c21_summary.py')
o=out.getvalue(); i=o.index('HEADLINE'); print('....'); print(o[i:i+1700]); print('....'); print(o[o.index('optimizer steps'):][:900])
df=ns['load_results'](os.path.join(ns['OUT_DIR'],'results.csv')); print('status',df.status.value_counts().to_dict(),'| rows',len(df))
print('tables:',sorted(f for f in os.listdir(os.path.join(ns['OUT_DIR'],'tables')) if f.startswith(('cell_stats__','gain_','regression__')))[:6],'... both tracks present:',
      all(any(f'__{m}.csv' in f for f in os.listdir(os.path.join(ns['OUT_DIR'],'tables'))) for m in ['primary_f1','review_f1']))
print('columns:', df[['profile','state_rows','max_state_share']].head(4).to_string()); print('arms', df['arms'].value_counts().to_dict()); print('files', sorted(os.listdir(ns['OUT_DIR']))); print('main folder untouched files:', sorted(os.listdir(ns['MAIN_OUT_DIR'])))
print('EXT banner in summary:', 'EXTENSION APPENDIX' in o, '| run_id sample:', df.run_id.iloc[0])

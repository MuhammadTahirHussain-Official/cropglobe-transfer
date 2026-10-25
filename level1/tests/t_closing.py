import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); L1 = os.path.abspath(os.path.join(HERE, '..')); CELLS = os.path.join(L1, 'src', 'cells'); CLOSING = os.path.join(L1, 'closing')
import sys, tempfile, os, io, contextlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
ns={'__name__':'__main__'}
for f in [''+CELLS+'/c02_config.py',''+CELLS+'/c04_core.py',CLOSING+'/c30_closing.py']: exec(compile(open(f).read(),f,'exec'),ns)
P = os.environ.get('LEVEL1_RESULTS_CSV', '')      # path of the frozen Level 1 results.csv (not part of the repository)
if not os.path.exists(P): print('SKIPPED: set LEVEL1_RESULTS_CSV to the frozen Level 1 results.csv to run these tests'); sys.exit(0)
res=[]
def T(name):
    def d(fn):
        try: fn(); res.append(('PASS',name))
        except Exception as e: res.append(('FAIL',name+' -> '+repr(e)[:200]))
        return fn
    return d
@T('C-01 freeze check passes on the real final file')
def _():
    v=ns['verify_freeze'](P); assert v['ok'] and v['hash_matches'] and v['per_seed_ok'], v
@T('C-02 freeze check STOPS when one byte / one row changes')
def _():
    with tempfile.TemporaryDirectory() as td:
        q=os.path.join(td,'r.csv'); s=open(P).read(); open(q,'w').write(s.replace('ok','ok',1)); assert ns['verify_freeze'](q)['ok']
        d=pd.read_csv(P); d.loc[5,'main_f1']=d.loc[5,'main_f1']+0.001; d.to_csv(q,index=False); v=ns['verify_freeze'](q); assert (not v['ok']) and not v['hash_matches']
        d=pd.read_csv(P).iloc[:-1]; d.to_csv(q,index=False); v=ns['verify_freeze'](q); assert (not v['ok']) and not v['rows_ok']
@T('C-03 robustness table reproduces the hand-checked figures on the real file')
def _():
    df=ns['load_final'](P); rb,dep=ns['robustness_table'](df,n_boot=300)
    r=rb.set_index('budget'); assert abs(r.loc[1000,'plain_gain']-9.97)<0.02 and abs(r.loc[2000,'plain_gain']-5.03)<0.02 and abs(r.loc[1000,'copy_control_gain']-9.11)<0.02 and abs(r.loc[2000,'copy_control_gain']-7.86)<0.02 and abs(r.loc[1000,'augmentation_gain']-9.08)<0.02 and abs(r.loc[2000,'augmentation_gain']-7.53)<0.02, rb
    assert abs(dep['slope']-0.30)<0.03 and abs(dep['p']-0.278)<0.01 and dep['n_combinations']==57, dep
    assert (rb.plain_lo<rb.plain_gain).all() and (rb.plain_gain<rb.plain_hi).all()
@T('C-04 own REML equals an independent dense-matrix REML on the real data (Argentina, China, France)')
def _():
    from scipy import optimize
    df=ns['load_final'](P); base=ns['_random_arm_runs'](df); base['y']=100*pd.to_numeric(base.main_f1)
    for t,(c,s) in {'ARG':(2.3418,0.3003),'CHN':(4.466,1.174),'FRA':(1.898,1.344)}.items():
        g,X,names=ns['_mixed_design'](base[base.target==t]); r=ns['reml_random_intercept'](g.y.values,X,g.combo.values); assert abs(r['beta'][1]-c)<0.002 and abs(np.sqrt(r['cov'][1,1])-s)<0.002,(t,r['beta'][1])
        if t=='CHN':          # dense check inside the test itself
            y=g.y.values;n=len(y);code=pd.factorize(g.combo)[0];Z=(code[:,None]==np.arange(code.max()+1)[None]).astype(float);ZZ=Z@Z.T
            def nll(th):
                su,se=np.exp(th);V=su*ZZ+se*np.eye(n);L=np.linalg.cholesky(V);A=X.T@np.linalg.solve(V,X);b=np.linalg.solve(A,X.T@np.linalg.solve(V,y));rr=y-X@b
                return .5*(2*np.log(np.diag(L)).sum()+np.linalg.slogdet(A)[1]+rr@np.linalg.solve(V,rr))
            o=optimize.minimize(nll,np.log([1,10]),method='Nelder-Mead',options=dict(xatol=1e-8,fatol=1e-10,maxiter=4000)); su,se=np.exp(o.x); V=su*ZZ+se*np.eye(n); Vi=np.linalg.inv(V); A=X.T@Vi@X; b=np.linalg.solve(A,X.T@Vi@y)
            assert abs(b[1]-r['beta'][1])<1e-3 and abs(su-r['s2u'])<0.05*max(su,1e-9)+1e-3
@T('C-05 own_vs_statsmodels runs without statsmodels and says NOT RUN (never claims agreement it did not check)')
def _():
    df=ns['load_final'](P); out=ns['own_vs_statsmodels'](df,try_install=False)
    try: import statsmodels; has=True
    except Exception: has=False
    if not has: assert len(out)==3 and out.agree.str.startswith('NOT RUN').all() and out.sm_coef.isna().all(), out
    else: assert len(out)==3
@T('C-06 robustness_table detects a planted dependence: copy gain that grows with the number of states')
def _():
    rng=np.random.default_rng(0); rows=[]
    for B in (1000,2000):
        for k in (1,2,4,8):
            for ci in range(10 if k>1 else 8):
                st=f'S{ci}' if k==1 else f'S{ci}-{k}'
                base=60+2*np.log2(k)+rng.normal(0,.5)
                for trk,extra in (('base',0),('aug_copy',2+1.5*np.log2(k)),('aug_on',2+1.5*np.log2(k)+1)):
                    rows.append(dict(target='ARG',budget=B,k=k,states=st,track=trk,y=base+extra+rng.normal(0,.3),seed=0))
    rb,dep=ns['robustness_table'](pd.DataFrame(rows),n_boot=200); assert dep['slope']>1.0 and dep['p']<0.001,dep; assert (rb.copy_control_gain>rb.plain_gain).all()
for r,n in res: print(f'  [{r}] {n}')
print('TOTAL',len(res),'FAIL',sum(1 for r,_ in res if r=='FAIL'))
# --- control-flow test of the optimiser fallback with a STAND-IN for statsmodels (this tests our loop, NOT statsmodels itself)
import types
class _R:
    def __init__(s,c,se): s.params={'lk':c}; s.bse={'lk':se}; s.pvalues={'lk':0.01}
def _mk(fail_methods, scale_must=None):
    calls=[]
    class M:
        def __init__(s,f,d,groups): s.d=d
        def fit(s,reml=True,method='lbfgs'):
            calls.append((method,float(s.d['y'].abs().max())))
            if method in fail_methods: raise np.linalg.LinAlgError('Singular matrix')
            sc=1.0 if s.d['y'].abs().max()>50 else 0.1; return _R(2.342*sc,0.300*sc)
    return types.SimpleNamespace(mixedlm=lambda f,d,groups: M(f,d,groups)),calls
@T('C-07 optimiser fallback: first optimisers raise LinAlgError -> the next one is used and the method is reported; all failing -> a plain message, never a made-up agreement')
def _():
    import sys as _s; df=ns['load_final'](P)
    for fails,expect in ((('lbfgs','bfgs'),'powell'),(('lbfgs','bfgs','powell','nm','cg'),None)):
        stub,calls=_mk(fails); mod=types.ModuleType('statsmodels'); sub=types.ModuleType('statsmodels.formula'); api=types.ModuleType('statsmodels.formula.api'); api.mixedlm=stub.mixedlm; mod.formula=sub; sub.api=api
        _s.modules.update({'statsmodels':mod,'statsmodels.formula':sub,'statsmodels.formula.api':api})
        try: out=ns['own_vs_statsmodels'](df,targets=('ARG',),try_install=False)
        finally:
            for k in ('statsmodels','statsmodels.formula','statsmodels.formula.api'): _s.modules.pop(k,None)
        r=out.iloc[0]
        if expect: assert r.sm_method==expect and r.agree=='YES' and abs(r.sm_coef-2.342)<1e-6, r.to_dict()
        else: assert np.isnan(r.sm_coef) and r.agree.startswith('statsmodels fit failed with every optimiser') and len(calls)==10, (r.to_dict(),len(calls))
for r,n in res[-1:]: print(f'  [{r}] {n}')
print('TOTAL',len(res),'FAIL',sum(1 for r,_ in res if r=='FAIL'))

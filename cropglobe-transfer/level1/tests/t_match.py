import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); L1 = os.path.abspath(os.path.join(HERE, '..')); CELLS = os.path.join(L1, 'src', 'cells'); CLOSING = os.path.join(L1, 'closing')
import os, glob, numpy as np, pandas as pd, tempfile, time, warnings; warnings.filterwarnings('ignore')
ns={'__name__':'__main__'}; exec(compile(open(CLOSING+'/c33_match.py').read(),'c33','exec'),ns)
U = os.environ.get('CROPGLOBE_FILES_DIR', '').rstrip('/') + '/'      # optional folder with the real raw France CSVs (only used for a printout)
files=sorted(glob.glob(U+'*S2_FRA_other_LAMB93_R*.csv')); print([os.path.basename(f)[-12:] for f in files])
if not files: print('SKIPPED: this test is built on the real raw France CSVs; set CROPGLOBE_FILES_DIR to a folder that contains S2_FRA_other_LAMB93_R*.csv'); sys.exit(0)
t0=time.time(); pts=[ns['read_raw_points'](f) for f in files]; print('parsed',sum(len(p['obs']) for p in pts),'points in',round(time.time()-t0,1),'s | names',[ (p['country'],p['crop_token'],p['region_code']) for p in pts])
# bin occupancy under each rule
for rule in ns['BIN_RULES']:
    occ=[ (~np.isnan(ns['bin_medians'](o,rule)[0])).sum() for p in pts for o in p['obs'][:200]]; print(rule,'non-empty bins of 73: mean %.1f min %d'%(np.mean(occ),min(occ)))
# synthetic npz built from the REAL raw points with an ASSUMED rule (floor(doy/5), scale 1e-4, band-major, linear interpolation of empty bins), plus decoys
def build(rule,scale,layout,shuffle=False,decoys=300,seed=0):
    rng=np.random.default_rng(seed); rows=[]
    for p in pts:
        for o in p['obs']:
            m=ns['bin_medians'](o,rule); 
            for b in range(10):
                x=np.arange(73); ok=~np.isnan(m[b]); m[b]=np.interp(x,x[ok],m[b][ok]) if ok.any() else 0
            rows.append(m*scale)
    A=np.array(rows); A=np.vstack([A,rng.normal(A.mean(),A.std(),(decoys,10,73))])
    A=A.reshape(len(A),730) if layout=='band-major (10,73)' else A.transpose(0,2,1).reshape(len(A),730)
    return A.astype('float32')
res=[]
def T(n):
    def d(f):
        try: f(); res.append(('PASS',n))
        except Exception as e: res.append(('FAIL',n+' -> '+repr(e)[:300]))
    return d
with tempfile.TemporaryDirectory() as td:
    A=build('floor(doy/5)',1e-4,'band-major (10,73)'); np.savez(td+'/a.npz',fea=A,lbl=np.zeros(len(A),int))
    @T('M-01 finds the planted rule (floor(doy/5), raw/10000, band-major), accepts it, and maps every real raw point to its true npz row')
    def _():
        r=ns['run_match'](files,td+'/a.npz',td+'/out'); assert r['accepted']; b=r['search'].iloc[0]; assert (b['rule'],b['layout'],b['scale'])==('floor(doy/5)','band-major (10,73)','raw/10000'),b
        m=r['map']; assert len(m)==sum(len(p['obs']) for p in pts) and (m.n_match==1).mean()>0.99
        true=np.arange(len(m)); got=m.npz_row.values; assert (got[m.n_match==1]==true[m.n_match==1]).all(), 'a matched row points to the wrong npz row'
        ev=r['evidence']; assert ev.increasing.all()
    # same data, different (time-major) layout and different rule
    A2=build('floor((doy-1)/5)',1.0,'time-major (73,10)'); np.savez(td+'/b.npz',fea=A2)
    @T('M-02 finds a DIFFERENT planted rule (floor((doy-1)/5), raw units, time-major)')
    def _():
        r=ns['run_match'](files[:2],td+'/b.npz',td+'/out2'); b=r['search'].iloc[0]; assert r['accepted'] and (b['rule'],b['layout'],b['scale'])==('floor((doy-1)/5)','time-major (73,10)','raw units'),b
    # an npz that was NOT built from these raw points (random) must be refused, never matched
    rng=np.random.default_rng(1); np.savez(td+'/c.npz',fea=rng.normal(0.2,0.1,(3000,730)).astype('float32'))
    @T('M-03 refuses an unrelated npz: no rule reaches 99%, STOP message, no row map written')
    def _():
        r=ns['run_match'](files,td+'/c.npz',td+'/out3'); assert (not r['accepted']) and not os.path.exists(td+'/out3/row_map.csv') and r['search'].iloc[0]['unique']<0.05
    # an npz built with a rule outside the declared set (bins of 7 days) must also be refused
    pts_bak=pts
    def b7(o):
        m=np.full((10,73),np.nan); b=np.clip(np.floor(o[:,0]/7).astype(int),0,72)
        for k in np.unique(b): m[:,k]=np.median(o[b==k,1:11],axis=0)
        for j in range(10):
            ok=~np.isnan(m[j]); m[j]=np.interp(np.arange(73),np.arange(73)[ok],m[j][ok])
        return m
    A3=np.array([b7(o) for p in pts for o in p['obs']]).reshape(-1,730).astype('float32'); np.savez(td+'/d.npz',fea=A3)
    @T('M-04 refuses an npz built with an undeclared rule (7-day bins) instead of forcing a match')
    def _():
        r=ns['run_match'](files[:1],td+'/d.npz',td+'/out4'); assert not r['accepted'], r['search'].head(3)
@T('M-05 label_and_coverage: one label per crop token, coverage share is correct, a token mapped to two labels is visible')
def _():
    import pandas as pd
    m=pd.DataFrame(dict(crop_token=['a']*3+['b']*2,npz_row=[0,1,2,3,4],n_match=1)); c,t=ns['label_and_coverage'](m,10,np.array([7,7,7,9,9]+[0]*5)); assert c['share']==0.5 and (t.labels_for_this_token==1).all()
    c,t=ns['label_and_coverage'](m,10,np.array([7,7,8,9,9]+[0]*5)); assert t[t.crop_token=='a'].labels_for_this_token.max()==2
@T('M-06 file-name parser: France names, upload-hash prefix, USA-style names (with and without state), unknown names')
def _():
    P=ns['parse_name']; assert P('S2_FRA_other_LAMB93_R76.csv')==dict(country='FRA',crop_token='other',region_code='76'); assert P('10f02bba-1792560435559_S2_FRA_winterwheat_LAMB93_R24.csv')['crop_token']=='winterwheat'
    assert P('S2_USA_springwheat_ND.csv')==dict(country='USA',crop_token='springwheat',region_code='ND'); assert P('S2_USA_corn.csv')['region_code'] is None and P('S2_USA_corn.csv')['crop_token']=='corn'; assert P('x.csv')['country'] is None
@T('M-07 a raw file with a different layout (no raw_data column) stops with a clear message, no crash deeper in')
def _():
    import pandas as pd, tempfile
    with tempfile.TemporaryDirectory() as td:
        pd.DataFrame({'a':[1],'.geo':['{}']}).to_csv(td+'/S2_USA_corn_IA.csv',index=False)
        try: ns['read_raw_points'](td+'/S2_USA_corn_IA.csv'); assert False
        except ValueError as e: assert 'raw layout differs' in str(e) and 'raw_data' in str(e)
for r,n in res: print(f'  [{r}] {n}')
print('TOTAL',len(res),'FAIL',sum(1 for r,_ in res if r=='FAIL'))

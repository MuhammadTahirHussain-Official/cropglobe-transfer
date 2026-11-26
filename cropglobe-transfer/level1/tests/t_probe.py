import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); L1 = os.path.abspath(os.path.join(HERE, '..')); CELLS = os.path.join(L1, 'src', 'cells'); CLOSING = os.path.join(L1, 'closing')
import os, json, tempfile, numpy as np, pandas as pd
ns={'__name__':'__main__'}; exec(compile(open(CLOSING+'/c31_probe.py').read(),'c31','exec'),ns)
res=[]
def T(n):
    def d(f):
        try: f(); res.append(('PASS',n))
        except Exception as e: res.append(('FAIL',n+' -> '+repr(e)[:250]))
    return d
with tempfile.TemporaryDirectory() as td:
    os.makedirs(td+'/data/for_CropNet'); os.makedirs(td+'/raw')
    np.savez(td+'/data/for_CropNet/median_S2_USA_month1-12_day5.npz',fea=np.random.rand(50,730).astype('float32'),lbl=np.repeat([1,2],25))
    np.savez(td+'/data/for_CropNet/median_S2_FRA_month1-12_day5.npz',fea=np.random.rand(30,730).astype('float32'),lbl=np.repeat([1,2,3],10),region=np.arange(30))
    pd.DataFrame({'system:index':range(5),'.geo':['{"type":"Point","coordinates":[-90.1,40.2]}']*5,**{f'b{i}':np.random.rand(5) for i in range(730)}}).to_csv(td+'/raw/S2_USA_corn_IA.csv',index=False)
    pd.DataFrame({'.geo':['{}']*5,**{f'b{i}':np.random.rand(5) for i in range(60)}}).to_csv(td+'/raw/S2_USA_soybeans.csv',index=False)
    @T('P-01 probe reads npz keys, shapes, label counts and flags coordinate/region-like keys')
    def _():
        a=ns['probe_npz'](td+'/data/for_CropNet/median_S2_FRA_month1-12_day5.npz'); assert a['keys']['fea']['shape']==[30,730] and a['has_coordinate_like_key']==['region'] and a['label_counts']=={'1':10,'2':10,'3':10}
        b=ns['probe_npz'](td+'/data/for_CropNet/median_S2_USA_month1-12_day5.npz'); assert b['has_coordinate_like_key']==[]
    @T('P-02 probe_geo_join: equal numeric width => says a match can be attempted; unequal => says it will NOT guess')
    def _():
        u=td+'/data/for_CropNet/median_S2_USA_month1-12_day5.npz'
        a=ns['probe_geo_join'](td+'/raw/S2_USA_corn_IA.csv',u); assert a['n_numeric_columns']>=730 and a['has_geo_column']
        b=ns['probe_geo_join'](td+'/raw/S2_USA_soybeans.csv',u); assert b['numeric_columns_equal_feature_width'] is False and 'not blind' in b['verdict']
    @T('P-03 full run_probe writes the report even when the internet is unreachable (graceful, no crash, no invented values)')
    def _():
        ns['REGION_SOURCES'].clear(); ns['REGION_SOURCES']['bad']='https://invalid.invalid/x.json'
        r=ns['run_probe'](td+'/data',td+'/raw',None,td+'/probe_report.json'); j=json.load(open(td+'/probe_report.json'))
        assert j['region_sources']['bad']['ok'] is False and j['raw_fra']=='directory not given / not found' and len(j['npz'])==2 and 'usa_join_probe' in j
for r,n in res: print(f'  [{r}] {n}')
print('TOTAL',len(res),'FAIL',sum(1 for r,_ in res if r=='FAIL'))

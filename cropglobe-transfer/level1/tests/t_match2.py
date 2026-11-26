import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); L1 = os.path.abspath(os.path.join(HERE, '..')); CELLS = os.path.join(L1, 'src', 'cells'); CLOSING = os.path.join(L1, 'closing')
import numpy as np, pandas as pd
ns={'__name__':'__main__'}; exec(compile(open(CLOSING+'/c33_match.py').read(),'c33','exec'),ns)
U = os.environ.get('CROPGLOBE_FILES_DIR', '').rstrip('/') + '/'      # folder with the verified France row map
if not os.path.exists(U + 'row_map.csv'): print('SKIPPED: put the verified row_map.csv in CROPGLOBE_FILES_DIR to run this check'); sys.exit(0)
m=pd.read_csv(U+'row_map.csv'); lab={'corn':0,'soybeans':1,'rice':2,'springwheat':3,'winterwheat':3,'other':6}
# synthetic label vector consistent with the real row map (label of each npz row = label of its token)
lbl=np.full(108599,-1); u=m[m.n_match==1]; lbl[u.npz_row.values]=u.crop_token.map(lab).values
oc=ns['order_candidates'](m,lbl); print(oc.to_string(index=False))
assert oc.iloc[0].identical_order and not oc.iloc[1].identical_order and not oc.iloc[2].identical_order and not oc.iloc[3].identical_order
# subset (wheat files only): still identifies the right convention
w=m[m.crop_token.isin(['springwheat','winterwheat'])]; oc2=ns['order_candidates'](w,lbl); print(oc2.to_string(index=False)); assert oc2.iloc[0].identical_order and not oc2.iloc[1].identical_order and not oc2.iloc[2].identical_order
print('ORDER TESTS PASS')

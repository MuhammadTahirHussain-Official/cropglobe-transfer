import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(ROOT, 'src', 'cells'))      # the cell files are read by relative name
import os, io, json, glob, tempfile, contextlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
ns = {'__name__': '__main__', 'OUT_DIR': __import__('tempfile').gettempdir(), 'MAIN_OUT_DIR': __import__('tempfile').gettempdir()}
for f in ['c02_config.py', 'c04_core.py', 'c09_data.py']:
    with contextlib.redirect_stdout(io.StringIO()): exec(compile(open(f).read(), f, 'exec'), ns)
src = open('c11b_usa_order.py').read(); src = src[:src.index('run_usa_order_cell(PROJECT_ROOT, DATA_DIR, OUT_DIR)')]
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, 'c11b', 'exec'), ns)
CID = ns['CROP_ID']; RTC = ns['RAW_TO_CROP']
PLAN = [('corn', ['IA', 'IL']), ('soybeans', ['IA', 'IL', 'MN']), ('springwheat', ['ND']), ('winterwheat', ['KS', 'OK']), ('other', ['NY', 'TX']), ('rice', ['AR'])]
def make_world(td, npts=60, seed=0, swap_soy_blocks=False, mislabel_soy=False, unrelated=False):
    rng = np.random.default_rng(seed); rawdir = os.path.join(td, 'raw'); os.makedirs(rawdir, exist_ok=True); order = sorted([(CID[RTC[t]], t, s) for t, sts in PLAN for s in sts]); blocks = []
    for lab, t, s in order:
        obs_l = []
        for j in range(npts):
            doy = np.sort(rng.choice(np.arange(1, 366), 40, replace=False)); vals = rng.integers(200, 4000, (40, 10)); obs_l.append(np.column_stack([doy, vals]).astype(float))
        blocks.append((lab, t, s, obs_l))
        rows = [dict(**{'system:index': f'{j:04d}', 'information': 'x', 'raw_data': json.dumps(o.tolist()), '.geo': json.dumps({'type': 'Point', 'coordinates': [-90.0 + j * .001, 40.0]})}) for j, o in enumerate(obs_l)]
        pd.DataFrame(rows).to_csv(os.path.join(rawdir, f'S2_USA_{t}_{s}.csv'), index=False)
    seq = list(blocks)
    if swap_soy_blocks:
        i = [k for k, b in enumerate(seq) if b[1] == 'soybeans']; seq[i[0]], seq[i[1]] = seq[i[1]], seq[i[0]]
    feas, labs, meta = [], [], []
    for lab, t, s, obs_l in seq:
        for o in obs_l:
            m = ns['bin_medians'](o, 'floor((doy-1)/5)')
            for b in range(10):
                x = np.arange(73); ok = ~np.isnan(m[b]); m[b] = np.interp(x, x[ok], m[b][ok]) if ok.any() else 0
            feas.append(m.reshape(730)); labs.append(CID['corn'] if (mislabel_soy and t == 'soybeans') else lab); meta.append((CID[RTC[t]], s))
    A = np.array(feas, dtype=np.float32)
    if unrelated: A = rng.normal(2000, 500, A.shape).astype(np.float32)
    ddir = os.path.join(td, 'data'); os.makedirs(ddir, exist_ok=True); np.savez(os.path.join(ddir, 'median_S2_USA_month1-12_day5.npz'), fea=A, lbl=np.array(labs))
    # Level 1 style state table (npz_row, crop_id, state) for the packaged rows (correct unless altered by the caller)
    tab = pd.DataFrame({'npz_row': np.arange(len(meta)), 'crop_id': [m[0] for m in meta], 'state': [m[1] for m in meta]})
    return rawdir, os.path.join(ddir, 'median_S2_USA_month1-12_day5.npz'), tab
res = []
def T(n):
    def d(f):
        try: f(); res.append(('PASS', n))
        except Exception as e: res.append(('FAIL', n + ' -> ' + repr(e)[:400]))
    return d
def run(td, tab=None, **kw):
    rawdir, npz, t = make_world(td, **kw); files = ns['find_usa_raw_files']([rawdir])
    with contextlib.redirect_stdout(io.StringIO()): rep, m = ns['verify_usa_order'](files, npz, os.path.join(td, 'out'), {'level1': tab if tab is not None else t} if tab is not False else {})
    return rep, m, t
with tempfile.TemporaryDirectory() as td:
    @T('U-01 correct data: rule found, every raw point of every crop (soybeans included) matches exactly one npz row, no label mismatch, blocks follow (label, token, state), Level 1 table agrees -> VERIFIED')
    def _():
        rep, m, t = run(td); assert rep['status'].startswith('VERIFIED'), rep['status'] + str(rep['reasons'])
        sb = [r for r in rep['by_crop'] if r['crop'] == 'soybeans'][0]; assert sb['unique'] == sb['raw_points'] == 3 * 60 and sb['label_mismatch'] == 0
        assert rep['order_convention_holds'] and rep['level1_disagreements'] == 0 and rep['npz_rows_matched'] == len(m)
        assert (m.n_match == 1).all()
with tempfile.TemporaryDirectory() as td:
    @T('U-02 a Level 1 table that puts 3 soybean rows in the WRONG state is caught: status names the number of disagreeing rows, the CSV lists them')
    def _():
        rawdir, npz, t = make_world(td); soy = t.index[(t.crop_id == CID['soybeans']) & (t.state == 'IA')][:3]; t2 = t.copy(); t2.loc[soy, 'state'] = 'IL'
        files = ns['find_usa_raw_files']([rawdir])
        with contextlib.redirect_stdout(io.StringIO()): rep, m = ns['verify_usa_order'](files, npz, os.path.join(td, 'out'), {'level1': t2})
        assert rep['level1_disagreements'] == 3 and 'disagrees' in rep['status'] and 'on 3 rows' in rep['status'], rep['status']
        w = pd.read_csv(os.path.join(td, 'out', 'usa_level1_disagreements.csv')); assert len(w) == 3 and set(w.crop_token) == {'soybeans'} and set(w.file_state) == {'IA'} and set(w.l1_state) == {'IL'}
with tempfile.TemporaryDirectory() as td:
    @T('U-03 an npz whose soybean state blocks are in the WRONG order (IA/IL swapped) is FAILED, never verified')
    def _():
        rep, m, t = run(td, swap_soy_blocks=True); assert rep['status'] == 'FAILED' and not rep['order_convention_holds'], rep['status']
with tempfile.TemporaryDirectory() as td:
    @T('U-04 soybean rows carrying the label of another crop are FAILED (label mismatch counted)')
    def _():
        rep, m, t = run(td, mislabel_soy=True); assert rep['status'] == 'FAILED' and any('label different' in r for r in rep['reasons']) and [r for r in rep['by_crop'] if r['crop'] == 'soybeans'][0]['label_mismatch'] == 180
with tempfile.TemporaryDirectory() as td:
    @T('U-05 an npz that is not built from these raw files is refused (no rule accepted); nothing is assumed')
    def _():
        rep, m, t = run(td, unrelated=True); assert rep['status'] == 'FAILED' and m is None and 'no declared rule' in rep['reasons'][0]
with tempfile.TemporaryDirectory() as td:
    @T('U-06 without a Level 1 table the order is still verified but the state comparison is stated as NOT done')
    def _():
        rep, m, t = run(td, tab=False); assert rep['status'].startswith('ORDER VERIFIED') and 'NOT done' in rep['status']
with tempfile.TemporaryDirectory() as td:
    @T('U-07 resumable: a second call reuses the per-file cache and gives the same result')
    def _():
        rep1, m1, t = run(td); n1 = len(glob.glob(os.path.join(td, 'out', 'cache', 'usa_match', '*.csv'))); rep2, m2, _t = run(td)
        assert n1 == 11 and rep1['status'] == rep2['status'] and (m1.npz_row.values == m2.npz_row.values).all()
with tempfile.TemporaryDirectory() as td:
    @T('U-08 no raw files / no npz: status says NOT VERIFIED with what to do, no exception, nothing assumed')
    def _():
        os.makedirs(os.path.join(td, 'p'), exist_ok=True); g = dict(ns); g['glob'] = glob
        class _NoHF: pass
        with contextlib.redirect_stdout(io.StringIO()): r = ns['run_usa_order_cell'](os.path.join(td, 'p'), os.path.join(td, 'p', 'd'), td)
        assert r['status'].startswith('NOT VERIFIED') and 'run this cell again' in r['status']
    @T('U-09 file-name parser: hash prefix ignored; only known crop tokens; two different files with one name STOP')
    def _():
        d2 = os.path.join(td, 'f'); os.makedirs(d2, exist_ok=True)
        for n in ['ab12cd34-1792616877071_S2_USA_soybeans_IA.csv', 'S2_USA_corn_IL.csv', 'S2_USA_banana_IL.csv', 'notes.csv']: open(os.path.join(d2, n), 'w').write('x')
        got = ns['find_usa_raw_files']([d2]); assert list(got) == ['S2_USA_corn_IL.csv', 'S2_USA_soybeans_IA.csv'], list(got)
        d3 = os.path.join(td, 'g'); os.makedirs(d3); open(os.path.join(d3, 'S2_USA_corn_IL.csv'), 'w').write('yy')
        try: ns['find_usa_raw_files']([d2, d3]); raise AssertionError('no stop')
        except RuntimeError as e: assert 'STOP' in str(e)
for s_, n in res: print(s_, n)
print('USA-ORDER TESTS:', sum(1 for s_, _ in res if s_ == 'PASS'), 'pass,', sum(1 for s_, _ in res if s_ != 'PASS'), 'fail')

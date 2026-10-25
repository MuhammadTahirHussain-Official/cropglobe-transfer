# ======================================================================================
# USA ROW-ORDER VERIFICATION (instructor review, Oct 2026): "verify the USA soybean (and corn / other) ordering exactly".  Runs EARLY, before the plan.
# WHY: the Level 1 headline (Argentina soybeans) depends on every USA soybean sample being assigned to the right STATE.  The packaged USA npz has no state column; Level 1 inferred it.
# WHAT THIS CELL DOES (exact, no inference): every raw point of every raw USA file S2_USA_<crop>_<ST>.csv is matched to its packaged npz row by an exact comparison of its 5-day medians
# (the same method that verified USA wheat 23,226 / 23,226 and all of France).  It then checks, with pre-declared criteria:
#   (a) a declared binning rule reproduces the npz (>= 99% on a sample);  (b) >= 99.9% of the raw points of EVERY crop token match exactly one npz row;
#   (c) the npz label of every matched row equals the label of its crop token;  (d) each file is one contiguous, increasing block and the blocks follow (label, token A-Z, state);
#   (e) the state in each raw FILE NAME equals the state in the Level 1 USA state table (cache/usa_meta.csv from the frozen Level 1 run), row by row.
# The result is printed per crop (soybeans first) and saved to tables/usa_order_verification.json.  It is NOT a gate for Level 2 runs (Level 2 uses the USA test set whole, no USA state enters
# any Level 2 run) -- it validates the Level 1 state assignment and is reported in the closing cell.  If the raw files cannot be found the status says NOT VERIFIED; nothing is assumed.
# ======================================================================================
import os, re, json, glob, time
import numpy as np, pandas as pd

BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
N_BINS, N_BANDS = 73, 10
BIN_RULES = {                                    # doy -> bin index 0..72 (declared candidates; clipped into range)
    'floor(doy/5)':        lambda d: np.floor(d / 5.0).astype(int),
    'floor((doy-1)/5)':    lambda d: np.floor((d - 1.0) / 5.0).astype(int),
    'floor((doy+2.5)/5)':  lambda d: np.floor((d + 2.5) / 5.0).astype(int),
}
SCALES = {'raw units': 1.0, 'raw/10000': 1e-4}
LAYOUTS = {'band-major (10,73)': lambda a: a.reshape(len(a), N_BANDS, N_BINS), 'time-major (73,10)': lambda a: a.reshape(len(a), N_BINS, N_BANDS).transpose(0, 2, 1)}
MATCH_ACCEPT = 0.99
MATCH_TOL = 1e-3                                 # in raw units (reflectance integers); scaled with the value scale

def parse_name(path):
    b = re.sub(r'^[0-9a-f]{8}-\d+_', '', os.path.basename(path))            # Drive/upload hash prefix, if any
    m = re.match(r'^S2_([A-Z]{3})_(.+?)_LAMB93_R(\d+)\.csv$', b)             # France: S2_FRA_<crop>_LAMB93_R<region>.csv
    if m: return dict(country=m.group(1), crop_token=m.group(2), region_code=m.group(3))
    m = re.match(r'^S2_([A-Z]{3})_(.+?)(?:_([A-Z]{2}))?\.csv$', b)             # USA-style: S2_USA_<crop>[_<ST>].csv  (best effort: not verified on real USA files)
    return dict(country=m.group(1), crop_token=m.group(2), region_code=m.group(3)) if m else dict(country=None, crop_token=None, region_code=None)

def read_raw_points(path):
    cols = list(pd.read_csv(path, nrows=0).columns); miss = [c for c in ('system:index', 'raw_data', '.geo') if c not in cols]
    if miss: raise ValueError(f'{os.path.basename(path)}: raw layout differs from the one this matcher was written for (missing columns {miss}; columns are {cols[:12]}). Send me this list instead of forcing a match.')
    d = pd.read_csv(path, usecols=['system:index', 'raw_data', '.geo']); obs = [np.asarray(json.loads(x), dtype=float) for x in d['raw_data'].values]
    xy = np.array([json.loads(g)['coordinates'][:2] for g in d['.geo'].values], dtype=float)
    return dict(path=path, index=d['system:index'].values, obs=obs, lon=xy[:, 0], lat=xy[:, 1], **parse_name(path))

def bin_medians(obs, rule):
    # (10, 73) array of per-bin medians, NaN where the bin has no observation
    out = np.full((N_BANDS, N_BINS), np.nan)
    if len(obs) == 0: return out
    b = np.clip(BIN_RULES[rule](obs[:, 0]), 0, N_BINS - 1)
    for k in np.unique(b): out[:, k] = np.median(obs[b == k, 1:1 + N_BANDS], axis=0)
    return out

class NpzIndex:
    # sorted-column index of one (band, bin) pair per bin for fast candidate lookup
    def __init__(self, fea, layout, scale):
        a = LAYOUTS[layout](np.asarray(fea, dtype=np.float32)); self.a = a; self.scale = scale; self.order = {}; self.sorted = {}
        for b in range(N_BINS):
            col = a[:, 0, b]; o = np.argsort(col, kind='stable'); self.order[b] = o; self.sorted[b] = col[o]
    def candidates(self, med, tol):
        have = np.flatnonzero(~np.isnan(med[0])); 
        if len(have) == 0: return None, have
        b = int(have[0]); v = med[0, b] * self.scale; t = tol * self.scale + 1e-9
        lo, hi = np.searchsorted(self.sorted[b], v - t), np.searchsorted(self.sorted[b], v + t, side='right'); cand = self.order[b][lo:hi]
        if len(cand) == 0: return cand, have
        ok = np.abs(self.a[np.ix_(cand, np.arange(N_BANDS), have)] - (med[:, have] * self.scale)[None]).max(axis=(1, 2)) <= t
        return cand[ok], have

def match_points(points, fea, rule, layout, scale, tol=MATCH_TOL, max_points=None, progress=0):
    idx = NpzIndex(fea, layout, scale); rows = []; t0 = time.time(); total = sum(len(p['obs']) for p in points)
    for p in points:
        for j, obs in enumerate(p['obs'][:max_points] if max_points else p['obs']):
            med = bin_medians(obs, rule); c, have = idx.candidates(med, tol)
            if progress and rows and len(rows) % progress == 0: print(f'   ... {len(rows)}/{total} raw points matched ({time.time() - t0:.0f}s)')
            rows.append(dict(file=os.path.basename(p['path']), raw_index=p['index'][j], n_bins_used=len(have), n_match=0 if c is None else len(c), npz_row=int(c[0]) if (c is not None and len(c) == 1) else -1, lon=p['lon'][j], lat=p['lat'][j], region_code=p['region_code'], crop_token=p['crop_token']))
    return pd.DataFrame(rows)

def find_rule(points, fea, sample_per_file=200, seed=0):
    # try every declared (rule, layout, scale) on a small sample; the winner must reach MATCH_ACCEPT unique matches
    rng = np.random.default_rng(seed); sub = []
    for p in points:
        n = len(p['obs']); sel = np.sort(rng.choice(n, min(sample_per_file, n), replace=False)); sub.append(dict(p, obs=[p['obs'][i] for i in sel], index=p['index'][sel], lon=p['lon'][sel], lat=p['lat'][sel]))
    res = []
    for rule in BIN_RULES:
        for layout in LAYOUTS:
            for sc_name, sc in SCALES.items():
                m = match_points(sub, fea, rule, layout, sc); res.append(dict(rule=rule, layout=layout, scale=sc_name, unique=float((m.n_match == 1).mean()), none=float((m.n_match == 0).mean()), ambiguous=float((m.n_match > 1).mean()), n=len(m)))
    r = pd.DataFrame(res).sort_values('unique', ascending=False).reset_index(drop=True); r['accepted'] = r['unique'] >= MATCH_ACCEPT
    return r

def ordering_evidence(m):
    # for matched rows: do raw rows of one file occupy ONE contiguous, increasing block of npz rows?  (this is the join the Level 1/2 pipelines otherwise have to infer)
    out = []
    for f, g in m[m.npz_row >= 0].groupby('file'):
        g = g.sort_values('raw_index'); r = g.npz_row.values; out.append(dict(file=f, n_matched=len(g), npz_row_min=int(r.min()), npz_row_max=int(r.max()), increasing=bool((np.diff(r) > 0).all()) if len(r) > 1 else True, span=int(r.max() - r.min() + 1)))
    return pd.DataFrame(out)

def label_and_coverage(m, n_npz, lbl=None):
    # coverage of the npz by the matched raw points, and (if labels exist) which npz label each crop token received
    u = m[m.n_match == 1]; cov = dict(npz_rows=int(n_npz), matched_unique=int(len(u)), share=float(len(u) / n_npz))
    tab = None
    if lbl is not None:
        t = pd.DataFrame(dict(crop_token=u.crop_token.values, lbl=np.asarray(lbl)[u.npz_row.values])); tab = t.groupby(['crop_token', 'lbl']).size().rename('rows').reset_index()
        tab['tokens_with_this_label'] = tab.groupby('lbl').crop_token.transform('nunique'); tab['labels_for_this_token'] = tab.groupby('crop_token').lbl.transform('nunique')
    return cov, tab

def order_candidates(m, lbl=None):
    # Which sorting convention explains the order of the npz blocks?  Files are ordered by their observed first npz row and compared with each candidate sort.
    # keys: label id (from the npz labels of the matched rows), crop token, region/state code.  Works on any subset of files (relative order is compared).
    u = m[m.n_match == 1]; f = u.groupby('file').agg(first=('npz_row', 'min'), crop_token=('crop_token', 'first'), region=('region_code', 'first')).reset_index()
    f['lab'] = [int(pd.Series(np.asarray(lbl)[u[u.file == x].npz_row.values]).mode().iloc[0]) if lbl is not None else 0 for x in f.file]; f['region'] = f['region'].astype(str)
    obs = list(f.sort_values('first').file); res = []
    for nm, by, asc in (('label, token A-Z, region/state', ['lab', 'crop_token', 'region'], [True, True, True]), ('label, token Z-A, region/state', ['lab', 'crop_token', 'region'], [True, False, True]),
                        ('label, region/state, token A-Z', ['lab', 'region', 'crop_token'], [True, True, True]), ('token A-Z, region/state (no label)', ['crop_token', 'region'], [True, True])):
        cand = list(f.sort_values(by, ascending=asc).file); res.append(dict(candidate=nm, files=len(f), identical_order=cand == obs, files_in_same_position=int(sum(a == b for a, b in zip(cand, obs)))))
    return pd.DataFrame(res)

def run_match(raw_files, npz_path, out_dir, fea_key='fea', lbl_key='lbl'):
    os.makedirs(out_dir, exist_ok=True); z = np.load(npz_path); fea = z[fea_key]; print('npz', os.path.basename(npz_path), 'fea', fea.shape, '| labels', ('yes' if lbl_key in z.files else 'no'))
    points = [read_raw_points(p) for p in raw_files]; print('raw files:', [(os.path.basename(p['path'])[-22:], len(p['obs'])) for p in points])
    r = find_rule(points, fea); r.to_csv(os.path.join(out_dir, 'match_rule_search.csv'), index=False); print('\nrule search (sample of raw points):'); print(r.round(3).to_string(index=False))
    best = r.iloc[0]
    if not best.accepted:
        print(f"\n!! STOP: no declared rule reaches {MATCH_ACCEPT:.0%} unique matches (best {best['unique']:.1%}).  The raw files and this npz are not related by a simple 5-day median, or the raw files are not the ones this npz was built from.  Nothing is assumed; send me match_rule_search.csv."); return dict(accepted=False, search=r)
    print(f"\nACCEPTED rule: {best['rule']} | {best['layout']} | {best['scale']} -> {best['unique']:.1%} unique matches")
    m = match_points(points, fea, best['rule'], best['layout'], SCALES[best['scale']], progress=20000); m.to_csv(os.path.join(out_dir, 'row_map.csv'), index=False)
    ev = ordering_evidence(m); ev.to_csv(os.path.join(out_dir, 'ordering_evidence.csv'), index=False)
    print(f"full match: unique {(m.n_match == 1).mean():.2%} | none {(m.n_match == 0).mean():.2%} | ambiguous {(m.n_match > 1).mean():.2%}  (raw points {len(m)})"); print(ev.to_string(index=False))
    dup = int(m[m.npz_row >= 0].npz_row.duplicated().sum()); print('npz rows claimed by more than one raw point:', dup)
    oc = order_candidates(m, z[lbl_key] if lbl_key in z.files else None); oc.to_csv(os.path.join(out_dir, 'order_candidates.csv'), index=False); print('\nwhich sort order explains the npz block order?'); print(oc.to_string(index=False))
    cov, tab = label_and_coverage(m, len(fea), z[lbl_key] if lbl_key in z.files else None); print(f"coverage: {cov['matched_unique']} of {cov['npz_rows']} npz rows matched ({cov['share']:.1%})")
    if tab is not None:
        tab.to_csv(os.path.join(out_dir, 'label_by_crop_token.csv'), index=False); print('npz label received by each raw crop token (each token should map to ONE label):'); print(tab.to_string(index=False))
    return dict(accepted=True, search=r, map=m, evidence=ev, coverage=cov, labels=tab)


# ---------------------------------------------------------------- USA driver
USA_ORDER_STATUS = 'NOT RUN'; USA_ORDER_REPORT = {}
USA_RAW_RE = re.compile(r'^S2_USA_(?P<tok>[A-Za-z]+)_(?P<st>[A-Z]{2})\.csv$')
USA_UNIQUE_ACCEPT = 0.999

def find_usa_raw_files(roots):
    # raw files S2_USA_<token>_<ST>.csv under any of the roots (an upload / Drive hash prefix such as 'ab12cd34-17926..._' is ignored).  Two different files with the same name -> STOP.
    found = {}
    for root in roots:
        for p in glob.glob(os.path.join(root, '**', '*S2_USA_*.csv'), recursive=True):
            b = re.sub(r'^[0-9a-f]{8}-\d+_', '', os.path.basename(p)); m = USA_RAW_RE.match(b)
            if not m or m.group('tok') not in RAW_TO_CROP: continue
            if b in found and os.path.getsize(found[b]) != os.path.getsize(p): raise RuntimeError(f'STOP: two different files named {b}: {found[b]} and {p}')
            found.setdefault(b, p)
    return {b: found[b] for b in sorted(found, key=lambda x: (USA_RAW_RE.match(x).group('tok'), USA_RAW_RE.match(x).group('st')))}

def _per_crop_summary(m, lbl, n_npz):
    rows = []
    for tok, g in m.groupby('crop_token'):
        u = g[g.n_match == 1]; want = CROP_ID[RAW_TO_CROP[tok]]; bad = int((np.asarray(lbl)[u.npz_row.values] != want).sum()) if len(u) else 0
        rows.append(dict(crop_token=tok, crop=RAW_TO_CROP[tok], files=int(g.file.nunique()), raw_points=int(len(g)), unique=int(len(u)), unique_share=float(len(u) / len(g)), ambiguous=int((g.n_match > 1).sum()), unmatched=int((g.n_match == 0).sum()), label_mismatch=bad))
    return pd.DataFrame(rows).sort_values('crop_token').reset_index(drop=True)

def _blocks_ok(ev, m):
    # every file = ONE block of npz rows, increasing with the raw position; span may exceed the matched count only by rows lost to ambiguous matches in that file
    amb = m[m.n_match > 1].groupby('file').size().to_dict(); none = m[m.n_match == 0].groupby('file').size().to_dict(); bad = []
    for r in ev.itertuples():
        slack = amb.get(r.file, 0) + none.get(r.file, 0)
        if not r.increasing or (r.span - r.n_matched) > slack: bad.append(r.file)
    e = ev.sort_values('npz_row_min'); overlap = int((e.npz_row_min.values[1:] <= e.npz_row_max.values[:-1]).sum()) if len(e) > 1 else 0
    return bad, overlap

def compare_level1_table(m, table, name='level1'):
    # row-by-row: state in the raw FILE NAME vs state in the Level 1 table, for every uniquely matched raw point
    t = table[['npz_row', 'crop_id', 'state']].drop_duplicates('npz_row').set_index('npz_row'); u = m[m.n_match == 1].copy()
    u['l1_state'] = u.npz_row.map(t['state']); u['l1_crop_id'] = u.npz_row.map(t['crop_id']); u['file_state'] = u.region_code.astype(str)
    u['in_table'] = u.l1_state.notna(); u['agree'] = u.in_table & (u.l1_state.astype(str) == u.file_state)
    out = u.groupby('crop_token').agg(matched=('npz_row', 'size'), in_level1_table=('in_table', 'sum'), agree=('agree', 'sum')).reset_index(); out['disagree'] = out.in_level1_table - out.agree; out['table'] = name
    return out, u[u.in_table & ~u.agree][['file', 'raw_index', 'npz_row', 'crop_token', 'file_state', 'l1_state']]

def verify_usa_order(raw_files, npz_path, out_dir, level1_tables=None, log=print, rule_files_per_token=1):
    os.makedirs(out_dir, exist_ok=True); cache = os.path.join(out_dir, 'cache', 'usa_match'); os.makedirs(cache, exist_ok=True)
    z = np.load(npz_path); fea = z['fea']; lbl = z['lbl'].astype(int); n_npz = len(fea); rep = dict(npz=os.path.basename(npz_path), npz_rows=int(n_npz), raw_files=len(raw_files), status='FAILED', reasons=[])
    # (a) rule: tried on ONE file per crop token (declared candidates only; nothing is assumed)
    first = {}
    for b, p in raw_files.items(): first.setdefault(USA_RAW_RE.match(b).group('tok'), p)
    pts = [read_raw_points(p) for p in first.values()]; sr = find_rule(pts, fea); sr.to_csv(os.path.join(out_dir, 'usa_match_rule_search.csv'), index=False); best = sr.iloc[0]; del pts
    log('rule search on one file per crop token:'); log(sr.head(4).round(3).to_string(index=False))
    if not best.accepted: rep['reasons'].append(f"no declared rule reaches {MATCH_ACCEPT:.0%} (best {best['unique']:.1%}): the raw files and this npz are not related by a simple 5-day median"); return rep, None
    rule, layout, scale = best['rule'], best['layout'], SCALES[best['scale']]; rep['rule'] = f"{rule} | {best['layout']} | {best['scale']}"; log(f"ACCEPTED rule: {rep['rule']} ({best['unique']:.1%} unique on the sample)")
    # (b) full match, streamed file by file (memory-safe, resumable: a finished file is read back from cache/usa_match)
    parts = []; idx = None; t0 = time.time()
    for b, p in raw_files.items():
        cp = os.path.join(cache, b + '.csv')
        if os.path.exists(cp): parts.append(pd.read_csv(cp, dtype={'region_code': str})); continue
        if idx is None: idx = NpzIndex(fea, layout, scale)
        pt = read_raw_points(p); rows = []
        for j, obs in enumerate(pt['obs']):
            med = bin_medians(obs, rule); c, have = idx.candidates(med, MATCH_TOL)
            rows.append(dict(file=b, raw_index=pt['index'][j], n_match=0 if c is None else len(c), npz_row=int(c[0]) if (c is not None and len(c) == 1) else -1, crop_token=pt['crop_token'], region_code=pt['region_code']))
        df_ = pd.DataFrame(rows); df_.to_csv(cp, index=False); parts.append(df_); del pt
        log(f'   {b}: {len(df_)} raw points | unique {(df_.n_match == 1).mean():.2%} | {time.time() - t0:.0f}s')
    m = pd.concat(parts, ignore_index=True); m['region_code'] = m['region_code'].astype(str); m.to_csv(os.path.join(out_dir, 'usa_row_map.csv'), index=False)
    summ = _per_crop_summary(m, lbl, n_npz); summ.to_csv(os.path.join(out_dir, 'usa_order_by_crop.csv'), index=False); rep['by_crop'] = summ.to_dict('records')
    log('\nper crop token (soybeans is the Level 1 headline crop):'); log(summ.round(4).to_string(index=False))
    low = summ[summ.unique_share < USA_UNIQUE_ACCEPT]
    if len(low): rep['reasons'].append(f"crop tokens below {USA_UNIQUE_ACCEPT:.1%} exact matches: {low.crop_token.tolist()}")
    if int(summ.label_mismatch.sum()): rep['reasons'].append(f"{int(summ.label_mismatch.sum())} matched rows carry an npz label different from their crop token's label")
    dup = int(m[m.npz_row >= 0].npz_row.duplicated().sum()); rep['npz_rows_claimed_twice'] = dup
    if dup: rep['reasons'].append(f'{dup} npz rows are claimed by more than one raw point')
    ev = ordering_evidence(m); ev.to_csv(os.path.join(out_dir, 'usa_ordering_evidence.csv'), index=False); bad_blocks, overlap = _blocks_ok(ev, m); rep['blocks_not_contiguous'] = bad_blocks; rep['block_overlaps'] = overlap
    if bad_blocks or overlap: rep['reasons'].append(f'files that are not one contiguous increasing block: {bad_blocks}; overlapping blocks: {overlap}')
    mm = m.copy(); mm['crop_token'] = mm['crop_token'].astype(str); oc = order_candidates(mm, lbl); oc.to_csv(os.path.join(out_dir, 'usa_order_candidates.csv'), index=False); rep['order_candidates'] = oc.to_dict('records')
    conv = oc[oc.candidate == 'label, token A-Z, region/state']; rep['order_convention_holds'] = bool(len(conv) and conv.identical_order.iloc[0]); log('\nsort order of the npz blocks:'); log(oc.to_string(index=False))
    if not rep['order_convention_holds']: rep['reasons'].append('the npz block order is not (label, token A-Z, state)')
    cov = int((m.n_match == 1).sum()); rep['npz_rows_matched'] = cov; rep['npz_rows_matched_share'] = cov / n_npz
    # (e) the Level 1 state table, row by row
    rep['level1'] = []; l1_bad = 0
    for name, tab in (level1_tables or {}).items():
        cmp_, wrong = compare_level1_table(m, tab, name); cmp_.to_csv(os.path.join(out_dir, f'usa_vs_level1_{re.sub("[^A-Za-z0-9]+", "_", name)[-40:]}.csv'), index=False); rep['level1'].append(dict(table=name, by_crop=cmp_.to_dict('records'), disagreements=int(cmp_.disagree.sum())))
        wrong.to_csv(os.path.join(out_dir, 'usa_level1_disagreements.csv'), index=False); l1_bad += int(cmp_.disagree.sum())
        log(f'\nLevel 1 state table ({name}) vs the state in the raw file name:'); log(cmp_.to_string(index=False))
    rep['level1_disagreements'] = l1_bad
    if rep['reasons']: rep['status'] = 'FAILED'
    elif not rep['level1']: rep['status'] = 'ORDER VERIFIED (Level 1 state table not found: row-by-row state comparison NOT done)'
    elif l1_bad: rep['status'] = f'ORDER VERIFIED, BUT the Level 1 state table disagrees with the raw file names on {l1_bad} rows'
    else: rep['status'] = 'VERIFIED (order exact; Level 1 state table equals the raw file names on every matched row)'
    return rep, m

USA_RAW_DIR = ''      # OPTIONAL: if your raw USA files S2_USA_<crop>_<ST>.csv sit in a folder that is not under the project folder, put its full path here, e.g. '/content/drive/MyDrive/cropglobe_assignment/USA'.  Left empty, the whole project folder and the data folder are searched (any sub-folder name works, including 'USA').
def run_usa_order_cell(project_root, data_dir, out_dir, log=print):
    global USA_ORDER_STATUS, USA_ORDER_REPORT
    roots = [r for r in (USA_RAW_DIR, project_root, data_dir) if r]; print('searching for raw USA files in:', roots)
    raw = find_usa_raw_files(roots)
    if not raw:
        try:                                                                      # best effort only; the repository layout of the raw files is not documented here
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id='x-ytong/CropGlobe', repo_type='dataset', local_dir=data_dir, allow_patterns=['**/S2_USA_*.csv']); raw = find_usa_raw_files(roots)
        except Exception as e: log(f'  (download attempt failed: {type(e).__name__})')
    npz = find_npz(data_dir, 'USA') if glob.glob(os.path.join(data_dir, '**', 'median_S2_USA_month1-12_day5.npz'), recursive=True) else None
    if npz is None:
        try:
            from huggingface_hub import snapshot_download; snapshot_download(repo_id='x-ytong/CropGlobe', repo_type='dataset', local_dir=data_dir, allow_patterns=['**/median_S2_USA_month1-12_day5.npz']); npz = find_npz(data_dir, 'USA')
        except Exception as e: log(f'  (USA npz not available: {type(e).__name__})')
    if not raw or npz is None:
        USA_ORDER_STATUS = 'NOT VERIFIED (' + ('raw USA files S2_USA_<crop>_<ST>.csv not found' if not raw else 'USA npz not found') + ': put them anywhere under the project folder and run this cell again)'
        USA_ORDER_REPORT = dict(status=USA_ORDER_STATUS); print('\n' + '=' * 100 + f'\nUSA ORDER VERIFICATION: {USA_ORDER_STATUS}\n' + '=' * 100); return USA_ORDER_REPORT
    print(f'{len(raw)} raw USA files: ' + ', '.join(f'{USA_RAW_RE.match(b).group("tok")}:{sum(1 for x in raw if USA_RAW_RE.match(x).group("tok") == USA_RAW_RE.match(b).group("tok"))}' for b in dict.fromkeys(raw)))
    tabs = {}
    for p in glob.glob(os.path.join(project_root, '**', 'cache', 'usa_meta.csv'), recursive=True):
        try:
            t = pd.read_csv(p)
            if {'npz_row', 'crop_id', 'state'} <= set(t.columns): tabs[os.path.relpath(p, project_root)] = t
        except Exception: pass
    print('Level 1 state tables found:', list(tabs) or 'none')
    rep, m = verify_usa_order(raw, npz, os.path.join(out_dir, 'tables'), tabs, log=log); USA_ORDER_REPORT = rep; USA_ORDER_STATUS = rep['status']
    atomic_write(os.path.join(out_dir, 'tables', 'usa_order_verification.json'), lambda f: f.write(json.dumps(rep, indent=1, default=str)), mode='w')
    sb = [r for r in rep.get('by_crop', []) if r['crop'] == 'soybeans']
    print('\n' + '=' * 100 + f'\nUSA ORDER VERIFICATION: {USA_ORDER_STATUS}')
    if sb: print('   soybeans: ' + ' | '.join(f"{r['crop_token']}: {r['unique']}/{r['raw_points']} exact matches, {r['label_mismatch']} label mismatches" for r in sb))
    if rep['reasons']: print('   reasons: ' + '; '.join(rep['reasons']))
    print('=' * 100)
    return rep


# ---- self-test of THIS verifier on a tiny synthetic world (runs every time; the verifier must say VERIFIED for correct data and FAILED for swapped blocks, wrong labels and unrelated data)
def _selftest_usa_order():
    import tempfile, io, contextlib
    ns = globals(); CID = CROP_ID; RTC = RAW_TO_CROP
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
    def run(td, **kw):
        rawdir, npz, t = make_world(td, **kw); files = find_usa_raw_files([rawdir])
        with contextlib.redirect_stdout(io.StringIO()): rep, m = verify_usa_order(files, npz, os.path.join(td, 'out'), {'level1': t})
        return rep, m
    with tempfile.TemporaryDirectory() as td: rep, m = run(td); assert rep['status'].startswith('VERIFIED') and rep['level1_disagreements'] == 0 and (m.n_match == 1).all(), rep['status']
    with tempfile.TemporaryDirectory() as td: rep, m = run(td, swap_soy_blocks=True); assert rep['status'] == 'FAILED', rep['status']
    with tempfile.TemporaryDirectory() as td: rep, m = run(td, mislabel_soy=True); assert rep['status'] == 'FAILED'
    with tempfile.TemporaryDirectory() as td: rep, m = run(td, unrelated=True); assert rep['status'] == 'FAILED' and m is None
    print('ordering-verifier self-test passed: correct data VERIFIED; swapped state blocks, wrong labels and unrelated data FAILED')
_selftest_usa_order()

run_usa_order_cell(PROJECT_ROOT, DATA_DIR, OUT_DIR)

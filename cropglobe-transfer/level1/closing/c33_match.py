# ======================================================================================
# EXACT ROW-BY-ROW MATCH: raw per-point CSVs (irregular Sentinel-2 observations)  <->  packaged npz (5-day medians, 73 steps x 10 bands = 730 values).
# Raw layout (seen in S2_FRA_other_LAMB93_R*.csv): columns system:index, information, raw_data, .geo.  raw_data = [[doy,B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12], ...] (start date 2023-01-01),
# .geo = {"type":"Point","coordinates":[lon, lat]} (degrees).  The 5-day binning rule, the value scale and the npz layout are NOT documented in the files we hold, so we do not assume them:
# a SMALL, DECLARED set of rules is tried and the one that reproduces the npz is reported.  Matching uses only bins that contain raw observations (gap-filled bins are ignored),
# so it does not depend on how the authors filled empty bins.  PRE-DECLARED acceptance: >= 99% of sampled raw points have exactly one npz row matching within tolerance.  Otherwise: STOP, nothing is assumed.
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

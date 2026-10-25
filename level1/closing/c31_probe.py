# ======================================================================================
# LEVEL 2 DATA PROBE (read-only; trains nothing).  Answers, from the REAL files, the questions that decide whether the instructor's next steps can be built honestly:
#   Q1  Does the packaged CropGlobe data (npz) carry coordinates or region ids, or only features + crop labels?
#   Q2  Do the raw per-sample CSVs (S2_USA_*.csv, S2_FRA_*.csv ...) contain the same features, so the npz-row <-> coordinate join can be VERIFIED row by row?
#   Q3  Are there region polygons for France that can be reached, and how many regions / how many samples fall into each?
# Nothing is guessed: the report lists what exists.  Upload probe_report.json back to continue.
# ======================================================================================
import os, json, glob, re, zipfile, io, time
import numpy as np, pandas as pd

def _human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB': return f'{n:.0f}{u}' if u == 'B' else f'{n:.1f}{u}'
        n /= 1024

def probe_tree(root, max_files=400):
    out = []
    for dp, dn, fn in os.walk(root):
        for f in sorted(fn):
            p = os.path.join(dp, f); out.append(dict(path=os.path.relpath(p, root), ext=os.path.splitext(f)[1].lower(), size=os.path.getsize(p)))
            if len(out) >= max_files: return out, True
    return out, False

def probe_npz(path):
    z = np.load(path, allow_pickle=False); info = dict(path=os.path.basename(path), keys={})
    for k in z.files:
        a = z[k]; info['keys'][k] = dict(shape=list(a.shape), dtype=str(a.dtype))
    info['has_coordinate_like_key'] = [k for k in z.files if re.search(r'lat|lon|coord|geo|region|state|admin|nuts|id', k, re.I)]
    if 'lbl' in z.files:
        u, c = np.unique(z['lbl'], return_counts=True); info['label_counts'] = {str(a): int(b) for a, b in zip(u, c)}
    return info

def probe_csv(path, nrows=3):
    d = pd.read_csv(path, nrows=nrows); n_rows = sum(1 for _ in open(path, 'rb')) - 1 if os.path.getsize(path) < 300e6 else None
    num = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    return dict(path=os.path.basename(path), n_columns=len(d.columns), n_numeric_columns=len(num), rows=n_rows, columns_head=list(d.columns[:12]), columns_tail=list(d.columns[-6:]), has_geo_column='.geo' in d.columns)

def probe_geo_join(raw_csv, npz_path, feature_key='fea'):
    # Q2: can a raw CSV row be matched to an npz row by its feature vector?  We only check whether the raw file has as many numeric columns as the npz feature width (or a multiple/divisor) -- no matching is attempted blind.
    z = np.load(npz_path); width = int(np.prod(z[feature_key].shape[1:])) if feature_key in z.files else None
    pr = probe_csv(raw_csv); pr['npz_feature_width'] = width
    pr['numeric_columns_equal_feature_width'] = (pr['n_numeric_columns'] == width) if width else None
    pr['verdict'] = ('raw CSV has the same number of numeric columns as the npz features: an exact row-by-row match can be attempted' if pr['numeric_columns_equal_feature_width']
                     else 'raw CSV columns do not obviously equal the npz feature width: a match needs the raw column layout (see columns_head / columns_tail); build it from this report, not blind')
    return pr

REGION_SOURCES = {                                  # candidate public polygon sources; we only test reachability and count France's regions -- nothing is assumed about names
    'natural_earth_admin1_10m': 'https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip',
    'gadm41_FRA_1': 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_FRA_1.json',
}
def probe_region_sources(timeout=40):
    import urllib.request
    res = {}
    for name, url in REGION_SOURCES.items():
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}); r = urllib.request.urlopen(req, timeout=timeout); data = r.read(); res[name] = dict(url=url, ok=True, status=getattr(r, 'status', 200), bytes=len(data), seconds=round(time.time() - t0, 1))
            if url.endswith('.json'):
                j = json.loads(data); feats = j.get('features', []); res[name]['n_features'] = len(feats); res[name]['names'] = [f['properties'].get('NAME_1') for f in feats][:30]
        except Exception as e: res[name] = dict(url=url, ok=False, error=f'{type(e).__name__}: {str(e)[:120]}')
    return res

def run_probe(data_dir, raw_usa_dir=None, raw_fra_dir=None, out_path='probe_report.json'):
    rep = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), data_dir=data_dir)
    files, truncated = probe_tree(data_dir); rep['tree'] = files; rep['tree_truncated'] = truncated
    npzs = sorted(glob.glob(os.path.join(data_dir, '**', '*.npz'), recursive=True)); rep['npz'] = [probe_npz(p) for p in npzs if re.search(r'median_S2_(USA|FRA|ARG|CHN)_month1-12_day5', os.path.basename(p))]
    for tag, d in (('raw_usa', raw_usa_dir), ('raw_fra', raw_fra_dir)):
        if d and os.path.isdir(d):
            cs = sorted(glob.glob(os.path.join(d, '**', '*.csv'), recursive=True)); rep[tag] = dict(n_csv=len(cs), first=[probe_csv(p) for p in cs[:3]], names=[os.path.basename(p) for p in cs[:60]])
        else: rep[tag] = 'directory not given / not found'
    usa_npz = [p for p in npzs if 'median_S2_USA_month1-12_day5' in p]
    if usa_npz and isinstance(rep.get('raw_usa'), dict) and rep['raw_usa']['n_csv']:
        c0 = sorted(glob.glob(os.path.join(raw_usa_dir, '**', '*.csv'), recursive=True))[0]; rep['usa_join_probe'] = probe_geo_join(c0, usa_npz[0])
    rep['region_sources'] = probe_region_sources()
    json.dump(rep, open(out_path, 'w'), indent=1, default=str)
    print('=== LEVEL 2 DATA PROBE ==='); print('files under', data_dir, ':', len(files), '(truncated)' if truncated else '')
    for n in rep['npz']: print(' ', n['path'], '| keys:', {k: v['shape'] for k, v in n['keys'].items()}, '| coordinate/region-like keys:', n['has_coordinate_like_key'] or 'NONE')
    for tag in ('raw_usa', 'raw_fra'):
        r = rep[tag]; print(' ', tag, ':', r if isinstance(r, str) else f"{r['n_csv']} csv files; first columns {r['first'][0]['columns_head'] if r['first'] else ''}; has .geo: {r['first'][0]['has_geo_column'] if r['first'] else ''}")
    if 'usa_join_probe' in rep: print('  USA join:', rep['usa_join_probe']['verdict'])
    for k, v in rep['region_sources'].items(): print('  region source', k, ':', 'reachable' if v['ok'] else 'NOT reachable', v.get('error', ''), ('| France regions: ' + str(v['n_features'])) if 'n_features' in v else '')
    print('\nSaved', out_path, '-- upload this file to continue.  Nothing was changed or trained.')
    return rep

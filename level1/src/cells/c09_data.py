# ======================================================================================
# DATA: load Sentinel-2 median time series (official files), de-duplicate like the authors' loader, attach a US STATE to every USA sample,
#       and VERIFY everything before any training.   (Torch-free: arrays stay numpy; the torch cell converts them.)
# ======================================================================================
ALL_STATES = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT',
              'NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']

def unique_row_index(a):
    # indices of the FIRST occurrence of every distinct feature row == what the authors' data_load() keeps via np.unique(axis=0, return_index=True)
    a = np.ascontiguousarray(np.asarray(a).reshape(len(a), -1)); seen, keep = set(), []
    for i in range(len(a)):
        h = hashlib.blake2b((a[i] + 0.0).tobytes(), digest_size=16).digest()   # +0.0 turns -0.0 into 0.0 (np.unique treats them as equal)
        if h not in seen: seen.add(h); keep.append(i)
    return np.asarray(keep, dtype=np.int64)

def _sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''): h.update(b)
    return h.hexdigest()

def find_npz(data_dir, country, log=print):
    # V33-05: several matching files are LISTED.  Byte-identical copies -> the documented rule picks one (path containing for_CropNet first, then the shortest path).
    # Files with DIFFERENT content -> STOP, because the choice would silently change the data.
    name = f'median_S2_{country}_month1-12_day5.npz'; hits = []
    for root, _d, files in os.walk(data_dir):
        if name in files: hits.append(os.path.join(root, name))
    _check(hits, f'{name} not found under {data_dir}')
    hits = sorted(set(hits))
    if len(hits) > 1:
        info = [(p, os.path.getsize(p), _sha256_file(p)) for p in hits]
        log(f'  {country}: {len(hits)} files match {name}:'); [log(f'      {p}  size {n}  sha256 {h[:16]}...') for p, n, h in info]
        if len({h for _p, _n, h in info}) > 1:
            raise RuntimeError(f'STOP: {len(hits)} DIFFERENT files match {name} under {data_dir}. Remove or move the wrong ones (or point DATA_DIR at one folder); no file is chosen silently.')
        log('      byte-identical copies -> deterministic rule: path containing for_CropNet first, then the shortest path')
    hits.sort(key=lambda p: ('for_CropNet' not in p, len(p)))
    return hits[0]

def load_country(path, country):
    z = np.load(path); lbl = z['lbl'].astype(int)
    cand = ['fea'] if 'fea' in z.files else [k for k in z.files if k != 'lbl' and z[k].ndim >= 2 and z[k].shape[0] == len(lbl)]
    _check(cand, f'{country}: no per-sample feature array (keys={z.files})')
    a = z[cand[0]].astype(np.float32); band_axes = [ax for ax in range(1, a.ndim) if a.shape[ax] == 10]
    _check(band_axes, f'{country}: cannot find the 10-band axis in shape {a.shape}')
    a = np.moveaxis(a, band_axes[0], 1).reshape(a.shape[0], 10, -1)
    _check(len(lbl) == len(a), f'{country}: label/feature length mismatch')
    _check(lbl.min() >= 0 and lbl.max() <= 6, f'{country}: label ids outside 0..6')
    nan = int((~np.isfinite(a)).sum()); _check(nan == 0, f'{country}: {nan} non-finite feature values (NaN/inf) -- CropNet would produce NaN')
    return lbl, a

# ---------------------------------------------------------------- USA state assignment (alignment logic reused from the audited Phase 3 notebook)
TOKENS_LONGEST_FIRST = sorted(RAW_TO_CROP, key=len, reverse=True)
def parse_usa_filename(fname):
    m = re.match(r'^S2_USA_(.+)\.csv$', fname)
    if not m: return None
    rest = m.group(1)
    for tok in TOKENS_LONGEST_FIRST:
        if rest == tok or rest.startswith(tok + '_'):
            tail = rest[len(tok) + 1:] if rest != tok else ''
            return tok, (tail.upper() if len(tail) == 2 else None)
    return None

def read_geo_column(path):
    df = pd.read_csv(path, usecols=['.geo']); lons = np.empty(len(df)); lats = np.empty(len(df))
    for i, g in enumerate(df['.geo'].values):
        c = json.loads(g)['coordinates']; lons[i] = float(c[0]); lats[i] = float(c[1])
    return lons, lats

def between_state_ratio_by_crop(F, crop_ids, state_codes, rng, n_perm=8):
    # per-crop lift: between-state variance share of the features inside ONE crop, divided by the same quantity under permuted state labels (V34-01: kept per crop so a disputed crop can be judged alone)
    def one(X, codes):
        if len(np.unique(codes)) < 2: return float('nan')
        order = np.argsort(codes, kind='stable'); Xs, gs = X[order], codes[order]
        starts = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]; cnt = np.diff(np.r_[starts, len(gs)])
        means = np.add.reduceat(Xs, starts, axis=0) / cnt[:, None]; gm = X.mean(0)
        return float((cnt[:, None] * (means - gm) ** 2).sum() / ((X - gm) ** 2).sum())
    lifts = {}
    for c in np.unique(crop_ids):
        m = np.flatnonzero(crop_ids == c); X, g = F[m], state_codes[m]; obs = one(X, g)
        perm = float(np.nanmean([one(X, g[rng.permutation(len(g))]) for _ in range(n_perm)]))
        if perm > 0: lifts[int(c)] = obs / perm
    return lifts

def between_state_ratio(F, crop_ids, state_codes, rng, n_perm=8):
    d = between_state_ratio_by_crop(F, crop_ids, state_codes, rng, n_perm); return float(np.mean(list(d.values()))) if d else float('nan')

def build_usa_meta(geo_by_file, usa_lbl_raw, F, sub, keep_usa, assign_states_fn, log=print, report_path=None, override=None):
    file_rows, lons, lats, fidx = [], [], [], []
    for fname in sorted(geo_by_file):
        parsed = parse_usa_filename(fname)
        if parsed is None: continue
        tok, st_fn = parsed; lo, la = geo_by_file[fname]; fi = len(file_rows)
        file_rows.append(dict(file_idx=fi, fname=fname, raw_tok=tok, crop=RAW_TO_CROP[tok], crop_id=CROP_ID[RAW_TO_CROP[tok]], state_fn=st_fn, n_rows=len(lo)))
        lons.append(np.asarray(lo)); lats.append(np.asarray(la)); fidx.append(np.full(len(lo), fi))
    _check(file_rows, 'no S2_USA_*.csv geometry found')
    file_table = pd.DataFrame(file_rows)
    geo = pd.DataFrame({'file_idx': np.concatenate(fidx), 'lat': np.concatenate(lats), 'lon': np.concatenate(lons)})
    if len(geo) != len(usa_lbl_raw):
        raise RuntimeError(f'raw USA points {len(geo)} (from {len(file_table)} files) != rows in the USA NPZ {len(usa_lbl_raw)}. '
                           'The raw USA CSV set is incomplete (typically a truncated Google-Drive listing). See the platform cell for how to fix it.')
    ORDERINGS = {
        'crop_id,token,state':      lambda ft: ft.sort_values(['crop_id', 'raw_tok', 'state_fn']),
        'crop_id,token_desc,state': lambda ft: ft.sort_values(['crop_id', 'raw_tok', 'state_fn'], ascending=[True, False, True]),
        'crop_id,state,token':      lambda ft: ft.sort_values(['crop_id', 'state_fn', 'raw_tok']),
        'filename':                 lambda ft: ft.sort_values('fname'),
    }
    offsets = np.r_[0, np.cumsum(file_table.sort_values('file_idx')['n_rows'].values)]
    def build_row_order(fo): return np.concatenate([np.arange(offsets[fi], offsets[fi] + n) for fi, n in zip(fo['file_idx'], fo['n_rows'])])
    candidates = {}
    for name, fn in ORDERINGS.items():
        fo = fn(file_table); pred = np.repeat(fo['crop_id'].values, fo['n_rows'].values)
        if (pred == usa_lbl_raw).all(): candidates[name] = build_row_order(fo); log(f'  ordering {name:28s} exact label-sequence match: True')
    _check(candidates, 'No file ordering reproduces the NPZ label sequence -- state alignment impossible, stopping.')
    uniq = {}
    for name, ro in candidates.items():                                   # identical row orders are ONE hypothesis (keep the first name)
        if not any(np.array_equal(ro, u) for u in uniq.values()): uniq[name] = ro
    candidates = uniq
    geo['state'] = assign_states_fn(geo['lat'].values, geo['lon'].values)
    _check(pd.notna(geo['state']).all(), 'some USA points got no state')
    state_code_of = {s: i for i, s in enumerate(ALL_STATES)}; names = list(candidates); crops_sub = usa_lbl_raw[sub]
    st_of = {n: np.array([state_code_of[x] for x in geo['state'].values[ro]]) for n, ro in candidates.items()}
    # V34-01: the candidate orderings can only disagree on SOME crops (e.g. a crop stored as two raw tokens such as spring/winter wheat).  The all-crop mean lift is blind to that (the undisputed crops
    # contribute the same number to every candidate), so candidates are ALSO compared on the disputed crops alone, with the SAME permutation stream for every candidate (paired, less noisy).
    disputed = sorted({int(c) for c in np.unique(usa_lbl_raw) for i_, a_ in enumerate(names) for b_ in names[i_ + 1:] if (st_of[a_][usa_lbl_raw == c] != st_of[b_][usa_lbl_raw == c]).any()})
    lifts, by_crop, lifts_disp = {}, {}, {}
    for name in names:
        by_crop[name] = between_state_ratio_by_crop(F, crops_sub, st_of[name][sub], np.random.default_rng(0), n_perm=8)
        lifts[name] = float(np.mean(list(by_crop[name].values()))) if by_crop[name] else float('nan')
        log(f'  ordering {name:28s} state-alignment lift (all crops): {lifts[name]:.2f}x')
    if disputed:
        log('  disputed crops (candidate orderings assign DIFFERENT states to some of their rows): ' + ', '.join(f'{CROP_LABELS[c]} ({int((usa_lbl_raw == c).sum())} rows)' for c in disputed))
        msk = np.isin(crops_sub, disputed)
        for name in names:
            bc = between_state_ratio_by_crop(F[msk], crops_sub[msk], st_of[name][sub][msk], np.random.default_rng(0), n_perm=24); by_crop[name + '|disputed'] = bc
            lifts_disp[name] = float(np.mean(list(bc.values()))) if bc else lifts[name]
            log(f'  ordering {name:28s} lift on the DISPUTED crops only: {lifts_disp[name]:.2f}x   ' + ', '.join(f'{CROP_LABELS[c]} {v:.1f}x' for c, v in bc.items()))
    else: lifts_disp = dict(lifts)
    top = max(lifts_disp.values())
    # near-tie (within 3%, i.e. permutation noise): keep the FIRST ordering in the audited list (the one validated in the Phase 3 notebook)
    best_name = next(n for n in ORDERINGS if n in lifts_disp and lifts_disp[n] >= 0.97 * top)
    override = override if override is not None else globals().get('ALIGNMENT_ORDERING_OVERRIDE')
    if override is not None: _check(override in candidates, f'ALIGNMENT_ORDERING_OVERRIDE={override!r} is not one of the candidate orderings {list(candidates)}'); best_name = override; log(f'  -> ordering "{override}" chosen EXPLICITLY by ALIGNMENT_ORDERING_OVERRIDE')
    best_lift, best_order = lifts[best_name], candidates[best_name]
    _check(best_lift >= 2.0, f'Best ordering "{best_name}" lift {best_lift:.2f} (<2): state alignment unsupported, stopping.')
    log(f'  -> aligned ordering "{best_name}" (lift {best_lift:.1f}x all crops; {lifts_disp[best_name]:.1f}x on the disputed crops)')
    # ambiguity guard: another ordering that assigns DIFFERENT states to some rows but explains the DISPUTED crops almost as well is a real risk (same 15% rule as before, applied where the candidates can differ)
    ambiguous = []; evidence = {n: dict(lift_all_crops=round(float(lifts[n]), 3), lift_disputed_crops=round(float(lifts_disp[n]), 3),
                                        per_crop_lift={CROP_LABELS[c]: round(float(v), 3) for c, v in by_crop[n].items()}) for n in names}
    for name in names:
        if name == best_name: continue
        differ = float((st_of[name] != st_of[best_name]).mean()); ld = lifts_disp[name]
        evidence[name].update(differs_from_chosen_pct=round(100 * differ, 3), disputed_crops=[CROP_LABELS[c] for c in disputed])
        if differ > 0 and ld > lifts_disp[best_name] / 1.15:
            ambiguous.append(name); log(f'  !! AMBIGUOUS: ordering "{name}" differs on {100 * differ:.1f}% of the rows and its lift on the disputed crops ({ld:.1f}x) is within 15% of the best ({lifts_disp[best_name]:.1f}x) -- state labels of those rows are uncertain')
    if report_path:                                                                  # V33-06: candidate alignments and their evidence are always saved
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f_: json.dump(dict(chosen=best_name, chosen_lift=float(best_lift), chosen_lift_disputed_crops=float(lifts_disp[best_name]), disputed_crops=[CROP_LABELS[c] for c in disputed], explicit_override=override, ambiguous_with=ambiguous, candidates=evidence), f_, indent=1)
    if ambiguous and override is None:                                               # V33-06: no silent pick between weakly identified orderings
        raise RuntimeError(f'STOP -- state alignment is AMBIGUOUS: orderings {ambiguous} give different states for some rows but explain the features almost as well as "{best_name}". Evidence in {report_path}. '
                           f'Decide explicitly: set ALIGNMENT_ORDERING_OVERRIDE to one of {list(candidates)} (and record why), or obtain an independent row identifier. Nothing was chosen silently.')
    full = pd.DataFrame({'npz_row': np.arange(len(usa_lbl_raw)), 'crop_id': usa_lbl_raw, 'state': geo['state'].values[best_order]})
    _check((full['crop_id'].values == usa_lbl_raw).all(), 'alignment broke the label order')
    meta = full.iloc[keep_usa].reset_index(drop=True)
    meta['crop'] = meta['crop_id'].map(CROP_LABELS)
    return meta, best_name, best_lift

def assign_states_census(lat, lon, cache_dir):
    # point-in-polygon on the US Census 2018 state polygons; nearest state as the fallback for coastal points
    import geopandas as gpd, urllib.request
    url = 'https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip'; zp = os.path.join(cache_dir, 'cb_2018_us_state_20m.zip')
    if not os.path.exists(zp):
        last = None
        for _ in range(4):
            try: urllib.request.urlretrieve(url, zp); break
            except Exception as e: last = e; time.sleep(3)
        _check(os.path.exists(zp), f'could not download the Census state polygons ({last}); check that Internet is ON')
    try: sg = gpd.read_file(zp)
    except Exception: sg = gpd.read_file('zip://' + zp)
    sg = sg[sg['STUSPS'].isin(ALL_STATES)][['STUSPS', 'geometry']].to_crs('EPSG:4326')
    pts = gpd.GeoDataFrame({'k': np.arange(len(lat))}, geometry=gpd.points_from_xy(lon, lat), crs='EPSG:4326')
    j = gpd.sjoin(pts, sg, how='left', predicate='within'); j = j[~j.index.duplicated(keep='first')]
    state = j['STUSPS'].reindex(pts.index).values.astype(object)
    un = pd.isna(state)
    if un.any():
        near = gpd.sjoin_nearest(pts[un].to_crs('EPSG:5070'), sg.to_crs('EPSG:5070'), how='left', distance_col='d')
        near = near[~near.index.duplicated(keep='first')]; state[un] = near['STUSPS'].reindex(pts.index[un]).values
    return state

def load_all_data(data_dir, out_dir, get_geo_fn, extra_meta_paths=(), assign_fn=None):
    cache_dir = os.path.join(out_dir, 'cache'); os.makedirs(cache_dir, exist_ok=True)
    needed = ['USA'] + [t for t in TARGETS_TO_RUN]; FEATURES, NPZ_LBL, INFO, F_lift, sub = {}, {}, {}, None, None
    keep_usa, usa_lbl_raw = None, None
    for c in needed:
        path = find_npz(data_dir, c); lbl, a = load_country(path, c); keep = unique_row_index(a)
        if c == 'USA':
            usa_lbl_raw, keep_usa = lbl, keep
            sub = np.random.default_rng(0).choice(len(lbl), size=min(30000, len(lbl)), replace=False)
            F = a.reshape(len(a), -1)[sub].astype(np.float32); F_lift = (F - F.mean(0)) / (F.std(0) + 1e-6)
        FEATURES[c] = np.ascontiguousarray(a[keep][:, None, :, :]); NPZ_LBL[c] = lbl[keep]
        INFO[c] = dict(path=path, raw=len(lbl), kept=len(keep), dup=len(lbl) - len(keep), shape=tuple(FEATURES[c].shape), keep=keep, raw_by_class=pd.Series(lbl).map(CROP_LABELS).value_counts().reindex(CROP_ORDER, fill_value=0).to_dict(), mean=float(a.mean()), std=float(a.std()),
                       vmin=float(a.min()), vmax=float(a.max())); del a
    # --- USA state table: cache -> attached meta -> build from raw geometry
    meta, how = None, ''
    for p in [os.path.join(cache_dir, 'usa_meta.csv')] + list(extra_meta_paths):
        if p and os.path.exists(p):
            m = pd.read_csv(p)
            # V33-04: besides length, crop ids and non-null states, the cached npz_row column must equal the CURRENT de-duplicated row index exactly (a stale / reordered cache is rebuilt, never reused)
            if ('npz_row' in m.columns and len(m) == len(NPZ_LBL['USA']) and np.array_equal(m['npz_row'].values.astype(np.int64), np.asarray(keep_usa, dtype=np.int64))
                    and (m['crop_id'].values == NPZ_LBL['USA']).all() and m['state'].notna().all()):
                m['crop'] = m['crop_id'].map(CROP_LABELS); meta, how = m[['npz_row', 'crop_id', 'state', 'crop']], f'cached state table: {p}'; break
            print(f'  (ignoring {p}: does not match the USA NPZ -- length, npz_row index, crop ids or missing states differ; it will not be reused)')
    if meta is None:
        meta, name, lift = build_usa_meta(get_geo_fn(), usa_lbl_raw, F_lift, sub, keep_usa, assign_fn or (lambda lat, lon: assign_states_census(lat, lon, cache_dir)), report_path=os.path.join(cache_dir, 'alignment_report.json'))
        meta[['npz_row', 'crop_id', 'state']].to_csv(os.path.join(cache_dir, 'usa_meta.csv'), index=False); how = f'built from raw geometry (ordering {name}, lift {lift:.1f}x)'
    print('USA state table:', how)
    return FEATURES, NPZ_LBL, INFO, meta

def row_hash_set(F):
    a = np.ascontiguousarray(F.reshape(len(F), -1)); return {hashlib.blake2b((a[i] + 0.0).tobytes(), digest_size=16).digest() for i in range(len(a))}

def verify_data(FEATURES, NPZ_LBL, INFO, meta):
    # EVERY check here is a STOP (spec section 1): raw counts, label sets, class counts vs paper v3 Table 2, no target row inside the USA table, state alignment.
    print('\n================ DATA VERIFICATION (paper arXiv 2509.03497 v3, Table 2) ================')
    rows = []
    for c in INFO:
        rows.append(dict(country=c, raw=INFO[c]['raw'], table2_total=TABLE2_TOTAL[c], kept_after_dedup=INFO[c]['kept'], removed_duplicates=INFO[c]['dup'], shape=INFO[c]['shape'], **{f'kept_{k}': v for k, v in
                         pd.Series(NPZ_LBL[c]).map(CROP_LABELS).value_counts().reindex(CROP_ORDER, fill_value=0).to_dict().items()}))
    print(pd.DataFrame(rows).to_string(index=False))
    problems = []
    for c in INFO:                                                                  # raw row counts == Table 2 totals; raw class counts == Table 2 class counts
        if INFO[c]['raw'] != TABLE2_TOTAL[c]: problems.append(f'{c}: raw rows {INFO[c]["raw"]} != Table 2 total {TABLE2_TOTAL[c]}')
        if INFO[c]['raw_by_class'] != TABLE2[c]: problems.append(f'{c}: raw class counts {INFO[c]["raw_by_class"]} != Table 2 {TABLE2[c]}')
    for t in TARGETS_TO_RUN:                                                        # label set per target == Table 2 (raw AND after de-duplication)
        got = sorted([CROP_LABELS[i] for i in np.unique(NPZ_LBL[t])], key=CROP_ORDER.index); exp = sorted(DEFINED_CLASSES[t], key=CROP_ORDER.index)
        rawset = sorted([c for c, n in INFO[t]['raw_by_class'].items() if n > 0], key=CROP_ORDER.index)
        print(f'  label set {t}: after dedup {got} | raw {rawset} | Table 2 {exp} -> {"OK" if got == exp and rawset == exp else "MISMATCH"}')
        if got != exp or rawset != exp: problems.append(f'{t}: label set data(raw {rawset}, dedup {got}) != Table 2 {exp}')
    for c in INFO:
        if 'median_S2_' not in os.path.basename(INFO[c]['path']): problems.append(f'{c}: file {INFO[c]["path"]} is not a Sentinel-2 median file (AlphaEarth embeddings must not be used here)')
        if INFO[c]['shape'][1:3] != (1, 10): problems.append(f'{c}: unexpected tensor shape {INFO[c]["shape"]} (need (N,1,10,T))')
        elif INFO[c]['shape'][3] < 67: problems.append(f'{c}: time axis {INFO[c]["shape"][3]} too short for the official DOY 121-334 window (needs >= 67 five-day steps)')
    if problems:
        raise RuntimeError('STOP -- data conflicts with the paper (Table 2). Report, do not resolve silently:\n  - ' + '\n  - '.join(problems))
    print('feature source:', FEATURE_SOURCE, '| files:', {c: os.path.basename(INFO[c]['path']) for c in INFO})
    print('\nde-duplication difference (raw -> kept):  ' + ' | '.join(f"{c}: -{INFO[c]['dup']}" for c in INFO))
    print('feature scale per country (mean / std / min / max):')
    for c in INFO: print(f"  {c}: {INFO[c]['mean']:.1f} / {INFO[c]['std']:.1f} / {INFO[c]['vmin']:.1f} / {INFO[c]['vmax']:.1f}")
    means = [INFO[c]['mean'] for c in INFO]
    if max(means) > 10 * max(min(means), 1e-9): print('  !! WARNING: country feature means differ by >10x -- check that all files use the same reflectance scale')
    _check(len(meta) == len(FEATURES['USA']) and (meta['crop_id'].values == NPZ_LBL['USA']).all(), 'USA state table is not aligned with the USA features')
    _check(set(meta['state']) <= set(ALL_STATES), 'unknown state codes in the USA table')
    # LEAKAGE: no target sample may be identical to any USA row (targets never enter training)
    hu = row_hash_set(FEATURES['USA'])
    for t in TARGETS_TO_RUN:
        n_dup = len(hu & row_hash_set(FEATURES[t])); print(f'  leakage check USA vs {t}: {n_dup} identical rows')
        _check(n_dup == 0, f'{t}: {n_dup} target rows are identical to USA rows -- leakage; STOP and report')
    big = meta.groupby('state').size(); print(f"USA: {len(meta)} samples, {meta['state'].nunique()} states ({int((big >= MIN_STATE_ROWS).sum())} with >= {MIN_STATE_ROWS} rows) -- state table aligned row by row")
    print('DATA VERIFICATION PASSED\n')
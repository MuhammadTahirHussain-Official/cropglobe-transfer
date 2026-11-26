# ======================================================================================
# DATA (Level 2): load the Sentinel-2 median time series (official files), de-duplicate like the authors' loader, attach the FRENCH REGION to every source sample from the verified
#       row map, and VERIFY everything before any training.   (Torch-free: arrays stay numpy; the torch cell converts them.)
#       Differences from Level 1: the source is France (units = regions); there is no point-in-polygon step -- the region of a row is read from the raw FILE NAME through the exact
#       row-by-row match of raw points to packaged rows (FRA_join_verification.md); the 96 packaged rows that match two identical raw points are placed only when their position is certain.
# ======================================================================================
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


FRA_FILE_RE = re.compile(r'^S2_FRA_(?P<tok>[a-z]+)_LAMB93_R(?P<reg>\d+)\.csv$')

def fra_blocks_from_row_map(row_map_path):
    # Re-derives the 64 file blocks from the verified row map (used to check the embedded FRA_BLOCKS, and in the tests).  STOP on any structural difference from what the verification found.
    h = _sha256_file(row_map_path); _check(h == FRA_ROW_MAP_SHA256, f'STOP: the row map {row_map_path} is not the verified file (sha256 {h[:16]}... != {FRA_ROW_MAP_SHA256[:16]}...).')
    r = pd.read_csv(row_map_path); need = ['file', 'raw_index', 'n_match', 'npz_row', 'region_code', 'crop_token']; miss = [c for c in need if c not in r.columns]
    _check(not miss, f'row map lacks columns {miss}')
    _check(len(r) == FRA_ROW_MAP_ROWS, f'row map has {len(r)} rows, expected {FRA_ROW_MAP_ROWS}')
    amb = r['n_match'] > 1; _check(int(amb.sum()) == FRA_ROW_MAP_AMBIGUOUS and bool((r.loc[amb, 'npz_row'] == -1).all()) and bool((r.loc[~amb, 'npz_row'] >= 0).all()) and bool((r['n_match'] >= 1).all()), 'row-map match structure differs from the verification')
    r['region'] = r['region_code'].astype(int).astype(str); _check(set(r['region']) == set(SOURCE_UNITS), f'regions in the map {sorted(set(r["region"]))} != {SOURCE_UNITS}')
    _check(r.groupby('region').size().to_dict() == FRA_REGION_TOTALS, 'points per region differ from the recorded table')
    pf = r['file'].map(lambda f: FRA_FILE_RE.match(str(f))); _check(pf.notna().all(), 'a file name in the row map does not match S2_FRA_<crop>_LAMB93_R<region>.csv')
    _check(bool((pf.map(lambda m: m.group('tok')) == r['crop_token']).all() and (pf.map(lambda m: str(int(m.group('reg')))) == r['region']).all()), 'file name disagrees with the crop_token / region columns')
    _check(set(r['crop_token']) <= set(RAW_TO_CROP), 'unknown crop token in the row map')
    u = r[~amb]; _check(u['npz_row'].is_unique, 'a packaged row is claimed by two raw points')
    for f_, g in u.groupby('file'): _check(bool((np.diff(g.sort_values('raw_index')['npz_row'].values) > 0).all()), f'{f_}: packaged rows do not increase with the raw position')
    b = u.groupby('file').agg(lo=('npz_row', 'min'), hi=('npz_row', 'max'), tok=('crop_token', 'first'), region=('region', 'first')).sort_values('lo')
    return [(t, g, int(lo), int(hi)) for t, g, lo, hi in zip(b['tok'], b['region'], b['lo'], b['hi'])]

def build_fra_meta(lbl_raw, row_map_path=None, log=print):
    # Returns (meta over ALL N packaged rows, report).  meta columns: npz_row, crop_id, state (region code, str), assigned (bool).
    #   * FRA_BLOCKS (embedded, derived from the verified row map) must be disjoint, ordered as (crop label, token A-Z, region), inside the file, and give the recorded rows per region
    #   * every row inside a block gets the block's region; rows outside every block (cannot be placed with certainty) stay unassigned and are excluded later
    #   * the packaged label of EVERY assigned row must equal the label of its raw crop token (full-row check against the real data)
    #   * if the row-map file is available it is re-read and must give exactly the embedded blocks
    N = len(lbl_raw); B = pd.DataFrame(FRA_BLOCKS, columns=['tok', 'region', 'lo', 'hi']); _check(len(B) == 64 and set(B['region']) == set(SOURCE_UNITS), 'embedded FRA_BLOCKS is not the 64-file table')
    _check(N == FRA_ROW_MAP_ROWS, f'the packaged France file has {N} rows, expected {FRA_ROW_MAP_ROWS}')
    B['label'] = B['tok'].map(lambda t: CROP_ID[RAW_TO_CROP[t]]); _check(bool((B['lo'].values[1:] > B['hi'].values[:-1]).all()) and bool((B['lo'] <= B['hi']).all()) and int(B['hi'].max()) < N, 'FRA_BLOCKS are not disjoint / ordered / inside the file')
    order_ok = B.sort_values(['label', 'tok', 'region'], key=lambda c: c.astype(int) if c.name == 'region' else c).index.tolist() == B.index.tolist()
    _check(order_ok, 'the block order is not (crop label, token A-Z, region): the ordering convention verified for France does not hold')
    region_of = np.full(N, '', dtype=object); lab_blk = np.full(N, -1)
    for b in B.itertuples(): region_of[b.lo:b.hi + 1] = b.region; lab_blk[b.lo:b.hi + 1] = b.label
    assigned = region_of != ''; crop_id = np.asarray(lbl_raw).astype(int)
    _check(bool((lab_blk[assigned] == crop_id[assigned]).all()), 'the packaged label of some row differs from the label of its raw crop token (full-row label check)')
    per = {k: int(v) for k, v in pd.Series(region_of[assigned]).value_counts().sort_index().items()}
    _check(per == FRA_ASSIGNED_PER_REGION, f'rows per region from the blocks {per} != recorded {FRA_ASSIGNED_PER_REGION}')
    n_excl = int((~assigned).sum()); gaps = np.flatnonzero(~assigned)
    if row_map_path is not None:
        _check(fra_blocks_from_row_map(row_map_path) == [tuple(x) for x in FRA_BLOCKS], 'the row-map file does not reproduce the embedded FRA_BLOCKS')
        log('  row-map file re-read: reproduces the embedded blocks exactly')
    meta = pd.DataFrame({'npz_row': np.arange(N), 'crop_id': crop_id, 'state': np.where(assigned, region_of, 'NA'), 'assigned': assigned})
    rep = dict(rows=N, blocks=len(B), block_order_ok=bool(order_ok), assigned=int(assigned.sum()), excluded_outside_blocks=n_excl, excluded_rows=gaps.tolist(), labels_checked=int(assigned.sum()), rows_per_region=per, row_map_sha256=FRA_ROW_MAP_SHA256)
    log(f'  FRA region table: 64 raw-file blocks (from the verified row map, sha256 {FRA_ROW_MAP_SHA256[:12]}...) tile the packaged rows in the order (label, token, region); labels of all {int(assigned.sum())} assigned rows equal their raw crop token;')
    log(f'  {n_excl} rows lie between blocks (matched two identical raw points; region NOT certain) -> excluded')
    return meta, rep

def load_all_data(data_dir, out_dir, row_map_path=None):
    cache_dir = os.path.join(out_dir, 'cache'); os.makedirs(cache_dir, exist_ok=True)
    needed = [SOURCE] + [t for t in TARGETS_TO_RUN]; FEATURES, NPZ_LBL, INFO, meta, report = {}, {}, {}, None, None
    for c in needed:
        path = find_npz(data_dir, c); lbl, a = load_country(path, c); keep = unique_row_index(a); extra = {}
        if c == SOURCE:
            meta_full, report = build_fra_meta(lbl, row_map_path); ok = meta_full['assigned'].values[keep]
            extra = dict(excluded_unassigned=int((~ok).sum())); keep = keep[ok]; meta = meta_full.iloc[keep].reset_index(drop=True)
        FEATURES[c] = np.ascontiguousarray(a[keep][:, None, :, :]); NPZ_LBL[c] = lbl[keep]
        INFO[c] = dict(path=path, raw=len(lbl), kept=len(keep), dup=len(lbl) - len(keep) - extra.get('excluded_unassigned', 0), shape=tuple(FEATURES[c].shape), keep=keep, raw_by_class=pd.Series(lbl).map(CROP_LABELS).value_counts().reindex(CROP_ORDER, fill_value=0).to_dict(),
                       mean=float(a.mean()), std=float(a.std()), vmin=float(a.min()), vmax=float(a.max()), **extra); del a
    meta['crop'] = meta['crop_id'].map(CROP_LABELS); meta = meta[['npz_row', 'crop_id', 'state', 'crop']]
    print(f"SOURCE {SOURCE}: raw {INFO[SOURCE]['raw']} | removed as exact duplicates {INFO[SOURCE]['dup']} | excluded (region not certain) {INFO[SOURCE]['excluded_unassigned']} | kept {INFO[SOURCE]['kept']}")
    return FEATURES, NPZ_LBL, INFO, meta, report

def row_hash_set(F):
    a = np.ascontiguousarray(F.reshape(len(F), -1)); return {hashlib.blake2b((a[i] + 0.0).tobytes(), digest_size=16).digest() for i in range(len(a))}


def verify_data(FEATURES, NPZ_LBL, INFO, meta):
    # EVERY check here is a STOP: raw counts, label sets, class counts vs paper v3 Table 2, no target row inside the source table, region alignment.
    print('\n================ DATA VERIFICATION (paper arXiv 2509.03497 v3, Table 2) ================')
    rows = []
    for c in INFO:
        rows.append(dict(country=c, raw=INFO[c]['raw'], table2_total=TABLE2_TOTAL[c], kept_after_dedup=INFO[c]['kept'], removed_duplicates=INFO[c]['dup'], shape=INFO[c]['shape'], **{f'kept_{k}': v for k, v in
                         pd.Series(NPZ_LBL[c]).map(CROP_LABELS).value_counts().reindex(CROP_ORDER, fill_value=0).to_dict().items()}))
    print(pd.DataFrame(rows).to_string(index=False))
    problems = []
    for c in INFO:
        if INFO[c]['raw'] != TABLE2_TOTAL[c]: problems.append(f'{c}: raw rows {INFO[c]["raw"]} != Table 2 total {TABLE2_TOTAL[c]}')
        if INFO[c]['raw_by_class'] != TABLE2[c]: problems.append(f'{c}: raw class counts {INFO[c]["raw_by_class"]} != Table 2 {TABLE2[c]}')
    for t in [SOURCE] + list(TARGETS_TO_RUN):
        got = sorted([CROP_LABELS[i] for i in np.unique(NPZ_LBL[t])], key=CROP_ORDER.index); exp = sorted(DEFINED_CLASSES[t], key=CROP_ORDER.index)
        rawset = sorted([c for c, n in INFO[t]['raw_by_class'].items() if n > 0], key=CROP_ORDER.index)
        print(f'  label set {t}: after dedup {got} | raw {rawset} | Table 2 {exp} -> {"OK" if got == exp and rawset == exp else "MISMATCH"}')
        if got != exp or rawset != exp: problems.append(f'{t}: label set data(raw {rawset}, dedup {got}) != Table 2 {exp}')
    for c in INFO:
        if 'median_S2_' not in os.path.basename(INFO[c]['path']): problems.append(f'{c}: file {INFO[c]["path"]} is not a Sentinel-2 median file (AlphaEarth embeddings must not be used here)')
        if INFO[c]['shape'][1:3] != (1, 10): problems.append(f'{c}: unexpected tensor shape {INFO[c]["shape"]} (need (N,1,10,T))')
        elif INFO[c]['shape'][3] < 67: problems.append(f'{c}: time axis {INFO[c]["shape"][3]} too short for the official DOY 121-334 window (needs >= 67 five-day steps)')
    if problems: raise RuntimeError('STOP -- data conflicts with the paper (Table 2). Report, do not resolve silently:\n  - ' + '\n  - '.join(problems))
    print('feature source:', FEATURE_SOURCE, '| files:', {c: os.path.basename(INFO[c]['path']) for c in INFO})
    print('\nde-duplication difference (raw -> kept):  ' + ' | '.join(f"{c}: -{INFO[c]['dup']}" for c in INFO))
    print('feature scale per country (mean / std / min / max):')
    for c in INFO: print(f"  {c}: {INFO[c]['mean']:.1f} / {INFO[c]['std']:.1f} / {INFO[c]['vmin']:.1f} / {INFO[c]['vmax']:.1f}")
    means = [INFO[c]['mean'] for c in INFO]
    if max(means) > 10 * max(min(means), 1e-9): print('  !! WARNING: country feature means differ by >10x -- check that all files use the same reflectance scale')
    _check(len(meta) == len(FEATURES[SOURCE]) and (meta['crop_id'].values == NPZ_LBL[SOURCE]).all(), 'source region table is not aligned with the source features')
    _check(set(meta['state']) <= set(SOURCE_UNITS), f'unknown region codes in the source table: {sorted(set(meta["state"]) - set(SOURCE_UNITS))}')
    hs = row_hash_set(FEATURES[SOURCE])                                              # LEAKAGE: no target sample may be identical to any source row (targets never enter training)
    for t in TARGETS_TO_RUN:
        n_dup = len(hs & row_hash_set(FEATURES[t])); print(f'  leakage check {SOURCE} vs {t}: {n_dup} identical rows')
        _check(n_dup == 0, f'{t}: {n_dup} target rows are identical to {SOURCE} rows -- leakage; STOP and report')
    big = meta.groupby('state').size(); print(f"{SOURCE}: {len(meta)} samples, {meta['state'].nunique()} regions ({int((big >= MIN_STATE_ROWS).sum())} with >= {MIN_STATE_ROWS} rows) -- region table aligned row by row")

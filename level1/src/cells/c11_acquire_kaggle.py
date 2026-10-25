# ======================================================================================
# [KAGGLE] VERIFY YOUR UPLOADED FILES FIRST, then fetch only what is missing.  The authors' repo is PINNED to the commit in the spec.
# Everything below is printed as a table: found / missing / downloaded, with sizes.  Nothing is trained here.
# ======================================================================================
import hashlib, zipfile, filecmp, pickle, re, fnmatch
NEEDED = ['USA'] + list(TARGETS_TO_RUN)
HF_REPO = 'x-ytong/CropGlobe'; HF_SUBDIR = 'for_CropNet'
APPROX_MB = {'USA': 253, 'ARG': 17.6, 'FRA': 202, 'CHN': 10.1, 'AUS': 34.2}                 # sizes shown on the Hugging Face page (rounded); a >10% difference only WARNS
INV = []                                                                                    # inventory rows
def _mb(p): return round(os.path.getsize(p) / 1e6, 1) if p and os.path.exists(p) else None
def _sha(p, n=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(n), b''): h.update(b)
    return h.hexdigest()
def _search(root, pattern):                                                                  # every file below root whose name matches (follows symlinks, any depth)
    out = []
    for r, _, fs in os.walk(root, followlinks=True):
        out += [os.path.join(r, f) for f in fs if fnmatch.fnmatch(f, pattern)]
    return sorted(out)

def _show_inventory():
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 150)
    print('\n================ FILE INVENTORY (verified before anything is used) ================'); print(pd.DataFrame(INV).to_string(index=False))
# ---------------- 1) Sentinel-2 median files (CropGlobe) ----------------
FARM = os.path.join(KAGGLE_WORK, 'CropGlobe_data', HF_SUBDIR); os.makedirs(FARM, exist_ok=True); DATA_DIR = os.path.dirname(FARM)
for c in NEEDED:
    name = f'median_S2_{c}_month1-12_day5.npz'; hits = _search(os.path.join(PROJECT_ROOT, 'CropGlobe'), name) or _search(KAGGLE_INPUT, name); src, status = None, ''
    if hits:
        if len(hits) > 1 and len({_sha(h) for h in hits}) > 1: raise RuntimeError(f'STOP: {len(hits)} DIFFERENT files named {name} in your dataset: {hits}. Remove the wrong ones; no file is chosen silently.')
        src, status = hits[0], 'found in your dataset' + (f' ({len(hits)} identical copies)' if len(hits) > 1 else '')
    else:
        if not INTERNET: INV.append(dict(item=name, status='MISSING (Internet OFF -> cannot download)', where='', MB=None)); continue
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(repo_id=HF_REPO, repo_type='dataset', filename=f'{HF_SUBDIR}/{name}', local_dir=os.path.join(KAGGLE_WORK, 'CropGlobe_downloaded')); status = 'MISSING in your dataset -> DOWNLOADED from Hugging Face'
    link = os.path.join(FARM, name)
    if os.path.lexists(link): os.remove(link)
    os.symlink(os.path.abspath(src), link)
    try:
        with np.load(link) as z: assert {'fea', 'lbl'} <= set(z.files), f'keys {z.files}'; n = int(z['lbl'].shape[0])
        ok = f'readable, {n} rows'
    except Exception as e: ok = 'UNREADABLE: ' + repr(e)[:80]
    w = ''
    if c in APPROX_MB and abs(_mb(link) - APPROX_MB[c]) > 0.10 * APPROX_MB[c]: w = f'  WARN size {_mb(link)} MB vs about {APPROX_MB[c]} MB on Hugging Face'
    INV.append(dict(item=name, status=status + ' | ' + ok + w, where=src, MB=_mb(link)))

# ---------------- 2) the authors' CropNet repo ----------------
REPO_NEED = ['models/CropNet.py', 'data/augment.py', 'trainers/torch_trainer.py']; REPO_OPT = ['run_CropNet_AUGM.py']
UP_REPO = next((p for p in [os.path.join(PROJECT_ROOT, 'CropNet')] + [os.path.dirname(os.path.dirname(h)) for h in _search(PROJECT_ROOT, 'CropNet.py') if h.endswith(os.path.join('models', 'CropNet.py'))] if os.path.isdir(p)), None)
REPO_DIR = os.path.join(KAGGLE_WORK, 'CropNet'); subprocess.run(['git', 'config', '--global', '--add', 'safe.directory', '*'], capture_output=True)
def _git(repo, *a): return subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True)
def _repo_state(repo):
    miss = [f for f in REPO_NEED if not os.path.isfile(os.path.join(repo, f))]
    if not os.path.isdir(os.path.join(repo, '.git')): return miss, 'no .git folder (Kaggle usually drops it on upload)', None
    head = _git(repo, 'rev-parse', 'HEAD').stdout.strip(); dirty = _git(repo, 'status', '--porcelain', '--untracked-files=no').stdout.strip()
    if head != EXPECTED_CROPNET_COMMIT: return miss, f'at commit {head[:10] or "?"}, spec says {EXPECTED_CROPNET_COMMIT[:10]}', head
    if dirty: return miss, 'local modifications: ' + dirty.replace(chr(10), '; ')[:120], head
    return miss, '', head
use_uploaded = False
if UP_REPO:
    if os.path.exists(REPO_DIR): shutil.rmtree(REPO_DIR)
    shutil.copytree(UP_REPO, REPO_DIR, symlinks=True); miss, why, _h = _repo_state(REPO_DIR)
    INV.append(dict(item='CropNet repo (uploaded)', status=('missing files ' + str(miss) if miss else 'core files present') + (' | NOT VERIFIABLE: ' + why if why else ' | pinned commit verified, clean'), where=UP_REPO, MB=None))
    use_uploaded = not miss and not why
else: INV.append(dict(item='CropNet repo (uploaded)', status='MISSING', where='', MB=None))
if not use_uploaded:
    _show_inventory()
    assert INTERNET, ('the uploaded CropNet copy cannot be verified against the pinned commit ' + EXPECTED_CROPNET_COMMIT + ' (see the table above) and Internet is OFF. '
                      'Turn Internet on so the pinned commit can be cloned from github.com/x-ytong/CropNet, or upload the repo WITH its .git folder at that commit.')
    pinned = os.path.join(KAGGLE_WORK, 'CropNet_pinned')
    if os.path.exists(pinned): shutil.rmtree(pinned)
    subprocess.run(['git', 'clone', '-q', 'https://github.com/x-ytong/CropNet.git', pinned], check=True)
    if _git(pinned, 'cat-file', '-e', EXPECTED_CROPNET_COMMIT + '^{commit}').returncode != 0:
        raise RuntimeError(f'STOP -- spec conflict: commit {EXPECTED_CROPNET_COMMIT} does not exist in x-ytong/CropNet. Report; do not substitute another commit silently.')
    r = _git(pinned, 'checkout', '--detach', EXPECTED_CROPNET_COMMIT); assert r.returncode == 0, r.stderr
    if UP_REPO:                                                                                      # how does the uploaded copy differ from the pinned one?  (information only)
        diff = [f for f in REPO_NEED + REPO_OPT if os.path.isfile(os.path.join(pinned, f)) and (not os.path.isfile(os.path.join(UP_REPO, f)) or not filecmp.cmp(os.path.join(pinned, f), os.path.join(UP_REPO, f), shallow=False))]
        print('uploaded CropNet vs pinned commit -- differing / missing core files:', diff or 'none (contents identical; only the git history could not be verified)')
    REPO_DIR = pinned; INV.append(dict(item='CropNet repo (used)', status='DOWNLOADED and pinned from GitHub', where=pinned, MB=None))
else: INV.append(dict(item='CropNet repo (used)', status='your upload', where=REPO_DIR, MB=None))
GIT_COMMIT = _git(REPO_DIR, 'rev-parse', 'HEAD').stdout.strip(); assert GIT_COMMIT == EXPECTED_CROPNET_COMMIT, f'repo at {GIT_COMMIT}, spec says {EXPECTED_CROPNET_COMMIT}'
_dirty = _git(REPO_DIR, 'status', '--porcelain', '--untracked-files=no').stdout.strip(); assert not _dirty, f'the authors\' code must be imported UNCHANGED, but the repo has local modifications:\n{_dirty}'
for f in REPO_NEED: assert os.path.isfile(os.path.join(REPO_DIR, f)), f'authors\' repo is incomplete: {f}'
for f in REPO_OPT: INV.append(dict(item=f'CropNet/{f}', status='present' if os.path.isfile(os.path.join(REPO_DIR, f)) else 'not found (only used as reference)', where='', MB=None))
print('authors\' repo pinned at', GIT_COMMIT, '(clean, unmodified)')

# ---------------- 3) USA state table: a cached usa_meta.csv (tiny) OR the raw USA csv files with geometry ----------------
RAW_USA_DIR = os.path.join(PROJECT_ROOT, 'CropGlobe_raw', 'Sentinel-2', 'USA')
_meta = _search(PROJECT_ROOT, 'usa_meta.csv') + [m for m in _search(KAGGLE_INPUT, 'usa_meta.csv') if m not in _search(PROJECT_ROOT, 'usa_meta.csv')]
EXTRA_META_PATHS = list(_meta)
if _meta: INV.append(dict(item='usa_meta.csv (state table)', status=f'found {len(_meta)} (checked against the USA npz row index before use; stale copies are ignored)', where=_meta[0], MB=_mb(_meta[0])))
if not os.path.isdir(RAW_USA_DIR):
    _any = _search(KAGGLE_INPUT, 'S2_USA_*.csv'); RAW_USA_DIR = os.path.dirname(_any[0]) if _any else RAW_USA_DIR
_csv = sorted(glob.glob(os.path.join(RAW_USA_DIR, '**', '*.csv'), recursive=True)) if os.path.isdir(RAW_USA_DIR) else []
_parsed = [parse_usa_filename(os.path.basename(p)) for p in _csv]; _bad = [os.path.basename(p) for p, q in zip(_csv, _parsed) if q is None]
INV.append(dict(item='raw USA csv files (geometry)', status=(f'{len(_csv)} files, {len(_csv) - len(_bad)} parse as S2_USA_<crop>[_<STATE>].csv' + (f'; UNPARSEABLE: {_bad[:5]}' if _bad else '')) if _csv else 'not found', where=RAW_USA_DIR, MB=None))
if not _csv and not _meta:
    _show_inventory()
    raise RuntimeError('STOP: neither a usa_meta.csv nor the raw USA csv files (CropGlobe_raw/Sentinel-2/USA/S2_USA_*.csv) were found in your dataset, and the Hugging Face dataset x-ytong/CropGlobe '
                       'does not contain them (its folders are AlphaEarth, Harmonic, Median, Presto, for_CropNet). Without a state table the USA rows cannot be assigned to states. Add the raw csv folder, or the usa_meta.csv exported by the Colab run, to your dataset.')

INVENTORY = pd.DataFrame(INV); _show_inventory()
_lost = [r['item'] for r in INV if str(r['status']).startswith('MISSING') or 'UNREADABLE' in str(r['status'])]
assert not _lost, f'STOP -- required files are missing or unreadable: {_lost}'
INVENTORY.to_csv(os.path.join(OUT_DIR, 'file_inventory.csv'), index=False)
print('all required files verified.\n')

def get_usa_geo_by_file():
    cache = os.path.join(OUT_DIR, 'cache', 'usa_geo.pkl'); os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(cache): return pickle.load(open(cache, 'rb'))
    assert os.path.isdir(RAW_USA_DIR), f'{RAW_USA_DIR} not found'
    paths = sorted(glob.glob(os.path.join(RAW_USA_DIR, '**', '*.csv'), recursive=True)); geo = {}
    for i, p in enumerate(paths):
        fn = os.path.basename(p)
        if parse_usa_filename(fn) is None: continue
        assert fn not in geo, f'duplicate file name {fn}'
        geo[fn] = read_geo_column(p)
        if (i + 1) % 25 == 0: print(f'  read geometry of {i + 1}/{len(paths)} raw USA files')
    pickle.dump(geo, open(cache, 'wb')); return geo

# ---- namespace guard: never mix with, or overwrite, results of an earlier notebook (spec 7)
def check_namespace(write=False):
    mp, rp = os.path.join(OUT_DIR, 'manifest.json'), os.path.join(OUT_DIR, 'results.csv')
    snap = {k: json.loads(json.dumps(globals()[k], default=str)) for k in LOCKED}
    if os.path.exists(mp):
        m = json.load(open(mp))
        if m.get('spec') != SPEC_TAG:
            raise RuntimeError(f'STOP: {OUT_DIR} holds a manifest from an EARLIER notebook (spec={m.get("spec")}). Spec 7 says never touch earlier outputs: rename/move that folder (keep it), then re-run.')
        diff = [k for k in LOCKED if m['locked'].get(k) != snap[k]]
        if diff: raise RuntimeError(f'STOP: locked design differs from the manifest in {diff}. Use a new namespace instead of mixing designs.')
        if write:                                                                  # v3.8: every design amendment is recorded (dated) in the manifest the first time it is seen; nothing changes silently
            _have = {a_.get('id') for a_ in m.get('amendments', [])}; _new = [dict(a_, recorded=datetime.datetime.now().isoformat(timespec='seconds'), notebook=NOTEBOOK_VERSION) for a_ in V38_AMENDMENTS if a_['id'] not in _have]
            if _new: m['amendments'] = m.get('amendments', []) + _new; atomic_write(mp, lambda f: f.write(json.dumps(m, indent=1)), mode='w'); print('manifest: recorded amendments', [a_['id'] for a_ in _new])
    elif os.path.exists(rp):
        raise RuntimeError(f'STOP: {rp} exists without a v3-spec manifest (results of an earlier notebook). Move that folder aside (do not delete) and re-run.')
    elif write:
        man = dict(spec=SPEC_TAG, experiment=EXPERIMENT_VERSION, created=datetime.datetime.now().isoformat(timespec='seconds'), locked=snap, cropnet_commit=GIT_COMMIT,
                   table2=TABLE2, table3_reference_context_only=TABLE3_REF_MF1,
                   amendments=[dict(a_, recorded=datetime.datetime.now().isoformat(timespec='seconds'), notebook=NOTEBOOK_VERSION) for a_ in V38_AMENDMENTS])
        atomic_write(mp, lambda f: f.write(json.dumps(man, indent=1)), mode='w')
    return True
check_namespace(write=False)
print('Kaggle data + code ready. Sentinel-2 files for', NEEDED, 'found. Namespace OK:', OUT_DIR)
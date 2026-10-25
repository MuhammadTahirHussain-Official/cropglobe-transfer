# ======================================================================================
# [COLAB] DATA + CODE: use the files already in Drive; download only what is missing.  The authors' repo is PINNED to the commit in the spec.
# ======================================================================================
NEEDED = ['USA'] + list(TARGETS_TO_RUN)
def _have(c): return bool(glob.glob(os.path.join(DATA_DIR, '**', f'median_S2_{c}_month1-12_day5.npz'), recursive=True))
if not all(_have(c) for c in NEEDED):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id='x-ytong/CropGlobe', repo_type='dataset', local_dir=DATA_DIR)
assert all(_have(c) for c in NEEDED), f'missing Sentinel-2 median files for {[c for c in NEEDED if not _have(c)]} in {DATA_DIR}'
if not os.path.isdir(REPO_DIR):
    subprocess.run(['git', 'clone', 'https://github.com/x-ytong/CropNet.git', REPO_DIR], check=True)
def _git(*a): return subprocess.run(['git', '-C', REPO_DIR, *a], capture_output=True, text=True)
if _git('rev-parse', 'HEAD').stdout.strip() != EXPECTED_CROPNET_COMMIT:
    _git('fetch', '--all', '--tags')
    if _git('cat-file', '-e', EXPECTED_CROPNET_COMMIT + '^{commit}').returncode != 0:
        raise RuntimeError(f'STOP -- spec conflict: commit {EXPECTED_CROPNET_COMMIT} does not exist in x-ytong/CropNet. Report; do not substitute another commit silently.')
    r = _git('checkout', '--detach', EXPECTED_CROPNET_COMMIT); assert r.returncode == 0, r.stderr
GIT_COMMIT = _git('rev-parse', 'HEAD').stdout.strip(); assert GIT_COMMIT == EXPECTED_CROPNET_COMMIT, f'repo at {GIT_COMMIT}, spec says {EXPECTED_CROPNET_COMMIT}'
_dirty = _git('status', '--porcelain', '--untracked-files=no').stdout.strip()
assert not _dirty, f'the authors\' code must be imported UNCHANGED, but the repo has local modifications:\n{_dirty}'
assert os.path.isfile(os.path.join(REPO_DIR, 'models', 'CropNet.py')), 'authors\' repo is incomplete'
print('authors\' repo pinned at', GIT_COMMIT, '(clean, unmodified)')

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
print('Colab data + code ready. Sentinel-2 files for', NEEDED, 'found. Namespace OK:', OUT_DIR)
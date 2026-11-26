# ======================================================================================
# TEST GATE - PART A2 (no GPU, no repo, no real data): the DATA functions on synthetic inputs (v3.3).  Runs after the data-function cell; blocks training like Part A.
# ======================================================================================
_n0 = len(TEST_REPORT)
@_t('V33-05 find_npz: one match -> that file; identical copies -> listed + documented rule; DIFFERENT files -> STOP')
def _():
    with tempfile.TemporaryDirectory() as td:
        name = 'median_S2_ARG_month1-12_day5.npz'
        for sub, payload in (('a', b'same'), ('for_CropNet/b', b'same')): os.makedirs(os.path.join(td, sub), exist_ok=True); open(os.path.join(td, sub, name), 'wb').write(payload)
        out = io.StringIO()
        with contextlib.redirect_stdout(out): p = find_npz(td, 'ARG', log=print)
        assert 'for_CropNet' in p and 'byte-identical' in out.getvalue() and out.getvalue().count('sha256') == 2
        open(os.path.join(td, 'a', name), 'wb').write(b'different')
        try:
            with contextlib.redirect_stdout(io.StringIO()): find_npz(td, 'ARG', log=print)
            raise RuntimeError('no stop')
        except RuntimeError as e: assert 'DIFFERENT files' in str(e)
        try: find_npz(td, 'FRA'); raise RuntimeError('no stop')
        except AssertionError as e: assert 'not found' in str(e)

def _p4_fixture(n_states=10, per=60, seed=0):
    rng = np.random.default_rng(seed); sts = [f'S{i:02d}' for i in range(n_states)]; st, cr, X = [], [], []
    shift = {s: rng.normal(0, 1, 730) * (0.2 + 0.25 * i) for i, s in enumerate(sts)}          # state-specific phenology shift, growing with i
    for s in sts:
        for ci_, c in enumerate(['soybeans', 'corn', 'wheat']):
            m = np.full(730, ci_ - 1.0)
            X.append((m + shift[s] + rng.normal(0, 0.3, (per, 730))).astype(np.float32)); st += [s] * per; cr += [c] * per
    return np.vstack(X).reshape(-1, 1, 10, 73), np.array(st), np.array(cr), sts

@_t('P4-01 diversity distances: symmetric, zero diagonal, USA-only inputs, NaN where a (state,crop) has < MIN_PROFILE_N samples')
def _():
    X, st, cr, sts = _p4_fixture(); DD = compute_diversity_distances(X, st, cr, min_n=20)
    for c in ('soybeans', 'corn', 'wheat'):
        M = DD['dist'][c]; assert M.shape == (10, 10) and np.allclose(M.values, M.values.T) and np.allclose(np.diag(M.values), 0) and M.notna().all().all()
    keep = ~((st == 'S03') & (cr == 'soybeans')) | (np.arange(len(st)) % 60 < 5); DD2 = compute_diversity_distances(X[keep], st[keep], cr[keep], min_n=20)
    assert DD2['dist']['soybeans'].loc['S03'].drop('S03').isna().all() and DD2['dist']['corn'].loc['S03'].notna().all()

@_t('P4-02 select_extreme_arms equals brute force (top / bottom N_ARM), is deterministic, and the two arms are disjoint; scores are mean pairwise distance')
def _():
    X, st, cr, sts = _p4_fixture(); DD = compute_diversity_distances(X, st, cr, min_n=20); D = DD['dist']['soybeans']; Dm = D.loc[sts, sts].values
    e1 = select_extreme_arms(sts, 3, Dm, 'ARG', 1000, BASE_MODE); e2 = select_extreme_arms(sts, 3, Dm, 'ARG', 1000, BASE_MODE); assert e1['diverse'] == e2['diverse'] and e1['clustered'] == e2['clustered']
    allc = list(itertools.combinations(range(10), 3)); sc = {c: np.mean([Dm[a, b] for a, b in itertools.combinations(c, 2)]) for c in allc}; srt = sorted(allc, key=lambda c: (-round(sc[c], 12), c))
    assert [tuple(sts.index(s) for s in c) for c in e1['diverse']] == srt[:N_ARM] and e1['enumerated'] and e1['n_scored'] == 120
    assert not (set(map(tuple, e1['diverse'])) & set(map(tuple, e1['clustered']))) and min(sc[c] for c in srt[:N_ARM]) >= max(sc[c] for c in srt[-N_ARM:])
    v, pct = combo_score_pct(e1['diverse'][0], sts, Dm, e1['scores']); assert abs(v - sc[srt[0]]) < 1e-9 and pct == 1.0

@_t('P4-03 large pools: scoring uses a fixed seeded sample of SCORE_SAMPLE distinct combinations (enumerated=False) and is reproducible')
def _():
    n = 40; rng = np.random.default_rng(1); A = rng.normal(0, 1, (n, 5)); Dm = np.sqrt(((A[:, None] - A[None]) ** 2).sum(-1)); pool = [f'P{i:02d}' for i in range(n)]
    old = (globals()['ENUM_LIMIT'], globals()['SCORE_SAMPLE'])
    try:
        globals()['ENUM_LIMIT'], globals()['SCORE_SAMPLE'] = 1000, 300
        a = select_extreme_arms(pool, 5, Dm, 'USA', 2000, BASE_MODE); b = select_extreme_arms(pool, 5, Dm, 'USA', 2000, BASE_MODE)
        assert not a['enumerated'] and a['n_scored'] == 300 and a['diverse'] == b['diverse'] and len(a['diverse']) == N_ARM and a['scores'].max() >= a['scores'].min()
    finally: globals()['ENUM_LIMIT'], globals()['SCORE_SAMPLE'] = old

@_t('P4-04 build_plan: SELECTION_ARMS=[random] reproduces the pre-Phase-4 random combinations; the arms share ONE pool; every combination passes combo_capacity_ok; k=1 is one shared census')
def _():
    X, st, cr, sts = _p4_fixture(12, 150); cap = make_cap(st, cr); g = globals(); names = ['BUDGETS', 'K_LIST', 'MIN_STATE_ROWS', 'TARGETS_TO_RUN', 'SELECTION_ARMS', 'MIN_KPOINTS', 'MIN_COMBOS', 'DIVERSITY_DIST', 'K_MAX']
    saved = {n_: g.get(n_) for n_ in names}
    try:
        g.update(BUDGETS=[300], K_LIST=[1, 2, 3, 4], MIN_STATE_ROWS=100, TARGETS_TO_RUN=['ARG'], MIN_KPOINTS=3, MIN_COMBOS=5, K_MAX={'ARG': 8, 'USA': 8, 'CHN': 2, 'AUS': 2, 'FRA': 8})
        g['SELECTION_ARMS'] = ['random']; c0, g0 = build_plan(cap); old_keys = {(c['k'], c['states']) for c in c0}
        g['SELECTION_ARMS'] = ['random', 'diverse', 'clustered']; g['DIVERSITY_DIST'] = compute_diversity_distances(X, st, cr, min_n=20); c1, g1 = build_plan(cap)
        assert {(c['k'], c['states']) for c in c1 if 'random' in c['arms']} == old_keys, 'the random arm changed'
        for c in c1: assert combo_capacity_ok(c['states'], c['target'], c['budget'], cap, c['composition'])
        k1 = [c for c in c1 if c['k'] == 1]; assert k1 and all(set(c['arms']) == {'random', 'diverse', 'clustered'} for c in k1)
        k2 = [c for c in c1 if c['k'] == 2]; ad = [c for c in k2 if 'diverse' in c['arms']]; ac = [c for c in k2 if 'clustered' in c['arms']]
        assert len(ad) == N_ARM and len(ac) == N_ARM and not ({c['states'] for c in ad} & {c['states'] for c in ac}) and all(c['div_score'] != '' for c in k2)
        assert min(c['div_score'] for c in ad) >= max(c['div_score'] for c in ac)
        r = g1[g1.k == 2].iloc[0]; assert r.n_diverse == N_ARM and r.n_clustered == N_ARM and r.n_random >= 5
    finally: g.update(saved)

@_t('P4-05 a state without a profile for a fixed crop leaves the pool of ALL arms (matched pool); undefined diversity yields NO diverse/clustered runs, never a silent substitute')
def _():
    X, st, cr, sts = _p4_fixture(12, 150); keep = ~((st == 'S05') & (cr == 'soybeans')); cap = make_cap(st, cr); g = globals(); names = ['BUDGETS', 'K_LIST', 'MIN_STATE_ROWS', 'TARGETS_TO_RUN', 'SELECTION_ARMS', 'DIVERSITY_DIST', 'K_MAX']; saved = {n_: g.get(n_) for n_ in names}
    try:
        g.update(BUDGETS=[300], K_LIST=[2], MIN_STATE_ROWS=100, TARGETS_TO_RUN=['ARG'], K_MAX={'ARG': 8, 'USA': 8, 'CHN': 2, 'AUS': 2, 'FRA': 8}, SELECTION_ARMS=['random', 'diverse', 'clustered'])
        g['DIVERSITY_DIST'] = compute_diversity_distances(X[keep], st[keep], cr[keep], min_n=20); c1, g1 = build_plan(cap)
        assert all('S05' not in c['states'] for c in c1) and 'S05' in g1.iloc[0].dropped_no_profile
        g['DIVERSITY_DIST'] = None; c2, g2 = build_plan(cap); assert all(c['arms'] == ('random',) for c in c2) and g2.iloc[0].n_diverse == 0 and 'not computed' in g2.iloc[0].arm_note
    finally: g.update(saved)

@_t('P4-06 pool rule: strict is the default and unchanged; equal_share is opt-in and only enlarges the pool')
def _():
    assert POOL_RULE == 'strict'; X, st, cr, sts = _p4_fixture(12, 150); cap = make_cap(st, cr); g = globals(); saved = (g['POOL_RULE'], g['MIN_STATE_ROWS'])
    try:
        g['MIN_STATE_ROWS'] = 100; s_ = pool_for('ARG', 2000, 4, cap); g['POOL_RULE'] = 'equal_share'; e_ = pool_for('ARG', 2000, 4, cap); assert set(s_) <= set(e_)
    finally: g['POOL_RULE'], g['MIN_STATE_ROWS'] = saved

@_t('P4-07 statistics: t_interval_fpc handles N=inf (selected arms: no finite-population correction); arm_contrast / expand_arms / diversity_assoc run and are labelled')
def _():
    m, lo, hi, se = t_interval_fpc(np.array([.5, .6, .55, .7]), float('inf')); assert np.isfinite([m, lo, hi, se]).all() and lo < m < hi
    rows = []
    for i, (arms, st_, f) in enumerate([('random', 'A-B', .60), ('random|diverse', 'A-C', .62), ('random', 'B-C', .58), ('diverse', 'C-D', .70), ('diverse', 'D-E', .72), ('clustered', 'A-E', .50), ('clustered', 'B-E', .52), ('random', 'B-D', .61), ('random', 'A-D', .59)]):
        rows.append(dict(run_id=f'r{i}', status='ok', composition=BASE_MODE, arms=arms, target='ARG', budget=1000, k=2, seed=0, states=st_, pool_size=6, n_total_combos=15, key_n=100, target_crop_n=100, n_target_classes_train_ge30=3, primary_f1=f, diversity_score=0.1 * i, diversity_pct=i / 9))
    d = pd.DataFrame(rows); E = expand_arms(d); assert (E.arm == 'diverse').sum() == 3 and (E.arm == 'random').sum() == 5 and (E.arm == 'clustered').sum() == 2
    cs = cell_stats(E[E.arm == 'diverse']); assert 'selected arm' in cs.iloc[0].ci_note and not cs.iloc[0].exhaustive
    ct = arm_contrast(d, 'primary_f1', 'diverse', 'random'); assert len(ct) == 1 and abs(ct.iloc[0]['diff'] - (np.mean([.62, .70, .72]) - np.mean([.60, .62, .58, .61, .59]))) < 1e-9
    assoc = diversity_assoc(d, 'primary_f1'); assert len(assoc) == 1 and assoc.iloc[0].n == 9

@_t('P4-08 results rows carry arms, diversity_score and diversity_pct; Phase-4 settings are in the LOCKED manifest and the run environment')
def _():
    for c in ('arms', 'diversity_score', 'diversity_pct'): assert c in COLUMNS
    for k in ('SELECTION_ARMS', 'POOL_RULE', 'N_ARM', 'ENUM_LIMIT', 'SCORE_SAMPLE', 'MIN_PROFILE_N', 'DIVERSITY_SPACE'): assert k in LOCKED, k
    df0, cells = _pipe_df(); assert df0['arms'].astype(str).str.contains('diverse').any() and df0['diversity_pct'].notna().any()

import hashlib
def _e3_scenario_hash():
    r = np.random.default_rng(7); sts = ['A', 'B', 'C', 'D']; st, cr = [], []
    for s in sts:
        for c, n in [('soybeans', 400), ('corn', 500), ('wheat', 500), ('rice', 300)]: st += [s] * n; cr += [c] * n
    st = np.array(st); cr = np.array(cr); crid = np.array([CROP_ID[c] for c in cr]); pools = {(s, c): np.where((st == s) & (cr == c))[0] for s in sts for c in set(cr)}
    h = hashlib.sha256()
    for tg, B, ss in [('ARG', 1000, ('A', 'B')), ('CHN', 1000, ('A', 'B', 'C'))]:
        for seed in (0, 1):
            sp = draw_split(tg, B, ss, seed, pools, cr)
            for key in sorted(sp) if isinstance(sp, dict) else range(len(sp)):
                v = sp[key]
                if key == 'sel': continue                                                          # key added in v3.7 (value 'random'); the draw itself is what is frozen
                h.update(str(key).encode()); h.update(np.ascontiguousarray(v).tobytes() if isinstance(v, np.ndarray) else repr(v).encode())
    return h.hexdigest()[:20]

_E_FROZEN_HASH = '666787c93d8b407377f2'          # Level 2: sha256 prefix of draw_split output (ARG + CHN, fixed scenario) computed with the FROZEN LEVEL 1 code (v3.8). USA is not in it: the target name is part of the draw seed, so a USA draw can never equal the old FRA draw; USA is checked structurally in L2-01
_E_FROZEN_LOCKED = ['BUDGETS', 'K_LIST', 'SEEDS', 'AUG', 'MAX_COMBOS', 'MIN_COMBOS', 'MIN_KPOINTS', 'MIN_STATE_ROWS', 'VAL_FRACTION', 'KEY_SHARES', 'K_MAX', 'FRA_K8_MIN_POOL']

def _e_cap(rows):
    # rows: {state: {crop: n}}; unlisted crops 0
    cap = pd.DataFrame({s: {c: v.get(c, 0) for c in CROP_ORDER} for s, v in rows.items()}).T; cap['total'] = cap[CROP_ORDER].sum(axis=1); cap.index.name = 'state'; return cap

@_t('E1 water_fill: exact total, never above capacity, as equal as capacity allows, deterministic, infeasible requests STOP')
def _():
    r0 = np.random.default_rng(0)
    for _i in range(300):
        k = int(r0.integers(1, 6)); caps = [int(x) for x in r0.integers(0, 400, k)]; n = int(r0.integers(0, max(1, sum(caps)) + 1))
        a = water_fill(n, caps, np.random.default_rng(_i)); b = water_fill(n, caps, np.random.default_rng(_i))
        assert a == b and sum(a) == n and all(0 <= x <= c for x, c in zip(a, caps))
        assert all(not (a[i] > a[j] + 1 and a[j] < caps[j]) for i in range(k) for j in range(k)), (n, caps, a)
    assert water_fill(10, [100, 100], np.random.default_rng(0)) in ([5, 5],) and sorted(water_fill(11, [100, 100], np.random.default_rng(0))) == [5, 6]
    assert water_fill(100, [10, 500, 500], np.random.default_rng(0)) == [10, 45, 45]
    try: water_fill(101, [50, 50], np.random.default_rng(0)); raise AssertionError('should STOP')
    except InfeasibleSplit: pass

@_t('E2 extension draw_split: a short state gives all it has and the others make up the surplus; rows are disjoint and verify_split accepts them; the minimum contribution is enforced')
def _():
    with profile_scope('extension'):
        rows = {'A': {'rice': 400, 'corn': 3000}, 'B': {'rice': 60, 'corn': 3000}, 'C': {'rice': 600, 'corn': 400}}   # CHN B=1000: rice 200, rest 800
        st = np.concatenate([np.full(sum(v.values()), s) for s, v in rows.items()]); cr = np.concatenate([np.array([c for c, n in v.items() for _ in range(n)]) for v in rows.values()])
        pools = {(s, c): np.where((st == s) & (cr == c))[0] for s in rows for c in set(cr)}
        sp = draw_split('CHN', 1000, ('A', 'B'), 0, pools, cr); v = verify_split(sp, 'CHN', 1000, ('A', 'B'), st, cr); assert v['ok'], v['msgs']
        allp = np.concatenate([sp['train_pos'], sp['val_pos']]); assert len(allp) == 1000 and len(set(allp.tolist())) == 1000
        nr = {s: int(((st[allp] == s) & (cr[allp] == 'rice')).sum()) for s in 'AB'}; assert nr == {'A': 140, 'B': 60}, nr          # B holds only 60 rice -> gives 60, A gives 140
        # tampering: swap B's rice for A's rice (B then gives less than it can while A gives more) must be caught
        bi = [i for i in sp['train_pos'] if st[i] == 'B' and cr[i] == 'rice'][:20]; ai = [i for i in pools[('A', 'rice')] if i not in set(allp.tolist())][:20]
        bad = dict(sp); bad['train_pos'] = np.concatenate([[i for i in sp['train_pos'] if i not in set(bi)], ai]).astype(sp['train_pos'].dtype); assert not verify_split(bad, 'CHN', 1000, ('A', 'B'), st, cr)['ok'], 'not-as-equal-as-possible split accepted'

@_t('E3 MAIN IS UNTOUCHED: draw_split gives byte-identical output to the v3.5 code (frozen hash), the main LOCKED list is unchanged, run ids carry no profile tag')
def _():
    assert _e3_scenario_hash() == _E_FROZEN_HASH, 'ARG/CHN draws differ from the frozen Level 1 code'
    assert LOCKED[:len(_E_FROZEN_LOCKED)] == _E_FROZEN_LOCKED
    assert ALLOC_RULE == 'equal' and PROFILE == 'main'; rid = make_run_id(dict(target='ARG', budget=1000, k=2, states=('A', 'B')), 0); assert 'prof-' not in rid
    with profile_scope('extension'): assert 'prof-extension' in make_run_id(dict(target='CHN', budget=3000, k=2, states=('A', 'B')), 0)

@_t('E4 extension plan: cells exist where the strict pool has none; every combination passes the pooled check and the floor; k=1 census kept; random arm only; MAIN plan unchanged on the same table')
def _():
    rows = {s: {'rice': 700, 'corn': 2500} for s in ('S1', 'S2')}                     # S1, S2 can each supply CHN B=3000 alone (rice 600 + rest 2400)
    rows.update({s: {'rice': 400, 'corn': 1100} for s in ('S3', 'S4', 'S5', 'S6')})   # these cannot alone, but can pool
    cap = _e_cap(rows); g = globals(); saved = {n_: g[n_] for n_ in ['MIN_STATE_ROWS', 'K_LIST', 'MIN_KPOINTS', 'K_MAX']}
    try:
        g.update(MIN_STATE_ROWS=100, K_LIST=[1, 2, 4], MIN_KPOINTS=3, K_MAX={'ARG': 8, 'USA': 8, 'CHN': 4, 'AUS': 2, 'FRA': 8})   # fixture of the DORMANT Level 1 extension; CHN k<=4 here is test-only
        cm, gm = build_plan(cap); main_cells = [c for c in cm if c['target'] == 'CHN' and c['budget'] == 3000]
        with profile_scope('extension'):
            ce, ge = build_plan(cap)
            assert ce and all(c['target'] == 'CHN' and c['budget'] in (3000, 4000) or c['target'] == 'USA' for c in ce) and all(c['arms'] == ('random',) for c in ce)
            c3 = [c for c in ce if c['target'] == 'CHN' and c['budget'] == 3000]; assert {c['k'] for c in c3} == {1, 2, 4}
            for c in ce: assert combo_feasible_pooled(c['states'], c['target'], c['budget'], cap)
            for c in c3:
                if c['k'] == 1: assert c['states'][0] in ('S1', 'S2')
            assert len([c for c in c3 if c['k'] == 1]) == 2 and any(c['k'] == 4 for c in c3)
            assert all(float(c['max_share']) >= 1.0 / c['k'] - 1e-9 for c in c3)
        assert all(set(c['states']) <= {'S1', 'S2'} for c in main_cells), 'strict main plan must contain only states that can supply the budget alone (CHN is exempt from MIN_COMBOS in Level 2, so the pair S1+S2 may appear)'
        cm2, gm2 = build_plan(cap); assert [(c['k'], c['states']) for c in cm2] == [(c['k'], c['states']) for c in cm]          # leaving the extension scope restores the main plan exactly
    finally: g.update(saved)

@_t('E5 extension settings: EXT_* keys are locked only when the extension is selected; extension output folder is a sub-folder of the main folder; results columns exist')
def _():
    for c in ('profile', 'state_rows', 'max_state_share'): assert c in COLUMNS
    has = [k for k in EXT_LOCKED if k in LOCKED]; assert len(has) in (0, len(EXT_LOCKED))
    if has: assert os.path.basename(os.path.normpath(OUT_DIR)) == 'extension' and os.path.dirname(os.path.normpath(OUT_DIR)) == os.path.normpath(MAIN_OUT_DIR)
    else: assert PROFILE == 'main'
    assert EXT_MIN_CONTRIB > 0 and set(EXT_TARGETS) <= {'ARG', 'USA', 'CHN', 'AUS'} and AUG == 'off'


@_t('V37-01 supervisor-review settings: 200-epoch cap, new results folder name, main/secondary metric, FRA coverage rule, input-feature text, new keys locked')
def _():
    assert MAX_EPOCHS == 200 and 'level2' in EXPERIMENT_VERSION and REPORT_PRIMARY == 'main_f1' and REPORT_SECONDARY == 'paper_f1' and FRA_MAIN_RULE == 'coverage_conditional' and 'Sentinel-2' in INPUT_FEATURES
    for k in ('FRA_MAIN_RULE', 'INPUT_FEATURES', 'KC_TARGETS', 'KC_COMBOS', 'AUG_TARGETS', 'AUG_BUDGETS', 'AUG_COMBOS', 'AUG_REPEAT', 'MAX_EPOCHS'): assert k in LOCKED, k
    assert 'aug_on' in APPROVED and AUTO_MODE in (True, False)
    for c in ('aug', 'sample_sel', 'input_features', 'main_f1', 'main_metric', 'train_n_effective'): assert c in COLUMNS

@_t('V37-02 main metric: ARG/CHN = binary key-crop (primary); FRA = coverage-conditional; old results get main_f1 filled in, existing values are never changed')
def _():
    assert main_metric_value('ARG', .7, .8)[0] == .7 and main_metric_value('CHN', .9, .95)[0] == .9 and main_metric_value('USA', .5, .68) == (.68, 'coverage_conditional_macro_f1')
    old = pd.DataFrame(dict(status=['ok'] * 3, target=['ARG', 'USA', 'USA'], primary_f1=[.7, .5, .5], cov_cond_primary_f1=[.8, .68, .68], main_f1=[np.nan, np.nan, .123]))
    new = add_main_metric(old); assert list(new.main_f1.round(3)) == [.7, .68, .123], list(new.main_f1)
    g = globals(); saved = g['FRA_MAIN_RULE']
    try: g['FRA_MAIN_RULE'] = 'primary'; assert main_metric_value('USA', .5, .68)[0] == .5
    finally: g['FRA_MAIN_RULE'] = saved

@_t('V37-03 kcenter_pick: exactly n DISTINCT rows of the pool, deterministic, covers the cloud better than a random pick of the same size')
def _():
    rng = np.random.default_rng(0); E = rng.normal(0, 1, (3000, 6)).astype(np.float32); pool = np.arange(500, 2500); n = 60
    a = kcenter_pick(pool, n, ('t', 1, 'S', 0, 'c'), embed=E); b = kcenter_pick(pool, n, ('t', 1, 'S', 0, 'c'), embed=E); c = kcenter_pick(pool, n, ('t', 1, 'S', 1, 'c'), embed=E)
    assert len(a) == n and len(set(a.tolist())) == n and set(a.tolist()) <= set(pool.tolist()) and (a == b).all() and not (a == c).all()
    def radius(sel): return float(np.max(np.min(((E[pool][:, None, :] - E[sel][None]) ** 2).sum(-1), axis=1)))
    rr = np.mean([radius(np.random.default_rng(i).choice(pool, n, replace=False)) for i in range(5)]); assert radius(a) < rr, (radius(a), rr)       # smaller covering radius than random
    assert set(kcenter_pick(pool[:10], 10, ('x',), embed=E).tolist()) == set(pool[:10].tolist()) and len(kcenter_pick(pool, 0, ('x',), embed=E)) == 0

@_t('V37-04 draw_split sel=kcenter: same per-state / per-crop counts as the random draw, rows come from the same pools, verify_split accepts; random path unchanged')
def _():
    X, st, cr, sts = _p4_fixture(6, 400); pools = build_pools(st, cr); g = globals(); saved = g['KC_EMBED']
    try:
        g['KC_EMBED'] = build_kc_embedding(X, dims=6, fit_n=1500); crops = np.asarray(cr)
        for tg, B, ss in (('ARG', 300, ('S00', 'S02')), ('ARG', 300, ('S01',)), ('ARG', 300, ('S00', 'S03', 'S04', 'S05'))):
            a = draw_split(tg, B, ss, 0, pools, cr); b = draw_split(tg, B, ss, 0, pools, cr, sel='kcenter'); v = verify_split(b, tg, B, ss, st, cr); assert v['ok'], v['msgs']
            ia, ib = np.concatenate([a['train_pos'], a['val_pos']]), np.concatenate([b['train_pos'], b['val_pos']]); assert len(set(ib.tolist())) == B
            for s in ss:
                for c in set(crops): assert int(((st[ia] == s) & (crops[ia] == c)).sum()) == int(((st[ib] == s) & (crops[ib] == c)).sum())
            assert set(ib.tolist()) != set(ia.tolist()) or len(ss) == 0
        assert a['sel'] == 'random' and b['sel'] == 'kcenter'
    finally: g['KC_EMBED'] = saved

@_t('V37-05 plan: common-pool / k-center / aug_on cells (shared runs not duplicated, common combos inside the common pool, aug_on pending until approved, distinct run ids); base plan unchanged')
def _():
    X, st, cr, sts = _p4_fixture(12, 150); cap = make_cap(st, cr); g = globals()
    names = ['BUDGETS', 'K_LIST', 'MIN_STATE_ROWS', 'TARGETS_TO_RUN', 'SELECTION_ARMS', 'MIN_KPOINTS', 'MIN_COMBOS', 'DIVERSITY_DIST', 'K_MAX', 'AUG_BUDGETS', 'AUG_TARGETS', 'KC_TARGETS']; saved = {n_: g.get(n_) for n_ in names}; ap = dict(APPROVED); APPROVED['aug_on'] = False
    try:
        g.update(BUDGETS=[300, 600], K_LIST=[1, 2, 3], MIN_STATE_ROWS=100, TARGETS_TO_RUN=['ARG'], MIN_KPOINTS=3, MIN_COMBOS=5, K_MAX={'ARG': 8, 'USA': 8, 'CHN': 2, 'AUS': 2, 'FRA': 8}, AUG_BUDGETS=[300, 600], AUG_TARGETS=['ARG'], KC_TARGETS=['ARG'], DIVERSITY_DIST=None)
        g['SELECTION_ARMS'] = ['random']; c0, _g0 = build_plan(cap); g['SELECTION_ARMS'] = ['random', 'common', 'kcenter', 'aug_on']; c1, _g1 = build_plan(cap)
        base = [c for c in c1 if c.get('sel', 'random') == 'random' and c.get('aug', 'off') == 'off' and 'random' in c['arms']]; assert {(c['k'], c['budget'], c['states']) for c in base} == {(c['k'], c['budget'], c['states']) for c in c0}
        ids = [make_run_id(c, 0) for c in c1]; assert len(ids) == len(set(ids)), 'duplicate run ids'
        for B in (300, 600):
            commonpool = set.intersection(*[set(pool_for('ARG', B2, 2, cap)) for B2 in (300, 600)])
            for c in c1:
                if 'common' in c['arms'] and c['budget'] == B: assert set(c['states']) <= commonpool and combo_ok(c['states'], 'ARG', B, cap)
        shared = [c for c in c1 if 'common' in c['arms'] and 'random' in c['arms']]; only = [c for c in c1 if c['arms'] == ('common',)]; assert len({_cell_key(c) for c in c1}) == len(c1)
        kc = [c for c in c1 if c.get('sel') == 'kcenter']; assert kc and all(c['arms'] == ('kcenter',) for c in kc) and all(sum(1 for x in kc if (x['budget'], x['k']) == (c['budget'], c['k'])) <= (KC_COMBOS if c['k'] > 1 else 99) for c in kc)
        au = [c for c in c1 if c.get('aug') == 'on']; assert au and all(c['pending'] for c in au)
        APPROVED['aug_on'] = True; c2, _g2 = build_plan(cap); assert not any(c['pending'] for c in c2 if c.get('aug') == 'on' and c['target'] != 'AUS')
        assert 'sel-kcenter' in make_run_id(kc[0], 0) and 'aug-on' in make_run_id(au[0], 0) and 'sel-' not in make_run_id(base[0], 0) and 'aug-off' in make_run_id(base[0], 0)
    finally: g.update(saved); APPROVED.clear(); APPROVED.update(ap)

@_t('V37-06 paired contrast: kcenter - random on the SAME combinations; unpaired combinations are ignored; units overlap -> labelled descriptive')
def _():
    rows = []
    for i, s in enumerate(['A-B', 'A-C', 'B-C', 'C-D', 'D-E']):
        for arm, f in (('random', .60 + .01 * i), ('kcenter', .58 + .01 * i)):
            for sd in (0, 1): rows.append(dict(run_id=f'{arm}{i}{sd}', status='ok', composition=BASE_MODE, arms=arm, target='ARG', budget=1000, k=2, seed=sd, states=s, pool_size=6, n_total_combos=15, key_n=100, target_crop_n=100, n_target_classes_train_ge30=3, diversity_score=np.nan, diversity_pct=np.nan, main_f1=f + .001 * sd))
    rows.append(dict(rows[0], run_id='extra', states='X-Y', arms='kcenter')); d = pd.DataFrame(rows)
    ct = arm_paired_contrast(d, 'main_f1', 'kcenter', 'random'); assert len(ct) == 1 and ct.iloc[0].n_pairs == 5 and abs(ct.iloc[0]['diff'] + .02) < 1e-9 and 'paired' in ct.iloc[0].contrast

def _v38_cells():
    X, st, cr, sts = _p4_fixture(12, 150); cap = make_cap(st, cr); g = globals()
    names = ['BUDGETS', 'K_LIST', 'MIN_STATE_ROWS', 'TARGETS_TO_RUN', 'SELECTION_ARMS', 'MIN_KPOINTS', 'MIN_COMBOS', 'DIVERSITY_DIST', 'K_MAX', 'AUG_BUDGETS', 'AUG_TARGETS', 'KC_TARGETS', 'AUG_CONTROL', 'KC_SEEDS', 'SEEDS', 'PLAN_CELLS_RUN']; saved = {n_: g.get(n_) for n_ in names}; ap = dict(APPROVED); APPROVED['aug_on'] = True
    g.update(BUDGETS=[300, 600], K_LIST=[1, 2, 3], MIN_STATE_ROWS=100, TARGETS_TO_RUN=['ARG'], MIN_KPOINTS=3, MIN_COMBOS=5, K_MAX={'ARG': 8, 'USA': 8, 'CHN': 2, 'AUS': 2, 'FRA': 8}, AUG_BUDGETS=[300, 600], AUG_TARGETS=['ARG'], KC_TARGETS=['ARG'], DIVERSITY_DIST=None, SELECTION_ARMS=['random', 'common', 'kcenter', 'aug_on'])
    return g, saved, ap, cap

@_t('V38-01 plan: aug_copy cells mirror aug_on (same states / budgets / k, aug="copy", distinct run ids, own arm); AUG_CONTROL=False removes them; plan_for_seed drops k-center only for seeds outside KC_SEEDS; the v3.7 run ids are unchanged')
def _():
    g, saved, ap, cap = _v38_cells()
    try:
        g['AUG_CONTROL'] = False; c_off, _x = build_plan(cap); g['AUG_CONTROL'] = True; c_on, _x = build_plan(cap)
        assert not [c for c in c_off if c.get('aug') == 'copy'] and [c for c in c_on if c.get('aug') == 'copy']
        au = [c for c in c_on if c.get('aug') == 'on']; cp = [c for c in c_on if c.get('aug') == 'copy']
        assert sorted((c['target'], c['budget'], c['k'], c['states']) for c in au) == sorted((c['target'], c['budget'], c['k'], c['states']) for c in cp) and all(c['arms'] == ('aug_copy',) for c in cp)
        ids = [make_run_id(c, 0) for c in c_on]; assert len(ids) == len(set(ids)) and all('aug-copy' in make_run_id(c, 0) for c in cp)
        assert [make_run_id(c, s) for c in c_off for s in (0, 1)] == [make_run_id(c, s) for c in c_on if c.get('aug') != 'copy' for s in (0, 1)], 'v3.7 run ids must not change'       # existing results.csv stays valid
        g['KC_SEEDS'] = [0, 1]; kc = [c for c in c_on if c.get('sel') == 'kcenter']; assert kc
        for sd, has in ((0, True), (1, True), (2, False), (3, False), (4, False)):
            pl = plan_for_seed(sd, c_on); assert (any(c.get('sel') == 'kcenter' for c in pl)) == has, sd; assert len(pl) == len(c_on) - (0 if has else len(kc))
            assert len([c for c in pl if c.get('aug') == 'copy']) == len(cp)
    finally: g.update(saved); APPROVED.clear(); APPROVED.update(ap)

def _v38_row(cell, seed, f1=0.7, **kw):
    r = {c: '' for c in COLUMNS}; r.update(run_id=make_run_id(cell, seed), status='ok', seed=seed, target=cell['target'], budget=cell['budget'], k=cell['k'], states='-'.join(cell['states']), arms='+'.join(cell['arms']), aug=cell.get('aug', 'off'), sample_sel=cell.get('sel', 'random'), main_f1=f1, primary_f1=f1, epochs_run=20, composition=cell['composition']); r.update(kw); return r

@_t('V38-02 plan_consistency_check: results of the current plan pass; a run id that is not in the plan (design drift) is reported as an orphan and blocks; k-center rows of seeds 0-1 stay valid')
def _():
    g, saved, ap, cap = _v38_cells()
    try:
        g['AUG_CONTROL'] = True; g['KC_SEEDS'] = [0, 1]; g['SEEDS'] = [0, 1, 2]; cells, _x = build_plan(cap); g['PLAN_CELLS_RUN'] = cells
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'results.csv'); assert plan_consistency_check(p)['ok']
            for c in cells[:5] + [c for c in cells if c.get('sel') == 'kcenter'][:2]:
                for sd in (0, 1): append_row(p, _v38_row(c, sd))
            r = plan_consistency_check(p); assert r['ok'] and r['existing'] == 14 and not r['orphans'], r
            kc = [c for c in cells if c.get('sel') == 'kcenter'][0]; append_row(p, _v38_row(kc, 2)); r = plan_consistency_check(p); assert not r['ok'] and len(r['orphans']) == 1 and 'sel-kcenter' in r['orphans'][0], r        # k-center at seed 2 is NOT in the plan
            g['KC_SEEDS'] = [0, 1, 2]; assert plan_consistency_check(p)['ok']
            g['KC_SEEDS'] = [0, 1]; d = dict(cells[0], budget=cells[0]['budget'] + 1); append_row(p, _v38_row(d, 0)); assert not plan_consistency_check(p)['ok']
    finally: g.update(saved); APPROVED.clear(); APPROVED.update(ap)

@_t('V38-03 reml_random_intercept recovers a simulated fixed effect and the variance components (own implementation, no statsmodels); equals ordinary least squares when the group variance is 0; mixed_model_report runs on a random-arm table')
def _():
    rng = np.random.default_rng(1); G, m = 60, 6; grp = np.repeat(np.arange(G), m); x = rng.normal(0, 1, G * m); u = rng.normal(0, 2.0, G)[grp]; y = 3.0 + 1.5 * x + u + rng.normal(0, 1.0, G * m)
    X = np.column_stack([np.ones(G * m), x]); r = reml_random_intercept(y, X, grp); se = np.sqrt(np.diag(r['cov']))
    assert abs(r['beta'][1] - 1.5) < 4 * se[1] and abs(np.sqrt(r['s2u']) - 2.0) < 0.6 and abs(np.sqrt(r['s2e']) - 1.0) < 0.2, (r['beta'], r['s2u'], r['s2e'])
    # exact cross-check: GLS with the estimated variances, built the slow way (dense covariance)
    V = np.zeros((G * m, G * m))
    for gi in range(G): ix = np.where(grp == gi)[0]; V[np.ix_(ix, ix)] = r['s2u']
    V += r['s2e'] * np.eye(G * m); Vi = np.linalg.inv(V); beta = np.linalg.solve(X.T @ Vi @ X, X.T @ Vi @ y); assert np.allclose(beta, r['beta'], atol=1e-6) and np.allclose(r['cov'], np.linalg.inv(X.T @ Vi @ X), atol=1e-8)
    y0 = 1.0 + 2.0 * x + rng.normal(0, 1.0, G * m); r0 = reml_random_intercept(y0, X, grp); ols = np.linalg.lstsq(X, y0, rcond=None)[0]; assert r0['s2u'] < 0.05 and np.allclose(r0['beta'], ols, atol=0.05)
    rows = []
    for ti, tg in enumerate(['ARG', 'CHN']):
        for ci in range(12):
            for k in (1, 2, 4):
                for sd in (0, 1):
                    for B in (1000, 2000):
                        rows.append(dict(run_id=f'{tg}{ci}{k}{sd}{B}', status='ok', arms='random', target=tg, budget=B, k=k, seed=sd, states=f'S{ci}-{k}', aug='off', sample_sel='random', main_f1=(60 + 3 * np.log2(k) + 2 * B / 1000 + (ci % 3)) / 100 + rng.normal(0, .01)))
    mm = mixed_model_report(pd.DataFrame(rows)); b = mm[(mm.term == 'log2(k)')].set_index('target'); assert set(b.index) == {'ARG', 'CHN'} and (abs(b.coef - 3.0) < 1.0).all() and (b.p < 0.05).all(), b

@_t('V38-04 single_state_report: average gain, worst-case gain and "beats the best single state" computed by hand on a small table')
def _():
    rows = []
    def add(states, k, f, sd=0): rows.append(dict(run_id=f'{states}{k}{sd}{f}', status='ok', arms='random', target='ARG', budget=1000, k=k, seed=sd, states=states, aug='off', sample_sel='random', main_f1=f))
    for s_, f in (('A', .50), ('B', .60), ('C', .70)): add(s_, 1, f); add(s_, 1, f + .02, sd=1)                      # seed-averaged singles: 0.51 0.61 0.71
    for s_, f in (('A-B', .65), ('A-C', .80), ('B-C', .55), ('A-D', .72)): add(s_, 2, f)
    r = single_state_report(pd.DataFrame(rows), n_boot=200).iloc[0]
    assert r.n_single == 3 and r.n_combos == 4 and abs(r.single_mean - 61) < 1e-9 and abs(r.single_best - 71) < 1e-9 and r.single_best_state == 'C' and abs(r.combo_mean - 68) < 1e-9
    assert abs(r.avg_gain - 7) < 1e-9 and abs(r.worst_case_gain - (55 - 61)) < 1e-9 and abs(r.share_beating_best_single - .5) < 1e-9 and abs(r.worst_vs_best_single - (55 - 71)) < 1e-9 and r.avg_gain_ci_lo < 7 < r.avg_gain_ci_hi

@_t('V38-05 replication_table: same sign in both seed groups and a confirmatory interval that excludes 0 = PASS; a reversed or null confirmatory group = NOT REPLICATED (never dropped)')
def _():
    rng = np.random.default_rng(2)
    def table(slope_conf):
        rows = []
        for sd in range(5):
            sl = 3.0 if sd < 2 else slope_conf
            for ci in range(14):
                for k in (1, 2, 4, 8):
                    for B in (1000, 2000): rows.append(dict(run_id=f'{sd}{ci}{k}{B}', status='ok', composition=BASE_MODE, pool_size=16, n_total_combos=(16 if k == 1 else 10), key_n=100, target_crop_n=100, n_target_classes_train_ge30=3, diversity_score=np.nan, diversity_pct=np.nan, arms='random', target='ARG', budget=B, k=k, seed=sd, states=f'S{ci}-{k}', aug='off', sample_sel='random', main_f1=(60 + sl * np.log2(k) + ci % 4) / 100 + rng.normal(0, .01)))
        return pd.DataFrame(rows)
    t = replication_table(table(3.0)); c1 = t[t.claim.str.startswith('C1')]; assert set(c1.verdict) == {'PASS'}, c1
    t = replication_table(table(-3.0)); c1 = t[t.claim.str.startswith('C1')]; assert set(c1.verdict) == {'NOT REPLICATED (sign differs)'}, c1
    t = replication_table(table(0.0)); c1 = t[t.claim.str.startswith('C1')]; assert c1.verdict.iloc[0].startswith('NOT REPLICATED'), c1.verdict.tolist()

@_t('V38-06 write_run_environment never overwrites an existing seed file (a resumed run writes a separate, time-stamped file)')
def _():
    with tempfile.TemporaryDirectory() as td:
        write_run_environment(td, 0); p = os.path.join(td, 'run_environment_seed0.json'); assert os.path.exists(p); a = open(p).read(); write_run_environment(td, 0)
        assert open(p).read() == a and len([f for f in os.listdir(td) if f.startswith('run_environment_seed0__resume_')]) == 1

_new = TEST_REPORT[_n0:]
for name, res, msg in _new: print(f'  [{res}] {name}' + (f'\n         -> {msg}' if msg else ''))
N_FAIL_A2 = sum(1 for r in _new if r[1] != 'PASS'); TESTS_A_PASSED = bool(TESTS_A_PASSED and N_FAIL_A2 == 0)
print('\nTEST GATE PART A2 (data functions):', 'ALL PASSED' if N_FAIL_A2 == 0 else f'{N_FAIL_A2} FAILED -- training is BLOCKED')

# ======================================================================================
# LEVEL 2 tests (added): France-source data, instructor gate, USA target, CHN descriptive, Level-1 setup equality
# ======================================================================================
@_t('L2-01 USA target: draw_split / verify_split accept a USA split (3 key crops 15% each + non-key remainder); a tampered split is caught')
def _():
    rows = {s: {'corn': 900, 'soybeans': 900, 'wheat': 900, 'other': 1500} for s in ('R1', 'R2', 'R3')}
    st = np.concatenate([np.full(sum(v.values()), s) for s, v in rows.items()]); cr = np.concatenate([np.array([c for c, n in v.items() for _ in range(n)]) for v in rows.values()])
    pools = {(s, c): np.where((st == s) & (cr == c))[0] for s in rows for c in set(cr)}
    sp = draw_split('USA', 2000, ('R1', 'R2'), 3, pools, cr); v = verify_split(sp, 'USA', 2000, ('R1', 'R2'), st, cr); assert v['ok'], v['msgs']
    allp = np.concatenate([sp['train_pos'], sp['val_pos']]); assert len(allp) == 2000 and len(set(allp.tolist())) == 2000
    for c in ('corn', 'soybeans', 'wheat'): assert int((cr[allp] == c).sum()) == int(0.15 * 2000), c
    bad = dict(sp); bad['train_pos'] = sp['train_pos'][:-1]; assert not verify_split(bad, 'USA', 2000, ('R1', 'R2'), st, cr)['ok']

@_t('L2-02 CHN is descriptive only: k>=4 infeasible, k<=2 feasible with the exemption; CHN is not a claim target; claim targets are exactly ARG and USA')
def _():
    assert 'CHN' in DESCRIPTIVE_TARGETS and 'CHN' in EXEMPT_RULES and sorted(CLAIM_TARGETS) == ['ARG', 'USA'] and 'CHN' not in CLAIM_TARGETS
    assert K_MAX['CHN'] == 2 and SOURCE == 'FRA' and len(SOURCE_UNITS) == 13 and SOURCE not in TARGETS_TO_RUN

@_t('L2-03 Level-1 training setup is unchanged: LEVEL1_TRAINING_SETUP equals the live configuration (lr, batch, epochs, patience, head, val fraction, seeds, budgets, k list)')
def _():
    g = globals(); assert 'LEVEL1_TRAINING_SETUP' in g
    for k, v in LEVEL1_TRAINING_SETUP.items():
        assert k in g, k
        assert g[k] == v, f'{k}: live {g[k]!r} != Level 1 {v!r}'

@_t('L2-04 instructor gate: refuses when not approved, when no hash is pasted, and when the hash differs; accepts only the exact current hash; hash changes when a locked setting changes OR the planned runs change')
def _():
    g = globals(); saved = (g['INSTRUCTOR_APPROVED'], g['APPROVED_DESIGN_SHA256'], g.get('PLAN_CELLS_RUN'))
    def refused():
        try:
            with contextlib.redirect_stdout(io.StringIO()): instructor_gate()
        except RuntimeError as e: return 'STOP' in str(e)
        return False
    try:
        g['PLAN_CELLS_RUN'] = [dict(target='ARG', budget=1000, k=2, states=('11', '24'), composition=BASE_MODE, sel='random', arms=('random',), aug='off', pending=False, combo=0)]
        h = design_hash(); assert len(h) == 64
        for ok_, hh in ((False, ''), (True, ''), (True, 'f' * 64), (False, h)):
            g['INSTRUCTOR_APPROVED'], g['APPROVED_DESIGN_SHA256'] = ok_, hh; assert refused(), (ok_, hh[:6])
        g['INSTRUCTOR_APPROVED'], g['APPROVED_DESIGN_SHA256'] = True, h
        with contextlib.redirect_stdout(io.StringIO()): assert instructor_gate() == h
        old = g['MAX_EPOCHS']
        try: g['MAX_EPOCHS'] = old + 1; assert design_hash() != h, 'hash ignores a locked setting'; assert refused()
        finally: g['MAX_EPOCHS'] = old
        g['PLAN_CELLS_RUN'] = [dict(g['PLAN_CELLS_RUN'][0], states=('11', '27'))]; assert design_hash() != h, 'hash ignores a change of the planned runs'; assert refused()
        g['PLAN_CELLS_RUN'] = [dict(target='ARG', budget=1000, k=2, states=('11', '24'), composition=BASE_MODE, sel='random', arms=('random',), aug='off', pending=False, combo=0)]
        assert design_hash() == h, 'hash not reproducible'
    finally: g['INSTRUCTOR_APPROVED'], g['APPROVED_DESIGN_SHA256'] = saved[0], saved[1]; (g.__setitem__('PLAN_CELLS_RUN', saved[2]) if saved[2] is not None else g.pop('PLAN_CELLS_RUN', None))

@_t('L2-05 France assignment (synthetic 64-block table): rows inside blocks get the block region, rows between blocks are excluded, labels are checked row by row; a swapped block order or a wrong label STOPs')
def _():
    g = globals(); keys = ['FRA_BLOCKS', 'FRA_ASSIGNED_PER_REGION', 'FRA_ROW_MAP_ROWS']; saved = {k: g[k] for k in keys}
    try:
        L = 3; gap_after = {5, 20}; nb = []; pos = 0
        for n_, (t_, r_, _lo, _hi) in enumerate(saved['FRA_BLOCKS']):
            nb.append((t_, r_, pos, pos + L - 1)); pos += L + (2 if n_ in gap_after else 0)
        N = pos; per = {}
        for t_, r_, lo, hi in nb: per[r_] = per.get(r_, 0) + (hi - lo + 1)
        g['FRA_BLOCKS'] = nb; g['FRA_ASSIGNED_PER_REGION'] = dict(sorted(per.items())); g['FRA_ROW_MAP_ROWS'] = N
        lab = np.full(N, CROP_ID['other'])
        for t_, r_, lo, hi in nb: lab[lo:hi + 1] = CROP_ID[RAW_TO_CROP[t_]]
        with contextlib.redirect_stdout(io.StringIO()): meta_, rep_ = build_fra_meta(lab)
        assert len(meta_) == N and int(meta_['assigned'].sum()) == 64 * L and rep_['excluded_outside_blocks'] == 4 and set(meta_.loc[meta_['assigned'], 'state']) == set(SOURCE_UNITS)
        assert (meta_.loc[~meta_['assigned'], 'state'] == 'NA').all()
        bad = lab.copy(); bad[nb[7][2]] = (bad[nb[7][2]] + 1) % 7
        try:
            with contextlib.redirect_stdout(io.StringIO()): build_fra_meta(bad)
            raise AssertionError('wrong label accepted')
        except RuntimeError as e: assert 'label' in str(e)
        except AssertionError as e:
            if 'wrong label accepted' in str(e): raise
            assert 'label' in str(e)
        g['FRA_BLOCKS'] = [nb[1], nb[0]] + nb[2:]
        try:
            with contextlib.redirect_stdout(io.StringIO()): build_fra_meta(lab)
            raise RuntimeError('swapped order accepted')
        except (RuntimeError, AssertionError) as e: assert 'accepted' not in str(e)
    finally: g.update(saved)

@_t('L2-06 USA clean score (L2-D11): same four classes in every run; TRUE rice/sugarcane/cotton removed; their predictions -> other; hand-computed; independent of the training mix; both scorers agree; v1 kept for provenance')
def _():
    I = CROP_ID; yt = np.array([I['corn']] * 4 + [I['soybeans']] * 4 + [I['wheat']] * 4 + [I['other']] * 4 + [I['rice']] * 2 + [I['cotton']])
    yp = np.array([I['corn']] * 4 + [I['soybeans']] * 3 + [I['sugarcane']] + [I['wheat']] * 4 + [I['other']] * 4 + [I['rice']] * 2 + [I['other']])          # one soybean predicted as sugarcane -> counts as 'other'
    r = fixed4_scores(yt, yp)
    # 16 samples scored (3 true rice/cotton dropped): corn F1 1, wheat F1 1, soybeans tp3 fn1 fp0 -> 6/7, other tp4 fp1 (the soybean) fn0 -> 8/9
    exp = (1 + 1 + 6 / 7 + 8 / 9) / 4; assert abs(r['f1'] - exp) < 1e-12 and abs(r['f1_numpy'] - exp) < 1e-12 and r['n_dropped'] == 3 and r['n_scored'] == 16 and r['n_pred_reassigned'] == 1, (r, exp)
    v1 = fixed4_v1_scores(yt, yp)['f1']; assert abs(v1 - exp) > 1e-3, 'clean and v1 must differ when true rice / cotton exist'
    sc = score_run('USA', yt, yp, np.array([0, 100, 100, 0, 0, 0, 100])); sc2 = score_run('USA', yt, yp, np.array([0, 100, 100, 50, 0, 0, 100]))
    assert abs(sc['fixed4_clean_f1'] - exp) < 1e-12 and sc['fixed4_clean_f1'] == sc2['fixed4_clean_f1'] and abs(sc['fixed4_f1'] - v1) < 1e-12
    assert sc['cov_cond_paper_f1'] != sc2['cov_cond_paper_f1'], 'the ORIGINAL rule depended on the training mix (the problem L2-D9 / D11 remove)'
    assert main_metric_value('USA', .5, .68, .6, .7) == (.7, 'fixed4_clean_macro_f1') and main_metric_value('ARG', .7, .8, .1, .2)[0] == .7
    try: fixed4_scores(np.array([I['corn']] * 3 + [I['rice']] * 3), np.array([I['corn']] * 6)); raise RuntimeError('no stop')
    except AssertionError as e: assert 'every class' in str(e)
    assert np.isnan(score_run('ARG', np.array([I['soybeans']] * 5 + [I['other']] * 5), np.array([I['soybeans']] * 5 + [I['other']] * 5), np.array([0, 50, 0, 0, 0, 0, 50]))['fixed4_clean_f1'])

@_t('L2-07 re-scoring ALL seeds from saved predictions: handles rows still on the original rule AND rows on L2-D9; reproduces their stored numbers first; keeps every earlier value and the original file; all-or-nothing; idempotent; the run cell refuses a mixed file')
def _():
    I = CROP_ID; rng = np.random.default_rng(3); cls = [I['corn'], I['soybeans'], I['wheat'], I['other'], I['rice'], I['cotton']]
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, 'runs')); rows = []
        for n_, kind in enumerate(['cov', 'v1', 'v1', 'cov']):
            yt = rng.choice(cls, 500); yp = np.where(rng.random(500) < .6, yt, rng.choice(cls + [I['sugarcane']], 500)); tc = np.array([0, 200, 200, 100, 0, 0, 100 if n_ % 2 else 0]); rid = f'USA|t{n_}'; present = np.unique(yt)
            np.savez_compressed(_npz_path(td, rid), y_true=yt.astype(np.uint8), y_pred=yp.astype(np.uint8), train_counts=tc)
            _p, _r, fs = _sk_prf(yt, yp, present); cov = [k for k, c_ in enumerate(present) if tc[c_] >= COVERAGE_MIN_TRAIN]
            row = dict(run_id=rid, status='ok', target='USA', states='11-24', seed=n_, sevenclass_f1=float(f1_score(yt, yp, average='macro', labels=present, zero_division=0)))
            if kind == 'cov': row.update(main_f1=float(np.mean(fs[cov])), main_metric='coverage_conditional_macro_f1')
            else: row.update(main_f1=fixed4_v1_scores(yt, yp)['f1'], fixed4_f1=fixed4_v1_scores(yt, yp)['f1'], main_f1_before_fixed4=.5, main_metric='fixed4_macro_f1')
            rows.append(row)
        rows.append(dict(run_id='ARG|x', status='ok', target='ARG', states='11', seed=0, main_f1=.5, main_metric='binary_key_crop_macro_f1', sevenclass_f1=.1))
        path = os.path.join(td, 'results.csv'); pd.DataFrame(rows).to_csv(path, index=False); orig = sha256_file(path)
        try: scoring_consistency_check(path); raise RuntimeError('mixed file accepted')
        except AssertionError as e: assert 'RE-SCORING' in str(e)
        with contextlib.redirect_stdout(io.StringIO()): rep_ = rescore_usa_fixed4(td)
        assert len(rep_) == 4 and sorted(rep_.old_rule.unique()) == ['coverage_conditional_macro_f1', 'fixed4_macro_f1'] and (rep_.n_true_dropped > 0).all()
        bak = [f for f in os.listdir(td) if f.startswith('results.csv.before_rescore_')]; assert len(bak) == 1 and sha256_file(os.path.join(td, bak[0])) == orig, 'the backup is not the original file'
        d2 = pd.read_csv(path); u = d2[d2.target == 'USA']; assert (u.main_metric == 'fixed4_clean_macro_f1').all() and np.allclose(u.main_f1, u.fixed4_clean_f1) and u.main_f1_before_clean.notna().all() and u.fixed4_f1.notna().all()
        assert u[u.run_id.isin(['USA|t0', 'USA|t3'])].main_f1_before_fixed4.notna().all() and (u[u.run_id.isin(['USA|t1', 'USA|t2'])].main_f1_before_fixed4 == .5).all(), 'an earlier value was lost or overwritten'
        assert d2[d2.target == 'ARG'].main_f1.iloc[0] == .5 and d2[d2.target == 'ARG'].main_metric.iloc[0] == 'binary_key_crop_macro_f1', 'a non-USA row was touched'
        assert scoring_consistency_check(path) is True
        with contextlib.redirect_stdout(io.StringIO()): assert len(rescore_usa_fixed4(td)) == 0                      # idempotent
        assert len(json.load(open(os.path.join(td, 'rescore_log.json')))) == 1
        d3 = pd.read_csv(os.path.join(td, bak[0])); d3.loc[1, 'main_f1'] += 0.01; p3 = os.path.join(td, 'r3'); os.makedirs(os.path.join(p3, 'runs')); [shutil.copy(os.path.join(td, 'runs', f), os.path.join(p3, 'runs', f)) for f in os.listdir(os.path.join(td, 'runs'))]
        d3.to_csv(os.path.join(p3, 'results.csv'), index=False); h3 = sha256_file(os.path.join(p3, 'results.csv'))
        try:
            with contextlib.redirect_stdout(io.StringIO()): rescore_usa_fixed4(p3)
            raise RuntimeError('mismatch accepted')
        except AssertionError as e: assert 'old main score' in str(e)
        assert sha256_file(os.path.join(p3, 'results.csv')) == h3 and not [f for f in os.listdir(p3) if f.startswith('results.csv.before')], 'a failed re-score changed something'

@_t('L2-08 the scoring rule is part of the design hash (changing the dropped classes or the rule changes the hash)')
def _():
    g = globals(); saved = (g.get('PLAN_CELLS_RUN'), list(g['USA_DROP_TRUE_CLASSES']))
    try:
        g['PLAN_CELLS_RUN'] = [dict(target='USA', budget=1000, k=1, states=('11',), composition=BASE_MODE, sel='random', arms=('random',), aug='off', pending=False, combo=0)]
        h = design_hash(); g['USA_DROP_TRUE_CLASSES'] = ['rice']; assert design_hash() != h
    finally: g['USA_DROP_TRUE_CLASSES'] = saved[1]; (g.__setitem__('PLAN_CELLS_RUN', saved[0]) if saved[0] is not None else g.pop('PLAN_CELLS_RUN', None))

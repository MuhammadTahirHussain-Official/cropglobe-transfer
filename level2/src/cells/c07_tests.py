# ======================================================================================
# PRE-TRAINING TEST GATE, PART A (spec section 8).  No GPU, no repo, no real data.  Training cells refuse to run unless Part A AND Part B pass.
# ======================================================================================
from sklearn.linear_model import LogisticRegression
TEST_REPORT = []
def _t(name):
    def deco(fn):
        try:
            with profile_scope('main'): fn()
            TEST_REPORT.append((name, 'PASS', ''))
        except Exception as e: TEST_REPORT.append((name, 'FAIL', f'{type(e).__name__}: {e}'))
        return fn
    return deco
def _near(a, b, tol=1e-9): assert abs(a - b) <= tol, f'{a} != {b}'
def _ids(*pairs): return np.concatenate([np.full(n, CROP_ID[c]) for c, n in pairs])
def _labels_from_table2(t): return np.concatenate([np.full(n, CROP_ID[c]) for c, n in TABLE2[t].items() if n > 0])

@_t('Table 2 label sets: ARG/CHN/AUS/FRA exactly as the paper')
def _():
    assert DEFINED_CLASSES['ARG'] == ['corn', 'soybeans', 'rice', 'sugarcane', 'cotton', 'other']
    assert DEFINED_CLASSES['CHN'] == ['corn', 'soybeans', 'rice', 'other'] and DEFINED_CLASSES['AUS'] == ['sugarcane', 'cotton', 'other']
    assert DEFINED_CLASSES['FRA'] == ['corn', 'soybeans', 'rice', 'wheat', 'other']

@_t('binary_collapse on the REAL label sets: other = every non-positive class (true OR predicted); nothing dropped, nothing -1')
def _():
    for t, pos in BINARY_TARGETS.items():
        yt = _labels_from_table2(t); rng = np.random.default_rng(0); yp = rng.integers(0, 7, len(yt))
        bt, bp = binary_collapse(yt, yp, pos)
        assert len(bt) == len(yt) and len(bp) == len(yp) and set(np.unique(bt)) <= {0, 1} and set(np.unique(bp)) <= {0, 1} and bt.min() >= 0 and bp.min() >= 0
        assert (bt == (yt == CROP_ID[pos])).all() and (bp == (yp == CROP_ID[pos])).all()
    # spec example "true corn predicted soybeans":  CHN (positive=rice) -> other/other (a correct 'other');  ARG (positive=soybeans) -> other/POSITIVE (a false positive)
    a, b = binary_collapse([CROP_ID['corn']], [CROP_ID['soybeans']], 'rice'); assert (a[0], b[0]) == (0, 0)
    a, b = binary_collapse([CROP_ID['corn']], [CROP_ID['soybeans']], 'soybeans'); assert (a[0], b[0]) == (0, 1)
    a, b = binary_collapse([CROP_ID['wheat'], CROP_ID['cotton']], [CROP_ID['wheat'], CROP_ID['rice']], 'rice'); assert list(a) == [0, 0] and list(b) == [0, 1]
    # a true crop outside {positive, other} is allowed (never required to be a subset)
    sc = score_run('CHN', _ids(('corn', 5), ('rice', 5), ('other', 5)), _ids(('corn', 5), ('rice', 5), ('other', 5)), np.array([50, 50, 50, 0, 0, 0, 50])); _near(sc['primary_f1'], 1.0)

@_t("all-'other' baseline reproduces 0.4375 (CHN, rice view) and 0.4496 (ARG, soybeans view) from the Table 2 counts")
def _():
    for t, exp in (('CHN', 0.4375), ('ARG', 0.4496)):
        yt = _labels_from_table2(t); sc = score_run(t, yt, np.full(len(yt), CROP_ID['other']), np.array([100] * 7))
        assert round(sc['baseline_all_other_primary'], 4) == exp, (t, sc['baseline_all_other_primary'])
        assert round(sc['primary_f1'], 4) == exp and sc['primary_view'] == f'binary_{BINARY_TARGETS[t]}'
        assert len(yt) == TABLE2_TOTAL[t]

def _authors_remap_reference(y_pred, target):
    # INDEPENDENT re-implementation of the spec text (lookup tables, no np.isin) -- compared with remap_existing_release
    out = []
    for y in np.asarray(y_pred).tolist():
        n = CROP_LABELS[y]
        if target == 'AUS': out.append(CROP_ID[n] if n in ('sugarcane', 'cotton') else CROP_ID['other'])
        elif target == 'CHN': out.append(CROP_ID[n] if n in ('corn', 'soybeans', 'rice') else CROP_ID['other'])
        elif target == 'ARG': out.append(CROP_ID['other'] if n == 'wheat' else CROP_ID[n])
        else: out.append(CROP_ID[n])
    return np.array(out)

@_t('existing-remap scorer == macro f1_score(y_true, remapped, labels=np.unique(y_true)) on random data, all targets (reference remap)')
def _():
    rng = np.random.default_rng(1)
    for t in ['ARG', 'CHN', 'AUS', 'FRA']:
        for _rep in range(5):
            yt = rng.choice([CROP_ID[c] for c in DEFINED_CLASSES[t]], 3000, p=None); yp = rng.integers(0, 7, 3000)
            assert (remap_existing_release(yp, t) == _authors_remap_reference(yp, t)).all()
            sc = score_run(t, yt, yp, np.array([100] * 7)); yr = _authors_remap_reference(yp, t)
            _near(sc['paper_f1'], f1_score(yt, yr, average='macro', labels=np.unique(yt)), 1e-12); _near(sc['paper_oa'], accuracy_score(yt, yr), 1e-12)

@_t('FRA: NO remap (wrong crop stays an error); ARG: only wheat->other; CHN keeps corn/soy/rice; AUS keeps sugarcane/cotton')
def _():
    allc = np.arange(7)
    assert (remap_existing_release(allc, 'FRA') == allc).all()
    assert list(remap_existing_release(allc, 'ARG')) == [0, 1, 2, 6, 4, 5, 6]
    assert list(remap_existing_release(allc, 'CHN')) == [0, 1, 2, 6, 6, 6, 6] and list(remap_existing_release(allc, 'AUS')) == [6, 6, 6, 6, 4, 5, 6]

@_t('two independent scorers (sklearn vs from-scratch numpy) agree to 1e-12 on random data (paper rule + binary view, per-class P/R/F1)')
def _():
    rng = np.random.default_rng(2)
    for t in ['ARG', 'CHN', 'AUS', 'FRA']:
        for _rep in range(8):
            n = int(rng.integers(50, 4000)); yt = rng.choice([CROP_ID[c] for c in DEFINED_CLASSES[t]], n); yp = rng.integers(0, 7, n)
            tc = rng.integers(0, 400, 7); sc = score_run(t, yt, yp, tc); assert sc['scorer_maxdiff'] <= SCORER_TOL, sc['scorer_maxdiff']

@_t('stop rule: fewer than 2 true classes raises TooFewClasses')
def _():
    try: score_run('ARG', _ids(('soybeans', 20)), _ids(('soybeans', 20)), np.ones(7)); raise RuntimeError('no stop')
    except TooFewClasses: pass

@_t('classes absent from training stay IN the score at their F1 (flagged); coverage-conditional macro-F1 is separate and labelled')
def _():
    yt = _ids(('corn', 50), ('soybeans', 50), ('rice', 50), ('sugarcane', 50), ('cotton', 50), ('other', 50)); yp = yt.copy()
    tc = np.array([100, 100, 100, 0, 0, 0, 100])                    # cotton / sugarcane absent from training
    yp[yt == CROP_ID['sugarcane']] = CROP_ID['other']
    sc = score_run('ARG', yt, yp, tc)
    assert 'sugarcane' in sc['absent_train_classes'] and 'cotton' in sc['absent_train_classes']
    assert sc['paper_f1_cls'][CROP_ID['sugarcane']] == 0.0 and not np.isnan(sc['paper_f1_cls'][CROP_ID['cotton']])      # still scored
    assert sc['cov_cond_paper_classes'] == ['corn', 'soybeans', 'rice', 'other'] and sc['cov_cond_paper_f1'] != sc['paper_f1']

@_t('no 7-class macro over crops the target lacks: per-class F1 is NaN for classes absent from y_true; macro only over np.unique(y_true)')
def _():
    yt = _ids(('rice', 30), ('other', 70)); sc = score_run('CHN', yt, yt.copy(), np.array([100] * 7))
    assert sc['present'] == ['rice', 'other'] and np.isnan(sc['paper_f1_cls'][CROP_ID['corn']]) and sc['paper_f1'] == 1.0

@_t('integrity vs skill: a skill failure (low val F1 / no skill / collapsed) is a WARNING; only integrity checks can fail the run')
def _():
    yt = _ids(('soybeans', 50), ('other', 50)); allo = np.full(100, CROP_ID['other']); sc = score_run('ARG', yt, allo, np.array([100] * 7))
    info = dict(val_f1=0.3, restore_delta=0.0, best_epoch=UNDERTRAINED_EPOCH + 5); I, ok = evaluate_integrity(sc, allo, dict(ok=True), info); fl = skill_flags(sc, info)
    assert ok and all(I.values()); assert any(f.startswith('WARN_VAL') for f in fl) and 'WARN_NOT_BETTER_THAN_BASELINE' in fl and 'WARN_COLLAPSED_PREDICTIONS' in fl and 'possibly_undertrained' in fl
    I2, ok2 = evaluate_integrity(sc, allo, dict(ok=False), info); assert not ok2 and not I2['I3_split_exact']
    I3, ok3 = evaluate_integrity(sc, allo, dict(ok=True), dict(val_f1=0.9, restore_delta=1e-3)); assert not ok3 and not I3['I4_restore_matches_best']
    I4, ok4 = evaluate_integrity(sc, allo, dict(ok=True), dict(val_f1=float('nan'), restore_delta=0.0)); assert not ok4 and not I4['I5_finite']
    bad = np.full(100, 9); 
    try: score_run('ARG', yt, bad, np.ones(7)); raise RuntimeError('no stop')
    except AssertionError: pass

@_t('Table 3 reference: only the three published values, no too-good flag, no PAPER_REF_F1')
def _():
    assert TABLE3_REF_MF1 == {'ARG': 75.63, 'AUS': 88.38, 'FRA': 90.00} and 'PAPER_REF_F1' not in globals()
    assert not any('TOO_GOOD' in f for f in skill_flags(score_run('ARG', _ids(('soybeans', 50), ('other', 50)), _ids(('soybeans', 50), ('other', 50)), np.ones(7) * 100), dict(val_f1=1.0, best_epoch=3)))

@_t('early stopping: strictly-greater improvement, patience 15, min 20 epochs, approved 200-epoch cap (v3.7; was 100), reasons logged')
def _():
    es = EarlyStopper(); stops = None
    for i in range(100):                                            # never improves after epoch 1
        _imp, stop = es.step(0.5)
        if stop: stops = es.epoch; break
    assert stops == 20 and es.reason == 'early_stop_patience' and es.best_epoch == 1          # min epochs 20 respected, bad count already >= 15
    es = EarlyStopper(); seq = [0.001 * i for i in range(1, MAX_EPOCHS + 1)]                # improves every epoch -> runs to the cap
    for s in seq:
        _imp, stop = es.step(s)
        if stop: break
    assert MAX_EPOCHS == 200 and es.epoch == MAX_EPOCHS and es.reason == 'max_epochs' and es.best_epoch == MAX_EPOCHS
    es = EarlyStopper()
    for s in [0.5] + [0.5] * 30:                                    # equal score is NOT an improvement (strictly greater)
        _imp, stop = es.step(s)
        if stop: break
    assert es.best_epoch == 1 and es.epoch == 20
    es = EarlyStopper()                                              # best at epoch 10 -> stops at 25
    for i in range(60):
        _imp, stop = es.step(0.9 if i == 9 else 0.5)
        if stop: break
    assert es.epoch == 25 and es.best_epoch == 10
    try: EarlyStopper(max_epochs=MAX_EPOCHS + 1); raise RuntimeError('no stop')
    except AssertionError: pass

@_t('class weights: absent crops get weight 0 (no division by zero) and match sklearn on the present classes')
def _():
    from sklearn.utils.class_weight import compute_class_weight
    y = np.array([1] * 30 + [6] * 10 + [2] * 60); w = balanced_class_weights(y, 7)
    assert w[0] == w[3] == w[4] == w[5] == 0 and np.isfinite(w).all() and np.allclose(w[[1, 2, 6]], compute_class_weight('balanced', classes=np.array([1, 2, 6]), y=y))

def _synthetic_usa(n_states=14, rows=1100, seed=0, floor=0.04):
    rng = np.random.default_rng(seed); st, cr = [], []
    for i in range(n_states):
        p = rng.dirichlet(np.ones(7) * 1.5); p = floor + p; p /= p.sum()
        for _ in range(rows): st.append('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]); cr.append(CROP_LABELS[int(rng.choice(7, p=p))])
    return np.array(st), np.array(cr)

@_t('design: key counts exact (ARG soy 20% / CHN rice 20% / USA 15% x3), identical at every k; pool = states that can supply the whole budget alone')
def _():
    for B in BUDGETS:
        assert key_counts('ARG', B) == {'soybeans': int(0.2 * B)} and key_counts('CHN', B) == {'rice': int(0.2 * B)}
        assert key_counts('USA', B) == {'corn': int(0.15 * B), 'soybeans': int(0.15 * B), 'wheat': int(0.15 * B)} and rest_count('USA', B) == B - 3 * int(0.15 * B)
    cap = pd.DataFrame({c: [900, 900, 900] for c in CROP_ORDER}, index=['X', 'Y', 'Z']); cap['total'] = cap[CROP_ORDER].sum(axis=1)
    cap.loc['X', 'soybeans'] = 100; cap.loc['Y', 'total'] = 1400                                    # X: too few soybeans for B=1000 (needs 200) ; Y: below the 1500-row rule
    assert eligible_states('ARG', 1000, cap) == ['Z'] and 'X' not in eligible_states('ARG', 1000, cap)

@_t('plan rules: k=1 census of the pool; k>=2 <=10 deterministic combos; <5 combos or <3 k-points -> infeasible; CHN k<=2 (Level 2: descriptive, exempt from the combination / k-point minimums); never shrinks a budget')
def _():
    st, cr = _synthetic_usa(14, 1800); cap = make_cap(st, cr); cells, grid = build_plan(cap)
    g = grid[(grid.target == 'ARG') & (grid.budget == 1000)].set_index('k'); P = int(g.pool_size.iloc[0]); pool = eligible_states('ARG', 1000, cap); assert P == len(pool)
    assert g.loc[1, 'n_combos'] == P and all(g.loc[k, 'n_combos'] <= MAX_COMBOS for k in g.index if k >= 2)
    assert not grid[(grid.target == 'CHN') & (grid.k >= 4)].feasible.any()
    c1 = make_combos(list('ABCDEFGHIJKL'), 4, 'ARG', 1000); assert len(c1) == len(set(c1)) == MAX_COMBOS and c1 == make_combos(list('ABCDEFGHIJKL'), 4, 'ARG', 1000)
    assert make_combos(list('ABCDEFGHIJKL'), 4, 'ARG', 1000) != make_combos(list('ABCDEFGHIJKL'), 4, 'ARG', 2000)
    small = st != 'Z'; s2, c2 = st[:], cr[:]                                                         # force a tiny pool -> everything infeasible
    cap_small = make_cap(*_synthetic_usa(4, 1800)); _c, gr = build_plan(cap_small)
    assert not gr[(gr.target == 'USA') & (gr.k == 8)].feasible.any()
    only = gr[(gr.target == 'ARG') & (gr.budget == 1000)]; assert (only.feasible.sum() == 0) or (only.feasible.sum() >= MIN_KPOINTS)
    assert all(c['pool_size'] >= 1 for c in cells) and set(c['budget'] for c in cells) <= set(BUDGETS)
    aus = [c for c in cells if c['target'] == 'AUS']; assert all(c['pending'] for c in aus) == (not AUS_SPEC_CONFIRMED)

@_t('sampling: exact budget, SAME key-crop count at k=1,2,4,8, 10% stratified validation, unique ids, no overlap, deterministic, seed-dependent')
def _():
    st, cr = _synthetic_usa(14, 1800); cap = make_cap(st, cr); pools = build_pools(st, cr)
    for t in ['ARG', 'CHN', 'USA']:
        for B in [1000, 4000]:
            pool = eligible_states(t, B, cap)
            if len(pool) < 8: continue
            ref = None
            for k in [1, 2, 4, 8]:
                states = tuple(pool[:k]); sp = draw_split(t, B, states, 0, pools, cr); v = verify_split(sp, t, B, states, st, cr); assert v['ok'], v['msgs']
                allp = np.concatenate([sp['train_pos'], sp['val_pos']]); kc = {c: int((cr[allp] == c).sum()) for c in key_counts(t, B)}
                assert len(sp['val_pos']) == B // 10 and len(allp) == B and len(np.unique(allp)) == B
                ref = ref or kc; assert kc == ref == key_counts(t, B), 'key-crop count changed with k'
    t, B = 'ARG', 1000; pool = eligible_states(t, B, cap); a = draw_split(t, B, tuple(pool[:2]), 0, pools, cr); b = draw_split(t, B, tuple(pool[:2]), 0, pools, cr); c = draw_split(t, B, tuple(pool[:2]), 1, pools, cr)
    assert (a['train_pos'] == b['train_pos']).all() and not np.array_equal(np.sort(a['train_pos']), np.sort(c['train_pos']))
    bad = dict(a); bad['val_pos'] = a['train_pos'][:100]; assert not verify_split(bad, t, B, tuple(pool[:2]), st, cr)['ok']            # overlap is caught
    bad2 = dict(a); bad2['train_pos'] = np.concatenate([a['train_pos'][:-1], a['train_pos'][:1]]); assert not verify_split(bad2, t, B, tuple(pool[:2]), st, cr)['ok']   # duplicate id is caught

@_t('statistics: FPC interval collapses when all combinations are evaluated; k=1 census has no CI; jackknife defined; regression reports the target-crop-count/budget aliasing')
def _():
    x = np.array([0.5, 0.6, 0.7, 0.55, 0.65]); m, lo, hi, se = t_interval_fpc(x, 5); assert se == 0 and lo == hi == m
    m2, lo2, hi2, se2 = t_interval_fpc(x, 100); assert lo2 < m2 < hi2 and se2 > 0
    rows = []
    for t, kc in (('ARG', 200), ('ARG', 400)):
        for k in (1, 2, 4):
            for ci in range(6):
                for s in range(2): rows.append(dict(run_id=f'{t}{kc}{k}{ci}{s}', status='ok', target=t, budget=kc * 5, k=k, states='-'.join(f'S{(ci + j) % 8}' for j in range(k)), seed=s, primary_f1=0.5 + 0.01 * k + 0.0001 * ci + 0.01 * s,
                                                  pool_size=8, n_total_combos=math.comb(8, k), key_n=kc, target_crop_n=kc, n_target_classes_train_ge30=3 + (1 if k > 1 else 0) + (1 if k > 2 else 0) + (ci % 2)))
    df = pd.DataFrame(rows); cs = cell_stats(df); assert (cs[cs.k == 1].ci_lo.isna()).all() and cs[cs.k == 1].census.all() and cs.jackknife_se.notna().all()
    reg = regression_report(df); assert 'aliased with budget' in ' '.join(reg['note'].astype(str)) and 'target_crop_count' not in set(reg.term) and 'n_target_classes_train_ge30' in set(reg.term) and {'k', 'budget'} <= set(reg.term)
    gn = gain_kmin_kmax(cs); assert len(gn) == 2 and (gn.gain > 0).all()

@_t('atomic write: no temp files left behind; content replaced as a whole')
def _():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'x.csv'); atomic_write(p, lambda f: f.write('a,b\n1,2\n'), mode='w'); atomic_write(p, lambda f: f.write('a,b\n3,4\n'), mode='w')
        assert open(p).read() == 'a,b\n3,4\n' and os.listdir(td) == ['x.csv']

def selftest_pipeline(return_report=False, return_df=False):
    with profile_scope('main'): return _selftest_pipeline_impl(return_report, return_df)

def _selftest_pipeline_impl(return_report=False, return_df=False):
    # END-TO-END dry run with a stand-in model (logistic regression): plan -> draw -> verify -> fit -> predict -> score (2 scorers) -> gates -> npz -> CSV (last) -> resume -> stats -> gate report.
    names = ['BUDGETS', 'SEEDS', 'K_LIST', 'MAX_COMBOS', 'MIN_STATE_ROWS', 'AUS_SPEC_CONFIRMED', 'AUS_STATES', 'AUS_KEY_SHARES_IF_CONFIRMED', 'FRA_K8_MIN_POOL', 'AUG_BUDGETS', 'SELECTION_ARMS', 'KC_TARGETS']
    g = globals(); saved_cfg = {n: g[n] for n in names}
    keys = ['KC_EMBED', 'DIVERSITY_DIST', 'FEATURES', 'SRC_UNIT', 'SRC_CROP', 'SRC_CROP_ID', 'SRC_NPZ_ROW', 'POOLS', 'EVAL_X', 'EVAL_Y', 'TARGET_IDS', 'FIT_FN', 'PREDICT_FN']; saved = {n: g.get(n, '<unset>') for n in keys}
    try:
        g.update(BUDGETS=[200, 400], SEEDS=[0, 1], K_LIST=[1, 2, 4, 8], MIN_STATE_ROWS=300, AUS_SPEC_CONFIRMED=True, AUS_STATES=['A', 'B'], AUS_KEY_SHARES_IF_CONFIRMED={'sugarcane': 0.2}, FRA_K8_MIN_POOL=10, AUG_BUDGETS=[200, 400], SELECTION_ARMS=['random', 'diverse', 'clustered', 'common', 'kcenter', 'aug_on'], KC_TARGETS=['ARG', 'CHN', 'USA'])      # the dormant Level 1 arms are switched ON here only so that their code stays tested
        st, cr = _synthetic_usa(14, 700); yid = np.array([CROP_ID[c] for c in cr]); rng = np.random.default_rng(0); mu = rng.normal(0, 1, (7, 730)).astype(np.float32)
        def feats(y): return (mu[y] + rng.normal(0, 1.5, (len(y), 730))).astype(np.float32).reshape(len(y), 1, 10, 73)
        g.update(FEATURES={SOURCE: feats(yid)}, SRC_UNIT=st, SRC_CROP=cr, SRC_CROP_ID=yid, SRC_NPZ_ROW=np.arange(len(st)) * 2, POOLS=build_pools(st, cr))
        g['DIVERSITY_DIST'] = compute_diversity_distances(g['FEATURES'][SOURCE], st, cr)
        EX, EY, TI = {}, {}, {}
        for t in TABLE2:
            if t == SOURCE: continue
            y = np.array([CROP_ID[c] for c in DEFINED_CLASSES[t] for _ in range(60)]); EY[t] = y; EX[t] = feats(y); TI[t] = np.arange(len(y))
        g.update(EVAL_X=EX, EVAL_Y=EY, TARGET_IDS=TI)
        def fake_fit(Xtr, ytr, Xva, yva, seed, ep=None, aug='off'):
            if aug in ('on', 'copy'): Xtr = np.concatenate([Xtr] * (1 + AUG_REPEAT)); ytr = np.concatenate([ytr] * (1 + AUG_REPEAT))             # stand-in for the 3x augmented training set
            m = LogisticRegression(max_iter=100, C=0.05).fit(Xtr.reshape(len(Xtr), -1), ytr); vf = float(f1_score(yva, m.predict(Xva.reshape(len(Xva), -1)), average='macro'))
            return dict(model=m, info=dict(val_f1=vf, best_epoch=3, epochs_run=23, stop_reason='early_stop_patience', restore_delta=0.0, history=[vf]))
        g.update(FIT_FN=fake_fit, PREDICT_FN=lambda m, Xe: m.predict(Xe.reshape(len(Xe), -1)))
        cap = make_cap(st, cr); cells, grid = build_plan(cap); assert len(cells) > 0 and {c['target'] for c in cells} >= {'ARG', 'CHN', 'USA'}
        g['KC_EMBED'] = build_kc_embedding(g['FEATURES'][SOURCE], dims=8, fit_n=3000)
        _variants = [c for c in cells if c.get('sel') == 'kcenter'][:4] + [c for c in cells if c.get('aug') == 'on'][:4]; assert len(_variants) == 8, 'plan must contain kcenter and aug_on cells'
        cells = [c for c in cells if (c['k'] <= 2 or c['target'] == 'USA') and c.get('sel', 'random') == 'random' and c.get('aug', 'off') == 'off'][:40] + _variants
        with tempfile.TemporaryDirectory() as td:
            csvp = os.path.join(td, 'results.csv'); buf = io.StringIO()
            with contextlib.redirect_stdout(buf): r1 = run_seed(0, cells, csvp, td); r1b = run_seed(1, cells, csvp, td)
            df = load_results(csvp); assert (df['sample_sel'] == 'kcenter').sum() == 8 and (df['aug'] == 'on').sum() == 8 and df['run_id'].is_unique and df['main_f1'].notna().all()
            assert (pd.to_numeric(df.loc[df['aug'] == 'on', 'train_n_effective']) == (1 + AUG_REPEAT) * pd.to_numeric(df.loc[df['aug'] == 'on', 'n_train'])).all() and (pd.to_numeric(df.loc[df['aug'] == 'on', 'steps_per_epoch']) == np.ceil((1 + AUG_REPEAT) * pd.to_numeric(df.loc[df['aug'] == 'on', 'n_train']) / BATCH_SIZE)).all()
            assert r1 == 'complete' and len(df) == 2 * len(cells) and (df['status'] == 'ok').all(), df['status'].value_counts().to_dict()
            assert len(os.listdir(os.path.join(td, 'runs'))) == 2 * len(cells) and not [f for f in os.listdir(td) if f.startswith('.tmp_')]
            z = np.load(os.path.join(td, 'runs', hashlib.md5(df.run_id.iloc[0].encode()).hexdigest()[:16] + '.npz'))
            assert {'y_true', 'y_pred', 'target_sample_ids', 'src_train_ids', 'cm7', 'cm_existing_release', 'cm_review_rule', 'paper_f1_per_class', 'train_counts'} <= set(z.files) and len(z['src_train_ids']) + len(z['src_val_ids']) == df.budget.iloc[0]
            assert (pd.to_numeric(df.scorer_maxdiff) <= SCORER_TOL).all()
            with contextlib.redirect_stdout(buf): run_seed(0, cells, csvp, td)
            assert len(load_results(csvp)) == 2 * len(cells), 'resume re-ran finished runs'
            kc = df[df.target == 'ARG'].groupby('budget')['key_n'].nunique().max(); assert kc == 1
            cs = cell_stats(df); assert len(cs) > 0; txt = gate_report(df, 0, cells); assert 'GATE REPORT' in txt and 'INTEGRITY' in txt
            for c_ in ['n_target_classes_train_ge30', 'n_absent_target_classes_train', 'steps_per_epoch', 'total_optimizer_steps']: assert df[c_].notna().all(), c_
            assert (pd.to_numeric(df.total_optimizer_steps) == pd.to_numeric(df.steps_per_epoch) * pd.to_numeric(df.epochs_run)).all()
            _tr = df.iloc[0]; _def = DEFINED_CLASSES[_tr.target]; assert int(_tr.n_target_classes_train_ge30) == sum(int(_tr[f'train_{c_}']) >= COVERAGE_MIN_TRAIN for c_ in _def) and int(_tr.n_absent_target_classes_train) == sum(int(_tr[f'train_{c_}']) == 0 for c_ in _def)
            if return_df: return df.copy(), list(cells)
            if return_report: return txt
            # skill failure never aborts:
            g['FIT_FN'] = lambda *a, **k: dict(model=None, info=dict(val_f1=0.1, best_epoch=47, epochs_run=50, stop_reason='max_epochs', restore_delta=0.0, history=[0.1]))
            g['PREDICT_FN'] = lambda m, Xe: np.full(len(Xe), CROP_ID['other'])
            csv2 = os.path.join(td, 'skill.csv')
            with contextlib.redirect_stdout(buf): run_seed(0, cells[:8], csv2, td)
            d2 = load_results(csv2); assert len(d2) == 8 and (d2.status == 'ok').all() and d2.skill_flags.str.contains('WARN_VAL').all()
            # 5 consecutive integrity failures abort:
            g['FIT_FN'] = lambda *a, **k: dict(model=None, info=dict(val_f1=0.9, best_epoch=3, epochs_run=23, stop_reason='x', restore_delta=0.5, history=[0.9]))
            csv3 = os.path.join(td, 'integ.csv')
            try:
                with contextlib.redirect_stdout(buf): run_seed(0, cells[:12], csv3, td)
                raise RuntimeError('did not abort')
            except RuntimeError as e: assert 'consecutive integrity' in str(e), e
            assert len(load_results(csv3)) == MAX_CONSEC_INTEGRITY
        return True
    finally:
        g.update(saved_cfg)
        for n, v in saved.items():
            if isinstance(v, str) and v == '<unset>': g.pop(n, None)
            else: g[n] = v

@_t('END-TO-END dry run with a stand-in model: plan, run, 2 scorers, gates, npz, CSV-last, resume, stats, gate report, skill warnings never abort, 5 integrity failures abort')
def _():
    assert selftest_pipeline()


# =================================================================== NEW TESTS for the dual-track update
@_t('three remap rules are separately named; existing rule PRESERVED unchanged (ARG wheat->other, FRA none); paper-text and review rules differ exactly where expected')
def _():
    allc = np.arange(7)
    assert list(REMAPS) == REMAP_RULES and 'remap_paper_rule' not in globals()
    ex = {t: list(remap_existing_release(allc, t)) for t in TABLE2_TOTAL if t != 'USA'}; pt = {t: list(remap_paper_text_v3(allc, t)) for t in ex}; rv = {t: list(remap_review_targetmap(allc, t)) for t in ex}
    assert ex['ARG'] == [0, 1, 2, 6, 4, 5, 6] and ex['FRA'] == list(allc) and ex['CHN'] == [0, 1, 2, 6, 6, 6, 6] and ex['AUS'] == [6, 6, 6, 6, 4, 5, 6]     # existing, unchanged
    assert pt['ARG'] == list(allc) and pt['FRA'] == list(allc) and pt['CHN'] == ex['CHN'] and pt['AUS'] == ex['AUS']                                      # paper text sec.4.4
    assert rv['ARG'] == ex['ARG'] and rv['CHN'] == ex['CHN'] and rv['AUS'] == ex['AUS'] and rv['FRA'] == [0, 1, 2, 3, 6, 6, 6]                            # target-map rule: only FRA differs from existing
    d = remap_difference_table().set_index('target'); assert not d.loc['FRA', 'existing_eq_review'] and d.loc['ARG', 'existing_eq_review'] and not d.loc['ARG', 'existing_eq_paper_text']

@_t('review track: FRA predicted sugarcane/cotton counts as Other (review) but stays an error (existing); both tracks reported, never merged')
def _():
    yt = _ids(('corn', 40), ('other', 40), ('wheat', 40)); yp = yt.copy(); yp[yt == CROP_ID['other']] = CROP_ID['sugarcane']                 # model calls every true Other 'sugarcane'
    sc = score_run('FRA', yt, yp, np.array([100] * 7))
    assert sc['paper_f1_cls'][CROP_ID['other']] == 0.0 and sc['review_f1_cls'][CROP_ID['other']] == 1.0 and sc['review_f1'] > sc['paper_f1']
    _near(sc['papertext_f1'], sc['paper_f1']); assert sc['scorer_maxdiff'] <= SCORER_TOL
    assert sc['sevenclass_f1'] == sc['paper_f1'] and sc['primary_f1'] == sc['paper_f1']                                                              # FRA: existing primary = existing paper rule

@_t('seven-class raw track: no remap, macro over classes present in truth only; differs from the paper rule when remapping changes predictions')
def _():
    yt = _ids(('corn', 30), ('rice', 30), ('other', 30)); yp = yt.copy(); yp[yt == CROP_ID['other']] = CROP_ID['cotton']          # CHN: cotton -> other under every remap, raw stays wrong
    sc = score_run('CHN', yt, yp, np.array([100] * 7)); assert sc['sevenclass_f1'] < 1.0 and sc['paper_f1'] == 1.0 and sc['review_f1'] == 1.0 and sc['papertext_f1'] == 1.0
    assert np.isnan(sc['paper_f1_cls'][CROP_ID['sugarcane']]) and sc['present'] == ['corn', 'rice', 'other']

@_t('binary truth with < 2 classes: STOP/flag (ARG truth without soybeans; CHN truth without rice), even though the 7-class truth has 2 classes')
def _():
    for t, pos in (('ARG', 'soybeans'), ('CHN', 'rice')):
        yt = _ids(('corn', 20), ('other', 20)); yp = yt.copy()
        try: score_run(t, yt, yp, np.array([100] * 7)); raise RuntimeError('no stop')
        except TooFewClasses as e: assert 'binary' in str(e)
    yt = _ids(('soybeans', 20)); 
    try: score_run('ARG', yt, yt, np.array([100] * 7)); raise RuntimeError('no stop')
    except TooFewClasses: pass

@_t('class weights: default = authors-equivalent (whole drawn sample, train+val); "train" option = training labels only; zero-count crops weight 0 in both')
def _():
    g = globals(); keep = g['CLASS_WEIGHT_SOURCE']
    try:
        assert keep == 'train_val'
        ytr, yva = np.array([1] * 30 + [2] * 10), np.array([3] * 5 + [1] * 5)
        w = training_class_weights(ytr, yva); assert w[3] > 0 and w[0] == 0 and np.isfinite(w).all()
        from sklearn.utils.class_weight import compute_class_weight
        ref = compute_class_weight('balanced', classes=np.unique(np.concatenate([ytr, yva])), y=np.concatenate([ytr, yva])); assert np.allclose(w[[1, 2, 3]], ref)
        g['CLASS_WEIGHT_SOURCE'] = 'train'; w2 = training_class_weights(ytr, yva); assert w2[3] == 0 and w2[0] == 0 and w2[2] > w2[1] > 0
        assert training_class_weights(np.array([1, 1, 2]), np.array([6]))[6] == 0
    finally: g['CLASS_WEIGHT_SOURCE'] = keep

@_t('eligibility table: EVERY state listed with need/have per fixed crop and an explicit reason for each exclusion; pool == eligible rows; combos verified for capacity')
def _():
    st, cr = _synthetic_usa(10, 1800); cap = make_cap(st, cr); cap.loc['A', 'soybeans'] = 5; cap.loc['B', 'total'] = 1000                      # A lacks soybeans, B below the row rule
    e = state_eligibility('ARG', 1000, cap); assert list(e.state) == list(cap.index) and len(e) == len(cap)
    assert not e.set_index('state').loc['A', 'eligible'] and 'soybeans' in e.set_index('state').loc['A', 'reason'] and '<1500' in e.set_index('state').loc['B', 'reason']
    assert (e.loc[~e.eligible, 'reason'] != '').all() and sorted(e.loc[e.eligible, 'state']) == eligible_states('ARG', 1000, cap)
    pool = eligible_states('ARG', 1000, cap)
    for k in (1, 2, 4): assert all(combo_capacity_ok(c, 'ARG', 1000, cap) for c in make_combos(pool, k, 'ARG', 1000))
    assert not combo_capacity_ok(('A', pool[0]), 'ARG', 1000, cap)                                                                        # a combination containing a short state is rejected

@_t('fixed_full composition (opt-in): proposals sum to B exactly; same counts at k=1,2,4,8; no approval -> pending/STOP; baseline untouched')
def _():
    g = globals(); keep = {n: g[n] for n in ('FIXED_COMPOSITION_CHOICE', 'COMPOSITION_MODES_TO_RUN')}; ap = dict(APPROVED)
    try:
        for ch in FIXED_COMPOSITION_PROPOSALS:
            for t in ('ARG', 'CHN', 'USA'):
                sh = FIXED_COMPOSITION_PROPOSALS[ch](t); _near(sum(sh.values()), 1.0, 1e-9); assert set(sh) <= set(DEFINED_CLASSES[t])        # only classes the target MAP defines
                for B in BUDGETS: assert sum(shares_to_counts(sh, B).values()) == B
        try: key_counts('ARG', 1000, FULL_MODE); raise RuntimeError('no stop')
        except AssertionError as e: assert 'not been chosen' in str(e)
        g['FIXED_COMPOSITION_CHOICE'] = 'A_key_plus_equal_rest'; g['COMPOSITION_MODES_TO_RUN'] = [BASE_MODE, FULL_MODE]
        assert key_counts('ARG', 1000, FULL_MODE)['soybeans'] == 200 and key_counts('ARG', 1000, BASE_MODE) == {'soybeans': 200}                  # baseline unchanged
        st, cr = _synthetic_usa(14, 1800, floor=0.4); cap = make_cap(st, cr); pools = build_pools(st, cr)
        cells, grid = build_plan(cap); full = [c for c in cells if c['composition'] == FULL_MODE]; assert full and all(c['pending'] for c in full)       # not approved -> excluded
        APPROVED['fixed_composition'] = True; cells, grid = build_plan(cap); full = [c for c in cells if c['composition'] == FULL_MODE]; assert full and not any(c['pending'] for c in full if c['target'] != 'AUS')
        pool = eligible_states('ARG', 1000, cap, FULL_MODE); ref = None
        for k in (1, 2, 4, 8):
            if len(pool) < k: continue
            states = tuple(pool[:k]); sp = draw_split('ARG', 1000, states, 0, pools, cr, FULL_MODE); v = verify_split(sp, 'ARG', 1000, states, st, cr, FULL_MODE); assert v['ok'], v['msgs']
            allp = np.concatenate([sp['train_pos'], sp['val_pos']]); cnt = {c: int((cr[allp] == c).sum()) for c in CROP_ORDER}; ref = ref or cnt; assert cnt == ref and sum(cnt.values()) == 1000
            assert all(cnt[c] == 0 for c in CROP_ORDER if c not in DEFINED_CLASSES['ARG'])                                                       # wheat never drawn: absent from training (flagged downstream)
        r_base = make_run_id(dict(target='ARG', budget=1000, k=1, states=('A',)), 0); r_full = make_run_id(dict(target='ARG', budget=1000, k=1, states=('A',), composition=FULL_MODE), 0)
        assert r_base == 'ARG|B1000|k1|A|aug-off|s0' and r_full != r_base and 'comp-fixed_full' in r_full
    finally:
        g.update(keep); APPROVED.clear(); APPROVED.update(ap)

@_t('baseline sampling unchanged by the update: identical draw for a fixed (target, budget, states, seed) and key counts')
def _():
    st, cr = _synthetic_usa(10, 1800); cap = make_cap(st, cr); pools = build_pools(st, cr); el = eligible_states('USA', 1000, cap); assert len(el) >= 2
    a = draw_split('USA', 1000, tuple(el[:2]), 3, pools, cr); b = draw_split('USA', 1000, tuple(el[:2]), 3, pools, cr, BASE_MODE)
    assert (a['train_pos'] == b['train_pos']).all() and a['key_counts'] == {'corn': 150, 'soybeans': 150, 'wheat': 150}

@_t('results reconciliation tool: reports duplicates, rows only in one file, conflicting metrics; refuses to call different files identical')
def _():
    A = pd.DataFrame({'run_id': list('abcd') + ['a'], 'primary_f1': [.5, .6, .7, .8, .5]}); B = pd.DataFrame({'run_id': list('bcdef'), 'primary_f1': [.6, .7, .81, .1, .2]})
    r = reconcile_results(A, B, names=('A', 'B')); assert r['duplicate_run_id_in_A'] == 1 and r['only_in_A'] == 1 and r['only_in_B'] == 2 and r['in_both'] == 3
    assert r['conflicting_rows_by_metric'] == {'primary_f1': 1} and r['verdict'].startswith('DIFFERENT'); assert reconcile_results(A.drop_duplicates('run_id'), A.drop_duplicates('run_id'))['verdict'].startswith('IDENTICAL')

@_t('feature source: S2 10-band guard text present; AlphaEarth (64-d) shape would fail the (N,1,10,T) check')
def _():
    assert 'NOT AlphaEarth' in FEATURE_SOURCE and 'median_S2_' in FEATURE_SOURCE


# =================================================================== NEW TESTS (authors alignment, coverage, safeguards)
@_t('authors-script recipe constants: Adam lr 1e-4 / weight decay 0 / batch 256 / no scheduler; TOTAL 200 epochs (v3.7; supervisor review: 38% of 8-state runs hit the old 100 cap) with early stopping (patience 15, min 20, strictly greater); deviations listed')
def _():
    assert (LR, WEIGHT_DECAY, BATCH_SIZE) == (AUTHORS_SCRIPT['lr'], 0.0, AUTHORS_SCRIPT['batch_size']) and AUTHORS_SCRIPT['scheduler'] is None
    assert (MAX_EPOCHS, PATIENCE, MIN_EPOCHS, MIN_DELTA) == (200, 15, 20, 0.0) and NUM_CLASSES == 7 and AUG == 'off'
    assert {'epochs', 'val_ratio', 'seeds', 'train_aug', 'class_weights'} <= set(DEVIATIONS_FROM_AUTHORS)

@_t('run_training_loop: <=200 total epochs, early stopping inside them, best epoch restored, restore verified (framework-free loop used by the torch cell)')
def _():
    def mk(scores):
        st = {'w': None, 'trained': 0}
        def train(ep): st['trained'] += 1; st['w'] = ('w', st['trained'])
        def val():
            cur = st['w']; return scores[cur[1] - 1] if cur is not None and cur[1] <= len(scores) else 0.0
        return st, train, val, (lambda: st['w']), (lambda s: st.__setitem__('w', s))
    st, tr, va, sn, rs = mk([0.001 * i for i in range(1, MAX_EPOCHS + 1)]); o = run_training_loop(tr, va, sn, rs); assert o['epochs_run'] == MAX_EPOCHS and o['stop_reason'] == 'max_epochs' and o['best_epoch'] == MAX_EPOCHS and o['restore_delta'] == 0 and st['trained'] == MAX_EPOCHS
    sc = [0.5 + 0.01 * i for i in range(30)] + [0.78 - 0.001 * i for i in range(1, 71)]; st, tr, va, sn, rs = mk(sc); o = run_training_loop(tr, va, sn, rs)
    assert o['best_epoch'] == 30 and o['epochs_run'] == 45 and o['stop_reason'] == 'early_stop_patience' and st['w'] == ('w', 30) and o['restore_delta'] == 0 and len(o['history']) == 45
    st, tr, va, sn, rs = mk([0.5] * MAX_EPOCHS); o = run_training_loop(tr, va, sn, rs); assert o['epochs_run'] == 20 and o['best_epoch'] == 1
    st, tr, va, sn, rs = mk([0.1 * (i + 1) / 10 for i in range(30)]); o = run_training_loop(tr, va, sn, lambda s: st.__setitem__('w', ('w', 1))); assert o['restore_delta'] > RESTORE_TOL
    st, tr, va, sn, rs = mk([float('nan')] * MAX_EPOCHS)
    try: run_training_loop(tr, va, sn, rs); raise RuntimeError('no stop')
    except FloatingPointError: pass
    try: run_training_loop(tr, va, sn, rs, max_epochs=MAX_EPOCHS + 1); raise RuntimeError('no stop')
    except AssertionError: pass

@_t('steps: steps_per_epoch = ceil(0.9*B/256) = 4/8/11/15; total_optimizer_steps = steps x epochs_run; both stored per run')
def _():
    assert [steps_per_epoch(int(0.9 * B)) for B in BUDGETS] == [4, 8, 11, 15] and 'steps_per_epoch' in COLUMNS and 'total_optimizer_steps' in COLUMNS

@_t('coverage columns: n_target_classes_train_ge30 / n_absent_target_classes_train in results; locked-design list contains every required name')
def _():
    for c_ in ['n_target_classes', 'n_target_classes_train_ge30', 'n_absent_target_classes_train']: assert c_ in COLUMNS
    for k in ['AUS_SPEC_CONFIRMED', 'AUS_KEY_SHARES_IF_CONFIRMED', 'TARGETS_TO_RUN', 'TARGET_CROPS', 'UNDERTRAINED_EPOCH', 'REVIEW_RULE', 'TRACKS']: assert k in LOCKED and k in globals(), k

@_t('geography STOP (France): rice only in regions 76 / 93; sugarcane / cotton reported; any violating row stops with the offending regions named')
def _():
    ok_st = np.array(['76', '93', '11', '24']); ok_cr = np.array(['rice', 'rice', 'corn', 'other'])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): out = check_geography(ok_st, ok_cr)
    assert set(out['rice']) == {'76', '93'} and 'report only' in buf.getvalue()
    try:
        with contextlib.redirect_stdout(io.StringIO()): check_geography(np.array(['11', '76']), np.array(['rice', 'rice']))
        raise RuntimeError('no stop')
    except RuntimeError as e: assert 'STOP' in str(e) and 'rice' in str(e) and "'11'" in str(e)

@_t('exhaustive cells flagged ("CI is degenerate by construction"); census k=1 flagged separately; sampled cells (C(P,k)>10) are neither')
def _():
    rows = []
    for bi, (k, P, ncomb) in enumerate(((1, 5, 5), (2, 5, 10), (2, 8, 10), (4, 12, 10))):
        for ci, cmb in enumerate(list(itertools.combinations(range(P), k))[:ncomb]):
            for s in range(2): rows.append(dict(run_id=f'{k}{P}{ci}{s}', status='ok', target='ARG', budget=1000 + 100 * bi, k=k, states='-'.join(f'S{j}' for j in cmb), seed=s, primary_f1=0.5 + 0.01 * ci + 0.001 * s, pool_size=P, n_total_combos=math.comb(P, k), key_n=200, target_crop_n=200, n_target_classes_train_ge30=4))
    df = pd.DataFrame(rows); cs = cell_stats(df)
    r = {(int(a), int(b)): (bool(c), bool(d), n) for a, b, c, d, n in zip(cs.k, cs.N_total_combos, cs.census, cs.exhaustive, cs.ci_note)}
    assert r[(1, 5)][0] and not r[(1, 5)][1]; assert r[(2, 10)][1] and 'degenerate by construction' in r[(2, 10)][2]; assert not r[(2, 28)][1] and not r[(4, 495)][1]

@_t('gate report: BOTH tracks side by side (EXISTING and REVIEW) with baselines + cov-cond, class coverage per target and k, optimizer steps, no pooled/ranked track')
def _():
    ok_pipeline = selftest_pipeline(return_report=True)
    assert 'EXISTING' in ok_pipeline and 'REVIEW' in ok_pipeline and 'cov-cond' in ok_pipeline and 'CLASS COVERAGE by target and k' in ok_pipeline and 'absent-class share' in ok_pipeline
    assert 'steps/epoch' in ok_pipeline and 'total optimizer steps' in ok_pipeline and 'EXISTING: mean by (target, budget, k)' in ok_pipeline and 'REVIEW-PROPOSED: mean by (target, budget, k)' in ok_pipeline

@_t('no track is declared "the paper\'s rule" in code labels (existing track = "existing release remap"; paper_text_v3 = literal sec.4.4 reading)')
def _():
    import re as _re
    labels = ' '.join(t_[1] for t_ in TRACKS) + ' ' + ' '.join(REMAP_RULES)
    assert not _re.search(r"paper'?s rule|the paper rule|paper-rule", labels, _re.I)
    sc = score_run('FRA', _ids(('corn', 20), ('other', 20)), _ids(('corn', 20), ('other', 20)), np.ones(7) * 100); assert sc['primary_view'] == 'existing_release_remap'


# ======================= v3.3 regression tests (each one belongs to a CHANGE ID in the change log) =======================
_PIPE_MEMO = {}
def _pipe_df():
    if 'v' not in _PIPE_MEMO: _PIPE_MEMO['v'] = selftest_pipeline(return_df=True)
    return _PIPE_MEMO['v'][0].copy(), list(_PIPE_MEMO['v'][1])

def _mk_err_row(template, **kw):
    e = template.copy()
    for c_ in e.index:
        if c_ not in ('run_id', 'status', 'target', 'budget', 'k', 'combo_id', 'states', 'seed', 'error', 'integrity_pass'): e[c_] = ''
    e['status'] = 'error'; e['error'] = 'boom'; e['integrity_pass'] = False
    for k_, v_ in kw.items(): e[k_] = v_
    return e

@_t('V33-01 gate_report: an error / blank-metric row FIRST in a budget group, a target/k group, a budget with only error rows, or zero ok rows never crashes; rows are listed')
def _():
    df0, cells = _pipe_df(); d0 = df0[pd.to_numeric(df0.seed) == 0].reset_index(drop=True); t0 = d0.iloc[0]
    base = gate_report(d0, 0, cells); assert 'RUNS WITHOUT METRICS (error / integrity_fail before scoring): 0' in base
    e1 = _mk_err_row(t0, run_id='E1')                                                                               # first of its budget AND target/k group
    r1 = gate_report(pd.concat([pd.DataFrame([e1]), d0], ignore_index=True), 0, cells); assert 'E1 [error] UNRESOLVED: boom' in r1 and 'RUNS WITHOUT METRICS (error / integrity_fail before scoring): 1' in r1
    e2 = _mk_err_row(t0, run_id='E2', budget=987654)                                                                 # a budget group that holds ONLY an error row
    r2 = gate_report(pd.concat([d0, pd.DataFrame([e2])], ignore_index=True), 0, cells); assert 'budget 987654' not in r2 and 'E2 [error]' in r2
    ok_id = d0.iloc[3]['run_id']; e3 = _mk_err_row(d0.iloc[3], run_id=ok_id)                                         # an error row that was later resolved by a resume
    r3 = gate_report(pd.concat([pd.DataFrame([e3]), d0], ignore_index=True), 0, cells); assert f'{ok_id} [error] SUPERSEDED' in r3
    e4 = _mk_err_row(t0, run_id='E4'); e4['status'] = 'integrity_fail'                                               # integrity_fail raised before scoring (blank metrics)
    r4 = gate_report(pd.concat([pd.DataFrame([e4]), d0], ignore_index=True), 0, cells); assert 'E4 [integrity_fail]' in r4
    r5 = gate_report(pd.DataFrame([e1]), 0, cells); assert 'RUNS WITHOUT METRICS (error / integrity_fail before scoring): 1' in r5        # nothing but errors
    # the abort-level integrity section still counts the error rows as failures (the gate is NOT weakened)
    assert int([l for l in r1.splitlines() if l.strip().startswith('I1_true_classes_ge2')][0].split('failures:')[1]) == 1

@_t('V33-02 gate_report skill flags: empty / NaN / one / several / mixed flag values all give a plain dict (no pandas .str accessor)')
def _():
    df0, cells = _pipe_df(); d0 = df0[pd.to_numeric(df0.seed) == 0].reset_index(drop=True); n = len(d0)
    def flags_line(vals):
        z = d0.copy(); z['skill_flags'] = vals; return [l for l in gate_report(z, 0, cells).splitlines() if l.strip().startswith('flags:')][0].strip()
    assert flags_line([''] * n) == 'flags: {}' and flags_line([np.nan] * n) == 'flags: {}' and flags_line([None] * n) == 'flags: {}'
    assert flags_line(['WARN_VAL:0.1'] + [''] * (n - 1)) == "flags: {'WARN_VAL': 1}"
    assert flags_line(['a:1|b:2'] * n) == "flags: {'a': %d, 'b': %d}" % (n, n)
    mixed = (['a:1|b:2', '', np.nan, 'a:9'] * n)[:n]; txt = flags_line(mixed); assert "'a'" in txt and "'b'" in txt
    z = d0.iloc[0:0]; assert 'GATE REPORT' in gate_report(z, 0, [])                                                   # zero rows

@_t('V33-08 score_run: y_true outside the target label set (paper Table 2) STOPs; valid truth, single-class, empty test set and undefined predictions behave as specified')
def _():
    for t in ['ARG', 'CHN', 'AUS', 'FRA']:
        yt = _labels_from_table2(t); undefined = [c for c in CROP_ORDER if c not in DEFINED_CLASSES[t]]
        for u in undefined:
            bad = np.r_[yt, np.full(30, CROP_ID[u])]
            try: score_run(t, bad, bad, np.array([100] * 7)); raise RuntimeError('no stop')
            except AssertionError as e: assert 'outside the target label set' in str(e), str(e)
        sc = score_run(t, yt, yt, np.array([100] * 7)); assert sc['existing_f1' if False else 'paper_f1'] == 1.0 and sc['review_f1'] == 1.0 and sc['papertext_f1'] == 1.0     # perfect predictions of a valid truth
        for u in undefined:                                                                                         # a prediction of an UNDEFINED class: existing / paper-text / review differ exactly as the rule table says
            yp = yt.copy(); yp[0] = CROP_ID[u]; sc = score_run(t, yt, yp, np.array([100] * 7)); assert 0 <= sc['review_f1'] <= 1 and 0 <= sc['paper_f1'] <= 1
    try: score_run('FRA', np.array([], int), np.array([], int), np.ones(7)); raise RuntimeError('no stop')
    except AssertionError as e: assert 'empty' in str(e)
    try: score_run('FRA', np.full(20, CROP_ID['corn']), np.full(20, CROP_ID['corn']), np.ones(7)); raise RuntimeError('no stop')
    except TooFewClasses: pass
    yt = _labels_from_table2('CHN'); yp = np.where(yt == CROP_ID['rice'], CROP_ID['corn'], yt); sc = score_run('CHN', yt, yp, np.array([100] * 7))      # a defined class never predicted
    assert sc['paper_f1_cls'][CROP_ID['rice']] == 0.0 and sc['review_f1'] < 1

@_t('V33-09 run_environment_seed<N>.json is written (software, device, settings, data identity, remap tables) without touching the manifest / LOCKED design')
def _():
    with tempfile.TemporaryDirectory() as td:
        locked_before = json.dumps({k_: globals()[k_] for k_ in LOCKED}, default=str, sort_keys=True)
        env = write_run_environment(td, 3, info={'ARG': dict(path=os.path.join(td, 'nope.npz'), shape=(10, 1, 10, 73), raw=10, kept=9, dup=1, raw_by_class={'corn': 10})}, geography={'rice': {'AR': 5}})
        p = os.path.join(td, 'run_environment_seed3.json'); assert os.path.exists(p); j = json.load(open(p))
        assert j['seed'] == 3 and j['settings']['class_weight_source'] == CLASS_WEIGHT_SOURCE and j['settings']['max_epochs'] == MAX_EPOCHS and 'ARG' in j['data_files'] and j['geography_observed'] == {'rice': {'AR': 5}}
        assert {'existing_release', 'paper_text_v3', 'review_targetmap'} <= set(j['remap_rules'][0]) and 'python' in j and 'numpy' in j
        assert locked_before == json.dumps({k_: globals()[k_] for k_ in LOCKED}, default=str, sort_keys=True)

@_t('V33-03 geography STOP message (France): counts, percentage, tolerance and the two ways forward are printed; tolerance 0 unchanged; a stated tolerance admits boundary rows')
def _():
    st = np.array(['76'] * 50 + ['11'] * 3 + ['93'] * 40); cr = np.array(['rice'] * 93); assert GEOGRAPHY_MAX_OUTSIDE_ROWS == 0
    try:
        with contextlib.redirect_stdout(io.StringIO()): check_geography(st, cr)
        raise RuntimeError('no stop')
    except RuntimeError as e: m = str(e); assert '3 of 93 rows' in m and "'11'" in m and 'GEOGRAPHY_MAX_OUTSIDE_ROWS=0' in m and 'NEIGHBOURING' in m
    old = globals()['GEOGRAPHY_MAX_OUTSIDE_ROWS']
    try:
        globals()['GEOGRAPHY_MAX_OUTSIDE_ROWS'] = 3
        with contextlib.redirect_stdout(io.StringIO()): out = check_geography(st, cr)
        assert out['rice']['11'] == 3
    finally: globals()['GEOGRAPHY_MAX_OUTSIDE_ROWS'] = old

@_t('V33-10 regression output is labelled EXPLORATORY (overlapping state combinations are not independent replicates)')
def _():
    df0, cells = _pipe_df(); rg = regression_report(df0, 'primary_f1'); assert len(rg) > 0 and rg['note'].str.contains('EXPLORATORY').all()

N_FAIL_A = sum(1 for r in TEST_REPORT if r[1] != 'PASS')
for name, res, msg in TEST_REPORT: print(f'  [{res}] {name}' + (f'\n         -> {msg}' if msg else ''))
TESTS_A_PASSED = (N_FAIL_A == 0)
print('\nTEST GATE PART A:', 'ALL PASSED' if TESTS_A_PASSED else f'{N_FAIL_A} FAILED -- training is BLOCKED')
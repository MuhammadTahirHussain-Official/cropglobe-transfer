# ======================================================================================
# SUMMARY (safe to run at any time).  Unit = REGION combination (French regions); the seeds are averaged FIRST.  Binary (ARG/CHN) and multiclass (USA) F1 are never compared with each other.
# ======================================================================================
import matplotlib
if not IN_NOTEBOOK: matplotlib.use('Agg')
import matplotlib.pyplot as plt
if PROFILE == 'extension': print('=' * 110 + f'\n  EXTENSION APPENDIX ({TARGETS_TO_RUN} at budgets {BUDGETS}; random arm only; allocation = as equal as each state\'s capacity allows, floor {EXT_MIN_CONTRIB:.0%} of the equal share).\n  These results are NOT part of the main study: do not pool them with the main results.csv; the combinations here overlap heavily (few states), so intervals are descriptive.\n  Compare with main results only at the same target and k, and always look at max_state_share (how much of the budget the largest single state supplies).\n' + '=' * 110)
RESULTS_CSV = os.path.join(OUT_DIR, 'results.csv'); df = load_results(RESULTS_CSV); _check(len(df) > 0, 'no results yet')
print('runs by status:', df['status'].value_counts().to_dict()); ok = df[df['status'] == 'ok']
print(f'seeds present: {sorted(pd.to_numeric(ok.seed).unique().astype(int).tolist())} (final statistics need all of {SEEDS}); integrity-failed / errored runs are EXCLUDED from the statistics')
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30); os.makedirs(os.path.join(OUT_DIR, 'tables'), exist_ok=True)
SUMMARY = {}
def summarize(comp, metric, label, full=True):
    dfc = df[(df['composition'].fillna(BASE_MODE).replace('', BASE_MODE) == comp) & df['arms'].fillna('random').replace('', 'random').astype(str).str.split('|').apply(lambda x: 'random' in x)]; CS = cell_stats(dfc, metric)   # legacy tables = the RANDOM arm only (unchanged method)
    if not len(CS): return None
    GAIN = gain_kmin_kmax(CS); REG = regression_report(dfc, metric); tag = f'{comp}__{metric}'
    for name, t in [('cell_stats', CS), ('gain_kmin_kmax', GAIN), ('regression', REG)]: atomic_write(os.path.join(OUT_DIR, 'tables', f'{name}__{tag}.csv'), lambda f, t=t: t.to_csv(f, index=False), mode='w')
    SUMMARY[(comp, metric)] = (CS, GAIN, REG)
    if full:
        print(f'\n########## [{comp}] {label}  (column {metric}) ##########')
        show = CS.copy()
        for c in ['mean', 'ci_lo', 'ci_hi', 'jackknife_lo', 'jackknife_hi', 'seed_sd_within_combo', 'seed_sd_of_cell_means']: show[c] = (100 * show[c]).round(2)
        print('--- per cell (macro-F1 %; k>=2: 95% t-interval with finite-population correction over state combinations; k=1 census: no CI; jackknife = delete-one-state) ---')
        print(show.drop(columns=['census_values', 'se', 'jackknife_se', 'census', 'exhaustive']).to_string(index=False))
        if show.exhaustive.any(): print('   * rows marked "exhaustive" had EVERY combination enumerated (C(P,k) <= %d): the interval is a point BY CONSTRUCTION, not evidence of certainty about unseen states (see the jackknife columns).' % MAX_COMBOS)
        for r in CS[CS.census].itertuples(): print(f'  k=1 census {r.target} B{r.budget}: {r.census_values}')
        g2 = GAIN.copy()
        for c in ['gain', 'ci_lo', 'ci_hi']: g2[c] = (100 * g2[c]).round(2)
        print('--- gain k_min -> k_max (pp, with CI) ---'); print(g2.to_string(index=False))
        print('--- regression of combination-level F1 on k, n_target_classes_train_ge30, budget ---'); print(REG.round(5).to_string(index=False))
    return CS
comps = sorted(set(df['composition'].fillna(BASE_MODE).replace('', BASE_MODE)))
if 'arms' in df.columns and (df['arms'].fillna('').astype(str).str.contains('diverse|clustered|common|kcenter|aug_on')).any():
    print('\n########## ARMS: random = baseline | diverse / clustered = state-diversity selection (Phase 4) | common = fixed common state pool | kcenter = k-center SAMPLE selection | aug_on = authors\' augmentation ON ##########')
    _E = expand_arms(df)
    for _comp in comps:
        _Ec = _E[_E['composition'].fillna(BASE_MODE).replace('', BASE_MODE) == _comp]
        for _arm in ARM_ORDER:
            _Ea = _Ec[_Ec['arm'] == _arm]
            if not len(_Ea): continue
            _CS = cell_stats(_Ea, REPORT_PRIMARY); atomic_write(os.path.join(OUT_DIR, 'tables', f'cell_stats__{_comp}__arm_{_arm}.csv'), lambda f, t=_CS: t.to_csv(f, index=False), mode='w')
            _sh = _CS[['target', 'budget', 'k', 'n_combos', 'mean', 'ci_lo', 'ci_hi', 'ci_note']].copy()
            for c_ in ['mean', 'ci_lo', 'ci_hi']: _sh[c_] = (100 * _sh[c_]).round(2)
            print(f'--- [{_comp}] arm = {_arm} (MAIN metric, %) ---'); print(_sh.to_string(index=False))
        for _a in ('diverse', 'clustered', 'common', 'kcenter', 'aug_on', 'aug_copy'):
            if _a in ('kcenter', 'aug_on', 'aug_copy'):
                _C = arm_paired_contrast(_Ec.drop(columns=['arm']), REPORT_PRIMARY, _a, 'random')
                if len(_C):
                    atomic_write(os.path.join(OUT_DIR, 'tables', f'arm_paired__{_comp}__{_a}_vs_random.csv'), lambda f, t=_C: t.to_csv(f, index=False), mode='w'); _s = _C.copy()
                    for c_ in ['diff', 'ci_lo', 'ci_hi']: _s[c_] = (100 * _s[c_]).round(2)
                    print(f'--- [{_comp}] PAIRED contrast {_a} - random on the SAME state combinations (pp, MAIN metric; units overlap in states: DESCRIPTIVE) ---'); print(_s.round(4).to_string(index=False))
                continue
            if _a == 'common': continue                                  # different pool: reported as its own arm table + the budget table below, not as a contrast
            _C = arm_contrast(_Ec.drop(columns=['arm']), REPORT_PRIMARY, _a, 'random')
            if len(_C):
                atomic_write(os.path.join(OUT_DIR, 'tables', f'arm_contrast__{_comp}__{_a}_vs_random.csv'), lambda f, t=_C: t.to_csv(f, index=False), mode='w')
                _s = _C.copy()
                for c_ in ['diff', 'ci_lo', 'ci_hi']: _s[c_] = (100 * _s[c_]).round(2)
                print(f'--- [{_comp}] contrast {_a} - random (pp; Welch interval over combinations, which share states: DESCRIPTIVE) ---'); print(_s.round(4).to_string(index=False))
        _A = diversity_assoc(_Ec[_Ec['arm'].isin(['random', 'diverse', 'clustered'])].drop(columns=['arm']).drop_duplicates('run_id'), REPORT_PRIMARY)
        if len(_A): print(f'--- [{_comp}] Spearman(F1, diversity percentile) per cell (EXPLORATORY) ---'); print(_A.round(3).to_string(index=False))
    _Cm = _E[_E['arm'] == 'common']
    if len(_Cm):
        print('\n--- FIXED COMMON STATE POOL: MAIN metric (%) by budget and k, every budget drawn from the SAME states (no pool-shrinkage confound) ---')
        _tab = _Cm.assign(m=pd.to_numeric(_Cm[REPORT_PRIMARY], errors='coerce')).groupby(['target', 'k', 'budget'])['m'].mean().unstack('budget').mul(100).round(2); print(_tab.to_string())
        atomic_write(os.path.join(OUT_DIR, 'tables', 'common_pool_budget_by_k.csv'), lambda f: _tab.to_csv(f), mode='w')
    _Au = _E[_E['arm'] == 'aug_on']
    if len(_Au): print('\n   NOTE aug_on: the augmented runs see %dx the training rows per epoch (more optimizer steps); the aug_on - random contrast mixes augmentation with training length.  The steps-matched comparison is aug_on - aug_copy below.' % (1 + AUG_REPEAT))
    for _comp in comps:
        _Ec = _E[_E['composition'].fillna(BASE_MODE).replace('', BASE_MODE) == _comp]
        if (_Ec['arm'] == 'aug_copy').any() and (_Ec['arm'] == 'aug_on').any():
            _C = arm_paired_contrast(_Ec.drop(columns=['arm']), REPORT_PRIMARY, 'aug_on', 'aug_copy')
            if len(_C):
                atomic_write(os.path.join(OUT_DIR, 'tables', f'arm_paired__{_comp}__aug_on_vs_aug_copy.csv'), lambda f, t=_C: t.to_csv(f, index=False), mode='w'); _s = _C.copy()
                for c_ in ['diff', 'ci_lo', 'ci_hi']: _s[c_] = (100 * _s[c_]).round(2)
                print(f'--- [{_comp}] STEPS-MATCHED contrast aug_on - aug_copy on the SAME combinations (pp, MAIN metric): the part of the augmentation gain that is NOT just more steps ---'); print(_s.round(4).to_string(index=False))
    print('   NOTE: diverse / clustered combinations are CHOSEN by score, not sampled; their intervals describe the selected sets only.  A single target country per track: no claim beyond the countries tested.')
print('\n########## v3.8 GENERALISATION STATISTICS (random arm; MAIN metric, points) ##########')
_MM = mixed_model_report(df, REPORT_PRIMARY)
if len(_MM):
    atomic_write(os.path.join(OUT_DIR, 'tables', 'mixed_model.csv'), lambda f: _MM.to_csv(f, index=False), mode='w')
    print('--- mixed model: F1 ~ log2(k) + budget/1000 + seed, random intercept = region combination (REML; t-based p, df = combinations - parameters) ---'); print(_MM.round(4).to_string(index=False))
_SS = single_state_report(df, REPORT_PRIMARY)
if len(_SS):
    atomic_write(os.path.join(OUT_DIR, 'tables', 'single_state_vs_combinations.csv'), lambda f: _SS.to_csv(f, index=False), mode='w'); _x = _SS.copy()
    for c_ in ['single_mean', 'single_best', 'single_worst', 'combo_mean', 'combo_worst', 'combo_best', 'avg_gain', 'avg_gain_ci_lo', 'avg_gain_ci_hi', 'worst_case_gain', 'worst_vs_best_single']: _x[c_] = _x[c_].round(2)
    _x['share_beating_best_single'] = (100 * _x['share_beating_best_single']).round(0)
    print('--- single states (k=1 census) vs k>=2 combinations: average gain (bootstrap interval over combinations), worst-case gain, share of combinations beating the BEST single state (%) ---'); print(_x.to_string(index=False))
_RT = replication_table(df, REPORT_PRIMARY)
if len(_RT):
    atomic_write(os.path.join(OUT_DIR, 'tables', 'replication_exploratory_vs_confirmatory.csv'), lambda f: _RT.to_csv(f, index=False), mode='w'); _y = _RT.copy()
    for c_ in ['estimate', 'ci_lo', 'ci_hi']: _y[c_] = _y[c_].round(2)
    print('--- PRE-DECLARED CLAIMS (V38-P1): exploratory seeds 0-1 vs confirmatory seeds 2-4.  PASS = same sign in both groups and the confirmatory interval excludes 0.  A failed claim is reported as NOT REPLICATED; no seed is dropped or re-run. ---'); print(_y.to_string(index=False))
print('compositions present:', comps, '| tracks are reported SEPARATELY; binary (ARG/CHN) is never compared with multiclass (USA)')
for comp in comps:
    summarize(comp, REPORT_PRIMARY, 'MAIN metric (ARG/CHN: binary key-crop macro-F1; USA: macro-F1 over the FIXED classes corn, soybeans, wheat, other; true rice/sugarcane/cotton removed, their predictions -> other (L2-D11))'); summarize(comp, REPORT_SECONDARY, 'SECONDARY: paper-style macro-F1 (read with the coverage columns)')
    if REPORT_ALL_TRACKS:
        for col, lab in TRACKS:
            if col not in (REPORT_PRIMARY, REPORT_SECONDARY): summarize(comp, col, lab, full=False)
print('   NOTE: the target-crop COUNT equals fixed share x budget exactly (aliased with budget), so the regression uses n_target_classes_train_ge30 instead. Rows from different budgets reuse the same states (not independent).')
print('\n--- HEADLINE: MAIN metric and SECONDARY (paper-style) metric, each with its coverage-conditional version beside it (%%; cov-cond over classes with >=%d train samples) ---' % COVERAGE_MIN_TRAIN)
for comp in comps:
    dfc = df[(df['composition'].fillna(BASE_MODE).replace('', BASE_MODE) == comp) & df['arms'].fillna('random').replace('', 'random').astype(str).str.split('|').apply(lambda x: 'random' in x)]; frames = []
    for metric, cov, lab in ((REPORT_PRIMARY, 'cov_cond_primary_f1', 'MAIN'), (REPORT_SECONDARY, 'cov_cond_paper_f1', 'SECONDARY')):
        a = combo_level(dfc, metric).groupby(['target', 'budget', 'k'])['f1'].mean().rename(f'{lab} main'); b_ = combo_level(dfc, cov).groupby(['target', 'budget', 'k'])['f1'].mean().rename(f'{lab} cov-cond')
        frames += [a, b_]
    H = pd.concat(frames, axis=1); H = (100 * H).round(1); print(f'[{comp}]'); print(H.to_string())
print('\n--- optimizer steps (budget is confounded with training length) ---')
_st = ok.copy()
for c_ in ['budget', 'steps_per_epoch', 'total_optimizer_steps', 'epochs_run']: _st[c_] = pd.to_numeric(_st[c_], errors='coerce')
_st = _st[_st['aug'].fillna('off').astype(str) == 'off'] if 'aug' in _st.columns else _st
print('(aug-OFF runs only; aug-ON runs have %dx the steps per epoch)' % (1 + AUG_REPEAT)); print(_st.groupby('budget').agg(steps_per_epoch=('steps_per_epoch', 'first'), epochs_run_median=('epochs_run', 'median'), total_steps_min=('total_optimizer_steps', 'min'), total_steps_median=('total_optimizer_steps', 'median'), total_steps_max=('total_optimizer_steps', 'max')).to_string())
print('\n--- class coverage by target and k ---')
_cv = ok.copy()
for c_ in ['k', 'n_absent_target_classes_train', 'n_target_classes_train_ge30']: _cv[c_] = pd.to_numeric(_cv[c_], errors='coerce')
print(_cv.groupby(['target', 'k']).agg(runs=('k', 'size'), share_with_absent_class=('n_absent_target_classes_train', lambda x: round(float((x > 0).mean()), 3)), mean_absent=('n_absent_target_classes_train', 'mean'), mean_classes_ge30=('n_target_classes_train_ge30', 'mean')).round(2).to_string())
SEED_WARN = seed_direction_warnings(df); print('\n--- seed-direction check (WARNING only) ---'); print('\n'.join(SEED_WARN) if SEED_WARN else '  no disagreement')
if REPORT_ALL_TRACKS:
    print('\n--- ALL TRACK COLUMNS (mean over runs, %; each column has its OWN definition, see TRACKS in the config cell; no track is pooled or ranked) ---')
    b = ok.copy(); cols = [c for c, _ in TRACKS] + ['baseline_all_other_primary', 'baseline_all_other_review', 'cov_cond_primary_f1', 'cov_cond_review_f1']
    for c in cols: b[c] = pd.to_numeric(b[c], errors='coerce')
    print((100 * b.groupby(['composition', 'target'])[cols].mean()).round(2).to_string())
    for col, lab in TRACKS: print(f'   {col:14s} = {lab}')
fl = ok['skill_flags'].fillna('').astype(str).str.split('|').explode().replace('', np.nan).dropna(); print('skill flags raised:', fl.str.replace(r':.*', '', regex=True).value_counts().to_dict())
CS = SUMMARY.get((comps[0], REPORT_PRIMARY), (None,))[0]
tg = [t for t in TARGETS_ORDER if CS is not None and t in set(CS.target)]
if tg:
    fig, axes = plt.subplots(2, len(tg), figsize=(5.2 * len(tg), 8), squeeze=False)
    for r_, (metric, rowlab) in enumerate(((REPORT_PRIMARY, 'MAIN'), (REPORT_SECONDARY, 'SECONDARY (paper-style)'))):
        CSm = SUMMARY.get((comps[0], metric), (None,))[0]
        for ax, t in zip(axes[r_], tg):
            if CSm is None or t not in set(CSm.target): continue
            for B, g in CSm[CSm.target == t].groupby('budget'):
                samp = g[~g.census & ~g.exhaustive]
                line = ax.errorbar(g.k, g['mean'], yerr=None, marker='o', ls='-', label=f'B={B}')
                if len(samp): ax.errorbar(samp.k, samp['mean'], yerr=[(samp['mean'] - samp.ci_lo).fillna(0), (samp.ci_hi - samp['mean']).fillna(0)], fmt='none', ecolor=line[0].get_color(), capsize=3)
                ex = g[g.exhaustive]; ce = g[g.census]
                if len(ex): ax.scatter(ex.k, ex['mean'], marker='D', facecolors='none', edgecolors=line[0].get_color(), s=70, zorder=5)        # hollow diamond = exhaustive, CI degenerate
                if len(ce): ax.scatter(ce.k, ce['mean'], marker='s', facecolors='none', edgecolors=line[0].get_color(), s=70, zorder=5)         # hollow square = k=1 census, no CI
            ax.set_xscale('log', base=2); ks = sorted(CSm[CSm.target == t].k.unique()); ax.set_xticks(ks); ax.set_xticklabels(ks); ax.set_xlabel('k (states)'); ax.set_ylabel('macro-F1'); ax.set_title(f'{t} - {rowlab}\n(bars = 95% CI; hollow square = census, hollow diamond = exhaustive: no CI drawn)', fontsize=8); ax.legend(fontsize=7)
    fig.tight_layout(); atomic_write(os.path.join(OUT_DIR, 'tables', 'f1_vs_k_by_track.png'), lambda f: fig.savefig(f, dpi=130), mode='wb'); plt.show() if IN_NOTEBOOK else None
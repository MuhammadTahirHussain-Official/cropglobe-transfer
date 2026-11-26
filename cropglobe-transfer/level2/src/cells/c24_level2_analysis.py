# ======================================================================================
# LEVEL 2 CLOSING ANALYSIS (read-only).  Run ONLY after all five seeds are finished.  Nothing here trains a model or writes to results.csv.
#   1. FREEZE: SHA-256 of results.csv is written to closing/FREEZE_level2.json (paste it into LEVEL2_FREEZE.md; after that nobody can change a number unnoticed)
#   2. CLAIMS (pre-registered): L2-C1 and L2-C2 on ARG and USA, evaluated separately on seeds 0-1 (exploratory) and 2-4 (confirmatory); PASS = same sign AND confirmatory 95% interval excludes 0;
#      anything else is printed as NOT REPLICATED.  CHN is descriptive only.  L2-C3 (aug_on - aug_copy) is reported with its interval, no direction assumed.
#   3. CROSS-CHECK of the mixed model with statsmodels (declared tolerance) and the training-quality robustness table.
#   The verdict words are produced by the code from the numbers; they are not edited by hand.
# ======================================================================================
import hashlib, json, os
import numpy as np, pandas as pd
from scipy import stats as scipy_stats
CLOSING_TOL = dict(coef=0.02, se=0.02)           # statsmodels vs own REML must agree this closely, declared before the cross-check was run
N_BOOT_CLOSING = 2000

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''): h.update(b)
    return h.hexdigest()

def load_final(path):
    d = load_results(path); d = d[d['status'] == 'ok'].drop_duplicates('run_id', keep='last').copy(); d['seed'] = pd.to_numeric(d['seed'])
    d['track'] = np.where(d['aug'].astype(str) == 'copy', 'aug_copy', np.where(d['aug'].astype(str) == 'on', 'aug_on', np.where(d['sample_sel'].astype(str) == 'kcenter', 'kcenter', 'base')))
    d['y'] = 100 * pd.to_numeric(d['main_f1'], errors='coerce'); return d

def _mixed_design(g):
    g = g.dropna(subset=['y']).copy(); g['lk'] = np.log2(g['k']); g['B'] = g['budget'] / 1000.0; g['combo'] = g['target'].astype(str) + '|' + g['states'].astype(str); seeds = sorted(g['seed'].dropna().unique())
    cols, names = [np.ones(len(g)), g['lk'].values], ['intercept', 'log2(k)']
    if g['B'].nunique() > 1: cols.append(g['B'].values); names.append('budget/1000')
    for s in seeds[1:]: cols.append((g['seed'].values == s).astype(float)); names.append(f'seed{int(s)}')
    return g, np.column_stack(cols), names

def own_vs_statsmodels(df, targets=('ARG', 'USA', 'CHN'), try_install=True):
    # fits the SAME model (random arm, F1 ~ log2(k) + budget/1000 + seed dummies, random intercept per state combination, REML) with the notebook's own code and with statsmodels MixedLM
    out = []; sm = None; err = ''
    try:
        import statsmodels.formula.api as smf; sm = smf
    except Exception:
        if try_install:
            try:
                import subprocess, sys; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'statsmodels'], check=True); import statsmodels.formula.api as smf; sm = smf
            except Exception as e: err = f'{type(e).__name__}: {str(e)[:80]}'
    base = _random_arm_runs(df); base['y'] = 100 * pd.to_numeric(base['main_f1'], errors='coerce')
    for t in targets:
        g, X, names = _mixed_design(base[base['target'] == t])
        if g['k'].nunique() < 2: continue
        r = reml_random_intercept(g['y'].values, X, g['combo'].values); row = dict(target=t, n=len(g), groups=int(r['G']), own_coef=float(r['beta'][1]), own_se=float(np.sqrt(r['cov'][1, 1])), sm_coef=np.nan, sm_se=np.nan, sm_p=np.nan, agree='NOT RUN (statsmodels unavailable' + (f': {err}' if err else '') + ')')
        if sm is not None:
            import warnings
            gg = g.copy(); gg['seedf'] = gg['seed'].astype(int).astype(str); terms = ['lk'] + (['B'] if g['B'].nunique() > 1 else []) + (['C(seedf)'] if g['seed'].nunique() > 1 else []); formula = 'y ~ ' + ' + '.join(terms)
            tried = []; fitted = None
            for scale in (1.0, 0.1):                              # second pass: y rescaled (coefficients and SEs are rescaled back) in case the optimiser failed on the scale of y
                for meth in ('lbfgs', 'bfgs', 'powell', 'nm', 'cg'):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore'); d2 = gg.assign(y=gg['y'] * scale); m = sm.mixedlm(formula, d2, groups=d2['combo']).fit(reml=True, method=meth)
                        if np.isfinite(m.params['lk']) and np.isfinite(m.bse['lk']) and m.bse['lk'] > 0: fitted = (m, meth, scale); break
                        tried.append(f'{meth}x{scale}: non-finite')
                    except Exception as e: tried.append(f'{meth}x{scale}: {type(e).__name__}')
                if fitted: break
            if fitted:
                m, meth, scale = fitted; row.update(sm_coef=float(m.params['lk']) / scale, sm_se=float(m.bse['lk']) / scale, sm_p=float(m.pvalues['lk']), sm_method=f'{meth}' + ('' if scale == 1.0 else ' (y x0.1)'))
                row['agree'] = 'YES' if (abs(row['sm_coef'] - row['own_coef']) <= CLOSING_TOL['coef'] and abs(row['sm_se'] - row['own_se']) <= CLOSING_TOL['se']) else 'NO -- report the statsmodels figures and investigate'
            else: row['agree'] = 'statsmodels fit failed with every optimiser: ' + '; '.join(tried[:4])
        out.append(row)
    return pd.DataFrame(out)

def robustness_table(df, target='ARG', budgets=(1000, 2000), k_lo=1, k_hi=8, n_boot=None, seed=0):
    # does the benefit of more states survive better training?  Gain k_lo -> k_hi on the SAME cells for plain training, the steps-matched copy control and augmentation.
    n_boot = n_boot or N_BOOT_CLOSING; rng = np.random.default_rng(seed); d = df[df['target'] == target].copy(); rows = []
    key = ['budget', 'k', 'states']
    cp = d[d['track'] == 'aug_copy'].groupby(key)['y'].mean(); on = d[d['track'] == 'aug_on'].groupby(key)['y'].mean(); pl = d[d['track'] == 'base'].groupby(key)['y'].mean()
    cells = cp.index   # like-for-like: plain training restricted to the cells that have a copy-control run
    pl = pl.reindex(cells)
    for B in budgets:
        out = dict(target=target, budget=B)
        for nm, s in (('plain', pl), ('copy_control', cp), ('augmentation', on.reindex(cells))):
            s = s[s.index.get_level_values('budget') == B]; lo = s[s.index.get_level_values('k') == k_lo].dropna(); hi = s[s.index.get_level_values('k') == k_hi].dropna()
            if len(lo) == 0 or len(hi) == 0: out[nm + '_gain'] = np.nan; continue
            out[nm + '_gain'] = float(hi.mean() - lo.mean())
            bs = [float(rng.choice(hi.values, len(hi)).mean() - rng.choice(lo.values, len(lo)).mean()) for _ in range(n_boot)]; out[nm + '_lo'], out[nm + '_hi'] = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        rows.append(out)
    # does the improvement from better training depend on the number of states?  (combination-level regression of copy-control minus plain on log2 k)
    gdiff = (cp - pl).dropna().reset_index(); gdiff.columns = ['budget', 'k', 'states', 'g']
    sl = scipy_stats.linregress(np.log2(gdiff['k']), gdiff['g']) if len(gdiff) > 3 and gdiff['k'].nunique() > 1 else None
    dep = dict(slope=float(sl.slope), p=float(sl.pvalue), n_combinations=int(len(gdiff))) if sl else dict(slope=np.nan, p=np.nan, n_combinations=int(len(gdiff)))
    return pd.DataFrame(rows), dep


def freeze_results(path, out_dir):
    # records the hash and the shape of the finished results file.  Refuses an unfinished file (a hash of a partial file would freeze nothing).
    d = pd.read_csv(path); ok = d[d['status'] == 'ok']; planned = {s_: len(plan_for_seed(s_)) for s_ in SEEDS}
    per = {int(k): int(v) for k, v in ok.groupby(pd.to_numeric(ok['seed'], errors='coerce')).size().to_dict().items()}
    complete = (per == planned) and int(d['run_id'].duplicated().sum()) == 0 and int((d['status'] != 'ok').sum()) == 0
    rec = dict(results_file=os.path.basename(path), sha256=sha256_file(path), n_rows=int(len(d)), runs_per_seed=per, planned_per_seed=planned, complete=bool(complete), design_sha256=APPROVED_DESIGN_SHA256, level1_results_sha256='38d1876cde08e54cfc827aa6b2813355ef98a09dbd048885a67d58991534f891')
    if complete: atomic_write(os.path.join(out_dir, 'FREEZE_level2.json'), lambda f: f.write(json.dumps(rec, indent=1)), mode='w')
    return rec

def c3_aug_contrast(df, target='ARG', budgets=(1000, 2000), exploratory=(0, 1), confirmatory=(2, 3, 4), n_boot=None, seed=0):
    # L2-C3: aug_on minus aug_copy (same cells, same optimizer steps).  Per budget and per seed group: mean over cells of (mean over the group's seeds), with a bootstrap-over-cells 95% interval.  No direction is assumed.
    n_boot = n_boot or N_BOOT_CLOSING; rng = np.random.default_rng(seed); d = df[(df['target'] == target) & df['track'].isin(['aug_on', 'aug_copy'])].copy(); rows = []
    for B in budgets:
        for label, seeds in (('exploratory', exploratory), ('confirmatory', confirmatory)):
            g = d[(d['budget'] == B) & d['seed'].isin(seeds)]
            m = g.groupby(['k', 'states', 'track'])['y'].mean().unstack('track')
            if not {'aug_on', 'aug_copy'} <= set(m.columns): continue
            diff = (m['aug_on'] - m['aug_copy']).dropna().values
            if len(diff) < 2: continue
            bs = [float(rng.choice(diff, len(diff)).mean()) for _ in range(n_boot)]
            rows.append(dict(claim='C3 aug_on - aug_copy (points)', target=target, budget=B, group=label, seeds=sorted(int(s) for s in g['seed'].unique()), estimate=float(diff.mean()), ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)), n_cells=int(len(diff))))
    t = pd.DataFrame(rows)
    if not len(t): return t
    v = {}
    for B, g in t.groupby('budget'):
        ex, co = g[g['group'] == 'exploratory'], g[g['group'] == 'confirmatory']
        if not len(co) or not len(ex): v[B] = 'confirmatory seeds not finished'; continue
        e, c = ex.iloc[0], co.iloc[0]; same = np.sign(e['estimate']) == np.sign(c['estimate']); excl = c['ci_lo'] > 0 or c['ci_hi'] < 0
        v[B] = ('REPLICATED: augmentation beats its steps-matched copy' if c['estimate'] > 0 else 'REPLICATED: augmentation is worse than its steps-matched copy') if (same and excl) else ('NOT REPLICATED (sign differs)' if not same else 'NOT REPLICATED (confirmatory interval includes 0)')
    t['verdict'] = t['budget'].map(v); return t


# ---------------- EXPLORATORY additions requested in the instructor review (NOT pre-registered claims; every table is labelled exploratory) ----------------
HIGH_REGIONS_ARG = ('75', '76')                  # the two Argentine-like French regions that are far better than the rest as single regions (descriptive label only)

def _base_runs(df, target):
    return df[(df['track'] == 'base') & (df['target'] == target)].copy()

def mix_vs_own(df, target, n_boot=None, seed=0):
    """EXPLORATORY.  For every region mix (k>=2) at a budget: its seed-averaged F1 minus (a) the mean and (b) the best of its OWN single regions at the SAME budget.
    Mixes are not independent (they share regions and single-region runs), so the p-values are descriptive; the bootstrap resamples mixes."""
    n_boot = n_boot or N_BOOT_CLOSING; g = _base_runs(df, target); sc = g.groupby(['budget', 'k', 'states'])['y'].mean().reset_index(); sc['states'] = sc['states'].astype(str)
    single = sc[sc['k'] == 1].set_index(['budget', 'states'])['y']; rows = []
    for x in sc[sc['k'] >= 2].itertuples():
        mem = x.states.split('-')
        if not all((x.budget, m) in single.index for m in mem): continue
        v = [float(single[(x.budget, m)]) for m in mem]; rows.append(dict(budget=int(x.budget), k=int(x.k), states=x.states, y=float(x.y), own_mean=float(np.mean(v)), own_best=float(np.max(v)), has_high=bool(set(HIGH_REGIONS_ARG) & set(mem))))
    return pd.DataFrame(rows)

def _gap_summary(gap, n_boot, seed):
    gap = np.asarray(gap, float); gap = gap[np.isfinite(gap)]
    if len(gap) < 3: return dict(n=len(gap), mean_gap=np.nan, lo=np.nan, hi=np.nan, share_better=np.nan, p_t=np.nan, p_wilcoxon=np.nan)
    r = np.random.default_rng(seed); bs = np.array([gap[r.integers(0, len(gap), len(gap))].mean() for _ in range(n_boot)])
    try: pw = float(scipy_stats.wilcoxon(gap).pvalue)
    except Exception: pw = np.nan
    return dict(n=len(gap), mean_gap=float(gap.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), share_better=float((gap > 0).mean()), p_t=float(scipy_stats.ttest_1samp(gap, 0).pvalue), p_wilcoxon=pw)

def mix_vs_own_table(df, targets=('ARG', 'USA'), n_boot=None):
    out = []
    for t in targets:
        m = mix_vs_own(df, t)
        if not len(m): continue
        for lab, sub in [('all mixes', m)] + ([('without 75/76', m[~m.has_high]), ('containing 75/76', m[m.has_high])] if t == 'ARG' else []):
            for ref, col in [('mean of own single regions', 'own_mean'), ('best own single region', 'own_best')]:
                r_ = _gap_summary(sub['y'] - sub[col], n_boot or N_BOOT_CLOSING, 0); r_.update(target=t, subset=lab, reference=ref, status='EXPLORATORY (not pre-registered)'); out.append(r_)
    cols = ['target', 'subset', 'reference', 'n', 'mean_gap', 'lo', 'hi', 'share_better', 'p_t', 'p_wilcoxon', 'status']
    return pd.DataFrame(out)[cols] if out else pd.DataFrame(columns=cols)

def fixed_pool_budget_effect(df, target='ARG'):
    """EXPLORATORY / correction.  Budget effect for single-region runs using ONLY regions that exist at every budget (the pool otherwise shrinks as the budget grows, which would change the average by itself)."""
    a = _base_runs(df, target); a = a[a['k'] == 1].copy(); a['r'] = a['states'].astype(str)
    if not len(a): return pd.DataFrame()
    nb = a['budget'].nunique(); av = a.groupby('r')['budget'].nunique(); common = sorted(av[av == nb].index.tolist())
    f = a[a['r'].isin(common)].groupby('budget')['y'].mean(); allr = a.groupby('budget')['y'].mean(); regs = a.groupby('budget')['r'].nunique()
    return pd.DataFrame(dict(target=target, budget=f.index.astype(int), fixed_pool_mean_f1=f.values, fixed_pool_regions=len(common), all_available_mean_f1=allr.reindex(f.index).values, regions_available=regs.reindex(f.index).values, fixed_pool_change_from_lowest_budget=f.values - f.values[0]))

def per_doubling_subset(df, target='ARG', exclude=HIGH_REGIONS_ARG):
    """EXPLORATORY.  Mixed-model log2(k) slope (same model as L2-C1, all seeds pooled) with and without runs whose region combination contains an excluded region."""
    rows = []
    for lab, drop in [('all combinations', False), ('without ' + '/'.join(exclude), True)]:
        b = _random_arm_runs(df); b = b[b['target'] == target]
        if drop: b = b[~b['states'].astype(str).apply(lambda s: bool(set(exclude) & set(s.split('-'))))]
        if b['k'].nunique() < 2: continue
        g, X, names = _mixed_design(b); r = reml_random_intercept(g['y'].values, X, g['combo'].values); se = float(np.sqrt(r['cov'][1, 1]))
        rows.append(dict(target=target, subset=lab, n=len(g), groups=int(r['G']), per_doubling=float(r['beta'][1]), se=se, lo=float(r['beta'][1]) - 1.96 * se, hi=float(r['beta'][1]) + 1.96 * se, status='EXPLORATORY'))
    return pd.DataFrame(rows)

def closing_report_l2(results_path, out_dir):
    os.makedirs(out_dir, exist_ok=True); fr = freeze_results(results_path, out_dir)
    print('FREEZE:', json.dumps({k: fr[k] for k in ('sha256', 'n_rows', 'runs_per_seed', 'planned_per_seed', 'complete')}))
    if not fr['complete']: print('STOPPED: the results file is not complete (see runs_per_seed vs planned_per_seed). Finish all seeds, then run this cell again. Nothing was frozen.'); return None
    df = load_final(results_path); out = dict(freeze=fr)
    print('\n--- pre-registered claims L2-C1 / L2-C2 (exploratory seeds 0-1 vs confirmatory seeds 2-4) ---'); rep = replication_table(df); print(rep.round(2).to_string(index=False)); rep.to_csv(os.path.join(out_dir, 'replication_exploratory_vs_confirmatory.csv'), index=False); out['replication'] = rep
    print('\n--- L2-C3 steps-matched augmentation contrast (Argentina; no direction assumed) ---'); c3 = c3_aug_contrast(df); print(c3.round(2).to_string(index=False) if len(c3) else '(no aug_on / aug_copy runs)'); c3.to_csv(os.path.join(out_dir, 'c3_aug_vs_copy.csv'), index=False); out['c3'] = c3
    print('\n--- mixed model: own REML code vs statsmodels (tolerance declared in advance: coef %.2f, SE %.2f) ---' % (CLOSING_TOL['coef'], CLOSING_TOL['se']))
    cmp_ = own_vs_statsmodels(df); print(cmp_.round(3).to_string(index=False)); cmp_.to_csv(os.path.join(out_dir, 'mixed_model_own_vs_statsmodels.csv'), index=False); out['cross'] = cmp_
    print('\n--- robustness: gain from 1 to 8 regions (points), Argentina, same cells: plain vs steps-matched copy vs augmentation ---')
    rb, dep = robustness_table(df); print(rb.round(2).to_string(index=False)); rb.to_csv(os.path.join(out_dir, 'robustness_more_regions_vs_training.csv'), index=False); out['robustness'] = rb; out['dep'] = dep
    print('copy-control improvement over plain training vs number of regions: slope %.2f points per doubling, p = %.3f (%d combinations)' % (dep['slope'], dep['p'], dep['n_combinations']))
    print('\nCHN is DESCRIPTIVE (rice exists in two regions only); its rows above carry the verdict "descriptive only (no claim)".')
    print('\n=== EXPLORATORY (requested in the instructor review; NOT pre-registered claims; not used for any verdict above) ===')
    mv = mix_vs_own_table(df); out['mix_vs_own'] = mv
    if len(mv): print('--- every region mix vs its OWN single regions at the same budget (mixes share regions: p-values are descriptive) ---'); print(mv.round(4).to_string(index=False)); mv.to_csv(os.path.join(out_dir, 'exploratory_mix_vs_own_regions.csv'), index=False)
    fp = pd.concat([fixed_pool_budget_effect(df, t) for t in ('ARG', 'USA')], ignore_index=True)
    out['fixed_pool'] = fp
    if len(fp): print('\n--- budget effect on single regions with the pool held FIXED (regions available at every budget); use this figure, not the change in the all-available average ---'); print(fp.round(2).to_string(index=False)); fp.to_csv(os.path.join(out_dir, 'exploratory_fixed_pool_budget_effect.csv'), index=False)
    pd_ = per_doubling_subset(df, 'ARG'); out['per_doubling_subset'] = pd_
    if len(pd_): print('\n--- per-doubling slope with and without regions 75/76 (exploratory; motivates the separate follow-up plan, which must be pre-registered and run on NEW seeds) ---'); print(pd_.round(3).to_string(index=False)); pd_.to_csv(os.path.join(out_dir, 'exploratory_per_doubling_with_without_75_76.csv'), index=False)
    print('\nUSA test-set ordering verification:', globals().get('USA_ORDER_STATUS', 'NOT RUN in this session (run the ordering-verification cell; the USA verdicts must not be submitted without it)'))
    print('USA scoring rule in force:', USA_SCORING, '| true classes removed:', USA_DROP_TRUE_CLASSES)
    print('Note: Argentina verdicts above are the pre-registered ones and stay as printed; the exploratory tables never change them.')
    return out

def _selftest_closing():
    # runs every time this cell is executed: the claim code must recover a KNOWN effect, and must say NOT REPLICATED for a sign flip and NOT TESTABLE without an interval
    r = np.random.default_rng(1); rows = []
    for seed in range(5):
        for ci in range(10):
            cell = ('11', '24', ci)
            base = 50 + r.normal(0, 1)
            rows.append(dict(target='ARG', budget=1000, k=2, states=str(cell), seed=seed, track='aug_copy', y=base))
            rows.append(dict(target='ARG', budget=1000, k=2, states=str(cell), seed=seed, track='aug_on', y=base + 2.0 + r.normal(0, .3)))                       # known effect +2
            rows.append(dict(target='ARG', budget=2000, k=2, states=str(cell), seed=seed, track='aug_copy', y=base))
            rows.append(dict(target='ARG', budget=2000, k=2, states=str(cell), seed=seed, track='aug_on', y=base + (1.0 if seed < 2 else -1.0) + r.normal(0, .1)))   # sign flips between groups
    t = c3_aug_contrast(pd.DataFrame(rows)); v = t.groupby('budget')['verdict'].first().to_dict()
    assert abs(t[(t.budget == 1000) & (t.group == 'confirmatory')]['estimate'].iloc[0] - 2.0) < 0.2, 'C3 contrast does not recover the known +2 effect'
    assert v[1000].startswith('REPLICATED') and v[2000].startswith('NOT REPLICATED (sign differs)'), v
    print('closing-analysis self-test passed: known +2 effect recovered (verdict:', v[1000] + '); sign flip ->', v[2000] + ')')
_selftest_closing()

def _selftest_exploratory():
    r = np.random.default_rng(5); rows = []
    for seed in range(3):
        for b in (1000, 2000):
            for reg, base in [('11', 50), ('24', 60), ('75', 70)]: rows.append(dict(target='ARG', budget=b, k=1, states=reg, seed=seed, track='base', y=base + (3 if b == 2000 else 0) + r.normal(0, .1)))
            for combo, v in [('11-24', 58), ('11-75', 63), ('24-75', 65)]: rows.append(dict(target='ARG', budget=b, k=2, states=combo, seed=seed, track='base', y=v + (3 if b == 2000 else 0) + r.normal(0, .1)))
    d = pd.DataFrame(rows); m = mix_vs_own(d, 'ARG'); assert len(m) == 6
    x = m[(m.states == '11-24') & (m.budget == 1000)].iloc[0]; assert abs(x.own_mean - 55) < .2 and abs(x.own_best - 60) < .2 and abs((x.y - x.own_mean) - 3) < .3
    t = mix_vs_own_table(d, ('ARG',), n_boot=200); a = t[(t.subset == 'all mixes') & (t.reference == 'mean of own single regions')].iloc[0]
    assert a.n == 6 and np.isfinite(a.mean_gap) and 0 <= a.share_better <= 1
    assert (t[t.subset == 'containing 75/76'].n == 4).all() and (t[t.subset == 'without 75/76'].n == 2).all()
    fp = fixed_pool_budget_effect(d); assert abs(fp.fixed_pool_change_from_lowest_budget.iloc[-1] - 3) < .2
    d2 = d[~((d.k == 1) & (d.states == '75') & (d.budget == 1000))]; fp2 = fixed_pool_budget_effect(d2); assert fp2.fixed_pool_regions.iloc[0] == 2           # a region missing at one budget is excluded from the fixed pool
    print('exploratory-analysis self-test passed (mix vs own mean/best, 75/76 split, fixed pool)')
_selftest_exploratory()

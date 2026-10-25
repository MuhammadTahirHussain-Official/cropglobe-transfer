# ======================================================================================
# LEVEL 1 CLOSING ANALYSIS (read-only).  Uses the FROZEN results.csv; nothing here trains a model or writes to the results file.
# Adds what the instructor's second review asked for: (1) a statsmodels cross-check of the mixed model, (2) the training-quality robustness table,
# (3) a freeze check (SHA-256) so nobody can later claim the numbers were changed.  Needs c02_config + c04_core cells (same code that produced the Level 1 tables).
# ======================================================================================
import hashlib, json, os
import numpy as np, pandas as pd
from scipy import stats as scipy_stats

FREEZE = dict(                                   # recorded on 2026-10-21 from the file the report was written from
    results_file='results.csv (final, 3160 runs)',
    results_sha256='38d1876cde08e54cfc827aa6b2813355ef98a09dbd048885a67d58991534f891',
    n_rows=3160, runs_per_seed={0: 725, 1: 725, 2: 570, 3: 570, 4: 570})
CLOSING_TOL = dict(coef=0.02, se=0.02)           # statsmodels vs own REML must agree this closely, declared before the cross-check was run
N_BOOT_CLOSING = 2000

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''): h.update(b)
    return h.hexdigest()

def verify_freeze(path, freeze=None):
    # STOP unless the file is byte-identical to the one the report was written from, and has exactly the planned number of runs per seed
    freeze = freeze or FREEZE; got = sha256_file(path); ok_hash = (got == freeze['results_sha256'])
    d = pd.read_csv(path); ok_n = (len(d) == freeze['n_rows']); per = d.groupby(pd.to_numeric(d['seed'], errors='coerce')).size().to_dict(); ok_seed = {int(k): int(v) for k, v in per.items()} == freeze['runs_per_seed']
    dup = int(d['run_id'].duplicated().sum()); bad = int((d['status'] != 'ok').sum())
    res = dict(ok=bool(ok_hash and ok_n and ok_seed and dup == 0 and bad == 0), sha256=got, hash_matches=ok_hash, rows=len(d), rows_ok=ok_n, per_seed=per, per_seed_ok=ok_seed, duplicates=dup, non_ok=bad)
    return res

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

def own_vs_statsmodels(df, targets=('ARG', 'CHN', 'FRA'), try_install=True):
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

def closing_report(results_path, out_dir):
    os.makedirs(out_dir, exist_ok=True); v = verify_freeze(results_path)
    print('FREEZE CHECK:', 'OK -- byte-identical to the file the report was written from' if v['ok'] else f'!! DIFFERENT: {v}')
    if not v['ok']: print('STOPPED: this is not the frozen results file.  Do not use these numbers in the report.'); return None
    df = load_final(results_path)
    print('\n--- mixed model: own REML code vs statsmodels (same model; tolerance declared in advance: coef %.2f, SE %.2f) ---' % (CLOSING_TOL['coef'], CLOSING_TOL['se']))
    cmp_ = own_vs_statsmodels(df); print(cmp_.round(3).to_string(index=False)); cmp_.to_csv(os.path.join(out_dir, 'mixed_model_own_vs_statsmodels.csv'), index=False)
    print('\n--- robustness: gain from 1 to 8 states (points), Argentina, same cells ---')
    rb, dep = robustness_table(df); print(rb.round(2).to_string(index=False)); rb.to_csv(os.path.join(out_dir, 'robustness_more_states_vs_training.csv'), index=False)
    print('copy-control improvement over plain training vs number of states: slope %.2f points per doubling, p = %.3f (%d combinations)' % (dep['slope'], dep['p'], dep['n_combinations']))
    print('\n--- pre-declared claims (all five seeds available) ---'); rep = replication_table(df); print(rep.round(2).to_string(index=False)); rep.to_csv(os.path.join(out_dir, 'replication_exploratory_vs_confirmatory.csv'), index=False)
    return dict(freeze=v, cross=cmp_, robustness=rb, dep=dep, replication=rep)

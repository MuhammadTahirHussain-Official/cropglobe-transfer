# ======================================================================================
# FOLLOW-UP CLOSING ANALYSIS (read-only).  Run ONLY after all five follow-up seeds (5-9) are finished.  Evaluates exactly the two claims declared in the plan cell (FOLLOWUP_CLAIMS); nothing else is a claim.
# Verdict words are produced by code from the numbers.  Requires the analysis functions of the Level 2 closing cell (same notebook, cell above).
# ======================================================================================
def followup_claims(df, claims=None):
    claims = claims or FOLLOWUP_CLAIMS; c2 = claims['FC2']; out = {}
    b = _random_arm_runs(df); b = b[b['target'] == 'ARG']
    g, X, names = _mixed_design(b); r = reml_random_intercept(g['y'].values, X, g['combo'].values); est = float(r['beta'][1]); se = float(np.sqrt(r['cov'][1, 1])); lo, hi = est - 1.96 * se, est + 1.96 * se
    out['FC1'] = dict(claim='F-C1 per-doubling effect of log2(k)', n_runs=len(g), n_combinations=int(r['G']), estimate=est, se=se, lo=lo, hi=hi,
                      verdict='PASS' if (est > 0 and lo > 0) else ('NOT REPLICATED (estimate > 0 but the 95% interval includes 0)' if est > 0 else 'NOT REPLICATED (estimate <= 0)'))
    m = mix_vs_own(df, 'ARG'); gap = (m['y'] - m['own_mean']).values; s_ = _gap_summary(gap, c2['n_boot'], c2['boot_seed'])
    per_seed = {}
    for sd in sorted(df['seed'].unique()):
        ms = mix_vs_own(df[df['seed'] == sd], 'ARG')
        if len(ms): per_seed[int(sd)] = float((ms['y'] - ms['own_mean']).mean())
    n_pos = sum(1 for v in per_seed.values() if v > 0); ok = (s_['mean_gap'] > 0) and (s_['lo'] > 0) and (n_pos >= c2['MIN_POSITIVE_SEEDS'])
    why = [] if ok else [x for x, bad in [('mean gap <= 0', not s_['mean_gap'] > 0), ('95% interval includes 0', not s_['lo'] > 0), (f'positive in only {n_pos} of {len(per_seed)} seeds (need {c2["MIN_POSITIVE_SEEDS"]})', n_pos < c2['MIN_POSITIVE_SEEDS'])] if bad]
    out['FC2'] = dict(claim='F-C2 mix vs mean of its own single regions', n_mixes=s_['n'], mean_gap=s_['mean_gap'], lo=s_['lo'], hi=s_['hi'], share_better=s_['share_better'], seeds_positive=f'{n_pos} of {len(per_seed)}', per_seed_gap=per_seed,
                      verdict='PASS' if ok else 'NOT REPLICATED (' + '; '.join(why) + ')')
    mb = mix_vs_own_table(df, ('ARG',)); out['descriptive_vs_best'] = mb[(mb.subset == 'all mixes') & (mb.reference == 'best own single region')]
    return out

def closing_report_followup(results_path, out_dir):
    os.makedirs(out_dir, exist_ok=True); fr = freeze_results(results_path, out_dir)
    print('FREEZE:', json.dumps({k: fr[k] for k in ('sha256', 'n_rows', 'runs_per_seed', 'planned_per_seed', 'complete')}))
    if not fr['complete']: print('STOPPED: the follow-up results are not complete. Finish seeds 5-9, then run this cell again. Nothing was frozen.'); return None
    df = load_final(results_path); res = followup_claims(df)
    print('\n=== PRE-REGISTERED FOLLOW-UP CLAIMS (Argentina, regions 75/76 removed, seeds 5-9, all confirmatory) ===')
    f1, f2 = res['FC1'], res['FC2']
    print(f"{f1['claim']}: {f1['estimate']:+.2f} points per doubling (SE {f1['se']:.2f}, 95% CI {f1['lo']:+.2f} .. {f1['hi']:+.2f}; {f1['n_runs']} runs, {f1['n_combinations']} combinations) -> {f1['verdict']}")
    print(f"{f2['claim']}: mean gap {f2['mean_gap']:+.2f} points (95% CI {f2['lo']:+.2f} .. {f2['hi']:+.2f}; {f2['n_mixes']} mixes; {100 * f2['share_better']:.0f}% better; positive in {f2['seeds_positive']} seeds) -> {f2['verdict']}")
    print('   per-seed mean gap:', {k: round(v, 2) for k, v in f2['per_seed_gap'].items()})
    print('descriptive only (NOT a claim) - mix vs the BEST of its own single regions:'); print(res['descriptive_vs_best'].round(3).to_string(index=False))
    print('\nCaveat: mixes share regions and single-region runs, so the intervals describe these 11 French regions; they do not describe new regions.  The Level 2 verdicts are not changed by this follow-up.')
    pd.DataFrame([{k: v for k, v in f1.items() if k != 'claim'} | dict(claim=f1['claim']), {k: v for k, v in f2.items() if k not in ('claim', 'per_seed_gap')} | dict(claim=f2['claim'])]).to_csv(os.path.join(out_dir, 'followup_claims.csv'), index=False)
    return res

def _selftest_followup():
    r = np.random.default_rng(2); rows = []
    for seed in range(5, 10):
        for b in (1000, 2000):
            base = {'11': 50, '24': 55, '27': 60}
            for reg, v in base.items(): rows.append(dict(target='ARG', budget=b, k=1, states=reg, seed=seed, track='base', y=v + r.normal(0, .3), aug='off', sample_sel='random', arms='random', combo=reg, main_f1=0))
            for combo in ('11-24', '11-27', '24-27'):
                m_ = np.mean([base[x] for x in combo.split('-')]); rows.append(dict(target='ARG', budget=b, k=2, states=combo, seed=seed, track='base', y=m_ + 2.0 + r.normal(0, .3), aug='off', sample_sel='random', arms='random', combo=combo, main_f1=0))
    d = pd.DataFrame(rows); d['main_f1'] = d['y'] / 100
    f2 = mix_vs_own(d, 'ARG'); gap = (f2['y'] - f2['own_mean']).mean(); assert abs(gap - 2.0) < .3, gap
    pf = {s: (mix_vs_own(d[d.seed == s], 'ARG').eval('y - own_mean')).mean() for s in range(5, 10)}; assert all(v > 0 for v in pf.values())
    d0 = d.copy(); d0.loc[d0.k == 2, 'y'] = d0.loc[d0.k == 2, 'y'] - 2.0                                                        # no gain: gap ~ 0 -> must NOT pass
    s0 = _gap_summary((mix_vs_own(d0, 'ARG')['y'] - mix_vs_own(d0, 'ARG')['own_mean']).values, 500, 0); assert s0['lo'] < 0 < s0['hi'], s0
    print('follow-up claim self-test passed: known +2.0 gap recovered (%.2f); a zero gap has an interval containing 0 (%.2f..%.2f)' % (gap, s0['lo'], s0['hi']))
_selftest_followup()

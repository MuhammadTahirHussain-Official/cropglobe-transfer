# ======================================================================================
# PLAN TABLE (nothing is trained here).  READ IT, then set APPROVED['plan'] = True in the config cell.
# ======================================================================================
assert TESTS_PASSED, 'the test gate (Part A + Part B) must pass first'
assert 'GEOGRAPHY_OBSERVED' in globals(), 'run the load cell (it contains the source-region geography check) first'
SRC_UNIT    = meta['state'].values.astype(str); SRC_CROP = meta['crop'].values.astype(str); SRC_CROP_ID = meta['crop_id'].values.astype(int)       # 'state' column = French REGION code
SRC_NPZ_ROW = meta['npz_row'].values.astype(np.int64)                                   # row index in the ORIGINAL (pre-dedup) France npz -> saved as sample ids
TARGET_IDS  = {t: INFO[t]['keep'] for t in TARGETS_TO_RUN}                              # original npz row of every kept target sample
POOLS = build_pools(SRC_UNIT, SRC_CROP); CAP = make_cap(SRC_UNIT, SRC_CROP)
EXCLUDED_SOURCE_UNITS = globals().get('EXCLUDED_SOURCE_UNITS', [])
if EXCLUDED_SOURCE_UNITS:                                                                   # FOLLOW-UP only: the removed source regions are taken out of the capacity table AND the row pools, so no plan cell can contain them and no row of them can be drawn
    _before = len(CAP); CAP = CAP.drop(index=[u_ for u_ in EXCLUDED_SOURCE_UNITS if u_ in CAP.index]); POOLS = {k_: v_ for k_, v_ in POOLS.items() if k_[0] not in EXCLUDED_SOURCE_UNITS}
    _check(len(CAP) == _before - len(EXCLUDED_SOURCE_UNITS) and not any(k_[0] in EXCLUDED_SOURCE_UNITS for k_ in POOLS), 'excluded source regions are still present in the capacity table / pools')
    print(f'FOLLOW-UP: source regions {EXCLUDED_SOURCE_UNITS} removed from the pool ({_before} -> {len(CAP)} regions); their rows can never be drawn.')
os.makedirs(os.path.join(OUT_DIR, 'tables'), exist_ok=True)
print('feature source check:', FEATURE_SOURCE)
print('\nREGION-ASSIGNMENT EVIDENCE (France; exact row-by-row match, see FRA_join_verification.md):'); print('  ', {k: v for k, v in FRA_META_REPORT.items() if k not in ('excluded_rows', 'rows_per_region')}); print('   rows per region (assigned):', FRA_META_REPORT['rows_per_region'])
CAP.sort_values('total', ascending=False).to_csv(os.path.join(OUT_DIR, 'tables', 'state_crop_table.csv'))
print(f'{SOURCE} regions: {len(CAP)} | regions with >= {MIN_STATE_ROWS} rows: {int((CAP.total >= MIN_STATE_ROWS).sum())}')
print(CAP.sort_values('total', ascending=False).head(25).to_string())

# ---- PHASE 4: phenology-spectral diversity of the USA source states (USA features only; no target data, labels or scores are used)
DIVERSITY_DIST = compute_diversity_distances(FEATURES[SOURCE], SRC_UNIT, SRC_CROP) if any(a in SELECTION_ARMS for a in ('diverse', 'clustered')) else None
if 'kcenter' in SELECTION_ARMS:
    KC_EMBED = build_kc_embedding(FEATURES[SOURCE]); _check(KC_EMBED.shape == (len(SRC_UNIT), KC_DIMS) and np.isfinite(KC_EMBED).all(), 'k-center embedding has the wrong shape / non-finite values')
    print(f'k-center embedding: USA window features (10 bands x 43 steps) -> standardised -> PCA {KC_DIMS} dims, shape {KC_EMBED.shape} (USA rows only; no target data is used)')
if DIVERSITY_DIST is not None:
    for _c, _D in DIVERSITY_DIST['dist'].items(): _D.to_csv(os.path.join(OUT_DIR, 'tables', f'diversity_distance_{_c}.csv'))
    print('Phase 4 diversity space:', DIVERSITY_SPACE); print('arms:', SELECTION_ARMS, '| pool rule:', POOL_RULE, '| N per arm:', N_ARM)
PLAN_CELLS, PLAN_GRID = build_plan(CAP)
PLAN_CELLS_RUN = [c for c in PLAN_CELLS if not c['pending']]
PLAN_GRID.to_csv(os.path.join(OUT_DIR, 'tables', 'feasibility_grid.csv'), index=False)
pd.DataFrame([dict(c, states='-'.join(c['states'])) for c in PLAN_CELLS]).to_csv(os.path.join(OUT_DIR, 'tables', 'plan_cells.csv'), index=False)

# ---- COMPLETE state-by-crop table is saved above; target-specific ELIGIBILITY tables (every state, with the reason for every exclusion) are saved per target/budget/mode
def _eligibility_report(mode):
    for t in TARGETS_TO_RUN:
        for B in BUDGETS:
            e = state_eligibility(t, B, CAP, mode); e.to_csv(os.path.join(OUT_DIR, 'tables', f'eligibility_{mode}_{t}_B{B}.csv'), index=False)
            bad = e[~e.eligible]; why = pd.Series([r.split(' ')[0] if r else '' for rs in bad.reason for r in rs.split('; ')]).value_counts().to_dict() if len(bad) else {}
            print(f'  [{mode}] {t} B={B}: {int(e.eligible.sum())}/{len(e)} states eligible; excluded {len(bad)} (reasons: {why or "-"}) -> tables/eligibility_{mode}_{t}_B{B}.csv')
print('\n================ ELIGIBILITY (every state is listed in the CSVs; nothing is excluded silently) ================')
for _m in COMPOSITION_MODES_TO_RUN:
    if _m == BASE_MODE or FIXED_COMPOSITION_CHOICE in FIXED_COMPOSITION_PROPOSALS: _eligibility_report(_m)
print('\n================ AUTHORS\' SCRIPT vs THIS NOTEBOOK (from run_CropNet_AUGM.py; deviations are deliberate and listed) ================')
print('  authors:', {k: v for k, v in AUTHORS_SCRIPT.items()}); print('  deviations:'); [print(f'    - {k}: {v}') for k, v in DEVIATIONS_FROM_AUTHORS.items()]
print(f'  this notebook: Adam lr={LR}, weight_decay={WEIGHT_DECAY}, batch {BATCH_SIZE}, no scheduler, TOTAL epochs {MAX_EPOCHS} with early stopping (patience {PATIENCE}, min {MIN_EPOCHS}, strictly greater, best weights restored), class weights from {CLASS_WEIGHT_SOURCE}')
print('  optimizer steps per epoch = ceil(train/256): ' + ' | '.join(f'B={B}: {steps_per_epoch(B - int(VAL_FRACTION * B))} (<= {MAX_EPOCHS * steps_per_epoch(B - int(VAL_FRACTION * B))} total)' for B in BUDGETS) + '  -> budget is confounded with training length; steps are reported per run')
print('\n================ REMAP RULES: where the three named rules differ (all three are scored, separately) ================')
print('  NOTE:', PLAN_NOTE_ARG)
print(remap_difference_table().to_string(index=False))
print('   existing_release = EXISTING (preserved) | paper_text_v3 = literal paper sec.4.4 text | review_targetmap = REVIEW-PROPOSED.  Part B compared all three with the authors\' released function (tables/authors_remap_comparison.csv). No rule is declared "the paper\'s rule".')

print('\n================ PLAN TABLE ================')
print(f'composition modes {COMPOSITION_MODES_TO_RUN} | state selector {STATE_SELECTOR} | epochs <= {MAX_EPOCHS} | class weights from {CLASS_WEIGHT_SOURCE} labels | feature source: {FEATURE_SOURCE}')
print(f'budgets {BUDGETS} | k {K_LIST} | seeds {SEEDS} | aug {AUG} | max {MAX_COMBOS} combos for k>=2 (k=1 census) | infeasible if <{MIN_COMBOS} combos or <{MIN_KPOINTS} k-points')
print('key-crop shares (fixed at every k; the rest sampled naturally):', {t: effective_key_shares(t) for t in TARGETS_TO_RUN})
for _m, t in [(m, t) for m in COMPOSITION_MODES_TO_RUN if (m == BASE_MODE or FIXED_COMPOSITION_CHOICE in FIXED_COMPOSITION_PROPOSALS) for t in TARGETS_TO_RUN]:
    print(f'\n--- [{_m}] {t} ({TARGET_ROLE[t]}; primary view: {PRIMARY_VIEW[t][0]}{"_" + PRIMARY_VIEW[t][1] if PRIMARY_VIEW[t][1] else " (existing release remap)"}; K_MAX={K_MAX[t]}) ---')
    g = PLAN_GRID[(PLAN_GRID.target == t) & (PLAN_GRID.composition == _m)]
    for B in BUDGETS:
        gb = g[g.budget == B]; kc = key_counts(t, B)
        print(f'  B={B}: key counts {kc if kc else "none (fully natural)"} + natural rest {B - sum(kc.values())} | pool = {int(gb.pool_size.iloc[0])} state(s): {gb.pool.iloc[0]}')
        print('      ' + ' | '.join(f'k={int(r.k)}: ' + (f'{int(r.n_combos)} combo(s) of {int(r.n_total_combos)}' + (' [census]' if r.k == 1 else '') if r.feasible else f'INFEASIBLE ({r.note.strip()})') for r in gb.itertuples()))
    _cm = [c for c in PLAN_CELLS if c['target'] == t and c['composition'] == _m]; n_runs = sum(len([c for c in plan_for_seed(s_) if c['target'] == t and c['composition'] == _m]) for s_ in SEEDS)
    print(f'  -> {len(_cm)} cells x {len(SEEDS)} seeds = {len(_cm) * len(SEEDS)} runs' + ('   *** EXCLUDED UNTIL APPROVED/CONFIRMED ***' if _cm and _cm[0]['pending'] else f'   ({n_runs} will run)'))
TOTAL_RUNS = sum(len(plan_for_seed(s_)) for s_ in SEEDS)
print(f'\nTOTAL training runs that will execute: {TOTAL_RUNS}  ({len(PLAN_CELLS_RUN)} combination-cells for seeds {KC_SEEDS}; seeds {[s_ for s_ in SEEDS if s_ not in KC_SEEDS]} skip the {sum(1 for c in PLAN_CELLS_RUN if c.get("sel") == "kcenter")} k-center cells; arms share combinations, so a combination is trained once); GPU time is estimated after the smoke test.')
_check(len(PLAN_CELLS_RUN) > 0, 'EMPTY PLAN: no cell is feasible -- see the tables above')
print('\n================ v3.7 ADDITIONS (listed in CHANGELOG_v3_7.md) ================')
print(f'  epochs: TOTAL cap {MAX_EPOCHS} (was 100: 38% of the 8-state runs hit the old cap) | results folder: {OUT_DIR}')
print(f'  MAIN metric: ARG/CHN = binary key-crop macro-F1 (primary_f1); USA (multi-class) = {"coverage-conditional macro-F1 (classes with >=" + str(COVERAGE_MIN_TRAIN) + " training samples; sugarcane and cotton do not exist in the French source)" if FRA_MAIN_RULE == "coverage_conditional" else "all-class macro-F1 (NO rule)"}.')
print(f'  SECONDARY metric: paper-style macro-F1 (read it next to the coverage columns: each absent crop costs ~11 points). All other score columns are stored but not summarised (REPORT_ALL_TRACKS={REPORT_ALL_TRACKS}).')
print('  input features recorded in every row:', INPUT_FEATURES)
_V = pd.DataFrame([dict(arm=a, target=c['target'], budget=c['budget'], pending=c['pending'], shared=('random' in c['arms'] and a != 'random')) for c in PLAN_CELLS for a in c['arms'] if a in ('common', 'kcenter', 'aug_on')])
if len(_V):
    print('  NEW TRACKS (cells; each cell x seeds = runs; "shared" = the run is also in the random arm and is trained once):')
    print(_V.groupby(['arm', 'target', 'budget']).agg(cells=('arm', 'size'), shared=('shared', 'sum'), pending=('pending', 'sum')).to_string())
    if 'common' in SELECTION_ARMS: print('  common = combinations re-drawn from the states eligible at EVERY budget of the target (fixed pool), so budgets can be compared on the same states; k above the common-pool size is not available.')
    if 'kcenter' in SELECTION_ARMS: print(f'  kcenter = same state combinations (k=1 census + up to {KC_COMBOS} per cell) and identical per-state/per-crop counts as the random arm; only WHICH rows are taken changes (greedy farthest-first in a {KC_DIMS}-dim PCA space, per state and crop).')
    if 'aug_on' in SELECTION_ARMS: print(f'  aug_on = authors\' augmentation (time-shift p=1, time-scale p=.5, magnitude-warp p=.25; original + {AUG_REPEAT} augmented copies) on the same combinations; targets {AUG_TARGETS}, budgets {AUG_BUDGETS}.  PENDING until APPROVED["aug_on"] = True.  It uses {1 + AUG_REPEAT}x the optimizer steps per epoch; compare with aug-OFF knowing this.')
_n_new = [c for c in PLAN_CELLS_RUN if c.get('sel', 'random') != 'random' or c.get('aug', 'off') != 'off' or c['arms'] == ('common',)]
print(f'  runs that exist only because of the new tracks: {sum(len(plan_for_seed(s_, _n_new)) for s_ in SEEDS)} ({len(_n_new)} cells; k-center only for seeds {KC_SEEDS}); drop an arm from SELECTION_ARMS in the config cell to remove it.')
if AUG_CONTROL and 'aug_on' in SELECTION_ARMS: print(f"  aug_copy (v3.8) = steps-matched control for aug_on: the SAME cells, the training rows copied {1 + AUG_REPEAT}x with NO transform (same rows per epoch and optimizer steps, no augmentation).  The decisive comparison is aug_on - aug_copy.")
DESIGN_HASH = design_hash()
atomic_write(os.path.join(OUT_DIR, 'tables', 'design_hash.txt'), lambda f: f.write(DESIGN_HASH + '\n'), mode='w')
print('\n================ INSTRUCTOR APPROVAL ================')
print('DESIGN HASH (locked settings + every planned run id of every seed):', DESIGN_HASH)
print('To run: after the instructor has signed the decisions, set INSTRUCTOR_APPROVED = True and APPROVED_DESIGN_SHA256 = this hash in the config cell, and re-run the config cell.  The run cell refuses any other plan.')
print('   Nothing has been trained.')
if PROFILE == 'extension':
    print('\n================ EXTENSION PLAN (appendix): CHN / FRA at budgets %s ================' % BUDGETS)
    print('  Allocation: each fixed crop and the natural remainder are split over the k states AS EQUALLY AS EACH STATE\'S CAPACITY ALLOWS (a short state gives all it has, the others make up the difference).')
    print(f'  A state joins a combination only if it can give at least {EXT_MIN_CONTRIB:.0%} of its equal share of every fixed quota and of the remainder; the combination as a whole must hold every quota.  k=1 is the census of states that can supply the whole budget alone.')
    _g = PLAN_GRID[PLAN_GRID.feasible].copy(); _g['max_share_worst'] = _g['max_share']
    print(_g[['target', 'budget', 'k', 'pool_size', 'n_total_combos', 'n_combos', 'max_share_worst']].to_string(index=False))
    print('  max_share_worst = the largest planned share of the budget supplied by ONE state in any combination of the cell (1/k would be perfectly equal).')
    _skipped = PLAN_GRID[~PLAN_GRID.feasible][['target', 'budget', 'k', 'note']]
    if len(_skipped): print('  NOT planned (reported, never silently dropped):'); print(_skipped.to_string(index=False))
    print(f'  runs: {sum(len(plan_for_seed(s_)) for s_ in SEEDS)} ({len(PLAN_CELLS_RUN)} cells x {len(SEEDS)} seeds)')

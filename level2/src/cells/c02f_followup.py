# ======================================================================================
# FOLLOW-UP PLAN  "Argentina without regions 75 and 76"  -- NEW pre-registered plan, written down and frozen BEFORE any run (instructor review, Oct 2026).  Nothing here is a result.
# WHY: in the Level 2 data the French regions 75 and 76 are far better than the others as single regions for Argentina, and Argentine region mixes that contain them lose accuracy when
#      more regions are added.  Whether more regions help when those two are absent was seen only in exploratory form (Level 2 seeds 0-4).  This plan tests it on NEW seeds (5-9).
# SCOPE: Argentina only | source France, regions 75 and 76 REMOVED from the source pool (EXCLUDED_SOURCE_UNITS) | k = 1, 2, 4 | budgets 1000 and 2000 | seeds 5-9 (ALL confirmatory) | plain training only
#        (no augmentation, no k-center, no other arm) | same model, optimiser, early stopping, scoring and data as Level 2.  Level 2 results are never read or written by this notebook (own results folder).
# THE TWO CLAIMS, DECLARED IN ADVANCE (the verdict words are produced by code from the numbers; nobody edits them):
#   F-C1  per-doubling claim:  in the mixed model  F1 ~ log2(k) + budget/1000 + seed + (1 | region combination)  fitted to all runs of seeds 5-9, the log2(k) coefficient is > 0 AND its 95% interval excludes 0.
#   F-C2  mix-vs-own-regions claim: for every region mix (k = 2, 4) the seed-averaged F1 minus the mean of the seed-averaged F1 of ITS OWN single regions at the SAME budget;
#         the mean of these gaps is > 0, its 95% bootstrap interval (mixes resampled) excludes 0, AND the mean gap is > 0 in at least 4 of the 5 seeds.
#         (the "at least 4 of 5 seeds" requirement is a PROPOSAL made by the analyst and needs the instructor's confirmation; it is part of the design hash, so it cannot change after approval.)
#   Anything else is reported as NOT REPLICATED.  "Mix vs the BEST of its own regions" is reported descriptively, it is NOT a claim.  No other comparison is a claim.
#   Mixes share regions, so the intervals describe these 11 French regions, not new regions; the report says so.
# ======================================================================================
FOLLOWUP = True
EXPERIMENT_VERSION = 'phase3_level2_followup_no75_76_v1'
SPEC_TAG = 'level2_followup_no75_76_prereg_v1'
EXCLUDED_SOURCE_UNITS = ['75', '76']
TARGETS_TO_RUN, BUDGETS, K_LIST, SEEDS = ['ARG'], [1000, 2000], [1, 2, 4], [5, 6, 7, 8, 9]
SELECTION_ARMS, AUG_TARGETS, AUG_BUDGETS, KC_TARGETS, CLAIM_TARGETS = ['random'], [], [], [], ['ARG']
KC_SEEDS = []
FOLLOWUP_CLAIMS = dict(
    FC1=dict(model='F1 ~ log2(k) + budget/1000 + seed dummies + (1 | region combination), REML, random-arm runs, seeds 5-9 pooled', pass_if='coefficient of log2(k) > 0 and the 95% interval excludes 0'),
    FC2=dict(quantity='mean over region mixes (k>=2) of [seed-averaged F1 of the mix - mean of the seed-averaged F1 of its own single regions, same budget]', pass_if='mean gap > 0, 95% bootstrap interval excludes 0, and mean gap > 0 in at least MIN_POSITIVE_SEEDS of the 5 seeds', MIN_POSITIVE_SEEDS=4, n_boot=2000, boot_seed=0),
    seed_roles='all of seeds 5-9 are confirmatory; there is no exploratory group', not_a_claim=['mix vs best own region', 'any k-by-budget cell', 'any other region subset'])
APPROVED.update({f'seed{s_}_gate_report': False for s_ in SEEDS})
if AUTO_MODE: APPROVED.update(plan=True)
LOCKED = LOCKED + [k_ for k_ in ('EXCLUDED_SOURCE_UNITS', 'FOLLOWUP', 'FOLLOWUP_CLAIMS') if k_ not in LOCKED]
INSTRUCTOR_APPROVED = False
APPROVED_DESIGN_SHA256 = ''
V38_AMENDMENTS = V38_AMENDMENTS + [dict(id='F-P1', item='follow-up plan', text='Separate pre-registered plan: Argentina only, regions 75 and 76 removed from the source pool, k in {1,2,4}, budgets {1000,2000}, seeds 5-9 (all confirmatory), plain training only; claims F-C1 and F-C2 declared before any run (FOLLOWUP_CLAIMS, part of the design hash).')]
print(f'FOLLOW-UP PLAN loaded: {EXPERIMENT_VERSION} | targets {TARGETS_TO_RUN} | budgets {BUDGETS} | k {K_LIST} | seeds {SEEDS} | source regions removed: {EXCLUDED_SOURCE_UNITS}')

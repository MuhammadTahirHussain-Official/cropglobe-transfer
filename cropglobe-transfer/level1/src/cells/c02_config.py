# ======================================================================================
# CONFIGURATION  (pure python; nothing here touches the network or the GPU)
# Level 1 - geographic diversity with the KEY-CROP share held fixed.  Implements the written spec EXACTLY.
# ======================================================================================
import os, sys, json, math, time, glob, re, csv, zlib, random, itertools, hashlib, datetime, pickle, warnings, traceback, subprocess, tempfile, io, contextlib
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats as scipy_stats
from scipy import optimize as scipy_opt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support

EXPERIMENT_VERSION = 'phase3_level1_v3_7_review'          # NEW results folder: v3.7 trains up to 200 epochs, so results from the 100-epoch cap (v3.2-v3.6) must never be mixed with these
SPEC_TAG = 'level1_fixedmix_spec_v3_dual_track_authors'          # written into the manifest; a folder without it is NOT ours -> the run stops instead of mixing

# ---------------- design (spec section 4) ----------------
BUDGETS      = [1000, 2000, 3000, 4000]       # TOTAL labelled samples per run (train + validation); never shrunk
K_LIST       = [1, 2, 4, 8]
SEEDS        = [0, 1, 2, 3, 4]
AUG          = 'off'                          # Level 1 = augmentation OFF only (spec 2).  ON needs a step-matched control + approval.
MAX_COMBOS   = 10                             # k>=2: up to 10 random combinations; k=1 is a census of EVERY pool state
MIN_COMBOS   = 5                              # a (target,budget,k) cell with fewer combinations is recorded as infeasible
MIN_KPOINTS  = 3                              # a (target,budget) with fewer feasible k-points is infeasible as a whole
MIN_STATE_ROWS = 1500                         # source pool = USA states with >= 1500 (de-duplicated) rows
VAL_FRACTION = 0.10                           # validation = 10% of the budget, stratified by crop, deterministic
TARGETS_ORDER = ['ARG', 'FRA', 'CHN', 'AUS']
TARGETS_TO_RUN = ['ARG', 'FRA', 'CHN', 'AUS']
TARGET_ROLE  = {'ARG': 'main', 'FRA': 'main', 'CHN': 'secondary', 'AUS': 'appendix'}
# KEY-CROP share (fixed, identical at every k); everything else is sampled NATURALLY from the state's remaining rows.
KEY_SHARES   = {'ARG': {'soybeans': 0.20}, 'CHN': {'rice': 0.20}, 'FRA': {'corn': 0.15, 'soybeans': 0.15, 'wheat': 0.15}, 'AUS': None}
K_MAX        = {'ARG': 8, 'FRA': 8, 'CHN': 4, 'AUS': 2}
FRA_K8_MIN_POOL = 10                          # FRA k=8 only if the pool has >= 10 states
AUS_STATES   = ['LA', 'FL']                   # AUS = appendix census of {LA, FL}, k <= 2, no sweep (exempt from MIN_COMBOS / MIN_KPOINTS)
# ---- composition modes.  'key_plus_natural' = EXISTING baseline (fixed key-crop share + natural remainder).
#      'fixed_full'        = REVIEW-PROPOSED (complete crop composition fixed at every budget and k).  The VALUES are a scientific choice: they are
#                            NOT set here.  Proposals are printed with their capacity checks in the plan cell; nothing runs until you approve one.
COMPOSITION_MODES_TO_RUN = ['key_plus_natural']
FIXED_COMPOSITION_CHOICE = None               # e.g. 'A_key_plus_equal_rest' | 'B_equal_over_target_classes'  (only after approval)
ALIGNMENT_ORDERING_OVERRIDE = None             # V33-06: None = STOP on an ambiguous state alignment; set to a candidate ordering name only after reading cache/alignment_report.json
NOTEBOOK_VERSION = 'v3.8'                       # display / run_environment only (not part of the locked design)
STATE_SELECTOR = 'random'                     # the baseline arm: random combinations (unchanged)
# ---- PHASE 4 (definition approved by the researcher): phenology-spectral diversity of the USA source states ------------------------------------------------------------------------------
SELECTION_ARMS = ['random', 'diverse', 'clustered', 'common', 'kcenter', 'aug_on']   # v3.7 adds common / kcenter / aug_on (see below); 'random' is the baseline and is mandatory; ['random'] reproduces the pre-Phase-4 plan exactly
POOL_RULE = 'strict'                          # 'strict' (as before: a pool state must supply the WHOLE budget alone) | 'equal_share' (OPT-IN, not approved yet: a state in a k-combination only supplies its 1/k share; the pool then depends on k)
# ---- v3.7 (supervisor review of the 409-run seed-0 results): reporting rules, new tracks.  Every item is listed in CHANGELOG_v3_7.md ------------------------------------------------------------
# R2 metrics: ONE main metric and ONE secondary metric are reported.  The other score columns are still computed and stored (audit trail) but are not summarised unless REPORT_ALL_TRACKS = True.
REPORT_PRIMARY, REPORT_SECONDARY, REPORT_ALL_TRACKS = 'main_f1', 'paper_f1', False
# R3 France: rice is never in the training mix (rice F1 = 0 in every run), so a rule is stated IN ADVANCE: FRA's main metric is the coverage-conditional macro-F1 (classes with >= COVERAGE_MIN_TRAIN training samples; the paper does the same for France->USA).
#    ARG / CHN main metric = binary key-crop macro-F1 (soybeans / rice vs. other) = the existing primary_f1.   'coverage_conditional' | 'primary' (= no rule; FRA scored on all classes)
FRA_MAIN_RULE = 'coverage_conditional'
# R4 which input features the CropNet runs use (written into every results row)
INPUT_FEATURES = 'Sentinel-2 5-day medians (CropGlobe for_CropNet/median_S2_*_month1-12_day5.npz; 10 bands x 73 steps; official DOY 121-334 window = 43 steps); AlphaEarth/Presto/Harmonic NOT used'
# R5 fixed common pool: the pool of USA states shrinks as the budget grows (ARG 16 -> 6), which confounds budget comparisons.  Arm 'common' re-draws the combinations of EVERY budget from the states eligible at ALL budgets of the target.
# R6 sample-level selection: arm 'kcenter' = same state combinations and same per-state/per-crop counts as 'random', but the rows inside each (state, crop) pool are chosen by greedy k-center (farthest-first) in a PCA space of the USA window features.
KC_TARGETS, KC_COMBOS, KC_DIMS, KC_MAX_CAND = ['ARG', 'CHN', 'FRA'], 5, 16, 20000     # k>=2: at most KC_COMBOS of the random-arm combinations (deterministic), k=1: the whole census
# R7 augmentation ON (authors' CropGlobeDataset aug_mode='augm', pinned commit; training set = original + AUG_REPEAT augmented copies; validation/test never augmented).  Arm 'aug_on' = the same combinations as the random arm.
AUG_TARGETS, AUG_BUDGETS, AUG_COMBOS, AUG_REPEAT = ['ARG'], [1000, 2000], 5, 2
# ---- v3.8 (instructor review of the 1,475-run file; seeds 0 and 1 complete).  These keys are deliberately NOT in LOCKED: the locked design of the runs already on disk is unchanged, so the same results.csv resumes.
#      Every item is written into manifest.json as a dated AMENDMENT (V38_AMENDMENTS) the first time the run cell executes, so nothing about the design changes silently.
KC_SEEDS = [0, 1]          # k-center sample selection runs for these seeds only: seeds 0-1 gave a clean null (-0.47 points over 310 paired runs).  Seeds >= 2 skip the k-center cells.  Set KC_SEEDS = SEEDS to restore every seed.
AUG_CONTROL = True         # steps-matched control for augmentation.  Arm 'aug_copy' = the SAME cells as 'aug_on', but the training rows are COPIED (1 + AUG_REPEAT)x WITHOUT any transform: same rows per epoch, same optimizer steps, no augmentation.
PREREG_TEXT = ("Exploratory seeds = 0 and 1 (all earlier findings came from them).  Confirmatory seeds = 2, 3, 4 (analysed separately, after they finish).  "
    "Claims: C1 Argentina random arm, mixed-model effect of log2(k) on main F1 is > 0 with a 95% interval excluding 0 in the confirmatory seeds; "
    "C2 Argentina gain from the fewest to the most states is > 0 at each budget (interval excluding 0); "
    "C3 steps-matched augmentation: aug_on - aug_copy (paired on the same combinations, all seeds that have both) is reported with its interval, no direction assumed; "
    "C4 k-center - random is reported for seeds 0-1 as a null/finding; C5 'beats the best single state' share and worst-case gain are reported descriptively; China and France are descriptive only.  "
    "A claim passes only if it has the same sign in both seed groups and its interval excludes 0 in the confirmatory group.  A failed claim is REPORTED AS NOT REPLICATED: no seed is dropped, replaced or re-run to change an outcome, and no run is repeated except to repair an error.  "
    "Disclosure: before this declaration the analyst had glimpsed partial seed-2 Argentina random-arm means (67 runs); they were not used for any decision.")
V38_AMENDMENTS = [
    dict(id='V38-A1', item='KC_SEEDS', text='k-center cells run for seeds 0 and 1 only (seeds 2-4 skip them): clean null on two seeds (-0.47 points, 310 paired runs); saves about 465 runs.'),
    dict(id='V38-A2', item='AUG_CONTROL', text="new arm 'aug_copy' (ARG, budgets 1000/2000, the same combinations as aug_on): training rows copied 3x without transforms = the steps-matched control for augmentation; run for seeds 0-4 (seeds 0 and 1 are NEW runs added after those seeds' other runs)."),
    dict(id='V38-A4', item='seed-consistency guards', text='before any run: (1) PLAN CHECK - every run id already in results.csv must exist in the current plan (stop otherwise); (2) CANARY - up to 3 finished runs (aug off, aug on, k-center) are re-run and compared with the stored scores; same device and a different score = STOP (code or design changed between seeds); different device = reported only.  Resumed seeds write a new run_environment file instead of overwriting the original.'),
    dict(id='V38-P1', item='analysis protocol (declared BEFORE seeds 2-4 are finished)', text=PREREG_TEXT),
    dict(id='V38-A3', item='analysis', text='summary adds a mixed model (state combination as a random effect), average gain, worst-case gain and the share of combinations that beat the best single state; plus a state-alignment evidence printout. No training change.'),
]
# APPROVED['aug_on'] (below) stays False: the aug-ON runs are PENDING (shown in the plan, not run) until you set it True.  They use 3x the optimizer steps per epoch, so compare them with aug-OFF knowing that.
# ---- EXTENSION PROFILE (appendix, opt-in): CHN and FRA at budgets 3000 / 4000.  PROFILE = 'main' leaves the locked study EXACTLY as it was ----------------------------------------
PROFILE = 'main'                              # 'main' | 'extension'.  'extension' runs ONLY CHN and FRA at EXT_BUDGETS, in its own sub-folder (<outputs>/extension) with its own manifest; nothing is mixed with the main results
EXT_TARGETS = ['CHN', 'FRA']
EXT_BUDGETS = [3000, 4000]
EXT_MIN_CONTRIB = 0.5                         # every state of a combination must be able to supply at least this fraction of its EQUAL share of every fixed crop and of the remainder (so a 4-state combination cannot be one big state in disguise)
ALLOC_RULE = 'equal'                          # set by PROFILE: 'equal' (main, unchanged) | 'capacity' (extension: as equal as each state's capacity allows; a short state gives its surplus to the others)
EXT_LOCKED = ['PROFILE', 'EXT_TARGETS', 'EXT_BUDGETS', 'EXT_MIN_CONTRIB', 'ALLOC_RULE']
N_ARM = 10                                    # combinations per arm (k>=2); the diverse and clustered arms need C(pool,k) >= 2*N_ARM so that they are disjoint
ENUM_LIMIT = 50000                            # score EVERY combination when C(pool,k) <= this; otherwise score a fixed random sample of SCORE_SAMPLE distinct combinations (seeded by target,budget,k)
SCORE_SAMPLE = 20000
MIN_PROFILE_N = 20                            # a (state, crop) profile needs >= this many USA samples; states without a profile for a fixed crop leave the matched pool of ALL arms for that target
DIVERSITY_SPACE = ('phenology_spectral: per (state, crop) mean of globally standardised Sentinel-2 features (10 bands x all time steps); state distance = RMS difference of the mean profiles; '
                   'set score = mean pairwise state distance, averaged over the fixed (key) crops of the target design; USA source features only, no target data or scores')
CLASS_WEIGHT_SOURCE = 'train_val'             # = the authors' script: compute_class_weight('balanced') over the WHOLE source set they train on (here: the drawn budget sample = train+val). 'train' = training labels only (earlier review request).
FEATURE_SOURCE = 'Sentinel-2 5-day median time series, 10 bands (median_S2_*_month1-12_day5.npz); NOT AlphaEarth embeddings (64-d)'
TARGET_CROPS = {'ARG': ['soybeans'], 'CHN': ['rice'], 'FRA': ['corn', 'soybeans', 'wheat'], 'AUS': ['sugarcane', 'cotton']}   # crops whose count enters the regression
AUS_SPEC_CONFIRMED = False                    # !! the spec gives NO key-crop share for AUS.  AUS stays OUT of the run until you decide (see plan table).
AUS_KEY_SHARES_IF_CONFIRMED = None            # e.g. {'sugarcane': 0.20}; None = fully natural sampling.  Set deliberately.

# ---------------- training (spec 2; do not change) ----------------
NUM_CLASSES  = 7                              # CropNet(data_channel=1, classes=7): ALL 7 outputs trained, never a binary head
LR, WEIGHT_DECAY, BATCH_SIZE = 1e-4, 0.0, 256 # Adam, no scheduler  (== authors' run_CropNet_AUGM.py: --lr 1e-4, --batch_size 256, optim.Adam(model.parameters(), lr) i.e. default weight_decay 0, no scheduler)
# What the authors' script (x-ytong/CropNet run_CropNet_AUGM.py, read in this project) does, and where this notebook DEVIATES on purpose (printed in the plan cell):
AUTHORS_SCRIPT = dict(lr=1e-4, batch_size=256, num_epochs=50, val_ratio=0.2, num_runs=10, base_seed=81, optimizer='Adam(lr), weight_decay default 0', scheduler=None,
                      loss="CrossEntropyLoss(weight=compute_class_weight('balanced', classes=np.unique(labels of the whole source set)))", model='CropNet(data_channel=1, classes=len(train_index))',
                      train_aug='augm (this notebook: none at Level 1)', val_test_aug='none')
DEVIATIONS_FROM_AUTHORS = {'epochs': '100 with early stopping (authors: 50, no early stopping)', 'val_ratio': '0.10 of the budget, stratified (authors: 0.20)', 'seeds': '0-4 (authors: 82-91)',
                           'train_aug': 'none at Level 1 (authors AUGM: augm)', 'class_weights': 'authors-equivalent: drawn sample (train+val); zero-count crops -> weight 0 because the head has 7 outputs'}
MAX_EPOCHS, PATIENCE, MIN_EPOCHS, MIN_DELTA = 200, 15, 20, 0.0    # TOTAL epochs = 100 (your instruction) with early stopping INSIDE them on VALIDATION macro-F1 (strictly greater, best weights restored). The authors' script runs 50 epochs, no early stopping.
UNDERTRAINED_EPOCH = 90                       # ASSUMPTION: 0.9 x the 100-epoch cap (was 45 of 50).  best_epoch (1-based) >= 90 -> flag 'possibly_undertrained'
RESTORE_TOL  = 1e-6                           # restored model must reproduce its best validation score
EXPECTED_CROPNET_COMMIT = 'bdcf63ec1704a412da73c534c579fe08c5d2fa9b'

# ---------------- label space ----------------
CROP_LABELS = {0: 'corn', 1: 'soybeans', 2: 'rice', 3: 'wheat', 4: 'sugarcane', 5: 'cotton', 6: 'other'}
CROP_ID     = {v: k for k, v in CROP_LABELS.items()}
CROP_ORDER  = [CROP_LABELS[i] for i in range(7)]
RAW_TO_CROP = {'corn': 'corn', 'soybeans': 'soybeans', 'rice': 'rice', 'springwheat': 'wheat', 'winterwheat': 'wheat',
               'wheat': 'wheat', 'sugarcane': 'sugarcane', 'cotton': 'cotton', 'other': 'other'}

# paper (arXiv 2509.03497 v3) Table 2 -- raw class counts.  Raw counts and label sets are ASSERTED against this (stop on mismatch).
TABLE2 = {
    'ARG': dict(corn=1630,  soybeans=1808,  rice=1319,  wheat=0,     sugarcane=1253, cotton=1443,  other=2425),
    'AUS': dict(corn=0,     soybeans=0,     rice=0,     wheat=0,     sugarcane=5098, cotton=4825,  other=8938),
    'CHN': dict(corn=1344,  soybeans=1200,  rice=1156,  wheat=0,     sugarcane=0,    cotton=0,     other=1502),
    'FRA': dict(corn=24902, soybeans=21002, rice=2294,  wheat=27686, sugarcane=0,    cotton=0,     other=32715),
    'USA': dict(corn=22224, soybeans=17093, rice=17613, wheat=23226, sugarcane=9897, cotton=14799, other=28427),
}
TABLE2_TOTAL = {'ARG': 9878, 'AUS': 18861, 'CHN': 5202, 'FRA': 108599, 'USA': 133279}
assert all(sum(TABLE2[c].values()) == TABLE2_TOTAL[c] for c in TABLE2), 'Table 2 transcription error'
DEFINED_CLASSES = {t: [c for c in CROP_ORDER if TABLE2[t][c] > 0] for t in TABLE2}     # label set per country = classes with count > 0 in Table 2

# ---------------- scoring: the tracks are kept SEPARATE and separately named ----------------
REMAP_RULES = ['existing_release', 'paper_text_v3', 'review_targetmap']
REVIEW_RULE = 'review_targetmap'
# (results column, human label).  Nothing below is ever averaged with, or compared against, another track.
TRACKS = [('primary_f1', 'EXISTING primary (binary: ARG soybeans / CHN rice; AUS & FRA: existing release remap)'),
          ('binary_f1', 'EXISTING binary-collapse view (ARG soybeans, CHN rice)'),
          ('paper_f1', 'EXISTING release remap (AUS keep {sugarcane,cotton}; CHN keep {corn,soybeans,rice}; ARG wheat->other; FRA none)'),
          ('sevenclass_f1', 'EXISTING seven-class raw (no remap; macro over classes present in the target truth)'),
          ('papertext_f1', 'paper_text_v3: literal reading of paper sec.4.4 (AUS/CHN undefined->other; ARG/FRA no alignment)'),
          ('review_f1', 'REVIEW-PROPOSED: target-map remap (every class the target map does not define -> other) + macro-F1 over np.unique(y_true)')]
TRACK_PAIR = [('primary_f1', 'EXISTING'), ('review_f1', 'REVIEW-PROPOSED')]     # the two equally-ranked headline tracks; reported side by side, never pooled or ranked
# ---------------- scoring (spec 3) ----------------
PRIMARY_VIEW = {'ARG': ('binary', 'soybeans'), 'CHN': ('binary', 'rice'), 'AUS': ('paper', None), 'FRA': ('paper', None)}
BINARY_TARGETS = {t: v[1] for t, v in PRIMARY_VIEW.items() if v[0] == 'binary'}
COVERAGE_MIN_TRAIN = 30                        # 'coverage-conditional' macro-F1: classes with >= 30 TRAINING samples (reported next to, never instead of, the main score)
SCORER_TOL = 1e-12                             # sklearn scorer vs from-scratch numpy scorer
# paper Table 3, CropNet_AUGM, FULL USA source, mF1 (%).  CONTEXT ONLY: NOT comparable to this design (budgets of 1-4k, augmentation off). No flag is derived from it.
TABLE3_REF_MF1 = {'ARG': 75.63, 'AUS': 88.38, 'FRA': 90.00}      # (the paper has no USA->CHN experiment)

# ---------------- USA state-table geography guard (STOP if violated) ----------------
GEOGRAPHY_RULES = {'sugarcane': ['LA', 'FL', 'TX', 'HI'], 'rice': ['AR', 'CA', 'LA', 'MS', 'MO', 'TX']}
GEOGRAPHY_REPORT_ONLY = ['cotton']             # observed state set is printed, not asserted
GEOGRAPHY_MAX_OUTSIDE_ROWS = 0                 # strict (as specified).  USDA lists FL/TN as minor rice states, so a legitimate trace could trip this: raise it only deliberately.
# ---------------- gates (spec 6) ----------------
VAL_F1_WARN = 0.60                             # SKILL outcome: warn only, never abort
MAX_CONSEC_INTEGRITY = 5                       # abort only after this many CONSECUTIVE integrity failures
INTEGRITY_CHECKS = ['I1_true_classes_ge2', 'I2_predictions_valid', 'I3_split_exact', 'I4_restore_matches_best', 'I5_finite', 'I6_scorers_agree']

# ---------------- approvals (spec 9): nothing trains until you flip these, in order ----------------
# ---- AUTO MODE: the whole experiment (all seeds, aug OFF and aug ON cells) runs from the run cell with no manual switches.  AUTO_MODE = False restores the manual approvals below.
#      What stays automatic-STOP either way: the test gate, the smoke test, data / geography / state-alignment checks, 5 consecutive integrity failures, and a seed with any error / integrity failure (the next seed does not start).
AUTO_MODE, AUTO_MAX_HOURS = True, None          # AUTO_MAX_HOURS: e.g. 11 on Kaggle (session limit) -> pauses cleanly between runs; run the same cell again to resume
APPROVED = {'plan': False, 'fixed_composition': False, 'seed0_gate_report': False, 'seed1_gate_report': False, 'seed2_gate_report': False, 'seed3_gate_report': False, 'aug_on': False}
RUN_SEED = 0                                   # which seed the run cell executes this time (seed 0 first; seed s>0 needs APPROVED[f'seed{s-1}_gate_report'])

LOCKED = ['BUDGETS', 'K_LIST', 'SEEDS', 'AUG', 'MAX_COMBOS', 'MIN_COMBOS', 'MIN_KPOINTS', 'MIN_STATE_ROWS', 'VAL_FRACTION', 'KEY_SHARES', 'K_MAX', 'FRA_K8_MIN_POOL',
          'AUS_STATES', 'NUM_CLASSES', 'LR', 'WEIGHT_DECAY', 'BATCH_SIZE', 'MAX_EPOCHS', 'PATIENCE', 'MIN_EPOCHS', 'MIN_DELTA', 'PRIMARY_VIEW', 'COVERAGE_MIN_TRAIN',
          'EXPECTED_CROPNET_COMMIT', 'AUS_SPEC_CONFIRMED', 'AUS_KEY_SHARES_IF_CONFIRMED', 'TARGETS_TO_RUN', 'TARGET_CROPS', 'UNDERTRAINED_EPOCH', 'REVIEW_RULE', 'TRACKS', 'GEOGRAPHY_RULES', 'CLASS_WEIGHT_SOURCE', 'COMPOSITION_MODES_TO_RUN', 'FIXED_COMPOSITION_CHOICE', 'STATE_SELECTOR', 'FEATURE_SOURCE', 'REMAP_RULES',
          'SELECTION_ARMS', 'POOL_RULE', 'N_ARM', 'ENUM_LIMIT', 'SCORE_SAMPLE', 'MIN_PROFILE_N', 'DIVERSITY_SPACE',
          'FRA_MAIN_RULE', 'INPUT_FEATURES', 'KC_TARGETS', 'KC_COMBOS', 'KC_DIMS', 'KC_MAX_CAND', 'AUG_TARGETS', 'AUG_BUDGETS', 'AUG_COMBOS', 'AUG_REPEAT']
if AUTO_MODE:
    APPROVED.update(plan=True, aug_on=True); print('*** AUTO_MODE: plan and augmentation-ON runs are pre-approved; seed gate reports are approved automatically when clean. The plan table is still printed. Set AUTO_MODE = False for manual approval. ***')
assert POOL_RULE in ('strict', 'equal_share') and 'random' in SELECTION_ARMS and set(SELECTION_ARMS) <= {'random', 'diverse', 'clustered', 'common', 'kcenter', 'aug_on'}, 'SELECTION_ARMS / POOL_RULE invalid'
assert PROFILE in ('main', 'extension'), "PROFILE must be 'main' or 'extension'"
MAIN_OUT_DIR = globals().get('MAIN_OUT_DIR') or globals().get('OUT_DIR')      # the main results folder (never written by the extension)
MAIN_DEFAULTS = dict(TARGETS_TO_RUN=list(TARGETS_TO_RUN), BUDGETS=list(BUDGETS), SELECTION_ARMS=list(SELECTION_ARMS))      # what the MAIN study uses (the test gate always runs against this)
if PROFILE == 'extension':
    TARGETS_TO_RUN = list(EXT_TARGETS); BUDGETS = list(EXT_BUDGETS); ALLOC_RULE = 'capacity'; SELECTION_ARMS = ['random']; LOCKED = LOCKED + [k_ for k_ in EXT_LOCKED if k_ not in LOCKED]
    if MAIN_OUT_DIR: OUT_DIR = os.path.join(MAIN_OUT_DIR, 'extension'); os.makedirs(OUT_DIR, exist_ok=True)
    print(f'*** PROFILE = extension (APPENDIX): {TARGETS_TO_RUN} at budgets {BUDGETS}; random arm only; outputs in {globals().get("OUT_DIR")}.  These results are NOT part of the main study and must not be pooled with it. ***')
else:
    ALLOC_RULE = 'equal'
    if MAIN_OUT_DIR: OUT_DIR = MAIN_OUT_DIR                                                                       # switching back to the main profile re-points the outputs to the main folder
IN_NOTEBOOK = 'ipykernel' in sys.modules
print('config OK |', EXPERIMENT_VERSION, '| spec', SPEC_TAG)
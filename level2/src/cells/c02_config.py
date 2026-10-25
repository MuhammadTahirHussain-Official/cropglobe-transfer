# ======================================================================================
# CONFIGURATION  (pure python; nothing here touches the network or the GPU)
# LEVEL 2 (France as the SOURCE country; French regions play the role the US states played in Level 1).  Same training setup as Level 1; design in LEVEL2_PREREGISTRATION.
# Derived from the audited Level 1 v3.8 code; every difference is listed in the manifest amendments (L2-A*) and in CHANGES_LEVEL2.md.
# ======================================================================================
import os, sys, json, math, time, glob, re, csv, zlib, random, itertools, hashlib, datetime, pickle, warnings, traceback, subprocess, tempfile, io, contextlib
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats as scipy_stats
from scipy import optimize as scipy_opt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support

EXPERIMENT_VERSION = 'phase3_level2_fra_source_v1'          # NEW results folder: v3.7 trains up to 200 epochs, so results from the 100-epoch cap (v3.2-v3.6) must never be mixed with these
SPEC_TAG = 'level2_fra_source_prereg_v1'          # written into the manifest; a folder without it is NOT ours -> the run stops instead of mixing

# ---------------- design (spec section 4) ----------------
BUDGETS      = [1000, 2000, 3000, 4000]       # TOTAL labelled samples per run (train + validation); never shrunk
K_LIST       = [1, 2, 4, 8]
SEEDS        = [0, 1, 2, 3, 4]
AUG          = 'off'                          # Level 1 = augmentation OFF only (spec 2).  ON needs a step-matched control + approval.
MAX_COMBOS   = 10                             # k>=2: up to 10 random combinations; k=1 is a census of EVERY pool state
MIN_COMBOS   = 5                              # a (target,budget,k) cell with fewer combinations is recorded as infeasible
MIN_KPOINTS  = 3                              # a (target,budget) with fewer feasible k-points is infeasible as a whole
MIN_STATE_ROWS = 1500                         # source pool = French regions with >= 1500 (de-duplicated) rows
VAL_FRACTION = 0.10                           # validation = 10% of the budget, stratified by crop, deterministic
TARGETS_ORDER = ['ARG', 'USA', 'CHN']
TARGETS_TO_RUN = ['ARG', 'USA', 'CHN']
TARGET_ROLE  = {'ARG': 'main', 'USA': 'main', 'CHN': 'descriptive', 'AUS': 'appendix', 'FRA': 'source'}
# KEY-CROP share (fixed, identical at every k); everything else is sampled NATURALLY from the state's remaining rows.
KEY_SHARES   = {'ARG': {'soybeans': 0.20}, 'CHN': {'rice': 0.20}, 'USA': {'corn': 0.15, 'soybeans': 0.15, 'wheat': 0.15}, 'AUS': None}     # USA shares = the Level 1 France design (PROPOSED in decision 2; instructor to confirm)
K_MAX        = {'ARG': 8, 'USA': 8, 'CHN': 2, 'AUS': 2, 'FRA': 8}     # CHN: rice exists in only 2 French regions -> k <= 2
FRA_K8_MIN_POOL = 10                          # FRA k=8 only if the pool has >= 10 states
AUS_STATES   = ['LA', 'FL']                   # AUS = appendix census of {LA, FL}, k <= 2, no sweep (exempt from MIN_COMBOS / MIN_KPOINTS)
# ---- composition modes.  'key_plus_natural' = EXISTING baseline (fixed key-crop share + natural remainder).
#      'fixed_full'        = REVIEW-PROPOSED (complete crop composition fixed at every budget and k).  The VALUES are a scientific choice: they are
#                            NOT set here.  Proposals are printed with their capacity checks in the plan cell; nothing runs until you approve one.
COMPOSITION_MODES_TO_RUN = ['key_plus_natural']
FIXED_COMPOSITION_CHOICE = None               # e.g. 'A_key_plus_equal_rest' | 'B_equal_over_target_classes'  (only after approval)
ALIGNMENT_ORDERING_OVERRIDE = None             # V33-06: None = STOP on an ambiguous state alignment; set to a candidate ordering name only after reading cache/alignment_report.json
NOTEBOOK_VERSION = 'L2-v1'                       # display / run_environment only (not part of the locked design)
STATE_SELECTOR = 'random'                     # the baseline arm: random combinations (unchanged)
# ---- PHASE 4 (definition approved by the researcher): phenology-spectral diversity of the USA source states ------------------------------------------------------------------------------
SELECTION_ARMS = ['random', 'aug_on']   # v3.7 adds common / kcenter / aug_on (see below); 'random' is the baseline and is mandatory; ['random'] reproduces the pre-Phase-4 plan exactly
POOL_RULE = 'strict'                          # 'strict' (as before: a pool state must supply the WHOLE budget alone) | 'equal_share' (OPT-IN, not approved yet: a state in a k-combination only supplies its 1/k share; the pool then depends on k)
# ---- v3.7 (supervisor review of the 409-run seed-0 results): reporting rules, new tracks.  Every item is listed in CHANGELOG_v3_7.md ------------------------------------------------------------
# R2 metrics: ONE main metric and ONE secondary metric are reported.  The other score columns are still computed and stored (audit trail) but are not summarised unless REPORT_ALL_TRACKS = True.
REPORT_PRIMARY, REPORT_SECONDARY, REPORT_ALL_TRACKS = 'main_f1', 'paper_f1', False
# R3 France: rice is never in the training mix (rice F1 = 0 in every run), so a rule is stated IN ADVANCE: FRA's main metric is the coverage-conditional macro-F1 (classes with >= COVERAGE_MIN_TRAIN training samples; the paper does the same for France->USA).
#    ARG / CHN main metric = binary key-crop macro-F1 (soybeans / rice vs. other) = the existing primary_f1.   'coverage_conditional' | 'primary' (= no rule; FRA scored on all classes)
FRA_MAIN_RULE = 'coverage_conditional'     # applies to the multi-class targets in FRA_LIKE_TARGETS (Level 2: USA): sugarcane / cotton never occur in the French source, so the main metric is the coverage-conditional macro-F1, as for France in Level 1
# R4 which input features the CropNet runs use (written into every results row)
INPUT_FEATURES = 'Sentinel-2 5-day medians (CropGlobe for_CropNet/median_S2_*_month1-12_day5.npz; 10 bands x 73 steps; official DOY 121-334 window = 43 steps); AlphaEarth/Presto/Harmonic NOT used'
# R5 fixed common pool: the pool of source units shrinks as the budget grows (ARG 16 -> 6), which confounds budget comparisons.  Arm 'common' re-draws the combinations of EVERY budget from the states eligible at ALL budgets of the target.
# R6 sample-level selection: arm 'kcenter' = same state combinations and same per-state/per-crop counts as 'random', but the rows inside each (state, crop) pool are chosen by greedy k-center (farthest-first) in a PCA space of the USA window features.
KC_TARGETS, KC_COMBOS, KC_DIMS, KC_MAX_CAND = [], 5, 16, 20000     # Level 2: no k-center (decision 4)     # k>=2: at most KC_COMBOS of the random-arm combinations (deterministic), k=1: the whole census
# R7 augmentation ON (authors' CropGlobeDataset aug_mode='augm', pinned commit; training set = original + AUG_REPEAT augmented copies; validation/test never augmented).  Arm 'aug_on' = the same combinations as the random arm.
AUG_TARGETS, AUG_BUDGETS, AUG_COMBOS, AUG_REPEAT = ['ARG'], [1000, 2000], 5, 2
# ---- v3.8 (instructor review of the 1,475-run file; seeds 0 and 1 complete).  These keys are deliberately NOT in LOCKED: the locked design of the runs already on disk is unchanged, so the same results.csv resumes.
#      Every item is written into manifest.json as a dated AMENDMENT (V38_AMENDMENTS) the first time the run cell executes, so nothing about the design changes silently.
KC_SEEDS = [0, 1]          # k-center sample selection runs for these seeds only: seeds 0-1 gave a clean null (-0.47 points over 310 paired runs).  Seeds >= 2 skip the k-center cells.  Set KC_SEEDS = SEEDS to restore every seed.
AUG_CONTROL = True         # steps-matched control for augmentation.  Arm 'aug_copy' = the SAME cells as 'aug_on', but the training rows are COPIED (1 + AUG_REPEAT)x WITHOUT any transform: same rows per epoch, same optimizer steps, no augmentation.
USA_SCORING = 'fixed4_clean'                 # L2-D11 (supersedes L2-D9): USA main metric = macro-F1 over the FIXED classes below, with true rice / sugarcane / cotton removed from the test set.  Not in LOCKED (the results file resumes); it IS in the design hash.
USA_FIXED_CLASSES = ['corn', 'soybeans', 'wheat', 'other']
USA_DROP_TRUE_CLASSES = ['rice', 'sugarcane', 'cotton']          # true samples of these classes are REMOVED from the USA test set before scoring (paper sec. 4.4 excludes sugarcane / cotton for FRA -> USA; rice added by the instructor)
USA_REASSIGN_PRED_TO_OTHER = ['rice', 'sugarcane', 'cotton']      # predictions of these classes (on the remaining samples) are reassigned to 'other'
PREREG_TEXT = ("Level 2 (France as source).  Seeds 0-1 exploratory, seeds 2-4 confirmatory.  Claims (main targets ARG and USA): L2-C1 mixed-model effect of log2(k) on main F1 > 0 (random intercept per region combination, budget and seed adjusted); "
    "L2-C2 gain from the fewest to the most regions > 0 at each budget; L2-C3 steps-matched augmentation aug_on - aug_copy (Argentina, budgets 1000/2000) reported with its interval, no direction assumed.  China is DESCRIPTIVE ONLY (rice exists in 2 French regions): no claim.  "
    "A claim passes only if its sign agrees in the exploratory and the confirmatory seeds AND the confirmatory 95% interval excludes 0.  A failed claim is REPORTED AS NOT REPLICATED: no seed or run is dropped, replaced or re-run to change an outcome, except to repair a recorded error.  "
    "Decisions were fixed before any run; they were not chosen from results.")
V38_AMENDMENTS = [      # name kept for compatibility with the manifest code; these are the LEVEL 2 design decisions (see LEVEL2_decisions_for_approval.pdf)
    dict(id='L2-D1', item='source regions', text='France, region = unit. Pool rule unchanged from Level 1 (>= MIN_STATE_ROWS rows; a region must supply the whole budget alone). Region of every row from the verified row map (FRA_join_verification); rows without a certain region are excluded and counted.'),
    dict(id='L2-D2', item='targets', text='Main: ARG (soybeans 20%) and USA (corn, soybeans, wheat 15% each; shares proposed by the analyst, pending instructor confirmation). BEL/NLD/GBR are NOT included (no verified label sets / key shares).'),
    dict(id='L2-D3', item='China', text='CHN descriptive only: k <= 2 (rice exists only in regions 76 and 93); exempt from MIN_COMBOS / MIN_KPOINTS; no claim.'),
    dict(id='L2-D4', item='arms', text='random arm + steps-matched copy control on ARG budgets 1000/2000 (aug_on and aug_copy, 5 combinations); no k-center, diversity or common-pool arms.'),
    dict(id='L2-D5', item='replication rule', text='seeds 0-1 exploratory, 2-4 confirmatory; same pass rule as Level 1.'),
    dict(id='L2-D6', item='reference model', text='statsmodels MixedLM primary (fallback optimisers), own REML alongside; tolerance coef/SE 0.02.'),
    dict(id='L2-D7', item='metric for USA target', text='multi-class macro-F1 with the coverage-conditional rule (classes with >= 30 training samples), as for France in Level 1; no remap (the authors release no USA rule).'),
    dict(id='L2-D8', item='untestable comparisons', text='L2-C2 is judged only where an interval exists. At ARG budget 1000 the largest k (8) enumerates ALL combinations of the 9-region pool and k=1 is a census, so no sampling interval exists by construction: the code prints NOT TESTABLE there (never PASS, never "interval includes 0"). L2-C1 (mixed model over all cells) is the primary test of the effect of k.'),
    dict(id='L2-D9', item='USA scoring (changed AFTER seed 0 was seen)', text='At the instructor\'s review of seed 0 (306 runs): the USA target is scored on a fixed class set (corn, soybeans, wheat, other); rice is not scored and rice predictions are reassigned to other. Reason: rice exists in only 2 French regions, so under the coverage-conditional rule the scored class set depended on the region mix (rice scored in 15% of k=1, 58% of k=4 combinations; mean 60.7 when rice was scored vs 54.0). Seed-0 USA rows were re-scored from their saved predictions (no retraining; original kept). This change was made after seed 0 was seen; it was requested by the instructor on the grounds above, not chosen from the size of any claim.'),
    dict(id='L2-D10', item='process note', text='Seed 0 (306 of 319 planned runs) was run before the instructor\'s written sign-off on the decision sheet. Seeds 0-1 are exploratory under the pre-registered rule, so no claim rests on them. The plan (design hash, now including the scoring rule) is frozen before seeds 2-4. Any further change made after seeing seed 0 must be logged here as such.'),
    dict(id='L2-D11', item='USA scoring v2 (changed AFTER seeds 0-3 were seen)', text='Instructor review of the interim report: true rice, sugarcane and cotton samples are REMOVED from the USA test set and predictions of rice / cotton / sugarcane are reassigned to other; the main metric is macro-F1 over corn, soybeans, wheat, other (supersedes L2-D9, where true labels were not changed). Basis: the paper (sec. 4.4) excludes sugarcane and cotton from evaluation for FRA -> USA because the source lacks them; the instructor extends this to rice, which is absent from most training mixes (2 of 13 French regions). ALL seeds re-scored from the saved predictions (no retraining; originals kept; every earlier value kept in a column). Made after seeds 0-3 were seen, at the instructor\'s request, not chosen from the size of any claim.'),
    dict(id='L2-D12', item='exploratory analyses added after seeing seeds 0-3', text='(i) mix versus the mean of its own single regions and versus its best single region (every k>=2 combination at the same budget); (ii) the budget effect on a FIXED pool of regions (those available at every budget); (iii) subset analyses with / without regions 75 and 76. All are labelled exploratory / post hoc; none replaces a pre-registered claim.'),
    dict(id='L2-P1', item='analysis protocol', text=PREREG_TEXT),
]
# APPROVED['aug_on'] (below) stays False: the aug-ON runs are PENDING (shown in the plan, not run) until you set it True.  They use 3x the optimizer steps per epoch, so compare them with aug-OFF knowing that.
# ---- EXTENSION PROFILE (appendix, opt-in): CHN and FRA at budgets 3000 / 4000.  PROFILE = 'main' leaves the locked study EXACTLY as it was ----------------------------------------
PROFILE = 'main'                              # 'main' | 'extension'.  'extension' runs ONLY CHN and USA at EXT_BUDGETS, in its own sub-folder (<outputs>/extension) with its own manifest; nothing is mixed with the main results
EXT_TARGETS = ['CHN', 'USA']                  # Level 2: dormant (PROFILE='main'); FRA is the SOURCE now, so the Level 1 value ['CHN','FRA'] could not run
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
TARGET_CROPS = {'ARG': ['soybeans'], 'CHN': ['rice'], 'FRA': ['corn', 'soybeans', 'wheat'], 'USA': ['corn', 'soybeans', 'wheat'], 'AUS': ['sugarcane', 'cotton']}   # crops whose count enters the regression
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

# ---------------- LEVEL 2 SOURCE (France) ----------------
SOURCE = 'FRA'; SOURCE_UNIT_NAME = 'region'
SOURCE_UNITS = ['11', '24', '27', '28', '32', '44', '52', '53', '75', '76', '84', '93', '94']       # the 13 metropolitan regions found in the raw file names (FRA_join_verification.md)
FRA_LIKE_TARGETS = ['FRA', 'USA']              # multi-class targets scored with the coverage-conditional main metric
EXEMPT_RULES = ['AUS', 'CHN']                  # targets exempt from MIN_COMBOS / MIN_KPOINTS in the plan (AUS: Level 1 appendix census; CHN: descriptive census of the 2 rice regions)
DESCRIPTIVE_TARGETS = ['CHN']                  # run, reported, but NO claim; exempt from MIN_COMBOS / MIN_KPOINTS (census of the 2 rice regions)
CLAIM_TARGETS = ['ARG', 'USA']
# exact row map produced by the student's Colab run of the raw<->npz matcher (cell 9 of Level1_closing_and_Level2_probe.ipynb).  The file is identified by its hash: a different file STOPS the run.
FRA_ROW_MAP_SHA256 = '93abeb51af5f68998818b770c8c5d853db5307e06f5aaf3dd1a6123d6ed69d9c'
FRA_ROW_MAP_ROWS, FRA_ROW_MAP_AMBIGUOUS = 108599, 96
# raw points per (region, crop token) in that row map (before de-duplication) -- checked against the file, so a wrong or edited map cannot pass
FRA_REGION_TOTALS = {'11': 3351, '24': 9546, '27': 12164, '28': 6493, '32': 7043, '44': 12968, '52': 4872, '53': 5639, '75': 17809, '76': 13224, '84': 10639, '93': 4275, '94': 576}
FRA_BLOCKS = [  # (crop token, region, first packaged row, last packaged row) of each of the 64 raw files, derived from the verified row map (sha256 above); rows outside every block are excluded
    ('corn', '11', 0, 755), ('corn', '24', 756, 3150), ('corn', '27', 3151, 5313), ('corn', '28', 5314, 7365), ('corn', '32', 7366, 9536),
    ('corn', '44', 9537, 12674), ('corn', '52', 12675, 14795), ('corn', '53', 14796, 16598), ('corn', '75', 16599, 20759),
    ('corn', '76', 20760, 22637), ('corn', '84', 22638, 24749), ('corn', '93', 24750, 24882), ('corn', '94', 24883, 24901),
    ('soybeans', '11', 24902, 25688), ('soybeans', '24', 25689, 27359), ('soybeans', '27', 27360, 31374), ('soybeans', '28', 31375, 31407),
    ('soybeans', '32', 31408, 31617), ('soybeans', '44', 31636, 33762), ('soybeans', '52', 33763, 34259), ('soybeans', '53', 34260, 34396),
    ('soybeans', '75', 34397, 39693), ('soybeans', '76', 39694, 43573), ('soybeans', '84', 43574, 45740), ('soybeans', '93', 45741, 45903),
    ('rice', '76', 45904, 46755), ('rice', '93', 46756, 48197), ('springwheat', '11', 48198, 48310), ('springwheat', '24', 48311, 48651),
    ('springwheat', '27', 48652, 49009), ('springwheat', '28', 49010, 49161), ('springwheat', '32', 49162, 49367),
    ('springwheat', '44', 49368, 49787), ('springwheat', '52', 49788, 49894), ('springwheat', '53', 49895, 49926),
    ('springwheat', '75', 49927, 50024), ('springwheat', '76', 50025, 50173), ('springwheat', '84', 50174, 50251),
    ('springwheat', '93', 50252, 50349), ('winterwheat', '11', 50350, 51205), ('winterwheat', '24', 51206, 53707),
    ('winterwheat', '27', 53708, 56165), ('winterwheat', '28', 56166, 58284), ('winterwheat', '32', 58285, 60510),
    ('winterwheat', '44', 60511, 63987), ('winterwheat', '52', 63988, 66134), ('winterwheat', '53', 66135, 67910),
    ('winterwheat', '75', 67911, 71112), ('winterwheat', '76', 71113, 73366), ('winterwheat', '84', 73367, 75350),
    ('winterwheat', '93', 75351, 75878), ('winterwheat', '94', 75879, 75883), ('other', '11', 75884, 76722), ('other', '24', 76723, 79359),
    ('other', '27', 79360, 82529), ('other', '28', 82530, 84666), ('other', '32', 84667, 86896), ('other', '44', 86897, 90684),
    ('other', '53', 90685, 92575), ('other', '75', 92576, 97626), ('other', '76', 97627, 101837), ('other', '84', 101838, 106135),
    ('other', '93', 106136, 108046), ('other', '94', 108047, 108598),
]
FRA_ASSIGNED_PER_REGION = {'11': 3351, '24': 9546, '27': 12164, '28': 6493, '32': 7043, '44': 12950, '52': 4872, '53': 5639, '75': 17809, '76': 13224, '84': 10639, '93': 4275, '94': 576}     # packaged rows that lie inside a file block (108,581); the other 18 rows are excluded
# Level 1 training setup, which Level 2 must keep unchanged (reviewer point); asserted again in the tests
LEVEL1_TRAINING_SETUP = dict(NUM_CLASSES=7, LR=1e-4, WEIGHT_DECAY=0.0, BATCH_SIZE=256, MAX_EPOCHS=200, PATIENCE=15, MIN_EPOCHS=20, MIN_DELTA=0.0, VAL_FRACTION=0.10, BUDGETS=[1000, 2000, 3000, 4000], K_LIST=[1, 2, 4, 8], SEEDS=[0, 1, 2, 3, 4], MAX_COMBOS=10, MIN_COMBOS=5, MIN_KPOINTS=3, MIN_STATE_ROWS=1500, COVERAGE_MIN_TRAIN=30, POOL_RULE='strict', CLASS_WEIGHT_SOURCE='train_val')
# INSTRUCTOR APPROVAL GATE.  The plan cell always runs (nothing trains).  The run cell refuses to start until this is True AND the design hash below is the one that was approved.
INSTRUCTOR_APPROVED = False
APPROVED_DESIGN_SHA256 = ''                    # paste the design hash printed by the plan cell once the instructor has signed the decisions
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
PRIMARY_VIEW = {'ARG': ('binary', 'soybeans'), 'CHN': ('binary', 'rice'), 'AUS': ('paper', None), 'FRA': ('paper', None), 'USA': ('paper', None)}
BINARY_TARGETS = {t: v[1] for t, v in PRIMARY_VIEW.items() if v[0] == 'binary'}
COVERAGE_MIN_TRAIN = 30                        # 'coverage-conditional' macro-F1: classes with >= 30 TRAINING samples (reported next to, never instead of, the main score)
SCORER_TOL = 1e-12                             # sklearn scorer vs from-scratch numpy scorer
# paper Table 3, CropNet_AUGM, FULL USA source, mF1 (%).  CONTEXT ONLY: NOT comparable to this design (budgets of 1-4k, augmentation off). No flag is derived from it.
TABLE3_REF_MF1 = {'ARG': 75.63, 'AUS': 88.38, 'FRA': 90.00}      # (the paper has no USA->CHN experiment)

# ---------------- USA state-table geography guard (STOP if violated) ----------------
GEOGRAPHY_RULES = {'rice': ['76', '93']}              # France: rice exists only in regions 76 and 93 (row map); sugarcane / cotton do not exist in France (checked in the data cell)
GEOGRAPHY_REPORT_ONLY = ['sugarcane', 'cotton']             # observed state set is printed, not asserted
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
          'FRA_MAIN_RULE', 'INPUT_FEATURES', 'KC_TARGETS', 'KC_COMBOS', 'KC_DIMS', 'KC_MAX_CAND', 'AUG_TARGETS', 'AUG_BUDGETS', 'AUG_COMBOS', 'AUG_REPEAT',
          'SOURCE', 'SOURCE_UNITS', 'FRA_LIKE_TARGETS', 'DESCRIPTIVE_TARGETS', 'CLAIM_TARGETS', 'FRA_ROW_MAP_SHA256', 'LEVEL1_TRAINING_SETUP']
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
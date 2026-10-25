# Level 2 (France as the source country) -- DRAFT pre-registration

**Status: DRAFT. Not signed off. Nothing below has been run, and no French data has been looked at.** Items marked **[OPEN]** are my proposals or unknowns; the instructor must confirm them before any run. Once confirmed this file is hashed and frozen exactly like Level 1.

## Why
The instructor's review: training on US states only is the biggest remaining weakness. Level 2 repeats the Level 1 question with France as the source (French regions play the role of US states).

## Fixed from Level 1 (for comparability)
Model and training (CropNet, Sentinel-2 5-day medians, DOY 121-334 window, Adam 1e-4, batch 256, max 200 epochs, patience 15), metric (`main_f1` = binary key-crop macro-F1; coverage rule for multi-crop targets), seeds 0-4 with 0-1 exploratory and 2-4 confirmatory, budgets, no seed ever dropped or re-run, the plan/canary guards, the namespace guard. Only the source country and its units change.

## Must be established from the real data first (the probe notebook)
1. **[RESOLVED 2026-10-22] Region labels for France.** The npz stores no region, but the raw file names carry it (`..._LAMB93_R<code>.csv`) and an exact row-by-row match assigns every one of the 108,599 npz rows to a raw point (see `FRA_join_verification.md`). 13 regions: 11, 24, 27, 28, 32, 44, 52, 53, 75, 76, 84, 93, 94.
2. **[OPEN] Number of usable regions** and their samples per key crop. Known now: rice exists in only 2 regions (76, 93); corn in region 94 has 19 points, winter wheat 5, soybeans in region 28 has 33; the full per-region table is in the row map. k values (1, 2, 4, 8 in Level 1) are limited by the pool size; the plan is built from the real table with the same feasibility rules (>= 5 combinations, >= 3 k points), not chosen by hand.
3. **[OPEN] Region polygons.** The probe tests reachability of two public sources and counts France's regions in them; it does not choose between them.

## Proposed design **[OPEN: all of this needs instructor sign-off]**
- Source: FRA, units = French regions (13 metropolitan, if confirmed).
- Targets: the authors' own France-source command uses BEL NLD GBR CHN USA ARG. Proposed primary targets: **USA** (corn/soybeans/wheat 15% each, mirroring France's role in Level 1), **ARG** (soybeans 20%), **CHN** (rice 20%). BEL, NLD, GBR secondary/descriptive. Key-crop shares for any target not in Level 1 must be stated by the instructor.
- Arms: random only (plain draws). Steps-matched copy control on a **subset** (as the review suggested); no k-center; no GroupDRO unless the instructor asks.
- Budgets and k: from the feasibility rules above.

## Claims, declared before any run
- **L2-C1.** For each primary target, the mixed-model effect of log2(k) on `main_f1` (random intercept per region combination, budget and seed adjusted) is > 0.
- **L2-C2.** The gain from the fewest to the most regions is > 0 at each budget.
- **L2-C3.** Steps-matched augmentation: `aug_on - aug_copy` is reported with its interval on the subset; no direction assumed.
- **L2-C4 (cross-source).** For the target shared with Level 1 (CHN) the sign of C1 matches Level 1's (+4.5 per doubling).
- **Rule:** a claim passes only if its sign agrees in the exploratory seeds (0-1) and the confirmatory seeds (2-4) AND the confirmatory 95% interval excludes 0. A failed claim is reported as NOT REPLICATED. No seed or run is dropped, replaced or repeated except to repair a recorded error.

## Analysis and integrity
- Mixed model fitted with statsmodels; the notebook's own REML code is reported next to it, tolerance 0.02 on the coefficient and on the SE, fixed in advance.
- Before the run: plan hash and the SHA-256 of every config/code file are recorded; before every seed the plan check and canary re-run (same device => mismatch = STOP).
- The final `results.csv` is hashed at the end, as for Level 1.

## Not part of Level 2
Mixing source countries (the review's optional "Level 2b") and any new method.

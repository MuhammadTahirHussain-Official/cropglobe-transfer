# CropGlobe Level 1 v3 (fixed key-crop share) - change list, conflicts, tests

Deliverables: `CropGlobe_Level1_v3_FIXEDMIX_COLAB.ipynb`, `test_report_part_A.txt`, this file.
**No GPU/real-data run was possible in this session** (no torch, no network from the sandbox). Everything below marked "run" was run on synthetic data with a stand-in model.

## A. Conflicts / gaps found (reported, NOT resolved silently)
1. **Old notebook vs paper v3 Table 2.** `verify_data` hard-coded ARG 12085, CHN 3063, FRA 108699 (older paper version). v3 Table 2 = ARG 9878, CHN 5202, FRA 108599 (AUS 18861, USA 133279 unchanged). Now asserted against v3 per country, per class.
2. **Old `PAPER_REF_F1`** (0.916/0.952/0.873/0.895) matches no Table 3 value. Removed, with the 'too good' flag. Table 3 (75.63 / 88.38 / 90.00; no USA->CHN row exists in the paper) is printed as context only.
3. **AUS is under-specified.** Spec: "no sweep, census of {LA, FL}, k<=2" but no key-crop share. AUS is therefore **excluded from the run** (`AUS_SPEC_CONFIRMED=False`); the plan table shows it with natural sampling for feasibility only. Needs your decision (e.g. `{'sugarcane': 0.20}`).
4. **Regression on "k, key-crop count and budget" is degenerate.** Key-crop count = share x budget exactly within a target, so it is perfectly collinear with budget. The code detects rank deficiency, drops key-crop count, and prints the aliasing note instead of hiding it.
5. **Test wording.** "true corn predicted soybeans counts as other" is true for CHN (positive = rice: other/other). For ARG (positive = soybeans) it is other/**positive**, i.e. a false positive. I implemented your `binary_collapse` literally and test both cases.
6. **Authors' repo files could not be read here** (github sub-pages are robots-blocked): `util/label.py`, `data/loading.py`, `models/CropNet.py`, and the commit page for `bdcf63ec...`. Consequences, all enforced in Colab as hard STOPs: the commit must exist and the repo must be clean/unmodified; Part B tests compare our paper rule with the authors' live `remap_prediction` (signature printed, call variants tried, any difference stops the run); `CropGlobeDataset` window must be (5, 121, 334, 43).
7. Observed in the authors' `run_CropNet_AUGM.py`: the train/val sets are built from `train_set.dataset` (the full dataset, not the split subset). We do not use their loop; flagged only so nobody reuses it for validation.
8. **Output namespace.** The old notebook already used `outputs/phase3_level1_v3_fixedmix`. A folder with results but no v3-spec manifest is refused (STOP); move it aside (keep it) and copy `cache/usa_meta.csv` into the new folder (or put its path in `EXTRA_META_PATHS`) to avoid rebuilding the state table.

## B. Interpretations I had to choose (please confirm at the plan approval)
- Pool per (target, budget) = states with >= 1500 rows **and** able to supply the whole budget alone (key crop(s) >= fixed count, non-key rows >= natural remainder). Same pool for every k.
- "Natural" rest = uniform draw from the state's non-key rows (all other crops incl. wheat/sugarcane/...), equal count per state; key counts split equally over states.
- <5 combinations or <3 feasible k-points rule applies literally to every cell of ARG/FRA/CHN (including k=1); AUS exempt (appendix).
- The sample draw depends on the seed (kept from the old notebook); the 10% stratified validation split is a deterministic function of the draw (largest-remainder per crop, exact B/10).
- Validation macro-F1 = sklearn macro (zero_division=0), as in the old notebook. `best_epoch` is 1-based; `possibly_undertrained` if >= 45.
- FPC = sqrt((N-n)/(N-1)), N = C(pool, k). Jackknife = delete-one-state over the states in the evaluated combinations. k_min->k_max gain CI is Welch-style (a k=1 census adds no variance).
- Integrity failures counted for the 5-consecutive abort: the six checks in the spec **plus** unexpected exceptions (e.g. non-finite loss). Fewer than 2 true classes stops the run immediately.
- Majority baseline = majority class of the true target labels (reported only).

## C. What changed in the notebook (all cells)
- **Config**: new locked design (budgets, k, key shares, K_MAX, 50 epochs, patience 15, min 20, delta 0), Table 2/3 constants, approval flags, `RUN_SEED`. Removed: augmentation ON, TEST_CAP, POOL_MODE, equal-share quotas, 100 epochs.
- **Scoring**: `binary_collapse` exactly as specified (eval view only, never -1, nothing dropped); paper rule mirrors `remap_prediction`; primary = binary (ARG soybeans, CHN rice) / paper rule (AUS, FRA); paper-rule score always also stored for ARG/CHN; all-other and majority baselines every run; coverage-conditional macro-F1 (>=30 train samples) separate; absent-train classes stay in the score, flagged; two independent scorers (1e-12) checked on every run.
- **Training**: 7-class CropNet, Adam lr 1e-4 wd 0 no scheduler, batch 256, balanced weights from train+val (0 for empty classes), official window for train/val/test, early stopping via a unit-tested `EarlyStopper`, best-weight restore re-verified, stop reason logged, hard cap 50.
- **Design/sampling**: fixed key share, natural rest, exact budget, exact 10% stratified validation, unique/overlap/leakage assertions, USA-vs-target identical-row leakage check.
- **Gates**: integrity (abort after 5 consecutive) vs skill (warn only); seed-direction disagreement = warning.
- **Statistics**: seeds averaged first; t-interval with FPC (k>=2); k=1 census values, no CI; jackknife; seed spread separately; regression; gain with CI.
- **Outputs**: new manifest-guarded namespace, atomic writes, npz per run (y_true, y_pred, ids, counts, confusion matrices, per-class P/R/F1), metrics CSV row written LAST, resumable, per-seed gate report files.
- **Data**: strict Table 2 asserts (raw totals + class counts + label sets), de-dup difference report, -0.0 normalisation in the dedupe hash. Repo pinned to the spec commit.
- **Flow control**: tests -> plan approval -> smoke -> seed 0 -> approval -> seeds 1-4 enforced by assertions.

## D. Tests
Part A (19 tests, no GPU) - **all pass**, see `test_report_part_A.txt`: binary_collapse on real label sets, 0.4375 (CHN) / 0.4496 (ARG) baselines from Table 2, paper-rule scorer vs reference remap, two scorers to 1e-12, stop rules, absent classes, early stopping, class weights, plan rules, sampling invariants (same key count at k=1,2,4,8), statistics, atomic writes, and a full dry run with a stand-in model (resume, integrity abort after 5, skill warnings never abort). Separately run: strict data-verification stop paths (count mismatch, leakage) and a full plan -> seed 0 -> approval -> seed 1 -> summary pass on synthetic data.
**Not yet run (needs Colab):** Part B (authors' own `remap_prediction`), real data verification, torch smoke test.

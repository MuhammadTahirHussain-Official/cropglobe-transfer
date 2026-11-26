# CropGlobe Level 1 - v3.1 (dual-track) update: change plan, change log, tests, open decisions

Files: `CropGlobe_Level1_v3_1_DUALTRACK_COLAB.ipynb` (new) | `test_report_part_A_v3_1.txt` | this file.
The earlier `CropGlobe_Level1_v3_FIXEDMIX_COLAB.ipynb` is left untouched. The v3.1 notebook is built from it (same cells, patched), not rewritten.

## 0. Verified facts vs assumptions
**Verified (read in this session):**
- arXiv 2509.03497v3 sec. 4.4 "Category Alignment": predictions of classes outside the target's label set are reassigned to Other **only when AUS or CHN is the target**. The paper says "no additional alignment is required for other transfer scenarios", so ARG and FRA get none.
- Table 1: CHN classes used = Rice, Corn, Soybeans + Other; AUS = Cotton, Sugarcane + Other. Table 2 counts and label sets match the notebook.
- The paper trains **50 epochs** (no early stopping) with Adam 1e-4 and batch 256. Table 3 mF1 values: ARG 75.63, AUS 88.38, FRA 90.00 (no USA->CHN).
- The paper uses both Sentinel-2 medians and AlphaEarth as separate representations. This notebook loads only `median_S2_*` files, with shape (N,1,10,T); AlphaEarth is 64-d. This is now asserted (file name and shape).
- The HF listing has the `median_S2_*` files.

**NOT verified (could not be read: github sub-pages are robots-blocked):**
- The authors' `util/label.py::remap_prediction` and the repo commit `bdcf63ec...`.
- The "existing" ARG branch (wheat->Other) is therefore only the spec's transcription of the release. It **disagrees with the paper text** (ARG: no alignment). Part B in Colab compares all three rules with the live authors' function and reports the result.

**Assumptions I made are flagged in section 4.**

## 1. Change plan (as requested, before the changes)
**Preserve (correct, kept as is):** per-crop training counts in every run; validation used only for early stopping/checkpoint selection; state combination as the unit of evidence (seeds averaged first); k=1 census of every pool state; 5 seeds; 95% t-intervals with finite-population correction + jackknife; model/feature inputs/labels/budgets/key shares/eligible states/seeds (unchanged); repo pin, namespace guard, atomic writes, prediction saving, resumability.
**Required fixes (explicit in the instruction) - done:** see section 2.
**Needs verification or approval - NOT applied:** see section 4.

## 2. What changed, and why
| # | Change | Why |
|---|---|---|
| 1 | Three **separately named remap rules**: `existing_release` (preserved unchanged; alias `remap_paper_rule` kept), `paper_text_v3` (verified paper sec. 4.4), `review_targetmap` (review: every class the target map does not define -> Other). Each is scored by both scorers with its own baselines, coverage-conditional F1, per-class F1, confusion matrix. | Review vs code vs paper disagree; none is overwritten. They differ only on ARG (existing/review: wheat->Other; paper text: none) and FRA (review: sugarcane/cotton->Other; existing/paper text: none). CHN and AUS agree everywhere. |
| 2 | Tracks kept apart in the results file and summary: `primary_f1` (existing primary), `binary_f1`, `paper_f1` (existing), `sevenclass_f1` (existing raw, no remap; macro over classes present in truth), `papertext_f1`, `review_f1`. Stats/CI/regression/gain tables are produced per track and per composition. | "Keep seven-class, binary-collapse and review scoring clearly separated." Never compare binary with multiclass across targets. |
| 3 | **Stop** if the binary truth has <2 classes (ARG without soybeans / CHN without rice) in addition to the existing >=2 true-class stop. | Review item 2. |
| 4 | Class weights from **training labels only** (`training_class_weights`); a class that exists only in validation gets weight 0. Column `class_weight_source` recorded. | Instruction 4. Reverses the earlier spec's train+val. |
| 5 | `MAX_EPOCHS` 50 -> **100**; `UNDERTRAINED_EPOCH` 45 -> 90 (assumption: 0.9 x cap). | Instruction ("approved 100-epoch maximum"). Reverses the earlier spec's cap; the paper itself uses 50. |
| 6 | **Eligibility**: `state_eligibility()` lists every state with need/have per fixed crop and an explicit reason for every exclusion; saved as `tables/eligibility_<mode>_<target>_B<budget>.csv` plus the full state-by-crop table; `combo_capacity_ok()` asserts every planned combination can supply its equal share. | Instruction 3; nothing excluded silently. |
| 7 | **Fixed-full composition mode** (`fixed_full`), opt-in and OFF by default; the baseline `key_plus_natural` is unchanged (identical draws, same run ids; tested). Two **proposals** (not defaults) are printed with their per-budget eligible-state counts. No run until `FIXED_COMPOSITION_CHOICE` is set and `APPROVED['fixed_composition']=True`. | Review item 4 without inventing proportions. |
| 8 | Regression uses the **realised target-crop count** (`target_crop_n`) vs k and budget; aliasing with budget is detected and reported, not hidden. | Review regression request. |
| 9 | `reconcile_results()` + optional cell (read-only comparison of two results files). | Review finding on the 5,004/5,590-row files; the files were not supplied. |
| 10 | Part B now **reports** differences between the three rules and the authors' function (saved to `tables/authors_remap_comparison.csv`) instead of stopping. It stops only if the authors' function cannot be called. | Instruction: preserve both until resolved. |
| 11 | Sentinel-2 guard in data verification (file name + (N,1,10,T) shape); `FEATURE_SOURCE` in the config/manifest. | Review item 1. |
| 12 | State-selector registry hook (`STATE_SELECTORS`; only `random` registered). | Where Phase 4 would plug in once defined. |
| 13 | `SPEC_TAG` bumped (new manifest); config items added to the locked-design check. | A folder from v3 is not mixed with v3.1. |

## 3. Tests: what was and was not executed
**Executed locally (CPU, synthetic data, stand-in logistic-regression model instead of CropNet):**
- Part A: **29 tests, all pass** (`test_report_part_A_v3_1.txt`). The 19 earlier ones were adapted to the 100-epoch cap and new column names. The 10 new ones cover: the three remap rules; the FRA review-vs-existing difference; the seven-class track; the binary-truth stop; train-only class weights; eligibility reasons and combination capacity; the fixed_full mode (exact counts at k=1,2,4,8, unapproved -> excluded, baseline unaffected); the unchanged baseline draw; reconciliation; the S2 guard text.
- Integration run of the plan, run, summary and gate cells (430 stand-in runs): the approval gates block seed 1 before seed 0 is approved; the summary reports both tracks; the legacy namespace is refused.
- Data verification (miniature Table 2): passes with S2 files and stops on an AlphaEarth-named file, a 64-d shape, a count mismatch and a train/target identical row.
- Part B logic against **fake** authors modules only (matches-existing, matches-paper-text, uncallable signature). This tests the comparison code, not the authors' real function.
- Syntax check of every code cell.

**NOT executed (needs Colab):**
- Part B against the real authors' repo; real-data verification against Table 2; the commit pin check; the torch/CropNet smoke test; any training. No GPU run and no real-data run was made.

## 4. Unresolved decisions and assumptions requiring approval
1. **Which remap is the "paper rule" / review primary?** The three rules are all scored. The decisive evidence (the authors' released function) is unread; the paper text contradicts the existing ARG branch. Also decide whether the review rule should apply to ARG/FRA (the paper text says it should not).
2. **Fixed-full composition values.** None were chosen. Proposals A (key share + equal rest over target-defined classes) and B (equal over target-defined classes) are printed with capacity checks. Either one gives 0 samples to USA crops the target map does not define, which makes them "absent from training" by design. Decide the proportions (or provide your own).
3. **Phase 4 diversity-aware selection:** no definition exists in the notebook, the review excerpt, or anything I can access. It is **not implemented**, and I have not invented one. Please supply the method (or the earlier code).
4. **Budgets:** the review says ~1,000/2,000/4,000; the notebook still runs 1,000/2,000/3,000/4,000. Dropping 3,000 changes the design, so I did not apply it.
5. **100-epoch cap and train-only weights** were applied as instructed but reverse the earlier spec and the paper (50 epochs). `UNDERTRAINED_EPOCH=90` is my assumption.
6. **China as a "secondary target with a feasible state sweep":** the plan encodes k<=4 with the feasibility rules; whether this holds on the real state table is only known after the plan table in Colab.
7. **AUS:** still excluded (no key share specified); needs `AUS_KEY_SHARES_IF_CONFIRMED` and `AUS_SPEC_CONFIRMED=True`. Whether AUS is eligible at all is also undecided.
8. **5,004/5,590-row reconciliation:** not performed (files not supplied); the tool is ready.
9. Earlier interpretation choices (pool definition, natural remainder, seed-dependent draw, FPC form, majority baseline) are unchanged from `CHANGES_AND_CONFLICTS.md`.

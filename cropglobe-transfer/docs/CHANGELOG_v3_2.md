# CropGlobe Level 1 v3.2 (authors-aligned, final) - change list, tests, open decisions

Files: `CropGlobe_Level1_v3_2_FINAL_COLAB.ipynb` | `test_report_part_A_v3_2.txt` | `test_report_part_B_logic_v3_2.txt` (three-rule table) | `test_report_data_verification_v3_2.txt` | this file.
Built from v3.1 (same 14 code cells, patched). Older files are untouched.

## 1. Errors fixed
1. **Part B was not a gate** (you were right): the name heuristic passed the predictions array as `crop_label`, silently fell back to a lowercase target (a no-op), and `TESTS_B_PASSED` was unconditionally True. Now: exact signature check `(test_pred, target, crop_label)`; call `remap_prediction(list(y_pred), target, CROP_ID)` with uppercase target and no fallback; ARG/CHN/AUS probes must actually change the input (a no-op STOPs); `existing_release` must be identical on all four targets or the run STOPs. Only then `TESTS_B_PASSED=True`. Three-rule table saved to `tables/authors_remap_comparison.csv`.
2. **Summary cell crash** found by the integration run (a literal `%` in a %-formatted header). Fixed.
3. Aliased regression term (target-crop count = share x budget) replaced by `n_target_classes_train_ge30`.

## 2. Training: authors' recipe, 100 epochs + early stopping (your instruction)
Read from `run_CropNet_AUGM.py`: Adam lr 1e-4 (default weight decay), no scheduler, batch 256, val/test aug none, class weights `compute_class_weight('balanced')` over the whole source set, no early stopping, 50 epochs.
Notebook: same optimizer/batch/scheduler/loss/model; **MAX_EPOCHS=100 with early stopping inside** (patience 15, min 20, strictly greater, best weights restored and verified); class weights default `train_val` (authors-equivalent; zero-count crops -> weight 0 because the head has 7 outputs). Printed deviations in the plan cell: 100 epochs+ES (authors 50); val 10% of the budget (authors 20%); seeds 0-4 (authors 82-91); no train augmentation at Level 1 (authors AUGM uses it).

## 3. Spec v3.1 items
- Both tracks side by side (EXISTING and REVIEW) plus `paper_text_v3`; never pooled/averaged/ranked; neither called "the paper's rule"; own baselines and cov-cond F1 per track; cov-cond printed beside the FRA headline.
- Plan note for ARG recorded verbatim. `n_target_classes_train_ge30`, per-target-and-k coverage stats in gate report. `steps_per_epoch`, `total_optimizer_steps` in results.csv and gate report.
- LOCKED: AUS_SPEC_CONFIRMED, AUS_KEY_SHARES_IF_CONFIRMED, TARGETS_TO_RUN, TARGET_CROPS, UNDERTRAINED_EPOCH, REVIEW_RULE, TRACKS, GEOGRAPHY_RULES.
- Geography STOP (sugarcane only LA/FL/TX/HI; rice only AR/CA/LA/MS/MO/TX; cotton reported; tolerance 0 rows).
- Exhaustive cells (C(P,k)<=10) flagged, no error bars drawn as certainty (hollow diamond; k=1 census hollow square).

## 4. What was executed (local CPU, synthetic data, stand-in models) and what was NOT
Executed: Part A **37/37 PASS**, run from the notebook's own cells (cells verified byte-identical to tested files); Part B logic vs faithful stand-in of the authors' function: PASS (paper_text_v3 differs on ARG only, review_targetmap on FRA only, existing identical) and STOP on no-op, wrong signature, ARG not remapped, CHN missing; data-verification stop paths (raw count, class count, AlphaEarth name, 64-d shape, leakage, missing label): all STOP, clean passes; end-to-end integration (plan -> seed 0 -> approval gate blocks seed 1 -> seed 1 -> summary, 248 runs, both tracks); syntax of every code cell.
**NOT executed:** the real authors' repo and real `remap_prediction`, real data vs Table 2, torch/CropNet smoke test, any training, Colab. Part B on the real repo is the decisive run and happens in Colab.

## 5. Unverified / risks
- Authors' `util/label.py` unread (GitHub subpages blocked); the sugarcane state list is general knowledge. The strict geography tolerance (0) may trip on legitimate FL/TN rice traces; if so it STOPs with the states named, and you decide.
- Phase 4 diversity-aware selection: not implemented (no definition supplied).
- 5,004/5,590 reconciliation: files not supplied; tool included.

## 6. Decisions for you (not made by me)
a) MAX_EPOCHS=100 (applied as you asked; paper and authors use 50). b) Class weights: train+val (default, authors-equivalent) vs train-only (`CLASS_WEIGHT_SOURCE='train'`). c) AUS key share / whether AUS runs (currently excluded). d) Run `fixed_full`? Which proposal (A or B)? Also budgets (3000 kept).

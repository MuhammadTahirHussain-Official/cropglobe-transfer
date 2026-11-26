# v3.6 change log (since v3.5) - ONE addition: optional EXTENSION profile

## What was added (nothing existing was edited in behaviour)
- `PROFILE = 'main' | 'extension'` (config cell; default `main`). `extension` runs CHN and FRA at budgets 3000 and 4000, random arm only, in `<outputs>/extension/` (own manifest, results.csv, runs). The main folder is never written (the USA state-table cache in `<main>/cache` is shared read/reuse).
- Allocation `ALLOC_RULE='capacity'` (extension only): each fixed crop and the natural remainder are split over the k states as equally as each state's capacity allows (`water_fill`); a short state gives all it has.
- Feasibility (`combo_feasible_pooled`): combination holds every quota together AND each state can give >= `EXT_MIN_CONTRIB` (0.5) of its equal share (k=1: the whole quota). k=1 census exempt from MIN_COMBOS in the extension only.
- `verify_split` extension checks: within capacity, >= floor, "as equal as capacity allows".
- Results columns: `profile`, `state_rows`, `max_state_share`. Run ids get `|prof-extension` only in the extension. Summary/plan print an APPENDIX banner.
- Test gate: tests now always run in the main profile (`profile_scope`); new tests E1-E5.

## Unchanged (verified)
- Main plan identical to v3.5 on the real 48-state table (415 cells with 3 arms; test_report_v3_6_real_table_plan_check.txt). Main `draw_split` byte-identical to v3.5 (frozen hash). Main LOCKED list unchanged; EXT_* keys locked only when the extension is selected.

## Real-table extension plan (offline planner, no training)
90 cells = 450 runs with 5 seeds: CHN 3000: k=1 4 states, k=2 and k=4 10 combos each; CHN 4000: k=1 2, k=2/4 10 each; FRA 3000/4000: k=1 2, k=2/4 10 each. FRA k=8 not feasible (reported). max_state_share ranges up to 1.0 (k=1 census, by definition).

## How to read the results (to avoid misleading conclusions)
- Appendix only; never pool with main results.csv. Compare with main at the same target and k, not across profiles by pooling.
- Combinations overlap heavily (few states): intervals are descriptive.
- Look at max_state_share: a k=4 cell where one state supplies most of the budget is not a diverse-state result.
- FRA rice absence in training (found earlier) still applies: report coverage-conditional F1 beside main F1.

## NOT executed by me
Real data, torch/GPU, Colab/Kaggle runs, and the dataset downloads. Executed: Part A + A2 (all pass incl. E1-E5), main and extension end-to-end dry runs with a stand-in logistic regression on synthetic data (extension dry run only produced FRA rows; the synthetic USA has no rice so CHN is not exercised end-to-end), real-table plan checks.

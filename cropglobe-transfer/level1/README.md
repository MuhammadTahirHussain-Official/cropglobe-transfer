# Level 1 – USA as the source (frozen)

**Design:** source = USA, the states are the units (a state enters the pool with ≥ 1,500 rows and when it can supply the whole budget alone). Targets: **ARG** (key crop soybeans 20 %, k up to 8), **FRA** (corn/soybeans/wheat 15 % each, k up to 8), **CHN** (rice 20 %, k up to 4), **AUS** (appendix census of {LA, FL}, k ≤ 2; excluded until its key-crop share is decided). Budgets 1,000–4,000, k ∈ {1,2,4,8}, seeds 0–4. Final run: 3,160 runs, SHA-256 `38d1876cde08e54cfc827aa6b2813355ef98a09dbd048885a67d58991534f891` (see `../docs/LEVEL1_FREEZE.md`).

**Arms in the code:** `random` (the baseline all claims use), `diverse` / `clustered` (state-diversity selection), `common` (a fixed pool of states available at every budget), `kcenter` (sample selection inside states; seeds 0–1 only), `aug_on` (the authors' augmentation, ARG budgets 1,000/2,000) and `aug_copy` (steps-matched control: the same rows copied 3× without any transform). An optional `extension` profile runs CHN and FRA at budgets 3,000/4,000 in its own sub-folder.

**Scoring:** ARG and CHN = binary key-crop macro-F1 (soybeans / rice versus everything else); FRA = coverage-conditional macro-F1 (classes with ≥ 30 training samples); the paper-style macro-F1 and the all-other / majority baselines are stored with every run. Pre-declared claims C1–C5 are in `PREREG_TEXT` in `src/cells/c02_config.py` (C1 mixed-model log2(k) effect on ARG, C2 fewest-to-most-states gain per budget, C3 `aug_on − aug_copy`, C4 k-center versus random, C5 share of combinations that beat the best single state and worst-case gain).

## Layout

| Path | Content |
|---|---|
| `notebooks/CropGlobe_Level1_v3_8_COLAB.ipynb` | **final** Level 1 notebook (Colab, GPU) |
| `notebooks/CropGlobe_Level1_v3_8_KAGGLE.ipynb` | same logic for Kaggle (two cells differ: environment and data acquisition) |
| `notebooks/archive/` | every earlier version, v3 fixed-mix … v3.7, kept for history – do not run them for new work |
| `src/cells/` | the notebook cells as plain `.py` files (`c01_env` … `c23_export`; `*_kaggle.py` are the Kaggle variants). `tools/check_notebook_matches_cells.py` verifies that the notebooks and these files are identical |
| `closing/` | `Level1_closing_and_Level2_probe.ipynb` and its sources: freeze check of the results by SHA-256 (`c30_closing.py`), statsmodels cross-check of the mixed model, training-quality robustness table, exact raw-CSV-to-npz row matching (`c33_match.py`) used to verify the France region assignment and USA wheat, and a data probe (`c31_probe.py`) |
| `tests/` | `integ38.py` (full synthetic flow: gates, run, resume, canary/plan guards – items F, G, H print `False` on purpose, they are the negative tests), `integ_auto.py`, `integ_ext.py` (extension profile), `t_probe.py`, and tests that need real files and skip otherwise: `t_closing.py` (set `LEVEL1_RESULTS_CSV`), `t_match.py` and `t_match2.py` (set `CROPGLOBE_FILES_DIR`) |

### Cell files

`c01_env` Colab environment · `c02_config` locked settings, approvals, pre-registered claims · `c04_core` scoring (two scorers), gates, sampling, plan, statistics, atomic outputs · `c07_tests`, `c10_tests_data` unit and data tests · `c09_data` strict verification of the data against paper Table 2 · `c11_acquire` data and pinned authors' code, namespace guard · `c12_testsB` our scorer against the authors' own `remap_prediction` · `c13_load` load and verify · `c15_plan` plan table · `c17_torch` CropNet backend and smoke test · `c19_run` run (resumable) · `c21_summary` tables, mixed model, figures · `c22_reconcile` compare two results files · `c23_export` export the state table.

## How to run (Colab)

1. Put the project folder on Drive; edit the candidate paths in `c01_env.py` (cell 1). Data goes to `<project>/CropGlobe`, results to `<project>/outputs/phase3_level1_v3_7_review` (the v3.8 notebook keeps this folder so an interrupted run resumes with the same run ids).
2. Run the cells in order. The test gate (parts A, A2, B) and the smoke test must pass; every mismatch with paper Table 2 is a STOP.
3. Read the plan table, then set `APPROVED['plan'] = True` (done automatically in `AUTO_MODE`, the default).
4. Run the run cell; run it again after any disconnect.
5. After all seeds: the summary cell, then the closing notebook in `closing/` (it recomputes the SHA-256 of `results.csv` and stops if the hash or the per-seed run counts differ from the freeze record).

## Notes

- A run started with an older version resumes only if the locked design is identical; new settings are recorded as dated *amendments* in `manifest.json`.
- Not done in Level 1 (needs data/GPU not available when it was written): GroupDRO / state-balanced batches, point-in-polygon alignment check. See `../docs/CHANGELOG_v3_8.md` and `../docs/VERIFICATION_v3_8.md`.

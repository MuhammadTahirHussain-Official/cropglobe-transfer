# v3.7 change log (since v3.6) - implements the supervisor review of the 409-run seed-0 results

Nothing in the locked design (budgets 1000-4000, key shares, strict pools, combinations, seeds, validation split, optimiser) changed. Base plan (random/diverse/clustered) is **identical to v3.5/v3.6 on the real 48-state table** (415 cells; checked). Results go to a NEW folder `phase3_level1_v3_7_review`.

## Review claims checked against your seed-0 file (409 runs) BEFORE changing code
| Claim | Result |
|---|---|
| ARG soybean-vs-other 0.616 / 0.662 / 0.703 / 0.718 at B1000 k=1/2/4/8 | matches `primary_f1` (this is the binary macro-F1 of soybeans-vs-other, not the soybean-class F1; the soybean-class F1 is 0.396/0.456/0.508/0.530) |
| "~2.2 points per doubling of states" | NOT reproduced: simple slope on B1000 gives ~3.3 points (primary) - the reviewer probably controlled for something; quote only what the final seeds support |
| 1000 -> 2000 labels adds ~0.9 points | roughly: +4.5, +2.0, -0.2, -1.1 at k=1,2,4,8 (mean ~+1.3) |
| Eligible ARG pool shrinks 16 -> 6 | confirmed (16/11/8/6) |
| 0% of 1-state runs vs 38% of 8-state runs hit the 100-epoch cap (ARG) | confirmed (0%, 16%, 30%, 38% for k=1,2,4,8) |
| CHN B2000 k=1 outlier is Louisiana 0.60 | confirmed (0.598) |
| FRA rice never in training, rice F1 = 0 | confirmed (train_rice max = 0 over all FRA runs) |
| Seed noise ~1.7 points; capped runs score higher; diversity-score correlation ~0.1 | NOT checkable: the file has seed 0 only |
| "Argentina soybeans only / China rice only" correction | agrees with our code: ARG 6 classes, CHN 4; binary view is a stated choice (`PRIMARY_VIEW`) |

## What changed (R1-R8)
- **R1 epochs**: `MAX_EPOCHS` 100 -> 200 (early stopping unchanged). Four tests that asserted 100 now assert 200. New output folder so cap-100 results are not mixed in.
- **R2 metrics**: `REPORT_PRIMARY='main_f1'`, `REPORT_SECONDARY='paper_f1'`. Summary tables, headline and plots show these two (+ coverage-conditional versions); `review/papertext/sevenclass/binary` columns are still computed and stored, but only summarised if `REPORT_ALL_TRACKS=True`. The per-seed gate report still lists all tracks (audit).
- **R3 France**: `FRA_MAIN_RULE='coverage_conditional'`: `main_f1` for FRA = coverage-conditional macro-F1 (classes with >=30 training samples; rice excluded in practice). ARG/CHN `main_f1` = `primary_f1`. `main_f1` is also filled for old results files on load (existing values never overwritten).
- **R4 input features**: column `input_features` (Sentinel-2 5-day medians, official DOY 121-334 window) in every row; locked in the manifest.
- **R5 common pool** (arm `common`): combinations of every budget are re-drawn from the states eligible at ALL budgets of the target (ARG 6, CHN 6, FRA 5 states; so k=8 is not available there). A combination already in the random arm is the same run (arms `random|common`), not re-trained. Table `common_pool_budget_by_k.csv`.
- **R6 k-center** (arm `kcenter`): same state combinations and the same per-state/per-crop counts as the random arm (k=1 census + up to 5 combinations per cell, deterministic subset); only WHICH rows of each (state, crop) pool are taken changes: greedy farthest-first in a 16-dim PCA space of the USA window features (USA rows only). Paired contrast `kcenter - random` on identical combinations. **This is my reading of "select individual samples with k-center"; I do not have the original Phase-4 text, so please confirm the embedding/space and the per-stratum design with your supervisor.**
- **R7 augmentation ON** (arm `aug_on`, ARG, B1000/2000, k=1 census + up to 5 combos): the authors' `CropGlobeDataset(aug_mode='augm')` from the pinned commit (time-shift p=1, time-scale p=.5, magnitude-warp p=.25; original + 2 augmented copies; validation/test never augmented). Their rng is unseeded, so it is seeded here (recorded deviation). **PENDING** until `APPROVED['aug_on']=True`; uses 3x the optimizer steps per epoch (`steps_per_epoch`, `train_n_effective` recorded) - the contrast mixes augmentation with training length. No step-matched control is built. NOTE: no earlier notebook ever ran augmentation ON (it was hard-asserted off).
- New columns: `aug`, `sample_sel`, `input_features`, `main_f1`, `main_metric`, `train_n_effective`. Run ids: `aug-on` / `|sel-kcenter` only for the new runs; all existing run ids unchanged.
- Tests: new V37-01..06; the end-to-end dry run now includes k-center and aug-ON cells; test cells always run in the main profile.

## NOT done / needs a decision
- **Fixed common pool for CHN** equals the full pool (6 states at both budgets) - shared with random, nothing new to train.
- Plan size: with all arms on, 599 runnable cells = ~3,000 runs over 5 seeds (+285 aug-ON runs pending). At ~33 s per run (more with 200 epochs) that is >30 GPU-hours; trim `SELECTION_ARMS` (e.g. drop 'diverse','clustered', which showed no effect) to cut it.
- Seeds 1-4 and the tables with confidence intervals require the real runs; **nothing here was run on real data, a GPU, Colab or Kaggle**.
- Still open from before: USA state alignment (`alignment_report.json`), the Kaggle dataset mount on your side, AUS spec.

## Executed here
Part A + A2 tests (66 pass, 0 fail), end-to-end dry runs (main with k-center and aug-ON cells, and the extension profile) with a stand-in logistic regression on synthetic data, offline plan checks on the real state table, simulated-/kaggle tests (downloads untested). I read the authors' `data/augment.py` at the pinned commit.

## NOT executed (torch could not be installed in this sandbox)
The authors' augmentation, the CropNet model and every GPU path. The torch cell therefore has new self-checks that run in YOUR smoke test before any sweep: aug-ON gives exactly 3x rows, the first block equals the unaugmented window, the result is identical under the same seed, a 2-epoch aug-ON fit runs, and one real k-center and one real aug-ON cell go through the full pipeline with integrity checks. If any of them fails, stop and send me the message.

## Addendum: AUTO MODE (full Level 1 on its own)
- `AUTO_MODE = True` (default, config cell): plan and `aug_on` are pre-approved; the run cell calls `run_all_seeds()`: seeds 0..4 in order, every planned run (aug OFF and aug ON cells in the same seed pass), resumable.
- Between seeds the gate report is auto-approved ONLY if that seed is complete with no `error` / `integrity_fail` row (`auto_gate_ok`; the latest row per run counts, so a failed run that later succeeds no longer blocks). Skill warnings never block (unchanged policy). Otherwise it stops with the reason and the next seed does not start.
- `AUTO_MAX_HOURS` (e.g. 11 on Kaggle): pauses cleanly between runs at the limit; run the cell again to resume.
- Still hard stops in both modes: test gate, smoke test, data/geography/state-alignment checks, 5 consecutive integrity failures. `AUTO_MODE = False` restores the manual one-seed-at-a-time approvals.
- Verified in the stand-in dry run (synthetic data, logistic regression): (1) a tiny time limit pauses with 0 rows; (2) one failing run in seed 0 -> stopped, seed 1 not started; (3) re-run retries the failed run and completes both seeds incl. aug-ON rows with gate reports for each seed. Not run on real data / GPU.
- Cost: the default plan is ~3,000 runs for 5 seeds plus aug-ON; auto mode will simply keep going for as long as that takes (30+ GPU hours), across several sessions. Remove arms from `SELECTION_ARMS` to shorten it.

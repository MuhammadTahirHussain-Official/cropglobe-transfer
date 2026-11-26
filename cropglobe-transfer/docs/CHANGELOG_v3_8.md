# CropGlobe Level 1 v3.8 — change log (since v3.7)

**Design frozen.** Same results folder (`phase3_level1_v3_7_review`), same LOCKED list, same run-id format. The existing `results.csv` (seeds 0–1, start of seed 2) resumes without change. New items are recorded in `manifest.json` as dated *amendments* (V38-A1, A2, A3, A4, P1) the first time the run cell executes.

## What changed
| ID | Change | Where |
|---|---|---|
| V38-A1 | `KC_SEEDS=[0,1]`: k-center cells exist only for seeds 0–1. Seeds 2–4 run random + augmentation only (about 465 fewer runs). Existing k-center rows stay valid. | config, `plan_for_seed` |
| V38-A2 | New arm `aug_copy`: same cells as `aug_on`, training rows copied 3× with **no** transform (same rows per epoch and optimizer steps). Decisive contrast: `aug_on − aug_copy`. Runs for seeds 0–4 (seeds 0–1 are new runs). | plan, `fit_model` |
| V38-A4 | Seed-consistency guards before every run: (1) plan check — every run id on disk must be in the current plan; (2) canary — re-run 3 finished runs (aug off, aug on, k-center) and compare with stored scores (tolerance 1e-6, epochs equal). Same device + mismatch = STOP; different device = reported only. | `run_guards()` |
| V38-P1 | Pre-declared analysis: seeds 0–1 exploratory, seeds 2–4 confirmatory; claims C1–C5; a claim passes only with the same sign in both groups and a confirmatory interval excluding 0. Failures are reported as NOT REPLICATED; no seed is dropped, replaced or re-run. Disclosure of the 67 partial seed-2 runs I glimpsed is included. | `PREREG_TEXT` |
| V38-A3 | Summary adds: mixed model (state combination as random intercept; own REML code, no statsmodels), average gain, worst-case gain, share of combinations beating the best single state, exploratory-vs-confirmatory replication table. Plan cell prints the state-alignment evidence. | summary / plan cells |
| — | `run_environment_seed<N>.json` is never overwritten (a resume writes a time-stamped copy). | `write_run_environment` |

## Not in v3.8
France-as-source replication, GroupDRO / state-balanced batches, point-in-polygon alignment check, and the Drive-safe results saving (proposed earlier, never approved).

## Tests
72 Part A / A2 tests pass (66 → 72: V38-01 … V38-06). See `VERIFICATION_v3_8.md` for what was and was not executed.

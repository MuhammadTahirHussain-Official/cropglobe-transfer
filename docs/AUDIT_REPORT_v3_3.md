# Verified audit report - CropGlobe Level 1 (v3.2 -> v3.3)

Sources actually read this session: the 2 findings you pasted; the audit brief; arXiv 2509.03497 v3 PDF (through the fetch tool's summary); raw files of github.com/x-ytong/CropNet: `util/label.py`, `data/loading.py`, `run_CropNet_AUGM.py`, `trainers/torch_trainer.py` (also through the fetch tool's summary, with short quotes - not a byte-level reading). **Not available:** CropGlobe data, Colab, torch, GPU, pandas 2.x. Everything marked "run" ran on CPU with synthetic data and stand-in models.

## 1. Verification table
| # | Claim | Verdict | Evidence | Action |
|---|---|---|---|---|
| U1 | `gate_report` crashes if an error row is first in a budget / target-k group | **Confirmed bug** | Reproduced on v3.2 (error row first: `cannot convert float NaN to integer`; last: fine). Resume re-runs `error` rows and appends after the old row, so it occurs in normal use. `integrity_fail` rows from the exception path are blank too. | Fixed V33-01 |
| U2 | Geography tolerance 0 can block on border rows | **Risk, not confirmed on data** | Code: tolerance 0, 1:20m Census file confirmed. Counts depend on data not available here. | Message improved (V33-03); tolerance unchanged; your decision after the printed counts |
| U-note | Guard cannot catch swaps between allowed states | **Correct** | By construction | Limitation stays; an exact row match to raw CSVs would close it (needs those files) |
| U-note | AUS in LOCKED; SPEC_TAG changed so a v3.1 folder must be moved | **Correct** | `LOCKED` list and `check_namespace` | None; v3.3 changed neither |
| 4 | `gate_report` empty-flags `.str` failure (2 tests) | **Not reproduced; unverified for pandas 2.x** | 37/37 passed here (pandas 3.0.5); all flag variants work on the OLD code; pandas 2.x could not be installed | Replaced by explicit parse (V33-02); cannot fail either way |
| 5a | `y_pred` remapped, `y_true` not | **Not an issue** | Authors' `remap_prediction` returns only remapped predictions; their test mF1 = `f1_score(average='macro', labels=np.unique(test_true))`; target data pass the Table 2 label check, so remapping y_true would be the identity | None; guard added for out-of-label truth (V33-08) |
| 5b | existing_release reproduces the release | **Confirmed** | `label.py`: AUS keep sugarcane+cotton; CHN keep corn/soy/rice; ARG wheat->other; others unchanged; Part B compares on 5000 random predictions (against a stand-in; the real call happens in Colab) | None |
| 5c | `papertext_v3` implements the paper | **Supported by text** | Sec. 4.4: only AUS/CHN reassign undefined predictions to Other | None; differs from release on ARG only |
| 5d | review rule supported | **Design choice** | Extends the label-set rule to ARG and FRA; the paper does not state it for those | Kept as a separate track |
| 5e | `DEFINED_CLASSES` from Table 2 non-zero counts | **Partly supported** | Table 2: ARG no wheat, AUS only sugarcane/cotton/other, CHN corn/soy/rice/other, FRA no sugarcane/cotton; matches the release for AUS, CHN, ARG; FRA's rule is the review proposal | None |
| 5f | single-class / empty / out-of-label truth | single-class: stops; empty: stops; out-of-label: **was scored silently** | probes | V33-08 |
| 6 | Class weights `train_val` | **Matches the authors** | Authors compute balanced weights over the whole source set before the split; test labels are never used; `train` option exists | None; decision (b) below |
| 7 | Composition modes / fixed composition | **Confirmed as configured** | `COMPOSITION_MODES_TO_RUN=['key_plus_natural']`, `FIXED_COMPOSITION_CHOICE=None` | Your decision |
| 8A | Pool = states able to supply the whole budget alone | **Confirmed (design choice)** | `state_eligibility` / `eligible_states`; `combo_capacity_ok` is looser (equal share per state) | Your decision |
| 8C | MIN_COMBOS also rejects k=1 | **Confirmed** | `t != 'AUS' and len(combos) < MIN_COMBOS` applies to k=1; this follows your first spec | Your decision |
| 8E | Only random selector | **Confirmed** | `STATE_SELECTORS={'random':...}`; no Phase 4 definition supplied anywhere | Need a definition |
| 9A | Alignment ambiguity only warns | **Confirmed** | `build_usa_meta` | V33-06 |
| 9B | Cache not checked against `npz_row` | **Confirmed** | `load_all_data` | V33-04 |
| 10 | Silent file choice | **Confirmed** | `find_npz` | V33-05 |
| 11 | Finite-loss on last batch only | **Confirmed, low severity** | `fit_model` | V33-07 (not executed: no torch) |
| 11 | Authors' recipe | **Confirmed** | Adam 1e-4, batch 256, no scheduler, 50 epochs, no early stopping, best epoch by val mF1, test once at the end | None (100 epochs + early stopping is your instruction) |
| 11 | Authors' train/val use the full dataset | **Confirmed in their script** | `.dataset` of the random_split discards the split | Not copied; we split properly |
| 12 | Manifest content | **Partly** | Missing versions/device/data identity | V33-09 |
| 13 | OLS on overlapping combinations | **Confirmed** | `regression_report` | Labelled exploratory (V33-10); better method = decision |

## 2. Decisions that are yours (nothing was changed here)
1. **Pool rule**: keep "each pool state supplies the whole budget" (comparable pool across k) or let states contribute partially (more states eligible, different design).
2. **K=1 and MIN_COMBOS**: your first spec skips cells with <5 combinations, including k=1; the brief says a k=1 census should never be rejected for that. Choose.
3. **Phase 4 selector**: give the definition (or approve an outcome-independent one, e.g. maximise spread of state centroids), otherwise random stays and no diversity claim should be made.
4. **fixed_full composition**: run it? proposal A or B? AUS key share / AUS at all?
5. **Class weights**: train+val (authors) or train only. **Epochs**: 100 + early stopping (your instruction) vs the authors' 50.
6. **Geography tolerance** and **boundary file** (1:20m vs 5m): decide after the printed counts; changing the file changes state labels and would rebuild the cache.
7. **Statistics**: add state-clustered / bootstrap intervals or keep descriptive + jackknife.

## 3. Remaining issues
Real-data checks (Table 2 counts, 10-band/43-step window, label map, alignment, geography) have not run. Real `remap_prediction` call, torch/CropNet smoke test (includes the new NaN check), GPU training, Colab: all unexecuted. pandas 2.x untested. Phase 4 undefined.

## 4. Colab steps
1. Runtime > GPU. Run cell 1 (mounts Drive, installs). Set `DATA_DIR` (folder containing `median_S2_<C>_month1-12_day5.npz`) in the config cell.
2. Run config, core, Part A, data functions, Part A2, acquire, Part B - each must end "ALL PASSED" / "Part B PASS". Any STOP: send me the text.
3. Run load/verify: read the geography counts and the alignment output. A STOP there is a decision point, not a bug to bypass.
4. Run the plan cell, read the table, then set `APPROVED['plan']=True`. Run the torch/smoke cell.
5. `RUN_SEED=0`, run, read `gate_reports/seed0.txt`, set `APPROVED['seed0_gate_report']=True`, `RUN_SEED=1`, repeat to 4. Then the summary cell.

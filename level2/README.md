# Level 2 – France as the source (main experiment and 75/76 follow-up)

See the top-level README for the overall study; this file covers Level 2 only. Level 1 (USA as the source) is in `../level1`. This folder contains code and documentation only – no data, no results files, no reports.

Sources: paper arXiv 2509.03497 (v3) · authors' code `github.com/x-ytong/CropNet` (pinned commit `bdcf63ec1704a412da73c534c579fe08c5d2fa9b`, checked out automatically) · data `huggingface.co/datasets/x-ytong/CropGlobe`.

## Layout

```
notebooks/   two ready-to-run Colab notebooks (generated from src/cells by tools/build_notebooks.py)
  CropGlobe_Level2_FRA_source_COLAB_v3.ipynb      main experiment, seeds 0-4
  CropGlobe_Level2_FOLLOWUP_no75_76_COLAB.ipynb   separate pre-registered follow-up, seeds 5-9
src/cells/   the code of every notebook cell (single source of truth)
tests/       synthetic tests and end-to-end dry runs (no GPU, no real data)
scripts/     run_unit_tests.py, analyze_results.py
tools/       build_notebooks.py  (re-creates the notebooks from src/cells)
data/        fra_region_crop.csv – real France region x crop row counts, used only by the dry runs
docs/        change log and checklist of the review points (the France region-assignment note is in ../docs)
```

### Cell files (notebook order)

| File | Notebook cell |
|---|---|
| `c01_env.py` | 1 Colab environment (Drive, paths, GPU) |
| `c02_config.py` | 2 All locked settings and design decisions (L2-D1…D12) |
| `c04_core.py` | 3 Core code: sampling, scoring, re-scoring, planning, design hash, run loop, claim tables |
| `c07_tests.py`, `c10_tests_data.py` | 4, 6 Unit and data tests (77 tests; training is blocked if any fails) |
| `c09_data.py` | 5 France region assignment and data loading |
| `c11_acquire.py`, `c12_testsB.py` | 7, 8 Download data, pin authors' code, compare remap with the authors' function |
| `c11b_usa_order.py` | 8b Exact verification of the USA row order and labels against the raw files |
| `c13_load.py`, `c15_plan.py` | 9, 10 Load and verify data; plan table and **design hash** |
| `c17_torch.py` | 11 Model, training, smoke test (needs torch/GPU) |
| `c18_approval.py` | 12 Approval gate (paste the approved hash) |
| `c18b_rescore.py` | 12b Re-score finished USA rows from saved predictions (no retraining) |
| `c19_run.py` | 13 Run (resumable) |
| `c21_summary.py` | 14 Summary tables and figures |
| `c24_level2_analysis.py` | 15 Closing analysis: freeze, pre-registered claims, exploratory tables |
| `c25_appendix_labels.py` | 16 Appendix: Belgium/Netherlands/UK label counts (descriptive) |
| `c02f_followup.py`, `c24f_followup_analysis.py` | follow-up only: plan overlay (cell 6b) and its closing analysis |

## Design in short

- Budgets 1,000–4,000, k ∈ {1,2,4,8} where the pool allows it (a region must be able to supply the whole budget alone), up to 10 random region combinations per cell for k ≥ 2 (k = 1 is a census), 319 runs per seed, seeds 0–4 (1,595 runs).
- Seeds 0–1 are exploratory, seeds 2–4 confirmatory. A claim passes only if the sign agrees in both groups **and** the confirmatory 95 % interval excludes 0; otherwise the code prints NOT REPLICATED, or NOT TESTABLE when no sampling interval exists. China is "descriptive only". Verdict words are produced by code, never edited.
- Claims: **L2-C1** (mixed model, gain per doubling of regions), **L2-C2** (fewest → most regions per budget), **L2-C3** (augmentation vs its steps-matched copy, ARG, budgets 1,000/2,000).
- **USA scoring (L2-D11):** macro-F1 over corn, soybeans, wheat, other; true rice, sugarcane and cotton are removed from the test set and predictions of those classes count as "other" (the paper removes sugarcane and cotton for FRA→USA, §4.4; the extension to rice was an instructor decision). Cell 12b re-scores all finished USA rows from saved predictions.
- **Exploratory tables** (labelled as such, never change a verdict): each region mix vs the mean and the best of its own single regions, the fixed-pool budget effect, per-doubling slope with/without regions 75 and 76.
- **Follow-up (separate notebook, own results folder and design hash):** Argentina only, regions 75 and 76 removed from the source, k ∈ {1,2,4}, budgets 1,000 and 2,000, new seeds 5–9, plain training, 53 runs per seed (265 in total). Claims fixed in `FOLLOWUP_CLAIMS` (part of the hash): **F-C1** per-doubling effect > 0 with 95 % interval excluding 0; **F-C2** mean of (mix − mean of its own single regions) > 0 with interval excluding 0 and positive in ≥ 4 of 5 seeds.

## How to run (Google Colab, GPU)

1. Put the project folder on Google Drive (data goes to `<project>/CropGlobe`, the authors' repo to `<project>/CropNet`, results to `<project>/outputs/<experiment>`). Edit `_candidates` in `c01_env.py` (cell 1) to your Drive path. Put the verified France row map (`row_map*.csv`, SHA-256 checked in cell 5) anywhere under the project folder.
2. Open `notebooks/CropGlobe_Level2_FRA_source_COLAB_v3.ipynb` and run **cells 1–11** top to bottom. Cell 8b looks for the raw USA files `S2_USA_<crop>_<ST>.csv` under the project folder (or set `USA_RAW_DIR` in that cell) and prints `VERIFIED` / `FAILED` / `NOT VERIFIED`.
3. Cell 10 prints the **design hash**. Get it approved, then in cell 12 set `INSTRUCTOR_APPROVED = True` and paste the hash into `APPROVED_DESIGN_SHA256`. Any change to a locked setting, the plan or the scoring rule changes the hash and the run cell refuses to start.
4. If results from an earlier scoring rule exist, run cell **12b** once (backs up `results.csv` byte-for-byte, checks that saved predictions reproduce stored numbers, writes `rescore_log.json`).
5. Run cell **13**. It is resumable: run it again after a disconnect. It stops on any error or integrity failure.
6. After all seeds: cell 14 (summary) and the closing call at the end of cell 15 (`closing_report_l2(...)`), which freezes `results.csv` by SHA-256 and prints the pre-registered claims first, the exploratory tables after.
7. Follow-up: open `notebooks/CropGlobe_Level2_FOLLOWUP_no75_76_COLAB.ipynb`, same procedure (its own hash, its own results folder `phase3_level2_followup_no75_76_v1`), closing call `closing_report_followup(...)`.

## Testing without GPU or data

```
pip install -r requirements.txt        # torch / huggingface_hub are only needed on Colab
python scripts/run_unit_tests.py       # 77 tests embedded in the cells
python tests/test_usa_order.py         # 9 tests of the USA order verifier
python tests/dryrun_level2.py run      # synthetic end-to-end run of the main notebook logic (gate refusal, run, re-scoring, closing)
python tests/dryrun_followup.py run    # same for the follow-up notebook (265 runs)
python tools/build_notebooks.py        # regenerate the notebooks from src/cells
```

The dry runs replace the network by a logistic-regression stand-in and use the real region/crop counts with synthetic features: they test the **plumbing, not the science**. The PyTorch training code, the GPU path and loading of the real data are only exercised on Colab. The statsmodels cross-check of the mixed model also runs on Colab (or wherever statsmodels is installable).

## Recomputing the analysis from finished results

```
python scripts/analyze_results.py path/to/main_results.csv [path/to/followup_results.csv]
```

Prints the claim tables, C3, the exploratory tables and (if given) the follow-up claims. It does not train and does not write anything.

## Honest limits

- Results here are conditional on these 13 French source regions; intervals do not describe new regions.
- Region mixes share regions, so p-values of the exploratory mix-vs-own-regions comparison are descriptive.
- Belgium/Netherlands/UK are appendix only (label counts); they contain only corn, wheat and other and carry no claim.
- Changing any locked setting after approval requires a new plan and a new seed set.

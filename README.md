# CropGlobe / CropNet – cross-region transfer experiments (Level 1 and Level 2)

Code for a two-stage, pre-registered study of whether training CropNet on **more source regions** improves crop-type transfer to a new country, at a fixed labelled-sample budget.

| | Source | Target(s) | Notebook |
|---|---|---|---|
| **Level 1** (frozen) | USA, the 50 **states** are the units | Argentina (ARG), France (FRA), China (CHN), Australia (AUS, appendix) | `level1/notebooks/CropGlobe_Level1_v3_8_COLAB.ipynb` (Kaggle variant alongside) |
| **Level 2** | France, 13 **regions** are the units | ARG and USA (claims), CHN (descriptive) | `level2/notebooks/CropGlobe_Level2_FRA_source_COLAB_v3.ipynb` |
| **Level 2 follow-up** | France without regions 75 and 76 | ARG only | `level2/notebooks/CropGlobe_Level2_FOLLOWUP_no75_76_COLAB.ipynb` |

This repository contains **code and documentation only** – no data, no results files, no reports.

Sources: paper arXiv 2509.03497 (v3) · authors' code `github.com/x-ytong/CropNet`, pinned to commit `bdcf63ec1704a412da73c534c579fe08c5d2fa9b` (cloned and checked out by the notebooks, never modified) · data `huggingface.co/datasets/x-ytong/CropGlobe`.

## Repository layout

```
README.md  requirements.txt  .gitignore
scripts/run_all_tests.py        runs the unit/data tests of both levels (no GPU, no data needed)
docs/                           change logs v3.1-v3.8, audits and verification notes, Level 1 freeze record,
                                France row-assignment verification, Level 2 pre-registration draft
level1/                         see level1/README.md
  notebooks/                    final v3.8 (Colab + Kaggle) and archive/ with every earlier version
  src/cells/                    the code of every v3.8 notebook cell (single readable source)
  closing/                      Level 1 closing analysis + raw-to-packaged row match (notebook and sources)
  tests/                        end-to-end synthetic dry runs and tests of the closing/match/probe code
  tools/check_notebook_matches_cells.py
level2/                         see level2/README.md
  notebooks/  src/cells/  tests/  scripts/  tools/  data/  docs/
```

## The common design (both levels)

- **Budget** = total labelled samples per run (train + validation, 10 % stratified validation): 1,000–4,000.
- **k** = number of source units mixed: 1, 2, 4, 8 where the pool allows it. A unit enters the pool only if it has enough rows and can supply the whole budget alone; k = 1 is a census of every pool unit, k ≥ 2 uses up to 10 random combinations.
- **Key-crop share** fixed per target (ARG: soybeans 20 %, CHN: rice 20 %, USA/FRA: corn, soybeans, wheat 15 % each); the rest sampled naturally.
- **Model/training** (identical in both levels, as in the authors' script): 7-class CropNet, Adam lr 1e-4, batch 256, up to 200 epochs with early stopping on validation macro-F1 (patience 15, minimum 20 epochs), best weights restored.
- **Seeds** 0–4. In Level 1 and 2, seeds 0–1 are *exploratory* and 2–4 *confirmatory*; a claim passes only if the sign agrees in both groups **and** the confirmatory 95 % interval excludes 0, otherwise the code prints NOT REPLICATED (or NOT TESTABLE when no sampling interval exists). Verdict words are produced by code.
- **Safety gates:** the test gate must pass before training; the authors' repo must be clean at the pinned commit; results folders carry a manifest and are never mixed; Level 2 additionally requires an **approved design hash** (settings + every planned run + scoring rule) before the run cell will start; seed-consistency guards (plan check + canary re-runs) stop a resumed run if anything drifted.
- **Level 1 is frozen:** 3,160 runs, SHA-256 `38d1876cde08e54cfc827aa6b2813355ef98a09dbd048885a67d58991534f891`. Level 2 never reads or writes it.

## Quick start

```
pip install -r requirements.txt            # torch and huggingface_hub are only needed on Colab/Kaggle
python scripts/run_all_tests.py            # Level 1: 72 tests, Level 2: 77 tests, plus ordering/probe tests
```

To run an experiment, open the notebook of your choice in Google Colab (GPU) and follow the README of that level. The dry-run scripts in `*/tests/` replace the network by a logistic-regression stand-in: they test the **plumbing, not the science**. The PyTorch training code, the GPU path and the loading of the real data are exercised only on Colab/Kaggle.

## What is not in the repository

Data (downloaded by the notebooks), results (`results.csv`, `runs/`, `manifest.json`: produced by the runs, frozen by SHA-256 in the closing cells), reports and PDFs. `.gitignore` keeps them out.

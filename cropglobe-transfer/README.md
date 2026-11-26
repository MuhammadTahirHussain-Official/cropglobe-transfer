# Does mixing source regions improve crop-type transfer?

Code, documentation and reproduction scripts for the study **"Does mixing source regions improve crop-type transfer? A hash-locked, two-stage study with Sentinel-2 time series"**.

The study asks one question: if you can label only a fixed number of samples in the source country, is it better to take them from one region or to spread them over several regions? It trains CropNet on Sentinel-2 time series from the CropGlobe benchmark and tests the trained models in other countries.

| Stage | Source (the "units" that are mixed) | Target(s) | Runs |
|---|---|---|---|
| **Level 1** (frozen) | USA, the **states** | Argentina (claims), France and China (descriptive) | 3,160 |
| **Level 2** | France, 13 **regions** | Argentina and USA (claims), China (descriptive) | 1,595 |
| **Level 2 follow-up** | France without regions 75 and 76 | Argentina | 265 |

Main findings (details and caveats in the paper):

- US states to Argentina: each doubling of the number of states raised soybean-versus-rest macro-F1 by 2.37 points (95% CI 1.70 to 3.04).
- French regions to the USA: 1.65 points per doubling (0.43 to 2.86); less robust (see the paper's sensitivity analyses and deviations).
- French regions to Argentina: not replicated, also not in the follow-up.

---

## Contents

1. [What is in this repository](#1-what-is-in-this-repository)
2. [What is not in this repository, and where to get it](#2-what-is-not-in-this-repository-and-where-to-get-it)
3. [Requirements](#3-requirements)
4. [Use 1: check the code (no data, no GPU, about 5 minutes)](#4-use-1-check-the-code-no-data-no-gpu-about-5-minutes)
5. [Use 2: reproduce every table and figure of the paper (no GPU, about 2 minutes)](#5-use-2-reproduce-every-table-and-figure-of-the-paper-no-gpu-about-2-minutes)
6. [Use 3: re-run the experiments (GPU, Google Colab)](#6-use-3-re-run-the-experiments-gpu-google-colab)
7. [Where each table and figure of the paper comes from](#7-where-each-table-and-figure-of-the-paper-comes-from)
8. [Study design in brief](#8-study-design-in-brief)
9. [Integrity safeguards](#9-integrity-safeguards)
10. [Documentation index](#10-documentation-index)
11. [Citation, license and contact](#11-citation-license-and-contact)

---

## 1. What is in this repository

```
cropglobe-transfer/
├── README.md                this guide
├── LICENSE                  MIT
├── requirements.txt         Python packages (training-only packages are marked)
├── .gitignore               keeps data, results and caches out of the repository
├── scripts/
│   └── run_all_tests.py     runs the unit and data tests of both levels
├── level1/                  Level 1: US states as source (frozen)      -> see level1/README.md
│   ├── notebooks/           final v3.8 notebook (Colab and Kaggle); archive/ holds earlier versions (history only)
│   ├── src/cells/           the code of every notebook cell as plain .py files
│   ├── closing/             closing analysis: freeze check, mixed-model cross-check, row matching
│   ├── tests/               synthetic end-to-end tests
│   └── tools/               check that the notebooks equal src/cells
├── level2/                  Level 2: French regions as source, and the follow-up   -> see level2/README.md
│   ├── notebooks/           main experiment and follow-up notebooks (Colab)
│   ├── src/cells/           the code of every notebook cell (single source of truth)
│   ├── tests/  scripts/  tools/  data/  docs/
│   └── requirements.txt
├── paper/                   LaTeX source of the paper and the scripts that rebuild its tables and figures
│   ├── main.tex  refs.bib   manuscript source
│   ├── *.py                 analysis and table/figure scripts (see section 5)
│   ├── tables/  figures/    the generated tables (.tex) and figures (.pdf, .png) used by main.tex
│   └── README.md
└── docs/                    design records, change logs, verification notes (see section 10)
```

## 2. What is not in this repository, and where to get it

| Item | Where | Needed for |
|---|---|---|
| CropGlobe data (Sentinel-2 time series and labels) | Hugging Face: `huggingface.co/datasets/x-ytong/CropGlobe` (downloaded by the notebooks) | Use 3 only |
| Authors' CropNet code | `github.com/x-ytong/CropNet`, pinned to commit `bdcf63ec1704a412da73c534c579fe08c5d2fa9b` (cloned and checked out by the notebooks, never modified) | Use 3 only |
| Result files of all 5,020 runs (`results.csv`, manifests, run-environment files, closing outputs, data-verification files) | Zenodo deposit, DOI: **[DOI to be added]** | Use 2 |

The CropGlobe/CropNet paper: arXiv 2509.03497 (v3).

Data, results, per-run prediction files, reports and PDFs are kept out of the repository by `.gitignore`. The only PDFs in the repository are the figure files in `paper/figures/`.

## 3. Requirements

**Python 3** (the reproduction in Use 2 was checked with Python 3.13). Install the packages from the repository root:

```
pip install -r requirements.txt
```

- Uses 1 and 2 need only numpy, pandas, scipy, scikit-learn and matplotlib. `torch`, `huggingface_hub` and `geopandas` are needed only for training on Colab or Kaggle. You can skip them on your own computer.
- To compile the paper you also need a LaTeX distribution with `pdflatex` and `bibtex` (for example TeX Live). These packages are used: geometry, microtype, amsmath, amssymb, booktabs, multirow, array, graphicx, caption, xcolor, natbib, setspace, url, placeins, lineno, hyperref.
- A GPU is needed only for Use 3. The experiments ran on Google Colab with a Tesla T4. The exact software versions of every session are recorded in the `run_environment_seed*.json` files of the Zenodo deposit.

## 4. Use 1: check the code (no data, no GPU, about 5 minutes)

From the repository root:

```
python scripts/run_all_tests.py
```

Expected output:

```
Level 1 v3.8: 72 tests, 72 passed
Level 2 v3: 77 tests, 77 passed
level2/tests/test_usa_order.py -> USA-ORDER TESTS: 9 pass, 0 fail
level1/tests/t_probe.py -> TOTAL 3 FAIL 0
```

Optional synthetic end-to-end runs (each takes about one minute):

```
python level2/tests/dryrun_level2.py run          # ends with: DRY RUN 2 OK
python level2/tests/dryrun_followup.py run        # ends with: FOLLOW-UP DRY RUN OK
python level1/tests/integ38.py                    # items F, G, H print False on purpose (they are the negative tests)
python level1/tools/check_notebook_matches_cells.py   # the notebooks equal the files in level1/src/cells
python level2/tools/build_notebooks.py            # rebuilds the Level 2 notebooks from level2/src/cells
```

The dry runs replace the neural network with a logistic-regression stand-in and use synthetic features. They test the **plumbing, not the science**. The PyTorch training, the GPU path and the loading of the real data run only on Colab or Kaggle.

## 5. Use 2: reproduce every table and figure of the paper (no GPU, about 2 minutes)

This rebuilds the paper's tables and figures from the deposited result files. No training is involved.

**Step 1.** Download the Zenodo deposit and unzip it. The folder must contain `1_results/` and `SHA256SUMS.txt`.

**Step 2.** Point the scripts to it and run them in this order from inside `paper/`:

```
cd paper
export CROPGLOBE_DATA=/path/to/deposit          # Windows PowerShell: $env:CROPGLOBE_DATA="C:\path\to\deposit"

python make_assets.py                            # claim tables, mix-versus-own, fixed pool, C3, follow-up -> assets.pkl
python sensitivity_checks.py "$CROPGLOBE_DATA"   # checks every deposit file against SHA256SUMS.txt, then the sensitivity analyses -> sensitivity_outputs.json
python rescoring_checks.py "$CROPGLOBE_DATA"     # numbers quoted in the text: near-baseline counts, USA claims under the earlier scoring rule -> rescoring_outputs.json
python revision_checks.py                        # class-share covariates, multi-class metric, seed-inclusive C2, Holm correction, selection arms -> revision_outputs.json
python gen_revision_tables.py                    # tables/tab_metric.tex, tab_c2boot.tex, tab_arms.tex
python make_graphical_abstract.py                # figures/graphical_abstract.png and .pdf
python make_figs.py                              # figures/fig1 ... fig5 (.pdf and .png)
python gen_tables.py                             # the other 11 tables in tables/
```

`sensitivity_checks.py` stops if any deposit file does not match its SHA-256 checksum. The scripts find the deposit through `CROPGLOBE_DATA` (or the path given as argument). `CROPGLOBE_REPO` is optional; it defaults to the parent folder of `paper/`.

**Step 3.** Compile the paper:

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**What you should get.** The regenerated files in `tables/` are byte-identical to the committed ones, so `git status` shows no table as changed. The figures are pixel-identical to the committed ones, but their files can differ in internal bytes (for example the creation date inside the PDFs), so git may list them as modified even though the images are the same.

## 6. Use 3: re-run the experiments (GPU, Google Colab)

Each level has its own README with the step-by-step procedure. In short:

| Experiment | Notebook | Instructions |
|---|---|---|
| Level 1 (US states as source) | `level1/notebooks/CropGlobe_Level1_v3_8_COLAB.ipynb` (Kaggle version alongside) | `level1/README.md` |
| Level 2 (French regions as source) | `level2/notebooks/CropGlobe_Level2_FRA_source_COLAB_v3.ipynb` | `level2/README.md` |
| Level 2 follow-up (without regions 75 and 76) | `level2/notebooks/CropGlobe_Level2_FOLLOWUP_no75_76_COLAB.ipynb` | `level2/README.md` |

General procedure:

1. Copy the project folder to Google Drive and set your Drive path in cell 1 (`c01_env.py`).
2. Run the cells in order. The notebook downloads the data, checks out the pinned CropNet commit, and runs the unit and data tests. Training is blocked if any test fails or if the data do not match the benchmark's published counts.
3. Level 2 and the follow-up print a **design hash** (all settings, every planned run and the scoring rule). Training starts only after this hash is pasted into the approval cell. Any later change to the design changes the hash, and the run cell refuses to start.
4. Run the run cell. It is resumable: run it again after a disconnect.
5. After all seeds, run the summary and closing cells. They freeze `results.csv` by SHA-256 and print the pre-specified claims first and the exploratory tables after.

Level 1 is frozen: the final `results.csv` has SHA-256 `38d1876cde08e54cfc827aa6b2813355ef98a09dbd048885a67d58991534f891` (`docs/LEVEL1_FREEZE.md`). Re-running the notebooks produces new runs in your own output folder and never modifies the deposited results.

## 7. Where each table and figure of the paper comes from

| Paper item | File in `paper/` | Made by |
|---|---|---|
| Table 1 (augmentation settings) | written directly in `main.tex` | - |
| Table 2 (Level 1 claims) | `tables/tab_l1.tex` | `gen_tables.py` |
| Table 3 (Level 2 claims) | `tables/tab_l2.tex` | `gen_tables.py` |
| Table 4 (mixes versus their own units) | `tables/tab_mix.tex` | `gen_tables.py` |
| Table 5 (follow-up claims) | `tables/tab_fu.tex` | `gen_tables.py` |
| Table 6 (C1 sensitivity analyses) | `tables/tab_sens.tex` | `gen_tables.py` (from `sensitivity_checks.py`) |
| Table 7 (multi-class metric) | `tables/tab_metric.tex` | `gen_revision_tables.py` (from `revision_checks.py`) |
| Table 8 (C2 with seed variation) | `tables/tab_c2boot.tex` | `gen_revision_tables.py` (from `revision_checks.py`) |
| Table 9 (gain per doubling under plain training, copy control and augmentation) | `tables/tab_train.tex` | `gen_tables.py` |
| Table 10 (how much the outcome depends on which units are drawn) | `tables/tab_spread.tex` | `gen_tables.py` |
| Table 11 (Level 1 selection arms) | `tables/tab_arms.tex` | `gen_revision_tables.py` (from `revision_checks.py`) |
| Table 12 (augmentation, C3) | `tables/tab_c3.tex` | `gen_tables.py` |
| Table A.1 (timeline) | `tables/tab_timeline.tex` | `gen_tables.py` |
| Table A.2 (gain per doubling by budget) | `tables/tab_budgetk.tex` | `gen_tables.py` |
| Table A.3 (USA scoring rules) | `tables/tab_scoring.tex` | `gen_tables.py` |
| Figure 1 (gain per doubling) | `figures/fig1_per_doubling.pdf` | `make_figs.py` |
| Figure 2 (F1 against k) | `figures/fig2_f1_vs_k.pdf` | `make_figs.py` |
| Figure 3 (mixes versus own units) | `figures/fig4_mix_vs_own.pdf` | `make_figs.py` |
| Figure 4 (single units) | `figures/fig3_single_units.pdf` | `make_figs.py` |
| Figure 5 (fixed-pool budget effect) | `figures/fig5_fixed_pool_budget.pdf` | `make_figs.py` |
| Graphical abstract | `figures/graphical_abstract.png` / `.pdf` | `make_graphical_abstract.py` |

Note that Figure 3 is the file `fig4_mix_vs_own.pdf` and Figure 4 is `fig3_single_units.pdf`. The files keep their original names, and the order in the paper follows the text.

## 8. Study design in brief

- **Budget**: the total number of labelled samples per run (training plus a 10% stratified validation split): 1,000, 2,000, 3,000 or 4,000.
- **k**: the number of source units mixed, 1, 2, 4 or 8 where the pool allows it.
  - A unit enters the pool only if it has at least 1,500 rows and can supply the whole budget alone.
  - k = 1 is a census of every pool unit.
  - k ≥ 2 uses up to 10 random combinations, with equal allocation per unit.
- **Key-crop share**: fixed per target (Argentina: soybeans 20%; USA and France: corn, soybeans and wheat 15% each; China: rice 20%). The rest is sampled naturally.
- **Model and training**: identical in all stages.
  - The authors' 7-class CropNet with their optimiser settings (Adam, learning rate 1e-4, batch 256, balanced class weights, day-of-year window 121 to 334).
  - Up to 200 epochs with early stopping on validation macro-F1.
- **Scoring**:
  - Argentina and China: binary key-crop macro-F1 (soybeans or rice versus the rest).
  - France (Level 1, descriptive): macro-F1 over the classes with at least 30 training samples.
  - USA (Level 2): four-class macro-F1 over corn, soybeans, wheat and other, with true rice, sugarcane and cotton samples removed.
- **Seeds and verdicts**:
  - Seeds 0-1 are exploratory and seeds 2-4 confirmatory; the follow-up uses new seeds 5-9.
  - A claim passes only if its sign agrees in both seed groups **and** the confirmatory 95% interval excludes 0.
  - The verdict words (PASS, NOT REPLICATED, NOT TESTABLE) are printed by the code, never edited by hand.
- **Claims**:
  - C1: gain per doubling of k, from a mixed model with a random intercept per combination.
  - C2: gain from the fewest to the most units at each budget.
  - C3: augmentation versus a steps-matched copy control.

## 9. Integrity safeguards

- **Test gate**: unit and data tests must pass before any training, and the data must match the benchmark's published counts.
- **Pinned code**: the authors' repository must be clean at the pinned commit.
- **Design hash** (Level 2 and follow-up): settings, plan and scoring rule are hashed, and training starts only with the approved hash.
- **Separate outputs**: results folders carry a manifest and are never mixed. Resumed runs are checked for drift by plan checks and canary re-runs.
- **Recorded changes**: every change after approval is recorded as a dated amendment in `manifest.json`, and the deviations are listed in the paper.
- **Freezing**: final result files are frozen by SHA-256 (`docs/LEVEL1_FREEZE.md`, and the FREEZE files in the Zenodo deposit). The reproduction scripts verify every deposit file before using it.

## 10. Documentation index

| File | Content |
|---|---|
| `docs/LEVEL1_FREEZE.md` | SHA-256 freeze record of the Level 1 results |
| `docs/LEVEL2_PREREGISTRATION_DRAFT.md` | Level 2 pre-registration draft (the design before any French data were seen) |
| `docs/FRA_join_verification.md` | Row-by-row verification of the France region assignment (raw files against the packaged data) |
| `docs/CHANGES_AND_CONFLICTS.md`, `docs/CHANGELOG_v3_1.md` ... `docs/CHANGELOG_v3_8.md` | Change log of every Level 1 notebook version |
| `docs/AUDIT_REPORT_v3_3.md`, `docs/VERIFICATION_v3_7_vs_review.md`, `docs/VERIFICATION_v3_8.md`, `docs/VERIFICATION_second_review.md` | Audits and verification notes for the Level 1 versions and review rounds |
| `level2/docs/CHANGES_v3_AND_REVIEW_CHECKLIST.md` | Level 2 notebook changes and how each review point is covered |
| `level1/README.md`, `level2/README.md`, `paper/README.md` | Detailed instructions for each part |

## 11. Citation, license and contact

- **Citation**: the manuscript is under review. A citation will be added here on publication. Please also cite the CropGlobe/CropNet paper (arXiv 2509.03497) and the Zenodo deposit of the result files.
- **License**: MIT (see `LICENSE`). The CropGlobe data and the authors' CropNet code are not part of this repository and keep their own licenses.
- **Questions and problems**: please open an issue on this repository.

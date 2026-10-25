# v3.7 notebooks verified against the supervisor review

Method: I read the delivered .ipynb files (not my source files): every code cell equals the tested source (checked by hash, both notebooks); the same code passes the 66-test gate; and I ran the v3.7 statistics code on your REAL seed-0 file (409 runs). Not possible here: torch / GPU / real training / downloads.

## Review item by item
| # | Review item | Status | Evidence / caveat |
|---|---|---|---|
| 0 | ARG has 6 classes, CHN 4; soybean-vs-other is a stated choice | OK | `DEFINED_CLASSES` ARG = corn, soybeans, rice, sugarcane, cotton, other; CHN = corn, soybeans, rice, other; `PRIMARY_VIEW` ARG binary soybeans, CHN binary rice, printed in the plan cell. The paper statement is yours to write |
| 1 | Seeds 1-4 | OK (untested on GPU) | AUTO_MODE runs seeds 0-4; auto-approval only on a clean seed (dry run: pause / stop / resume all behaved) |
| 2 | Epoch cap 200 | OK | `MAX_EPOCHS=200`, new results folder, tests updated. Whether 200 is enough is only visible in the gate report (stop reasons) after seed 0 |
| 3 | Fixed common state pool for budget comparison | OK, but see finding B | arm `common` + table `common_pool_budget_by_k.csv`; ARG pool 6 so k<=4 only |
| 4 | France rice | OK | `FRA_MAIN_RULE='coverage_conditional'`; real file: FRA main 0.541 -> 0.676. Caveat: the rule drops ANY class with <30 training samples, not only rice (same definition as before) |
| 5 | Augmentation ON, ARG B1000 / B2000 | Implemented, NOT executed | authors' `augm` (seeded). Caveats: 3x steps per epoch, no step-matched control, torch path never run here (your smoke test checks it) |
| 6 | Two metrics only | OK with a deviation | main + secondary in tables/plots; other columns are still stored and the per-seed gate report still lists all tracks (audit) |
| 7 | Record input features | OK | `input_features` column in every row |
| 8 | Main table with CIs across combinations | OK | ran `cell_stats` / gain / regression on the real file without error |
| 9 | Sample-level selection (k-center) | Implemented, my interpretation | the "original Phase 4" text was not available to me; design = per (state, crop) farthest-first in a 16-dim PCA space, same counts as random. Confirm with the supervisor |
| 10 | Report selection-strategy result as a finding | OK | diverse / clustered arms kept (real file: diverse - random = 0.0 pp) |
| 11 | Level 2 | Out of scope | not started |

## Findings from running the v3.7 code on your seed-0 file
**A. The review's ARG table pools all arms.** Its B1000 numbers (0.616 / 0.662 / 0.703 / 0.718) reproduce exactly as the mean over ALL runs (random + diverse + clustered). The random arm alone (the designed baseline) gives 0.616 / 0.693 / 0.694 / 0.720: k=2 -> 4 is flat. So the "+2.2 points per doubling" statement depends on mixing the selected arms; ask which one the supervisor meant. The notebook's tables use the random arm and report the other arms separately.

**B. The pool-shrinkage explanation is only partly supported.** On the SAME six states (the B4000 pool: KS, MN, ND, NE, SD, WI), k=1 seed-0 scores are still 0.653 (B1000), 0.658 (B2000), 0.709 (B3000), 0.699 (B4000). The 2000 -> 3000 jump (+5 points) remains with identical states, so it is not only a change of states. Single seed, 6 runs per cell: treat as a flag, the new `common` arm will test it properly.

**C. Reported numbers I could not reproduce:** "~2.2 points per doubling" (see A), seed noise 1.7 points and "capped runs score higher" (one seed only).

**D. Minor:** the regression table prints the budget coefficient as 0.0000 because it is per label; read it as per 1000 labels x 1000. Cosmetic, left unchanged.

## Not verified
Any real training (CropNet, augmentation, k-center on real features), run time, whether the 200-epoch cap is sufficient, the Kaggle / Colab download paths, USA state alignment (needs `alignment_report.json`).

# Level 2 notebook v3 and follow-up notebook: what changed and how each review point is covered

Files: `CropGlobe_Level2_FRA_source_COLAB_v3.ipynb` (replaces the v2 notebook), `CropGlobe_Level2_FOLLOWUP_no75_76_COLAB.ipynb` (new, separate plan).
Level 1 stays frozen (SHA-256 `38d1876cde08e54cfc827aa6b2813355ef98a09dbd048885a67d58991534f891`). Neither notebook touches it.

## 0. Things you must do / know first
1. **New design hash.** The USA scoring rule is part of the hash and changed (L2-D11). The old hash no longer matches. Run cells 1-11, copy the hash printed by the plan cell, send it to the instructor, and paste it into cell 12 only after he approves it. The run cell refuses anything else.
2. **Cell 12b** re-scores every finished USA row (all seeds in your file) from the saved predictions, no retraining. It first checks that the saved predictions reproduce the numbers already stored; if not it stops and writes nothing. The original `results.csv` is kept byte-for-byte as `results.csv.before_rescore_<time>`; earlier scores stay in `main_f1_before_fixed4` / `main_f1_before_clean`; the change is logged in `rescore_log.json` and in the manifest as L2-D11.
3. **Cell 8b (USA order check)** needs the raw USA files `S2_USA_<crop>_<ST>.csv` somewhere under the project folder. If they are not there it prints **NOT VERIFIED** and says what to do. It does not assume anything.

## 1. Review point -> what the notebook does
| Review point | Where / how |
|---|---|
| Verify the USA soybean ordering first | Cell 8b: every raw point of every raw USA file is matched exactly to its packaged row (5-day medians), per crop with soybeans printed first; label check; block-order check (label, token A-Z, state); state in the file name vs the Level 1 state table, row by row. Pre-declared criteria; status VERIFIED / FAILED / NOT VERIFIED; saved to `tables/usa_order_verification.json` plus CSVs. Includes a built-in self-test of the verifier. Status is printed again in the closing cell. |
| Report stays honest; Argentina stays NOT REPLICATED; USA PASS fragile; do not lead with ARG budget 4000; no augmentation gain beyond copy control | Verdict words are produced by code (pre-registered rule). Closing cell prints the pre-registered claims first; the exploratory tables are printed after them and never change a verdict. The report text I wrote (not the notebook) must carry the wording about fragility; the notebook prints the intervals so the lowest CI value is visible. |
| Mix vs average of its own single regions, and vs its best single region (every mix, same budget) | `mix_vs_own_table` in cell 15, per target, ARG also split "all / without 75-76 / containing 75-76"; mean gap, 95 % bootstrap interval, share better, paired t and Wilcoxon p; labelled **EXPLORATORY (not pre-registered)**; caveat that mixes share regions so p-values are descriptive. On the interim file it reproduces the instructor's figures: ARG +1.35 (73 % better) and -3.98 (17 % beat best); USA +2.07 (89 %) and -1.31 (39 %); ARG mixes with 75/76 +1.17. His p-values correspond to the Wilcoxon column (see 4). |
| Correct the budget effect to the fixed pool (+6, not +8) | `fixed_pool_budget_effect`: single regions available at every budget only. Interim file: ARG 56.5 -> 62.5 (+6.1; +6.7 at 3000). The all-available average is printed beside it so the difference is visible. |
| (a) Remove true rice, sugarcane, cotton from the USA test set; re-score all seeds; log as a change | L2-D11 scoring + cell 12b (above). Paper Sec. 4.4 states the sugarcane / cotton exclusion for FRA -> USA; extending it to rice is the instructor's decision and is labelled as such in the code and the notebook header. |
| (b) Reassign cotton / sugarcane (and rice) predictions to "other" | Same rule: predictions of rice, sugarcane, cotton count as "other". Counts of dropped true samples and reassigned predictions are stored per run (`tables/rescore_fixed4_clean_rows.csv`). |
| (c) Belgium / Netherlands / UK: appendix only, descriptive, no claims, low priority | Cell 16: label counts only, writes `tables/appendix_label_counts.csv`, says a country may be scored only after its label set is checked. Files that are not downloaded are reported as not found. |
| Finish seed 4, then the closing analysis | Not done by me: needs your GPU. Run cell 13 until it says complete, then run the closing call in cell 15b. |
| Run the 75/76 follow-up as a NEW plan, written down and frozen before running; Argentina only; without 75 and 76; k = 1, 2, 4; budgets 1000 and 2000; new seeds 5-9; declare BOTH claims in advance | Follow-up notebook, cell 6b (`FOLLOWUP_CLAIMS` is in the locked settings, so it is in the design hash): F-C1 per-doubling (mixed model, log2(k) > 0, 95 % CI excludes 0); F-C2 mix vs mean of its own regions (mean gap > 0, bootstrap CI excludes 0, and positive in at least 4 of 5 seeds). Own results folder and own approval hash. 53 runs per seed, 265 in total (checked on a synthetic run). |
| Verify the USA soybean ordering before submission | Cell 8b, see first row. |

## 2. Decisions that are mine, not the instructor's - he should confirm them
- Follow-up rule "positive in at least 4 of 5 seeds" for F-C2, and treating all of seeds 5-9 as confirmatory. Both are inside the hash.
- Follow-up excludes 75/76 by removing them from the capacity table and row pools after loading (the data checks still use all 13 regions).
- Bootstrap: 2000 resamples of mixes, seed 0.

## 3. What was tested here, and what was NOT
- 77 unit/data tests pass (including new tests for the clean scoring rule, re-scoring all seeds from saved predictions, hash dependence on the scoring rule); 9 tests for the ordering verifier; new self-tests for the exploratory analysis and the follow-up claims.
- Synthetic end-to-end dry runs (stand-in classifier, real region/crop counts): main notebook (gate refusal without approval and with a wrong hash, run, re-score from an old-rule file reproduces run-time scores to 0.0, closing) and follow-up notebook (265 runs, no run contains 75 or 76, claims evaluated). These test the plumbing only; their numbers mean nothing.
- **Never executed here:** the PyTorch training code, the GPU, loading of the real data, and the ordering check on the real raw USA files (no access from this sandbox). The statsmodels cross-check could not run here (no package access); it runs on Colab.
- The seed-4 / closing results are not available, so no final verdict exists yet. Interim figures are from 1,327 runs.

## 4. The instructor's p-values
His p = 2e-7 (ARG) and 3e-11 (USA) match the **Wilcoxon** p-values the notebook prints (2.2e-7 and 2.5e-11). The paired t-test p-values are smaller (3e-8 and 5e-16). Effect sizes and shares match his numbers exactly. Because mixes share regions, none of these p-values is a clean significance test; the notebook labels them descriptive.

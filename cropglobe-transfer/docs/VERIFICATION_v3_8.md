# v3.8 verification — what was executed, and what was not

## Executed (CPU, in the build environment)
- **Part A + A2 unit tests: 72 PASS, 0 FAIL**, run from the code cells of **both** built notebooks (Colab, Kaggle). Cell text is byte-identical to the tested `.py` files. (`test_report_v3_8_part_A.txt`)
- **Full-flow dry run** with synthetic data and a logistic-regression stand-in for CropNet (`integration_full_flow_v3_8.txt`): pause on time limit → stop after one failing run → resume → complete; aug_on and aug_copy rows present; summary cell runs including the new statistics. The synthetic scores are all 100%, so this proves the flow, not any result.
- **Seed-guard dry run, 3 seeds** (`integration_seed_guards_v3_8.txt`): seeds 0–1 have 116 runs, seed 2 has 101 (k-center skipped); manifest records all 5 amendments; three `run_environment_seed*.json` files kept; canary reproduces stored scores exactly (diff 0.0) with unchanged code; **changing the fit function → canary FAILS and `run_guards()` returns False**; **changing the plan → plan check FAILS** and names the orphan run ids.
- **Real-data check of resume safety** (`realplan_check_v3_8.txt`): plan rebuilt from your 48-state table; **all 887 checkable runs** of your 1,475-run file (random, common, k-center, aug_on) are in the v3.8 plan.
- **Statistics on your real 1,475-run file** (`real_data_stats_v3_8.txt`). ARG random arm: +2.15 points per doubling of states (mixed model, 104 combinations, p<0.001); CHN +4.4; FRA +0.9 (not significant). The replication table runs; with only 139 seed-2 runs it correctly says "confirmatory seeds not finished".

## NOT executed
- **Any GPU/torch code** (real CropNet training, the `copy` branch of `fit_model` on real tensors). Only syntax and the CPU stand-in were run. Part B and the smoke test still run on your Colab/Kaggle before training.
- **Diverse / clustered run ids** (588 of the 1,475 rows) — they depend on the real features, so only your Colab can confirm them; the plan check there will do so.
- France-as-source, GroupDRO, point-in-polygon alignment: not built.
- statsmodels was not available, so the mixed model is my own REML code; it was checked against a dense GLS (V38-03) but not against statsmodels itself.

## Open points
- The k-center design is my reading of the original Phase 4 and should be confirmed with the instructor.
- Wheat's state assignment is disputed between candidate orderings (affects France only).
- C2 labels in the replication table depend on the (k_min, k_max) found in each group; with complete seeds they will match.

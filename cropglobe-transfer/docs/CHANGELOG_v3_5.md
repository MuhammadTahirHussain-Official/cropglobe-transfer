# CHANGELOG v3.5 — Phase 4 (diversity-aware state selection) added

Base: v3.4 (untouched copies of v3.2 and v3.4 remain in the outputs folder). Nothing in the baseline method was changed: with `SELECTION_ARMS = ['random']` the plan is identical to v3.4 (verified below).

## Approved definition implemented (your decision table)
| Item | Implementation | Config name |
|---|---|---|
| Diversity space | Per (state, crop): mean of globally standardised Sentinel-2 features of the USA source data (10 bands × all 73 time steps). State distance = RMS difference of two mean profiles. | `DIVERSITY_SPACE`, `MIN_PROFILE_N=20` |
| Score of a state set | Mean pairwise state distance, averaged over the **fixed (key) crops of the target design** (ARG soybeans; CHN rice; FRA corn+soybeans+wheat). | `diversity_matrix()` |
| Arms | `random` (baseline, unchanged), `diverse` (N highest scores), `clustered` (N lowest scores). k=1 is one shared census for all arms. | `SELECTION_ARMS`, `N_ARM=10` |
| Number of combinations | Score **every** combination when C(P,k) ≤ 50,000; otherwise score a fixed seeded sample of 20,000 distinct combinations. The `enumerated` flag is saved in the feasibility grid. | `ENUM_LIMIT`, `SCORE_SAMPLE` |
| Fair comparison | Same pool, k, budget, key-crop counts, training settings, seeds for all arms; a state without a profile for a fixed crop leaves the pool of **all** arms. Diverse/clustered need C(P,k) ≥ 2·N_ARM so they are disjoint. | `build_plan()` |
| Data restriction | The distance function takes only USA features, state and crop labels. No target data, labels or scores are used. | `compute_diversity_distances()` |
| Statistics | Per-arm cell statistics; diverse/clustered intervals use no finite-population correction (the combinations are *selected*, not sampled) and are labelled so; Welch contrasts `diverse − random`, `clustered − random`; Spearman(F1, diversity percentile) per cell (EXPLORATORY). | `arm_contrast`, `diversity_assoc` |
| Legacy tables | The existing summary tables use the **random arm only**, so they stay comparable with v3.4. | c21 |

## Change log (every change)
| ID | Cell | Original → new | Reason | Kind | Scientific impact |
|---|---|---|---|---|---|
| V40-01 | config | added SELECTION_ARMS, POOL_RULE, N_ARM, ENUM_LIMIT, SCORE_SAMPLE, MIN_PROFILE_N, DIVERSITY_SPACE (all in LOCKED); NOTEBOOK_VERSION v3.5 | Phase 4 definition | feature | none for random arm |
| V40-02 | core | `compute_diversity_distances`, `diversity_matrix`, `pool_for`, `select_extreme_arms`, `combo_score_pct` | Phase 4 | feature | new arms only |
| V40-03 | core | `build_plan` rewritten to build arms over one matched pool; random arm output identical to v3.4 | Phase 4 | feature | none for random arm (tested) |
| V40-04 | core | results columns `arms`, `diversity_score`, `diversity_pct` | traceability | feature | none |
| V40-05 | core | `t_interval_fpc`: N = ∞ → no correction | selected arms | feature | only selected arms |
| V40-06 | core | `cell_stats`: selected arms flagged and not "exhaustive" | honesty of intervals | feature | labels only |
| V40-07 | core | `combo_level` tolerates results without diversity columns | old results files | robustness | none |
| V40-08 | plan cell | computes distances from the USA features, saves `diversity_distance_<crop>.csv`; text no longer says Phase 4 is unimplemented | Phase 4 | feature | none |
| V40-09 | summary | Phase 4 section (per-arm tables, contrasts, association); legacy tables restricted to random-arm rows | Phase 4 | feature | none for random arm |
| V40-10 | tests | P4-01…P4-08 added to Part A2; pipeline self-test now runs all three arms | verification | tests | – |

## Verification — what was and was not executed
**Executed by me (synthetic data, no GPU):** Part A and Part A2 (all pass, including P4-01…P4-08); end-to-end dry run with a stand-in model through plan → run → gate → summary with all three arms; plan comparison on your real USA state × crop table.
**Not executed by me:** real Sentinel-2 data, the real diversity distances, torch/CropNet, GPU training, Colab. Diversity values on real data are therefore unknown until you run it.

### Plan on your real state table (`MIN_STATE_ROWS=1500`, distances replaced by random stand-ins — counts do not depend on distance values)
- `SELECTION_ARMS=['random']` gives **exactly the v3.4 plan** (250 combination-cells; identical sets).
- Runs (5 seeds), N_ARM=10: strict pool **2,075** (random-only 1,250); equal_share pool **2,795** (random-only 1,380).
- With N_ARM=10, **CHN has no diverse/clustered arm at all** (pool of 6 states gives C(6,2)=15 < 20), FRA only B=1000 (strict). AUS has only census cells. Phase 4 is testable on ARG (all budgets) and FRA.
- With N_ARM=5: CHN and FRA get arms (16–17 per arm-cell); runs fall to 1,740 (strict) / 2,135 (equal_share). **Not applied** — your decision.

## Open decisions (unchanged defaults, nothing silently chosen)
1. `N_ARM` 10 (matches the random arm size, leaves CHN without arms) or 5 (CHN covered, fewer runs).
2. `POOL_RULE` strict (default) or equal_share (needed for ARG 4000 at k=8 and FRA 2000 coverage; not approved).
3. Diversity uses all 73 time steps, not only the DOY 121–334 window the model sees (the window is applied by the authors' class, which is only available later). Say if you want the window slice.
4. AUS key share, fixed_full proposal A/B, geography tolerance, alignment override if the USA alignment stop persists.
5. Geographic-distance control (secondary analysis) is **not implemented** — it needs state coordinates that are not in the data supplied.
6. Namespace: the locked manifest changed, so an existing v3.4 output folder that already has a manifest will STOP the guard. Use a new folder (or move the old one aside). Nothing is deleted.

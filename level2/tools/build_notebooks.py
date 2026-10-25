import json, re, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(ROOT, 'src', 'cells'))
OUT = os.path.join(ROOT, 'notebooks') + os.sep
MD = lambda t: dict(cell_type='markdown', metadata={}, source=t.strip('\n').splitlines(True))
def CODE(f, sub=None):
    src = open(f).read().rstrip('\n')
    for a, b in (sub or []): assert a in src, (f, a); src = src.replace(a, b)
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=src.splitlines(True))
def nb(cells): return dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python'), accelerator='GPU', colab=dict(provenance=[])), nbformat=4, nbformat_minor=5)

HEAD1 = """# Phase 3 - Level 2: France as the source, ARG / USA / CHN as targets  (notebook v3)

**Status: DESIGN NOT YET APPROVED.** Nothing trains until the instructor has signed the decisions *and* the approved design hash is pasted into the approval cell (cell 12). The scoring rule for the USA changed in this version (L2-D11), so the design hash is **new**: the old hash no longer matches and the new one must be approved. If results of earlier seeds exist, cell 12b re-scores them from the saved predictions (no retraining) before the run cell will continue.

| | |
|---|---|
| Source | France - 13 regions are the "states" of Level 1 (region of every row from the verified row map; 18 rows that cannot be placed with certainty are excluded and counted) |
| Targets | Argentina (soybeans 20 %) and USA (corn / soybeans / wheat 15 % each) carry the claims; China is descriptive only (rice exists in two regions). Belgium / Netherlands / UK: appendix label check only (cell 16), no claims |
| Training setup | identical to Level 1 (lr 1e-4, batch 256, <= 200 epochs, patience 15, 7-class head, budgets 1000-4000, k in {1,2,4,8}, seeds 0-4) |
| Seeds | 0-1 exploratory, 2-4 confirmatory. A claim passes only if the sign agrees in both groups **and** the confirmatory 95 % interval excludes 0. Otherwise the table says NOT REPLICATED / NOT TESTABLE |
| USA scoring (L2-D11) | macro-F1 over corn, soybeans, wheat, other; TRUE rice / sugarcane / cotton removed from the USA test set (paper Sec. 4.4 does this for sugarcane and cotton in FRA->USA; extending it to rice is the instructor's decision); predictions of rice / sugarcane / cotton count as "other". Every seed is re-scored from saved predictions and logged |
| USA ordering | cell 8b verifies the USA soybean (and corn / other) row order against the raw files, exactly, and prints the status; it is reported again in the closing cell |
| Exploratory (not claims) | closing cell: every region mix vs the mean of its own single regions and vs its best single region; fixed-pool budget effect; per-doubling slope with / without regions 75 and 76 |
| Level 1 | frozen (3,160 runs, SHA-256 `38d1876c...34f891`); this notebook never reads or writes that folder |

**What this notebook cannot do:** it cannot make the results support any claim. It implements the pre-registered claims and tests them honestly. Argentina verdicts stay as the code prints them. Run order: cells 1-11 (setup, tests, data, USA-order check, plan) -> instructor signs -> cell 12 (approval) -> cell 12b (re-score, once) -> cell 13 (run) -> cells 14-15 (summary, closing analysis) -> cell 16 (appendix label check).
"""
S1 = [("1. Environment (Colab, GPU)", 'c01_env.py'), ("2. Configuration (all locked settings, design decisions L2-D1..D12)", 'c02_config.py'), ("3. Core code", 'c04_core.py'),
 ("4. Test gate, part A (unit tests; training is blocked if any fail)", 'c07_tests.py'), ("5. France region assignment + data loading code", 'c09_data.py'), ("6. Data / Level 2 tests", 'c10_tests_data.py'),
 ("7. Acquire data + authors' code (pinned commit)", 'c11_acquire.py'), ("8. Test gate, part B (our remap vs the authors' released function)", 'c12_testsB.py'),
 ("8b. USA row-order verification (soybeans first; exact match of every raw point to its packaged row; needs the raw USA files, otherwise prints NOT VERIFIED)", 'c11b_usa_order.py'),
 ("9. Load + verify the data (STOPs on any inconsistency)", 'c13_load.py'), ("10. Plan table (nothing trains) + design hash", 'c15_plan.py'), ("11. Model + training code + smoke test", 'c17_torch.py'), ("12. INSTRUCTOR APPROVAL", 'c18_approval.py'),
 ("12b. Re-score ALL finished USA rows with the clean fixed-class rule (L2-D11; once, from saved predictions, no retraining)", 'c18b_rescore.py'),
 ("13. RUN (resumable: run again after a disconnect)", 'c19_run.py'), ("14. Summary tables and figures (safe at any time)", 'c21_summary.py'),
 ("15. Closing analysis: freeze, pre-registered claims, exploratory additions, cross-checks (after ALL seeds)", 'c24_level2_analysis.py'),
 ("16. Appendix: Belgium / Netherlands / UK label check (descriptive only, no claims)", 'c25_appendix_labels.py')]
cells = [MD(HEAD1)]
for t, f in S1: cells += [MD('## ' + t), CODE(f)]
cells.append(MD("## 15b. Run the closing analysis\nAfter every seed is finished: `closing_report_l2(os.path.join(OUT_DIR, 'results.csv'), os.path.join(OUT_DIR, 'closing'))`. Pre-registered claims come first and are never changed by the exploratory tables printed after them."))
cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=["# run ONLY after all five seeds are finished (it refuses an incomplete results file)\n", "# CLOSING = closing_report_l2(os.path.join(OUT_DIR, 'results.csv'), os.path.join(OUT_DIR, 'closing'))\n"]))
json.dump(nb(cells), open(OUT + 'CropGlobe_Level2_FRA_source_COLAB_v3.ipynb', 'w'), indent=1, ensure_ascii=False); print('main', len(cells))

HEAD2 = """# Phase 3 - Level 2 FOLLOW-UP: Argentina without French regions 75 and 76  (separate pre-registered plan)

**Status: DESIGN NOT YET APPROVED.** This is a NEW plan, written down and frozen before any run. It does not change any Level 2 result or verdict.

| | |
|---|---|
| Question | In Level 2, French regions 75 and 76 are far better than the others as single regions for Argentina, and mixes containing them lose accuracy when regions are added. Do more regions help when those two are absent? |
| Scope | Argentina only; source France with regions **75 and 76 removed** from the pool; k = 1, 2, 4; budgets 1000 and 2000; **new seeds 5-9**, all confirmatory; plain training (no augmentation, no other arm); same model, training, scoring and data as Level 2 |
| Claim F-C1 (declared in advance) | mixed model F1 ~ log2(k) + budget/1000 + seed + (1 \\| region combination): the log2(k) coefficient is > 0 and its 95 % interval excludes 0 |
| Claim F-C2 (declared in advance) | mean over region mixes of [F1 of the mix - mean F1 of its own single regions at the same budget] is > 0, its 95 % bootstrap interval excludes 0, and it is > 0 in at least 4 of the 5 seeds (the "4 of 5" rule is the analyst's proposal; it is part of the design hash, so the instructor must confirm it before approving) |
| Not claims | mix vs the BEST of its own regions; any single (k, budget) cell; any other subset |
| Results folder | `outputs/phase3_level2_followup_no75_76_v1` (never mixed with Level 2) |

Run order: cells 1-6 (setup, tests) -> cell 6b (follow-up plan) -> cells 7-11 -> instructor signs -> cell 12 (paste the hash) -> cell 13 (run) -> 14-15 (summary, closing).
"""
S2 = [("1. Environment (Colab, GPU)", 'c01_env.py', [("NAMESPACE  = 'phase3_level2_fra_source_v1'", "NAMESPACE  = 'phase3_level2_followup_no75_76_v1'")]),
 ("2. Configuration (inherited from Level 2; the follow-up overlay is cell 6b)", 'c02_config.py', None), ("3. Core code", 'c04_core.py', None),
 ("4. Test gate, part A (unit tests; training is blocked if any fail)", 'c07_tests.py', None), ("5. France region assignment + data loading code", 'c09_data.py', None), ("6. Data / Level 2 tests", 'c10_tests_data.py', None),
 ("6b. FOLLOW-UP PLAN: scope, claims F-C1 / F-C2, excluded regions (declared before any run; part of the design hash)", 'c02f_followup.py', None),
 ("7. Acquire data + authors' code (pinned commit)", 'c11_acquire.py', None), ("8. Test gate, part B", 'c12_testsB.py', None), ("9. Load + verify the data (STOPs on any inconsistency)", 'c13_load.py', None),
 ("10. Plan table (nothing trains) + design hash", 'c15_plan.py', None), ("11. Model + training code + smoke test", 'c17_torch.py', None), ("12. INSTRUCTOR APPROVAL", 'c18_approval.py', None),
 ("13. RUN (resumable: run again after a disconnect)", 'c19_run.py', None), ("14. Summary tables and figures (safe at any time)", 'c21_summary.py', None),
 ("15a. Analysis functions (shared with Level 2)", 'c24_level2_analysis.py', None), ("15b. Follow-up closing analysis: freeze + claims F-C1 / F-C2 (after ALL seeds 5-9)", 'c24f_followup_analysis.py', None)]
cells = [MD(HEAD2)]
for t, f, sub in S2: cells += [MD('## ' + t), CODE(f, sub)]
cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=["# run ONLY after seeds 5-9 are finished\n", "# FOLLOWUP_RESULT = closing_report_followup(os.path.join(OUT_DIR, 'results.csv'), os.path.join(OUT_DIR, 'closing'))\n"]))
json.dump(nb(cells), open(OUT + 'CropGlobe_Level2_FOLLOWUP_no75_76_COLAB.ipynb', 'w'), indent=1, ensure_ascii=False); print('followup', len(cells))

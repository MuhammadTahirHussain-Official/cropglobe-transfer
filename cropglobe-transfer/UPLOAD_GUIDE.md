# How to upload the repository to GitHub

Repository: https://github.com/MuhammadTahirHussain-Official/cropglobe-transfer

The folder `cropglobe-transfer/` holds the **complete** repository: 138 files. It replaces everything that is on GitHub now.
GitHub's web page accepts up to 100 files per upload, so upload in the three batches below. Each batch is one commit.

## Web browser (no software needed)

For each batch:

1. Open the repository page and click **Add file -> Upload files**.
2. Open the folder `cropglobe-transfer` on your computer and drag the items listed for that batch onto the page.
   Drag the folders themselves (for example the folder `paper`), not their contents: GitHub keeps the sub-folders.
3. Enter the commit message given below, select **Commit directly to the main branch**, and click **Commit changes**.

Hidden files: `.gitignore`, `paper/.gitignore` and `level2/.gitignore` start with a dot. On a Mac, press Cmd+Shift+. in Finder to show them before dragging.
If one is missing on GitHub after the upload, click **Add file -> Create new file**, type its path (for example `.gitignore`) and paste its contents.

| Batch | Drag these items | Files | Commit message |
|---|---|---|---|
| 1 | `README.md`, `LICENSE`, `requirements.txt`, `.gitignore`, folders `paper`, `docs`, `scripts` | 61 | Update README, add LICENSE, requirements and .gitignore; add paper/ |
| 2 | folder `level1` | 44 | Sync level1/ with the final v3.8 version |
| 3 | folder `level2` | 33 | Sync level2/ with the final version (main experiment and follow-up) |

Files that are already on GitHub and unchanged are simply kept. `LICENSE` has the same MIT text as the one on GitHub.

## After the upload

1. On GitHub, open the repository page: the new README should display, with the contents list at the top.
2. Check that `paper/figures/` has 12 files and `paper/tables/` has 14 files.
3. If GitHub shows any file that is **not** in the list below, it is from an older version. Delete it (open the file, click the three-dot menu, **Delete file**) only if you are sure it is not needed.
4. When the Zenodo deposit is published, replace **[DOI to be added]** in `README.md` (section 2) and add the DOI to the Data availability section of `paper/main.tex`.

## Do NOT upload

- Result files (`results*.csv`, `runs/`, `*.npz`, `manifest.json`): they belong in the Zenodo deposit.
- The submission package (`manuscript.pdf`, `main.bbl`, `REVISION_NOTES.md`, the submission zip) and the audit PDFs.

## Command line alternative (git)

```
git clone https://github.com/MuhammadTahirHussain-Official/cropglobe-transfer.git
# copy the CONTENTS of the provided cropglobe-transfer/ folder over the clone (including the hidden .gitignore files), then:
cd cropglobe-transfer
git add -A
git status          # review: new, changed and deleted files
git commit -m "Update repository: paper/, README guide, LICENSE, requirements, .gitignore"
git push
```

## Complete file list (138 files)


### Batch 1: top-level files, paper/, docs/, scripts/ (61 files)

1. `README.md`
2. `LICENSE`
3. `requirements.txt`
4. `.gitignore`
5. `paper/.gitignore`
6. `paper/README.md`
7. `paper/analysis.py`
8. `paper/gen_revision_tables.py`
9. `paper/gen_tables.py`
10. `paper/main.tex`
11. `paper/make_assets.py`
12. `paper/make_figs.py`
13. `paper/make_graphical_abstract.py`
14. `paper/refs.bib`
15. `paper/rescoring_checks.py`
16. `paper/revision_checks.py`
17. `paper/revision_outputs.json`
18. `paper/sensitivity_checks.py`
19. `paper/figures/fig1_per_doubling.pdf`
20. `paper/figures/fig1_per_doubling.png`
21. `paper/figures/fig2_f1_vs_k.pdf`
22. `paper/figures/fig2_f1_vs_k.png`
23. `paper/figures/fig3_single_units.pdf`
24. `paper/figures/fig3_single_units.png`
25. `paper/figures/fig4_mix_vs_own.pdf`
26. `paper/figures/fig4_mix_vs_own.png`
27. `paper/figures/fig5_fixed_pool_budget.pdf`
28. `paper/figures/fig5_fixed_pool_budget.png`
29. `paper/figures/graphical_abstract.pdf`
30. `paper/figures/graphical_abstract.png`
31. `paper/tables/tab_arms.tex`
32. `paper/tables/tab_budgetk.tex`
33. `paper/tables/tab_c2boot.tex`
34. `paper/tables/tab_c3.tex`
35. `paper/tables/tab_fu.tex`
36. `paper/tables/tab_l1.tex`
37. `paper/tables/tab_l2.tex`
38. `paper/tables/tab_metric.tex`
39. `paper/tables/tab_mix.tex`
40. `paper/tables/tab_scoring.tex`
41. `paper/tables/tab_sens.tex`
42. `paper/tables/tab_spread.tex`
43. `paper/tables/tab_timeline.tex`
44. `paper/tables/tab_train.tex`
45. `docs/AUDIT_REPORT_v3_3.md`
46. `docs/CHANGELOG_v3_1.md`
47. `docs/CHANGELOG_v3_2.md`
48. `docs/CHANGELOG_v3_3.md`
49. `docs/CHANGELOG_v3_4.md`
50. `docs/CHANGELOG_v3_5.md`
51. `docs/CHANGELOG_v3_6.md`
52. `docs/CHANGELOG_v3_7.md`
53. `docs/CHANGELOG_v3_8.md`
54. `docs/CHANGES_AND_CONFLICTS.md`
55. `docs/FRA_join_verification.md`
56. `docs/LEVEL1_FREEZE.md`
57. `docs/LEVEL2_PREREGISTRATION_DRAFT.md`
58. `docs/VERIFICATION_second_review.md`
59. `docs/VERIFICATION_v3_7_vs_review.md`
60. `docs/VERIFICATION_v3_8.md`
61. `scripts/run_all_tests.py`

### Batch 2: level1/ (44 files)

62. `level1/README.md`
63. `level1/closing/Level1_closing_and_Level2_probe.ipynb`
64. `level1/closing/c30_closing.py`
65. `level1/closing/c31_probe.py`
66. `level1/closing/c32_env_closing.py`
67. `level1/closing/c33_match.py`
68. `level1/notebooks/CropGlobe_Level1_v3_8_COLAB.ipynb`
69. `level1/notebooks/CropGlobe_Level1_v3_8_KAGGLE.ipynb`
70. `level1/notebooks/archive/CropGlobe_Level1_v3_1_DUALTRACK_COLAB.ipynb`
71. `level1/notebooks/archive/CropGlobe_Level1_v3_2_FINAL_COLAB.ipynb`
72. `level1/notebooks/archive/CropGlobe_Level1_v3_3_VERIFIED_COLAB.ipynb`
73. `level1/notebooks/archive/CropGlobe_Level1_v3_4_VERIFIED_COLAB.ipynb`
74. `level1/notebooks/archive/CropGlobe_Level1_v3_5_PHASE4_COLAB.ipynb`
75. `level1/notebooks/archive/CropGlobe_Level1_v3_5_PHASE4_KAGGLE.ipynb`
76. `level1/notebooks/archive/CropGlobe_Level1_v3_6_EXTENSION_COLAB.ipynb`
77. `level1/notebooks/archive/CropGlobe_Level1_v3_6_EXTENSION_KAGGLE.ipynb`
78. `level1/notebooks/archive/CropGlobe_Level1_v3_7_REVIEW_COLAB.ipynb`
79. `level1/notebooks/archive/CropGlobe_Level1_v3_7_REVIEW_KAGGLE.ipynb`
80. `level1/notebooks/archive/CropGlobe_Level1_v3_FIXEDMIX_COLAB.ipynb`
81. `level1/src/cells/c01_env.py`
82. `level1/src/cells/c01_env_kaggle.py`
83. `level1/src/cells/c02_config.py`
84. `level1/src/cells/c04_core.py`
85. `level1/src/cells/c07_tests.py`
86. `level1/src/cells/c09_data.py`
87. `level1/src/cells/c10_tests_data.py`
88. `level1/src/cells/c11_acquire.py`
89. `level1/src/cells/c11_acquire_kaggle.py`
90. `level1/src/cells/c12_testsB.py`
91. `level1/src/cells/c13_load.py`
92. `level1/src/cells/c15_plan.py`
93. `level1/src/cells/c17_torch.py`
94. `level1/src/cells/c19_run.py`
95. `level1/src/cells/c21_summary.py`
96. `level1/src/cells/c22_reconcile.py`
97. `level1/src/cells/c23_export.py`
98. `level1/tests/integ38.py`
99. `level1/tests/integ_auto.py`
100. `level1/tests/integ_ext.py`
101. `level1/tests/t_closing.py`
102. `level1/tests/t_match.py`
103. `level1/tests/t_match2.py`
104. `level1/tests/t_probe.py`
105. `level1/tools/check_notebook_matches_cells.py`

### Batch 3: level2/ (33 files)

106. `level2/.gitignore`
107. `level2/README.md`
108. `level2/requirements.txt`
109. `level2/data/fra_region_crop.csv`
110. `level2/docs/CHANGES_v3_AND_REVIEW_CHECKLIST.md`
111. `level2/notebooks/CropGlobe_Level2_FOLLOWUP_no75_76_COLAB.ipynb`
112. `level2/notebooks/CropGlobe_Level2_FRA_source_COLAB_v3.ipynb`
113. `level2/scripts/analyze_results.py`
114. `level2/scripts/run_unit_tests.py`
115. `level2/src/cells/c01_env.py`
116. `level2/src/cells/c02_config.py`
117. `level2/src/cells/c02f_followup.py`
118. `level2/src/cells/c04_core.py`
119. `level2/src/cells/c07_tests.py`
120. `level2/src/cells/c09_data.py`
121. `level2/src/cells/c10_tests_data.py`
122. `level2/src/cells/c11_acquire.py`
123. `level2/src/cells/c11b_usa_order.py`
124. `level2/src/cells/c12_testsB.py`
125. `level2/src/cells/c13_load.py`
126. `level2/src/cells/c15_plan.py`
127. `level2/src/cells/c17_torch.py`
128. `level2/src/cells/c18_approval.py`
129. `level2/src/cells/c18b_rescore.py`
130. `level2/src/cells/c19_run.py`
131. `level2/src/cells/c21_summary.py`
132. `level2/src/cells/c24_level2_analysis.py`
133. `level2/src/cells/c24f_followup_analysis.py`
134. `level2/src/cells/c25_appendix_labels.py`
135. `level2/tests/dryrun_followup.py`
136. `level2/tests/dryrun_level2.py`
137. `level2/tests/test_usa_order.py`
138. `level2/tools/build_notebooks.py`

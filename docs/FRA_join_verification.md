# France: exact row-by-row match, raw CSVs <-> packaged npz (all 64 raw files)

Run on Colab by the user (cell 9), 2026-10-22; every output file reviewed by me, and the full row map re-analysed here.

**Setup.** `median_S2_FRA_month1-12_day5.npz`: 108,599 rows (10 bands x 73 steps). Raw files: all 64 `S2_FRA_<crop>_LAMB93_R<region>.csv`, 108,599 points in total = exactly the number of npz rows.

**Rule found (not assumed).** Of 12 declared candidates one reproduces the npz: 5-day bins `floor((doy-1)/5)`, band-major (10,73), raw reflectance units (99.7% unique on an 11,317-point sample; next best 50.5%). Matching used only bins that contain observations, so it does not depend on how empty bins were filled.

**Full match.** 108,503 of 108,599 points match exactly one npz row (99.91%); 96 match two npz rows (identical feature vectors: soybeans 46, springwheat 28, winterwheat 12, corn 8, other 2); none match zero; no npz row is claimed by two raw points.

**Ordering (the join that otherwise has to be inferred).**
- Each raw file occupies one contiguous block of npz rows, and inside every one of the 64 files `npz_row - position_in_file` is constant.
- The block order is: label id (corn 0, soybeans 1, rice 2, wheat 3, other 6), then token A-Z (springwheat before winterwheat), then region code ascending. This rule predicts the first npz row of **all 64** files exactly. Token Z-A predicts 39 of 64 correctly, region-before-token 42 of 64, so they are rejected.
- Labels: each raw crop token received exactly one npz label (springwheat and winterwheat both label 3).
- The 96 two-way matches lie inside their blocks, so their npz rows are fixed by position.

**What this establishes.** For France, region labels can be read exactly from the raw file names for every one of the 108,599 rows (13 regions: 11, 24, 27, 28, 32, 44, 52, 53, 75, 76, 84, 93, 94). The dataset's ordering convention is crop label, token A-Z, region A-Z. **Not established:** the USA join. The Level 1 pipeline chose the ordering "crop, token, state" (spring before winter); that is the same convention, so this is strong supporting evidence, but the USA wheat files have to be matched before it counts as verified.

**Feasibility note for Level 2.** Points per French region are very uneven: rice exists in only 2 regions (76: 852 points, 93: 1,442), corn in region 94 has 19 points and winter wheat 5, soybeans in region 28 has 33. The usual capacity rules will shrink the usable region pools; the plan must be built from this table, not chosen by hand.

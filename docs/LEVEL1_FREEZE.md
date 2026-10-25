# Level 1 freeze record

Recorded 2026-10-21, **before** the instructor's second-round checks were run. Anything computed for the paper must come from files that match these hashes.

| File | SHA-256 | Notes |
|---|---|---|
| final `results.csv` (3,160 runs) | `38d1876cde08e54cfc827aa6b2813355ef98a09dbd048885a67d58991534f891` | seeds 0-1: 725 runs each; seeds 2-4: 570 each; all status `ok` |
| `manifest.json` (as uploaded) | `c953d4df9503cf7b325d1beedc71dc2c0a0410cced30a48405c5dad3ab7d270b` | version uploaded after seed 0; the run cell appends the v3.8 amendments to the copy on your drive, so **hash your final copy and add it here** |
| `run_environment_seed0.json` | `c7686091fd34d55b27ac17b1888f720b5bf6610ca81b1eccd8546ce48962a114` | |

**To do on your side (2 minutes):** run `sha256sum` (or the closing-analysis notebook, which prints it) on the final `manifest.json`, `run_environment_seed1..4*.json` and `gate_reports/*.txt`, and add the values to this table. Keep a copy of this file with the paper.

**How to verify:** the closing-analysis notebook recomputes the hash and the per-seed run counts and STOPS if either differs. A single changed byte changes the hash.

**What is frozen:** every number in the Level 1 report. **What is allowed to change:** the wording of the paper, the cross-check of the mixed model, and the Level 2 work. If any Level 1 run is ever repeated, that is a new, separately labelled experiment, not a replacement.

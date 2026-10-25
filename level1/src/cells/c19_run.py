# ======================================================================================
# LEVEL 1 RUN.  AUTO_MODE (default): runs EVERY seed and EVERY planned run (aug OFF and aug ON, all arms) in one go, resumable -- run this cell again after a disconnect or time limit.
#   Between seeds the gate report is auto-approved only if the seed is complete with no error / integrity failure; otherwise the run stops and tells you why.
#   AUTO_MODE = False: one seed per execution (set RUN_SEED, approve the previous gate report in the config cell), as before.
# ======================================================================================
assert TESTS_PASSED and SMOKE_OK and APPROVED['plan'], 'test gate, smoke test and plan approval are required'
if AUTO_MODE:
    AUTO_RESULT = run_all_seeds(AUTO_MAX_HOURS)
else:
    assert RUN_SEED in SEEDS, f'RUN_SEED must be one of {SEEDS}'
    assert run_guards(), 'seed-consistency guards failed (see above)'
    if run_seed_cell(RUN_SEED) == 'complete':
        print(f'\n>>> Gate report saved. To continue: review it, then set APPROVED["seed{RUN_SEED}_gate_report"] = True and RUN_SEED = {RUN_SEED + 1}.' if RUN_SEED < SEEDS[-1] else '\n>>> All seeds done: run the summary cell.')

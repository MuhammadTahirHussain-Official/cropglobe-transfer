# ======================================================================================
# LOAD + VERIFY THE DATA  (nothing is trained here)
# ======================================================================================
if PROFILE == 'extension': print('extension profile: the USA state table cache is SHARED with the main study (<main outputs>/cache; same data, reused only if it matches the current index)')
FEATURES, NPZ_LBL, INFO, meta = load_all_data(DATA_DIR, (MAIN_OUT_DIR or OUT_DIR), get_usa_geo_by_file, extra_meta_paths=EXTRA_META_PATHS)
verify_data(FEATURES, NPZ_LBL, INFO, meta)
print('\n================ USA STATE-TABLE GEOGRAPHY CHECK (STOP if violated) ================')
GEOGRAPHY_OBSERVED = check_geography(meta['state'].values, meta['crop'].values)
print('GEOGRAPHY CHECK PASSED: sugarcane and rice rows fall only in the agronomically possible states.')

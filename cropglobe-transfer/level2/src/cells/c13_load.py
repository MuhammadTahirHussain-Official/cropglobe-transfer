# ======================================================================================
# LOAD + VERIFY THE DATA (Level 2: source = France).  Nothing is trained here.
# ======================================================================================
FEATURES, NPZ_LBL, INFO, meta, FRA_META_REPORT = load_all_data(DATA_DIR, OUT_DIR, ROW_MAP_PATH)
os.makedirs(os.path.join(OUT_DIR, 'tables'), exist_ok=True)
atomic_write(os.path.join(OUT_DIR, 'tables', 'fra_region_assignment.json'), lambda f: f.write(json.dumps(FRA_META_REPORT, indent=1)), mode='w')
verify_data(FEATURES, NPZ_LBL, INFO, meta)
print('\n================ SOURCE REGION-TABLE GEOGRAPHY CHECK (STOP if violated) ================')
GEOGRAPHY_OBSERVED = check_geography(meta['state'].values, meta['crop'].values)
_bad = {c: int((meta['crop'] == c).sum()) for c in ('sugarcane', 'cotton') if (meta['crop'] == c).any()}; _check(not _bad, f'France cannot contain sugarcane / cotton rows: {_bad}')
print('GEOGRAPHY CHECK PASSED: rice rows fall only in regions 76 and 93; no sugarcane / cotton rows.')

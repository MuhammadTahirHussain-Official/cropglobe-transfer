# ======================================================================================
# APPENDIX (instructor item c): Belgium / Netherlands / United Kingdom -- label check ONLY.  Descriptive, NO claims, low priority, never gates anything, writes only tables/appendix_label_counts.csv.
# Before any of these countries could be SCORED, its real label set must be verified (a score needs the classes that exist in the test data).  This cell only counts labels; if a file is absent it says so.
# ======================================================================================
def appendix_label_counts(data_dir, countries=('BEL', 'NLD', 'GBR'), log=print):
    rows = []
    for c in countries:
        try: path = find_npz(data_dir, c, log=lambda *a, **k: None)
        except Exception as e: path = None; log(f'  {c}: {type(e).__name__}: {str(e)[:80]}')
        if not path: rows.append(dict(country=c, status='file not found (not downloaded; nothing assumed)', n=0)); continue
        lbl = np.load(path)['lbl'].astype(int); cnt = np.bincount(lbl, minlength=7)
        rows.append(dict(country=c, status='counted', n=int(len(lbl)), **{CROP_LABELS[i]: int(cnt[i]) for i in range(7)}, classes_present=', '.join(CROP_LABELS[i] for i in range(7) if cnt[i] > 0)))
    return pd.DataFrame(rows)
APPENDIX_LABELS = appendix_label_counts(DATA_DIR)
print(APPENDIX_LABELS.to_string(index=False)); os.makedirs(os.path.join(OUT_DIR, 'tables'), exist_ok=True); APPENDIX_LABELS.to_csv(os.path.join(OUT_DIR, 'tables', 'appendix_label_counts.csv'), index=False)
print('APPENDIX ONLY: no claim is made for these countries; a country may be scored only after its label set above has been checked against the paper.')

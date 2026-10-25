# ======================================================================================
# OPTIONAL: export the tiny USA state table so the KAGGLE notebook can skip the raw-data download entirely
# ======================================================================================
_src = os.path.join(OUT_DIR, 'cache', 'usa_meta.csv'); _dst = os.path.join(OUT_DIR, 'for_kaggle'); os.makedirs(_dst, exist_ok=True)
import shutil; shutil.copy(_src, os.path.join(_dst, 'usa_meta.csv'))
print('Upload this file as a Kaggle Dataset and attach it to the Kaggle notebook:\n  ', os.path.join(_dst, 'usa_meta.csv'))
# ======================================================================================
# SETUP for the Level 1 closing analysis + Level 2 data probe (Colab or Kaggle).  Edit only the PATHS below.
# ======================================================================================
import os, sys, glob
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    try:
        from google.colab import drive; drive.mount('/content/drive')
    except Exception as e: print('Drive not mounted:', e)

RESULTS_CSV = ''          # the FINAL results.csv (3,160 runs).  Leave '' to search the usual places.
DATA_DIR    = ''          # folder holding CropGlobe/for_CropNet/*.npz  (needed only for the Level 2 probe)
RAW_USA_DIR = ''          # folder with the raw S2_USA_*.csv files (Level 2 probe; optional)
RAW_FRA_DIR = ''          # folder with the raw S2_FRA_*.csv files (Level 2 probe; optional)

if not RESULTS_CSV:
    found = [p for pat in ('/content/drive/**/phase3_level1_v3_7_review/results.csv', '/kaggle/working/**/phase3_level1_v3_7_review/results.csv', '/kaggle/input/**/results*.csv', './**/results*.csv') for p in glob.glob(pat, recursive=True)]
    print('results.csv candidates:', found[:10] if found else 'NONE FOUND -- set RESULTS_CSV above')
    if len(found) == 1: RESULTS_CSV = found[0]
    elif found: print('More than one candidate: set RESULTS_CSV to the right one (the freeze check below will refuse a wrong file).')
print('RESULTS_CSV =', RESULTS_CSV or '(not set)')

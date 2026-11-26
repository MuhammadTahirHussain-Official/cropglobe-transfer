# ======================================================================================
# [COLAB] ENVIRONMENT: mount Drive, paths (your existing files), libraries, GPU check.
# ======================================================================================
import os, sys, subprocess, importlib.util, glob, json, time
PLATFORM = 'colab'
assert importlib.util.find_spec('google.colab') is not None, 'This notebook is for Google Colab. Use the Kaggle notebook on Kaggle.'
from google.colab import drive
drive.mount('/content/drive')
_candidates = ['/content/drive/.shortcut-targets-by-id/1HTRsB7bxydWFYCZG8Wrj-8IQZJKj2Z8w/cropglobe_assignment',
               '/content/drive/MyDrive/cropglobe_assignment']
PROJECT_ROOT = next((p for p in _candidates if os.path.isdir(p)), None)
assert PROJECT_ROOT, f'project folder not found. Looked in: {_candidates}. Edit PROJECT_ROOT here.'
DATA_DIR   = os.path.join(PROJECT_ROOT, 'CropGlobe')                     # contains for_CropNet/
REPO_DIR   = os.path.join(PROJECT_ROOT, 'CropNet')
NAMESPACE  = 'phase3_level2_fra_source_v1'
OUT_DIR    = os.path.join(PROJECT_ROOT, 'outputs', NAMESPACE)            # NEW folder: the legacy results (outputs/phase3 + backup) are never touched
os.makedirs(OUT_DIR, exist_ok=True)
PLATFORM_TIME_LIMIT_MIN = None                                           # no artificial limit; re-run the run cell after a disconnect to resume
# Optional: the row-map CSV from the French matching run.  The 64 region blocks derived from it are EMBEDDED in the config cell, so this file is not required; if it is found
# (and its SHA-256 equals the verified one) it is re-read and must reproduce the embedded blocks.
ROW_MAP_PATH = None
import hashlib
for _p in glob.glob(os.path.join(PROJECT_ROOT, '**', 'row_map*.csv'), recursive=True):
    if os.path.getsize(_p) > 1_000_000 and hashlib.sha256(open(_p, 'rb').read()).hexdigest() == '93abeb51af5f68998818b770c8c5d853db5307e06f5aaf3dd1a6123d6ed69d9c': ROW_MAP_PATH = _p; break
print('row map (optional cross-check):', ROW_MAP_PATH or 'not found -> the embedded region blocks are used (nothing is lost)')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub', 'scikit-learn', 'scipy', 'tqdm', 'statsmodels'], check=True)
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE -> Runtime > Change runtime type > GPU')
assert torch.cuda.is_available(), 'enable a GPU runtime'
print('project:', PROJECT_ROOT, '\noutputs:', OUT_DIR)
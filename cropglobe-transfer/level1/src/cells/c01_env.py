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
RAW_USA_DIR = os.path.join(PROJECT_ROOT, 'CropGlobe_raw', 'Sentinel-2', 'USA')
REPO_DIR   = os.path.join(PROJECT_ROOT, 'CropNet')
NAMESPACE  = 'phase3_level1_v3_7_review'
OUT_DIR    = os.path.join(PROJECT_ROOT, 'outputs', NAMESPACE)            # NEW folder: the legacy results (outputs/phase3 + backup) are never touched
os.makedirs(OUT_DIR, exist_ok=True)
PLATFORM_TIME_LIMIT_MIN = None                                           # no artificial limit; re-run the run cell after a disconnect to resume
EXTRA_META_PATHS = []
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub', 'geopandas', 'shapely', 'pyproj', 'scikit-learn', 'scipy', 'tqdm'], check=True)
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE -> Runtime > Change runtime type > GPU')
assert torch.cuda.is_available(), 'enable a GPU runtime'
print('project:', PROJECT_ROOT, '\noutputs:', OUT_DIR)
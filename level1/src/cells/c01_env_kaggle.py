# ======================================================================================
# [KAGGLE] ENVIRONMENT: no Drive mount -- your uploaded dataset is read from /kaggle/input (read-only); everything written goes to /kaggle/working.
# Turn ON: Accelerator = GPU (T4 / P100) and, if any file turns out to be missing, Internet.  Add your dataset (cropglobe_assignment) via "Add Input".
# ======================================================================================
import os, sys, subprocess, importlib.util, glob, json, time, socket, shutil
PLATFORM = 'kaggle'
assert os.path.isdir('/kaggle/input') and os.path.isdir('/kaggle/working'), 'This notebook is for Kaggle. Use the Colab notebook on Google Colab.'
KAGGLE_INPUT, KAGGLE_WORK = '/kaggle/input', '/kaggle/working'

def _online(host='huggingface.co', port=443, timeout=6):
    try: socket.create_connection((host, port), timeout=timeout).close(); return True
    except Exception: return False
INTERNET = _online()
print('Internet:', 'ON' if INTERNET else 'OFF  (only needed if a file is missing; Kaggle: Settings > Internet)')

# ---- find YOUR dataset: the folder that holds CropGlobe / CropGlobe_raw / CropNet (Kaggle mounts it as /kaggle/input/<slug>/ or /kaggle/input/datasets/<owner>/<slug>/ ...)
def _find_project_root(maxdepth=6):
    names, hits = {'CropGlobe', 'CropGlobe_raw', 'CropNet'}, {}
    for root, dirs, _ in os.walk(KAGGLE_INPUT):
        if root[len(KAGGLE_INPUT):].count(os.sep) >= maxdepth: dirs[:] = []; continue
        if names & set(dirs): hits[root] = len(names & set(dirs))
    return hits
_hits = _find_project_root()
PROJECT_ROOT = max(_hits, key=lambda p: (_hits[p], -len(p))) if _hits else None            # the folder with the most of the three sub-folders
assert PROJECT_ROOT, ('no folder containing CropGlobe / CropGlobe_raw / CropNet found under /kaggle/input. Did you add the dataset via "Add Input"?  '
                      'Contents of /kaggle/input: ' + str(os.listdir(KAGGLE_INPUT)))
print('your dataset root:', PROJECT_ROOT, '| sub-folders:', sorted(os.listdir(PROJECT_ROOT)))

NAMESPACE = 'phase3_level1_v3_7_review'
OUT_DIR = os.path.join(KAGGLE_WORK, 'outputs', NAMESPACE); os.makedirs(OUT_DIR, exist_ok=True)
# ---- resume after a session ended: attach the previous notebook's OUTPUT (or a dataset made from it) as an input; its results are copied here (never modified in place)
if not os.path.exists(os.path.join(OUT_DIR, 'results.csv')):
    _prev = [r for r, _, f in os.walk(KAGGLE_INPUT) if os.path.basename(r) == NAMESPACE and 'results.csv' in f]
    if _prev:
        print('previous results found in', _prev[0], '-> copying into', OUT_DIR, '(resume).')
        shutil.copytree(_prev[0], OUT_DIR, dirs_exist_ok=True)
PLATFORM_TIME_LIMIT_MIN = 12 * 60            # Kaggle stops a session after ~12 h (and the weekly GPU quota is limited): the run cell is resumable -> "Save Version", then attach the output next session
EXTRA_META_PATHS = []                        # filled by the data cell (a usa_meta.csv found in your dataset)
_du = shutil.disk_usage(KAGGLE_WORK); print(f'/kaggle/working free: {_du.free / 1e9:.1f} GB (limit about 20 GB for files you write)')

_need = {'huggingface_hub': 'huggingface_hub', 'geopandas': 'geopandas', 'shapely': 'shapely', 'pyproj': 'pyproj', 'sklearn': 'scikit-learn', 'scipy': 'scipy', 'tqdm': 'tqdm', 'pandas': 'pandas', 'numpy': 'numpy'}
_missing = [pip for mod, pip in _need.items() if importlib.util.find_spec(mod) is None]
if _missing:
    assert INTERNET, f'python packages missing ({_missing}) and Internet is OFF -> turn Internet on (Settings) and re-run this cell'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *_missing], check=True)
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE -> Settings > Accelerator > GPU')
assert torch.cuda.is_available(), 'enable a GPU accelerator'
print('outputs:', OUT_DIR)

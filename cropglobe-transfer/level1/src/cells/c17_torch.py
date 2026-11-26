# ======================================================================================
# TORCH BACKEND -- the AUTHORS' CropNet and CropGlobeDataset (unchanged, pinned commit), official recipe:
#   CropNet(data_channel=1, classes=7), ALL 7 outputs trained | Adam lr 1e-4, weight_decay 0, no scheduler | batch 256
#   CrossEntropyLoss with balanced class weights over the drawn sample (= the authors' whole source set; CLASS_WEIGHT_SOURCE), zero-sample crops -> weight 0
#   official DOY 121-334 window (aug_mode='none') for validation AND test; training: aug_mode='none' (aug OFF) or the authors' 'augm' (aug ON, arm aug_on only) | validation/test are never augmented
#   TOTAL 200 epochs, early stopping inside them on VALIDATION macro-F1 (patience 15, min 20 epochs, strictly greater), best weights restored and re-verified
# ======================================================================================
assert TESTS_PASSED, 'test gate (Part A + Part B) has not passed -- no GPU time'
assert APPROVED['plan'], 'the plan table has not been approved: set APPROVED["plan"] = True in the config cell after reading it'
import torch
from torch.utils.data import TensorDataset, DataLoader
os.environ.setdefault('TQDM_DISABLE', '1')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE, '|', torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'NO GPU (very slow) -- enable a GPU accelerator')
for _m in ('models', 'data'):
    if _m in sys.modules and REPO_DIR not in str(getattr(sys.modules[_m], '__file__', '') or ''): del sys.modules[_m]
if REPO_DIR in sys.path: sys.path.remove(REPO_DIR)
sys.path.insert(0, REPO_DIR)
from models.CropNet import CropNet
from data.augment import CropGlobeDataset
for _cls in (CropNet, CropGlobeDataset): assert os.path.abspath(inspect.getfile(_cls)).startswith(os.path.abspath(REPO_DIR)), f'{_cls} is not the authors\' class'
assert subprocess.run(['git', '-C', REPO_DIR, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() == EXPECTED_CROPNET_COMMIT
_probe = CropGlobeDataset()
assert (_probe.dayItv, _probe.staDOY, _probe.endDOY, _probe.target_len) == (5, 121, 334, 43), 'unexpected official DOY window'
print(f'official DOY window: day {_probe.staDOY}-{_probe.endDOY}, step {_probe.dayItv} -> {_probe.target_len} steps; model input = 10 bands x {_probe.target_len}')
assert AUG == 'off' and MAX_EPOCHS == 200 and CLASS_WEIGHT_SOURCE == 'train_val' and (LR, WEIGHT_DECAY, BATCH_SIZE) == (AUTHORS_SCRIPT['lr'], 0.0, AUTHORS_SCRIPT['batch_size'])

try:                                                                                   # informational: the authors' own trainer (NOT used here)
    from trainers.torch_trainer import trainer as _auth_trainer
    print('\n--- authors\' trainers/torch_trainer.py::trainer (read-only; compare with this notebook\'s early stopping) ---'); print(inspect.getsource(_auth_trainer)[:3500])
except Exception as _e: print('(could not display the authors\' trainer source:', repr(_e)[:100], ')')

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def official_window(X, y):
    ds = CropGlobeDataset(aug_mode='none')(TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y).long()))
    return ds.tensors

@torch.no_grad()
def predict_tensor(model, X, bs=4096):
    model.eval(); out = []; X = torch.as_tensor(X) if not torch.is_tensor(X) else X
    for i in range(0, len(X), bs): out.append(model(X[i:i + bs].to(DEVICE, dtype=torch.float32)).argmax(1).cpu())
    return torch.cat(out).numpy()

def fit_model(Xtr, ytr, Xva, yva, seed, epochs_override=None, aug='off'):
    max_ep = int(epochs_override or MAX_EPOCHS)                                                    # TOTAL epochs (100); EarlyStopper refuses more
    set_seed(seed)
    ytr_np, yva_np = np.asarray(ytr).astype(int), np.asarray(yva).astype(int)
    w = training_class_weights(ytr_np, yva_np, NUM_CLASSES)                                         # balanced weights over the drawn sample (authors-equivalent); zero-sample crops -> 0
    assert aug in ('off', 'on', 'copy'), aug
    if aug == 'on':                                                                                 # the AUTHORS' augmentation (pinned commit): original + AUG_REPEAT augmented copies of every training row; their rng is unseeded, so we seed it (recorded deviation)
        _ds = CropGlobeDataset(aug_mode='augm', re__peat=AUG_REPEAT); _ds.rng = np.random.default_rng(stable_seed('aug', seed, len(ytr_np)))
        train_ds = _ds(TensorDataset(torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(ytr_np).long()))
        assert len(train_ds) == (1 + AUG_REPEAT) * len(ytr_np) and bool(torch.isfinite(train_ds.tensors[0]).all()), 'augmentation produced an unexpected training set'
    elif aug == 'copy':                                                                             # v3.8 steps-matched control: the SAME number of training rows as aug ON, all exact copies of the originals (no transform) -> same rows per epoch and optimizer steps, no augmentation
        _X = torch.as_tensor(Xtr, dtype=torch.float32); _y = torch.as_tensor(ytr_np).long()
        train_ds = CropGlobeDataset(aug_mode='none')(TensorDataset(torch.cat([_X] * (1 + AUG_REPEAT)), torch.cat([_y] * (1 + AUG_REPEAT))))
        assert len(train_ds) == (1 + AUG_REPEAT) * len(ytr_np), 'copy control produced an unexpected training set'
    else: train_ds = CropGlobeDataset(aug_mode='none')(TensorDataset(torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(ytr_np).long()))
    val_X, val_y = official_window(Xva, yva_np); val_X = val_X.to(DEVICE); val_y_np = val_y.numpy()
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)                 # drop_last=False -> ceil(train/256) steps per epoch
    set_seed(seed)
    model = CropNet(data_channel=1, classes=NUM_CLASSES).to(DEVICE)                               # authors' architecture, 7-class head
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)                  # == authors' optim.Adam(lr); no scheduler
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    last = {}
    def train_one_epoch(ep):
        model.train()
        for bi, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad(set_to_none=True); loss = crit(model(xb), yb)
            if not torch.isfinite(loss): raise FloatingPointError(f'training loss non-finite at epoch {ep + 1}, batch {bi}')            # V33-07: EVERY batch, before the step (was: last batch only)
            loss.backward(); opt.step()
        last['loss'] = float(loss)
        if not math.isfinite(last['loss']): raise FloatingPointError(f'training loss became non-finite at epoch {ep + 1}')
        if not all(bool(torch.isfinite(p_).all()) for p_ in model.parameters()): raise FloatingPointError(f'model parameters non-finite after epoch {ep + 1}')
    def val_score(): return float(f1_score(val_y_np, predict_tensor(model, val_X), average='macro', zero_division=0))
    snapshot = lambda: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out = run_training_loop(train_one_epoch, val_score, snapshot, model.load_state_dict, max_ep)  # <= MAX_EPOCHS (200), early stopping, restore + verification
    return dict(model=model, info=out)

def prepare_eval_sets():
    EX, EY = {}, {}
    for t in TARGETS_TO_RUN:
        y = np.asarray(NPZ_LBL[t]).astype(int); Xw, _ = official_window(FEATURES[t], y); EX[t] = Xw.to(DEVICE); EY[t] = y
        print(f'  eval set {t}: {len(y)} samples (full, de-duplicated), window tensor {tuple(Xw.shape)}')
    return EX, EY

# ---------------------------------------------------------------- 2-EPOCH SMOKE TEST with the REAL model (spec 9: before any sweep)
print('\nSMOKE TEST'); SMOKE_OK = False
m = CropNet(data_channel=1, classes=NUM_CLASSES).to(DEVICE); out = m(torch.randn(8, 1, 10, 43, device=DEVICE)); assert out.shape == (8, 7); out.sum().backward()
print('  [PASS] official CropNet forward/backward (8,1,10,43) ->', tuple(out.shape))
wx, _ = official_window(np.random.rand(6, 1, 10, 73).astype(np.float32), np.zeros(6, dtype=int)); assert tuple(wx.shape) == (6, 1, 10, 43); print('  [PASS] official window (6,1,10,73) -> (6,1,10,43)')
_r = np.random.default_rng(0); _mu = _r.normal(0, 1, (3, 730)).astype(np.float32)
def _fake(n, labs): y = np.repeat(labs, n); return (_mu[y] + _r.normal(0, 1, (len(y), 730))).astype(np.float32).reshape(len(y), 1, 10, 73), y
Xa, ya = _fake(80, [0, 1, 2]); Xb, yb_ = _fake(20, [0, 1, 2]); t0 = time.time(); r = fit_model(Xa, ya, Xb, yb_, 0, epochs_override=3)
assert set(r['info']) >= {'val_f1', 'best_epoch', 'epochs_run', 'stop_reason', 'restore_delta', 'history'} and r['info']['restore_delta'] <= RESTORE_TOL and math.isfinite(r['info']['val_f1']) and r['info']['epochs_run'] == 3 and r['info']['stop_reason'] == 'max_epochs'
print(f"  [PASS] fit_model (3 epochs; 4 crops absent -> weight 0): val F1 {r['info']['val_f1']:.2f}, best epoch {r['info']['best_epoch']}, restore delta {r['info']['restore_delta']:.1e}, {time.time() - t0:.1f}s")
# R7: augmentation ON = authors' 'augm' (seeded here): 3x the training rows, deterministic under the same seed, validation untouched
r2 = fit_model(Xa, ya, Xb, yb_, 0, epochs_override=2, aug='on'); assert r2['info']['epochs_run'] == 2 and math.isfinite(r2['info']['val_f1'])
_d1 = CropGlobeDataset(aug_mode='augm', re__peat=AUG_REPEAT); _d1.rng = np.random.default_rng(1); _d2 = CropGlobeDataset(aug_mode='augm', re__peat=AUG_REPEAT); _d2.rng = np.random.default_rng(1)
_t1 = _d1(TensorDataset(torch.as_tensor(Xa[:30], dtype=torch.float32), torch.as_tensor(ya[:30]).long())); _t2 = _d2(TensorDataset(torch.as_tensor(Xa[:30], dtype=torch.float32), torch.as_tensor(ya[:30]).long()))
assert len(_t1) == 30 * (1 + AUG_REPEAT) and torch.equal(_t1.tensors[0], _t2.tensors[0]) and torch.equal(_t1.tensors[0][:30], official_window(Xa[:30], ya[:30])[0]), 'augmentation is not seed-deterministic / does not keep the unaugmented rows first'
print('  [PASS] augmentation ON (authors\' augm): 3x rows, first block = unaugmented official window, identical under the same seed, 2-epoch fit runs')
r3 = fit_model(Xa, ya, Xb, yb_, 0, epochs_override=2, aug='copy'); assert r3['info']['epochs_run'] == 2 and math.isfinite(r3['info']['val_f1'])
_dc = CropGlobeDataset(aug_mode='none')(TensorDataset(torch.cat([torch.as_tensor(Xa[:30], dtype=torch.float32)] * (1 + AUG_REPEAT)), torch.cat([torch.as_tensor(ya[:30]).long()] * (1 + AUG_REPEAT))))
assert len(_dc) == 30 * (1 + AUG_REPEAT) and all(torch.equal(_dc.tensors[0][i * 30:(i + 1) * 30], official_window(Xa[:30], ya[:30])[0]) for i in range(1 + AUG_REPEAT)), 'copy control is not (1+AUG_REPEAT) exact copies of the unaugmented official window'
print('  [PASS] copy control (v3.8): (1 + AUG_REPEAT)x rows, every block = the unaugmented official window, 2-epoch fit runs')
# V33-07: a non-finite value in ONE batch must stop the run even if the LAST batch of the epoch is finite (the shuffled loader puts the bad sample in a random batch)
_Xl, _yl = _fake(200, [0, 1, 2]); _Xl[[0, 100, 200, 300, 400, 500]] = np.nan; _caught = None          # 600 samples = 3 batches of <=256; NaN samples spread over the shuffled batches
try: fit_model(_Xl, _yl, Xb, yb_, 0, epochs_override=2)
except FloatingPointError as _e: _caught = str(_e)
assert _caught and ('non-finite' in _caught), f'a NaN training sample was not caught: {_caught}'; print(f'  [PASS] non-finite sample in an earlier batch is caught ({_caught})')
EVAL_X, EVAL_Y = prepare_eval_sets(); FIT_FN, PREDICT_FN = fit_model, predict_tensor
print('\n  one REAL cell per target, 2 epochs (full pipeline incl. both scorers + integrity checks; results NOT kept):')
for t in TARGETS_TO_RUN:
    cl = [c for c in PLAN_CELLS_RUN if c['target'] == t]
    if not cl: print(f'   {t}: no runnable cell -- skipped'); continue
    cell = cl[0]; t0 = time.time(); row, preds, y_true, _sc = run_one(cell, 0, 'SMOKE|' + make_run_id(cell, 0), tempfile.gettempdir(), epochs_override=2, save=False)
    assert len(preds) == len(y_true) == row['test_n'] and row['epochs_run'] <= 2 and row['integrity_pass'] is True, f'{t}: smoke run failed integrity: {row}'
    print(f"   [PASS] {t} B{cell['budget']} k{cell['k']} {row['states']}: {row['primary_view']} mF1 {_pct(row['primary_f1'])} | paper-rule mF1 {_pct(row['paper_f1'])} | integrity PASS | {time.time() - t0:.1f}s (2 epochs: skill is not meaningful)")
for _tag, _cl in (('k-center', [c for c in PLAN_CELLS_RUN if c.get('sel') == 'kcenter']), ('aug ON', [c for c in PLAN_CELLS if c.get('aug') == 'on']), ('aug COPY (steps-matched control)', [c for c in PLAN_CELLS if c.get('aug') == 'copy'])):
    if not _cl: continue
    cell = _cl[0]; t0 = time.time(); row, preds, y_true, _sc = run_one(cell, 0, 'SMOKE|' + make_run_id(cell, 0), tempfile.gettempdir(), epochs_override=2, save=False)
    assert row['integrity_pass'] is True and row['aug'] == cell.get('aug', 'off') and row['sample_sel'] == cell.get('sel', 'random'), f'{_tag}: smoke run failed: {row}'
    print(f"   [PASS] {_tag} {cell['target']} B{cell['budget']} k{cell['k']} {row['states']}: steps/epoch {row['steps_per_epoch']} (train rows seen per epoch {row['train_n_effective']}) | integrity PASS | {time.time() - t0:.1f}s")
SMOKE_OK = True; print('\nSMOKE TEST PASSED  -> run the seed-0 cell next.')
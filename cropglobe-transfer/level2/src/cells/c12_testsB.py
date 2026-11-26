# ======================================================================================
# PRE-TRAINING TEST GATE, PART B -- a REAL gate.  Our `existing_release` remap must equal the AUTHORS' OWN util/label.py::remap_prediction on 5000 random predictions
# for ARG, CHN, AUS and FRA, or training stays blocked.  All three named rules are compared with the authors' function and the full table is printed and saved.
# The authors' signature is (test_pred, target, crop_label): called exactly as remap_prediction(list(y_pred), TARGET, CROP_ID), uppercase target, NO fallback spellings.
# ======================================================================================
import inspect, importlib
for _m in ('util', 'util.label'):
    if _m in sys.modules and REPO_DIR not in str(getattr(sys.modules[_m], '__file__', '') or ''): del sys.modules[_m]
if REPO_DIR in sys.path: sys.path.remove(REPO_DIR)
sys.path.insert(0, REPO_DIR)
UL = importlib.import_module('util.label'); assert os.path.abspath(inspect.getfile(UL)).startswith(os.path.abspath(REPO_DIR)), 'util.label is not the authors\' module'
AUTH_REMAP = UL.remap_prediction
_sig = list(inspect.signature(AUTH_REMAP).parameters)
print('authors\' remap_prediction signature:', inspect.signature(AUTH_REMAP)); print(inspect.getsource(AUTH_REMAP))
TESTS_B_PASSED = False; AUTH_REMAP_REPORT = []; PLAN_NOTE_ARG = ''
if _sig != ['test_pred', 'target', 'crop_label']:
    raise RuntimeError(f'STOP -- the authors\' remap_prediction signature is {_sig}, not (test_pred, target, crop_label) as the spec states. Report; nothing is guessed.')

def call_authors_remap(y_pred, target):
    out = AUTH_REMAP(list(np.asarray(y_pred).tolist()), target, CROP_ID)                  # uppercase target, CROP_ID as crop_label, no retries
    out = out.tolist() if hasattr(out, 'tolist') else list(out); arr = np.asarray(out).astype(int)
    if arr.shape != (len(y_pred),): raise RuntimeError(f'authors\' remap_prediction returned shape {arr.shape} for {len(y_pred)} predictions')
    return arr

# ---- 1) the authors' function must REALLY change a probe (so a no-op can never pass for a match)
_O = CROP_ID['other']; _allc = np.arange(7); _probe_expect = {
    'ARG': [c if c != CROP_ID['wheat'] else _O for c in _allc],                                         # only wheat -> other
    'CHN': [c if c in (CROP_ID['corn'], CROP_ID['soybeans'], CROP_ID['rice']) else _O for c in _allc], # keeps ONLY corn / soybeans / rice
    'AUS': [c if c in (CROP_ID['sugarcane'], CROP_ID['cotton']) else _O for c in _allc]}                # keeps ONLY sugarcane / cotton
_probe_bad = []
for _t_, _exp in _probe_expect.items():
    _got = call_authors_remap(_allc, _t_).tolist()
    print(f'  probe {_t_}: authors map {dict(zip(CROP_ORDER, [CROP_LABELS[g] for g in _got]))}' + ('' if _got == _exp else f'   <-- expected {[CROP_LABELS[e] for e in _exp]}'))
    if _got == list(_allc): _probe_bad.append(f'{_t_}: the authors\' function returned the probe UNCHANGED (a no-op)')
    elif _got != _exp: _probe_bad.append(f'{_t_}: authors map {[CROP_LABELS[g] for g in _got]} but the spec says {[CROP_LABELS[e] for e in _exp]}')
if _probe_bad: raise RuntimeError('STOP -- the authors\' remap_prediction does not behave as the spec states. Report, do not resolve silently:\n  - ' + '\n  - '.join(_probe_bad))
print('  probes OK: the authors\' function really changes predictions (ARG wheat->other; CHN keeps only corn/soybeans/rice; AUS keeps only sugarcane/cotton)')

# ---- 2) full three-rule comparison on 5000 random predictions per target
_rng = np.random.default_rng(7)
for _t_ in ['ARG', 'CHN', 'AUS', 'FRA', 'USA']:
    yp = _rng.integers(0, 7, 5000); ref = call_authors_remap(yp, _t_); row = dict(target=_t_, n=5000)
    for name, fn in REMAPS.items(): row[f'{name}_identical'] = bool((fn(yp, _t_) == ref).all()); row[f'{name}_n_diff'] = int((fn(yp, _t_) != ref).sum())
    _mp = call_authors_remap(_allc, _t_); row['authors_class_mapping'] = ','.join(f'{CROP_LABELS[a]}->{CROP_LABELS[int(m)]}' for a, m in zip(_allc, _mp) if a != m) or '(none)'
    AUTH_REMAP_REPORT.append(row)
REP = pd.DataFrame(AUTH_REMAP_REPORT); print('\nTHREE-RULE COMPARISON vs the authors\' remap_prediction (identical on 5000 random predictions?):'); print(REP.to_string(index=False))
os.makedirs(os.path.join(OUT_DIR, 'tables'), exist_ok=True); atomic_write(os.path.join(OUT_DIR, 'tables', 'authors_remap_comparison.csv'), lambda f: REP.to_csv(f, index=False), mode='w')
_diff = lambda rule: sorted(r['target'] for r in AUTH_REMAP_REPORT if not r[f'{rule}_identical'])
print(f'\n  paper_text_v3 differs from the released code on: {_diff("paper_text_v3") or "(nowhere)"}   (expected: ARG only)')
print(f'  review_targetmap differs from the released code on: {_diff("review_targetmap") or "(nowhere)"}   (expected: FRA only; USA defines all 7 classes, so it never differs)')
print(f'  existing_release differs from the released code on: {_diff("existing_release") or "(nowhere)"}   (required: nowhere)')
_arg_map = [r for r in AUTH_REMAP_REPORT if r['target'] == 'ARG'][0]['authors_class_mapping']
PLAN_NOTE_ARG = ('paper sec.4.4 says no alignment for ARG; the released code maps wheat->other. Both are scored; no choice is made.' if 'wheat->other' in _arg_map
                 else f'paper sec.4.4 says no alignment for ARG; the released code maps: {_arg_map}. Both are scored; no choice is made.')
# ---- 3) the gate
_fail = [r['target'] for r in AUTH_REMAP_REPORT if not r['existing_release_identical']]
if _fail:
    raise RuntimeError(f'PART B FAILED -- existing_release differs from the authors\' released remap_prediction on {_fail} (see the table above / tables/authors_remap_comparison.csv). Training is BLOCKED. Report; nothing is resolved silently.')
TESTS_B_PASSED = True
TESTS_PASSED = bool(TESTS_A_PASSED and TESTS_B_PASSED)
print('\nTEST GATE: Part A', 'PASS' if TESTS_A_PASSED else 'FAIL', '| Part B', 'PASS' if TESTS_B_PASSED else 'FAIL', '->', 'training allowed after the plan approval' if TESTS_PASSED else 'TRAINING BLOCKED')

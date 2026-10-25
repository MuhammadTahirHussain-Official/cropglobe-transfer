# ======================================================================================
# CORE LOGIC (pure numpy / pandas / sklearn -- no torch, no network).  Unit-tested in the next cell.
# ======================================================================================
class TooFewClasses(Exception): pass
class InfeasibleSplit(Exception): pass
class IntegrityFailure(Exception): pass

def _check(cond, msg):
    if not cond:                      # cannot be disabled with python -O
        raise AssertionError(msg)

def stable_seed(*parts):
    return zlib.crc32('|'.join(str(p) for p in parts).encode()) & 0x7fffffff

def atomic_write(path, writer, mode='wb'):
    # write to a temp file in the same folder, fsync, then os.replace: a reader never sees a half-written file
    d = os.path.dirname(path); os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp_', suffix=os.path.basename(path))
    try:
        with os.fdopen(fd, mode) as f:
            writer(f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp): os.remove(tmp)
        raise

# ------------------------------------------------------------------------------ SCORING (spec 3)
def binary_collapse(y_true, y_pred, positive_name):
    # EVALUATION VIEW ONLY.  Every non-positive class, true OR predicted, is 'other'.  Nothing is dropped, nothing is set to -1.
    pid = CROP_ID[positive_name]
    return (np.asarray(y_true) == pid).astype(int), (np.asarray(y_pred) == pid).astype(int)

def remap_existing_release(y_pred, target):
    # EXISTING rule, PRESERVED (transcribed from the spec's description of the authors' released util/label.py::remap_prediction; Part B compares it with the
    # released function in Colab and is a real gate):  AUS keep {sugarcane, cotton} | CHN keep {corn, soybeans, rice} | ARG only wheat -> other | FRA no remap
    y = np.asarray(y_pred).astype(int); o = CROP_ID['other']
    if target == 'AUS': return np.where(np.isin(y, [CROP_ID['sugarcane'], CROP_ID['cotton']]), y, o)
    if target == 'CHN': return np.where(np.isin(y, [CROP_ID['corn'], CROP_ID['soybeans'], CROP_ID['rice']]), y, o)
    if target == 'ARG': return np.where(y == CROP_ID['wheat'], o, y)
    if target == 'FRA': return y.copy()
    raise ValueError(f'unknown target {target}')

def _keep_defined(y_pred, target):
    keep = [CROP_ID[c] for c in DEFINED_CLASSES[target]]; y = np.asarray(y_pred).astype(int)
    return np.where(np.isin(y, keep), y, CROP_ID['other'])

def remap_paper_text_v3(y_pred, target):
    # VERIFIED against arXiv 2509.03497v3 sec.4.4 "Category Alignment": only when AUS or CHN are the target are predictions of classes outside their label set
    # reassigned to Other; "no additional alignment is required for other transfer scenarios" (ARG, FRA: predictions untouched).
    if target in ('AUS', 'CHN'): return _keep_defined(y_pred, target)
    if target in ('ARG', 'FRA'): return np.asarray(y_pred).astype(int).copy()
    raise ValueError(f'unknown target {target}')

def remap_review_targetmap(y_pred, target):
    # REVIEW-PROPOSED: for EVERY target, a prediction of a crop that the target map does not define (paper Table 2 label set) is mapped to Other.
    #   ARG: wheat | CHN: wheat, sugarcane, cotton | AUS: corn, soybeans, rice, wheat | FRA: sugarcane, cotton
    if target not in DEFINED_CLASSES: raise ValueError(f'unknown target {target}')
    return _keep_defined(y_pred, target)

REMAPS = {'existing_release': remap_existing_release, 'paper_text_v3': remap_paper_text_v3, 'review_targetmap': remap_review_targetmap}

def remap_difference_table():
    # which targets / predicted classes the three rules treat differently (printed in the plan cell so the disagreement is visible, not silent)
    rows = []
    for t in ['ARG', 'CHN', 'AUS', 'FRA']:
        allc = np.arange(7); maps = {k: f(allc, t) for k, f in REMAPS.items()}
        rows.append(dict(target=t, **{k: ','.join(f'{CROP_LABELS[a]}->{CROP_LABELS[b]}' for a, b in zip(allc, m) if a != b) or '(none)' for k, m in maps.items()},
                         existing_eq_paper_text=bool((maps['existing_release'] == maps['paper_text_v3']).all()), existing_eq_review=bool((maps['existing_release'] == maps['review_targetmap']).all())))
    return pd.DataFrame(rows)

def _np_prf(yt, yp, labels):
    # scorer #2: from scratch, numpy only (no sklearn)
    P, R, F = [], [], []
    for c in labels:
        tp = float(np.sum((yt == c) & (yp == c))); fp = float(np.sum((yt != c) & (yp == c))); fn = float(np.sum((yt == c) & (yp != c)))
        P.append(tp / (tp + fp) if tp + fp > 0 else 0.0); R.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
        d = 2 * tp + fp + fn; F.append(2 * tp / d if d > 0 else 0.0)
    return np.array(P), np.array(R), np.array(F)

def _sk_prf(yt, yp, labels):
    p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=list(labels), zero_division=0)
    return p, r, f

def _expand(vals, labels, n=7):
    out = np.full(n, np.nan); out[np.asarray(labels, dtype=int)] = vals; return out

def macro_labels_present_sklearn(y_true, y_pred_remapped):
    return float(f1_score(y_true, y_pred_remapped, average='macro', labels=np.unique(y_true), zero_division=0))

def score_run(target, y_true, y_pred, train_counts):
    # y_true / y_pred : integer ids 0..6 of the FULL target set.  train_counts : crop counts (length 7) of the TRAIN split.
    yt = np.asarray(y_true).astype(int); yp = np.asarray(y_pred).astype(int); tc = np.asarray(train_counts).astype(int)
    _check(yt.ndim == 1 and yt.shape == yp.shape and len(yt) > 0, 'y_true / y_pred empty or length mismatch')
    _check(yt.min() >= 0 and yt.max() <= 6 and yp.min() >= 0 and yp.max() <= 6, 'class ids outside 0..6')
    present = np.unique(yt); O = CROP_ID['other']
    outside = [CROP_LABELS[int(c)] for c in present if CROP_LABELS[int(c)] not in DEFINED_CLASSES[target]]          # V33-08: scoring assumes y_true is already in the target label space (paper Table 2)
    _check(not outside, f'{target}: y_true contains classes outside the target label set {DEFINED_CLASSES[target]}: {outside} -- the predictions-only remap would be inconsistent with such a truth; STOP')
    if len(present) < 2:
        raise TooFewClasses(f'{target}: true labels contain {len(present)} class(es) -- need >= 2; run stopped')
    yr = remap_existing_release(yp, target)
    # ---- EXISTING release remap: macro-F1 over the classes present in y_true (labels=np.unique(y_true)), OA over ALL samples
    p_s, r_s, f_s = _sk_prf(yt, yr, present); p_n, r_n, f_n = _np_prf(yt, yr, present)
    paper_f1, paper_oa = macro_labels_present_sklearn(yt, yr), float(accuracy_score(yt, yr))
    paper_f1_n, paper_oa_n = float(f_n.mean()), float((yt == yr).mean())
    diffs = [abs(paper_f1 - paper_f1_n), abs(paper_oa - paper_oa_n), *np.abs(f_s - f_n), *np.abs(p_s - p_n), *np.abs(r_s - r_n)]
    # ---- baselines in the existing-release-remap view (all-'other'; majority class of the true labels), scored with the same remap
    maj = int(np.bincount(yt, minlength=7).argmax())
    base = dict(paper_all_other=float(_np_prf(yt, remap_existing_release(np.full_like(yt, O), target), present)[2].mean()),
                paper_majority=float(_np_prf(yt, remap_existing_release(np.full_like(yt, maj), target), present)[2].mean()))
    # ---- coverage-conditional (labelled; never replaces the main score): classes with >= 30 TRAINING samples
    cov_idx = [i for i, c in enumerate(present) if tc[c] >= COVERAGE_MIN_TRAIN]
    cov_paper = float(np.mean(f_s[cov_idx])) if cov_idx else float('nan')
    absent_train = [CROP_LABELS[int(c)] for c in present if tc[c] == 0]            # stay IN the score at their F1; only flagged
    res = dict(n=int(len(yt)), n_true_classes=int(len(present)), present=[CROP_LABELS[int(c)] for c in present],
               paper_f1=paper_f1, paper_oa=paper_oa, paper_prec=_expand(p_s, present), paper_rec=_expand(r_s, present), paper_f1_cls=_expand(f_s, present),
               cov_cond_paper_f1=cov_paper, cov_cond_paper_classes=[CROP_LABELS[int(present[i])] for i in cov_idx], absent_train_classes=absent_train,
               yp_remapped=yr, cm7=confusion_matrix(yt, yp, labels=list(range(7))), cm_paper=confusion_matrix(yt, yr, labels=list(range(7))),
               bin_f1=float('nan'), bin_oa=float('nan'), cm_binary=None, cov_cond_binary_f1=float('nan'), bin_prec=None, bin_rec=None, bin_f1_cls=None,
               baseline_all_other_primary=base['paper_all_other'], baseline_majority_primary=base['paper_majority'], baseline_all_other_paper=base['paper_all_other'],
               baseline_majority_paper=base['paper_majority'], primary_view='existing_release_remap', positive=None)
    # ---- seven-class RAW track (existing): no remap; macro over the classes present in the target truth (never over crops the target lacks)
    p7_s, r7_s, f7_s = _sk_prf(yt, yp, present); p7_n, r7_n, f7_n = _np_prf(yt, yp, present)
    res.update(sevenclass_f1=float(f1_score(yt, yp, average='macro', labels=present, zero_division=0)), sevenclass_oa=float(accuracy_score(yt, yp)))
    diffs += [abs(res['sevenclass_f1'] - float(f7_n.mean())), abs(res['sevenclass_oa'] - float((yt == yp).mean())), *np.abs(f7_s - f7_n)]
    # ---- PAPER-TEXT and REVIEW rules (new, separately named; each scored by BOTH scorers, with its own baselines and coverage-conditional score)
    for tag, fn in (('papertext', remap_paper_text_v3), ('review', remap_review_targetmap)):
        yrr = fn(yp, target); _p, _r, f_s2 = _sk_prf(yt, yrr, present); _p2, _r2, f_n2 = _np_prf(yt, yrr, present)
        f1_s2 = float(f1_score(yt, yrr, average='macro', labels=present, zero_division=0)); oa_s2 = float(accuracy_score(yt, yrr))
        diffs += [abs(f1_s2 - float(f_n2.mean())), abs(oa_s2 - float((yt == yrr).mean())), *np.abs(f_s2 - f_n2)]
        res[f'{tag}_f1'], res[f'{tag}_oa'], res[f'{tag}_f1_cls'] = f1_s2, oa_s2, _expand(f_s2, present); res[f'cm_{tag}'] = confusion_matrix(yt, yrr, labels=list(range(7)))
        res[f'baseline_all_other_{tag}'] = float(_np_prf(yt, fn(np.full_like(yt, O), target), present)[2].mean()); res[f'baseline_majority_{tag}'] = float(_np_prf(yt, fn(np.full_like(yt, maj), target), present)[2].mean())
        ci = [i for i, c_ in enumerate(present) if tc[c_] >= COVERAGE_MIN_TRAIN]; res[f'cov_cond_{tag}_f1'] = float(np.mean(f_s2[ci])) if ci else float('nan')
        res[f'{tag}_pred_unique'] = int(len(np.unique(yrr)))
    if target in BINARY_TARGETS:
        pos = BINARY_TARGETS[target]; pid = CROP_ID[pos]; bt, bp = binary_collapse(yt, yp, pos); lab = [0, 1]
        if len(np.unique(bt)) < 2:                                   # STOP/FLAG: the binary ground truth must contain both classes
            raise TooFewClasses(f'{target}: the binary ({pos} vs other) ground truth contains only {len(np.unique(bt))} class -- binary view undefined; run stopped')
        pb_s, rb_s, fb_s = _sk_prf(bt, bp, lab); pb_n, rb_n, fb_n = _np_prf(bt, bp, lab)
        bin_f1 = float(f1_score(bt, bp, average='macro', labels=lab, zero_division=0)); bin_oa = float(accuracy_score(bt, bp))
        diffs += [abs(bin_f1 - float(fb_n.mean())), abs(bin_oa - float((bt == bp).mean())), *np.abs(fb_s - fb_n), *np.abs(pb_s - pb_n), *np.abs(rb_s - rb_n)]
        n_pos_tr, n_oth_tr = int(tc[pid]), int(tc.sum() - tc[pid]); cov_b = [i for i, n in enumerate([n_oth_tr, n_pos_tr]) if n >= COVERAGE_MIN_TRAIN]   # index 0 = other, 1 = positive
        maj_b = int(np.bincount(bt, minlength=2).argmax())
        res.update(bin_f1=bin_f1, bin_oa=bin_oa, cm_binary=confusion_matrix(bt, bp, labels=lab), bin_prec=pb_s, bin_rec=rb_s, bin_f1_cls=fb_s,
                   cov_cond_binary_f1=float(np.mean(fb_s[cov_b])) if cov_b else float('nan'), primary_view=f'binary_{pos}', positive=pos,
                   baseline_all_other_primary=float(_np_prf(bt, np.zeros_like(bt), lab)[2].mean()), baseline_majority_primary=float(_np_prf(bt, np.full_like(bt, maj_b), lab)[2].mean()),
                   binary_pred_unique=int(len(np.unique(bp))))
        if n_pos_tr == 0: res['absent_train_classes'] = sorted(set(res['absent_train_classes']) | {pos})
    res['primary_f1'] = res['bin_f1'] if target in BINARY_TARGETS else res['paper_f1']
    res['primary_oa'] = res['bin_oa'] if target in BINARY_TARGETS else res['paper_oa']
    res['baseline_best_primary'] = max(res['baseline_all_other_primary'], res['baseline_majority_primary'])
    res['baseline_best_review'] = max(res['baseline_all_other_review'], res['baseline_majority_review'])
    res['scorer_maxdiff'] = float(max(diffs))
    res['primary_pred_unique'] = res.get('binary_pred_unique', int(len(np.unique(yr))))
    return res

# ------------------------------------------------------------------------------ GATES (spec 6)
def evaluate_integrity(sc, preds, split_chk, info):
    # INTEGRITY failures (abort after 5 consecutive).  Skill outcomes are NOT here.
    p = np.asarray(preds); n_test = sc['n']
    finite_vals = [sc['primary_f1'], sc['paper_f1'], sc['paper_oa'], sc['review_f1'], sc['review_oa'], sc['papertext_f1'], sc['sevenclass_f1'], sc['baseline_all_other_primary'], sc['baseline_majority_primary'], info.get('val_f1', np.nan), info.get('restore_delta', np.nan)]
    if sc['cm_binary'] is not None: finite_vals += [sc['bin_f1'], sc['bin_oa']]
    I = {'I1_true_classes_ge2': bool(sc['n_true_classes'] >= 2),
         'I2_predictions_valid': bool(len(p) == n_test and p.min() >= 0 and p.max() <= 6 and int(sc['cm7'].sum()) == n_test and int(sc['cm_paper'].sum()) == n_test and int(sc['cm_review'].sum()) == n_test
                                       and (sc['cm_binary'] is None or int(sc['cm_binary'].sum()) == n_test)),
         'I3_split_exact': bool(split_chk['ok']),
         'I4_restore_matches_best': bool(info.get('restore_delta', np.inf) <= RESTORE_TOL),
         'I5_finite': bool(all(np.isfinite(finite_vals))),
         'I6_scorers_agree': bool(sc['scorer_maxdiff'] <= SCORER_TOL)}
    return I, all(I.values())

def skill_flags(sc, info):
    # SKILL outcomes: recorded and warned, NEVER an abort
    f = []
    if info.get('val_f1', 1.0) < VAL_F1_WARN: f.append(f'WARN_VAL_F1_LT_{VAL_F1_WARN:.2f}')
    if not (sc['primary_f1'] > sc['baseline_best_primary']): f.append('WARN_NOT_BETTER_THAN_BASELINE')
    if not (sc['review_f1'] > sc['baseline_best_review']): f.append('WARN_REVIEW_TRACK_NOT_BETTER_THAN_BASELINE')
    if sc['primary_pred_unique'] < 2: f.append('WARN_COLLAPSED_PREDICTIONS')
    if info.get('best_epoch', 0) >= UNDERTRAINED_EPOCH: f.append('possibly_undertrained')
    if sc['absent_train_classes']: f.append('FLAG_ABSENT_TRAIN_CLASS:' + '+'.join(sc['absent_train_classes']))
    return f

# ------------------------------------------------------------------------------ EARLY STOPPING (pure; the torch loop uses it)
class EarlyStopper:
    # validation macro-F1; strictly greater than best + MIN_DELTA counts as an improvement; never more than max_epochs
    def __init__(self, max_epochs=None, patience=None, min_epochs=None, min_delta=None):
        self.max_epochs = MAX_EPOCHS if max_epochs is None else max_epochs; self.patience = PATIENCE if patience is None else patience
        self.min_epochs = MIN_EPOCHS if min_epochs is None else min_epochs; self.min_delta = MIN_DELTA if min_delta is None else min_delta
        _check(self.max_epochs <= MAX_EPOCHS, f'max_epochs {self.max_epochs} would exceed the hard cap {MAX_EPOCHS}')
        self.best, self.best_epoch, self.bad, self.epoch, self.reason = -np.inf, 0, 0, 0, None
    def step(self, score):
        self.epoch += 1; improved = score > self.best + self.min_delta
        if improved: self.best, self.best_epoch, self.bad = score, self.epoch, 0
        else: self.bad += 1
        if self.epoch >= self.min_epochs and self.bad >= self.patience: self.reason = 'early_stop_patience'
        elif self.epoch >= self.max_epochs: self.reason = 'max_epochs'
        return improved, self.reason is not None

def balanced_class_weights(y, n_classes=7):
    # == sklearn compute_class_weight('balanced') over the classes present; classes with ZERO samples get weight 0 (no division by zero)
    y = np.asarray(y).astype(int); counts = np.bincount(y, minlength=n_classes).astype(float)
    present = counts > 0; w = np.zeros(n_classes)
    w[present] = len(y) / (present.sum() * counts[present])
    return w

def training_class_weights(ytr, yva, n_classes=7):
    # CLASS_WEIGHT_SOURCE = 'train_val' (default): balanced weights over the WHOLE drawn sample = what the authors' script does with its whole source set;
    #                       'train': training labels only.  A crop with ZERO samples gets weight 0 (the head has 7 outputs; the authors' vector is only as long as the classes present).
    _check(CLASS_WEIGHT_SOURCE in ('train_val', 'train'), 'CLASS_WEIGHT_SOURCE must be train_val or train')
    y = np.concatenate([np.asarray(ytr), np.asarray(yva)]) if CLASS_WEIGHT_SOURCE == 'train_val' else np.asarray(ytr)
    return balanced_class_weights(y, n_classes)

def run_training_loop(train_one_epoch, val_score, snapshot, restore, max_epochs=None):
    # The epoch loop, framework-free (unit-tested): <= max_epochs TOTAL epochs (100), early stopping inside them on validation macro-F1
    # (patience 15, min 20 epochs, strictly greater), best weights restored, and the restored model must reproduce its best validation score.
    max_epochs = MAX_EPOCHS if max_epochs is None else max_epochs; es = EarlyStopper(max_epochs=max_epochs); best, hist = None, []
    for ep in range(max_epochs):
        train_one_epoch(ep); f = float(val_score()); hist.append(f); improved, stop = es.step(f)
        if improved: best = snapshot()
        if stop: break
    if best is None: raise FloatingPointError('no finite validation score was ever produced')
    restore(best); delta = abs(float(val_score()) - es.best)
    return dict(val_f1=es.best, best_epoch=es.best_epoch, epochs_run=es.epoch, stop_reason=es.reason, restore_delta=delta, history=hist)

def steps_per_epoch(n_train, batch_size=None): return int(math.ceil(n_train / (BATCH_SIZE if batch_size is None else batch_size)))     # DataLoader(drop_last=False)

import collections
def check_geography(state, crop):
    # STOP if sugarcane / rice rows fall outside the agronomically possible US states; print the observed state sets (with row counts) for sugarcane, rice and cotton.
    st, cr = np.asarray(state).astype(str), np.asarray(crop).astype(str); out = {}; viol = []
    for name in list(GEOGRAPHY_RULES) + [c for c in GEOGRAPHY_REPORT_ONLY if c not in GEOGRAPHY_RULES]:
        vc = pd.Series(st[cr == name]).value_counts(); out[name] = vc.to_dict(); allowed = GEOGRAPHY_RULES.get(name)
        print(f'  {name:10s}: {int(vc.sum())} rows in {len(vc)} states -> ' + ', '.join(f'{s}:{n}' for s, n in vc.items()) + (f'   [allowed: {allowed}]' if allowed else '   [report only]'))
        if allowed:
            bad = {s: int(n) for s, n in vc.items() if s not in allowed}
            if sum(bad.values()) > GEOGRAPHY_MAX_OUTSIDE_ROWS:
                viol.append(f'{name}: {sum(bad.values())} of {int(vc.sum())} rows ({100 * sum(bad.values()) / max(int(vc.sum()), 1):.2f}%) lie outside {allowed}: {bad}  [tolerance GEOGRAPHY_MAX_OUTSIDE_ROWS={GEOGRAPHY_MAX_OUTSIDE_ROWS}]')
    if viol: raise RuntimeError('STOP -- USA state table contradicts crop geography (wrong state assignment or labels). Report, do not resolve silently:\n  - ' + '\n  - '.join(viol) +
                                '\nDecide deliberately: (a) a handful of rows in a NEIGHBOURING state may be boundary error (states are assigned from the 1:20m Census polygons) -> set GEOGRAPHY_MAX_OUTSIDE_ROWS to a stated number and record why; '
                                '(b) many rows or non-neighbouring states -> the state table or labels are wrong; do not run. The observed counts are saved in run_environment_seed*.json.')
    return out

# ------------------------------------------------------------------------------ DESIGN: two composition modes kept side by side
BASE_MODE, FULL_MODE = 'key_plus_natural', 'fixed_full'
def effective_key_shares(target):
    if target == 'AUS': return dict(AUS_KEY_SHARES_IF_CONFIRMED or {})
    return dict(KEY_SHARES[target])

def shares_to_counts(shares, B, exact=False):
    # exact=True: every share x B must be an integer (existing baseline); exact=False: largest-remainder rounding to EXACTLY B (fixed_full proposals)
    raw = {c: s * B for c, s in shares.items() if s > 0}
    _check(exact or abs(sum(shares.values()) - 1.0) < 1e-9, 'a complete composition must sum to 1')
    if exact:
        for c, n in raw.items(): _check(abs(n - round(n)) < 1e-9, f'share {shares[c]} x B={B} = {n} is not an integer')
        return {c: int(round(n)) for c, n in raw.items()}
    fl = {c: int(math.floor(n + 1e-9)) for c, n in raw.items()}; rem = B - sum(fl.values())
    for c in sorted(raw, key=lambda c: (-(raw[c] - fl[c]), CROP_ID[c]))[:rem]: fl[c] += 1
    _check(sum(fl.values()) == B, 'largest-remainder rounding failed'); return fl

# PROPOSALS for the complete fixed composition.  They are NOT approved defaults: no proportion is invented, each is a documented candidate whose capacity is checked
# in the plan cell.  They use only the crop definitions of the target map (paper Table 2 label set) and the existing key shares -- never target sample counts.
def _prop_A(target):                                  # key-crop share as in the baseline; the remainder split EQUALLY over the other classes the target map defines
    ks = effective_key_shares(target); rest = [c for c in DEFINED_CLASSES[target] if c not in ks]
    return {**ks, **{c: (1 - sum(ks.values())) / len(rest) for c in rest}} if rest else dict(ks)
def _prop_B(target):                                  # EQUAL share over all classes the target map defines (key crop not privileged)
    d = DEFINED_CLASSES[target]; return {c: 1.0 / len(d) for c in d}
FIXED_COMPOSITION_PROPOSALS = {'A_key_plus_equal_rest': _prop_A, 'B_equal_over_target_classes': _prop_B}

def fixed_shares(target):
    _check(FIXED_COMPOSITION_CHOICE in FIXED_COMPOSITION_PROPOSALS, 'fixed_full composition values have not been chosen/approved: set FIXED_COMPOSITION_CHOICE (and APPROVED["fixed_composition"]) after reviewing the proposals')
    return FIXED_COMPOSITION_PROPOSALS[FIXED_COMPOSITION_CHOICE](target)

def key_counts(target, B, mode=BASE_MODE):
    # counts that are FIXED in every draw.  baseline: key crop(s) only; fixed_full: every crop of the composition (the natural remainder is then empty)
    if mode == BASE_MODE: return shares_to_counts(effective_key_shares(target), B, exact=True)
    if mode == FULL_MODE: return shares_to_counts(fixed_shares(target), B, exact=False)
    raise ValueError(mode)

def rest_count(target, B, mode=BASE_MODE): return B - sum(key_counts(target, B, mode).values())

def state_eligibility(target, B, cap, mode=BASE_MODE):
    # COMPLETE per-state table: what the (target, budget, mode) needs vs what each state holds, and the REASON for every exclusion.  No state is dropped silently.
    kc = key_counts(target, B, mode); rest = B - sum(kc.values()); rows = []
    for s in cap.index:
        row = dict(state=s, total=int(cap.loc[s, 'total']), need_total=B, rows_rule_min=MIN_STATE_ROWS); why = []
        if cap.loc[s, 'total'] < MIN_STATE_ROWS: why.append(f'<{MIN_STATE_ROWS} rows')
        if cap.loc[s, 'total'] < B: why.append(f'total {int(cap.loc[s, "total"])} < budget {B}')
        for c, n in kc.items():
            row[f'have_{c}'], row[f'need_{c}'] = int(cap.loc[s, c]), n
            if cap.loc[s, c] < n: why.append(f'{c} {int(cap.loc[s, c])}<{n}')
        nonkey = int(cap.loc[s, 'total'] - sum(cap.loc[s, c] for c in kc)); row['have_nonkey'], row['need_nonkey'] = nonkey, rest
        if nonkey < rest: why.append(f'non-key {nonkey}<{rest}')
        if target == 'AUS' and s not in AUS_STATES: why.append('not in AUS census {LA, FL}')
        row['eligible'] = not why; row['reason'] = '; '.join(why); rows.append(row)
    return pd.DataFrame(rows)

def eligible_states(target, B, cap, mode=BASE_MODE):
    # pool = states that can supply the WHOLE budget ALONE (rows rule + every fixed crop count + natural remainder).  Same pool for every k.
    e = state_eligibility(target, B, cap, mode); return sorted(e.loc[e['eligible'], 'state'].tolist())

def combo_capacity_ok(states, target, B, cap, mode=BASE_MODE):
    # actual capacity of a COMBINATION: each state must supply its equal share (ceil of count / k) of every fixed crop and of the remainder
    kc = key_counts(target, B, mode); rest = B - sum(kc.values()); k = len(states)
    for s in states:
        for c, n in kc.items():
            if cap.loc[s, c] < math.ceil(n / k): return False
        if cap.loc[s, 'total'] - sum(cap.loc[s, c] for c in kc) < math.ceil(rest / k): return False
    return True

# ------------------------------------------------------------------------------ EXTENSION: capacity-aware allocation (used only when ALLOC_RULE == 'capacity'; the main study never calls any of this)
import contextlib
@contextlib.contextmanager
def profile_scope(which):
    # temporarily put the notebook into the MAIN or the EXTENSION settings (used by the test gate so that tests of the main design never depend on which profile is selected)
    g = globals(); names = ['PROFILE', 'ALLOC_RULE', 'TARGETS_TO_RUN', 'BUDGETS', 'SELECTION_ARMS']; saved = {n: g[n] for n in names}
    try:
        if which == 'main': g.update(PROFILE='main', ALLOC_RULE='equal', **{k_: list(v) for k_, v in MAIN_DEFAULTS.items()})
        else: g.update(PROFILE='extension', ALLOC_RULE='capacity', TARGETS_TO_RUN=list(EXT_TARGETS), BUDGETS=list(EXT_BUDGETS), SELECTION_ARMS=['random'])
        yield
    finally: g.update(saved)

def _floor(n, k): return int(n) if k == 1 else math.ceil(EXT_MIN_CONTRIB * n / k)      # a one-state combination must supply everything itself

def water_fill(n, caps, rng):
    # split n rows over states as EQUALLY as the capacities allow: a state that cannot reach the common level gives its whole capacity, the surplus is shared by the others; the integer remainder goes to a random subset of the uncapped states
    caps = [int(c) for c in caps]; k = len(caps)
    if sum(caps) < n: raise InfeasibleSplit(f'capacity {sum(caps)} < {n} requested')
    alloc, free, left = [0] * k, list(range(k)), int(n)
    while free:
        share = left // len(free); capped = [i for i in free if caps[i] <= share]
        if not capped: break
        for i in capped: alloc[i] = caps[i]; left -= caps[i]
        free = [i for i in free if i not in capped]
    if free:
        base, rem = divmod(left, len(free)); extra = set(rng.choice(len(free), size=rem, replace=False).tolist()) if rem else set()
        for j, i in enumerate(free): alloc[i] = base + (1 if j in extra else 0)
    _check(sum(alloc) == n and all(0 <= a <= c for a, c in zip(alloc, caps)), 'water_fill produced an invalid allocation')
    return alloc

def _caps_for(states, target, cap, kc):
    # per fixed crop and for the natural remainder: how many rows each state of the combination holds
    out = {c: [int(cap.loc[s, c]) for s in states] for c in kc}; out[None] = [int(cap.loc[s, 'total'] - sum(cap.loc[s, c] for c in kc)) for s in states]; return out

def combo_feasible_pooled(states, target, B, cap, mode=BASE_MODE):
    # a combination is feasible when its states TOGETHER hold the quota of every fixed crop and of the remainder, and EACH state can give at least EXT_MIN_CONTRIB of its equal share
    kc = key_counts(target, B, mode); rest = B - sum(kc.values()); k = len(states); caps = _caps_for(states, target, cap, kc)
    for c, n in list(kc.items()) + [(None, rest)]:
        if n and (sum(caps[c]) < n or min(caps[c]) < _floor(n, k)): return False
    return True

def combo_ok(states, target, B, cap, mode=BASE_MODE):
    return combo_feasible_pooled(states, target, B, cap, mode) if ALLOC_RULE == 'capacity' else combo_capacity_ok(states, target, B, cap, mode)

def make_combos_pooled(pool, k, target, B, cap, mode=BASE_MODE, max_combos=None):
    # every FEASIBLE combination of the pool is counted (n_total); k=1 is the census of the pool; otherwise all of them if <= max_combos, else a fixed seeded sample
    max_combos = MAX_COMBOS if max_combos is None else max_combos; pool = sorted(pool)
    if k > len(pool): return [], 0
    if k == 1: return [(s,) for s in pool], len(pool)
    _check(math.comb(len(pool), k) <= 200000, f'pool of {len(pool)} states is too large to enumerate k={k}')
    allc = [c for c in itertools.combinations(pool, k) if combo_feasible_pooled(c, target, B, cap, mode)]; n_total = len(allc)
    if n_total <= max_combos: return allc, n_total
    idx = sorted(np.random.default_rng(stable_seed('combo', target, B, k, 'pooled')).choice(n_total, size=max_combos, replace=False).tolist()); return [allc[i] for i in idx], n_total

def alloc_max_share(states, target, B, cap, mode=BASE_MODE):
    # planned share of the budget supplied by the largest contributor (fixed crops + remainder, deterministic preview of water_fill)
    kc = key_counts(target, B, mode); rest = B - sum(kc.values()); caps = _caps_for(states, target, cap, kc); rng = np.random.default_rng(stable_seed('preview', target, B, '-'.join(states))); tot = np.zeros(len(states))
    for c, n in list(kc.items()) + [(None, rest)]:
        if n: tot += np.array(water_fill(n, caps[c], rng))
    return float(tot.max() / B)

def make_combos(pool, k, target, B, max_combos=None):
    max_combos = MAX_COMBOS if max_combos is None else max_combos
    pool = sorted(pool)
    if k > len(pool): return []
    if k == 1: return [(s,) for s in pool]                       # CENSUS: every pool state
    total = math.comb(len(pool), k)
    if total <= max_combos: return list(itertools.combinations(pool, k))
    rng = np.random.default_rng(stable_seed('combo', target, B, k))   # deterministic seed from (target, budget, k)
    seen = set()
    while len(seen) < max_combos:
        seen.add(tuple(sorted(rng.choice(len(pool), size=k, replace=False).tolist())))
    return [tuple(pool[i] for i in idx) for idx in sorted(seen)]
ARM_ORDER = ['random', 'diverse', 'clustered', 'common', 'kcenter', 'aug_on', 'aug_copy']
BASE_ARMS = ['random', 'diverse', 'clustered']            # chosen inside the plan grid; 'common' / 'kcenter' / 'aug_on' are added by add_variant_cells
STATE_SELECTORS = {'random': make_combos}              # the baseline arm; the two Phase-4 arms are built from the SAME pool by select_extreme_arms()

# ------------------------------------------------------------------------------ PHASE 4: phenology-spectral diversity of the USA source states (definition: see DIVERSITY_SPACE)
def compute_diversity_distances(X, state, crop, min_n=None):
    # X: (N, ...) USA Sentinel-2 features, state/crop: per-row labels aligned with X.  Features are standardised GLOBALLY (USA mean/std per feature); for every (state, crop) with >= min_n samples the mean
    # profile is taken; dist[crop][s, s'] = RMS difference of the two mean profiles (NaN if either has < min_n samples).  Uses USA source features only.
    min_n = MIN_PROFILE_N if min_n is None else int(min_n); A = np.asarray(X).reshape(len(X), -1); st, cr = np.asarray(state).astype(str), np.asarray(crop).astype(str)
    s1 = np.zeros(A.shape[1]); s2 = np.zeros(A.shape[1])
    for i in range(0, len(A), 20000): blk = A[i:i + 20000].astype(np.float64); s1 += blk.sum(0); s2 += (blk * blk).sum(0)
    gm = s1 / len(A); gs = np.sqrt(np.maximum(s2 / len(A) - gm ** 2, 0)) + 1e-6; states = sorted(set(st)); dist, cnt = {}, {}
    for c in sorted(set(cr)):
        mu, n_ = {}, {}
        for s_ in states:
            idx = np.flatnonzero((st == s_) & (cr == c)); n_[s_] = len(idx)
            if len(idx) >= min_n: mu[s_] = (A[idx].astype(np.float64).mean(0) - gm) / gs
        M = pd.DataFrame(np.nan, index=states, columns=states); ks = list(mu)
        if ks: P = np.vstack([mu[s_] for s_ in ks]); M.loc[ks, ks] = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).mean(-1))
        dist[c] = M; cnt[c] = pd.Series(n_)
    return dict(dist=dist, n=cnt, min_n=min_n)

def diversity_matrix(t, B, mode):
    # state-by-state distance for the target design = mean over its FIXED crops; also the set of states that have a profile for every one of them.  Returns (D or None, valid set, note)
    DD = globals().get('DIVERSITY_DIST')
    if DD is None: return None, set(), 'diversity distances not computed'
    crops = list(key_counts(t, B, mode))
    if not crops: return None, set(), 'no fixed crops in this design: diversity undefined'
    miss = [c for c in crops if c not in DD['dist']]
    if miss: return None, set(), f'no USA profiles for {miss}'
    D = sum(DD['dist'][c] for c in crops) / len(crops); valid = set(D.index[D.notna().any(axis=1)]) & set.intersection(*[set(DD['n'][c].index[DD['n'][c] >= DD['min_n']]) for c in crops])
    return D, valid, ''

def pool_for(t, B, k, cap, mode=BASE_MODE):
    # strict (default, unchanged): states that can supply the WHOLE budget alone.  equal_share (opt-in): states that can supply their 1/k share of every fixed crop and of the remainder.
    if ALLOC_RULE == 'capacity':                                                                                  # extension: a state can join a k-combination if it can give its floor share of every quota
        kc = key_counts(t, B, mode); rest = B - sum(kc.values()); out = []
        for s in cap.index:
            if cap.loc[s, 'total'] < MIN_STATE_ROWS or (t == 'AUS' and s not in AUS_STATES): continue
            caps = _caps_for((s,), t, cap, kc)
            if all((not n) or caps[c][0] >= _floor(n, k) for c, n in list(kc.items()) + [(None, rest)]): out.append(s)
        return sorted(out)
    if POOL_RULE == 'strict': return eligible_states(t, B, cap, mode)
    return sorted(s for s in cap.index if cap.loc[s, 'total'] >= MIN_STATE_ROWS and (t != 'AUS' or s in AUS_STATES) and combo_capacity_ok(tuple([s] * k), t, B, cap, mode))

def _score_idx(Dm, cidx):
    k = cidx.shape[1]; tot = np.zeros(len(cidx))
    for i_ in range(k):
        for j_ in range(i_ + 1, k): tot += Dm[cidx[:, i_], cidx[:, j_]]
    return tot / (k * (k - 1) / 2)

def diversity_population(pool, k, Dm, t, B, mode):
    # the scored combinations of one cell: ALL of them when C(P,k) <= ENUM_LIMIT, otherwise a fixed seeded random sample of SCORE_SAMPLE distinct ones
    P = len(pool); total = math.comb(P, k)
    if total <= ENUM_LIMIT: cidx = np.array(list(itertools.combinations(range(P), k))); enumerated = True
    else:
        rng = np.random.default_rng(stable_seed('divscore', t, B, k, mode)); seen = set()
        while len(seen) < SCORE_SAMPLE: seen.add(tuple(sorted(rng.choice(P, size=k, replace=False).tolist())))
        cidx = np.array(sorted(seen)); enumerated = False
    return cidx, _score_idx(Dm, cidx), enumerated

def select_extreme_arms(pool, k, Dm, t, B, mode):
    # diverse = the N_ARM highest-scoring combinations, clustered = the N_ARM lowest (ties broken by the combination itself -> deterministic); the same function also gives every combination its percentile
    cidx, sc, enumerated = diversity_population(pool, k, Dm, t, B, mode); rows = [(round(float(x), 12), tuple(int(v) for v in c)) for x, c in zip(sc, cidx)]
    hi = sorted(rows, key=lambda r: (-r[0], r[1]))[:N_ARM]; lo = sorted(rows, key=lambda r: (r[0], r[1]))[:N_ARM]; sc_sorted = np.sort(sc)
    mk = lambda lst: [tuple(pool[i] for i in c) for _, c in lst]
    return dict(diverse=mk(hi), clustered=mk(lo), scores=sc_sorted, enumerated=enumerated, n_scored=len(sc))

def combo_score_pct(combo, pool, Dm, sc_sorted):
    idx = np.array([[pool.index(s) for s in combo]]); v = float(_score_idx(Dm, idx)[0]); return v, float(np.searchsorted(sc_sorted, v + 1e-12, side='right') / len(sc_sorted))

# ------------------------------------------------------------------------------ v3.7 tracks: common pool, k-center sample selection, augmentation ON
def train_rows_multiplier(aug):
    # rows seen per epoch relative to the drawn training set: aug ON = original + AUG_REPEAT augmented copies; aug 'copy' (the v3.8 steps-matched control) = AUG_REPEAT exact copies, no transform
    return (1 + AUG_REPEAT) if aug in ('on', 'copy') else 1

def _cell_key(c): return (c['composition'], c['target'], c['budget'], c['k'], tuple(c['states']), c.get('sel', 'random'), c.get('aug', 'off'))

def add_variant_cells(cells, cap):
    # arms 'common' / 'kcenter' / 'aug_on' are added AFTER the base plan; the base cells are never altered except that a shared run also lists 'common' in its arms
    want = [a for a in ('common', 'kcenter', 'aug_on') if a in SELECTION_ARMS]
    if not want: return cells
    out = list(cells); index = {_cell_key(c): c for c in out}
    def nxt(mode, t, B, k): return 1 + max([c['combo_id'] for c in out if (c['composition'], c['target'], c['budget'], c['k']) == (mode, t, B, k)] or [-1])
    if 'common' in want:
        for mode in [BASE_MODE]:                                                   # the new tracks exist for the locked baseline composition only
            for t in [x for x in TARGETS_TO_RUN if x != 'AUS']:                    # AUS has no confirmed spec: no extra tracks
                run_B = sorted({c['budget'] for c in cells if c['target'] == t and c['composition'] == mode and 'random' in c['arms']})
                if len(run_B) < 2: continue
                new_by_B = {}
                for B in run_B:
                    for k in sorted({c['k'] for c in cells if c['target'] == t and c['composition'] == mode and c['budget'] == B}):
                        pk = sorted(set.intersection(*[set(pool_for(t, B2, k, cap, mode)) for B2 in run_B]))
                        if k > len(pk) or k > K_MAX[t]: continue
                        combos = [(s,) for s in pk] if k == 1 else make_combos(pk, k, t, B, MAX_COMBOS)
                        combos = [c for c in combos if combo_ok(c, t, B, cap, mode)]
                        if k >= 2 and len(combos) < MIN_COMBOS and t != 'AUS': continue
                        if combos: new_by_B.setdefault(B, {})[k] = (combos, len(pk))
                for B, byk in new_by_B.items():
                    if len(byk) < MIN_KPOINTS and t != 'AUS': continue                      # the k-point rule applies to this arm too
                    for k, (combos, P) in sorted(byk.items()):
                        for c in combos:
                            key = (mode, t, B, k, tuple(c), 'random', 'off')
                            if key in index: index[key]['arms'] = tuple(index[key]['arms']) + (('common',) if 'common' not in index[key]['arms'] else ()); index[key]['common_pool'] = P
                            else:
                                pend = any(x['pending'] for x in cells if x['target'] == t and x['composition'] == mode)
                                cell = dict(composition=mode, target=t, budget=B, k=k, combo_id=nxt(mode, t, B, k), states=tuple(c), pool_size=P, n_total_combos=math.comb(P, k), pending=pend, arms=('common',), div_score='', div_pct='', max_share='', common_pool=P)
                                out.append(cell); index[key] = cell
    def pick(base, n):
        # deterministic subset of the random-arm combinations: k=1 = the whole census, k>=2 = the n combinations with the smallest stable hash
        grp = {}
        for c in base: grp.setdefault((c['composition'], c['target'], c['budget'], c['k']), []).append(c)
        sel = []
        for (mode, t, B, k), lst in grp.items():
            lst = sorted(lst, key=lambda c: stable_seed('variant', t, B, k, '-'.join(c['states'])))
            sel += lst if k == 1 else lst[:n]
        return sel
    base = [c for c in cells if c['target'] != 'AUS' and c['composition'] == BASE_MODE and 'random' in c['arms'] and c.get('sel', 'random') == 'random' and c.get('aug', 'off') == 'off']
    if 'kcenter' in want:
        for c in pick([c for c in base if c['target'] in KC_TARGETS], KC_COMBOS):
            n = dict(c, arms=('kcenter',), sel='kcenter', div_score='', div_pct='', combo_id=c['combo_id']); out.append(n)
    if 'aug_on' in want:
        for c in pick([c for c in base if c['target'] in AUG_TARGETS and c['budget'] in AUG_BUDGETS], AUG_COMBOS):
            n = dict(c, arms=('aug_on',), aug='on', div_score='', div_pct='', pending=bool(c['pending'] or not APPROVED.get('aug_on', False))); out.append(n)
        if AUG_CONTROL:                                                            # v3.8: the steps-matched control, same combinations as aug_on, rows copied without transforms
            for c in pick([c for c in base if c['target'] in AUG_TARGETS and c['budget'] in AUG_BUDGETS], AUG_COMBOS):
                n = dict(c, arms=('aug_copy',), aug='copy', div_score='', div_pct='', pending=bool(c['pending'] or not APPROVED.get('aug_on', False))); out.append(n)
    return out

# ---- k-center sample selection (arm 'kcenter')
KC_EMBED = None
KC_WINDOW = ((121 - 1) // 5, (334 - 1) // 5 + 1)           # official DOY 121-334 at 5-day steps -> indices 24:67 (43 steps), same as the authors' CropGlobeDataset

def _np(a): return a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)

def build_kc_embedding(features, dims=None, fit_n=20000, seed=0, chunk=40000):
    # USA rows only: official-window features (10 bands x 43 steps) -> standardise -> PCA(dims) fitted on a seeded subsample.  Returns float32 (N, dims).
    dims = KC_DIMS if dims is None else dims; N = len(features); w0, w1 = KC_WINDOW
    def win(idx): return _np(features[idx])[:, 0, :, w0:w1].reshape(len(idx), -1).astype(np.float32)
    rng = np.random.default_rng(stable_seed('kc-fit', seed)); fit_idx = np.sort(rng.choice(N, size=min(fit_n, N), replace=False)); S = win(fit_idx)
    _check(np.isfinite(S).all(), 'k-center embedding: non-finite USA features')
    mu, sd = S.mean(0), S.std(0) + 1e-6; Z = (S - mu) / sd; U, sing, Vt = np.linalg.svd(Z, full_matrices=False); P = Vt[:dims].T.astype(np.float32)
    out = np.empty((N, dims), dtype=np.float32)
    for i in range(0, N, chunk): idx = np.arange(i, min(N, i + chunk)); out[idx] = ((win(idx) - mu) / sd) @ P
    return out

def kcenter_pick(pool, n, key, embed=None):
    # greedy farthest-first traversal INSIDE one (state, crop) pool; deterministic given the key (start point from the key); exactly n distinct rows of the pool
    embed = KC_EMBED if embed is None else embed; _check(embed is not None, 'k-center needs KC_EMBED (built in the plan cell from the USA features)')
    pool = np.asarray(pool); _check(0 <= n <= len(pool), f'k-center asked for {n} of {len(pool)} rows')
    if n == 0: return pool[:0]
    r = np.random.default_rng(stable_seed('kc', *key)); cand = pool if len(pool) <= KC_MAX_CAND else np.sort(r.choice(pool, size=KC_MAX_CAND, replace=False))
    if n > len(cand): cand = pool
    E = embed[cand].astype(np.float32); first = int(r.integers(len(cand))); sel = [first]; d = ((E - E[first]) ** 2).sum(1); d[first] = -1.0
    for _ in range(n - 1):
        j = int(d.argmax()); sel.append(j); d = np.minimum(d, ((E - E[j]) ** 2).sum(1)); d[sel] = -1.0
    out = cand[np.array(sel)]; _check(len(set(out.tolist())) == n, 'k-center produced duplicate rows'); return out

def build_plan(cap):
    cells, grid = [], []; arms_req = [a for a in BASE_ARMS if a in SELECTION_ARMS]; need_div = len(arms_req) > 1
    for mode in COMPOSITION_MODES_TO_RUN:
        mode_pending = (mode == FULL_MODE and not (APPROVED.get('fixed_composition') and FIXED_COMPOSITION_CHOICE in FIXED_COMPOSITION_PROPOSALS))
        if mode == FULL_MODE and FIXED_COMPOSITION_CHOICE not in FIXED_COMPOSITION_PROPOSALS: continue          # nothing to plan until a composition is chosen
        for t in TARGETS_TO_RUN:
            pending = (t == 'AUS' and not AUS_SPEC_CONFIRMED) or mode_pending
            for B in BUDGETS:
                kc = key_counts(t, B, mode); rest = B - sum(kc.values()); Dt, valid, div_note = diversity_matrix(t, B, mode) if need_div else (None, set(), '')
                rows = []
                for k in K_LIST:
                    pool = pool_for(t, B, k, cap, mode); dropped = [s for s in pool if Dt is not None and s not in valid]; pool = [s for s in pool if Dt is None or s in valid]; P = len(pool)
                    ntot = math.comb(P, k) if k <= P else 0; note, combos = '', []
                    if k > K_MAX[t]: note = f'k > K_MAX({K_MAX[t]})'
                    elif t == 'FRA' and k == 8 and P < FRA_K8_MIN_POOL: note = f'FRA k=8 needs pool >= {FRA_K8_MIN_POOL} (pool={P})'
                    elif k > P: note = 'pool smaller than k'
                    elif ALLOC_RULE == 'capacity':
                        combos, ntot = make_combos_pooled(pool, k, t, B, cap, mode)
                        if not combos: note = 'no combination of the pool can supply the quotas'
                    else: combos = STATE_SELECTORS[STATE_SELECTOR](pool, k, t, B)
                    if t != 'AUS' and combos and len(combos) < MIN_COMBOS and not (ALLOC_RULE == 'capacity' and k == 1): note, combos = f'{len(combos)} combinations < {MIN_COMBOS}', []      # extension: the k=1 CENSUS is kept however few states it has
                    arm_c = {'random': (list(combos), note)}; Dm = Dt.loc[pool, pool].values if (Dt is not None and P >= k >= 2) else None; ext = None
                    if Dm is not None: ext = select_extreme_arms(pool, k, Dm, t, B, mode) if ntot >= 2 * N_ARM else None
                    for arm in arms_req[1:]:
                        if combos and k == 1: arm_c[arm] = (list(combos), '')                                          # the one-state census is shared by every arm
                        elif not combos: arm_c[arm] = ([], note)
                        elif Dt is None: arm_c[arm] = ([], div_note or 'diversity undefined')
                        elif ntot < 2 * N_ARM: arm_c[arm] = ([], f'only {ntot} combinations: the {N_ARM} most and {N_ARM} least diverse sets cannot be disjoint')
                        else: arm_c[arm] = (list(ext[arm]), '')
                    rows.append(dict(k=k, pool=pool, P=P, ntot=ntot, arm_c=arm_c, Dm=Dm, ext=ext, dropped=dropped, note=note))
                for arm in arms_req:                                                                                   # the k-point rule is applied to every arm separately
                    if t != 'AUS' and sum(1 for r in rows if r['arm_c'][arm][0]) < MIN_KPOINTS:
                        for r in rows:
                            if r['arm_c'][arm][0]: r['arm_c'][arm] = ([], f'fewer than {MIN_KPOINTS} feasible k-points for this budget')
                for r in rows:
                    k, pool, P, ntot = r['k'], r['pool'], r['P'], r['ntot']; union = {}
                    for arm in arms_req:
                        for c in r['arm_c'][arm][0]: union.setdefault(tuple(c), []).append(arm)
                    for c in union: _check(combo_ok(c, t, B, cap, mode), f'{mode} {t} B={B} k={k}: combination {c} cannot supply its share (pool construction bug)')
                    arm_note = '; '.join(f'{a}: {r["arm_c"][a][1]}' for a in arms_req[1:] if r['arm_c'][a][1] and not r['arm_c'][a][0])
                    grid.append(dict(composition=mode, target=t, role=TARGET_ROLE[t], budget=B, k=k, pool_size=P, n_total_combos=ntot, n_combos=len(union), feasible=bool(union),
                                     note=r['arm_c']['random'][1] + (' [PENDING approval/decision]' if pending and union else ''), key_counts=json.dumps(kc), rest=rest, pool=','.join(pool) if P <= 20 else f'{P} states',
                                     **{f'n_{a}': len(r['arm_c'].get(a, ([], ''))[0]) for a in ARM_ORDER}, arm_note=arm_note, max_share=(max([alloc_max_share(c, t, B, cap, mode) for c in union] or [0.0]) if ALLOC_RULE == 'capacity' else ''), dropped_no_profile=','.join(r['dropped']), arms_enumerated=(r['ext']['enumerated'] if r['ext'] else '')))
                    for ci, (c, arms_of) in enumerate(union.items()):
                        sc, pct = (combo_score_pct(c, pool, r['Dm'], r['ext']['scores']) if (r['Dm'] is not None and r['ext'] is not None and k >= 2) else ('', ''))
                        cells.append(dict(composition=mode, target=t, budget=B, k=k, combo_id=ci, states=tuple(c), pool_size=P, n_total_combos=ntot, pending=pending, arms=tuple(arms_of), div_score=sc, div_pct=pct,
                                      max_share=(alloc_max_share(c, t, B, cap, mode) if ALLOC_RULE == 'capacity' else '')))
    cells = add_variant_cells(cells, cap)
    return cells, pd.DataFrame(grid)

def build_pools(usa_state, usa_crop):
    df = pd.DataFrame({'s': np.asarray(usa_state), 'c': np.asarray(usa_crop), 'i': np.arange(len(usa_state))})
    return {(s, c): g['i'].values.astype(np.int64) for (s, c), g in df.groupby(['s', 'c'])}

def make_cap(usa_state, usa_crop):
    cap = pd.crosstab(np.asarray(usa_state), np.asarray(usa_crop)).reindex(columns=CROP_ORDER, fill_value=0)
    cap['total'] = cap[CROP_ORDER].sum(axis=1); return cap

def allocate_val(counts, total, frac):
    # largest-remainder allocation of EXACTLY `total` validation rows across crops (stratified); every crop with >= 2 rows keeps >= 1 row in train
    ideal = {c: frac * n for c, n in counts.items()}
    v = {c: min(int(math.floor(ideal[c] + 1e-9)), max(n - 1, 0)) for c, n in counts.items()}
    for strict in (True, False):
        order = sorted(counts, key=lambda c: (-(ideal[c] - v[c]), CROP_ID[c]))
        while total - sum(v.values()) > 0:
            moved = False
            for c in order:
                if total - sum(v.values()) == 0: break
                if v[c] < counts[c] - 1 and (not strict or v[c] < math.ceil(ideal[c] - 1e-9)): v[c] += 1; moved = True
            if not moved: break
        if sum(v.values()) == total: return v
    raise InfeasibleSplit(f'cannot allocate {total} validation rows over {counts}')

def draw_split(target, B, states, seed, pools, usa_crop, mode=BASE_MODE, sel='random'):
    # EXACTLY B rows.  Fixed crop count(s) split equally over the states; any remainder = natural draw from each state's non-fixed rows (equal count per state).
    states = tuple(sorted(states)); k = len(states); kc = key_counts(target, B, mode); rest = B - sum(kc.values())
    rng = np.random.default_rng(stable_seed('draw', target, B, '-'.join(states), seed) if mode == BASE_MODE else stable_seed('draw', mode, FIXED_COMPOSITION_CHOICE, target, B, '-'.join(states), seed)); chosen = []
    def split_counts(n, caps=None):
        if ALLOC_RULE == 'capacity' and caps is not None: return water_fill(n, caps, rng)                       # extension only
        base, rem = divmod(n, k); extra = set(rng.choice(k, size=rem, replace=False).tolist()) if rem else set()
        return [base + (1 if i in extra else 0) for i in range(k)]
    for c in sorted(kc, key=lambda x: CROP_ID[x]):
        cnt = split_counts(kc[c], [len(pools.get((s, c), ())) for s in states])
        for i, s in enumerate(states):
            pool = pools.get((s, c), np.zeros(0, dtype=np.int64))
            if len(pool) < cnt[i]: raise InfeasibleSplit(f'{target} B={B}: state {s} has {len(pool)} {c} rows, needs {cnt[i]}')
            pick_ = rng.choice(pool, size=cnt[i], replace=False)
            chosen.append(pick_ if sel == 'random' else kcenter_pick(pool, cnt[i], (target, B, s, seed, c)))
    cnt = split_counts(rest, [sum(len(pools[(s, c)]) for c in CROP_ORDER if c not in kc and (s, c) in pools) for s in states])
    for i, s in enumerate(states):
        if cnt[i] == 0: continue
        parts = [pools[(s, c)] for c in CROP_ORDER if c not in kc and (s, c) in pools]; pool = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        if len(pool) < cnt[i]: raise InfeasibleSplit(f'{target} B={B}: state {s} has {len(pool)} non-key rows, needs {cnt[i]}')
        pick_ = rng.choice(pool, size=cnt[i], replace=False)
        if sel == 'random': chosen.append(pick_)
        else:                                                                                  # same crop mix as the random draw; only WHICH rows of each (state, crop) pool are taken changes
            cc_ = np.asarray(usa_crop)[pick_]
            for c2 in CROP_ORDER:
                n2 = int((cc_ == c2).sum())
                if n2: chosen.append(kcenter_pick(pools[(s, c2)], n2, (target, B, s, seed, 'rest', c2)))
    allp = np.concatenate(chosen); crops = np.asarray(usa_crop)[allp]
    nval = int(round(VAL_FRACTION * B)); _check(abs(VAL_FRACTION * B - nval) < 1e-9, 'validation size must be an integer')
    counts = {c: int((crops == c).sum()) for c in CROP_ORDER if (crops == c).any()}; vq = allocate_val(counts, nval, VAL_FRACTION); tr, va = [], []
    for c in counts:
        pos = allp[crops == c]; perm = np.random.default_rng(stable_seed('val', target, B, '-'.join(states), seed, c)).permutation(len(pos))
        va.append(pos[perm[:vq[c]]]); tr.append(pos[perm[vq[c]:]])
    return dict(train_pos=np.concatenate(tr), val_pos=np.concatenate(va), key_counts=kc, rest=rest, states=states, val_quota=vq, mode=mode, sel=sel)

def _alloc_checks(need, label, counts, caps, n):
    # extension: within capacity, at least the floor, and "as equal as possible" = a state below the others by more than 1 row must be at its capacity
    k = len(counts); fl = _floor(n, k)
    need(all(c <= cp for c, cp in zip(counts, caps)), f'{label}: a state supplies more than it holds {counts} vs {caps}')
    need(all(c >= fl for c in counts), f'{label}: a state supplies less than the minimum contribution {fl}: {counts}')
    need(all(not (counts[i] > counts[j] + 1 and counts[j] < caps[j]) for i in range(k) for j in range(k)), f'{label}: not as equal as capacity allows {counts} (capacities {caps})')

def verify_split(split, target, B, states, usa_state, usa_crop, mode=BASE_MODE):
    msgs = []; tr, va = split['train_pos'], split['val_pos']; allp = np.concatenate([tr, va]); kc = key_counts(target, B, mode); k = len(states)
    st, cr = np.asarray(usa_state), np.asarray(usa_crop)
    def need(c, m):
        if not c: msgs.append(m)
    nval = int(round(VAL_FRACTION * B))
    need(len(allp) == B, f'train+val={len(allp)} != budget {B}')
    need(len(va) == nval and len(tr) == B - nval, f'split sizes train={len(tr)} val={len(va)} != {B - nval}/{nval}')
    need(len(np.unique(allp)) == len(allp), 'duplicate rows inside the drawn sample (unique-id check)')
    need(len(np.intersect1d(tr, va)) == 0, 'train/validation overlap')
    need(allp.min() >= 0 and allp.max() < len(st), 'row id outside the USA table')
    need(set(st[allp].tolist()) <= set(states), 'a drawn row comes from a state outside the combination')
    for c, n in kc.items():
        need(int((cr[allp] == c).sum()) == n, f'fixed crop {c}: {int((cr[allp] == c).sum())} drawn vs fixed {n}')
        counts = [int(((cr[allp] == c) & (st[allp] == s)).sum()) for s in states]
        if ALLOC_RULE == 'capacity': _alloc_checks(need, f'fixed crop {c}', counts, [int(((cr == c) & (st == s)).sum()) for s in states], n)
        else: need(max(counts) - min(counts) <= 1, f'fixed crop {c}: states not equally represented {counts}')
    nonkey = ~np.isin(cr[allp], list(kc)); rest_counts = [int((nonkey & (st[allp] == s)).sum()) for s in states]
    need(int(nonkey.sum()) == B - sum(kc.values()), 'non-fixed count != natural remainder')
    if ALLOC_RULE == 'capacity': _alloc_checks(need, 'non-fixed rows', rest_counts, [int(((~np.isin(cr, list(kc))) & (st == s)).sum()) for s in states], B - sum(kc.values()))
    else: need(max(rest_counts) - min(rest_counts) <= 1, f'non-fixed rows not equally split over states {rest_counts}')
    for c in set(cr[allp].tolist()):
        n_c = int((cr[allp] == c).sum()); v_c = int((cr[va] == c).sum())
        need(abs(v_c - VAL_FRACTION * n_c) <= 1.0 + 1e-9, f'validation not stratified for {c}: {v_c} of {n_c}')
    return dict(ok=not msgs, msgs=msgs)

# ------------------------------------------------------------------------------ RESULTS TABLE (atomic; the metrics file is written LAST)
def make_run_id(cell, seed):
    mode = cell.get('composition', BASE_MODE); tag = '' if mode == BASE_MODE else f'|comp-{mode}-{FIXED_COMPOSITION_CHOICE}'
    if PROFILE != 'main': tag += f'|prof-{PROFILE}'
    if cell.get('sel', 'random') != 'random': tag += f"|sel-{cell['sel']}"
    return f"{cell['target']}|B{cell['budget']}|k{cell['k']}|{'-'.join(cell['states'])}|aug-{cell.get('aug', AUG)}{tag}|s{seed}"

COLUMNS = (['run_id', 'status', 'composition', 'arms', 'aug', 'sample_sel', 'input_features', 'main_f1', 'main_metric', 'train_n_effective', 'diversity_score', 'diversity_pct', 'profile', 'state_rows', 'max_state_share', 'class_weight_source', 'target', 'role', 'budget', 'k', 'combo_id', 'states', 'pool_size', 'n_total_combos', 'seed', 'n_train', 'n_val', 'epochs_run', 'best_epoch', 'stop_reason',
            'val_f1', 'restore_delta', 'primary_view', 'primary_f1', 'primary_oa', 'paper_f1', 'paper_oa', 'binary_f1', 'binary_oa',
            'baseline_all_other_primary', 'baseline_majority_primary', 'baseline_all_other_paper', 'baseline_majority_paper', 'cov_cond_primary_f1', 'cov_cond_paper_f1',
            'sevenclass_f1', 'sevenclass_oa', 'papertext_f1', 'papertext_oa', 'review_f1', 'review_oa', 'baseline_all_other_review', 'baseline_majority_review', 'baseline_all_other_papertext',
            'baseline_majority_papertext', 'cov_cond_review_f1', 'remap_rule_review',
            'absent_train_classes', 'n_target_classes', 'n_target_classes_train_ge30', 'n_absent_target_classes_train', 'steps_per_epoch', 'total_optimizer_steps',
            'scorer_maxdiff', 'test_n', 'key_n', 'target_crop_n']
           + [f'train_{c}' for c in CROP_ORDER] + [f'val_{c}' for c in CROP_ORDER] + [f'true_{c}' for c in CROP_ORDER] + [f'pred_{c}' for c in CROP_ORDER]
           + [f'f1_{c}' for c in CROP_ORDER] + [f'review_f1_{c}' for c in CROP_ORDER] + INTEGRITY_CHECKS + ['integrity_pass', 'skill_flags', 'seconds', 'error'])

def main_metric_value(target, primary_f1, cov_cond_primary_f1):
    # ARG / CHN: binary key-crop macro-F1 (= primary_f1).  FRA: coverage-conditional macro-F1 (rule FRA_MAIN_RULE, stated in advance because rice is never in the FRA training mix)
    if target == 'FRA' and FRA_MAIN_RULE == 'coverage_conditional': return cov_cond_primary_f1, 'coverage_conditional_macro_f1'
    return primary_f1, ('binary_key_crop_macro_f1' if target in BINARY_TARGETS else 'macro_f1_existing_release_remap')

def add_main_metric(d):
    # fills main_f1 / main_metric where they are missing (results written before v3.7); existing values are never changed
    d = d.copy()
    if 'main_f1' not in d.columns: d['main_f1'] = np.nan
    if 'main_metric' not in d.columns: d['main_metric'] = ''
    miss = pd.to_numeric(d['main_f1'], errors='coerce').isna() & d['primary_f1'].notna() if 'primary_f1' in d.columns else pd.Series(False, index=d.index)
    for i in d.index[miss]:
        v, name = main_metric_value(d.at[i, 'target'], pd.to_numeric(d.at[i, 'primary_f1'], errors='coerce'), pd.to_numeric(d.at[i, 'cov_cond_primary_f1'], errors='coerce')); d.at[i, 'main_f1'] = v; d.at[i, 'main_metric'] = name
    return d

def load_results(path):
    if not os.path.exists(path): return pd.DataFrame(columns=COLUMNS)
    return add_main_metric(pd.read_csv(path))

def append_row(path, row):
    df = load_results(path); df = pd.concat([df, pd.DataFrame([{c: row.get(c, '') for c in COLUMNS}])], ignore_index=True)
    atomic_write(path, lambda f: df.to_csv(f, index=False), mode='w')

def plan_for_seed(seed, cells=None):
    # the cells that exist for ONE seed: all of PLAN_CELLS_RUN, except that k-center cells run only for seeds in KC_SEEDS (v3.8 amendment V38-A1)
    cells = PLAN_CELLS_RUN if cells is None else cells
    return [c for c in cells if c.get('sel', 'random') != 'kcenter' or seed in KC_SEEDS]

def auto_gate_ok(seed, results_csv=None):
    # automatic stand-in for "read the gate report and approve": the seed is complete and has NO error / integrity_fail row.  Skill warnings never block (same policy as before).
    df = load_results(results_csv or os.path.join(OUT_DIR, 'results.csv'));
    if not len(df): return False, 'no results'
    d = df[pd.to_numeric(df['seed'], errors='coerce') == seed].drop_duplicates('run_id', keep='last')      # latest row per run: an error that was later re-run OK no longer blocks
    bad = int((~d['status'].isin(['ok'])).sum()); n_ok = int((d['status'] == 'ok').sum())
    if bad: return False, f'{bad} error / integrity_fail row(s) in seed {seed}'
    n_plan = len(plan_for_seed(seed))
    if n_ok != n_plan: return False, f'seed {seed} incomplete ({n_ok}/{n_plan} runs)'
    return True, f'seed {seed}: {n_ok}/{n_plan} runs, 0 errors, 0 integrity failures'

def plan_consistency_check(results_csv=None):
    # STOP-level guard against a plan that changed between seeds: every run id already on disk must be a run id of the CURRENT plan (otherwise earlier and later seeds would not be the same experiment).
    df = load_results(results_csv or os.path.join(OUT_DIR, 'results.csv'))
    expected = {make_run_id(c, s): (c, s) for s in SEEDS for c in plan_for_seed(s)}
    if not len(df): return dict(ok=True, existing=0, orphans=[], new=len(expected), msg='no results yet')
    have = set(df.loc[df['status'].isin(['ok', 'integrity_fail']), 'run_id'].astype(str)); orphans = sorted(have - set(expected))
    return dict(ok=not orphans, existing=len(have), orphans=orphans, new=len(set(expected) - have), msg=(f'{len(have)} existing run ids are all in the current plan; {len(set(expected) - have)} runs still to do' if not orphans else f'{len(orphans)} existing run ids are NOT in the current plan, e.g. {orphans[:3]}'))

CANARY_TOL = 1e-6
def canary_check(results_csv=None, out_dir=None, n=3):
    # re-run up to n FINISHED runs (aug off, aug on, k-center; seed 0 preferred) with the current code and compare with the stored scores.  Same device + any difference => the code or design changed between seeds: STOP.  Different device => report only (GPU arithmetic may differ).
    out_dir = out_dir or OUT_DIR; df = load_results(results_csv or os.path.join(out_dir, 'results.csv'))
    if not len(df): return dict(ok=True, rows=[], note='no finished runs yet')
    ok_rows = df[df['status'] == 'ok'].copy(); ok_rows['seed'] = pd.to_numeric(ok_rows['seed'], errors='coerce'); by_id = {make_run_id(c, s): (c, s) for s in SEEDS for c in plan_for_seed(s)}
    picks = []
    for pred in (lambda r: str(r['aug']) == 'off' and str(r['sample_sel']) in ('random', '', 'nan') and r['target'] == 'ARG', lambda r: str(r['aug']) == 'on', lambda r: str(r['sample_sel']) == 'kcenter'):
        for _, r in ok_rows.sort_values(['seed', 'run_id']).iterrows():
            if pred(r) and r['run_id'] in by_id and r['run_id'] not in [p['run_id'] for p in picks]: picks.append(r.to_dict()); break
    dev_now = None
    try:
        import torch; dev_now = 'cuda: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    except Exception: pass
    dev_then = None; p0 = os.path.join(out_dir, 'run_environment_seed0.json')
    if os.path.exists(p0):
        try: dev_then = json.load(open(p0)).get('device')
        except Exception: pass
    same_dev = (dev_now is None) or (dev_then is None) or (dev_now == dev_then); res = []
    for r in picks[:n]:
        cell, s = by_id[r['run_id']]; row, _p, _y, _sc = run_one(cell, int(s), r['run_id'], out_dir, save=False)
        d_f1 = abs(float(row['main_f1']) - float(r['main_f1'])); d_ep = abs(int(row['epochs_run']) - int(r['epochs_run']))
        res.append(dict(run_id=r['run_id'], stored=float(r['main_f1']), rerun=float(row['main_f1']), abs_diff=d_f1, epochs_stored=int(r['epochs_run']), epochs_rerun=int(row['epochs_run']), match=(d_f1 <= CANARY_TOL and d_ep == 0)))
    bad = [x for x in res if not x['match']]; ok = (not bad) or (not same_dev)
    return dict(ok=ok, rows=res, same_device=same_dev, device_now=dev_now, device_then=dev_then, mismatches=len(bad),
                note=('all canaries reproduce the stored scores exactly' if not bad else (f'{len(bad)} canary mismatch(es) on a DIFFERENT device ({dev_then} -> {dev_now}): reported, not a stop' if not same_dev else f'{len(bad)} canary mismatch(es) on the SAME device: the code or design changed since the stored runs')))

def replication_table(df, metric='main_f1', exploratory=(0, 1), confirmatory=(2, 3, 4)):
    # the pre-declared claim C1 (and C2) evaluated SEPARATELY on the exploratory and the confirmatory seeds; PASS = same sign in both groups and the 95% interval of the confirmatory group excludes 0.
    rows = []; d = df.copy(); d['seed'] = pd.to_numeric(d['seed'], errors='coerce')
    for label, seeds in (('exploratory', exploratory), ('confirmatory', confirmatory)):
        g = d[d['seed'].isin(seeds)]
        if not len(g): continue
        present = sorted(int(x) for x in g['seed'].dropna().unique()); mm = mixed_model_report(g, metric)
        for _, r in mm[mm['term'] == 'log2(k)'].iterrows():
            h = float(scipy_stats.t.ppf(0.975, max(int(r['groups']) - 3, 1))) * r['se']; rows.append(dict(claim='C1 gain per doubling of states (points)', target=r['target'], group=label, seeds=present, estimate=float(r['coef']), ci_lo=float(r['coef'] - h), ci_hi=float(r['coef'] + h)))
        gc = cell_stats(_random_arm_runs(g), metric)
        for _, r in gain_kmin_kmax(gc).iterrows(): rows.append(dict(claim=f"C2 gain k{int(r['k_min'])}->k{int(r['k_max'])} at budget {int(r['budget'])} (points)", target=r['target'], group=label, seeds=present, estimate=100 * float(r['gain']), ci_lo=100 * float(r['ci_lo']), ci_hi=100 * float(r['ci_hi'])))
    t = pd.DataFrame(rows)
    if not len(t): return t
    verdict = []
    for (cl, tg), g in t.groupby(['claim', 'target']):
        ex, co = g[g['group'] == 'exploratory'], g[g['group'] == 'confirmatory']
        if not len(co): v = 'confirmatory seeds not finished'
        elif not len(ex): v = 'no exploratory estimate'
        else:
            e, c = ex.iloc[0], co.iloc[0]; same = np.sign(e['estimate']) == np.sign(c['estimate']); excl = np.isfinite(c['ci_lo']) and (c['ci_lo'] > 0 or c['ci_hi'] < 0)
            v = 'PASS' if (same and excl) else ('NOT REPLICATED (sign differs)' if not same else 'NOT REPLICATED (confirmatory interval includes 0)')
        verdict.append(((cl, tg), v))
    vm = dict(verdict); t['verdict'] = [vm[(a, b)] for a, b in zip(t['claim'], t['target'])]
    return t

def run_seed_cell(seed, deadline=None):
    assert TESTS_PASSED and SMOKE_OK and APPROVED['plan'], 'test gate, smoke test and plan approval are required'
    assert seed in SEEDS, f'seed must be one of {SEEDS}'
    results_csv = os.path.join(OUT_DIR, 'results.csv'); log_path = os.path.join(OUT_DIR, 'run_log.txt')
    if seed > 0:
        assert APPROVED[f'seed{seed - 1}_gate_report'], f'seed {seed - 1} gate report has not been approved (APPROVED["seed{seed - 1}_gate_report"])'
        prev = load_results(results_csv); n = int(((pd.to_numeric(prev['seed'], errors='coerce') == seed - 1) & prev['status'].isin(['ok', 'integrity_fail'])).sum()) if len(prev) else 0
        assert n == len(plan_for_seed(seed - 1)), f'seed {seed - 1} is incomplete ({n}/{len(plan_for_seed(seed - 1))} runs) -- finish it first'
    check_namespace(write=True)
    os.makedirs(os.path.join(OUT_DIR, 'runs'), exist_ok=True); os.makedirs(os.path.join(OUT_DIR, 'gate_reports'), exist_ok=True)
    write_run_environment(OUT_DIR, seed)
    status = run_seed(seed, plan_for_seed(seed), results_csv, OUT_DIR, log_path, deadline=deadline)
    if status == 'paused': return 'paused'
    df_ = load_results(results_csv); report = gate_report(df_, seed, plan_for_seed(seed))
    atomic_write(os.path.join(OUT_DIR, 'gate_reports', f'seed{seed}.txt'), lambda f: f.write(report), mode='w'); print('\n' + report)
    return 'complete'

def run_guards():
    # the two seed-consistency guards (V38-A4).  True = safe to run.
    pc = plan_consistency_check(); print('  PLAN CHECK:', pc['msg'])
    if not pc['ok']: print('\n!! STOPPED: the plan changed since the earlier seeds were run (see above).  Do not continue: earlier and later seeds would not be the same experiment.'); return False
    cc = canary_check(); print('  CANARY CHECK:', cc['note'])
    for x in cc.get('rows', []): print(f"     {x['run_id']}: stored {x['stored']:.6f} | re-run {x['rerun']:.6f} | epochs {x['epochs_stored']} vs {x['epochs_rerun']} | {'OK' if x['match'] else 'DIFFERENT'}")
    if not cc['ok']: print('\n!! STOPPED: finished runs do not reproduce on the same device, so the code or design changed since they were run.  Report this; do not continue.'); return False
    return True

def run_all_seeds(max_hours=None):
    # THE WHOLE EXPERIMENT: every seed, every planned run (all arms, augmentation ON and OFF cells together).  Resumable: run it again after a disconnect / time limit.
    # Between seeds the gate report is approved AUTOMATICALLY only if the seed is complete with no error / integrity failure; otherwise it stops and says why.
    deadline = time.time() + 3600 * max_hours if max_hours else None; t0 = time.time()
    print(f'AUTO RUN: seeds {SEEDS}, planned runs per seed {[len(plan_for_seed(s_)) for s_ in SEEDS]} ({sum(1 for c in PLAN_CELLS_RUN if c.get("aug", "off") == "on")} augmentation ON + {sum(1 for c in PLAN_CELLS_RUN if c.get("aug", "off") == "copy")} steps-matched copy control per seed; k-center only for seeds {KC_SEEDS}), time limit {max_hours or "none"} h')
    if not run_guards(): return 'stopped'
    for s in SEEDS:
        if s > 0:
            ok, why = auto_gate_ok(s - 1)
            if not ok: print(f'\n!! AUTO RUN STOPPED before seed {s}: {why}.  Read the gate report / errors.log, fix the cause, then run this cell again (finished runs are skipped).'); return 'stopped'
            APPROVED[f'seed{s - 1}_gate_report'] = True; print(f'  auto-approved gate report of seed {s - 1} ({why})')
        status = run_seed_cell(s, deadline)
        if status == 'paused': print(f'\n>>> PAUSED at the time limit after {(time.time() - t0) / 3600:.1f} h: run this cell again to continue with seed {s}.'); return 'paused'
    ok, why = auto_gate_ok(SEEDS[-1])
    if not ok: print(f'\n!! LAST SEED NOT CLEAN: {why}'); return 'stopped'
    print(f'\n>>> ALL {len(SEEDS)} SEEDS COMPLETE ({(time.time() - t0) / 3600:.1f} h this session). Run the summary cell next.'); return 'complete'

def build_row(rid, cell, seed, split, sc, I, ok, skills, info, preds, y_true, seconds, usa_crop):
    tr, va = split['train_pos'], split['val_pos']; row = {c: '' for c in COLUMNS}; cr = np.asarray(usa_crop)
    row.update(run_id=rid, status='ok' if ok else 'integrity_fail', target=cell['target'], role=TARGET_ROLE[cell['target']], budget=cell['budget'], k=cell['k'], combo_id=cell['combo_id'],
               states='-'.join(cell['states']), pool_size=cell.get('pool_size', ''), n_total_combos=cell.get('n_total_combos', ''), seed=seed, n_train=len(tr), n_val=len(va),
               epochs_run=info.get('epochs_run', ''), best_epoch=info.get('best_epoch', ''), stop_reason=info.get('stop_reason', ''), val_f1=info['val_f1'], restore_delta=info.get('restore_delta', ''),
               primary_view=sc['primary_view'], primary_f1=sc['primary_f1'], primary_oa=sc['primary_oa'], paper_f1=sc['paper_f1'], paper_oa=sc['paper_oa'], binary_f1=sc['bin_f1'], binary_oa=sc['bin_oa'],
               baseline_all_other_primary=sc['baseline_all_other_primary'], baseline_majority_primary=sc['baseline_majority_primary'],
               baseline_all_other_paper=sc['baseline_all_other_paper'], baseline_majority_paper=sc['baseline_majority_paper'],
               cov_cond_primary_f1=(sc['cov_cond_binary_f1'] if cell['target'] in BINARY_TARGETS else sc['cov_cond_paper_f1']), cov_cond_paper_f1=sc['cov_cond_paper_f1'],
               absent_train_classes='+'.join(sc['absent_train_classes']), scorer_maxdiff=sc['scorer_maxdiff'], test_n=sc['n'], key_n=sum(split['key_counts'].values()),
               profile=PROFILE, state_rows='|'.join(f'{s_}:{int((np.asarray(USA_STATE)[np.concatenate([tr, va])] == s_).sum())}' for s_ in cell['states']), max_state_share=round(max(int((np.asarray(USA_STATE)[np.concatenate([tr, va])] == s_).sum()) for s_ in cell['states']) / len(np.concatenate([tr, va])), 4),
               main_f1=main_metric_value(cell['target'], sc['primary_f1'], (sc['cov_cond_binary_f1'] if cell['target'] in BINARY_TARGETS else sc['cov_cond_paper_f1']))[0], main_metric=main_metric_value(cell['target'], 0, 0)[1],
               aug=cell.get('aug', AUG), sample_sel=cell.get('sel', 'random'), input_features=INPUT_FEATURES, train_n_effective=len(tr) * train_rows_multiplier(cell.get('aug', AUG)),
               composition=cell.get('composition', BASE_MODE), arms='|'.join(cell.get('arms', ('random',))), diversity_score=cell.get('div_score', ''), diversity_pct=cell.get('div_pct', ''), class_weight_source=CLASS_WEIGHT_SOURCE,
               sevenclass_f1=sc['sevenclass_f1'], sevenclass_oa=sc['sevenclass_oa'], papertext_f1=sc['papertext_f1'], papertext_oa=sc['papertext_oa'], review_f1=sc['review_f1'], review_oa=sc['review_oa'],
               baseline_all_other_review=sc['baseline_all_other_review'], baseline_majority_review=sc['baseline_majority_review'], baseline_all_other_papertext=sc['baseline_all_other_papertext'],
               baseline_majority_papertext=sc['baseline_majority_papertext'], cov_cond_review_f1=sc['cov_cond_review_f1'], remap_rule_review=REVIEW_RULE,
               n_target_classes=len(DEFINED_CLASSES[cell['target']]),
               n_target_classes_train_ge30=int(sum(int((cr[tr] == c_).sum()) >= COVERAGE_MIN_TRAIN for c_ in DEFINED_CLASSES[cell['target']])),
               n_absent_target_classes_train=int(sum(int((cr[tr] == c_).sum()) == 0 for c_ in DEFINED_CLASSES[cell['target']])),
               steps_per_epoch=steps_per_epoch(len(tr) * train_rows_multiplier(cell.get('aug', AUG))), total_optimizer_steps=steps_per_epoch(len(tr) * train_rows_multiplier(cell.get('aug', AUG))) * int(info.get('epochs_run', 0) or 0),
               target_crop_n=int(sum(int((cr[np.concatenate([tr, va])] == c_).sum()) for c_ in TARGET_CROPS[cell['target']])),
               integrity_pass=ok, skill_flags='|'.join(skills), seconds=round(seconds, 2), error='')
    for c in CROP_ORDER:
        row[f'train_{c}'] = int((cr[tr] == c).sum()); row[f'val_{c}'] = int((cr[va] == c).sum())
        row[f'true_{c}'] = int((np.asarray(y_true) == CROP_ID[c]).sum()); row[f'pred_{c}'] = int((np.asarray(preds) == CROP_ID[c]).sum())
        row[f'f1_{c}'] = sc['paper_f1_cls'][CROP_ID[c]]; row[f'review_f1_{c}'] = sc['review_f1_cls'][CROP_ID[c]]
    row.update(I); return row

# ------------------------------------------------------------------------------ RUN ONE JOB + RUN LOOP
#   FIT_FN(Xtr, ytr, Xva, yva, seed, epochs_override) -> dict(model=..., info=dict(val_f1, best_epoch, epochs_run, stop_reason, restore_delta, history))
#   PREDICT_FN(model, X) -> integer class ids 0..6        (both injected by the torch cell)
FIT_FN, PREDICT_FN = None, None

def run_one(cell, seed, rid, out_dir, epochs_override=None, save=True):
    t0 = time.time(); target, B, states = cell['target'], cell['budget'], tuple(cell['states'])
    mode = cell.get('composition', BASE_MODE)
    split = draw_split(target, B, states, seed, POOLS, USA_CROP, mode, sel=cell.get('sel', 'random'))
    chk = verify_split(split, target, B, states, USA_STATE, USA_CROP, mode)
    if not chk['ok']: raise IntegrityFailure('SPLIT NOT EXACT: ' + '; '.join(chk['msgs']))
    tr, va = split['train_pos'], split['val_pos']
    out = FIT_FN(take(FEATURES['USA'], tr), USA_CROP_ID[tr], take(FEATURES['USA'], va), USA_CROP_ID[va], seed, epochs_override, **({'aug': cell['aug']} if cell.get('aug', AUG) in ('on', 'copy') else {}))
    preds = np.asarray(PREDICT_FN(out['model'], EVAL_X[target])).astype(int); y_true = np.asarray(EVAL_Y[target]).astype(int)
    train_counts = np.bincount(USA_CROP_ID[tr], minlength=7); val_counts = np.bincount(USA_CROP_ID[va], minlength=7)
    sc = score_run(target, y_true, preds, train_counts)
    I, ok = evaluate_integrity(sc, preds, chk, out['info']); skills = skill_flags(sc, out['info'])
    row = build_row(rid, cell, seed, split, sc, I, ok, skills, out['info'], preds, y_true, time.time() - t0, USA_CROP)
    if save:                                                                    # 1) per-run arrays (atomic)   2) the metrics row is appended by the caller LAST
        def w(f):
            np.savez_compressed(f, run_id=rid, y_true=y_true.astype(np.uint8), y_pred=preds.astype(np.uint8), y_pred_existing_release=sc['yp_remapped'].astype(np.uint8), y_pred_review_rule=remap_review_targetmap(preds, target).astype(np.uint8), y_pred_papertext_rule=remap_paper_text_v3(preds, target).astype(np.uint8),
                                target_sample_ids=np.asarray(TARGET_IDS[target]).astype(np.int64), usa_train_ids=USA_NPZ_ROW[tr].astype(np.int64), usa_val_ids=USA_NPZ_ROW[va].astype(np.int64),
                                train_counts=train_counts, val_counts=val_counts, cm7=sc['cm7'], cm_existing_release=sc['cm_paper'], cm_review_rule=sc['cm_review'], cm_papertext_rule=sc['cm_papertext'], review_f1_per_class=sc['review_f1_cls'],
                                cm_binary=(sc['cm_binary'] if sc['cm_binary'] is not None else np.zeros((0, 0), dtype=int)),
                                paper_precision=sc['paper_prec'], paper_recall=sc['paper_rec'], paper_f1_per_class=sc['paper_f1_cls'],
                                binary_precision=(sc['bin_prec'] if sc['bin_prec'] is not None else np.zeros(0)), binary_recall=(sc['bin_rec'] if sc['bin_rec'] is not None else np.zeros(0)),
                                binary_f1_per_class=(sc['bin_f1_cls'] if sc['bin_f1_cls'] is not None else np.zeros(0)), val_history=np.asarray(out['info'].get('history', []), dtype=np.float32))
        atomic_write(os.path.join(out_dir, 'runs', hashlib.md5(rid.encode()).hexdigest()[:16] + '.npz'), w)
    return row, preds, y_true, sc

def take(X, idx):
    idx = np.asarray(idx)
    if type(X).__module__.startswith('torch'):
        import torch
        return X[torch.as_tensor(idx, dtype=torch.long)]
    return X[idx]

def _log(msg, path=None):
    print(msg, flush=True)
    if path:
        with open(path, 'a') as f: f.write(msg + '\n')

def _pct(v): return '  n/a' if v is None or not np.isfinite(v) else f'{100 * v:5.1f}'

def seed_line(row, sc):
    sk = row['skill_flags']
    return (f"  [{row['target']} B{row['budget']} k{row['k']} {row['states']} s{row['seed']}] {row['primary_view']} mF1 {_pct(row['primary_f1'])} "
            f"(all-other {_pct(row['baseline_all_other_primary'])}, majority {_pct(row['baseline_majority_primary'])}) | EXISTING release-remap {_pct(row['paper_f1'])} | REVIEW {_pct(row['review_f1'])} (base {_pct(row['baseline_all_other_review'])}) | 7-class raw {_pct(row['sevenclass_f1'])} | cov-cond EXISTING {_pct(row['cov_cond_primary_f1'])} REVIEW {_pct(row['cov_cond_review_f1'])} "
            f"| steps {row['total_optimizer_steps']} | val {_pct(row['val_f1'])} best-ep {row['best_epoch']} ran {row['epochs_run']}/{MAX_EPOCHS} ({row['stop_reason']}) "
            f"| INTEGRITY {'PASS' if row['integrity_pass'] else 'FAIL'}" + (f' | {sk}' if sk else ''))

def run_seed(seed, cells, results_csv, out_dir, log_path=None, epochs_override=None, deadline=None):
    # one seed over ALL planned cells; resumable (finished runs are skipped); aborts only after MAX_CONSEC_INTEGRITY consecutive integrity failures/errors
    prev = load_results(results_csv); done = set(prev.loc[prev['status'].isin(['ok', 'integrity_fail']), 'run_id']) if len(prev) else set()
    todo = [c for c in cells if make_run_id(c, seed) not in done]
    _log(f'SEED {seed}: {len(cells)} planned runs | already done {len(cells) - len(todo)} | pending {len(todo)}', log_path)
    t_start, n_new, consec = time.time(), 0, 0
    for cell in todo:
        if deadline is not None and time.time() > deadline:
            _log(f'SEED {seed} PAUSED at the time limit ({n_new} new runs this session); run the same cell again to resume.', log_path); return 'paused'
        rid = make_run_id(cell, seed)
        try:
            row, preds, y_true, sc = run_one(cell, seed, rid, out_dir, epochs_override=epochs_override)
        except TooFewClasses: raise                                              # target data problem: stop the whole run
        except Exception as e:
            consec += 1; msg = f'{type(e).__name__}: {e}'[:400]; _log(f'  !! INTEGRITY/ERROR in {rid}: {msg}', log_path)
            with open(os.path.join(out_dir, 'errors.log'), 'a') as f: f.write(f'{rid}\n{traceback.format_exc()}\n')
            er = {c: '' for c in COLUMNS}; er.update(run_id=rid, status='integrity_fail' if isinstance(e, IntegrityFailure) else 'error', target=cell['target'], budget=cell['budget'], k=cell['k'],
                                                     combo_id=cell['combo_id'], states='-'.join(cell['states']), seed=seed, error=msg, integrity_pass=False)
            append_row(results_csv, er)
            if consec >= MAX_CONSEC_INTEGRITY: raise RuntimeError(f'{consec} consecutive integrity failures/errors -- aborted. Last: {msg}')
            continue
        append_row(results_csv, row); n_new += 1                                 # the metrics file is written LAST
        _log(seed_line(row, sc), log_path)
        consec = consec + 1 if row['status'] == 'integrity_fail' else 0
        if consec >= MAX_CONSEC_INTEGRITY: raise RuntimeError(f'{consec} consecutive integrity failures -- aborted (see the lines above).')
        if n_new % 10 == 0:
            el = time.time() - t_start; _log(f'  .. {n_new} new runs, {el / n_new:.1f}s/run, ~{(len(todo) - n_new) * el / n_new / 3600:.1f} h left for this seed', log_path)
    _log(f'SEED {seed} COMPLETE: {n_new} new runs this session', log_path)
    return 'complete'

# ------------------------------------------------------------------------------ STATISTICS (spec 5)
def t_interval_fpc(x, N, level=0.95):
    # t-interval over state COMBINATIONS with finite-population correction sqrt((N-n)/(N-1)); N = all C(pool,k) combinations
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]; n = len(x); m = float(x.mean()) if n else float('nan')
    if n < 2: return m, float('nan'), float('nan'), float('nan')
    fpc = (1.0 if not np.isfinite(N) else math.sqrt(max(N - n, 0) / (N - 1))) if N > 1 else 0.0           # N = inf: the combinations are NOT a random sample of a finite population -> no correction
    se = float(x.std(ddof=1)) / math.sqrt(n) * fpc; h = float(scipy_stats.t.ppf(0.5 + level / 2, n - 1)) * se
    return m, m - h, m + h, se

def combo_level(df, metric='primary_f1'):
    # unit = state combination; the seeds are averaged FIRST
    d = df[df['status'] == 'ok'].copy()
    for c_ in ('diversity_score', 'diversity_pct'):
        if c_ not in d.columns: d[c_] = np.nan             # results written before Phase 4 have no diversity columns
    for c in [metric, 'budget', 'k', 'seed', 'pool_size', 'n_total_combos', 'key_n', 'target_crop_n', 'n_target_classes_train_ge30', 'diversity_score', 'diversity_pct']: d[c] = pd.to_numeric(d[c], errors='coerce')
    g = d.groupby(['target', 'budget', 'k', 'states']).agg(f1=(metric, 'mean'), seed_sd=(metric, lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else float('nan')),
                                                         n_seeds=(metric, 'count'), pool_size=('pool_size', 'first'), n_total=('n_total_combos', 'first'), key_n=('key_n', 'first'), target_crop_n=('target_crop_n', 'mean'), n_cls_ge30=('n_target_classes_train_ge30', 'mean'), div_score=('diversity_score', 'first'), div_pct=('diversity_pct', 'first')).reset_index()
    return g

def expand_arms(df):
    # one copy of every run per arm it belongs to (a combination can be in several arms, e.g. the k=1 census is in all of them); column 'arm' added
    d = df.copy(); d['arms'] = d['arms'].fillna('random').replace('', 'random').astype(str); out = []
    for a in ARM_ORDER:
        m = d['arms'].str.split('|').apply(lambda x: a in x)
        if m.any(): out.append(d[m].assign(arm=a))
    return pd.concat(out, ignore_index=True) if out else d.assign(arm=[])

def arm_contrast(df, metric='primary_f1', a='diverse', b='random'):
    # per (target,budget,k>=2): mean difference (arm a - arm b) over combinations, Welch t-interval.  Combinations are NOT independent (they share states) -> descriptive, plus the jackknife of cell_stats.
    E = expand_arms(df); rows = []
    if not len(E): return pd.DataFrame()
    cl = pd.concat([combo_level(E[E['arm'] == x], metric).assign(arm=x) for x in (a, b) if (E['arm'] == x).any()], ignore_index=True)
    for (t, B, k), g in cl.groupby(['target', 'budget', 'k']):
        xa, xb = g[g.arm == a]['f1'].values, g[g.arm == b]['f1'].values
        if k == 1 or len(xa) < 2 or len(xb) < 2: continue
        d_ = float(xa.mean() - xb.mean()); va, vb = xa.var(ddof=1) / len(xa), xb.var(ddof=1) / len(xb); se = math.sqrt(va + vb)
        dfw = (va + vb) ** 2 / (va ** 2 / (len(xa) - 1) + vb ** 2 / (len(xb) - 1)) if se > 0 else float('nan'); h = float(scipy_stats.t.ppf(0.975, dfw)) * se if se > 0 else float('nan')
        rows.append(dict(target=t, budget=B, k=k, contrast=f'{a} - {b}', n_a=len(xa), n_b=len(xb), diff=d_, ci_lo=d_ - h, ci_hi=d_ + h, p_welch=float(scipy_stats.ttest_ind(xa, xb, equal_var=False).pvalue)))
    return pd.DataFrame(rows)

def arm_paired_contrast(df, metric, a, b='random'):
    # same target / budget / k / state combination in BOTH arms: difference of the seed-averaged scores, t-interval over combinations (k=1: over states).  Units overlap in states -> DESCRIPTIVE.
    E = expand_arms(df); rows = []
    if not len(E) or not (E['arm'] == a).any() or not (E['arm'] == b).any(): return pd.DataFrame()
    A = combo_level(E[E['arm'] == a], metric)[['target', 'budget', 'k', 'states', 'f1']].rename(columns={'f1': 'fa'}); Bn = combo_level(E[E['arm'] == b], metric)[['target', 'budget', 'k', 'states', 'f1']].rename(columns={'f1': 'fb'})
    M = A.merge(Bn, on=['target', 'budget', 'k', 'states'])
    for (t, B, k), g in M.groupby(['target', 'budget', 'k']):
        dlt = (g['fa'] - g['fb']).values
        if len(dlt) < 3: continue
        m = float(dlt.mean()); se = float(dlt.std(ddof=1)) / math.sqrt(len(dlt)); h = float(scipy_stats.t.ppf(0.975, len(dlt) - 1)) * se
        rows.append(dict(target=t, budget=int(B), k=int(k), contrast=f'{a} - {b} (paired)', n_pairs=len(dlt), diff=m, ci_lo=m - h, ci_hi=m + h, p_paired=float(scipy_stats.ttest_1samp(dlt, 0.0).pvalue) if se > 0 else float('nan'), unit='states' if k == 1 else 'combinations'))
    return pd.DataFrame(rows)

# ------------------------------------------------------------------------------ v3.8 generalisation statistics (instructor review): mixed model, average / worst-case gain, "beats the best single state"
def _random_arm_runs(df):
    # the random arm only (plain draws, aug OFF, random sample selection): the baseline the review's claims are about
    E = expand_arms(df); d = E[(E['arm'] == 'random') & (E['aug'].fillna('off').astype(str) == 'off') & (E['sample_sel'].fillna('random').replace('', 'random').astype(str) == 'random')].copy()
    for c in ['k', 'budget', 'seed']: d[c] = pd.to_numeric(d[c], errors='coerce')
    return d

def reml_random_intercept(y, X, groups):
    # y = X beta + u_g + e,  u_g ~ N(0, s2u),  e ~ N(0, s2e): REML estimate of lambda = s2u / s2e by a 1-D search, then GLS beta and its covariance.  No external dependency (statsmodels is not needed).
    y = np.asarray(y, float); X = np.asarray(X, float); g = pd.factorize(np.asarray(groups))[0]; G = g.max() + 1; N, p = X.shape
    n_g = np.bincount(g, minlength=G).astype(float); Xs = np.zeros((G, p)); ys = np.zeros(G); np.add.at(Xs, g, X); np.add.at(ys, g, y)
    XtX, Xty, yty = X.T @ X, X.T @ y, float(y @ y)
    def parts(lam):
        c = lam / (1.0 + n_g * lam)                                                  # V_g^-1 = I - c_g * 11'
        A = XtX - (Xs * c[:, None]).T @ Xs; bvec = Xty - (Xs * c[:, None]).T @ ys; q = yty - float((c * ys * ys).sum())
        return A, bvec, q
    def negll(loglam):
        lam = float(np.exp(loglam)); A, bvec, q = parts(lam)
        try: beta = np.linalg.solve(A, bvec)
        except np.linalg.LinAlgError: return 1e12
        rss = q - float(bvec @ beta); s2e = max(rss / (N - p), 1e-12); sign, logdetA = np.linalg.slogdet(A)
        return (N - p) * np.log(s2e) + float(np.log1p(n_g * lam).sum()) + logdetA
    grid = np.linspace(-12, 8, 81); vals = [negll(x) for x in grid]; i = int(np.argmin(vals)); lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = scipy_opt.minimize_scalar(negll, bounds=(lo, hi), method='bounded', options=dict(xatol=1e-8)); loglam = res.x if res.fun <= vals[i] else grid[i]
    lam = 0.0 if (loglam <= -11.9) else float(np.exp(loglam)); A, bvec, q = parts(lam); beta = np.linalg.solve(A, bvec); s2e = max((q - float(bvec @ beta)) / (N - p), 1e-12)
    cov = s2e * np.linalg.inv(A); return dict(beta=beta, cov=cov, s2e=s2e, s2u=lam * s2e, lam=lam, G=int(G), N=int(N), p=int(p))

def mixed_model_report(df, metric='main_f1'):
    # per target: F1 (points) ~ log2(k) + budget/1000 [+ seed dummies] with a random intercept per STATE COMBINATION (the same combination is scored at several budgets / seeds, and combinations share states).
    # Estimated with reml_random_intercept; p-values use a t distribution with (groups - parameters) degrees of freedom (conservative).  Never raises.
    d = _random_arm_runs(df); rows = []
    if not len(d): return pd.DataFrame()
    d['y'] = 100 * pd.to_numeric(d[metric], errors='coerce'); d['lk'] = np.log2(d['k']); d['B'] = d['budget'] / 1000.0; d['combo'] = d['target'].astype(str) + '|' + d['states'].astype(str)
    for t, g in d.dropna(subset=['y']).groupby('target'):
        if g['k'].nunique() < 2 or g['combo'].nunique() < 3: rows.append(dict(target=t, term='(not estimable)', coef=float('nan'), se=float('nan'), p=float('nan'), n=len(g), groups=g['combo'].nunique(), note='too few k values / combinations')); continue
        names, cols = ['intercept', 'log2(k)'], [np.ones(len(g)), g['lk'].values]
        if g['B'].nunique() > 1: names.append('budget/1000'); cols.append(g['B'].values)
        sds = sorted(g['seed'].dropna().unique())
        for s_ in sds[1:]: names.append(f'seed{int(s_)}'); cols.append((g['seed'].values == s_).astype(float))
        try:
            X = np.column_stack(cols); r = reml_random_intercept(g['y'].values, X, g['combo'].values); se = np.sqrt(np.diag(r['cov'])); dfree = max(r['G'] - r['p'], 1)
            for i_, nm in enumerate(names):
                if nm == 'intercept' or nm.startswith('seed'): continue
                tv = r['beta'][i_] / se[i_]; rows.append(dict(target=t, term=nm, coef=float(r['beta'][i_]), se=float(se[i_]), p=float(2 * scipy_stats.t.sf(abs(tv), dfree)), n=r['N'], groups=r['G'], note=''))
            rows.append(dict(target=t, term='between-combination SD', coef=float(np.sqrt(r['s2u'])), se=float('nan'), p=float('nan'), n=r['N'], groups=r['G'], note=f"residual SD {np.sqrt(r['s2e']):.2f}"))
        except Exception as e: rows.append(dict(target=t, term='(fit failed)', coef=float('nan'), se=float('nan'), p=float('nan'), n=len(g), groups=g['combo'].nunique(), note=f'{type(e).__name__}: {str(e)[:70]}'))
    return pd.DataFrame(rows)

def single_state_report(df, metric='main_f1', n_boot=2000, seed=0):
    # per (target, budget): the single-state census (k=1) versus every k>=2 combination of the random arm.  Scores are seed-averaged per state / combination.
    d = _random_arm_runs(df); out = []; rng = np.random.default_rng(seed)
    if not len(d): return pd.DataFrame()
    d['y'] = 100 * pd.to_numeric(d[metric], errors='coerce')
    for (t, B), g in d.dropna(subset=['y']).groupby(['target', 'budget']):
        s1 = g[g['k'] == 1].groupby('states')['y'].mean()
        if len(s1) < 2: continue
        for k, gk in sorted(g[g['k'] >= 2].groupby('k')):
            cm = gk.groupby('states')['y'].mean(); v = cm.values
            if len(v) < 2: continue
            boots = [float(rng.choice(v, size=len(v), replace=True).mean()) for _ in range(n_boot)]; mean_gain = float(v.mean() - s1.mean())
            out.append(dict(target=t, budget=int(B), k=int(k), n_single=len(s1), n_combos=len(v), single_mean=float(s1.mean()), single_best=float(s1.max()), single_best_state=str(s1.idxmax()), single_worst=float(s1.min()),
                            combo_mean=float(v.mean()), combo_worst=float(v.min()), combo_best=float(v.max()), avg_gain=mean_gain, avg_gain_ci_lo=float(np.percentile(boots, 2.5)) - float(s1.mean()), avg_gain_ci_hi=float(np.percentile(boots, 97.5)) - float(s1.mean()),
                            worst_case_gain=float(v.min() - s1.mean()), share_beating_best_single=float((v > s1.max()).mean()), worst_vs_best_single=float(v.min() - s1.max())))
    return pd.DataFrame(out)

def alignment_evidence(cache_dir, usa_state, usa_crop, raw_counts, log=print):
    # prints the state-alignment evidence (which ordering was chosen, how close the alternatives were, which crops are disputed) and checks that assigned rows add up to the raw class counts.  Evidence, not a stop: the data cell already stops on a weak or ambiguous alignment.
    p = os.path.join(cache_dir, 'alignment_report.json'); res = dict(report_found=os.path.exists(p), counts_match=None, disputed=[])
    if res['report_found']:
        r = json.load(open(p)); res['disputed'] = list(r.get('disputed_crops', []))
        log(f"  alignment: chosen ordering {r.get('chosen')} (lift {r.get('chosen_lift', float('nan')):.1f}); ambiguous with: {r.get('ambiguous_with') or 'none'}; disputed crops: {res['disputed'] or 'none'}")
        for n_, c_ in (r.get('candidates') or {}).items(): log(f"     candidate {n_:28s} lift {c_.get('lift_all_crops', float('nan')):7.1f}" + (f" | differs from chosen on {c_['differs_from_chosen_pct']:.1f}% of rows" if 'differs_from_chosen_pct' in c_ else ' | chosen'))
        if 'wheat' in res['disputed']: log("  CAVEAT: wheat's state assignment is disputed between orderings.  France's key crops include wheat (state-level France results depend on it); Argentina (soybeans) and China (rice) key crops are identical under every candidate ordering.")
    else: log(f'  alignment: {p} not found (older cache); no ordering evidence printed.')
    ass = pd.Series(np.asarray(usa_crop)).value_counts(); bad = {c: (int(ass.get(c, 0)), int(n)) for c, n in raw_counts.items() if int(ass.get(c, 0)) != int(n)}
    res['counts_match'] = not bad; log('  assigned rows per crop equal the raw USA class counts: ' + ('YES (every row has exactly one state)' if not bad else f'NO {bad}'))
    return res

def diversity_assoc(df, metric='primary_f1'):
    # all evaluated combinations (all arms pooled, k>=2): Spearman correlation of F1 with the diversity percentile inside each (target,budget,k) cell.  Exploratory; the percentile is relative to the SAME pool and k.
    cl = combo_level(df, metric); rows = []
    for (t, B, k), g in cl.groupby(['target', 'budget', 'k']):
        g = g.dropna(subset=['div_pct', 'f1'])
        if k == 1 or len(g) < 5 or g['div_pct'].nunique() < 3: continue
        r, p = scipy_stats.spearmanr(g['div_pct'], g['f1']); rows.append(dict(target=t, budget=B, k=k, n=len(g), spearman=float(r), p=float(p)))
    return pd.DataFrame(rows)

def jackknife_states(cl_group):
    # delete-one-STATE jackknife over the states that appear in the evaluated combinations (secondary generalisation interval)
    combos = [(set(s.split('-')), f) for s, f in zip(cl_group['states'], cl_group['f1'])]; states = sorted(set().union(*[c for c, _ in combos])); thetas = []
    for s in states:
        rest = [f for c, f in combos if s not in c]
        if rest: thetas.append(float(np.mean(rest)))
    m = len(thetas)
    if m < 2: return float('nan'), float('nan'), float('nan')
    tb = float(np.mean(thetas)); se = math.sqrt((m - 1) / m * sum((t - tb) ** 2 for t in thetas)); h = float(scipy_stats.t.ppf(0.975, m - 1)) * se
    return se, float(np.mean([f for _, f in combos])) - h, float(np.mean([f for _, f in combos])) + h

def cell_stats(df, metric='primary_f1'):
    cl = combo_level(df, metric); rows = []; arm = df['arm'].iloc[0] if ('arm' in df.columns and len(df) and df['arm'].nunique() == 1) else 'random'; selected = arm in ('diverse', 'clustered')
    for (t, B, k), g in cl.groupby(['target', 'budget', 'k']):
        x = g['f1'].values; N = int(g['n_total'].iloc[0]) if pd.notna(g['n_total'].iloc[0]) else len(x); census = (k == 1)
        if selected and not census: N = float('inf')          # Phase-4 arms are chosen by their score, not drawn at random
        m, lo, hi, se = t_interval_fpc(x, N) if not census else (float(np.mean(x)), float('nan'), float('nan'), float('nan'))
        jse, jlo, jhi = jackknife_states(g)
        sub = df[(df['status'] == 'ok') & (df['target'] == t) & (df['budget'] == B) & (df['k'] == k)]
        per_seed_means = sub.groupby('seed')[metric].mean().values if len(sub) else np.array([])
        exhaustive = bool((not census) and not selected and N <= MAX_COMBOS and len(x) >= N)      # every combination was enumerated -> the t/FPC interval collapses to a point BY CONSTRUCTION
        rows.append(dict(target=t, budget=int(B), k=int(k), n_combos=len(x), N_total_combos=N, n_seeds_min=int(g['n_seeds'].min()), mean=m, ci_lo=lo, ci_hi=hi, se=se, census=census, exhaustive=exhaustive,
                         ci_note=('census: no CI' if census else 'EXTENSION, few combinations (overlapping states): descriptive only' if (PROFILE == 'extension' and len(x) < MAX_COMBOS) else 'selected arm: interval describes variation among the selected combinations only (not a random sample of all combinations)' if selected else 'exhaustive: CI is degenerate by construction' if exhaustive else ''),
                         census_values=';'.join(f'{s}:{f:.4f}' for s, f in zip(g['states'], g['f1'])) if census else '',
                         jackknife_se=jse, jackknife_lo=jlo, jackknife_hi=jhi,
                         seed_sd_within_combo=float(np.nanmean(g['seed_sd'])) if g['seed_sd'].notna().any() else float('nan'),
                         seed_sd_of_cell_means=float(np.std(per_seed_means, ddof=1)) if len(per_seed_means) > 1 else float('nan')))
    return pd.DataFrame(rows).sort_values(['target', 'budget', 'k']).reset_index(drop=True) if rows else pd.DataFrame()

def gain_kmin_kmax(cs):
    # gain from the smallest to the largest feasible k with a CI (Welch-style; a k=1 census contributes zero sampling variance)
    out = []
    for (t, B), g in cs.groupby(['target', 'budget']):
        g = g.sort_values('k')
        if len(g) < 2: continue
        lo, hi = g.iloc[0], g.iloc[-1]; d = float(hi['mean'] - lo['mean'])
        se2 = [s ** 2 for s in (lo['se'], hi['se']) if np.isfinite(s)]; ns = [n for s, n in ((lo['se'], lo['n_combos']), (hi['se'], hi['n_combos'])) if np.isfinite(s)]
        if se2 and sum(se2) > 0:
            se = math.sqrt(sum(se2)); df_w = (sum(se2) ** 2) / sum((s2 ** 2) / (n - 1) for s2, n in zip(se2, ns)); h = float(scipy_stats.t.ppf(0.975, df_w)) * se
        else: se, h = float('nan'), float('nan')
        out.append(dict(target=t, budget=int(B), k_min=int(lo['k']), k_max=int(hi['k']), gain=d, ci_lo=d - h if np.isfinite(h) else float('nan'), ci_hi=d + h if np.isfinite(h) else float('nan')))
    return pd.DataFrame(out)

def regression_report(df, metric='primary_f1'):
    # OLS of combination-level F1 on k, n_target_classes_train_ge30 (target-defined classes with >= 30 TRAINING samples) and budget, per target.
    # The target-crop COUNT is NOT used: it equals fixed share x budget exactly (perfectly aliased with budget).  A regressor with zero variance is dropped and said so.
    cl = combo_level(df, metric); res = []; alias_note = ('target-crop count (= fixed share x budget) is aliased with budget and was replaced by n_target_classes_train_ge30 | EXPLORATORY: combinations overlap in states and are not '
                  'independent replicates, so these standard errors / p-values are descriptive, not confirmatory')
    for t, g in cl.groupby('target'):
        cols = {'intercept': np.ones(len(g)), 'k': g['k'].values.astype(float), 'n_target_classes_train_ge30': g['n_cls_ge30'].values.astype(float), 'budget': g['budget'].values.astype(float)}
        note = alias_note; names = list(cols)
        for nm in [n for n in names if n != 'intercept' and np.nanstd(cols[n]) == 0]: names.remove(nm); note += f' | {nm} has zero variance in this target and was dropped'
        X = np.column_stack([cols[n] for n in names]); y = g['f1'].values.astype(float)
        if len(y) <= X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
            res.append(dict(target=t, term='(not estimable)', coef=np.nan, se=np.nan, t=np.nan, p=np.nan, n=len(y), r2=np.nan, note=note + ' | design matrix rank-deficient or too few rows')); continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None); resid = y - X @ beta; dfree = len(y) - X.shape[1]; s2 = float(resid @ resid) / dfree
        cov = s2 * np.linalg.inv(X.T @ X); se = np.sqrt(np.diag(cov)); tv = beta / se; pv = 2 * scipy_stats.t.sf(np.abs(tv), dfree)
        r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum()) if ((y - y.mean()) ** 2).sum() > 0 else float('nan')
        for i_, nm in enumerate(names): res.append(dict(target=t, term=nm, coef=float(beta[i_]), se=float(se[i_]), t=float(tv[i_]), p=float(pv[i_]), n=len(y), r2=r2, note=note))
    return pd.DataFrame(res)

def seed_direction_warnings(df, metric='primary_f1'):
    # WARNING only, never a stop: do the seeds disagree on the SIGN of the k_min -> k_max change?
    w = []; d = df[df['status'] == 'ok'].copy()
    for c in [metric, 'budget', 'k', 'seed']: d[c] = pd.to_numeric(d[c], errors='coerce')
    for (t, B), g in d.groupby(['target', 'budget']):
        ks = sorted(g['k'].unique())
        if len(ks) < 2: continue
        signs = {}
        for s, gs in g.groupby('seed'):
            a, b = gs[gs['k'] == ks[0]][metric].mean(), gs[gs['k'] == ks[-1]][metric].mean()
            if np.isfinite(a) and np.isfinite(b): signs[int(s)] = float(np.sign(b - a))
        if len(set(signs.values())) > 1: w.append(f'WARNING seed-direction disagreement [{t} B{int(B)} k{int(ks[0])}->k{int(ks[-1])}]: per-seed sign {signs}')
    return w

def gate_report(df, seed, cells):
    # the per-seed gate report you approve before the next seed.  Both headline tracks are reported side by side; they are never pooled, averaged or ranked.
    d = df[pd.to_numeric(df['seed'], errors='coerce') == seed].copy(); L = []; planned = len(cells)
    L.append(f'================ GATE REPORT  seed {seed}  ({EXPERIMENT_VERSION}) ================')
    L.append(f'planned runs {planned} | recorded {len(d)} | status {d["status"].value_counts().to_dict()}')
    L.append('-- INTEGRITY (abort-level) --')
    for c_ in INTEGRITY_CHECKS:
        bad = int((d[c_].astype(str) != 'True').sum()) if len(d) else 0; L.append(f'   {c_:28s} failures: {bad}')
    rd = pd.to_numeric(d['restore_delta'], errors='coerce'); sd = pd.to_numeric(d['scorer_maxdiff'], errors='coerce')
    L.append(f'   max restore delta {rd.max():.2e} (tolerance {RESTORE_TOL:.0e}) | max scorer disagreement {sd.max():.2e} (tolerance {SCORER_TOL:.0e})')
    for c_ in ['best_epoch', 'epochs_run', 'steps_per_epoch', 'total_optimizer_steps', 'n_target_classes_train_ge30', 'n_absent_target_classes_train', 'val_f1', 'target_crop_n']: d[c_] = pd.to_numeric(d[c_], errors='coerce')
    # V33-01: rows written by the exception path (status error / integrity_fail raised before scoring) have BLANK metric columns.  They stay in the status and INTEGRITY counts above and are
    # listed here one by one; every training / track / coverage section below uses only rows that carry metrics, so a blank first row can no longer crash the report.
    d_all = d; has_m = pd.to_numeric(d_all['primary_f1'], errors='coerce').notna(); d = d_all[has_m].copy(); ok_ids = set(d_all.loc[d_all['status'] == 'ok', 'run_id'])
    L.append(f'-- RUNS WITHOUT METRICS (error / integrity_fail before scoring): {int((~has_m).sum())}  [excluded from the sections below, listed here] --')
    for _, r_ in d_all[~has_m].iterrows(): L.append(f'   {r_["run_id"]} [{r_["status"]}] {"SUPERSEDED by a later ok row (resume)" if r_["run_id"] in ok_ids else "UNRESOLVED"}: {str(r_["error"])[:160]}')
    L.append('-- TRAINING (<= %d total epochs, early stopping inside them) --' % MAX_EPOCHS); L.append(f'   stop reasons: {d["stop_reason"].value_counts().to_dict()}')
    L.append(f'   epochs run: min {d.epochs_run.min():.0f} median {d.epochs_run.median():.0f} max {d.epochs_run.max():.0f} | best_epoch >= {UNDERTRAINED_EPOCH} (possibly_undertrained): {int((d.best_epoch >= UNDERTRAINED_EPOCH).sum())}')
    for B, g in d.groupby(pd.to_numeric(d['budget'], errors='coerce')):
        L.append(f'   budget {int(B)}: steps/epoch {int(g.steps_per_epoch.iloc[0])} | total optimizer steps min {g.total_optimizer_steps.min():.0f} median {g.total_optimizer_steps.median():.0f} max {g.total_optimizer_steps.max():.0f}')
    L.append('-- SKILL outcomes (warnings, never aborts) --')
    _fc = collections.Counter(f.split(':')[0] for s_ in d['skill_flags'].fillna('').astype(str) for f in s_.split('|') if f)       # V33-02: explicit parse (no .str on a possibly empty/float Series)
    L.append(f'   flags: {dict(_fc.most_common())}')
    L.append('-- TRACKS, side by side (each its own definition; never pooled, averaged or ranked; binary is never compared with multiclass) --')
    for t in [x for x in TARGETS_ORDER if x in set(d['target'])]:
        g = d[d['target'] == t].copy()
        for c_ in ['primary_f1', 'baseline_all_other_primary', 'baseline_majority_primary', 'cov_cond_primary_f1', 'review_f1', 'baseline_all_other_review', 'baseline_majority_review', 'cov_cond_review_f1', 'paper_f1', 'papertext_f1', 'sevenclass_f1']: g[c_] = pd.to_numeric(g[c_], errors='coerce')
        L.append(f'   {t}:')
        L.append(f'      EXISTING  [{g.primary_view.iloc[0]}] main {100 * g.primary_f1.mean():.1f} | cov-cond {100 * g.cov_cond_primary_f1.mean():.1f} | all-other {100 * g.baseline_all_other_primary.mean():.1f} | majority {100 * g.baseline_majority_primary.mean():.1f} | '
                 f'runs not better than baseline {int((g.primary_f1 <= g[["baseline_all_other_primary", "baseline_majority_primary"]].max(axis=1)).sum())}/{len(g)}')
        L.append(f'      REVIEW    [target-map remap] main {100 * g.review_f1.mean():.1f} | cov-cond {100 * g.cov_cond_review_f1.mean():.1f} | all-other {100 * g.baseline_all_other_review.mean():.1f} | majority {100 * g.baseline_majority_review.mean():.1f} | '
                 f'runs not better than baseline {int((g.review_f1 <= g[["baseline_all_other_review", "baseline_majority_review"]].max(axis=1)).sum())}/{len(g)}')
        L.append(f'      also reported: existing release-remap {100 * g.paper_f1.mean():.1f} | paper_text_v3 {100 * g.papertext_f1.mean():.1f} | existing seven-class raw {100 * g.sevenclass_f1.mean():.1f} | val mF1 min {100 * g.val_f1.min():.1f}')
    L.append('-- CLASS COVERAGE by target and k (a k effect can be a coverage effect): share of runs with an absent target class | mean absent classes | mean classes with >=%d train samples --' % COVERAGE_MIN_TRAIN)
    for (t, k), g in d.groupby(['target', pd.to_numeric(d['k'], errors='coerce')]):
        L.append(f'   {t} k={int(k)}: runs {len(g)} | absent-class share {100 * (g.n_absent_target_classes_train > 0).mean():.0f}% | mean absent {g.n_absent_target_classes_train.mean():.2f} | mean classes>={COVERAGE_MIN_TRAIN}: {g.n_target_classes_train_ge30.mean():.2f} of {int(pd.to_numeric(g.n_target_classes).iloc[0])}')
    for metric, lab in TRACK_PAIR:
        cs = cell_stats(df[pd.to_numeric(df['seed'], errors='coerce') == seed], metric)
        if len(cs):
            L.append(f'-- {lab}: mean by (target, budget, k) for THIS seed  [E = exhaustive, CI degenerate by construction; C = k=1 census] --')
            for t, gt in cs.groupby('target'): L.append(f'   {t}: ' + ' | '.join(f'B{int(r.budget)}k{int(r.k)}:{100 * r["mean"]:.1f}{"C" if r.census else ("E" if r.exhaustive else "")}' for _, r in gt.iterrows()))
        L += [f'[{lab}] ' + w for w in seed_direction_warnings(df, metric)]
    return '\n'.join(L)

# V33-09: what actually ran -- software, device, data identity, settings, scoring maps.  One file per seed (never overwritten by another seed); the manifest and LOCKED are untouched.
def write_run_environment(out_dir, seed, info=None, geography=None):
    import platform
    env = dict(notebook_version=globals().get('NOTEBOOK_VERSION'), experiment=EXPERIMENT_VERSION, spec=SPEC_TAG, seed=int(seed), written=datetime.datetime.now().isoformat(timespec='seconds'),
               python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy_stats.__name__ and __import__('scipy').__version__, cropnet_commit=globals().get('GIT_COMMIT'))
    try:
        import sklearn; env['sklearn'] = sklearn.__version__
        import torch; env.update(torch=torch.__version__, cuda_available=bool(torch.cuda.is_available()), device='cuda: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
    except Exception as e: env['torch_or_sklearn'] = f'not importable here: {type(e).__name__}'
    env['settings'] = dict(profile=PROFILE, alloc_rule=ALLOC_RULE, class_weight_source=CLASS_WEIGHT_SOURCE, max_epochs=MAX_EPOCHS, patience=PATIENCE, min_epochs=MIN_EPOCHS, batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
                           val_fraction=VAL_FRACTION, budgets=BUDGETS, k_list=K_LIST, seeds=SEEDS, composition_modes=COMPOSITION_MODES_TO_RUN, state_selector=STATE_SELECTOR,
                           geography_max_outside_rows=GEOGRAPHY_MAX_OUTSIDE_ROWS, alignment_override=globals().get('ALIGNMENT_ORDERING_OVERRIDE'))
    env['label_map'] = {int(i): CROP_LABELS[i] for i in CROP_LABELS}; env['target_label_sets_table2'] = DEFINED_CLASSES; env['remap_rules'] = remap_difference_table().to_dict('records')
    info = info if info is not None else globals().get('INFO')
    if info: env['data_files'] = {c: dict(path=v.get('path'), size_bytes=(os.path.getsize(v['path']) if v.get('path') and os.path.exists(v['path']) else None), shape=list(v.get('shape', [])),
                                          raw_rows=v.get('raw'), kept_rows=v.get('kept'), duplicates_removed=v.get('dup'), raw_class_counts=v.get('raw_by_class')) for c, v in info.items()}
    env['geography_observed'] = geography if geography is not None else globals().get('GEOGRAPHY_OBSERVED')
    path = os.path.join(out_dir, f'run_environment_seed{int(seed)}.json')
    if os.path.exists(path): path = os.path.join(out_dir, f"run_environment_seed{int(seed)}__resume_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.json")     # a resumed seed keeps its ORIGINAL record (device, library versions) and gets a new one beside it
    atomic_write(path, lambda f: f.write(json.dumps(env, indent=1, default=str)), mode='w'); return env

# ------------------------------------------------------------------------------ RESULTS-FILE RECONCILIATION (e.g. the 5,004-row vs 5,590-row files of the review)
def reconcile_results(a, b, key='run_id', value_cols=('primary_f1', 'paper_f1', 'binary_f1', 'val_f1'), tol=1e-9, names=('A', 'B')):
    # generic, read-only comparison of two results tables: duplicates, rows only in one, rows in both with conflicting metrics, column differences
    rep = {}; na, nb = names
    rep['rows'] = {na: len(a), nb: len(b)}; rep['columns_only_in_' + na] = sorted(set(a.columns) - set(b.columns)); rep['columns_only_in_' + nb] = sorted(set(b.columns) - set(a.columns))
    for nm, d in ((na, a), (nb, b)): rep[f'duplicate_{key}_in_{nm}'] = int(d[key].duplicated().sum()) if key in d else 'key missing'
    if key not in a or key not in b: rep['error'] = f'"{key}" must exist in both files'; return rep
    ka, kb = set(a[key]), set(b[key]); rep[f'only_in_{na}'] = len(ka - kb); rep[f'only_in_{nb}'] = len(kb - ka); rep['in_both'] = len(ka & kb)
    m = a.drop_duplicates(key).merge(b.drop_duplicates(key), on=key, suffixes=('_A', '_B')); conflicts = {}
    for col in value_cols:
        if f'{col}_A' in m and f'{col}_B' in m:
            x, y = pd.to_numeric(m[f'{col}_A'], errors='coerce'), pd.to_numeric(m[f'{col}_B'], errors='coerce'); bad = ~(np.isclose(x, y, atol=tol, equal_nan=True))
            if bad.any(): conflicts[col] = m.loc[bad, key].tolist()[:20]
    rep['conflicting_rows_by_metric'] = {k: len(v) for k, v in conflicts.items()}; rep['conflict_examples'] = conflicts
    rep['verdict'] = 'IDENTICAL on shared keys' if not conflicts and not rep[f'only_in_{na}'] and not rep[f'only_in_{nb}'] else 'DIFFERENT -- inspect before merging; do not pool silently'
    return rep
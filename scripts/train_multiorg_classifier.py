"""Phase 3+4 — train multi-organism essentiality classifier, test on Syn3A.

Loads outputs/multiorg_per_gene_features.csv (built by build_multiorg_features.py),
trains on 6 bacterial organisms (Caulobacter, S.aureus, M.tb, S.Typhimurium,
A.baylyi, M.pneumoniae), tests on Syn3A held out.

Three classifier baselines:
  1. Class-weighted logistic regression
  2. Class-weighted gradient boosting
  3. Same logreg trained ONLY on the closest organism (M. pneumoniae)
     — for comparison against v24's mpne->syn3a baseline (MCC=0.20)

Outputs: outputs/multiorg_classifier_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
INP = ROOT / 'outputs/multiorg_per_gene_features.csv'
OUT = ROOT / 'outputs/multiorg_classifier_results.json'

NUMERIC_FEATURES = [
    'log_length_bp', 'gc', 'upstream_gc', 'upstream_at_skew',
    'position_norm', 'operon_run_length',
    'is_hypothetical', 'is_uncharacterized', 'is_putative',
    'kw_replication', 'kw_transcription', 'kw_translation',
    'kw_atp', 'kw_membrane', 'kw_synthase', 'kw_kinase',
    'kw_dehydrogenase', 'kw_protease', 'kw_rrna', 'kw_chaperone',
]

TRAIN_ORGS = ['ccrescentus', 'saureus', 'mtuberculosis',
              'styphimurium', 'abaylyi', 'mpne']
TEST_ORG = 'syn3a'
SEED = 42


def threshold_search(y_true, probs):
    best_t, best_mcc = 0.5, -2.0
    for t in np.linspace(0.05, 0.95, 19):
        p = (probs >= t).astype(int)
        if p.sum() in (0, len(p)): continue
        m = matthews_corrcoef(y_true, p)
        if m > best_mcc: best_mcc, best_t = m, t
    return best_t, best_mcc


def evaluate(model, X_train, y_train, X_test, y_test, label):
    model.fit(X_train, y_train)
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    best_t, train_mcc = threshold_search(y_train, train_probs)
    test_pred = (test_probs >= best_t).astype(int)
    if pd.Series(test_pred).nunique() < 2 or pd.Series(y_test).nunique() < 2:
        test_mcc = 0.0
        tp, fp, tn, fn = 0, 0, 0, 0
    else:
        test_mcc = matthews_corrcoef(y_test, test_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
    print(f'  [{label}] train MCC@best_t={train_mcc:.4f}  '
          f'test MCC={test_mcc:.4f}  '
          f'(t={best_t:.2f}, TP={tp} FP={fp} TN={tn} FN={fn})')
    return {'label': label, 'best_threshold': float(best_t),
            'train_mcc': float(train_mcc), 'test_mcc': float(test_mcc),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}


def main():
    print('[1] loading multiorg features...')
    df = pd.read_csv(INP)
    print(f'  total rows: {len(df)}')
    print(f'  per-org: {df.organism.value_counts().to_dict()}')

    train = df[df.organism.isin(TRAIN_ORGS)].copy()
    test = df[df.organism == TEST_ORG].copy()
    print(f'\n  train: {len(train)} rows ({train.essential.sum()} ess, '
          f'{(train.essential==0).sum()} ne) across {train.organism.nunique()} orgs')
    print(f'  test ({TEST_ORG}): {len(test)} rows ({test.essential.sum()} ess, '
          f'{(test.essential==0).sum()} ne)')

    Xtr = train[NUMERIC_FEATURES].fillna(0.0).values.astype(np.float32)
    ytr = train.essential.astype(int).values
    Xte = test[NUMERIC_FEATURES].fillna(0.0).values.astype(np.float32)
    yte = test.essential.astype(int).values

    print('\n[2] training classifiers...')
    results = {}

    # 1. Class-weighted logistic regression
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', C=1.0, max_iter=500,
                            solver='liblinear', random_state=SEED),
    )
    results['logreg_6orgs_to_syn3a'] = evaluate(
        logreg, Xtr, ytr, Xte, yte, 'LogReg 6-org -> syn3a')

    # 2. Class-weighted gradient boosting
    sw = np.where(ytr == 1, (ytr == 0).sum() / len(ytr),
                  (ytr == 1).sum() / len(ytr))
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.1, random_state=SEED)
    # Wrap fit to pass sample_weight
    class _GBWrap:
        def __init__(self, m, sw): self.m = m; self.sw = sw
        def fit(self, X, y): self.m.fit(X, y, sample_weight=self.sw); return self
        def predict_proba(self, X): return self.m.predict_proba(X)
    gb_wrap = _GBWrap(gb, sw)
    results['gb_6orgs_to_syn3a'] = evaluate(
        gb_wrap, Xtr, ytr, Xte, yte, 'GB 6-org -> syn3a')

    # 3. mpne-only -> syn3a (v24 baseline reproduction)
    train_mpne = df[df.organism == 'mpne']
    Xtr_m = train_mpne[NUMERIC_FEATURES].fillna(0.0).values.astype(np.float32)
    ytr_m = train_mpne.essential.astype(int).values
    logreg_mpne = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', C=1.0, max_iter=500,
                            solver='liblinear', random_state=SEED),
    )
    results['logreg_mpne_only_to_syn3a'] = evaluate(
        logreg_mpne, Xtr_m, ytr_m, Xte, yte, 'LogReg mpne-only -> syn3a')

    # 4. Leave-one-organism-out: how does training on N-1 orgs compare to ALL?
    # Specifically: leave out mpne, train on 5 distant orgs, test on syn3a.
    # This isolates whether the closest-relative (mpne) is the one carrying
    # the signal vs whether broader bacterial training helps.
    train_no_mpne = df[df.organism.isin(['ccrescentus', 'saureus',
                                          'mtuberculosis', 'styphimurium',
                                          'abaylyi'])]
    Xtr_nm = train_no_mpne[NUMERIC_FEATURES].fillna(0.0).values.astype(np.float32)
    ytr_nm = train_no_mpne.essential.astype(int).values
    logreg_nm = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', C=1.0, max_iter=500,
                            solver='liblinear', random_state=SEED),
    )
    results['logreg_5orgs_no_mpne_to_syn3a'] = evaluate(
        logreg_nm, Xtr_nm, ytr_nm, Xte, yte, 'LogReg 5-org (no mpne) -> syn3a')

    # Scoreboard
    print('\n[3] scoreboard:')
    for k, v in results.items():
        print(f'  {k:40s}  test MCC = {v["test_mcc"]:.4f}')
    print(f'  {"v24 mpne->syn3a annotation (reference)":40s}  test MCC = 0.2034')
    print(f'  {"v18 ESM-2 cross-organism (reference)":40s}  test MCC = 0.1376')
    print(f'  {"v15 within-organism (reference)":40s}  test MCC = 0.5372')

    out = {
        'method': 'multiorg_classifier_train_6orgs_test_syn3a',
        'features': NUMERIC_FEATURES,
        'train_organisms': TRAIN_ORGS,
        'test_organism': TEST_ORG,
        'train_size': int(len(train)),
        'test_size': int(len(test)),
        'train_class_balance': {'essential': int(train.essential.sum()),
                                  'nonessential': int((train.essential==0).sum())},
        'test_class_balance': {'essential': int(test.essential.sum()),
                                'nonessential': int((test.essential==0).sum())},
        'results': results,
        'baselines': {
            'v15_within_organism': 0.5372,
            'v18_xgboost_esm2_cross_organism': 0.1376,
            'v24_mpne_to_syn3a_annotation_only': 0.2034,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()

"""Phase 7 training — class-weighted GB on v25 features + 5 organism-specific
expression-rate features (CAI, RBS_strength, CAI_rank, RBS_rank, expression_rank).

Same train/test split as v25/v27: 6 organisms train, Syn3A held out.

Output: outputs/multiorg_classifier_with_expression_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef

ROOT = Path(__file__).resolve().parent.parent
INP = ROOT / 'outputs/multiorg_per_gene_features_with_expression.csv'
OUT = ROOT / 'outputs/multiorg_classifier_with_expression_results.json'

ORIGINAL_FEATURES = [
    'log_length_bp', 'gc', 'upstream_gc', 'upstream_at_skew',
    'position_norm', 'operon_run_length',
    'is_hypothetical', 'is_uncharacterized', 'is_putative',
    'kw_replication', 'kw_transcription', 'kw_translation',
    'kw_atp', 'kw_membrane', 'kw_synthase', 'kw_kinase',
    'kw_dehydrogenase', 'kw_protease', 'kw_rrna', 'kw_chaperone',
]
EXPRESSION_FEATURES = ['cai', 'rbs_strength', 'cai_rank', 'rbs_rank', 'expression_rank']
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


def evaluate(model, X_train, y_train, X_test, y_test, label, sample_weight=None):
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    train_probs = model.predict_proba(X_train)[:, 1]
    best_t, train_mcc = threshold_search(y_train, train_probs)
    probs = model.predict_proba(X_test)[:, 1]
    pred = (probs >= best_t).astype(int)
    if pd.Series(pred).nunique() < 2 or pd.Series(y_test).nunique() < 2:
        test_mcc = 0.0; tp, fp, tn, fn = 0, 0, 0, 0
    else:
        test_mcc = matthews_corrcoef(y_test, pred)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f'  [{label}]  best_t={best_t:.2f}  train MCC={train_mcc:.4f}  '
          f'test MCC={test_mcc:.4f}  (TP={tp} FP={fp} TN={tn} FN={fn})')
    return {'label': label, 'best_threshold': float(best_t),
            'train_mcc': float(train_mcc), 'test_mcc': float(test_mcc),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}


def main():
    print('[1] loading enriched matrix (v25 + expression features)...')
    df = pd.read_csv(INP)
    print(f'  rows: {len(df)}; total cols: {len(df.columns)}')

    train = df[df.organism.isin(TRAIN_ORGS)].copy()
    test = df[df.organism == TEST_ORG].copy()
    print(f'  train: {len(train)} rows  test: {len(test)} rows')

    Xtr_orig = train[ORIGINAL_FEATURES].fillna(0.0).values.astype(np.float32)
    Xtr_expr = train[EXPRESSION_FEATURES].fillna(0.0).values.astype(np.float32)
    Xtr_all = train[ORIGINAL_FEATURES + EXPRESSION_FEATURES].fillna(0.0).values.astype(np.float32)
    ytr = train.essential.astype(int).values

    Xte_orig = test[ORIGINAL_FEATURES].fillna(0.0).values.astype(np.float32)
    Xte_expr = test[EXPRESSION_FEATURES].fillna(0.0).values.astype(np.float32)
    Xte_all = test[ORIGINAL_FEATURES + EXPRESSION_FEATURES].fillna(0.0).values.astype(np.float32)
    yte = test.essential.astype(int).values

    sw = np.where(ytr == 1, (ytr == 0).sum() / len(ytr),
                   (ytr == 1).sum() / len(ytr))

    print('\n[2] training classifiers...')
    results = {}

    gb1 = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.1, random_state=SEED)
    results['gb_original'] = evaluate(gb1, Xtr_orig, ytr, Xte_orig, yte,
                                       'GB original 20 (v25 reproduction)',
                                       sample_weight=sw)

    gb2 = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.1, random_state=SEED)
    results['gb_expression_only'] = evaluate(gb2, Xtr_expr, ytr, Xte_expr, yte,
                                              'GB expression features only (5)',
                                              sample_weight=sw)

    gb3 = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.1, random_state=SEED)
    results['gb_all_features'] = evaluate(gb3, Xtr_all, ytr, Xte_all, yte,
                                           'GB original 20 + expression 5',
                                           sample_weight=sw)

    # Feature importance
    fi = sorted(
        zip(ORIGINAL_FEATURES + EXPRESSION_FEATURES, gb3.feature_importances_),
        key=lambda x: -x[1])[:15]
    print('\n[3] top 15 feature importances (all-features GB):')
    for name, imp in fi:
        print(f'  {name:35s} {imp:.4f}')

    print('\n[4] scoreboard:')
    for k in ['gb_original', 'gb_expression_only', 'gb_all_features']:
        r = results[k]
        print(f'  {k:35s}  test MCC = {r["test_mcc"]:.4f}')
    print(f'  v25 GB 6-org (reference)             test MCC = 0.2268')
    print(f'  v27 GB+regulatory (failed)           test MCC = 0.1069')

    delta = results['gb_all_features']['test_mcc'] - 0.2268
    print(f'\n  delta vs v25 baseline: {delta:+.4f}')

    out = {
        'method': 'gb_on_v25_features_plus_organism_specific_expression_proxies',
        'expression_features': EXPRESSION_FEATURES,
        'expression_feature_description': {
            'cai': 'Codon Adaptation Index against organism\'s ribosomal proteins',
            'rbs_strength': 'identity match score vs organism-specific anti-SD template',
            'cai_rank': 'within-organism rank of CAI [0,1]',
            'rbs_rank': 'within-organism rank of rbs_strength [0,1]',
            'expression_rank': 'mean(cai_rank, rbs_rank)',
        },
        'training_organisms': TRAIN_ORGS,
        'test_organism': TEST_ORG,
        'results': results,
        'top_15_feature_importances': [{'feature': n, 'importance': float(i)}
                                          for n, i in fi],
        'baselines': {
            'v15_within_organism': 0.5372,
            'v24_mpne_to_syn3a_annotation': 0.2034,
            'v25_gb_6orgs_original': 0.2268,
            'v27_gb_with_regulatory_pwms': 0.1069,
        },
        'delta_vs_v25_baseline': float(delta),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()

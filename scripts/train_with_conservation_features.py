"""Phase 9 — class-weighted GB on v25 features + conservation features
(n_orthologs_total, n_orthologs_within_mollicutes, paralog_count_internal,
mean_identity_to_orthologs, plus 16 binary phyletic-profile features).

Key diagnostic: of v25's 148 false negatives on Syn3A, how many does
adding conservation features now catch correctly?

Output: outputs/multiorg_classifier_with_conservation_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef

ROOT = Path(__file__).resolve().parent.parent
INP = ROOT / 'outputs/multiorg_per_gene_features_with_conservation.csv'
OUT = ROOT / 'outputs/multiorg_classifier_with_conservation_results.json'

ORIGINAL_FEATURES = [
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
    print(f'  [{label:<55s}]  best_t={best_t:.2f}  train MCC={train_mcc:.4f}  '
          f'test MCC={test_mcc:.4f}  (TP={tp} FP={fp} TN={tn} FN={fn})')
    return {'label': label, 'best_threshold': float(best_t),
            'train_mcc': float(train_mcc), 'test_mcc': float(test_mcc),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'predictions': pred.tolist(), 'probs': probs.tolist()}


def main():
    print('[1] loading enriched matrix (v25 + conservation)...')
    df = pd.read_csv(INP)
    print(f'  rows: {len(df)}; total cols: {len(df.columns)}')

    base_cols = {'organism', 'locus_tag', 'gene_name', 'product', 'essential'}
    feat_cols = [c for c in df.columns if c not in base_cols]
    cons_cols = [c for c in feat_cols if c not in ORIGINAL_FEATURES]
    print(f'  conservation feature columns ({len(cons_cols)}): {cons_cols[:6]}...')

    # Pre-aggregate: simple conservation summary
    print('\n  conservation distribution by class for Syn3A:')
    syn = df[df.organism == 'syn3a']
    print(f'    syn3a essentials (n={int(syn.essential.sum())}): '
          f'mean_n_orth={syn[syn.essential==1].n_orthologs_total.mean():.2f}, '
          f'paralog={syn[syn.essential==1].paralog_count_internal.mean():.2f}')
    print(f'    syn3a nonessentials (n={int((syn.essential==0).sum())}): '
          f'mean_n_orth={syn[syn.essential==0].n_orthologs_total.mean():.2f}, '
          f'paralog={syn[syn.essential==0].paralog_count_internal.mean():.2f}')

    train = df[df.organism.isin(TRAIN_ORGS)].copy()
    test = df[df.organism == TEST_ORG].copy()

    Xtr_orig = train[ORIGINAL_FEATURES].fillna(0.0).values.astype(np.float32)
    Xtr_cons = train[cons_cols].fillna(0.0).values.astype(np.float32)
    Xtr_all = train[ORIGINAL_FEATURES + cons_cols].fillna(0.0).values.astype(np.float32)
    ytr = train.essential.astype(int).values

    Xte_orig = test[ORIGINAL_FEATURES].fillna(0.0).values.astype(np.float32)
    Xte_cons = test[cons_cols].fillna(0.0).values.astype(np.float32)
    Xte_all = test[ORIGINAL_FEATURES + cons_cols].fillna(0.0).values.astype(np.float32)
    yte = test.essential.astype(int).values

    sw = np.where(ytr == 1, (ytr == 0).sum() / len(ytr),
                   (ytr == 1).sum() / len(ytr))

    print('\n[2] training classifiers...')
    results = {}

    gb1 = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.1, random_state=SEED)
    results['gb_original_only'] = evaluate(gb1, Xtr_orig, ytr, Xte_orig, yte,
                                            'GB original 20 (v25 baseline)',
                                            sample_weight=sw)

    gb2 = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.1, random_state=SEED)
    results['gb_conservation_only'] = evaluate(gb2, Xtr_cons, ytr, Xte_cons, yte,
                                                'GB conservation features only',
                                                sample_weight=sw)

    gb3 = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.1, random_state=SEED)
    results['gb_all_features'] = evaluate(gb3, Xtr_all, ytr, Xte_all, yte,
                                           'GB original 20 + conservation',
                                           sample_weight=sw)

    # Diagnostic: per-confusion-class lift
    v25_pred = np.array(results['gb_original_only']['predictions'])
    new_pred = np.array(results['gb_all_features']['predictions'])
    v25_was_FN_now_TP = ((v25_pred == 0) & (new_pred == 1) & (yte == 1)).sum()
    v25_was_FP_now_TN = ((v25_pred == 1) & (new_pred == 0) & (yte == 0)).sum()
    v25_was_TN_now_FP = ((v25_pred == 0) & (new_pred == 1) & (yte == 0)).sum()
    v25_was_TP_now_FN = ((v25_pred == 1) & (new_pred == 0) & (yte == 1)).sum()
    print(f'\n[3] per-confusion-class lift (v25 -> v29):')
    print(f'  fixes: v25 FN -> TP : {v25_was_FN_now_TP} (essentials we now catch)')
    print(f'  fixes: v25 FP -> TN : {v25_was_FP_now_TN} (nonessentials we no longer mis-flag)')
    print(f'  errors: v25 TN -> FP : {v25_was_TN_now_FP} (nonessentials we now mis-flag)')
    print(f'  errors: v25 TP -> FN : {v25_was_TP_now_FN} (essentials we lost)')

    # Feature importance from all-features GB
    feat_names = ORIGINAL_FEATURES + cons_cols
    fi = sorted(zip(feat_names, gb3.feature_importances_),
                 key=lambda x: -x[1])[:15]
    print('\n[4] top 15 feature importances (all-features GB):')
    for name, imp in fi:
        print(f'  {name:35s} {imp:.4f}')

    print('\n[5] scoreboard:')
    for k in ['gb_original_only', 'gb_conservation_only', 'gb_all_features']:
        r = results[k]
        print(f'  {k:35s}  test MCC = {r["test_mcc"]:.4f}')
    print(f'  v25 GB 6-org (reference)             test MCC = 0.2268')
    print(f'  v27 GB+regulatory (failed)           test MCC = 0.1069')
    print(f'  v28 GB+expression (failed)           test MCC = 0.1942')
    delta = results['gb_all_features']['test_mcc'] - 0.2268
    print(f'\n  delta vs v25 baseline: {delta:+.4f}')

    out = {
        'method': 'gb_on_v25_features_plus_mmseqs2_rbh_conservation_features',
        'n_features_original': len(ORIGINAL_FEATURES),
        'n_features_conservation': len(cons_cols),
        'conservation_feature_keys': cons_cols,
        'training_organisms': TRAIN_ORGS,
        'test_organism': TEST_ORG,
        'results': {k: {kk: vv for kk, vv in v.items()
                          if kk not in ('predictions', 'probs')}
                     for k, v in results.items()},
        'per_confusion_class_lift_v25_to_v29': {
            'fixed_v25_FN_to_TP': int(v25_was_FN_now_TP),
            'fixed_v25_FP_to_TN': int(v25_was_FP_now_TN),
            'introduced_TN_to_FP': int(v25_was_TN_now_FP),
            'introduced_TP_to_FN': int(v25_was_TP_now_FN),
        },
        'top_15_feature_importances': [
            {'feature': n, 'importance': float(i)} for n, i in fi
        ],
        'baselines': {
            'v15_within_organism': 0.5372,
            'v25_gb_6orgs_original': 0.2268,
            'v27_gb_with_regulatory_pwms': 0.1069,
            'v28_gb_with_expression_features': 0.1942,
        },
        'delta_vs_v25_baseline': float(delta),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()

"""Phase 6 train+test — class-weighted gradient boosting on the v25 feature
matrix AUGMENTED with predicted regulatory features (50 new columns from
scripts/predict_regulatory_features.py: RBS / -10 / -35 / TF presence-and-
score for 14 major TFs, plus has_promoter_pair and predicted_tf_count).

Same train/test split as v25: 6 organisms train, Syn3A held out.

Output: outputs/multiorg_classifier_with_regulatory_results.json
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
INP = ROOT / 'outputs/multiorg_per_gene_features_with_regulatory.csv'
OUT = ROOT / 'outputs/multiorg_classifier_with_regulatory_results.json'

ORIGINAL_FEATURES = [
    'log_length_bp', 'gc', 'upstream_gc', 'upstream_at_skew',
    'position_norm', 'operon_run_length',
    'is_hypothetical', 'is_uncharacterized', 'is_putative',
    'kw_replication', 'kw_transcription', 'kw_translation',
    'kw_atp', 'kw_membrane', 'kw_synthase', 'kw_kinase',
    'kw_dehydrogenase', 'kw_protease', 'kw_rrna', 'kw_chaperone',
]

# All regulatory feature columns added by predict_regulatory_features.py
REG_FEATURES = []
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
    test_pred = (probs >= best_t).astype(int)
    if pd.Series(test_pred).nunique() < 2 or pd.Series(y_test).nunique() < 2:
        test_mcc = 0.0; tp, fp, tn, fn = 0, 0, 0, 0
    else:
        test_mcc = matthews_corrcoef(y_test, test_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
    print(f'  [{label}]  best_t={best_t:.2f}  train MCC={train_mcc:.4f}  '
          f'test MCC={test_mcc:.4f}  (TP={tp} FP={fp} TN={tn} FN={fn})')
    return {'label': label, 'best_threshold': float(best_t),
            'train_mcc': float(train_mcc), 'test_mcc': float(test_mcc),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}


def main():
    print('[1] loading enriched feature matrix...')
    df = pd.read_csv(INP)
    print(f'  rows: {len(df)}; cols: {len(df.columns)}')

    # All regulatory feature columns: anything not in ORIGINAL_FEATURES + base
    base_cols = ['organism', 'locus_tag', 'gene_name', 'product', 'essential']
    feat_cols = [c for c in df.columns if c not in base_cols]
    reg_cols = [c for c in feat_cols if c not in ORIGINAL_FEATURES]
    print(f'  original features: {len(ORIGINAL_FEATURES)}')
    print(f'  predicted regulatory features: {len(reg_cols)}')
    print(f'  sample regulatory: {reg_cols[:8]}')

    train = df[df.organism.isin(TRAIN_ORGS)].copy()
    test = df[df.organism == TEST_ORG].copy()
    print(f'\n  train: {len(train)} rows  test: {len(test)} rows')

    Xtr_all = train[ORIGINAL_FEATURES + reg_cols].fillna(0.0).values.astype(np.float32)
    Xtr_orig = train[ORIGINAL_FEATURES].fillna(0.0).values.astype(np.float32)
    Xtr_reg_only = train[reg_cols].fillna(0.0).values.astype(np.float32)
    ytr = train.essential.astype(int).values

    Xte_all = test[ORIGINAL_FEATURES + reg_cols].fillna(0.0).values.astype(np.float32)
    Xte_orig = test[ORIGINAL_FEATURES].fillna(0.0).values.astype(np.float32)
    Xte_reg_only = test[reg_cols].fillna(0.0).values.astype(np.float32)
    yte = test.essential.astype(int).values

    print('\n[2] training classifiers...')
    sw = np.where(ytr == 1, (ytr == 0).sum() / len(ytr),
                  (ytr == 1).sum() / len(ytr))

    results = {}

    # 1. GB on ORIGINAL features only (v25 reproduction baseline)
    gb1 = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.1, random_state=SEED)
    results['gb_original_only'] = evaluate(gb1, Xtr_orig, ytr, Xte_orig, yte,
                                            'GB original 20 features (v25 reproduction)',
                                            sample_weight=sw)

    # 2. GB on REGULATORY features only (does the regulatory signal alone work?)
    gb2 = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.1, random_state=SEED)
    results['gb_regulatory_only'] = evaluate(gb2, Xtr_reg_only, ytr, Xte_reg_only, yte,
                                              'GB regulatory features only',
                                              sample_weight=sw)

    # 3. GB on ALL features (original + regulatory)
    gb3 = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.1, random_state=SEED)
    results['gb_all_features'] = evaluate(gb3, Xtr_all, ytr, Xte_all, yte,
                                           'GB original + predicted regulatory',
                                           sample_weight=sw)

    # 4. Class-weighted logreg on all features
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', C=1.0, max_iter=500,
                            solver='liblinear', random_state=SEED),
    )
    results['logreg_all_features'] = evaluate(lr, Xtr_all, ytr, Xte_all, yte,
                                                'LogReg original + predicted regulatory')

    # Feature importances from the all-features GB
    fi = sorted(
        zip(ORIGINAL_FEATURES + reg_cols, gb3.feature_importances_),
        key=lambda x: -x[1])[:15]
    print('\n[3] top 15 feature importances (all-features GB):')
    for name, imp in fi:
        print(f'  {name:35s} {imp:.4f}')

    # Scoreboard
    print('\n[4] scoreboard:')
    for k in ['gb_original_only', 'gb_regulatory_only', 'gb_all_features',
              'logreg_all_features']:
        r = results[k]
        print(f'  {k:35s}  test MCC = {r["test_mcc"]:.4f}')
    print(f'  v25 GB 6-org (reference)             test MCC = 0.2268')
    print(f'  v24 mpne->syn3a (reference)          test MCC = 0.2034')
    print(f'  v15 within-organism                  test MCC = 0.5372')

    out = {
        'method': 'gb_on_v25_features_plus_predicted_regulatory_motifs_from_salmonella_pwms',
        'n_features_original': len(ORIGINAL_FEATURES),
        'n_features_regulatory_predicted': len(reg_cols),
        'regulatory_feature_keys': reg_cols,
        'training_organisms': TRAIN_ORGS,
        'test_organism': TEST_ORG,
        'train_size': int(len(train)),
        'test_size': int(len(test)),
        'results': results,
        'top_15_feature_importances_all_features_gb':
            [{'feature': n, 'importance': float(i)} for n, i in fi],
        'baselines': {
            'v15_within_organism': 0.5372,
            'v24_mpne_to_syn3a_annotation': 0.2034,
            'v25_gb_6orgs_original_only': 0.2268,
            'v18_xgboost_esm2': 0.1376,
        },
    }
    delta = float(results['gb_all_features']['test_mcc'] - 0.2268)
    out['delta_vs_v25_gb_baseline'] = delta
    print(f'\n  delta vs v25 GB baseline: {delta:+.4f}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()

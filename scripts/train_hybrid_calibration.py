"""Phase 10 / Path C — hybrid transfer with Syn3A threshold calibration.

Hypothesis: v25-v29 all hit a recall-vs-specificity trade-off because the
training threshold (tuned on 6 orgs with 5-70% essential class balance)
is wrong for Syn3A's 84% essential test distribution. Per-gene
probabilities may carry the right signal; the decision boundary is
mis-calibrated.

Method
------

1. Train ONE GB on 6 train organisms (Syn3A held out, same as v25).
2. Run inference on all 455 Syn3A genes -> fixed probability vector.
3. 30-iteration stratified shuffle-split:
     - sample N Syn3A genes for calibration (preserving 84/16 ratio)
     - search best threshold on calibration (maximize calibration MCC)
     - apply threshold to remaining 455-N test genes
     - record test MCC
4. Sweep calibration size: 25, 50, 100.
5. Compare to v25's 0.2268 (training-threshold, full 455).

Two model variants:
  A. v25 features (20)
  B. v25 + conservation features (42 from v29 build)

Output: outputs/hybrid_calibration_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
INP_V25 = ROOT / 'outputs/multiorg_per_gene_features.csv'
INP_V29 = ROOT / 'outputs/multiorg_per_gene_features_with_conservation.csv'
OUT = ROOT / 'outputs/hybrid_calibration_results.json'

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
N_ITERATIONS = 30
CALIBRATION_SIZES = [25, 50, 100]


def threshold_search(y_true, probs):
    """Return (best_threshold, best_mcc) over a fine threshold grid."""
    best_t, best_mcc = 0.5, -2.0
    # Use percentile thresholds on probs to avoid sparse-grid issues
    cand_thresholds = sorted(set(np.percentile(probs, np.linspace(2, 98, 49))))
    cand_thresholds = [0.05, 0.1, 0.5, 0.9, 0.95] + list(cand_thresholds)
    for t in cand_thresholds:
        p = (probs >= t).astype(int)
        if p.sum() in (0, len(p)): continue
        if len(set(y_true)) < 2: return t, 0.0
        m = matthews_corrcoef(y_true, p)
        if m > best_mcc: best_mcc, best_t = m, t
    return best_t, best_mcc


def train_inference(features_df, feat_cols):
    """Train GB on 6 orgs, infer on Syn3A. Return (test_probs, test_y, train_threshold)."""
    train = features_df[features_df.organism.isin(TRAIN_ORGS)].copy()
    test = features_df[features_df.organism == TEST_ORG].copy()

    Xtr = train[feat_cols].fillna(0.0).values.astype(np.float32)
    ytr = train.essential.astype(int).values
    Xte = test[feat_cols].fillna(0.0).values.astype(np.float32)
    yte = test.essential.astype(int).values

    sw = np.where(ytr == 1, (ytr == 0).sum() / len(ytr),
                   (ytr == 1).sum() / len(ytr))
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                      learning_rate=0.1, random_state=SEED)
    gb.fit(Xtr, ytr, sample_weight=sw)
    train_probs = gb.predict_proba(Xtr)[:, 1]
    train_t, _ = threshold_search(ytr, train_probs)
    test_probs = gb.predict_proba(Xte)[:, 1]
    return test_probs, yte, float(train_t)


def hybrid_calibration_eval(test_probs, test_y, calib_size, n_iters=30,
                              base_seed=42):
    """Run shuffle-split calibration: sample calib_size genes from Syn3A,
    tune threshold on them, apply to the rest."""
    test_mccs = []
    test_thresholds = []
    test_confusions = []

    sss = StratifiedShuffleSplit(n_splits=n_iters,
                                   train_size=calib_size,
                                   random_state=base_seed)
    for i, (calib_idx, test_idx) in enumerate(sss.split(test_probs, test_y)):
        cal_p = test_probs[calib_idx]
        cal_y = test_y[calib_idx]
        tst_p = test_probs[test_idx]
        tst_y = test_y[test_idx]
        # Tune threshold on calibration set
        t, _ = threshold_search(cal_y, cal_p)
        # Apply to test set
        pred = (tst_p >= t).astype(int)
        if len(set(pred)) > 1 and len(set(tst_y)) > 1:
            mcc = matthews_corrcoef(tst_y, pred)
            tn, fp, fn, tp = confusion_matrix(tst_y, pred).ravel()
        else:
            mcc = 0.0; tp = fp = tn = fn = 0
        test_mccs.append(mcc)
        test_thresholds.append(t)
        test_confusions.append((int(tp), int(fp), int(tn), int(fn)))
    return {
        'mean_mcc': float(np.mean(test_mccs)),
        'std_mcc': float(np.std(test_mccs)),
        'median_mcc': float(np.median(test_mccs)),
        'min_mcc': float(np.min(test_mccs)),
        'max_mcc': float(np.max(test_mccs)),
        'q25_mcc': float(np.percentile(test_mccs, 25)),
        'q75_mcc': float(np.percentile(test_mccs, 75)),
        'mean_threshold': float(np.mean(test_thresholds)),
        'std_threshold': float(np.std(test_thresholds)),
        'mean_confusion': {
            'tp': float(np.mean([c[0] for c in test_confusions])),
            'fp': float(np.mean([c[1] for c in test_confusions])),
            'tn': float(np.mean([c[2] for c in test_confusions])),
            'fn': float(np.mean([c[3] for c in test_confusions])),
        },
        'all_test_mccs': test_mccs,
        'all_thresholds': test_thresholds,
    }


def fixed_threshold_eval(test_probs, test_y, threshold):
    """Apply a fixed (e.g. training-tuned) threshold to all of Syn3A. v25
    baseline reproduction."""
    pred = (test_probs >= threshold).astype(int)
    if len(set(pred)) > 1 and len(set(test_y)) > 1:
        mcc = matthews_corrcoef(test_y, pred)
        tn, fp, fn, tp = confusion_matrix(test_y, pred).ravel()
    else:
        mcc = 0.0; tp = fp = tn = fn = 0
    return {'mcc': float(mcc), 'threshold': float(threshold),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}


def main():
    print('[1] loading feature matrices...')
    df_v25 = pd.read_csv(INP_V25)
    df_v29 = pd.read_csv(INP_V29)
    print(f'  v25 features: {len(df_v25)} rows, {len(df_v25.columns)} cols')
    print(f'  v29 features: {len(df_v29)} rows, {len(df_v29.columns)} cols')

    # Determine conservation feature columns (in v29 but not v25)
    base_cols = {'organism', 'locus_tag', 'gene_name', 'product', 'essential'}
    cons_cols = [c for c in df_v29.columns
                  if c not in base_cols and c not in ORIGINAL_FEATURES]
    print(f'  conservation features ({len(cons_cols)}): {cons_cols[:4]}...')

    print('\n[2] training each model variant + running inference on Syn3A...')
    variants = {
        'v25_original': (df_v25, ORIGINAL_FEATURES),
        'v29_with_conservation': (df_v29, ORIGINAL_FEATURES + cons_cols),
    }

    inference_outputs = {}
    for name, (df, cols) in variants.items():
        probs, y, train_t = train_inference(df, cols)
        baseline = fixed_threshold_eval(probs, y, train_t)
        print(f'  [{name:<25s}]  train-tuned threshold = {train_t:.3f}  '
              f'baseline MCC (full 455) = {baseline["mcc"]:.4f}  '
              f'TP={baseline["tp"]} FP={baseline["fp"]} '
              f'TN={baseline["tn"]} FN={baseline["fn"]}')
        inference_outputs[name] = {
            'probs': probs, 'y': y, 'train_threshold': train_t,
            'baseline_fixed_threshold': baseline,
        }

    print('\n[3] hybrid calibration (shuffle-split, 30 iters per (variant, size))...')
    print(f'  {"variant":<25s} {"calib":>5s} {"mean":>7s} {"std":>6s} '
          f'{"median":>7s} {"q25":>6s} {"q75":>6s} {"mean_t":>7s}')
    print('  ' + '-' * 80)
    hybrid_results = {}
    for name, out in inference_outputs.items():
        for csize in CALIBRATION_SIZES:
            r = hybrid_calibration_eval(out['probs'], out['y'], csize,
                                         n_iters=N_ITERATIONS, base_seed=SEED)
            hybrid_results[(name, csize)] = r
            print(f'  {name:<25s} {csize:>5d} '
                  f'{r["mean_mcc"]:>7.4f} {r["std_mcc"]:>6.4f} '
                  f'{r["median_mcc"]:>7.4f} {r["q25_mcc"]:>6.4f} {r["q75_mcc"]:>6.4f} '
                  f'{r["mean_threshold"]:>7.3f}')

    print('\n[4] scoreboard:')
    print(f'  v25 fixed-threshold (training-tuned, full 455)        MCC = '
          f'{inference_outputs["v25_original"]["baseline_fixed_threshold"]["mcc"]:.4f}')
    print(f'  v29 fixed-threshold (training-tuned, full 455)        MCC = '
          f'{inference_outputs["v29_with_conservation"]["baseline_fixed_threshold"]["mcc"]:.4f}')
    for (name, csize), r in hybrid_results.items():
        print(f'  {name + " hybrid calib=" + str(csize):<48s} '
              f'mean MCC = {r["mean_mcc"]:.4f} (+/- {r["std_mcc"]:.4f})')

    # Best result for the verdict
    best_key, best_r = max(hybrid_results.items(),
                            key=lambda kv: kv[1]['mean_mcc'])
    delta_v25 = best_r['mean_mcc'] - 0.2268
    print(f'\n  best hybrid: {best_key[0]} @ calib_size={best_key[1]}')
    print(f'    mean test MCC = {best_r["mean_mcc"]:.4f} (vs v25 0.2268, delta = {delta_v25:+.4f})')

    out_dict = {
        'method': 'hybrid_transfer_with_syn3a_threshold_calibration_shuffle_split',
        'training_organisms': TRAIN_ORGS,
        'test_organism': TEST_ORG,
        'n_iterations_per_split': N_ITERATIONS,
        'calibration_sizes_tested': CALIBRATION_SIZES,
        'fixed_threshold_baselines': {
            name: out['baseline_fixed_threshold']
            for name, out in inference_outputs.items()
        },
        'hybrid_results': {
            f'{name}__calib_{csize}': {k: v for k, v in r.items()
                                          if k not in ('all_test_mccs', 'all_thresholds')}
            for (name, csize), r in hybrid_results.items()
        },
        'baselines': {
            'v15_within_organism': 0.5372,
            'v25_gb_6orgs_original_fixed_threshold': 0.2268,
            'v27_gb_with_regulatory_pwms': 0.1069,
            'v28_gb_with_expression_features': 0.1942,
            'v29_gb_with_conservation_features': 0.1840,
        },
        'best_hybrid_variant': best_key[0],
        'best_calibration_size': best_key[1],
        'best_mean_mcc': best_r['mean_mcc'],
        'delta_vs_v25_baseline': float(delta_v25),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out_dict, indent=2, default=str))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()

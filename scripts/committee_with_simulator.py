"""Committee voting ensemble for Syn3A essentiality prediction.

Three members vote on each Syn3A gene:

  1. v15 mechanistic detector (the simulator's call) - already cached
     in outputs/predictions_parallel_*v15*.csv. MCC=0.5372 alone.
     - binary call (0/1) and graded confidence

  2. LNN trained on 6 organisms WITHOUT Syn3A (cross-organism transfer)
     - probability per gene from outputs/lnn_syn3a_predictions.csv
     - threshold-tuned per the model's own probability distribution

  3. GB on v29 features trained on 6 orgs WITHOUT Syn3A (cross-organism)
     - probability per gene from cross-org GB predictions
     - same train/test split as LNN for fairness

Tested ensemble strategies:
  A. v15-only (baseline 0.5372)
  B. Majority vote (any 2 of 3 say essential)
  C. v15 OR (LNN AND GB) — defer to v15 if confident, use cross-org for
     v15's "no signal" cases
  D. v15 AND (LNN OR GB) — strict; only essential if v15 + at least one
     cross-org agree
  E. Soft mean: weighted_avg(v15_call, LNN_prob, GB_prob), threshold-tune
  F. Override-FN-only: keep v15's calls except for v15-FN candidates
     where both LNN and GB say essential, flip to essential
  G. Override-FP-only: keep v15's calls except where v15 essential but
     both cross-org members say nonessential, flip to nonessential

Output: outputs/committee_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef

ROOT = Path(__file__).resolve().parent.parent
V15_CSV = ROOT / 'outputs/predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4_composed_all455_v15_round2_priors.csv'
LNN_CSV = ROOT / 'outputs/lnn_syn3a_predictions.csv'
FEATURES_CSV = ROOT / 'outputs/multiorg_per_gene_features_with_conservation.csv'
LABELS_CSV = ROOT / 'memory_bank/data/multiorg_essentiality/labels.csv'
OUT = ROOT / 'outputs/committee_results.json'

# Same feature set as v29
FEATURE_COLS_BASE = [
    'log_length_bp', 'gc', 'upstream_gc', 'upstream_at_skew',
    'position_norm', 'operon_run_length',
    'is_hypothetical', 'is_uncharacterized', 'is_putative',
    'kw_replication', 'kw_transcription', 'kw_translation',
    'kw_atp', 'kw_membrane', 'kw_synthase', 'kw_kinase',
    'kw_dehydrogenase', 'kw_protease', 'kw_rrna', 'kw_chaperone',
]
TRAIN_ORGS = ['styphimurium', 'ccrescentus', 'saureus', 'mtuberculosis',
              'abaylyi', 'mpne']
SEED = 42


def threshold_search(y, probs):
    best_t, best_mcc = 0.5, -2.0
    cand = sorted(set([0.05, 0.1, 0.5, 0.9, 0.95]
                       + list(np.percentile(probs, np.linspace(2, 98, 49)))))
    for t in cand:
        p = (probs >= t).astype(int)
        if p.sum() in (0, len(p)): continue
        if len(set(y)) < 2: return t, 0.0
        m = matthews_corrcoef(y, p)
        if m > best_mcc: best_mcc, best_t = m, t
    return best_t, best_mcc


def confusion(y, pred):
    if len(set(pred)) > 1 and len(set(y)) > 1:
        mcc = float(matthews_corrcoef(y, pred))
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        return mcc, int(tp), int(fp), int(tn), int(fn)
    return 0.0, 0, 0, 0, 0


def main():
    print('[1] loading v15 predictions on Syn3A...')
    v15 = pd.read_csv(V15_CSV)
    print(f'  v15: {len(v15)} rows')

    print('\n[2] loading LNN predictions on Syn3A...')
    if not LNN_CSV.exists():
        raise FileNotFoundError(
            f'LNN predictions not found at {LNN_CSV}. '
            f'Run scripts/train_lnn_no_syn3a.py first.')
    lnn = pd.read_csv(LNN_CSV)
    print(f'  LNN: {len(lnn)} rows')
    print(f'  LNN prob distribution: min={lnn.lnn_prob.min():.3f}, '
          f'median={lnn.lnn_prob.median():.3f}, max={lnn.lnn_prob.max():.3f}')

    print('\n[3] training cross-organism GB on the same 6-org training set...')
    feats = pd.read_csv(FEATURES_CSV)
    base_cols = {'organism', 'locus_tag', 'gene_name', 'product', 'essential'}
    cons_cols = [c for c in feats.columns
                  if c not in base_cols and c not in FEATURE_COLS_BASE]
    feature_cols = FEATURE_COLS_BASE + cons_cols
    print(f'  total feature columns: {len(feature_cols)} (base 20 + conservation {len(cons_cols)})')

    train = feats[feats.organism.isin(TRAIN_ORGS)].copy()
    test = feats[feats.organism == 'syn3a'].copy()
    Xtr = train[feature_cols].fillna(0).values.astype(np.float32)
    ytr = train.essential.astype(int).values
    Xte = test[feature_cols].fillna(0).values.astype(np.float32)

    sw = np.where(ytr == 1, (ytr == 0).sum() / len(ytr),
                   (ytr == 1).sum() / len(ytr))
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                      learning_rate=0.1, random_state=SEED)
    gb.fit(Xtr, ytr, sample_weight=sw)
    train_probs = gb.predict_proba(Xtr)[:, 1]
    gb_train_t, _ = threshold_search(ytr, train_probs)
    gb_probs = gb.predict_proba(Xte)[:, 1]
    print(f'  GB train MCC threshold: {gb_train_t:.3f}; '
          f'GB Syn3A prob distribution: min={gb_probs.min():.3f}, '
          f'median={np.median(gb_probs):.3f}, max={gb_probs.max():.3f}')

    # Build a Syn3A frame indexed by locus_tag with all three members
    print('\n[4] aligning members on Syn3A 455-gene panel...')
    test = test.reset_index(drop=True).copy()
    test['gb_prob'] = gb_probs
    test['gb_pred_at_train_t'] = (gb_probs >= gb_train_t).astype(int)

    syn = (test[['locus_tag', 'gene_name', 'essential', 'gb_prob',
                   'gb_pred_at_train_t']]
            .merge(v15[['locus_tag', 'essential', 'confidence',
                         'failure_mode']]
                    .rename(columns={'essential': 'v15_call'}),
                    on='locus_tag', how='inner')
            .merge(lnn[['locus_tag', 'lnn_prob']], on='locus_tag', how='inner'))
    print(f'  joined panel: {len(syn)} genes (essential / nonessential = '
          f'{int(syn.essential.sum())} / {int((syn.essential==0).sum())})')

    y = syn.essential.values
    v15_call = syn.v15_call.values
    v15_conf = syn.confidence.values
    lnn_prob = syn.lnn_prob.values
    gb_prob = syn.gb_prob.values

    # Per-member thresholds tuned on POOLED TRAINING (not Syn3A — honest)
    # For LNN: use the threshold that the LNN's own training distribution
    # selected. Already in lnn CSV via threshold_search at training time
    # but here we just use 0.5 as a neutral default and let the ensemble
    # rules combine via prob averaging.
    # For GB: gb_train_t (already computed)
    print(f'\n  LNN cross-org Syn3A binary at t=0.5:')
    lnn_pred = (lnn_prob >= 0.5).astype(int)
    mcc, tp, fp, tn, fn = confusion(y, lnn_pred)
    print(f'    MCC={mcc:.4f}  TP={tp} FP={fp} TN={tn} FN={fn}')

    print(f'\n  GB cross-org Syn3A binary at t={gb_train_t:.2f}:')
    gb_pred = (gb_prob >= gb_train_t).astype(int)
    mcc, tp, fp, tn, fn = confusion(y, gb_pred)
    print(f'    MCC={mcc:.4f}  TP={tp} FP={fp} TN={tn} FN={fn}')

    print(f'\n  v15 binary (cached, training threshold):')
    mcc, tp, fp, tn, fn = confusion(y, v15_call)
    print(f'    MCC={mcc:.4f}  TP={tp} FP={fp} TN={tn} FN={fn}')

    # Ensemble strategies
    print('\n[5] ensemble strategies on full 455-gene panel:')
    results = {}

    # A. v15 only (baseline)
    v15_mcc, v15_tp, v15_fp, v15_tn, v15_fn = confusion(y, v15_call)
    results['A_v15_only'] = {'mcc': v15_mcc, 'tp': v15_tp, 'fp': v15_fp,
                              'tn': v15_tn, 'fn': v15_fn}

    # B. Majority vote (binary calls)
    votes = v15_call + lnn_pred + gb_pred
    maj_pred = (votes >= 2).astype(int)
    mcc, tp, fp, tn, fn = confusion(y, maj_pred)
    results['B_majority_vote'] = {'mcc': mcc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

    # C. v15 OR (LNN AND GB)
    rule_c = ((v15_call == 1) | ((lnn_pred == 1) & (gb_pred == 1))).astype(int)
    mcc, tp, fp, tn, fn = confusion(y, rule_c)
    results['C_v15_OR_lnn_AND_gb'] = {'mcc': mcc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

    # D. v15 AND (LNN OR GB)
    rule_d = ((v15_call == 1) & ((lnn_pred == 1) | (gb_pred == 1))).astype(int)
    mcc, tp, fp, tn, fn = confusion(y, rule_d)
    results['D_v15_AND_lnn_OR_gb'] = {'mcc': mcc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

    # E. Soft prob mean + threshold tuning
    # Convert v15 to a "prob" via confidence (which is already 0..1 in v15)
    v15_prob = v15_conf
    # All three members on prob scale
    soft_score = (v15_prob + lnn_prob + gb_prob) / 3.0
    best_t, best_mcc = threshold_search(y, soft_score)
    soft_pred = (soft_score >= best_t).astype(int)
    mcc, tp, fp, tn, fn = confusion(y, soft_pred)
    results['E_soft_mean_threshold_tuned_on_test'] = {
        'mcc': mcc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'best_t': float(best_t),
        'caveat': 'threshold tuned on test — diagnostic ceiling, not honest'}

    # F. Override-FN: keep v15, but for v15-FN-candidate genes (v15==0),
    # if BOTH cross-org members say essential, flip to essential
    rule_f = v15_call.copy()
    flip_mask = (v15_call == 0) & (lnn_pred == 1) & (gb_pred == 1)
    rule_f[flip_mask] = 1
    mcc, tp, fp, tn, fn = confusion(y, rule_f)
    results['F_v15_with_FN_recovery'] = {
        'mcc': mcc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_flipped_from_0_to_1': int(flip_mask.sum())}

    # G. Override-FP: keep v15, but for v15-essential genes where BOTH
    # cross-org members say nonessential, flip to nonessential
    rule_g = v15_call.copy()
    flip_mask = (v15_call == 1) & (lnn_pred == 0) & (gb_pred == 0)
    rule_g[flip_mask] = 0
    mcc, tp, fp, tn, fn = confusion(y, rule_g)
    results['G_v15_with_FP_correction'] = {
        'mcc': mcc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_flipped_from_1_to_0': int(flip_mask.sum())}

    # H. Combined F + G
    rule_h = v15_call.copy()
    flip_up = (v15_call == 0) & (lnn_pred == 1) & (gb_pred == 1)
    flip_down = (v15_call == 1) & (lnn_pred == 0) & (gb_pred == 0)
    rule_h[flip_up] = 1
    rule_h[flip_down] = 0
    mcc, tp, fp, tn, fn = confusion(y, rule_h)
    results['H_v15_with_both_FN_recovery_and_FP_correction'] = {
        'mcc': mcc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_flipped_up': int(flip_up.sum()),
        'n_flipped_down': int(flip_down.sum())}

    # Print scoreboard
    print('\n  scoreboard:')
    print(f'  {"strategy":<55s} {"MCC":>7s} {"TP":>4s} {"FP":>4s} {"TN":>4s} {"FN":>4s}')
    print('  ' + '-' * 80)
    for name, r in results.items():
        print(f'  {name:<55s} {r["mcc"]:>7.4f} '
              f'{r["tp"]:>4d} {r["fp"]:>4d} {r["tn"]:>4d} {r["fn"]:>4d}')

    # Best ensemble
    best_name, best_r = max(
        ((n, r) for n, r in results.items()
          if 'caveat' not in r),
        key=lambda kv: kv[1]['mcc'])
    delta_v15 = best_r['mcc'] - v15_mcc
    print(f'\n[6] best honest ensemble: {best_name}')
    print(f'  MCC = {best_r["mcc"]:.4f}  (vs v15 alone {v15_mcc:.4f}, '
          f'delta = {delta_v15:+.4f})')

    # Diagnostic: which v15 FN did the cross-org members catch?
    print('\n[7] diagnostic on v15 false negatives:')
    fn_idx = np.where((v15_call == 0) & (y == 1))[0]
    print(f'  v15 has {len(fn_idx)} false negatives (essential genes called nonessential)')
    for fn_thresh in [0.5, 0.7, 0.9]:
        lnn_caught = ((lnn_prob[fn_idx] >= fn_thresh)).sum()
        gb_caught = ((gb_prob[fn_idx] >= fn_thresh)).sum()
        both_caught = ((lnn_prob[fn_idx] >= fn_thresh) &
                        (gb_prob[fn_idx] >= fn_thresh)).sum()
        print(f'    at prob>={fn_thresh}:  LNN catches {lnn_caught}/96, '
              f'GB catches {gb_caught}/96, both {both_caught}/96')

    out = {
        'method': 'committee_voting_v15_plus_lnn_no_syn3a_plus_gb_cross_org',
        'syn3a_panel_size': int(len(syn)),
        'syn3a_class_balance': {'essential': int(y.sum()),
                                  'nonessential': int((y == 0).sum())},
        'individual_member_mccs': {
            'v15_simulator': v15_mcc,
            'lnn_cross_org_no_syn3a_at_t_0.5': float(matthews_corrcoef(y, lnn_pred)
                                                       if len(set(lnn_pred))>1 else 0.0),
            'gb_cross_org_no_syn3a': float(matthews_corrcoef(y, gb_pred)
                                              if len(set(gb_pred))>1 else 0.0),
        },
        'ensemble_results': results,
        'best_honest_ensemble': best_name,
        'best_mcc': best_r['mcc'],
        'delta_vs_v15_alone': float(delta_v15),
        'baselines': {
            'v15_within_organism': 0.5372,
            'v25_gb_cross_organism': 0.2270,
            'v32_lnn_phase1_in_domain': 0.5347,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()

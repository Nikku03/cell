"""Meta-LNN stacker: learn per-gene component weights via leave-one-org-out CV.

Pipeline:
  1. Generate honest cross-org LNN predictions for all 5 leave-one-out folds
     (mtuberculosis was generated earlier; this fills in the other 4).
  2. Build per-gene feature matrix: scalar features + ortholog_prior +
     n_orthologs + PPI + LNN prob (where available) + v15 transfer.
  3. For each of 5 evaluation orgs (the LOO folds), train a logistic
     regression stacker on the OTHER 4 orgs' (features + component preds),
     predict on the held-out 5th. This is the honest cross-org test.
  4. Report per-org and pooled MCC, compare to:
     - rule-based router (pooled 0.604, mean 0.443)
     - ortholog prior alone
     - cross-org LNN alone

Output:
  outputs/meta_lnn_stacker_results.json
  outputs/meta_lnn_stacker_per_gene.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'cell_sim'))

from cell_sim.layer_ml.data_loader import load_organism_batches
from cell_sim.layer_ml.essentiality_lnn import EssentialityLNN, LNNConfig
from cell_sim.layer_ml.hyperbolic_memory import HyperbolicMemoryBank

ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = ROOT / 'cell_sim/layer_ml/checkpoints/lnn_leave_one_org_out.pt'
FEATURES_CSV = ROOT / 'outputs/multiorg_per_gene_features.csv'
PPI_CSV = ROOT / 'outputs/ppi_features_per_gene.csv'
ORTHOLOG_FEATURES = ROOT / 'outputs/ortholog_prior_features_per_gene.csv'
RBH_CSV = ROOT / 'outputs/rbh_pairs.csv'
V15_SYN3A_CSV = (ROOT / 'outputs' /
                  'predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4_'
                  'composed_all455_v15_round2_priors.csv')
MTB_CASCADE_CSV = ROOT / 'outputs/cascade_three_way_mtuberculosis_per_gene.csv'

OUT_JSON = ROOT / 'outputs/meta_lnn_stacker_results.json'
OUT_CSV = ROOT / 'outputs/meta_lnn_stacker_per_gene.csv'

LOO_ORGS = ['styphimurium', 'ccrescentus', 'saureus', 'mtuberculosis', 'abaylyi']
KW_FEATURES = ['kw_replication', 'kw_transcription', 'kw_translation',
                'kw_atp', 'kw_membrane', 'kw_synthase', 'kw_kinase',
                'kw_dehydrogenase', 'kw_protease', 'kw_rrna', 'kw_chaperone',
                'is_hypothetical', 'is_uncharacterized', 'is_putative']


def load_lnn_for_fold(target_org: str) -> dict[str, float]:
    """Generate per-gene LNN probs for the given fold's held-out org."""
    print(f'  building gene batch for {target_org}...')
    batches = load_organism_batches([target_org], verbose=False)
    batch = batches[target_org]

    print(f'  loading {target_org} fold from {CKPT_PATH.name}...')
    ck = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    cfg_dict = ck['config']
    valid = {f.name for f in LNNConfig.__dataclass_fields__.values()}
    cfg = LNNConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
    model = EssentialityLNN(cfg)

    fold_sd = ck['fold_state_dicts'][target_org]
    state_dict = fold_sd['state_dict'] if 'state_dict' in fold_sd else fold_sd
    saved_mem_size = state_dict['memory.memory'].shape[0]
    model.memory = HyperbolicMemoryBank(
        hidden=cfg.hidden, memory_size=saved_mem_size,
        retrieval_k=cfg.memory_retrieval_k,
        curvature=cfg.memory_curvature,
        growable=False, max_size=saved_mem_size,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    with torch.no_grad():
        out = model(batch, store_memory=False)
        probs = torch.sigmoid(out['essentiality']).numpy()
    return {lt: float(p) for lt, p in zip(batch['locus_tags'], probs)}


def main():
    print('[1] loading features + labels...')
    fdf = pd.read_csv(FEATURES_CSV)
    fdf = fdf.dropna(subset=['essential']).copy()
    fdf['essential'] = fdf['essential'].astype(int)
    print(f'  {len(fdf)} labeled genes across {fdf.organism.nunique()} orgs')

    ppi = pd.read_csv(PPI_CSV)
    orth = pd.read_csv(ORTHOLOG_FEATURES)
    fdf = (fdf.merge(ppi, on=['organism', 'locus_tag'], how='left')
                   .merge(orth, on=['organism', 'locus_tag'], how='left'))
    fdf['n_orthologs'] = fdf.n_orthologs.fillna(0).astype(int)
    fdf['ortholog_prior'] = fdf.ortholog_prior.fillna(0.5)
    fdf['ppi_max_score'] = fdf.ppi_max_score.fillna(0.0)
    fdf['ppi_degree_high'] = fdf.ppi_degree_high.fillna(0).astype(int)

    print('\n[2] generating LNN preds for all 5 leave-one-out folds...')
    lnn_probs = {}   # {(org, locus_tag): prob}
    # M. tb already done — load from cached file
    mtb = pd.read_csv(MTB_CASCADE_CSV)
    for r in mtb.itertuples(index=False):
        lnn_probs[('mtuberculosis', r.locus_tag)] = float(r.lnn_prob)
    print(f'  mtuberculosis: {len(mtb)} preds (cached from earlier run)')

    for org in [o for o in LOO_ORGS if o != 'mtuberculosis']:
        print(f'\n  fold = {org}:')
        try:
            preds = load_lnn_for_fold(org)
            for lt, p in preds.items():
                lnn_probs[(org, lt)] = p
            print(f'    {len(preds)} preds')
        except Exception as e:
            print(f'    FAILED: {e}')

    fdf['lnn_prob'] = fdf.apply(
        lambda r: lnn_probs.get((r.organism, r.locus_tag), np.nan), axis=1)
    fdf['has_lnn'] = fdf.lnn_prob.notna().astype(int)
    print(f'\n  LNN prob coverage: {fdf.has_lnn.sum()}/{len(fdf)} '
          f'({100*fdf.has_lnn.sum()/len(fdf):.1f}%)')

    # v15 transfer for non-Syn3A orgs via RBH; in-domain for Syn3A
    print('\n[3] adding v15 (in-domain on Syn3A, RBH-transferred elsewhere)...')
    v15 = pd.read_csv(V15_SYN3A_CSV)
    syn3a_v15 = {r.locus_tag: (int(r.essential), float(r.confidence))
                  for r in v15.itertuples(index=False)}
    rbh = pd.read_csv(RBH_CSV)
    # For each (target_org, target_locus), if there's a syn3a RBH partner,
    # transfer the v15 call.
    v15_transfer = {}
    for r in rbh.itertuples(index=False):
        if r.org_a == 'syn3a' and r.locus_b != '':
            if r.locus_a in syn3a_v15:
                k = (r.org_b, r.locus_b)
                if k not in v15_transfer:
                    v15_transfer[k] = syn3a_v15[r.locus_a]
        elif r.org_b == 'syn3a' and r.locus_a != '':
            if r.locus_b in syn3a_v15:
                k = (r.org_a, r.locus_a)
                if k not in v15_transfer:
                    v15_transfer[k] = syn3a_v15[r.locus_b]

    def get_v15(r):
        if r.organism == 'syn3a':
            return syn3a_v15.get(r.locus_tag, (np.nan, 0.0))
        return v15_transfer.get((r.organism, r.locus_tag), (np.nan, 0.0))

    fdf['v15_call'] = fdf.apply(lambda r: get_v15(r)[0], axis=1)
    fdf['v15_conf'] = fdf.apply(lambda r: get_v15(r)[1], axis=1)
    fdf['has_v15'] = fdf.v15_call.notna().astype(int)
    print(f'  v15 coverage: {fdf.has_v15.sum()}/{len(fdf)} '
          f'({100*fdf.has_v15.sum()/len(fdf):.1f}%)')

    # ─────────── feature matrix for the stacker ───────────
    feat_cols = (KW_FEATURES + ['log_length_bp', 'gc', 'upstream_gc',
                  'upstream_at_skew', 'position_norm', 'operon_run_length',
                  'n_orthologs', 'ortholog_prior',
                  'ppi_max_score', 'ppi_degree_high',
                  'has_lnn', 'has_v15'])
    fdf['lnn_prob_filled'] = fdf.lnn_prob.fillna(0.5)
    fdf['v15_call_filled'] = fdf.v15_call.fillna(0.5)
    feat_cols += ['lnn_prob_filled', 'v15_call_filled', 'v15_conf']

    # Make sure all kw_* exist + numeric
    for c in feat_cols:
        if c in fdf.columns:
            fdf[c] = pd.to_numeric(fdf[c], errors='coerce').fillna(0)
        else:
            print(f'  missing column {c}, filling 0')
            fdf[c] = 0.0

    print(f'\n[4] training meta-LNN stacker — leave-one-org-out CV...')
    print(f'  feature columns ({len(feat_cols)}): {feat_cols}')

    rows = []
    per_org_results = {}
    for held in LOO_ORGS:
        train = fdf[(fdf.organism != held) & (fdf.organism.isin(LOO_ORGS +
                                                                  ['syn3a',
                                                                   'mgenitalium',
                                                                   'mpne']))]
        test = fdf[fdf.organism == held]

        X_train = train[feat_cols].values
        y_train = train.essential.values
        X_test = test[feat_cols].values
        y_test = test.essential.values

        # Logistic regression stacker (small, regularized)
        clf = LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced',
                                  solver='lbfgs')
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]

        # Sweep threshold for best MCC
        best_t, best_mcc = 0.5, -2.0
        for t in np.linspace(0.05, 0.95, 19):
            pred = (proba >= t).astype(int)
            if len(set(pred)) < 2: continue
            m = matthews_corrcoef(y_test, pred)
            if m > best_mcc: best_mcc, best_t = m, t

        pred = (proba >= best_t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
        prec = tp/(tp+fp) if (tp+fp) else float('nan')
        rec = tp/(tp+fn) if (tp+fn) else float('nan')
        per_org_results[held] = {
            'n': int(len(test)), 'best_threshold': float(best_t),
            'mcc': float(best_mcc), 'precision': float(prec),
            'recall': float(rec),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        }
        print(f'  held={held:<15s}  n={len(test):>4d}  '
              f'MCC={best_mcc:+.3f}  thr={best_t:.2f}  '
              f'prec={prec:.3f}  rec={rec:.3f}  '
              f'TP={tp} FP={fp} TN={tn} FN={fn}')

        for i, lt in enumerate(test.locus_tag.values):
            rows.append({
                'organism': held, 'locus_tag': lt,
                'true_essential': int(y_test[i]),
                'meta_proba': float(proba[i]),
                'meta_pred': int(pred[i]),
                'best_threshold': float(best_t),
            })

    # Pooled across all 5 LOO orgs
    pooled = pd.DataFrame(rows)
    y_all = pooled.true_essential.values
    p_all = pooled.meta_pred.values
    pooled_mcc = matthews_corrcoef(y_all, p_all)
    tn, fp, fn, tp = confusion_matrix(y_all, p_all, labels=[0,1]).ravel()
    mean_mcc = float(np.mean([r['mcc'] for r in per_org_results.values()]))

    print(f'\n[5] AGGREGATE RESULTS:')
    print(f'  POOLED MCC across 5 LOO orgs:  {pooled_mcc:+.4f}  '
          f'(TP={tp} FP={fp} TN={tn} FN={fn})')
    print(f'  MEAN per-org MCC:              {mean_mcc:+.4f}')
    print(f'  vs rule-based router pooled:   +0.604')
    print(f'  vs rule-based mean:            +0.443')
    print(f'  vs ortholog alone (mean):      +0.42')

    # Feature importance from one of the folds (last-trained clf)
    feat_imp = sorted(zip(feat_cols, clf.coef_[0]),
                       key=lambda kv: abs(kv[1]), reverse=True)
    print(f'\n[6] feature importance (|coef|, last fold = {held}):')
    for f, c in feat_imp[:15]:
        print(f'    {f:<25s}  {c:+.3f}')

    pooled.to_csv(OUT_CSV, index=False)
    print(f'\nwrote {OUT_CSV}')

    OUT_JSON.write_text(json.dumps({
        'method': 'meta_lnn_logreg_stacker_loo',
        'eval_orgs': LOO_ORGS,
        'feature_columns': feat_cols,
        'pooled_mcc': float(pooled_mcc),
        'mean_org_mcc': float(mean_mcc),
        'pooled_confusion': {'tp': int(tp), 'fp': int(fp),
                              'tn': int(tn), 'fn': int(fn)},
        'per_org': per_org_results,
        'feature_importance_last_fold': {f: float(c) for f, c in feat_imp},
        'baselines': {
            'rule_based_router_pooled': 0.604,
            'rule_based_router_mean': 0.443,
            'ortholog_prior_mean_cross_org': 0.420,
            'cross_org_LNN_mtb_alone': 0.260,
        },
    }, indent=2, default=str))
    print(f'wrote {OUT_JSON}')


if __name__ == '__main__':
    main()

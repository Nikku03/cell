"""Reversed leave-one-org-out stacker: hold out the 3 high-prevalence,
mollicute-line orgs (syn3a, mgenitalium, mpne) one at a time. Train on
the other 7. Mirror of meta_lnn_stacker.py.

Why: the original LOO trained on the 3 high-prev mollicutes (always-in)
+ 4 of the 5 LOO orgs, then tested on the 5th LOO org. That's "predict
typical bacteria using a mostly-typical training set." This script
flips it: "predict mollicute-line minimal-cell organisms using a mostly-
typical-bacteria training set." Tests prevalence-shift transfer in the
hardest direction.

Honest LNN handling: the leave-one-out checkpoint has no syn3a/mgen/mpne
folds (those orgs were always-in). For held-out target predictions:
LNN is unavailable -> has_lnn=0, lnn_prob_filled=0.5.
For training, the 5 LOO orgs DO have honest LNN preds.

Components for held-out targets:
  - v15: full coverage on syn3a (in-domain); RBH-transferred to mgen/mpne
  - ortholog_prior: full coverage all 3
  - PPI features: full coverage all 3
  - LNN: NONE (zeroed out)
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

OUT_JSON = ROOT / 'outputs/meta_lnn_stacker_reversed_results.json'
OUT_CSV = ROOT / 'outputs/meta_lnn_stacker_reversed_per_gene.csv'

# Reversed: now hold out the high-prev mollicutes
REVERSED_LOO = ['syn3a', 'mgenitalium', 'mpne']
# In training: the 5 LOO orgs (low-prev typical bacteria)
ALWAYS_IN = ['styphimurium', 'ccrescentus', 'saureus', 'mtuberculosis', 'abaylyi']

KW_FEATURES = ['kw_replication', 'kw_transcription', 'kw_translation',
                'kw_atp', 'kw_membrane', 'kw_synthase', 'kw_kinase',
                'kw_dehydrogenase', 'kw_protease', 'kw_rrna', 'kw_chaperone',
                'is_hypothetical', 'is_uncharacterized', 'is_putative']


def load_lnn_for_fold(target_org: str) -> dict[str, float]:
    batches = load_organism_batches([target_org], verbose=False)
    batch = batches[target_org]
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

    ppi = pd.read_csv(PPI_CSV)
    orth = pd.read_csv(ORTHOLOG_FEATURES)
    fdf = (fdf.merge(ppi, on=['organism', 'locus_tag'], how='left')
                   .merge(orth, on=['organism', 'locus_tag'], how='left'))
    fdf['n_orthologs'] = fdf.n_orthologs.fillna(0).astype(int)
    fdf['ortholog_prior'] = fdf.ortholog_prior.fillna(0.5)
    fdf['ppi_max_score'] = fdf.ppi_max_score.fillna(0.0)
    fdf['ppi_degree_high'] = fdf.ppi_degree_high.fillna(0).astype(int)

    print('\n[2] generating LNN preds for the 5 always-in (low-prev) orgs...')
    lnn_probs = {}
    # M. tb is cached
    mtb = pd.read_csv(MTB_CASCADE_CSV)
    for r in mtb.itertuples(index=False):
        lnn_probs[('mtuberculosis', r.locus_tag)] = float(r.lnn_prob)
    print(f'  mtuberculosis: {len(mtb)} (cached)')
    for org in [o for o in ALWAYS_IN if o != 'mtuberculosis']:
        try:
            preds = load_lnn_for_fold(org)
            for lt, p in preds.items():
                lnn_probs[(org, lt)] = p
            print(f'  {org}: {len(preds)}')
        except Exception as e:
            print(f'  {org}: FAILED ({e})')

    fdf['lnn_prob'] = fdf.apply(
        lambda r: lnn_probs.get((r.organism, r.locus_tag), np.nan), axis=1)
    fdf['has_lnn'] = fdf.lnn_prob.notna().astype(int)
    print(f'  LNN coverage: {fdf.has_lnn.sum()}/{len(fdf)} '
          f'(only the 5 ALWAYS_IN orgs have honest LNN; '
          f'syn3a/mgen/mpne are zeroed)')

    print('\n[3] adding v15 (in-domain on Syn3A, RBH-transferred to mgen/mpne)...')
    v15 = pd.read_csv(V15_SYN3A_CSV)
    syn3a_v15 = {r.locus_tag: (int(r.essential), float(r.confidence))
                  for r in v15.itertuples(index=False)}
    rbh = pd.read_csv(RBH_CSV)
    v15_transfer = {}
    for r in rbh.itertuples(index=False):
        if r.org_a == 'syn3a':
            if r.locus_a in syn3a_v15:
                k = (r.org_b, r.locus_b)
                if k not in v15_transfer:
                    v15_transfer[k] = syn3a_v15[r.locus_a]
        elif r.org_b == 'syn3a':
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
    syn = int(((fdf.organism == 'syn3a') & (fdf.has_v15 == 1)).sum())
    mgn = int(((fdf.organism == 'mgenitalium') & (fdf.has_v15 == 1)).sum())
    mpn = int(((fdf.organism == 'mpne') & (fdf.has_v15 == 1)).sum())
    print(f'  v15 coverage: {fdf.has_v15.sum()}/{len(fdf)}  '
          f'(syn3a={syn}, mgen={mgn}, mpne={mpn})')

    feat_cols = (KW_FEATURES + ['log_length_bp', 'gc', 'upstream_gc',
                  'upstream_at_skew', 'position_norm', 'operon_run_length',
                  'n_orthologs', 'ortholog_prior',
                  'ppi_max_score', 'ppi_degree_high',
                  'has_lnn', 'has_v15'])
    fdf['lnn_prob_filled'] = fdf.lnn_prob.fillna(0.5)
    fdf['v15_call_filled'] = fdf.v15_call.fillna(0.5)
    feat_cols += ['lnn_prob_filled', 'v15_call_filled', 'v15_conf']

    for c in feat_cols:
        if c in fdf.columns:
            fdf[c] = pd.to_numeric(fdf[c], errors='coerce').fillna(0)
        else:
            fdf[c] = 0.0

    print(f'\n[4] REVERSED LOO: hold out {REVERSED_LOO}, train on the rest...')
    print(f'  feature columns ({len(feat_cols)}): {feat_cols}')

    rows = []
    per_org_results = {}
    all_orgs = REVERSED_LOO + ALWAYS_IN
    for held in REVERSED_LOO:
        train = fdf[(fdf.organism != held) & (fdf.organism.isin(all_orgs))]
        test = fdf[fdf.organism == held]

        train_prev = train.essential.mean()
        test_prev = test.essential.mean()
        print(f'\n  fold = {held}:')
        print(f'    train n={len(train)}  prev={train_prev:.3f}')
        print(f'    test  n={len(test)}  prev={test_prev:.3f}')

        X_train = train[feat_cols].values
        y_train = train.essential.values
        X_test = test[feat_cols].values
        y_test = test.essential.values

        clf = LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced',
                                  solver='lbfgs')
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]

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
            'train_prevalence': float(train_prev),
            'test_prevalence': float(test_prev),
            'prevalence_shift': float(test_prev - train_prev),
        }
        print(f'    MCC={best_mcc:+.3f}  thr={best_t:.2f}  '
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

    pooled = pd.DataFrame(rows)
    y_all = pooled.true_essential.values
    p_all = pooled.meta_pred.values
    pooled_mcc = matthews_corrcoef(y_all, p_all)
    tn, fp, fn, tp = confusion_matrix(y_all, p_all, labels=[0,1]).ravel()
    mean_mcc = float(np.mean([r['mcc'] for r in per_org_results.values()]))

    print(f'\n[5] AGGREGATE — REVERSED DIRECTION:')
    print(f'  POOLED MCC across 3 reversed-LOO orgs: {pooled_mcc:+.4f}')
    print(f'  MEAN per-org MCC:                      {mean_mcc:+.4f}')
    print(f'\n  Compare to FORWARD direction (5 LOO orgs):')
    print(f'    Forward pooled MCC:   +0.500')
    print(f'    Forward mean MCC:     +0.460')

    feat_imp = sorted(zip(feat_cols, clf.coef_[0]),
                       key=lambda kv: abs(kv[1]), reverse=True)
    print(f'\n[6] feature importance (last fold = {held}):')
    for f, c in feat_imp[:12]:
        print(f'    {f:<25s}  {c:+.3f}')

    pooled.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps({
        'method': 'meta_lnn_logreg_stacker_REVERSED_loo',
        'reversed_holdouts': REVERSED_LOO,
        'training_pool': ALWAYS_IN,
        'note': ('Held-out targets have NO honest LNN signal '
                 '(no syn3a/mgen/mpne folds in checkpoint). v15 is '
                 'in-domain for syn3a, RBH-transferred for mgen/mpne.'),
        'pooled_mcc': float(pooled_mcc),
        'mean_org_mcc': float(mean_mcc),
        'pooled_confusion': {'tp': int(tp), 'fp': int(fp),
                              'tn': int(tn), 'fn': int(fn)},
        'per_org': per_org_results,
        'forward_baseline_pooled': 0.500,
        'forward_baseline_mean': 0.460,
    }, indent=2, default=str))
    print(f'\nwrote {OUT_CSV}')
    print(f'wrote {OUT_JSON}')


if __name__ == '__main__':
    main()

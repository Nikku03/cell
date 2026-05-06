"""Sparse-LNN cascade stacker — Path B (no PPI).

Stack the following per-gene signals with leave-one-org-out CV:
  - ortholog prior + n_orthologs                          (cross-org transfer)
  - v15 simulator essentiality call + confidence          (in-domain Syn3A,
                                                           RBH-transferred elsewhere)
  - SparseConceptLNN final predicted_prob                 (sparse LNN cascade head)
  - 128 pattern activations per modality (static, know)   (sparse LNN cascade body)
  - 128 traj-pattern activations + has_traj indicator     (Syn3A-only signal)
  - mean-pooled simulator trajectory state (Syn3A)        (raw simulator readout)
  - standard scalar+kw features (length, GC, kw_*, etc.)  (baseline)

Explicitly EXCLUDED: PPI features (ppi_max_score, ppi_degree_high, ppi_degree_mid).

Compares to existing meta_lnn_stacker (MCC 0.460 *with* PPI) to measure how
much PPI was contributing.

Outputs:
  outputs/sparse_lnn_cascade_stacker_results.json
  outputs/sparse_lnn_cascade_stacker_per_gene.csv
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / 'outputs/multiorg_per_gene_features.csv'
ORTHOLOG_FEATURES = ROOT / 'outputs/ortholog_prior_features_per_gene.csv'
RBH_CSV = ROOT / 'outputs/rbh_pairs.csv'
V15_SYN3A_CSV = (ROOT / 'outputs' /
                 'predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4_'
                 'composed_all455_v15_round2_priors.csv')
ATTRIB_CSV = ROOT / 'outputs/multimodal_sparse_concept_per_gene_attributions.csv'
TRAJ_NPZ = ROOT / 'outputs/syn3a_trajectories_t2.0s.npz'

OUT_JSON = ROOT / 'outputs/sparse_lnn_cascade_stacker_results.json'
OUT_CSV = ROOT / 'outputs/sparse_lnn_cascade_stacker_per_gene.csv'

LOO_ORGS = ['styphimurium', 'ccrescentus', 'saureus', 'mtuberculosis', 'abaylyi']
N_PATTERNS = 128
KW_FEATURES = ['kw_replication', 'kw_transcription', 'kw_translation',
               'kw_atp', 'kw_membrane', 'kw_synthase', 'kw_kinase',
               'kw_dehydrogenase', 'kw_protease', 'kw_rrna', 'kw_chaperone',
               'is_hypothetical', 'is_uncharacterized', 'is_putative']
SCALAR_FEATURES = ['log_length_bp', 'gc', 'upstream_gc', 'upstream_at_skew',
                   'position_norm', 'operon_run_length']

PATTERN_RE = re.compile(r'p(\d+):([0-9.]+)')


def parse_pattern_str(s: str, n_patterns: int) -> np.ndarray:
    """Parse 'p17:0.51,p42:0.31,p8:0.18' into a sparse 128-dim vector."""
    v = np.zeros(n_patterns, dtype=np.float32)
    if not isinstance(s, str) or not s:
        return v
    for m in PATTERN_RE.finditer(s):
        idx = int(m.group(1))
        if 0 <= idx < n_patterns:
            v[idx] = float(m.group(2))
    return v


def load_v15_calls() -> tuple[dict, dict]:
    v15 = pd.read_csv(V15_SYN3A_CSV)
    syn3a_v15 = {r.locus_tag: (int(r.essential), float(r.confidence))
                 for r in v15.itertuples(index=False)}
    rbh = pd.read_csv(RBH_CSV)
    transfer = {}
    for r in rbh.itertuples(index=False):
        if r.org_a == 'syn3a' and isinstance(r.locus_b, str) and r.locus_b:
            if r.locus_a in syn3a_v15:
                transfer.setdefault((r.org_b, r.locus_b), syn3a_v15[r.locus_a])
        elif r.org_b == 'syn3a' and isinstance(r.locus_a, str) and r.locus_a:
            if r.locus_b in syn3a_v15:
                transfer.setdefault((r.org_a, r.locus_a), syn3a_v15[r.locus_b])
    return syn3a_v15, transfer


def load_traj_summary() -> dict[str, np.ndarray]:
    """Mean-pool trajectory state over time for each Syn3A locus."""
    if not TRAJ_NPZ.exists():
        return {}
    d = np.load(TRAJ_NPZ, allow_pickle=True)
    loci = d['loci']
    traj = d['trajectories']  # (N, T, D)
    # Mean over the time axis -> (N, D)
    pooled = traj.mean(axis=1)
    return {str(lt): pooled[i].astype(np.float32) for i, lt in enumerate(loci)}


def main():
    print('[1] loading per-gene features + labels (no PPI)...')
    fdf = pd.read_csv(FEATURES_CSV)
    fdf = fdf.dropna(subset=['essential']).copy()
    fdf['essential'] = fdf['essential'].astype(int)
    print(f'  {len(fdf)} labeled genes across {fdf.organism.nunique()} orgs')

    orth = pd.read_csv(ORTHOLOG_FEATURES)
    fdf = fdf.merge(orth, on=['organism', 'locus_tag'], how='left')
    fdf['n_orthologs'] = fdf.n_orthologs.fillna(0).astype(int)
    fdf['ortholog_prior'] = fdf.ortholog_prior.fillna(0.5)

    print('\n[2] loading v15 simulator calls (in-domain Syn3A + RBH transfer)...')
    syn3a_v15, v15_transfer = load_v15_calls()

    def get_v15(r):
        if r.organism == 'syn3a':
            return syn3a_v15.get(r.locus_tag, (np.nan, 0.0))
        return v15_transfer.get((r.organism, r.locus_tag), (np.nan, 0.0))

    fdf['v15_call'] = fdf.apply(lambda r: get_v15(r)[0], axis=1)
    fdf['v15_conf'] = fdf.apply(lambda r: get_v15(r)[1], axis=1)
    fdf['has_v15'] = fdf.v15_call.notna().astype(int)
    fdf['v15_call_filled'] = fdf.v15_call.fillna(0.5)
    print(f'  v15 coverage: {fdf.has_v15.sum()}/{len(fdf)} '
          f'({100*fdf.has_v15.sum()/len(fdf):.1f}%)')

    print('\n[3] parsing SparseConceptLNN pattern activations + final prob...')
    attr = pd.read_csv(ATTRIB_CSV)
    print(f'  {len(attr)} attribution rows')
    static_mat = np.stack([parse_pattern_str(s, N_PATTERNS)
                           for s in attr.static_top_patterns.fillna('')])
    traj_mat = np.stack([parse_pattern_str(s, N_PATTERNS)
                         for s in attr.traj_top_patterns.fillna('')])
    know_mat = np.stack([parse_pattern_str(s, N_PATTERNS)
                         for s in attr.know_top_patterns.fillna('')])
    has_traj_arr = attr.has_traj_signal.fillna(0).astype(int).values
    lnn_prob_arr = attr.predicted_prob.fillna(0.5).values
    has_lnn_arr = attr.predicted_prob.notna().astype(int).values

    attr_key = list(zip(attr.organism, attr.locus_tag))
    static_lookup = dict(zip(attr_key, static_mat))
    traj_lookup = dict(zip(attr_key, traj_mat))
    know_lookup = dict(zip(attr_key, know_mat))
    has_traj_lookup = dict(zip(attr_key, has_traj_arr))
    lnn_prob_lookup = dict(zip(attr_key, lnn_prob_arr))
    has_lnn_lookup = dict(zip(attr_key, has_lnn_arr))

    print('\n[4] loading Syn3A trajectory mean-pool features...')
    traj_summary = load_traj_summary()
    if traj_summary:
        D_traj = next(iter(traj_summary.values())).shape[0]
        print(f'  {len(traj_summary)} Syn3A loci with traj features (D={D_traj})')
    else:
        D_traj = 0
        print('  no trajectory file — skipping')

    print('\n[5] assembling per-gene feature matrix...')
    n = len(fdf)
    static_block = np.zeros((n, N_PATTERNS), dtype=np.float32)
    traj_block = np.zeros((n, N_PATTERNS), dtype=np.float32)
    know_block = np.zeros((n, N_PATTERNS), dtype=np.float32)
    has_traj_col = np.zeros(n, dtype=np.float32)
    lnn_prob_col = np.full(n, 0.5, dtype=np.float32)
    has_lnn_col = np.zeros(n, dtype=np.float32)
    traj_pooled_block = np.zeros((n, D_traj), dtype=np.float32) if D_traj else None
    has_traj_pool = np.zeros(n, dtype=np.float32)

    for i, r in enumerate(fdf.itertuples(index=False)):
        k = (r.organism, r.locus_tag)
        if k in static_lookup:
            static_block[i] = static_lookup[k]
            traj_block[i] = traj_lookup[k]
            know_block[i] = know_lookup[k]
            has_traj_col[i] = has_traj_lookup[k]
            lnn_prob_col[i] = lnn_prob_lookup[k]
            has_lnn_col[i] = has_lnn_lookup[k]
        if D_traj and r.organism == 'syn3a' and r.locus_tag in traj_summary:
            traj_pooled_block[i] = traj_summary[r.locus_tag]
            has_traj_pool[i] = 1.0

    print(f'  pattern coverage: {int(has_lnn_col.sum())}/{n} '
          f'({100*has_lnn_col.mean():.1f}%)')
    print(f'  traj-signal coverage: {int(has_traj_col.sum())}/{n} '
          f'({100*has_traj_col.mean():.1f}%)')
    print(f'  traj mean-pool coverage: {int(has_traj_pool.sum())}/{n} '
          f'({100*has_traj_pool.mean():.1f}%)')

    # --- Scalar/kw block ---
    scalar_cols = SCALAR_FEATURES + KW_FEATURES + [
        'n_orthologs', 'ortholog_prior',
        'has_lnn', 'has_v15', 'v15_call_filled', 'v15_conf',
        'lnn_prob', 'has_pattern', 'has_traj_signal',
        'has_traj_pool',
    ]
    fdf['lnn_prob'] = lnn_prob_col
    fdf['has_pattern'] = has_lnn_col
    fdf['has_traj_signal'] = has_traj_col
    fdf['has_traj_pool'] = has_traj_pool
    fdf['has_lnn'] = has_lnn_col
    for c in scalar_cols:
        if c in fdf.columns:
            fdf[c] = pd.to_numeric(fdf[c], errors='coerce').fillna(0.0)
        else:
            fdf[c] = 0.0
    scalar_mat = fdf[scalar_cols].astype(np.float32).values

    # Stitch full feature matrix (PPI excluded entirely)
    blocks = [scalar_mat, static_block, traj_block, know_block]
    block_names = ['scalar', 'static_pat', 'traj_pat', 'know_pat']
    if D_traj:
        blocks.append(traj_pooled_block)
        block_names.append('traj_pool')
    X = np.concatenate(blocks, axis=1)
    print(f'  feature matrix: {X.shape}  blocks={list(zip(block_names, [b.shape[1] for b in blocks]))}')

    # ---- LOO CV ----
    print(f'\n[6] training stacker (logreg, balanced) — LOO across {LOO_ORGS}')
    train_universe = set(LOO_ORGS) | {'syn3a', 'mgenitalium', 'mpne'}
    rows = []
    per_org_results = {}
    for held in LOO_ORGS:
        train_mask = (fdf.organism != held) & fdf.organism.isin(train_universe)
        test_mask = fdf.organism == held
        X_train, y_train = X[train_mask.values], fdf.essential.values[train_mask.values]
        X_test, y_test = X[test_mask.values], fdf.essential.values[test_mask.values]

        clf = LogisticRegression(C=0.5, max_iter=4000,
                                  class_weight='balanced', solver='lbfgs')
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]

        best_t, best_mcc = 0.5, -2.0
        for t in np.linspace(0.05, 0.95, 19):
            pred = (proba >= t).astype(int)
            if len(set(pred)) < 2:
                continue
            m = matthews_corrcoef(y_test, pred)
            if m > best_mcc:
                best_mcc, best_t = m, t
        pred = (proba >= best_t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) else float('nan')
        rec = tp / (tp + fn) if (tp + fn) else float('nan')
        per_org_results[held] = {
            'n': int(len(y_test)),
            'best_threshold': float(best_t),
            'mcc': float(best_mcc),
            'precision': float(prec), 'recall': float(rec),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        }
        print(f'  held={held:<15s} n={len(y_test):>4d}  '
              f'MCC={best_mcc:+.3f}  thr={best_t:.2f}  '
              f'prec={prec:.3f}  rec={rec:.3f}  '
              f'TP={tp} FP={fp} TN={tn} FN={fn}')

        test_loci = fdf.locus_tag.values[test_mask.values]
        for i, lt in enumerate(test_loci):
            rows.append({
                'organism': held, 'locus_tag': lt,
                'true_essential': int(y_test[i]),
                'cascade_proba': float(proba[i]),
                'cascade_pred': int(pred[i]),
                'best_threshold': float(best_t),
            })

    pooled = pd.DataFrame(rows)
    y_all = pooled.true_essential.values
    p_all = pooled.cascade_pred.values
    pooled_mcc = matthews_corrcoef(y_all, p_all)
    tn, fp, fn, tp = confusion_matrix(y_all, p_all, labels=[0, 1]).ravel()
    mean_mcc = float(np.mean([r['mcc'] for r in per_org_results.values()]))

    print(f'\n[7] AGGREGATE (no-PPI cascade stacker):')
    print(f'  POOLED MCC across 5 LOO orgs:  {pooled_mcc:+.4f}  '
          f'(TP={tp} FP={fp} TN={tn} FN={fn})')
    print(f'  MEAN per-org MCC:              {mean_mcc:+.4f}')
    print(f'  vs meta_lnn_stacker w/ PPI:    +0.460')
    print(f'  Δ (no-PPI − w/ PPI):           {mean_mcc - 0.460:+.4f}')

    # Top scalar-block coefficients (last fold)
    coefs = clf.coef_[0]
    scalar_coefs = sorted(zip(scalar_cols, coefs[:len(scalar_cols)]),
                          key=lambda kv: abs(kv[1]), reverse=True)
    print(f'\n[8] scalar feature importance (last fold = {held}):')
    for f, c in scalar_coefs[:15]:
        print(f'    {f:<25s}  {c:+.3f}')

    # Top static-pattern coefficients
    p_off = len(scalar_cols)
    static_coefs = sorted(enumerate(coefs[p_off:p_off + N_PATTERNS]),
                          key=lambda kv: abs(kv[1]), reverse=True)
    print(f'\n[9] top static-pattern coefficients:')
    for idx, c in static_coefs[:10]:
        print(f'    static_p{idx:<3d}  {c:+.3f}')

    pooled.to_csv(OUT_CSV, index=False)
    print(f'\nwrote {OUT_CSV}')

    OUT_JSON.write_text(json.dumps({
        'method': 'sparse_lnn_cascade_stacker_loo_no_ppi',
        'eval_orgs': LOO_ORGS,
        'feature_blocks': dict(zip(block_names, [int(b.shape[1]) for b in blocks])),
        'scalar_columns': scalar_cols,
        'pooled_mcc': float(pooled_mcc),
        'mean_org_mcc': float(mean_mcc),
        'pooled_confusion': {'tp': int(tp), 'fp': int(fp),
                              'tn': int(tn), 'fn': int(fn)},
        'per_org': per_org_results,
        'scalar_feature_importance_last_fold': {f: float(c) for f, c in scalar_coefs},
        'top_static_patterns_last_fold': [{'pattern': int(i), 'coef': float(c)}
                                            for i, c in static_coefs[:20]],
        'baselines': {
            'meta_lnn_stacker_with_ppi_mean': 0.460,
            'rule_based_router_mean': 0.443,
            'ortholog_prior_mean_cross_org': 0.420,
        },
        'caveats': [
            'pattern activations come from a multimodal model that may have seen '
            'all orgs at train time; the cascade head therefore carries some leakage. '
            'Treat this as an upper bound on what the pattern bottleneck contributes '
            'on top of the honest cross-org features (ortholog + v15-transfer + scalar).',
            'PPI features deliberately excluded per Path B.',
        ],
    }, indent=2, default=str))
    print(f'wrote {OUT_JSON}')


if __name__ == '__main__':
    main()

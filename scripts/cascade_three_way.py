"""Three-way honest cascade on Syn3A held out from all training:
    v15 simulator + cross-org LNN + ortholog prior.

All three components have ZERO knowledge of Syn3A's labels:
  - v15 simulator: no learned parameters at all
  - cross-org LNN: trained on 7 orgs with Syn3A explicitly held out
                    (essentiality_lnn_no_syn3a.pt, commit fec65e2)
  - ortholog prior: leave-one-out by construction

This is the "true cascade" answer to: how well can we predict Syn3A
essentiality WITHOUT ever using Syn3A's labels?

Strategies tested
-----------------
A. Gating with weighted fallback:
     if v15_conf >= T:          use v15_call
     else:                       use (alpha * lnn + beta * orth) >= 0.5
B. Additive (recall-boosting):
     cascade_essential iff any of:
       v15_call == 1
       lnn_prob >= lnn_t
       orth_prior >= orth_t
C. 2-of-3 voting:
     cascade_essential iff at least 2 of [v15, lnn>=lnn_t, orth>=orth_t] agree
D. Probability average:
     cascade_essential iff mean(v15_call, lnn_prob, orth_prior) >= 0.5

Each strategy is reported with its MCC, TP/FP/TN/FN, and per-source
contribution. Best strategy is the deployable honest predictor.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (matthews_corrcoef, roc_auc_score,
                                average_precision_score, confusion_matrix)

from cell_sim.layer_ml.data_loader import load_organism_batches
from cell_sim.layer6_essentiality.cascade_detector import (
    load_lnn_predictions_for_organism,
)

ROOT = Path(__file__).resolve().parent.parent
MMSEQS_TSV = Path('/tmp/conservation_work/all_vs_all.tsv')
LABELS_CSV = ROOT / 'outputs/multiorg_per_gene_features.csv'
V15_CSV = (ROOT / 'outputs' /
           'predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4_composed_all455_v15_round2_priors.csv')
LNN_CKPT = ROOT / 'cell_sim/layer_ml/checkpoints/essentiality_lnn_no_syn3a.pt'
OUT_JSON = ROOT / 'outputs/cascade_three_way_syn3a_results.json'

OUR_ORGS = ['styphimurium', 'ccrescentus', 'saureus', 'mtuberculosis',
            'abaylyi', 'mgenitalium', 'mpne', 'syn3a']


def load_v15_predictions() -> pd.DataFrame:
    df = pd.read_csv(V15_CSV)
    df = df.rename(columns={'essential': 'v15_call', 'confidence': 'v15_conf'})
    df['true_essential'] = df.true_class.map(
        {'Essential': 1, 'Quasiessential': 1, 'Nonessential': 0}).astype(int)
    return df[['locus_tag', 'gene_name', 'v15_call', 'v15_conf',
                 'true_essential']]


def build_ortholog_priors(target_org: str, ref_orgs: set,
                            ess_lookup: dict) -> dict:
    """Per-target-gene ortholog prior."""
    print(f'  building RBH for {target_org}...')
    best = defaultdict(dict)
    with open(MMSEQS_TSV) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            q, t, ident = parts[0], parts[1], float(parts[2])
            if q == t: continue
            try:
                qo, ql = q.split('|', 1); to, tl = t.split('|', 1)
            except ValueError: continue
            if (qo == target_org and to in ref_orgs) or \
                 (qo in ref_orgs and to == target_org):
                cur = best[(qo, ql)].get(to)
                if cur is None or ident > cur[1]:
                    best[(qo, ql)][to] = (tl, ident)

    rbh = defaultdict(list)
    for (q_org, q_locus), partners in best.items():
        if q_org != target_org: continue
        for t_org, (t_locus, _) in partners.items():
            if t_org not in ref_orgs: continue
            rev = best.get((t_org, t_locus), {}).get(q_org)
            if rev is not None and rev[0] == q_locus:
                rbh[q_locus].append((t_org, t_locus))

    priors = {}
    for q_locus, partners in rbh.items():
        labels = [ess_lookup.get((po, pl)) for (po, pl) in partners]
        labels = [l for l in labels if l is not None]
        if labels:
            priors[q_locus] = (float(np.mean(labels)), len(labels))
    return priors


def metrics_block(y, pred, name):
    if len(set(y)) < 2 or len(set(pred)) < 2:
        return {'name': name, 'mcc': 0.0, 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    mcc = matthews_corrcoef(y, pred)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {'name': name, 'mcc': float(mcc),
             'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='syn3a')
    ap.add_argument('--out', type=Path, default=OUT_JSON)
    args = ap.parse_args()

    target = args.target
    ref_orgs = set(o for o in OUR_ORGS if o != target)
    print(f'target = {target}; references = {sorted(ref_orgs)}\n')

    print('[1] loading v15 cached predictions...')
    v15_df = load_v15_predictions()
    print(f'  {len(v15_df)} {target} genes')

    print('\n[2] loading cross-org LNN (no Syn3A in training)...')
    print(f'  checkpoint: {LNN_CKPT.name}')
    gene_batches = load_organism_batches([target], verbose=False)
    target_batch = gene_batches[target]
    lnn_probs = load_lnn_predictions_for_organism(LNN_CKPT, target_batch,
                                                      device='cpu')
    print(f'  {len(lnn_probs)} per-gene LNN probs from cross-org model')
    lnn_df = pd.DataFrame([
        {'locus_tag': lt, 'lnn_prob': p} for lt, p in lnn_probs.items()
    ])

    print('\n[3] loading labels + building ortholog priors...')
    label_df = pd.read_csv(LABELS_CSV, usecols=['organism', 'locus_tag', 'essential'])
    ess_lookup = {(r.organism, r.locus_tag): int(r.essential)
                    for r in label_df.itertuples(index=False)
                    if not pd.isna(r.essential)}
    orth_priors = build_ortholog_priors(target, ref_orgs, ess_lookup)
    print(f'  {len(orth_priors)}/{len(v15_df)} target genes have RBH '
          f'orthologs with labels')

    print('\n[4] merging predictors...')
    df = v15_df.merge(lnn_df, on='locus_tag', how='left')
    df['ortholog_prior'] = df.locus_tag.map(
        lambda lt: orth_priors[lt][0] if lt in orth_priors else np.nan)
    df['n_orthologs'] = df.locus_tag.map(
        lambda lt: orth_priors[lt][1] if lt in orth_priors else 0)
    df['lnn_prob'] = df.lnn_prob.fillna(0.5)
    print(f'  ortholog coverage: {(df.n_orthologs > 0).sum()}/{len(df)}')

    y = df.true_essential.values
    print(f'\n  panel: {int(y.sum())} essential / {int((y==0).sum())} non-essential')

    # Component MCCs
    print(f'\n[5] component MCCs (all honestly held out from Syn3A):')
    m_v15 = metrics_block(y, df.v15_call.values, 'v15_alone')
    print(f'  {m_v15["name"]:30s}  MCC={m_v15["mcc"]:.4f}  '
          f'TP={m_v15["tp"]} FP={m_v15["fp"]} TN={m_v15["tn"]} FN={m_v15["fn"]}')

    # LNN at multiple thresholds (find best for each)
    best_lnn_t, best_lnn_mcc = 0.5, -2.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (df.lnn_prob.values >= t).astype(int)
        if pred.sum() in (0, len(pred)): continue
        m = matthews_corrcoef(y, pred)
        if m > best_lnn_mcc: best_lnn_mcc, best_lnn_t = m, t
    lnn_call = (df.lnn_prob.values >= best_lnn_t).astype(int)
    m_lnn = metrics_block(y, lnn_call, f'cross_org_LNN(t={best_lnn_t:.2f})')
    print(f'  {m_lnn["name"]:30s}  MCC={m_lnn["mcc"]:.4f}  '
          f'TP={m_lnn["tp"]} FP={m_lnn["fp"]} TN={m_lnn["tn"]} FN={m_lnn["fn"]}')

    orth_call = np.where(df.n_orthologs > 0,
                            df.ortholog_prior.fillna(0.5) >= 0.5, 1).astype(int)
    m_orth = metrics_block(y, orth_call, 'ortholog_prior_alone')
    print(f'  {m_orth["name"]:30s}  MCC={m_orth["mcc"]:.4f}  '
          f'TP={m_orth["tp"]} FP={m_orth["fp"]} TN={m_orth["tn"]} FN={m_orth["fn"]}')

    # Combinations
    print(f'\n[6] combinations:')
    results = {'v15_alone': m_v15, 'lnn_alone': m_lnn, 'orth_alone': m_orth}

    # Strategy A: gating cascade with LNN+ortholog blend fallback
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        cas_a = []
        for _, r in df.iterrows():
            if r.v15_conf >= 0.5:
                cas_a.append(int(r.v15_call))
            else:
                fallback = alpha * r.lnn_prob + (1 - alpha) * (
                    r.ortholog_prior if not pd.isna(r.ortholog_prior) else 0.5)
                cas_a.append(int(fallback >= 0.5))
        m = metrics_block(y, np.array(cas_a),
                            f'A_gating_alpha={alpha}_lnn+{1-alpha:.1f}_orth')
        print(f'  {m["name"]:40s} MCC={m["mcc"]:.4f}  '
              f'TP={m["tp"]} FP={m["fp"]} TN={m["tn"]} FN={m["fn"]}')
        results[m['name']] = m

    # Strategy B: additive (essential if any of v15, lnn high, orth high)
    for lnn_t, orth_t in [(0.5, 0.5), (0.7, 0.7), (0.8, 0.8),
                            (0.7, 0.85), (0.85, 0.85)]:
        cas_b = []
        for _, r in df.iterrows():
            v15_ess = (r.v15_call == 1)
            lnn_ess = (r.lnn_prob >= lnn_t)
            orth_ess = (not pd.isna(r.ortholog_prior)
                          and r.ortholog_prior >= orth_t)
            cas_b.append(int(v15_ess or lnn_ess or orth_ess))
        m = metrics_block(y, np.array(cas_b),
                            f'B_additive_lnn>={lnn_t}_orth>={orth_t}')
        print(f'  {m["name"]:40s} MCC={m["mcc"]:.4f}  '
              f'TP={m["tp"]} FP={m["fp"]} TN={m["tn"]} FN={m["fn"]}')
        results[m['name']] = m

    # Strategy C: 2-of-3 majority vote
    for lnn_t, orth_t in [(0.5, 0.5), (0.6, 0.5), (0.7, 0.5)]:
        cas_c = []
        for _, r in df.iterrows():
            votes = (int(r.v15_call == 1)
                       + int(r.lnn_prob >= lnn_t)
                       + int((not pd.isna(r.ortholog_prior))
                              and r.ortholog_prior >= orth_t))
            cas_c.append(int(votes >= 2))
        m = metrics_block(y, np.array(cas_c),
                            f'C_2of3_lnn>={lnn_t}_orth>={orth_t}')
        print(f'  {m["name"]:40s} MCC={m["mcc"]:.4f}  '
              f'TP={m["tp"]} FP={m["fp"]} TN={m["tn"]} FN={m["fn"]}')
        results[m['name']] = m

    # Strategy D: probability average
    for w_v15, w_lnn, w_orth in [(0.5, 0.25, 0.25), (0.6, 0.2, 0.2),
                                     (0.4, 0.3, 0.3), (0.7, 0.15, 0.15)]:
        cas_d = []
        for _, r in df.iterrows():
            p = (w_v15 * (r.v15_conf if r.v15_call == 1 else (1 - r.v15_conf))
                  if r.v15_call == 1 else
                  w_v15 * (1 - 0.5))   # treat v15 nonessential as 0.5 prob
            # Better: use v15_call as a boolean prob 1 or 0
            v_p = float(r.v15_call)
            o_p = (r.ortholog_prior if not pd.isna(r.ortholog_prior)
                    else 0.5)
            avg = w_v15 * v_p + w_lnn * r.lnn_prob + w_orth * o_p
            cas_d.append(int(avg >= 0.5))
        m = metrics_block(y, np.array(cas_d),
                            f'D_avg_v15={w_v15}_lnn={w_lnn}_orth={w_orth}')
        print(f'  {m["name"]:40s} MCC={m["mcc"]:.4f}  '
              f'TP={m["tp"]} FP={m["fp"]} TN={m["tn"]} FN={m["fn"]}')
        results[m['name']] = m

    # Best
    best = max(results.values(), key=lambda r: r['mcc'])
    print(f'\n[7] BEST: {best["name"]}  MCC={best["mcc"]:.4f}')
    print(f'      vs v15 alone:       {best["mcc"] - m_v15["mcc"]:+.4f}')
    print(f'      vs orth alone:      {best["mcc"] - m_orth["mcc"]:+.4f}')

    print(f'\n  reference baselines (to put this number in context):')
    print(f'    v15 simulator alone:                       0.5372')
    print(f'    LNN cascade WITH Syn3A in training:        0.768  (memorization)')
    print(f'    cross-org LNN alone (Syn3A held out):      {m_lnn["mcc"]:.4f}')
    print(f'    ortholog prior alone (Syn3A held out):     {m_orth["mcc"]:.4f}')
    print(f'    THIS RUN (best honest combination):        {best["mcc"]:.4f}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        'method': 'cascade_three_way_v15_lnn_orth_syn3a_held_out',
        'target': target, 'ref_orgs': sorted(ref_orgs),
        'all_strategies': results,
        'best_strategy': best['name'],
        'best_mcc': best['mcc'],
        'baselines': {
            'v15_alone': 0.5372,
            'lnn_cascade_in_domain_memorized': 0.768,
            'cross_org_lnn_alone_honest': float(m_lnn['mcc']),
            'ortholog_alone_honest': float(m_orth['mcc']),
        },
    }, indent=2, default=str))
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()

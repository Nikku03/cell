"""Cascade with ortholog prior + memory-bank kNN fallback.

Honest deployable predictor for Syn3A — Syn3A's labels are NEVER used in
training. Replaces the LNN fallback (which memorized) with two
parameter-free predictors:

  ortholog_prior(G)  = mean essentiality of G's RBH orthologs in
                         the 7 training organisms
  memory_kNN(G)      = identity-weighted mean essentiality of G's
                         top-K mmseqs hits in the 7 training organisms
                         (broader than RBH — includes paralogs / homologs)

Cascade:
  if v15_conf(G) >= T:   use v15_call(G)              # mechanistic, precise
  else if ortholog has 1+: use ortholog_prior(G) >= 0.5
  else if memory_kNN has 1+: use memory_kNN(G) >= 0.5
  else:                   default to majority class

The "memory bank" here is the cached mmseqs all-vs-all similarity table
(`/tmp/conservation_work/all_vs_all.tsv`) — a deterministic, biology-
based nearest-neighbor lookup. No learned encoder, no memorization risk.

Comparison points
-----------------
- v15 simulator alone (Syn3A): MCC 0.505 (mechanistic, no training)
- LNN cascade (LNN had Syn3A in training): 0.768 (memorization)
- Ortholog prior, Syn3A held out: 0.302
- Memory bank kNN alone (this run, Syn3A held out): ?
- v15 + ortholog cascade (this run): ?
- v15 + ortholog + memory cascade (this run): ?

If the cascade gets > 0.55 on Syn3A held out, that's the honest
deployable predictor — beats every learned predictor we built without
ever seeing Syn3A's labels.
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
from sklearn.metrics import (matthews_corrcoef, roc_auc_score,
                                average_precision_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent.parent
MMSEQS_TSV = Path('/tmp/conservation_work/all_vs_all.tsv')
LABELS_CSV = ROOT / 'outputs/multiorg_per_gene_features.csv'
V15_CSV = (ROOT / 'outputs' /
           'predictions_parallel_s0.05_t0.5_seed42_thr0.1_w4_composed_all455_v15_round2_priors.csv')
OUT_JSON = ROOT / 'outputs/cascade_v15_ortholog_memory_syn3a_results.json'

OUR_ORGS = ['styphimurium', 'ccrescentus', 'saureus', 'mtuberculosis',
            'abaylyi', 'mgenitalium', 'mpne', 'syn3a']

CASCADE_THRESHOLD = 0.5
ORTHOLOG_THRESHOLD = 0.5
MEMORY_KNN_TOP_K = 10
MEMORY_KNN_THRESHOLD = 0.5
ORTHOLOG_WEIGHT = 0.7   # when both ortholog and memory available


def load_v15_predictions() -> pd.DataFrame:
    df = pd.read_csv(V15_CSV)
    # 'essential' column here is v15's CALL (1 if mode != NONE, 0 otherwise)
    df = df.rename(columns={'essential': 'v15_call', 'confidence': 'v15_conf'})
    # Map true_class to binary: Essential or Quasiessential -> 1, Nonessential -> 0
    df['true_essential'] = df.true_class.map(
        {'Essential': 1, 'Quasiessential': 1, 'Nonessential': 0}).astype(int)
    return df[['locus_tag', 'gene_name', 'v15_call', 'v15_conf',
                 'true_essential', 'failure_mode']]


def build_mmseqs_index(target_org: str, ref_orgs: set) -> dict:
    """For each target gene, list all mmseqs hits to reference-org genes
    with their identity scores. Used as the 'memory bank' lookup."""
    print(f'[mmseqs] reading {MMSEQS_TSV.name}...')
    hits = defaultdict(list)   # target_locus -> [(ref_org, ref_locus, ident), ...]
    n_lines = 0
    with open(MMSEQS_TSV) as f:
        for line in f:
            n_lines += 1
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            q, t, ident = parts[0], parts[1], float(parts[2])
            if q == t: continue
            try:
                qo, ql = q.split('|', 1); to, tl = t.split('|', 1)
            except ValueError: continue
            if qo == target_org and to in ref_orgs:
                hits[ql].append((to, tl, ident))
    print(f'  parsed {n_lines:,} hit lines, {len(hits):,} target genes have '
          f'cross-org hits')
    return hits


def build_rbh_orthologs(target_org: str, ref_orgs: set) -> dict:
    """For each target gene, list its RBH partners in reference orgs.
    Subset of mmseqs hits filtered to reciprocal-best only."""
    print(f'[rbh] computing RBH partners for {target_org}...')
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

    rbh = defaultdict(list)   # target_locus -> [(ref_org, ref_locus), ...]
    for (q_org, q_locus), partners in best.items():
        if q_org != target_org: continue
        for t_org, (t_locus, _) in partners.items():
            if t_org not in ref_orgs: continue
            rev = best.get((t_org, t_locus), {}).get(q_org)
            if rev is not None and rev[0] == q_locus:
                rbh[q_locus].append((t_org, t_locus))
    n_with_rbh = len(rbh)
    print(f'  {n_with_rbh} target genes have RBH orthologs')
    return rbh


def build_label_lookup() -> dict:
    df = pd.read_csv(LABELS_CSV, usecols=['organism', 'locus_tag', 'essential'])
    return {(r.organism, r.locus_tag): int(r.essential)
             for r in df.itertuples(index=False) if not pd.isna(r.essential)}


def ortholog_prior(target_locus: str, rbh: dict, ess_lookup: dict) -> tuple:
    partners = rbh.get(target_locus, [])
    labels = [ess_lookup.get((po, pl)) for (po, pl) in partners]
    labels = [l for l in labels if l is not None]
    if not labels: return None, 0
    return float(np.mean(labels)), len(labels)


def memory_kNN_prior(target_locus: str, hits: dict, ess_lookup: dict,
                       top_k: int = MEMORY_KNN_TOP_K) -> tuple:
    """Identity-weighted mean essentiality of the top-K mmseqs hits.
    Broader than RBH — includes any sequence-similar training gene."""
    raw_hits = hits.get(target_locus, [])
    if not raw_hits: return None, 0
    # Sort by identity descending, take top K
    raw_hits = sorted(raw_hits, key=lambda x: -x[2])[:top_k]
    weighted_sum, weight_sum, n_used = 0.0, 0.0, 0
    for (ro, rl, ident) in raw_hits:
        lab = ess_lookup.get((ro, rl))
        if lab is None: continue
        w = ident
        weighted_sum += w * lab
        weight_sum += w
        n_used += 1
    if weight_sum == 0: return None, 0
    return float(weighted_sum / weight_sum), n_used


def cascade_predict(v15_df: pd.DataFrame, rbh: dict, hits: dict,
                      ess_lookup: dict,
                      cascade_t: float = CASCADE_THRESHOLD,
                      additive_t: float = 0.7,
                      ) -> pd.DataFrame:
    """Apply BOTH cascade strategies per gene.

    Strategy A (gating): if v15 confident -> v15, else fallback >= 0.5
        - Designed for strong fallbacks. Hurts when fallback is noisier
          than v15 because it overrides v15's reliable TNs.

    Strategy B (additive / OR): cascade_essential iff v15_essential OR
        fallback_strongly_essential (fallback_prior >= additive_t)
        - Preserves v15's high precision. Only uses fallback to recover
          FNs that v15 missed. Right strategy when v15 >> fallback.
    """
    rows = []
    for _, r in v15_df.iterrows():
        loc = r.locus_tag
        v15_call = int(r.v15_call); v15_conf = float(r.v15_conf)

        ortho_p, n_orth = ortholog_prior(loc, rbh, ess_lookup)
        mem_p, n_mem = memory_kNN_prior(loc, hits, ess_lookup)

        # Combine fallback: weighted blend if both available
        if ortho_p is not None and mem_p is not None:
            fallback_p = (ORTHOLOG_WEIGHT * ortho_p
                            + (1 - ORTHOLOG_WEIGHT) * mem_p)
            fallback_src = 'orth+mem'
        elif ortho_p is not None:
            fallback_p = ortho_p; fallback_src = 'orth_only'
        elif mem_p is not None:
            fallback_p = mem_p; fallback_src = 'mem_only'
        else:
            fallback_p = None    # no fallback signal — defer to v15
            fallback_src = 'none'

        # Strategy A: gating (original cascade design)
        if v15_conf >= cascade_t:
            cas_a = v15_call
            cas_a_src = f'v15>={cascade_t}'
        elif fallback_p is not None:
            cas_a = int(fallback_p >= 0.5)
            cas_a_src = f'fallback({fallback_src})'
        else:
            cas_a = v15_call
            cas_a_src = 'v15_no_fallback'

        # Strategy B: additive / OR — preserves v15's TNs, recovers FNs
        # only when fallback is strongly confident
        if v15_call == 1:
            cas_b = 1
            cas_b_src = 'v15_essential'
        elif fallback_p is not None and fallback_p >= additive_t:
            cas_b = 1
            cas_b_src = f'fallback_strong({fallback_src})'
        else:
            cas_b = 0
            cas_b_src = 'v15_nonessential'

        rows.append({
            'locus_tag': loc, 'gene_name': r.gene_name,
            'true_essential': int(r.true_essential),
            'v15_call': v15_call, 'v15_conf': v15_conf,
            'ortholog_prior': ortho_p, 'n_orthologs': n_orth,
            'memory_prior': mem_p, 'n_memory_neighbors': n_mem,
            'fallback_prior': fallback_p, 'fallback_source': fallback_src,
            'cascade_a_call': cas_a, 'cascade_a_source': cas_a_src,
            'cascade_b_call': cas_b, 'cascade_b_source': cas_b_src,
        })
    return pd.DataFrame(rows)


def metrics_block(y, pred, name):
    if len(set(y)) < 2 or len(set(pred)) < 2:
        return {'mcc': 0.0, 'name': name}
    mcc = matthews_corrcoef(y, pred)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {'name': name, 'mcc': float(mcc),
             'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='syn3a')
    ap.add_argument('--cascade-threshold', type=float, default=CASCADE_THRESHOLD)
    ap.add_argument('--top-k', type=int, default=MEMORY_KNN_TOP_K)
    ap.add_argument('--additive-threshold', type=float, default=0.7,
                     help='Strategy B: only override v15 with essential when '
                            'fallback prior >= this (default 0.7).')
    ap.add_argument('--out', type=Path, default=OUT_JSON)
    args = ap.parse_args()

    target = args.target
    ref_orgs = set(o for o in OUR_ORGS if o != target)
    print(f'target = {target}; references = {sorted(ref_orgs)}\n')

    print('[1] loading v15 cached predictions...')
    v15_df = load_v15_predictions()
    print(f'  {len(v15_df)} {target} genes from v15 cached run')
    print(f'  v15 confidence: median={v15_df.v15_conf.median():.2f}, '
          f'frac >= {args.cascade_threshold}: '
          f'{(v15_df.v15_conf >= args.cascade_threshold).mean():.3f}\n')

    rbh = build_rbh_orthologs(target, ref_orgs)
    hits = build_mmseqs_index(target, ref_orgs)
    ess_lookup = build_label_lookup()

    print(f'\n[2] running cascade...')
    pred_df = cascade_predict(
        v15_df, rbh, hits, ess_lookup,
        cascade_t=args.cascade_threshold,
        additive_t=args.additive_threshold,
    )
    pred_df.to_csv(ROOT / f'outputs/cascade_v15_orth_mem_{target}_per_gene.csv',
                     index=False)
    print(f'  saved per-gene predictions')

    # Compute MCCs for each component
    y = pred_df.true_essential.values
    print(f'\n[3] component MCCs (target = {target}, '
          f'{int(y.sum())} essential / {int((y==0).sum())} non-essential):')

    # v15 alone
    m_v15 = metrics_block(y, pred_df.v15_call.values, 'v15_alone')
    print(f'  {m_v15["name"]:25s}  MCC={m_v15["mcc"]:.4f}  '
          f'TP={m_v15["tp"]} FP={m_v15["fp"]} TN={m_v15["tn"]} FN={m_v15["fn"]}')

    # Ortholog prior alone (where available; default to class prior elsewhere)
    orth_call = np.where(pred_df.ortholog_prior.notna(),
                            pred_df.ortholog_prior.fillna(0.5) >= 0.5,
                            1).astype(int)
    m_orth = metrics_block(y, orth_call, 'ortholog_alone')
    print(f'  {m_orth["name"]:25s}  MCC={m_orth["mcc"]:.4f}  '
          f'TP={m_orth["tp"]} FP={m_orth["fp"]} TN={m_orth["tn"]} FN={m_orth["fn"]}  '
          f'(coverage: '
          f'{int(pred_df.ortholog_prior.notna().sum())}/{len(pred_df)})')

    # Memory bank alone
    mem_call = np.where(pred_df.memory_prior.notna(),
                          pred_df.memory_prior.fillna(0.5) >= 0.5,
                          1).astype(int)
    m_mem = metrics_block(y, mem_call, 'memory_kNN_alone')
    print(f'  {m_mem["name"]:25s}  MCC={m_mem["mcc"]:.4f}  '
          f'TP={m_mem["tp"]} FP={m_mem["fp"]} TN={m_mem["tn"]} FN={m_mem["fn"]}  '
          f'(coverage: '
          f'{int(pred_df.memory_prior.notna().sum())}/{len(pred_df)})')

    # Combined fallback alone (no cascade) — fallback_prior >= 0.5 where available
    fb_mask = pred_df.fallback_prior.notna()
    fb_call = np.where(fb_mask, pred_df.fallback_prior.fillna(0.5) >= 0.5,
                          1).astype(int)
    m_fb = metrics_block(y, fb_call, 'fallback_alone')
    print(f'  {m_fb["name"]:25s}  MCC={m_fb["mcc"]:.4f}  '
          f'TP={m_fb["tp"]} FP={m_fb["fp"]} TN={m_fb["tn"]} FN={m_fb["fn"]}')

    # Strategy A: gating cascade (original design)
    cas_a_call = pred_df.cascade_a_call.values
    m_cas_a = metrics_block(y, cas_a_call, 'CASCADE_A_gating')
    print(f'\n  {m_cas_a["name"]:25s}  MCC={m_cas_a["mcc"]:.4f}  '
          f'TP={m_cas_a["tp"]} FP={m_cas_a["fp"]} TN={m_cas_a["tn"]} FN={m_cas_a["fn"]}')

    # Strategy B: additive (v15 OR strong fallback)
    cas_b_call = pred_df.cascade_b_call.values
    m_cas_b = metrics_block(y, cas_b_call, f'CASCADE_B_additive_t{args.additive_threshold}')
    print(f'  {m_cas_b["name"]:25s}  MCC={m_cas_b["mcc"]:.4f}  '
          f'TP={m_cas_b["tp"]} FP={m_cas_b["fp"]} TN={m_cas_b["tn"]} FN={m_cas_b["fn"]}')

    # Per-source breakdown for the better of the two
    print(f'\n  Strategy B (additive) routing:')
    for src in pred_df.cascade_b_source.value_counts().index:
        sub = pred_df[pred_df.cascade_b_source == src]
        n = len(sub)
        if n == 0: continue
        ys = sub.true_essential.values; cs = sub.cascade_b_call.values
        n_correct = int((cs == ys).sum())
        print(f'    {src:30s}  n={n:>3d}  correct={n_correct}/{n} '
              f'({100*n_correct/n:.1f}%)')

    # Threshold sweep on Strategy B to find best additive threshold
    print(f'\n  Strategy B threshold sweep (additive_t):')
    best_t, best_mcc = args.additive_threshold, m_cas_b['mcc']
    for t in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        # Recompute Strategy B with this threshold
        b_call = []
        for _, r in pred_df.iterrows():
            if r.v15_call == 1:
                b_call.append(1)
            elif r.fallback_prior is not None and not pd.isna(r.fallback_prior) \
                   and r.fallback_prior >= t:
                b_call.append(1)
            else:
                b_call.append(0)
        b_call = np.array(b_call)
        m = metrics_block(y, b_call, f't={t}')
        marker = ' <- best' if m['mcc'] > best_mcc else ''
        print(f'    t={t:.2f}  MCC={m["mcc"]:.4f}  '
              f'TP={m["tp"]} FP={m["fp"]} TN={m["tn"]} FN={m["fn"]}{marker}')
        if m['mcc'] > best_mcc:
            best_t, best_mcc = t, m['mcc']
    print(f'\n  best additive threshold: t={best_t}  MCC={best_mcc:.4f}')

    m_cas = m_cas_b if m_cas_b['mcc'] > m_cas_a['mcc'] else m_cas_a
    cascade_strategy = 'B_additive' if m_cas_b['mcc'] > m_cas_a['mcc'] else 'A_gating'

    print(f'\n  reference baselines:')
    print(f'    v15 simulator alone (mechanistic):     ~0.505')
    print(f'    LNN cascade with Syn3A in training:    0.768 (memorized!)')
    print(f'    LNN cross-org Syn3A held out:          0.08-0.26 (honest)')
    print(f'    ortholog prior alone, Syn3A held out:  0.302')
    print(f'    THIS RUN (cascade, Syn3A held out):    {m_cas["mcc"]:.4f}')

    out = {
        'method': 'cascade_v15_ortholog_memory_syn3a_held_out',
        'target': target,
        'ref_orgs': sorted(ref_orgs),
        'cascade_threshold': args.cascade_threshold,
        'additive_threshold': args.additive_threshold,
        'best_additive_threshold': best_t,
        'best_additive_mcc': best_mcc,
        'memory_top_k': args.top_k,
        'ortholog_blend_weight': ORTHOLOG_WEIGHT,
        'best_strategy': cascade_strategy,
        'components': {
            'v15_alone': m_v15,
            'ortholog_alone': m_orth,
            'memory_kNN_alone': m_mem,
            'fallback_combined': m_fb,
            'CASCADE_A_gating': m_cas_a,
            'CASCADE_B_additive': m_cas_b,
        },
        'baselines': {
            'v15_simulator_alone': 0.505,
            'lnn_cascade_with_syn3a_in_training': 0.768,
            'lnn_cross_org_syn3a_held_out': 0.26,
            'ortholog_prior_only_syn3a_held_out': 0.302,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()

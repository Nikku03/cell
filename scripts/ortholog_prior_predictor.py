"""Ortholog-based essentiality prior — the simplest cross-organism predictor.

Predict gene essentiality by averaging the essentiality of its
reciprocal-best-hit (RBH) orthologs in reference organisms. No learned
parameters; cross-organism transfer is bounded only by how much
essentiality is conserved across orthologs.

Pipeline
--------
1. Build RBH orthology table from `/tmp/conservation_work/all_vs_all.tsv`
   (the cached mmseqs all-vs-all from `build_conservation_features.py`).
2. Load per-organism essentiality labels from
   `outputs/multiorg_per_gene_features.csv`.
3. For each target organism, for each gene:
     orthologs = RBH partners in the 7 reference organisms
     prior(G)  = mean of [is_essential(o) for o in orthologs]
     prediction(G) = (prior(G) >= 0.5)
4. Report MCC, AUC, AP per target organism + leave-one-out summary.

Comparison points (commit history)
-----------------------------------
- LNN leave-one-org-out, no PPI: mean MCC 0.264, std 0.065
- LNN+PPI held-out mtb (regularized): MCC 0.158
- LNN+PPI held-out mtb (unregularized): MCC 0.115

If the ortholog prior beats 0.264 mean, the LNN's 432k parameters add
nothing for cross-org deployment and the simple thing is the right
thing.
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
OUT_JSON = ROOT / 'outputs/ortholog_prior_results.json'

OUR_ORGS = ['styphimurium', 'ccrescentus', 'saureus', 'mtuberculosis',
            'abaylyi', 'mgenitalium', 'mpne', 'syn3a']


def build_rbh_pairs(mmseqs_tsv: Path, organisms: set) -> set:
    """Walk mmseqs all-vs-all output, build the set of RBH ortholog pairs.

    For each (q_org, q_locus, t_org), the best hit is the highest-identity
    target. RBH = target's best hit back is the original query.
    """
    print(f'[1] reading {mmseqs_tsv}...')
    best = defaultdict(dict)   # (q_org, q_locus) -> t_org -> (t_locus, ident)
    n_lines = 0
    with open(mmseqs_tsv) as f:
        for line in f:
            n_lines += 1
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            q, t, ident = parts[0], parts[1], float(parts[2])
            if q == t: continue
            try:
                qo, ql = q.split('|', 1); to, tl = t.split('|', 1)
            except ValueError: continue
            if qo not in organisms or to not in organisms: continue
            cur = best[(qo, ql)].get(to)
            if cur is None or ident > cur[1]:
                best[(qo, ql)][to] = (tl, ident)
    print(f'  parsed {n_lines:,} lines, '
          f'{len(best):,} (org, locus) keys with at least one cross-org hit')

    print(f'[2] computing reciprocal-best-hit pairs...')
    rbh_pairs = set()
    for (q_org, q_locus), partners in best.items():
        for t_org, (t_locus, _ident) in partners.items():
            if t_org == q_org: continue
            rev = best.get((t_org, t_locus), {}).get(q_org)
            if rev is not None and rev[0] == q_locus:
                # Canonical ordering keeps each pair once
                key = tuple(sorted([(q_org, q_locus), (t_org, t_locus)]))
                rbh_pairs.add(key)
    print(f'  {len(rbh_pairs):,} RBH ortholog pairs across '
          f'{len(organisms)} organisms')
    return rbh_pairs


def build_ortholog_index(rbh_pairs: set) -> dict:
    """For each (org, locus), list all (other_org, other_locus) RBH partners."""
    idx = defaultdict(list)
    for ((oa, la), (ob, lb)) in rbh_pairs:
        idx[(oa, la)].append((ob, lb))
        idx[(ob, lb)].append((oa, la))
    return idx


def predict_organism(target_org: str, ref_orgs: list, ortholog_idx: dict,
                       label_df: pd.DataFrame, ess_lookup: dict
                       ) -> tuple[pd.DataFrame, dict]:
    """Compute ortholog prior for each gene in target_org."""
    target = label_df[label_df.organism == target_org].copy()
    rows = []
    for _, r in target.iterrows():
        loc = r.locus_tag
        true_ess = int(r.essential) if not pd.isna(r.essential) else None
        partners = ortholog_idx.get((target_org, loc), [])
        # Filter to reference organisms only (exclude target itself)
        partner_labels = []
        for (po, pl) in partners:
            if po not in ref_orgs: continue
            lab = ess_lookup.get((po, pl))
            if lab is not None:
                partner_labels.append(lab)
        n_orth = len(partner_labels)
        if n_orth == 0:
            prior = np.nan
            pred = 0  # abstain → predict majority class (non-essential default;
                       # we'll also compute a "skip-no-ortholog" MCC below)
        else:
            prior = float(np.mean(partner_labels))
            pred = int(prior >= 0.5)
        rows.append({
            'organism': target_org, 'locus_tag': loc,
            'true_essential': true_ess, 'n_orthologs': n_orth,
            'ortholog_prior': prior, 'prediction_at_0.5': pred,
        })
    df = pd.DataFrame(rows)

    # Metrics
    has_lab = df.true_essential.notna()
    has_orth = df.n_orthologs > 0
    y_all = df[has_lab].true_essential.astype(int).values
    pred_all = df[has_lab]['prediction_at_0.5'].values
    prior_all = df[has_lab].ortholog_prior.fillna(0.5).values

    metrics = {'n_total': int(has_lab.sum()),
                'n_with_orthologs': int((has_lab & has_orth).sum()),
                'n_without_orthologs': int((has_lab & ~has_orth).sum())}

    # Including no-ortholog genes (default predict 0)
    if len(set(y_all)) > 1 and len(set(pred_all)) > 1:
        metrics['mcc_with_default'] = float(matthews_corrcoef(y_all, pred_all))
        tn, fp, fn, tp = confusion_matrix(y_all, pred_all).ravel()
        metrics.update({'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)})
    else:
        metrics['mcc_with_default'] = 0.0

    # Excluding genes without orthologs (only on covered subset)
    sub = df[has_lab & has_orth]
    if len(sub) > 0:
        y = sub.true_essential.astype(int).values
        p = sub['prediction_at_0.5'].values
        priors = sub.ortholog_prior.values
        if len(set(y)) > 1 and len(set(p)) > 1:
            metrics['mcc_covered_only'] = float(matthews_corrcoef(y, p))
        else:
            metrics['mcc_covered_only'] = 0.0
        if len(set(y)) > 1:
            try: metrics['auc'] = float(roc_auc_score(y, priors))
            except ValueError: metrics['auc'] = float('nan')
            try: metrics['ap'] = float(average_precision_score(y, priors))
            except ValueError: metrics['ap'] = float('nan')
        # Best-threshold MCC (diagnostic)
        best_t, best_mcc = 0.5, -2.0
        cand = sorted(set([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        for t in cand:
            pp = (priors >= t).astype(int)
            if pp.sum() in (0, len(pp)): continue
            m = matthews_corrcoef(y, pp)
            if m > best_mcc: best_mcc, best_t = m, t
        metrics['mcc_covered_best_t'] = float(best_mcc)
        metrics['best_t'] = float(best_t)

    return df, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default=','.join(OUR_ORGS),
                     help='comma-separated organisms to evaluate (each tested '
                            'while held out from references)')
    ap.add_argument('--out', type=Path, default=OUT_JSON)
    ap.add_argument('--save-predictions', action='store_true',
                     help='save per-gene predictions CSV per target')
    args = ap.parse_args()

    targets = [o.strip() for o in args.targets.split(',') if o.strip()]
    print(f'targets: {targets}')

    # Build RBH ortholog index once for all 8 organisms
    rbh_pairs = build_rbh_pairs(MMSEQS_TSV, set(OUR_ORGS))
    ortholog_idx = build_ortholog_index(rbh_pairs)

    # Load labels
    print(f'\n[3] loading labels from {LABELS_CSV.name}...')
    label_df = pd.read_csv(LABELS_CSV, usecols=['organism', 'locus_tag', 'essential'])
    label_df = label_df[label_df.organism.isin(OUR_ORGS)]
    print(f'  {len(label_df):,} labeled rows across '
          f'{label_df.organism.nunique()} organisms')

    # essentiality lookup
    ess_lookup = {(r.organism, r.locus_tag): int(r.essential)
                    for r in label_df.itertuples(index=False)
                    if not pd.isna(r.essential)}
    print(f'  {len(ess_lookup):,} (org, locus) -> essential lookup entries')

    # Per-organism leave-one-out
    print(f'\n[4] running leave-one-out evaluation...')
    print(f'{"target":15s}  {"n":>4s}  {"orthologs":>9s}  {"covered MCC":>11s}  '
          f'{"with-default":>13s}  {"AUC":>6s}  {"best-t":>7s}  {"MCC@best":>9s}')
    print('-' * 100)
    all_results = {}
    all_predictions = {}
    for target in targets:
        ref_orgs = [o for o in OUR_ORGS if o != target]
        preds, m = predict_organism(target, ref_orgs, ortholog_idx,
                                       label_df, ess_lookup)
        all_results[target] = m
        all_predictions[target] = preds
        cov = m.get('mcc_covered_only', 0.0)
        wd = m.get('mcc_with_default', 0.0)
        auc = m.get('auc', float('nan'))
        bt = m.get('best_t', 0.5)
        bm = m.get('mcc_covered_best_t', 0.0)
        n_o = m.get('n_with_orthologs', 0)
        n_t = m.get('n_total', 0)
        print(f'{target:15s}  {n_t:>4d}  {n_o:>9d}  {cov:>11.4f}  '
              f'{wd:>13.4f}  {auc:>6.4f}  {bt:>7.2f}  {bm:>9.4f}')

    # Aggregate
    cov_mccs = [r.get('mcc_covered_only', 0.0) for r in all_results.values()]
    wd_mccs = [r.get('mcc_with_default', 0.0) for r in all_results.values()]
    aucs = [r.get('auc', float('nan')) for r in all_results.values()
             if not np.isnan(r.get('auc', float('nan')))]
    print('-' * 100)
    print(f'{"mean":15s}              '
          f'  {np.mean(cov_mccs):>11.4f}  {np.mean(wd_mccs):>13.4f}  '
          f'{np.mean(aucs) if aucs else float("nan"):>6.4f}')

    # Reference comparison
    print(f'\nReference baselines:')
    print(f'  LNN leave-one-org-out, no PPI:        0.264 mean (commit 02630d9)')
    print(f'  LNN+PPI mtb held out (unregularized): 0.115')
    print(f'  LNN+PPI mtb held out (regularized):   0.158')
    print(f'  ortholog prior on mtb (this run):     '
          f'{all_results.get("mtuberculosis", {}).get("mcc_covered_only", 0.0):.4f} '
          f'(covered) / '
          f'{all_results.get("mtuberculosis", {}).get("mcc_with_default", 0.0):.4f} '
          f'(with default)')
    print(f'  ortholog prior mean (8 leave-one-out):'
          f'  {np.mean(cov_mccs):.4f} (covered) / {np.mean(wd_mccs):.4f} (with default)')

    # Save
    out = {
        'method': 'ortholog_rbh_prior',
        'organisms': targets,
        'n_rbh_pairs_total': len(rbh_pairs),
        'per_organism': all_results,
        'aggregate': {
            'mean_mcc_covered_only': float(np.mean(cov_mccs)),
            'mean_mcc_with_default': float(np.mean(wd_mccs)),
            'mean_auc': float(np.mean(aucs)) if aucs else float('nan'),
            'std_mcc_covered_only': float(np.std(cov_mccs)),
        },
        'baselines': {
            'lnn_leave_one_org_out_no_ppi_mean': 0.264,
            'lnn_with_ppi_mtb_unregularized': 0.115,
            'lnn_with_ppi_mtb_regularized': 0.158,
            'v15_simulator_syn3a_only': 0.505,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nwrote {args.out}')

    if args.save_predictions:
        for org, df in all_predictions.items():
            p = ROOT / f'outputs/ortholog_prior_predictions_{org}.csv'
            df.to_csv(p, index=False)
            print(f'  per-gene predictions: {p}')


if __name__ == '__main__':
    main()

"""Score per-gene ortholog-prior features for the LNN, leave-one-out.

For each (organism, locus_tag) in our 8-org panel, compute:
  ortholog_prior   = mean essentiality of RBH partners in the OTHER 7 orgs
  n_orthologs      = count of RBH partners with labels in those 7 orgs

This is the leave-one-out variant: each gene's prior is built from
organisms OTHER than its own. So when the LNN trains on a gene G in org
O, the prior already excludes O's labels — no information leak.

Usage
-----
  python scripts/score_ortholog_prior_feature.py
  -> writes outputs/ortholog_prior_features_per_gene.csv

Schema: organism, locus_tag, ortholog_prior, n_orthologs

Comparison
----------
This is the same ortholog prior the standalone predictor used (commit
97bd0f6, mean leave-one-out MCC 0.420). Now exposed as a per-gene
feature for the LNN to mix with sequence + scalar + regulatory inputs.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MMSEQS_TSV = Path('/tmp/conservation_work/all_vs_all.tsv')
LABELS_CSV = ROOT / 'outputs/multiorg_per_gene_features.csv'
OUT_CSV = ROOT / 'outputs/ortholog_prior_features_per_gene.csv'

OUR_ORGS = ['styphimurium', 'ccrescentus', 'saureus', 'mtuberculosis',
            'abaylyi', 'mgenitalium', 'mpne', 'syn3a']


def build_rbh(organisms: set) -> set:
    """Walk mmseqs all-vs-all, return set of RBH ortholog pairs (canonical
    ordering)."""
    print(f'[1] reading {MMSEQS_TSV.name}...')
    best = defaultdict(dict)
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
            if qo not in organisms or to not in organisms: continue
            cur = best[(qo, ql)].get(to)
            if cur is None or ident > cur[1]:
                best[(qo, ql)][to] = (tl, ident)
    print(f'  parsed {n_lines:,} hit lines, {len(best):,} cross-org keys')

    print('[2] computing RBH pairs...')
    rbh = set()
    for (q_org, q_locus), partners in best.items():
        for t_org, (t_locus, _) in partners.items():
            if t_org == q_org: continue
            rev = best.get((t_org, t_locus), {}).get(q_org)
            if rev is not None and rev[0] == q_locus:
                key = tuple(sorted([(q_org, q_locus), (t_org, t_locus)]))
                rbh.add(key)
    print(f'  {len(rbh):,} RBH pairs')
    return rbh


def main():
    rbh_pairs = build_rbh(set(OUR_ORGS))

    # Index: (org, locus) -> [(other_org, other_locus), ...]
    idx = defaultdict(list)
    for ((oa, la), (ob, lb)) in rbh_pairs:
        idx[(oa, la)].append((ob, lb))
        idx[(ob, lb)].append((oa, la))

    print('[3] loading labels...')
    label_df = pd.read_csv(LABELS_CSV, usecols=['organism', 'locus_tag', 'essential'])
    label_df = label_df[label_df.organism.isin(OUR_ORGS)]
    ess_lookup = {(r.organism, r.locus_tag): int(r.essential)
                    for r in label_df.itertuples(index=False)
                    if not pd.isna(r.essential)}
    print(f'  {len(ess_lookup):,} (org, locus) -> essential entries')

    print('[4] computing per-gene leave-one-out ortholog priors...')
    rows = []
    for org in OUR_ORGS:
        ref_orgs = set(o for o in OUR_ORGS if o != org)
        sub = label_df[label_df.organism == org]
        n_with = 0
        for r in sub.itertuples(index=False):
            partners = idx.get((org, r.locus_tag), [])
            partner_labels = [ess_lookup.get((po, pl)) for (po, pl) in partners
                                if po in ref_orgs]
            partner_labels = [l for l in partner_labels if l is not None]
            if partner_labels:
                prior = float(np.mean(partner_labels))
                n = len(partner_labels)
                n_with += 1
            else:
                # No labeled orthologs -> use a neutral prior (0.5) so the
                # LNN can learn it doesn't carry information for this gene
                prior = 0.5
                n = 0
            rows.append({
                'organism': org, 'locus_tag': r.locus_tag,
                'ortholog_prior': prior, 'n_orthologs': n,
            })
        print(f'  {org:15s}  {len(sub):>5d} genes  {n_with:>5d} with labeled orthologs')

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nwrote {OUT_CSV}: {len(df)} rows')

    # Sanity: per-org correlation of prior with truth
    print('\nper-org sanity (prior vs. true essential):')
    label_df2 = label_df.merge(df, on=['organism', 'locus_tag'], how='left')
    for org in OUR_ORGS:
        sub = label_df2[(label_df2.organism == org) & (label_df2.n_orthologs > 0)]
        if len(sub) < 5: continue
        y = sub.essential.values; p = sub.ortholog_prior.values
        # Pearson (cheap correlation)
        if len(set(y)) > 1:
            corr = np.corrcoef(y, p)[0, 1]
            print(f'  {org:15s}  n_covered={len(sub):>4d}  '
                  f'corr(prior, essential)={corr:+.3f}')


if __name__ == '__main__':
    main()

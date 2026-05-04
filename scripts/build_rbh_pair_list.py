"""Build the cross-organism RBH ortholog pair list for the ortholog
consistency loss in the minimal generalizing LNN training.

For each pair of training organisms in our 8-org panel, finds reciprocal-
best-hit ortholog pairs from the cached mmseqs all-vs-all output. Writes
`outputs/rbh_pairs.csv` with columns:
  org_a, locus_a, org_b, locus_b, identity

The training loop will then sample N pairs per step, look up each gene's
prediction in its organism's batch, and penalize divergence.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MMSEQS_TSV = Path('/tmp/conservation_work/all_vs_all.tsv')
OUT_CSV = ROOT / 'outputs/rbh_pairs.csv'

OUR_ORGS = {'styphimurium', 'ccrescentus', 'saureus', 'mtuberculosis',
            'abaylyi', 'mgenitalium', 'mpne', 'syn3a'}


def main():
    print(f'reading {MMSEQS_TSV.name}...')
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
            if qo not in OUR_ORGS or to not in OUR_ORGS: continue
            cur = best[(qo, ql)].get(to)
            if cur is None or ident > cur[1]:
                best[(qo, ql)][to] = (tl, ident)

    print(f'parsed {n_lines:,} lines, computing RBH pairs...')
    rows = []
    seen = set()
    for (q_org, q_locus), partners in best.items():
        for t_org, (t_locus, ident) in partners.items():
            if t_org == q_org: continue
            rev = best.get((t_org, t_locus), {}).get(q_org)
            if rev is not None and rev[0] == q_locus:
                # canonical ordering so we don't double-count (a,b) vs (b,a)
                a = (q_org, q_locus); b = (t_org, t_locus)
                if a > b: a, b = b, a
                key = (*a, *b)
                if key in seen: continue
                seen.add(key)
                rows.append({
                    'org_a': a[0], 'locus_a': a[1],
                    'org_b': b[0], 'locus_b': b[1],
                    'identity': ident,
                })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f'wrote {OUT_CSV}: {len(df):,} RBH pairs')

    # Pair counts per organism pair
    print('\npairs per (org_a, org_b):')
    for (oa, ob), n in df.groupby(['org_a', 'org_b']).size().items():
        print(f'  {oa:15s} <-> {ob:15s}  {n:>5d} pairs')


if __name__ == '__main__':
    main()

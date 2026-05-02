"""Phase 8 — build cross-organism conservation features for the 17 GenBanks
on disk via mmseqs2 all-vs-all protein search.

Plan
----

1. Extract all proteomes to a single FASTA with headers `<organism>|<locus_tag>`.
2. Run mmseqs2 easy-search self-vs-self.
3. For each protein in (organism, locus_tag), compute:
     - n_orthologs_total (count of OTHER organisms with at least one hit
       passing e-value < 1e-5, coverage > 50%, identity > 30%)
     - n_orthologs_within_mollicutes (subset: mpne, syn3a)
     - paralog_count_internal (hits within same organism, excluding self)
     - mean_identity_to_orthologs (numeric)
     - 16 binary phyletic-profile features
4. Sanity-check: dnaA in syn3a should have orthologs in most/all 16. Hypothetical
   uncharacterized proteins should have fewer.
5. Save augmented per-gene feature matrix.

Output: outputs/multiorg_per_gene_features_with_conservation.csv
"""
from __future__ import annotations

import json
import subprocess
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
GB_DIR = ROOT / 'memory_bank/data/multiorg_essentiality/raw/multi_organism_genbank'
INP_FEATURES = ROOT / 'outputs/multiorg_per_gene_features.csv'
OUT_FEATURES = ROOT / 'outputs/multiorg_per_gene_features_with_conservation.csv'
WORK_DIR = Path('/tmp/conservation_work')
WORK_DIR.mkdir(parents=True, exist_ok=True)
PROTEOME_FA = WORK_DIR / 'all_proteomes.fa'
MMSEQS_DB = WORK_DIR / 'all_db'
MMSEQS_RESULT = WORK_DIR / 'all_vs_all.tsv'

GB_PATHS = {
    'ccrescentus': GB_DIR / 'ccrescentus_NA1000_CP001340.1.gb',
    'saureus': GB_DIR / 'saureus_N315_BA000018.3.gb',
    'mtuberculosis': GB_DIR / 'mtuberculosis_H37Rv_AL123456.3.gb',
    'styphimurium': GB_DIR / 'styphimurium_LT2_AE006468.2.gb',
    'abaylyi': GB_DIR / 'abaylyi_ADP1_CR543861.1.gb',
    'mpne': ROOT / 'memory_bank/data/multiorg_essentiality/raw/mpneumoniae_M129_NC_000912.gb',
    'syn3a': ROOT / 'cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb',
    # Additional organisms with GenBank but no essentiality (still useful for
    # conservation counts as 'extra reference' organisms).
    'abaumannii': GB_DIR / 'abaumannii_AB5075_CP008706.1.gb',
    'bsubtilis': GB_DIR / 'bsubtilis_168_AL009126.3.gb',
    'ecoli': GB_DIR / 'ecoli_K12_MG1655_U00096.3.gb',
    'ftularensis': GB_DIR / 'ftularensis_SCHU_S4_AJ749949.2.gb',
    'hinfluenzae': GB_DIR / 'hinfluenzae_Rd_KW20_L42023.1.gb',
    'hpylori': GB_DIR / 'hpylori_26695_AE000511.1.gb',
    'kpneumoniae': GB_DIR / 'kpneumoniae_KPNIH1_CP008827.1.gb',
    'mgenitalium': GB_DIR / 'mgenitalium_G37_L43967.2.gb',
    'paeruginosa': GB_DIR / 'paeruginosa_PAO1_AE004091.2.gb',
    'spneumoniae': GB_DIR / 'spneumoniae_TIGR4_AE005672.3.gb',
    'vcholerae': GB_DIR / 'vcholerae_N16961_chr1_AE003852.1.gb',
}

# Mycoplasmataceae / Mollicute organisms among our set
MOLLICUTES = {'mpne', 'syn3a', 'mgenitalium'}


def organism_locus_policy(organism: str, f) -> str | None:
    """Match the locus_tag policy from build_multiorg_features.py so the
    conservation features can be joined onto the v25 feature CSV."""
    locus = f.qualifiers.get('locus_tag', [None])[0]
    gene_sym = (f.qualifiers.get('gene', [''])[0] or '').strip()
    if organism == 'mpne':
        ot = f.qualifiers.get('old_locus_tag', [None])[0]
        if ot: locus = ot
    elif organism == 'saureus':
        locus = gene_sym if gene_sym else locus
    elif organism == 'styphimurium':
        locus = gene_sym if gene_sym else locus
    return locus


def step1_extract_proteomes() -> dict:
    """Write a single FASTA with all proteins keyed by `organism|locus_tag`.
    Returns metadata: {(organism, locus_tag): protein_length}."""
    print('[1] extracting proteomes...')
    n_total = 0
    meta = {}
    seen = set()
    with open(PROTEOME_FA, 'w') as out:
        for org, gb_path in GB_PATHS.items():
            if not gb_path.exists():
                print(f'  skip {org}: GenBank missing'); continue
            rec = next(SeqIO.parse(str(gb_path), 'genbank'))
            n_org = 0
            for f in rec.features:
                if f.type != 'CDS': continue
                locus = organism_locus_policy(org, f)
                if not locus: continue
                trans = f.qualifiers.get('translation', [''])[0]
                if not trans or len(trans) < 30: continue
                key = (org, locus)
                if key in seen: continue
                seen.add(key)
                # Sanitize header (no | in locus)
                header = f'{org}|{locus}'
                out.write(f'>{header}\n{trans}\n')
                meta[key] = len(trans)
                n_org += 1
                n_total += 1
            print(f'  {org}: {n_org} proteins')
    print(f'  total: {n_total} proteins -> {PROTEOME_FA}')
    return meta


def step2_run_mmseqs():
    """All-vs-all mmseqs2 easy-search."""
    print('\n[2] running mmseqs2 all-vs-all (this takes ~5-10 min)...')
    cmd = [
        'mmseqs', 'easy-search',
        str(PROTEOME_FA), str(PROTEOME_FA), str(MMSEQS_RESULT),
        str(WORK_DIR / 'tmp'),
        '--min-seq-id', '0.30',
        '-c', '0.50',
        '-e', '1e-5',
        '--threads', '4',
        '-s', '5.7',
        '--max-seqs', '500',
        '--format-output', 'query,target,fident,alnlen,bits,evalue',
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f'  STDERR (last 30 lines):\n{res.stderr[-3000:]}')
        raise RuntimeError(f'mmseqs2 failed (exit {res.returncode})')
    n_hits = sum(1 for _ in open(MMSEQS_RESULT))
    print(f'  {n_hits} pairwise hits saved to {MMSEQS_RESULT}')


def step3_build_conservation_features(meta: dict) -> dict:
    """Walk mmseqs2 output, compute per-protein conservation metrics."""
    print('\n[3] building conservation features per gene...')

    # For each query, group hits by target organism
    # We track best-hit-per-target-organism for "ortholog" reciprocal logic.
    by_query = defaultdict(lambda: defaultdict(list))  # query -> target_org -> [(target_locus, ident)]

    n_lines = 0
    with open(MMSEQS_RESULT) as f:
        for line in f:
            n_lines += 1
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            q, t, ident = parts[0], parts[1], float(parts[2])
            if q == t: continue  # self-hit
            try:
                q_org, q_locus = q.split('|', 1)
                t_org, t_locus = t.split('|', 1)
            except ValueError:
                continue
            by_query[q][t_org].append((t_locus, ident, t))

    print(f'  parsed {n_lines} hit lines')

    # Reciprocal-best-hit (RBH) detection: A's best hit in B must hit A back as B's best hit in A.
    # For efficiency, for each query find best hit in each target organism. Then check
    # the symmetric condition: target's best hit in query's organism is the query.
    best_hits = {}  # (q_org, q_locus, t_org) -> (t_locus, ident)
    for q, targets in by_query.items():
        try: q_org, q_locus = q.split('|', 1)
        except ValueError: continue
        for t_org, hits in targets.items():
            if t_org == q_org:
                continue
            # Best hit in this target organism
            best = max(hits, key=lambda x: x[1])
            best_hits[(q_org, q_locus, t_org)] = best  # (t_locus, ident, full_target_id)

    # Now check for RBH: gene A in org Q, best hit B in org T.
    # Look up (t_org=Q's gene from B's perspective) -> is it A?
    rbh_pairs = set()  # set of (q_org, q_locus, t_org, t_locus)
    for (q_org, q_locus, t_org), (t_locus, _, _) in best_hits.items():
        # Find what t's best hit in q_org is
        sym = best_hits.get((t_org, t_locus, q_org))
        if sym is not None and sym[0] == q_locus:
            rbh_pairs.add((q_org, q_locus, t_org, t_locus))

    print(f'  RBH ortholog pairs: {len(rbh_pairs):,}')

    # Per-gene conservation features
    organisms_list = sorted(GB_PATHS.keys())
    feats = {}
    for (org, locus) in meta:
        # Find all RBH partners for this gene
        partners = [(to, tl) for (qo, ql, to, tl) in rbh_pairs
                     if qo == org and ql == locus]
        partner_orgs = {to for to, _ in partners}

        n_total = len(partner_orgs)
        n_mollicutes = sum(1 for o in partner_orgs if o in MOLLICUTES)

        # Mean identity to orthologs
        ident_vals = []
        for to, tl in partners:
            bh = best_hits.get((org, locus, to))
            if bh is not None:
                ident_vals.append(bh[1])
        mean_ident = float(np.mean(ident_vals)) if ident_vals else 0.0

        # Paralog count: best_hits within own organism (excluding self)
        # We need to look at ALL by_query[org|locus] hits in own organism
        own_hits = by_query.get(f'{org}|{locus}', {}).get(org, [])
        paralog_count = sum(1 for tl, _, _ in own_hits if tl != locus)

        f = {
            'n_orthologs_total': n_total,
            'n_orthologs_within_mollicutes': n_mollicutes,
            'paralog_count_internal': paralog_count,
            'mean_identity_to_orthologs': mean_ident,
        }
        # 16 binary phyletic profile (1 if RBH present in that organism)
        for o in organisms_list:
            if o == org: continue
            f[f'has_ortholog_in_{o}'] = float(o in partner_orgs)
        feats[(org, locus)] = f

    # Sanity check
    print('\n  --- sanity check ---')
    for org, locus in [('syn3a', 'JCVISYN3A_0001'), ('mpne', 'MPN001')]:
        if (org, locus) in feats:
            f = feats[(org, locus)]
            print(f'  {org}/{locus} (dnaA): orthologs in {f["n_orthologs_total"]}/16 organisms, '
                  f'mean_ident={f["mean_identity_to_orthologs"]:.2f}')

    return feats, organisms_list


def step5_join_to_v25_csv(conservation_feats, organisms_list):
    """Augment the v25 feature CSV with conservation features."""
    print('\n[4] joining onto v25 feature matrix...')
    df = pd.read_csv(INP_FEATURES)
    print(f'  base rows: {len(df)}')

    new_cols = ['n_orthologs_total', 'n_orthologs_within_mollicutes',
                 'paralog_count_internal', 'mean_identity_to_orthologs']
    for o in organisms_list:
        new_cols.append(f'has_ortholog_in_{o}')
    for c in new_cols:
        df[c] = 0.0

    n_matched = 0
    for idx, row in df.iterrows():
        key = (row['organism'], row['locus_tag'])
        gf = conservation_feats.get(key)
        if not gf: continue
        for c in new_cols:
            if c in gf:
                df.at[idx, c] = gf[c]
        n_matched += 1
    print(f'  matched {n_matched}/{len(df)} rows')

    # Diagnostic per organism
    print('\n  per-organism conservation distribution:')
    for org in df.organism.unique():
        sub = df[df.organism == org]
        print(f'    {org:15s}: n_genes={len(sub):4d}  '
              f'mean_n_orthologs={sub.n_orthologs_total.mean():.2f}  '
              f'median_paralog_count={sub.paralog_count_internal.median():.1f}')

    # Sanity: do essentials have higher conservation than nonessentials?
    print('\n  conservation by essential class (within each organism):')
    print(f'  {"organism":<15} {"ess_n":>5} {"ess_n_orth":>10} {"ne_n":>5} {"ne_n_orth":>10} {"diff":>8}')
    for org in df.organism.unique():
        sub = df[df.organism == org]
        if sub.essential.nunique() < 2: continue
        ess_mean = sub[sub.essential == 1].n_orthologs_total.mean()
        ne_mean = sub[sub.essential == 0].n_orthologs_total.mean()
        print(f'  {org:<15} {(sub.essential==1).sum():>5} {ess_mean:>10.2f} '
              f'{(sub.essential==0).sum():>5} {ne_mean:>10.2f}  {ess_mean-ne_mean:>+8.2f}')

    df.to_csv(OUT_FEATURES, index=False)
    print(f'\nwrote {OUT_FEATURES}')
    return df


def main():
    if not PROTEOME_FA.exists():
        meta = step1_extract_proteomes()
    else:
        # Load meta from existing fasta
        meta = {}
        for line in open(PROTEOME_FA):
            if line.startswith('>'):
                h = line[1:].strip()
                try:
                    org, locus = h.split('|', 1)
                    meta[(org, locus)] = 0  # length not needed
                except ValueError:
                    pass
        print(f'[1] reused existing FASTA: {len(meta)} proteins')

    if not MMSEQS_RESULT.exists():
        step2_run_mmseqs()
    else:
        print(f'[2] reused existing mmseqs result: {MMSEQS_RESULT}')

    feats, organisms_list = step3_build_conservation_features(meta)
    df = step5_join_to_v25_csv(feats, organisms_list)
    print(f'\nfinal feature columns ({len(df.columns)}): {list(df.columns)[-22:]}')


if __name__ == '__main__':
    main()

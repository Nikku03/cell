"""Phase 7 — predict per-gene expression rate from sequence using
ORGANISM-SPECIFIC features (addressing the v27 finding that Salmonella-PWMs
don't transfer to other organisms because regulatory grammar varies).

Per-gene features (all organism-specific by construction):

  1. Codon Adaptation Index (CAI)
     Reference set: each organism's ribosomal proteins (typically the
     most highly-expressed genes in any cell). Per-organism codon
     frequencies derived from these. Each gene's CAI = geometric mean
     of relative-synonymous-codon-usage of its codons. Higher CAI -> higher
     expression.

  2. RBS strength (anti-SD complementarity)
     For each organism, extract its anti-Shine-Dalgarno sequence — the
     last ~13 bp at the 3' end of its 16S rRNA gene. Score each gene's
     -20 to -1 bp upstream region for complementary pairing to the anti-SD,
     allowing a 5-15 bp spacer.

  3. Predicted expression rank
     Combined organism-rank score = (CAI_rank + RBS_rank) / 2

  4. Operon leader flag
     The first gene in a same-strand contiguous run is the operon leader;
     its predicted promoter activity drives the whole operon. Mark this.

Output: outputs/multiorg_per_gene_features_with_expression.csv
  augmented v25 features + new expression columns
"""
from __future__ import annotations

import warnings
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
GB_DIR = ROOT / 'memory_bank/data/multiorg_essentiality/raw/multi_organism_genbank'
INP_FEATURES = ROOT / 'outputs/multiorg_per_gene_features.csv'
OUT_FEATURES = ROOT / 'outputs/multiorg_per_gene_features_with_expression.csv'

GB_PATHS = {
    'ccrescentus': GB_DIR / 'ccrescentus_NA1000_CP001340.1.gb',
    'saureus': GB_DIR / 'saureus_N315_BA000018.3.gb',
    'mtuberculosis': GB_DIR / 'mtuberculosis_H37Rv_AL123456.3.gb',
    'styphimurium': GB_DIR / 'styphimurium_LT2_AE006468.2.gb',
    'abaylyi': GB_DIR / 'abaylyi_ADP1_CR543861.1.gb',
    'mpne': ROOT / 'memory_bank/data/multiorg_essentiality/raw/mpneumoniae_M129_NC_000912.gb',
    'syn3a': ROOT / 'cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb',
}

CODONS = [''.join(c) for c in product('ACGT', repeat=3)]
STOP_CODONS = {'TAA', 'TAG', 'TGA'}


# ─────────────── CAI ───────────────


def codon_count(seq: str) -> Counter:
    """Count codons in a CDS sequence (skip incomplete trailing codon)."""
    counts = Counter()
    seq = seq.upper()
    n = (len(seq) // 3) * 3
    for i in range(0, n, 3):
        c = seq[i:i+3]
        if c in CODONS:
            counts[c] += 1
    return counts


def find_ribosomal_protein_cdses(rec) -> list:
    """Identify ribosomal protein CDSes (rpsX, rplX, rpmX) — typical
    high-expression reference set for CAI."""
    out = []
    for f in rec.features:
        if f.type != 'CDS': continue
        product = (f.qualifiers.get('product', [''])[0] or '').lower()
        gene = (f.qualifiers.get('gene', [''])[0] or '').lower()
        if (gene.startswith(('rps', 'rpl', 'rpm')) or
            'ribosomal protein' in product):
            try:
                seq = str(f.extract(rec.seq)).upper()
                if 50 <= len(seq) <= 900:
                    out.append((f, seq))
            except Exception:
                pass
    return out


def build_cai_table(rec) -> dict[str, float]:
    """Build Codon Adaptation Index lookup: codon -> w_codon (relative
    synonymous codon usage normalized within each amino acid). High-expression
    reference set is this organism's ribosomal proteins."""
    rprot = find_ribosomal_protein_cdses(rec)
    if not rprot:
        # Fall back to the full genome's CDSes (less specific)
        rprot = []
        for f in rec.features:
            if f.type != 'CDS': continue
            try:
                seq = str(f.extract(rec.seq)).upper()
                if 100 <= len(seq) <= 1500:
                    rprot.append((f, seq))
            except Exception:
                pass
        rprot = rprot[:50]  # limit

    if not rprot:
        return {c: 1.0 for c in CODONS}

    # Aggregate codon counts across reference set
    agg = Counter()
    for _, seq in rprot:
        agg.update(codon_count(seq))

    # Group codons by amino acid
    aa_to_codons = {}
    for c in CODONS:
        if c in STOP_CODONS: continue
        aa = str(Seq(c).translate())
        aa_to_codons.setdefault(aa, []).append(c)

    # w_codon = count / max_count_in_synonymous_group
    w = {}
    for aa, codons in aa_to_codons.items():
        max_count = max(agg[c] for c in codons) if codons else 1
        max_count = max(max_count, 1)
        for c in codons:
            w[c] = max(agg[c] / max_count, 0.001)  # avoid log(0)
    for c in STOP_CODONS:
        w[c] = 1.0
    return w


def gene_cai(seq: str, w_table: dict) -> float:
    """CAI = geometric mean of w_codon for each codon in the gene."""
    seq = seq.upper()
    n = (len(seq) // 3) * 3
    log_w = []
    for i in range(0, n, 3):
        c = seq[i:i+3]
        if c in STOP_CODONS or c not in w_table: continue
        log_w.append(np.log(w_table[c]))
    if not log_w: return 1.0
    return float(np.exp(np.mean(log_w)))


# ─────────────── RBS strength via anti-SD ───────────────


def find_anti_sd(rec) -> str:
    """Extract the last ~13 bp at the 3' end of a 16S rRNA gene.
    Reverse-complemented this is the anti-Shine-Dalgarno sequence that
    pairs with the gene's RBS."""
    for f in rec.features:
        if f.type != 'rRNA': continue
        product = (f.qualifiers.get('product', [''])[0] or '').lower()
        if '16s' in product:
            try:
                rrna = str(f.extract(rec.seq)).upper()
                # Last 13bp at 3' end
                tail = rrna[-13:]
                # Anti-SD = reverse complement of tail
                anti_sd = str(Seq(tail).reverse_complement())
                return anti_sd
            except Exception:
                pass
    # Fall back to a generic E. coli-style anti-SD
    return 'ACCUCCUUA'.replace('U', 'T')


def rbs_score(upstream_seq: str, anti_sd: str) -> float:
    """Score how RBS-like the upstream region is. anti_sd is the
    reverse-complement of the 16S 3' tail (so it reads as a canonical
    SD template, e.g. 'TAAGGAGG'-like). The score is the max identity
    count between any L-bp window of upstream seq and the SD template.
    Higher = more RBS-like."""
    if not upstream_seq or len(upstream_seq) < 5:
        return 0.0
    best = 0
    L = len(anti_sd)
    region = upstream_seq[-20:] if len(upstream_seq) >= 20 else upstream_seq
    for offset in range(len(region) - L + 1):
        sub = region[offset:offset+L]
        # Identity match against the SD template (Watson-Crick pairing
        # already handled by reverse_complementing the 16S tail upstream).
        matches = sum(1 for i in range(L) if sub[i] == anti_sd[i])
        if matches > best:
            best = matches
    return float(best)


# ─────────────── Per-organism feature extraction ───────────────


def extract_expression_features(organism: str) -> dict:
    """For each CDS in this organism's GenBank, compute CAI + RBS_score.
    Returns {locus_tag: feature_dict}.  Locus tag policy matches the rest of
    the pipeline (saureus/styphimurium use gene symbol)."""
    gb_path = GB_PATHS[organism]
    if not gb_path.exists():
        return {}
    rec = next(SeqIO.parse(str(gb_path), 'genbank'))
    print(f'  parsing {organism}: building CAI table from ribosomal proteins...')

    # Per-organism CAI weights
    w_table = build_cai_table(rec)
    n_rprot = len(find_ribosomal_protein_cdses(rec))
    print(f'    CAI ref set: {n_rprot} ribosomal proteins')

    # Per-organism anti-SD
    anti_sd = find_anti_sd(rec)
    print(f'    anti-SD: {anti_sd}')

    # Score each CDS
    out = {}
    for f in rec.features:
        if f.type != 'CDS': continue
        # Locus_tag policy
        locus = f.qualifiers.get('locus_tag', [None])[0]
        gene_sym = (f.qualifiers.get('gene', [''])[0] or '').strip()
        if organism == 'mpne':
            ot = f.qualifiers.get('old_locus_tag', [None])[0]
            if ot: locus = ot
        elif organism == 'saureus':
            locus = gene_sym if gene_sym else locus
        elif organism == 'styphimurium':
            locus = gene_sym if gene_sym else locus
        if not locus: continue

        try:
            seq = str(f.extract(rec.seq)).upper()
        except Exception:
            continue
        if len(seq) < 30: continue

        # Upstream 30bp for RBS scoring
        strand = int(f.location.strand) if f.location.strand else 1
        if strand == 1:
            up_start = max(0, int(f.location.start) - 30)
            up_end = int(f.location.start)
            ups = str(rec.seq[up_start:up_end]).upper()
        else:
            up_start = int(f.location.end)
            up_end = min(len(rec.seq), int(feature.location.end) + 30) \
                if False else min(len(rec.seq), int(f.location.end) + 30)
            ups = str(rec.seq[up_start:up_end].reverse_complement()).upper()

        cai = gene_cai(seq, w_table)
        rbs = rbs_score(ups, anti_sd)
        out[locus] = {
            'cai': float(cai),
            'rbs_strength': float(rbs),
        }

    # Within-organism ranks (so ranks are comparable across organisms)
    cai_vals = sorted([f['cai'] for f in out.values()])
    rbs_vals = sorted([f['rbs_strength'] for f in out.values()])
    cai_rank = {v: i / max(len(cai_vals) - 1, 1) for i, v in enumerate(cai_vals)}
    rbs_rank = {v: i / max(len(rbs_vals) - 1, 1) for i, v in enumerate(rbs_vals)}
    for k, v in out.items():
        v['cai_rank'] = float(cai_rank[v['cai']])
        v['rbs_rank'] = float(rbs_rank[v['rbs_strength']])
        v['expression_rank'] = float((v['cai_rank'] + v['rbs_rank']) / 2)
    return out


def main():
    print('[1] computing per-organism expression features...')
    organism_feats = {}
    for org in ['ccrescentus','saureus','mtuberculosis','styphimurium',
                'abaylyi','mpne','syn3a']:
        feats = extract_expression_features(org)
        organism_feats[org] = feats
        if feats:
            cais = [f['cai'] for f in feats.values()]
            rbs = [f['rbs_strength'] for f in feats.values()]
            print(f'    {org}: {len(feats)} CDSes;  '
                  f'CAI median={np.median(cais):.3f}, range=[{min(cais):.2f}, {max(cais):.2f}];  '
                  f'RBS median={np.median(rbs):.1f}, range=[{min(rbs):.0f}, {max(rbs):.0f}]')

    print('\n[2] joining onto v25 feature matrix...')
    df = pd.read_csv(INP_FEATURES)
    print(f'  base rows: {len(df)}')

    new_cols = ['cai', 'rbs_strength', 'cai_rank', 'rbs_rank', 'expression_rank']
    for c in new_cols:
        df[c] = 0.5  # default neutral rank for unmatched

    n_matched = 0
    for idx, row in df.iterrows():
        org_feats = organism_feats.get(row['organism'])
        if not org_feats: continue
        gf = org_feats.get(row['locus_tag'])
        if not gf: continue
        for c in new_cols:
            if c in gf:
                df.at[idx, c] = gf[c]
        n_matched += 1
    print(f'  matched {n_matched}/{len(df)} rows')

    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FEATURES, index=False)
    print(f'\nwrote {OUT_FEATURES}')


if __name__ == '__main__':
    main()

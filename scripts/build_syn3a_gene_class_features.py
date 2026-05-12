"""Parse syn3A.gb -> per-locus annotation features for downstream
essentiality classifiers.

Output columns (saved to memory_bank/data/syn3a_gene_class_features.csv):

  locus_tag         JCVISYN3A_LLLL
  gene_name         e.g. dnaA (NaN if unnamed)
  product           text from /product
  feature_type      CDS / rRNA / tRNA / ncRNA
  length_bp         nt length of the gene feature
  protein_length_aa AA length of the translation (NaN for RNA features)
  strand            +1 / -1
  start_bp          1-based start coordinate
  rel_position      start_bp / 543379  (position on the 543kb circle)
  gravy             Kyte-Doolittle GRAVY (mean hydrophobicity, NaN for RNA)
  pI                isoelectric point (NaN for RNA)
  net_charge_ph7    net charge at pH 7 (NaN for RNA)
  aromaticity       fraction F/Y/W (NaN for RNA)
  has_signal_pep    first 20 AAs have GRAVY>1 and at least 5 hydrophobics
  tm_helix_count    number of 18+ consecutive hydrophobic AA runs
  ... + 20 amino-acid composition columns (aa_A, aa_C, ...)
  ... + 14 functional-keyword bool columns (is_ribosomal, is_synthetase, ...)

This is purely descriptive; no Breuer info is used. The file is checked
in so downstream essentiality tests can stack these features against
the v15 / LGNN / variance signals.
"""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

GB_PATH = Path('cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb')
OUT_PATH = Path('memory_bank/data/syn3a_gene_class_features.csv')

# Kyte-Doolittle hydrophobicity index
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'E':-3.5,'Q':-3.5,
      'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
      'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}

AAS = list('ACDEFGHIKLMNPQRSTVWY')

# Functional-keyword categories (case-insensitive substring match on /product)
KEYWORD_CATEGORIES = {
    'is_ribosomal':      [r'\bribosom', r'\brRNA', r'\b(rps|rpl|rpm)\b'],
    'is_synthetase':     [r'tRNA ligase', r'synthetase', r'aminoacyl-tRNA'],
    'is_trna':           [r'\btRNA\b'],
    'is_polymerase':     [r'polymerase', r'RNA pol', r'DNA pol'],
    'is_dna_machinery':  [r'replication', r'topoisomerase', r'gyrase',
                          r'helicase', r'primase', r'ligase', r'recomb',
                          r'repair', r'\bdna[A-Z]\b'],
    'is_translation':    [r'translation', r'initiation factor',
                          r'elongation factor', r'release factor',
                          r'ribosome recycling'],
    'is_transcription':  [r'transcription', r'sigma factor', r'nusG',
                          r'NusA', r'termin'],
    'is_membrane':       [r'membrane', r'transporter', r'permease',
                          r'translocase', r'secY', r'secA'],
    'is_atp_binding':    [r'ATP-binding', r'ATPase'],
    'is_kinase':         [r'kinase'],
    'is_phosphatase':    [r'phosphatase'],
    'is_synthase':       [r'synthase'],
    'is_uncharacterized':[r'[Uu]ncharacteriz', r'[Hh]ypothetical',
                          r'putative protein', r'unknown function'],
    'is_metabolism':     [r'dehydrogenase', r'reductase', r'isomerase',
                          r'transferase', r'oxidase', r'mutase',
                          r'carboxylase', r'aldolase', r'epimerase'],
}


def keyword_flags(product: str) -> dict:
    p = (product or '').lower()
    out = {}
    for cat, patterns in KEYWORD_CATEGORIES.items():
        out[cat] = int(any(re.search(pat, p, flags=re.IGNORECASE)
                            for pat in patterns))
    return out


def hydrophobic_run_count(seq: str, window: int = 18, kd_thresh: float = 1.6
                           ) -> int:
    """Count non-overlapping runs of `window` consecutive AAs whose mean
    Kyte-Doolittle is > kd_thresh. Rough TM-helix proxy.
    """
    if len(seq) < window:
        return 0
    runs = 0
    i = 0
    while i <= len(seq) - window:
        sub = seq[i:i + window]
        avg = sum(KD.get(a, 0) for a in sub) / window
        if avg > kd_thresh:
            runs += 1
            i += window
        else:
            i += 1
    return runs


def signal_peptide_flag(seq: str) -> int:
    if len(seq) < 20:
        return 0
    first20 = seq[:20]
    avg = sum(KD.get(a, 0) for a in first20) / 20
    n_hydro = sum(1 for a in first20 if KD.get(a, 0) > 1.5)
    return int(avg > 1.0 and n_hydro >= 5)


def parse() -> pd.DataFrame:
    rec = next(SeqIO.parse(GB_PATH, 'genbank'))
    genome_length = len(rec.seq)
    print(f'Genome length: {genome_length} bp')

    rows = []
    for feat in rec.features:
        if feat.type not in ('CDS', 'rRNA', 'tRNA', 'ncRNA'):
            continue
        q = feat.qualifiers
        if 'locus_tag' not in q:
            continue
        locus_tag = q['locus_tag'][0]
        gene_name = q.get('gene', [None])[0]
        product   = q.get('product', [''])[0]
        protein_seq = q.get('translation', [None])[0]
        start_bp  = int(feat.location.start) + 1
        end_bp    = int(feat.location.end)
        strand    = int(feat.location.strand) if feat.location.strand else 0
        length_bp = end_bp - start_bp + 1
        rel_pos   = start_bp / genome_length

        row = {
            'locus_tag':         locus_tag,
            'gene_name':         gene_name,
            'product':           product,
            'feature_type':      feat.type,
            'length_bp':         length_bp,
            'strand':            strand,
            'start_bp':          start_bp,
            'rel_position':      rel_pos,
            'has_gene_name':     int(gene_name is not None),
        }

        if protein_seq:
            try:
                pa = ProteinAnalysis(protein_seq.replace('*', ''))
                aa_pct = pa.get_amino_acids_percent()
            except Exception:
                pa = None; aa_pct = {a: 0.0 for a in AAS}
            row['protein_length_aa'] = len(protein_seq)
            if pa is not None:
                try:    row['gravy'] = pa.gravy()
                except: row['gravy'] = float('nan')
                try:    row['pI'] = pa.isoelectric_point()
                except: row['pI'] = float('nan')
                try:    row['aromaticity'] = pa.aromaticity()
                except: row['aromaticity'] = float('nan')
                try:    row['net_charge_ph7'] = pa.charge_at_pH(7.0)
                except: row['net_charge_ph7'] = float('nan')
            else:
                row.update(gravy=float('nan'), pI=float('nan'),
                            aromaticity=float('nan'),
                            net_charge_ph7=float('nan'))
            row['has_signal_pep']  = signal_peptide_flag(protein_seq)
            row['tm_helix_count']  = hydrophobic_run_count(protein_seq)
            for a in AAS:
                row[f'aa_{a}'] = aa_pct.get(a, 0.0)
        else:
            row['protein_length_aa'] = float('nan')
            row['gravy'] = float('nan')
            row['pI'] = float('nan')
            row['aromaticity'] = float('nan')
            row['net_charge_ph7'] = float('nan')
            row['has_signal_pep'] = 0
            row['tm_helix_count'] = 0
            for a in AAS:
                row[f'aa_{a}'] = 0.0

        row.update(keyword_flags(product))
        rows.append(row)

    df = pd.DataFrame(rows)
    # Locus tags can repeat (gene + CDS for same locus). Keep CDS preferentially,
    # then rRNA/tRNA/ncRNA. Drop any duplicates.
    df['_priority'] = df['feature_type'].map(
        {'CDS': 0, 'rRNA': 1, 'tRNA': 2, 'ncRNA': 3},
    ).fillna(99).astype(int)
    df = df.sort_values(['locus_tag', '_priority']).drop_duplicates(
        'locus_tag', keep='first',
    ).drop(columns=['_priority']).reset_index(drop=True)
    print(f'Extracted {len(df)} unique loci')
    return df


if __name__ == '__main__':
    df = parse()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f'wrote {len(df)} rows × {df.shape[1]} cols -> {OUT_PATH}')

    # Summary
    print(f'\nFeature-type counts:')
    print(df['feature_type'].value_counts())
    print(f'\nGenes with sequence-derived features (have translation): '
          f'{df["protein_length_aa"].notna().sum()}')
    print(f'\nKeyword-category positive counts:')
    for cat in KEYWORD_CATEGORIES:
        n = int(df[cat].sum())
        print(f'  {cat:24s} {n:4d}')

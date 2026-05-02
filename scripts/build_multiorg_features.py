"""Phase 2 — build per-gene structured feature matrix across 6 bacteria
plus Syn3A test set, for the LNN cross-organism essentiality experiment.

Sources
-------

Training organisms (essentiality + GenBank):
  ccrescentus  — Christen 2011 MSB DT2_Essential_ORFs (3876 ORFs)
  saureus      — Chaudhuri 2009 BMC Genomics (466 NCTC genes)
  mtuberculosis — DeJesus 2017 mBio st3 Final Call (4019 ORFs)
  styphimurium — Barquist 2013 NAR TraDIS SL1344 protein (4530 ORFs)
  abaylyi      — DeBerardinis 2008 MSB s2.xls (deletion mutants)
  mpne         — Lluch-Senar 2015 MSB labels.csv (737 genes)

Test organism:
  syn3a        — Breuer 2019 eLife labels.csv (455 genes)

Per-gene features (cross-organism portable)
-------------------------------------------

  log_length_bp         numeric
  gc                    fraction of GC in CDS
  upstream_gc           GC content of 100bp upstream of CDS start
  upstream_at_skew      AT skew of 100bp upstream
  position_norm         CDS midpoint / genome length (origin-rel)
  operon_run_length     length of same-strand contiguous CDS run
  is_hypothetical       /product contains 'hypothetical'
  is_uncharacterized    /product contains 'uncharacterized'
  is_putative           /product contains 'putative'
  kw_replication        DNA pol / helicase / gyrase / topo
  kw_transcription      RNA pol / sigma / rho / transcription
  kw_translation        ribosomal / tRNA / aminoacyl / elongation
  kw_atp                ATP synthase / F0F1
  kw_membrane           membrane / transporter / permease
  kw_synthase           synthase / synthetase
  kw_kinase             kinase
  kw_dehydrogenase      dehydrogenase / reductase / oxidase
  kw_protease           protease / peptidase
  kw_rrna               rRNA / 23S / 16S / 5S
  kw_chaperone          chaperone / DnaK / GroEL

Output
------

  outputs/multiorg_per_gene_features.csv
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

warnings.filterwarnings('ignore', message='Unknown extension is not supported')
warnings.filterwarnings('ignore', message='.*UserWarning.*')

ROOT = Path(__file__).resolve().parent.parent
SUPP = ROOT / 'memory_bank/data/multiorg_essentiality/raw/multi_organism_essentiality_supp'
GB = ROOT / 'memory_bank/data/multiorg_essentiality/raw/multi_organism_genbank'
OUT = ROOT / 'outputs/multiorg_per_gene_features.csv'

KEYWORDS = {
    'kw_replication': ['replication', 'polymerase', 'helicase', 'gyrase', 'topoisomerase'],
    'kw_transcription': ['rna polymerase', 'sigma', 'transcription', 'rho '],
    'kw_translation': ['ribosomal', 'trna', 'aminoacyl', 'elongation factor'],
    'kw_atp': ['atp synthase', 'f0f1', 'f-atpase'],
    'kw_membrane': ['membrane', 'transporter', 'permease', 'abc '],
    'kw_synthase': ['synthase', 'synthetase'],
    'kw_kinase': ['kinase'],
    'kw_dehydrogenase': ['dehydrogenase', 'reductase', 'oxidase'],
    'kw_protease': ['protease', 'peptidase'],
    'kw_rrna': ['rrna', '23s', '16s', '5s'],
    'kw_chaperone': ['chaperone', 'dnak', 'groel'],
}


# ──────────────────────────── essentiality parsers ─────────────────────────


def _unzip(zip_name: str) -> Path:
    """Extract a supp ZIP to /tmp/supp_unpacked/<name> if not already."""
    import zipfile
    target = Path('/tmp/supp_unpacked') / Path(zip_name).stem
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(SUPP / zip_name) as zf:
            zf.extractall(target)
    return target


def parse_christen_ccrescentus() -> pd.DataFrame:
    p = _unzip('christen_2011_ccrescentus_PMC3202797.zip') / 'msb201158-s2.xls'
    df = pd.read_excel(p, sheet_name='DT2_Essential_ORFs')
    df = df.rename(columns={'CCNA_number': 'locus_tag', 'essentiality': 'ess_call',
                            'annotation': 'product'})
    df['organism'] = 'ccrescentus'
    df['gene_name'] = ''
    # Christen uses lowercase 'essential' and 'Nonessential'; 'High_Fitness_Costs'
    # is an intermediate category — drop it.
    norm = df['ess_call'].astype(str).str.strip().str.lower()
    df['essential'] = norm.map(lambda v: 1 if v == 'essential'
                                else (0 if v == 'nonessential' else None))
    df = df.dropna(subset=['essential', 'locus_tag'])
    df['essential'] = df['essential'].astype(int)
    return df[['organism', 'locus_tag', 'gene_name', 'essential']]


def parse_chaudhuri_saureus() -> pd.DataFrame:
    """Chaudhuri's NCTC gene column is SAOUHSC_#### (NCTC 8325 strain).
    Our GenBank is N315 (BA000018.3) which uses note=ORFID:SA####.
    Match via the N315 gene column (gene symbol) which is present in the supp.
    Locus tag in the unified dataframe will be the gene symbol; the GenBank
    parser will index by gene symbol for saureus."""
    p = _unzip('chaudhuri_2009_saureus_PMC2721850.zip') / '1471-2164-10-291-S1.xls'
    df = pd.read_excel(p)
    df = df.rename(columns={'Essential in S.aureus': 'ess_call'})
    df['organism'] = 'saureus'
    # Use the gene symbol column (N315 gene) as the locus_tag for matching.
    # Some rows have only NCTC gene; fall back to that (will fail join but
    # we keep the row for downstream).
    df['locus_tag'] = df['N315 gene'].fillna(df.get('NCTC gene', '')).astype(str).str.strip()
    df = df[df['locus_tag'].str.len() > 0]
    df['gene_name'] = df['locus_tag']
    df['essential'] = df['ess_call'].astype(str).str.strip().str.lower().map(
        {'y': 1, 'n': 0}).where(df['ess_call'].astype(str).str.strip().str.lower().isin(['y', 'n']))
    df = df.dropna(subset=['essential', 'locus_tag'])
    df['essential'] = df['essential'].astype(int)
    return df[['organism', 'locus_tag', 'gene_name', 'essential']]


def parse_dejesus_mtb() -> pd.DataFrame:
    p = _unzip('dejesus_2017_mtuberculosis_PMC5241402.zip') / 'mbo002173137st3.xlsx'
    df = pd.read_excel(p, skiprows=1)
    df = df.rename(columns={'ORF ID': 'locus_tag', 'Name': 'gene_name',
                            'Final Call': 'ess_call'})
    df['organism'] = 'mtuberculosis'
    # Map ES/ESD -> essential, NE -> nonessential, drop GD/GA/Uncertain
    df['essential'] = df['ess_call'].astype(str).str.strip().map(
        lambda v: 1 if v in ('ES', 'ESD') else (0 if v == 'NE' else None))
    df = df.dropna(subset=['essential', 'locus_tag'])
    df['essential'] = df['essential'].astype(int)
    df['gene_name'] = df['gene_name'].fillna('').astype(str)
    return df[['organism', 'locus_tag', 'gene_name', 'essential']]


def parse_barquist_styphimurium() -> pd.DataFrame:
    """Barquist's SL1344 panel doesn't share locus tags with our LT2 GenBank.
    Match via gene symbol — both strains have the same gene names for
    orthologs. Many genes will have empty gene_name and be lost; that's the
    cost of not having SL1344 GenBank on disk."""
    p = (_unzip('barquist_2013_styphimurium_PMC3632133.zip')
         / 'supp_gkt148_nar-02998-f-2012-File007.xlsx')
    df = pd.read_excel(p, sheet_name='TraDIS screen of SL1344 protein')
    df = df.rename(columns={'stm gene symbol': 'gene_name'})
    df['organism'] = 'styphimurium'
    df['gene_name'] = df['gene_name'].fillna('').astype(str).str.strip()
    df['locus_tag'] = df['gene_name']  # match key = gene symbol
    df = df[df['locus_tag'].str.len() > 0]
    ii = pd.to_numeric(df['insertion index'], errors='coerce')
    pv = pd.to_numeric(df['p-value'], errors='coerce')
    df['essential'] = ((ii < 0.005) & (pv < 0.05)).astype(int)
    return df[['organism', 'locus_tag', 'gene_name', 'essential']]


def parse_deberardinis_abaylyi() -> pd.DataFrame:
    p = _unzip('deberardinis_2008_abaylyi_PMC2290942.zip') / 'msb200810-s2.xls'
    # Header is multi-row; data starts deeper. Probe for the row containing 'Gene' or 'ACIAD'.
    raw = pd.read_excel(p, header=None, sheet_name=0)
    # Find header row: the one that contains 'Gene' as a header-like token
    header_row = None
    for i in range(min(20, len(raw))):
        row_vals = [str(v).strip() for v in raw.iloc[i].tolist()]
        if any('Gene name' in v or 'gene name' in v.lower() for v in row_vals):
            header_row = i; break
        if any(v.lower() in ('locus', 'orf', 'gene') for v in row_vals):
            header_row = i; break
    if header_row is None:
        # fallback: row containing 'ACIAD' is data; header is the row above
        for i in range(len(raw)):
            if any('ACIAD' in str(v) for v in raw.iloc[i].tolist()):
                header_row = max(0, i - 1); break
    if header_row is None:
        return pd.DataFrame(columns=['organism', 'locus_tag', 'gene_name', 'essential'])
    df = pd.read_excel(p, sheet_name=0, skiprows=header_row)
    df.columns = [str(c).strip() for c in df.columns]
    # Heuristic match: locus_tag column = whichever contains 'ACIAD'
    locus_col = next((c for c in df.columns if df[c].astype(str).str.contains('ACIAD').any()), None)
    if locus_col is None:
        return pd.DataFrame(columns=['organism', 'locus_tag', 'gene_name', 'essential'])
    df['locus_tag'] = df[locus_col].astype(str).str.strip()
    # essentiality column: contains 'Ess'/'D'/'DB mutant'
    ess_col = None
    for c in df.columns:
        vals = df[c].astype(str).str.strip().str.lower()
        if vals.isin(['ess', 'd', 'db mutant', 'n.a', 'n.a.', 't.c.', 't.c', 'na']).sum() > 100:
            ess_col = c; break
    if ess_col is None:
        return pd.DataFrame(columns=['organism', 'locus_tag', 'gene_name', 'essential'])
    df['ess_call'] = df[ess_col].astype(str).str.strip().str.lower()
    df['essential'] = df['ess_call'].map(
        lambda v: 1 if v == 'ess' else (0 if v == 'd' else None))
    df = df.dropna(subset=['essential', 'locus_tag'])
    df = df[df['locus_tag'].str.startswith('ACIAD')]
    df['essential'] = df['essential'].astype(int)
    df['gene_name'] = ''
    df['organism'] = 'abaylyi'
    return df[['organism', 'locus_tag', 'gene_name', 'essential']]


def parse_mpne_syn3a_from_labels() -> pd.DataFrame:
    df = pd.read_csv(ROOT / 'memory_bank/data/multiorg_essentiality/labels.csv')
    df = df[df['organism'].isin(['mpne', 'syn3a'])].copy()
    df['gene_name'] = df['gene_name'].fillna('').astype(str)
    return df[['organism', 'locus_tag', 'gene_name', 'essential']]


# ──────────────────────────── GenBank feature extractor ─────────────────────


def _genbank_for(organism: str) -> Path:
    candidates = {
        'ccrescentus': 'ccrescentus_NA1000_CP001340.1.gb',
        'saureus': 'saureus_N315_BA000018.3.gb',
        'mtuberculosis': 'mtuberculosis_H37Rv_AL123456.3.gb',
        'styphimurium': 'styphimurium_LT2_AE006468.2.gb',
        'abaylyi': 'abaylyi_ADP1_CR543861.1.gb',
        'mpne': '../mpneumoniae_M129_NC_000912.gb',
        'syn3a': None,  # special-cased below
    }
    name = candidates.get(organism)
    if name is None:
        return None
    return GB / name


def _extract_genbank_features(gb_path: Path, organism: str) -> dict[str, dict]:
    """Return {locus_tag: feature_dict} for one organism."""
    if organism == 'syn3a':
        gb_path = ROOT / 'cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb'
    print(f'  parsing {organism} GenBank: {gb_path}')
    rec = next(SeqIO.parse(str(gb_path), 'genbank'))
    genome_seq = str(rec.seq).upper()
    genome_len = len(genome_seq)

    # First pass: collect CDSes with start, end, strand, locus_tag
    cdses = []
    for f in rec.features:
        if f.type != 'CDS': continue
        locus = f.qualifiers.get('locus_tag', [None])[0]
        gene_sym = (f.qualifiers.get('gene', [''])[0] or '').strip()
        # Per-organism locus_tag policy:
        if organism == 'mpne':
            ot = f.qualifiers.get('old_locus_tag', [None])[0]
            if ot: locus = ot
        elif organism == 'saureus':
            # N315 GenBank stores ORFID in /note like 'ORFID:SA0001'.
            # We index BY GENE SYMBOL because Chaudhuri matches via N315 gene
            # symbol; ORFID is captured for traceability but not the match key.
            note = (f.qualifiers.get('note', [''])[0] or '')
            m = re.search(r'ORFID:([A-Za-z0-9_.-]+)', note)
            if m:
                _orfid = m.group(1)
            else:
                _orfid = None
            locus = gene_sym if gene_sym else _orfid
        elif organism == 'styphimurium':
            # SL1344 supp -> LT2 GenBank: cross-strain. Match via gene symbol.
            locus = gene_sym if gene_sym else locus
        if not locus: continue
        start = int(f.location.start)
        end = int(f.location.end)
        strand = int(f.location.strand) if f.location.strand else 1
        cdses.append({
            'locus_tag': locus,
            'gene_name': gene_sym,
            'product': (f.qualifiers.get('product', [''])[0] or '').strip(),
            'start': start, 'end': end, 'strand': strand,
            'feature': f,
        })
    cdses.sort(key=lambda d: d['start'])

    # Second pass: compute features per CDS
    out = {}
    for i, c in enumerate(cdses):
        seq = str(c['feature'].extract(rec.seq)).upper()
        if not seq: continue
        gc = (seq.count('G') + seq.count('C')) / len(seq)
        # Upstream 100bp
        if c['strand'] == 1:
            up_start = max(0, c['start'] - 100); up_end = c['start']
            up_seq = genome_seq[up_start:up_end]
        else:
            up_start = c['end']; up_end = min(genome_len, c['end'] + 100)
            up_seq = str(rec.seq[up_start:up_end].reverse_complement()).upper()
        if up_seq:
            up_gc = (up_seq.count('G') + up_seq.count('C')) / len(up_seq)
            up_a, up_t = up_seq.count('A'), up_seq.count('T')
            up_at_skew = (up_a - up_t) / max(up_a + up_t, 1)
        else:
            up_gc = 0.0; up_at_skew = 0.0
        # Operon proxy: same-strand contiguous run length
        run_len = 1
        # forward through neighbors with same strand and gap < 100bp
        for j in range(i + 1, len(cdses)):
            if cdses[j]['strand'] != c['strand']: break
            if cdses[j]['start'] - cdses[j-1]['end'] > 100: break
            run_len += 1
        # backward
        for j in range(i - 1, -1, -1):
            if cdses[j]['strand'] != c['strand']: break
            if cdses[j+1]['start'] - cdses[j]['end'] > 100: break
            run_len += 1

        product_l = c['product'].lower()
        feats = {
            'log_length_bp': float(np.log1p(len(seq))),
            'gc': float(gc),
            'upstream_gc': float(up_gc),
            'upstream_at_skew': float(up_at_skew),
            'position_norm': float((c['start'] + c['end']) / 2 / genome_len),
            'operon_run_length': float(run_len),
            'is_hypothetical': float('hypothetical' in product_l),
            'is_uncharacterized': float('uncharacterized' in product_l),
            'is_putative': float('putative' in product_l),
            'gene_name_gb': c['gene_name'],
            'product_gb': c['product'][:80],
        }
        for kw_name, terms in KEYWORDS.items():
            feats[kw_name] = float(any(t in product_l for t in terms))
        out[c['locus_tag']] = feats
    return out


# ──────────────────────────── main ────────────────────────────


def main():
    print('[1] parsing essentiality calls per organism...')
    parsers = [
        ('christen_ccrescentus', parse_christen_ccrescentus),
        ('chaudhuri_saureus', parse_chaudhuri_saureus),
        ('dejesus_mtb', parse_dejesus_mtb),
        ('barquist_styphimurium', parse_barquist_styphimurium),
        ('deberardinis_abaylyi', parse_deberardinis_abaylyi),
    ]
    ess_frames = []
    for name, fn in parsers:
        try:
            df = fn()
            print(f'  {name}: {len(df)} rows '
                  f'({df.essential.sum()} ess, {(df.essential==0).sum()} ne)')
            ess_frames.append(df)
        except Exception as e:
            print(f'  {name}: PARSE FAILED — {e}')
    # mpne + syn3a from existing labels.csv
    df = parse_mpne_syn3a_from_labels()
    for org in ['mpne', 'syn3a']:
        sub = df[df.organism == org]
        print(f'  {org}: {len(sub)} rows '
              f'({sub.essential.sum()} ess, {(sub.essential==0).sum()} ne)')
    ess_frames.append(df)
    ess_all = pd.concat(ess_frames, ignore_index=True)
    print(f'\ntotal essentiality rows: {len(ess_all)}')

    print('\n[2] extracting GenBank features per organism...')
    feature_lookup = {}  # (organism, locus_tag) -> features
    for org in ['ccrescentus', 'saureus', 'mtuberculosis', 'styphimurium',
                'abaylyi', 'mpne', 'syn3a']:
        gb = _genbank_for(org)
        if gb is None:
            if org == 'syn3a':
                gb = ROOT / 'cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb'
            else:
                continue
        if not gb.exists():
            print(f'  {org}: GenBank missing at {gb}'); continue
        try:
            feats = _extract_genbank_features(gb, org)
            print(f'  {org}: {len(feats)} CDS features extracted')
            for lt, fd in feats.items():
                feature_lookup[(org, lt)] = fd
        except Exception as e:
            print(f'  {org}: GENBANK PARSE FAILED — {e}')

    print('\n[3] joining essentiality + features...')
    rows = []
    matched_per_org = {}
    for _, r in ess_all.iterrows():
        key = (r['organism'], r['locus_tag'])
        feats = feature_lookup.get(key)
        if feats is None:
            continue
        row = {'organism': r['organism'], 'locus_tag': r['locus_tag'],
               'gene_name': r['gene_name'] or feats.get('gene_name_gb', ''),
               'product': feats.get('product_gb', ''),
               'essential': int(r['essential'])}
        for k, v in feats.items():
            if k in ('gene_name_gb', 'product_gb'): continue
            row[k] = v
        rows.append(row)
        matched_per_org[r['organism']] = matched_per_org.get(r['organism'], 0) + 1

    df_out = pd.DataFrame(rows)
    print(f'\n  joined: {len(df_out)} (essentiality ∩ GenBank)')
    print('  per-organism matched:')
    for org, n in sorted(matched_per_org.items()):
        sub = df_out[df_out.organism == org]
        print(f'    {org:15s} {n:5d}  ({sub.essential.sum()} ess, '
              f'{(sub.essential==0).sum()} ne)')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()

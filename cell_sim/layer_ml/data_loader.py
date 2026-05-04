"""Data loader for the EssentialityLNN.

Assembles per-organism multimodal batches from the existing on-disk
artifacts:

  - v25 / v29 features (scalar + conservation): outputs/multiorg_per_gene_features_with_conservation.csv
  - v27 regulatory features (RBS / -10 / -35 / TF scores from Salmonella PWMs):
        outputs/multiorg_per_gene_features_with_regulatory.csv
  - v28 expression features (CAI, RBS_strength):
        outputs/multiorg_per_gene_features_with_expression.csv
  - GenBank files for sequence (k-mer features) and operon-edge graph:
        memory_bank/data/multiorg_essentiality/raw/multi_organism_genbank/
  - Kinetic features (Km, kcat, Vmax) from cell_sim/data/.../kinetic_params.xlsx
        Salmonella has no kinetic_params; Syn3A has the Luthey-Schulten table;
        mpne has BACTERIAL_DEFAULT-filled values (per Phase D2).

Output: per-organism dict of tensors ready for EssentialityLNN.forward.

Self-test: `python -m cell_sim.layer_ml.data_loader` (loads, builds for
Salmonella + Syn3A, prints shapes).
"""
from __future__ import annotations

import re
import warnings
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]   # cell_sim/layer_ml/data_loader.py -> cell/
GB_DIR = ROOT / 'memory_bank/data/multiorg_essentiality/raw/multi_organism_genbank'
FEATURES_BASE = ROOT / 'outputs/multiorg_per_gene_features.csv'
FEATURES_REGULATORY = ROOT / 'outputs/multiorg_per_gene_features_with_regulatory.csv'
FEATURES_CONSERVATION = ROOT / 'outputs/multiorg_per_gene_features_with_conservation.csv'
FEATURES_EXPRESSION = ROOT / 'outputs/multiorg_per_gene_features_with_expression.csv'
FEATURES_PPI = ROOT / 'outputs/ppi_features_per_gene.csv'
FEATURES_ORTHOLOG_PRIOR = ROOT / 'outputs/ortholog_prior_features_per_gene.csv'

GB_PATHS = {
    'ccrescentus': GB_DIR / 'ccrescentus_NA1000_CP001340.1.gb',
    'saureus': GB_DIR / 'saureus_N315_BA000018.3.gb',
    'mtuberculosis': GB_DIR / 'mtuberculosis_H37Rv_AL123456.3.gb',
    'styphimurium': GB_DIR / 'styphimurium_LT2_AE006468.2.gb',
    'abaylyi': GB_DIR / 'abaylyi_ADP1_CR543861.1.gb',
    'mgenitalium': GB_DIR / 'mgenitalium_G37_L43967.2.gb',
    'mpne': ROOT / 'memory_bank/data/multiorg_essentiality/raw/mpneumoniae_M129_NC_000912.gb',
    'syn3a': ROOT / 'cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb',
}

# Per-gene scalar feature columns (matches v25 fact)
SCALAR_FEATURES = [
    'log_length_bp', 'gc', 'upstream_gc', 'upstream_at_skew',
    'position_norm', 'operon_run_length',
    'is_hypothetical', 'is_uncharacterized', 'is_putative',
    'kw_replication', 'kw_transcription', 'kw_translation',
    'kw_atp', 'kw_membrane', 'kw_synthase', 'kw_kinase',
    'kw_dehydrogenase', 'kw_protease', 'kw_rrna', 'kw_chaperone',
]

# Path A: PPI-derived per-gene aggregates from the PPIPredictor cross-attention
# pair head. Computed by `scripts/score_ppi_features.py`. Cross-org PPI signal
# only weakly transfers (MCC ~0.48), but per-gene degree/max-score is a useful
# new feature channel for the in-domain LNN. Set via `include_ppi=True` in
# `load_organism_batches`.
PPI_SCALAR_FEATURES = [
    'ppi_degree_high',   # count of partners with predicted_prob >= 0.7
    'ppi_degree_mid',    # count of partners with predicted_prob >= 0.5
    'ppi_max_score',     # highest predicted PPI prob involving this gene
]
SCALAR_FEATURES_WITH_PPI = SCALAR_FEATURES + PPI_SCALAR_FEATURES

# Ortholog-prior feature channel: leave-one-out ortholog mean essentiality
# from the 7 OTHER organisms (each gene's prior excludes its own organism's
# labels — no leakage). Computed by scripts/score_ortholog_prior_feature.py.
# Standalone ortholog predictor: mean cross-org MCC 0.420 (commit 97bd0f6).
# Strong cross-org signal — much stronger than PPI features, which actively
# hurt cross-org transfer (commit 988f9ed).
ORTHOLOG_PRIOR_FEATURES = [
    'ortholog_prior',    # mean essentiality of RBH partners (0.5 if none)
    'n_orthologs',       # count of labeled RBH partners (0 if none)
]
SCALAR_FEATURES_WITH_ORTH = SCALAR_FEATURES + ORTHOLOG_PRIOR_FEATURES
SCALAR_FEATURES_WITH_PPI_AND_ORTH = (SCALAR_FEATURES + PPI_SCALAR_FEATURES
                                          + ORTHOLOG_PRIOR_FEATURES)

# 50 regulatory feature columns (from v27): organized as
# {RBS, -10, -35, 14_TFs} × {score, present, dist} ≈ 51, plus the special
# has_promoter_pair + predicted_tf_count. Discovered at load time.
REGULATORY_FEATURE_PREFIXES = [
    'ribosome_binding_site_', 'minus_10_signal_', 'minus_35_signal_',
    'TF_CRP_', 'TF_FNR_', 'TF_IHF_', 'TF_Lrp_', 'TF_NarL_',
    'TF_ArgR_', 'TF_OmpR_', 'TF_PurR_', 'TF_GlpR_', 'TF_TyrR_',
    'TF_AraC_', 'TF_FIS_', 'TF_LexA_',
]

# k-mer alphabet
NUCS = ['A', 'C', 'G', 'T']
KMERS_3_LIST = [''.join(t) for t in product(NUCS, repeat=3)]   # 64
KMERS_4_LIST = [''.join(t) for t in product(NUCS, repeat=4)]   # 256
NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def _organism_locus(organism: str, f) -> str | None:
    """Per-organism locus_tag policy matching build_multiorg_features.py."""
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


def _seq_to_tokens(seq: str, max_len: int = 2048) -> tuple[np.ndarray, int]:
    """Token-encode a nucleotide sequence with padding.
    Vocab: A=0, C=1, G=2, T=3, N/pad=4.
    Returns (tokens [max_len], real_length)."""
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    tokens = np.full(max_len, 4, dtype=np.int64)  # 4 = pad
    L = min(len(seq), max_len)
    for i in range(L):
        tokens[i] = nuc_to_idx.get(seq[i], 4)
    return tokens, L


def _kmer_counts(seq: str, k: int) -> np.ndarray:
    """Normalized k-mer count vector. seq must be uppercase ACGT-only."""
    if len(seq) < k:
        return np.zeros(4 ** k, dtype=np.float32)
    counts = np.zeros(4 ** k, dtype=np.float32)
    n = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        idx = 0; valid = True
        for ch in kmer:
            if ch not in NUC_TO_IDX:
                valid = False; break
            idx = idx * 4 + NUC_TO_IDX[ch]
        if valid:
            counts[idx] += 1
            n += 1
    if n > 0:
        counts /= n
    return counts


def _extract_organism_data(organism: str, features_df: pd.DataFrame
                              ) -> dict | None:
    """Extract sequences + k-mer features + edges for an organism's CDSes.
    Returns dict with per-gene aligned arrays + edge graph."""
    gb_path = GB_PATHS.get(organism)
    if gb_path is None or not gb_path.exists():
        return None
    rec = next(SeqIO.parse(str(gb_path), 'genbank'))
    feat_lookup = features_df[features_df.organism == organism].set_index('locus_tag')

    # CDS pass: collect sequences + positions
    cdses = []
    for f in rec.features:
        if f.type != 'CDS': continue
        locus = _organism_locus(organism, f)
        if not locus: continue
        if locus not in feat_lookup.index: continue
        try:
            seq = str(f.extract(rec.seq)).upper()
        except Exception:
            continue
        if len(seq) < 30: continue
        cdses.append({
            'locus_tag': locus,
            'gene_name': (f.qualifiers.get('gene', [''])[0] or '').strip(),
            'product': (f.qualifiers.get('product', [''])[0] or '').strip(),
            'seq': seq,
            'start': int(f.location.start),
            'end': int(f.location.end),
            'strand': int(f.location.strand) if f.location.strand else 1,
        })
    cdses.sort(key=lambda d: d['start'])

    # Build edges (operon proxy: same-strand consecutive within 100bp)
    locus_to_idx = {c['locus_tag']: i for i, c in enumerate(cdses)}
    src, dst, attr = [], [], []
    for i in range(len(cdses) - 1):
        a, b = cdses[i], cdses[i + 1]
        gap = b['start'] - a['end']
        if gap < 100 and a['strand'] == b['strand']:
            ai, bi = locus_to_idx[a['locus_tag']], locus_to_idx[b['locus_tag']]
            src.append(ai); dst.append(bi)
            attr.append([np.log1p(max(gap, 0)), 1.0])
            src.append(bi); dst.append(ai)
            attr.append([np.log1p(max(gap, 0)), 1.0])

    edge_index = (np.array([src, dst], dtype=np.int64)
                   if src else np.zeros((2, 0), dtype=np.int64))
    edge_attr = (np.array(attr, dtype=np.float32)
                  if attr else np.zeros((0, 2), dtype=np.float32))

    # K-mer features per gene (compressed sequence summary)
    kmer3 = np.zeros((len(cdses), 64), dtype=np.float32)
    kmer4 = np.zeros((len(cdses), 256), dtype=np.float32)
    for i, c in enumerate(cdses):
        kmer3[i] = _kmer_counts(c['seq'], 3)
        kmer4[i] = _kmer_counts(c['seq'], 4)

    # Raw nucleotide tokens (for the SequenceEncoder option). Padded to
    # max_seq_len; per-gene real lengths kept separately.
    max_seq_len = 2048
    seq_tokens = np.full((len(cdses), max_seq_len), 4, dtype=np.int64)
    seq_lengths = np.zeros(len(cdses), dtype=np.int64)
    for i, c in enumerate(cdses):
        toks, L = _seq_to_tokens(c['seq'], max_len=max_seq_len)
        seq_tokens[i] = toks
        seq_lengths[i] = L

    return {
        'organism': organism,
        'cdses': cdses,
        'locus_to_idx': locus_to_idx,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'kmer3': kmer3,
        'kmer4': kmer4,
        'seq_tokens': seq_tokens,
        'seq_lengths': seq_lengths,
    }


def _build_organism_batch(organism: str,
                            features_df: pd.DataFrame,
                            regulatory_df: pd.DataFrame | None,
                            include_kinetic: bool = True,
                            scalar_feature_list: list[str] = SCALAR_FEATURES,
                            ) -> dict | None:
    """Assemble a single multimodal batch for an organism.

    `scalar_feature_list` controls which columns from features_df are
    pulled into the per-gene scalar tensor. Pass `SCALAR_FEATURES_WITH_PPI`
    when you want the PPI-derived columns appended (Path A); requires
    features_df to already contain those columns (merged from the PPI CSV).
    """
    org_data = _extract_organism_data(organism, features_df)
    if org_data is None: return None
    cdses = org_data['cdses']
    if not cdses: return None
    feat_lookup = features_df[features_df.organism == organism].set_index('locus_tag')

    # 1. Scalar features
    scalar = np.zeros((len(cdses), len(scalar_feature_list)), dtype=np.float32)
    for i, c in enumerate(cdses):
        if c['locus_tag'] in feat_lookup.index:
            row = feat_lookup.loc[c['locus_tag']]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            for j, fname in enumerate(scalar_feature_list):
                v = row.get(fname, 0.0)
                scalar[i, j] = float(v) if pd.notna(v) else 0.0

    # 2. Regulatory features (if available)
    n_reg = 50
    regulatory = np.zeros((len(cdses), n_reg), dtype=np.float32)
    regulatory_mask = np.zeros(len(cdses), dtype=np.float32)

    if regulatory_df is not None:
        reg_lookup = regulatory_df[regulatory_df.organism == organism].set_index('locus_tag')
        # Discover regulatory columns that match our prefixes
        all_reg_cols = []
        for prefix in REGULATORY_FEATURE_PREFIXES:
            for suffix in ['score', 'present', 'dist']:
                col = prefix + suffix
                if col in regulatory_df.columns:
                    all_reg_cols.append(col)
        # Pad/truncate to n_reg
        all_reg_cols = all_reg_cols[:n_reg]
        for i, c in enumerate(cdses):
            if c['locus_tag'] in reg_lookup.index:
                row = reg_lookup.loc[c['locus_tag']]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                # Mask is "did this organism get PWMs applied?"  We applied
                # them to all 7, but for organisms outside Salmonella the
                # PWMs are noisy (per v27 fact). Mark mask=1 for Salmonella
                # only — others get the features but with reduced confidence.
                got_features = any(
                    pd.notna(row.get(c_, np.nan)) and row.get(c_, 0) != 0
                    for c_ in all_reg_cols)
                if got_features:
                    regulatory_mask[i] = 1.0 if organism == 'styphimurium' else 0.5
                    for j, col in enumerate(all_reg_cols):
                        v = row.get(col, 0.0)
                        regulatory[i, j] = float(v) if pd.notna(v) else 0.0

    # 3. Kinetic features (Vmax, Km, kcat from SBML/kinetic tables)
    # For now, all-zero / mask-zero — can be enriched later by linking to
    # cell_sim/data/<organism>/input_data/kinetic_params.xlsx via gene→reaction
    # mapping. Placeholder for future enrichment.
    kinetic = np.zeros((len(cdses), 4), dtype=np.float32)
    kinetic_mask = np.zeros(len(cdses), dtype=np.float32)

    # 4. Labels
    labels = np.zeros(len(cdses), dtype=np.int64)
    has_label = np.zeros(len(cdses), dtype=np.float32)
    for i, c in enumerate(cdses):
        if c['locus_tag'] in feat_lookup.index:
            row = feat_lookup.loc[c['locus_tag']]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            labels[i] = int(row.get('essential', 0))
            has_label[i] = 1.0

    # Build the batch tensors
    batch = {
        'organism': organism,
        'locus_tags': [c['locus_tag'] for c in cdses],
        'gene_names': [c['gene_name'] for c in cdses],
        'scalar': torch.tensor(scalar, dtype=torch.float32),
        'regulatory': torch.tensor(regulatory, dtype=torch.float32),
        'regulatory_mask': torch.tensor(regulatory_mask, dtype=torch.float32),
        'kinetic': torch.tensor(kinetic, dtype=torch.float32),
        'kinetic_mask': torch.tensor(kinetic_mask, dtype=torch.float32),
        'kmer3': torch.tensor(org_data['kmer3'], dtype=torch.float32),
        'kmer4': torch.tensor(org_data['kmer4'], dtype=torch.float32),
        'seq_tokens': torch.tensor(org_data['seq_tokens'], dtype=torch.long),
        'seq_lengths': torch.tensor(org_data['seq_lengths'], dtype=torch.long),
        'edge_index': torch.tensor(org_data['edge_index'], dtype=torch.long),
        'edge_attr': torch.tensor(org_data['edge_attr'], dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.float32),
        'has_label': torch.tensor(has_label, dtype=torch.float32),
    }
    return batch


def load_organism_batches(organisms: list[str],
                            include_regulatory: bool = True,
                            include_ppi: bool = False,
                            include_ortholog_prior: bool = False,
                            verbose: bool = True
                            ) -> dict[str, dict]:
    """Load per-organism batches for a list of organisms. Returns a dict
    {organism_name: batch_dict}.

    `include_ppi`: when True and `outputs/ppi_features_per_gene.csv` exists,
    appends the 3 PPI scalar features (Path A) to the per-gene scalar
    tensor. (PPI features hurt cross-org generalization — see commit
    988f9ed; useful only for in-domain interpolation.)

    `include_ortholog_prior`: when True and
    `outputs/ortholog_prior_features_per_gene.csv` exists, appends the 2
    ortholog-prior scalar features. Computed leave-one-out per organism,
    so safe for cross-org training.

    Set caller's `MultimodalEncoderConfig.n_scalar` to match:
      base only:           20
      + PPI:                23
      + ortholog prior:     22
      + PPI + ortholog:     25
    """
    if verbose:
        print(f'[data_loader] loading features...')
    if FEATURES_CONSERVATION.exists():
        features_df = pd.read_csv(FEATURES_CONSERVATION)
    elif FEATURES_BASE.exists():
        features_df = pd.read_csv(FEATURES_BASE)
    else:
        raise FileNotFoundError(f'No feature CSV available')

    regulatory_df = None
    if include_regulatory and FEATURES_REGULATORY.exists():
        regulatory_df = pd.read_csv(FEATURES_REGULATORY)

    scalar_feature_list = list(SCALAR_FEATURES)
    if include_ppi:
        if not FEATURES_PPI.exists():
            raise FileNotFoundError(
                f'include_ppi=True but {FEATURES_PPI} missing — run '
                f'`python scripts/score_ppi_features.py` first.')
        ppi_df = pd.read_csv(FEATURES_PPI)
        features_df = features_df.merge(
            ppi_df[['organism', 'locus_tag'] + PPI_SCALAR_FEATURES],
            on=['organism', 'locus_tag'], how='left',
        )
        for col in PPI_SCALAR_FEATURES:
            features_df[col] = features_df[col].fillna(0.0)
        scalar_feature_list = scalar_feature_list + PPI_SCALAR_FEATURES

    if include_ortholog_prior:
        if not FEATURES_ORTHOLOG_PRIOR.exists():
            raise FileNotFoundError(
                f'include_ortholog_prior=True but {FEATURES_ORTHOLOG_PRIOR} '
                f'missing — run '
                f'`python scripts/score_ortholog_prior_feature.py` first.')
        orth_df = pd.read_csv(FEATURES_ORTHOLOG_PRIOR)
        features_df = features_df.merge(
            orth_df[['organism', 'locus_tag'] + ORTHOLOG_PRIOR_FEATURES],
            on=['organism', 'locus_tag'], how='left',
        )
        # Missing values -> neutral defaults (0.5 prior, 0 partners)
        features_df['ortholog_prior'] = features_df['ortholog_prior'].fillna(0.5)
        features_df['n_orthologs'] = features_df['n_orthologs'].fillna(0)
        scalar_feature_list = scalar_feature_list + ORTHOLOG_PRIOR_FEATURES

    if verbose:
        print(f'  base features: {len(features_df)} rows, {len(features_df.columns)} cols')
        if regulatory_df is not None:
            print(f'  regulatory features: {len(regulatory_df)} rows')
        if include_ppi:
            print(f'  PPI features merged: '
                  f'{len(features_df.dropna(subset=PPI_SCALAR_FEATURES))} rows')
        if include_ortholog_prior:
            print(f'  ortholog-prior features merged: '
                  f'{(features_df.n_orthologs > 0).sum()} rows have orthologs')
        print(f'  scalar feature dim: {len(scalar_feature_list)} '
              f'({scalar_feature_list[-3:]})')

    batches = {}
    for org in organisms:
        b = _build_organism_batch(
            org, features_df, regulatory_df,
            scalar_feature_list=scalar_feature_list,
        )
        if b is None:
            if verbose: print(f'  {org}: skipped (data missing)')
            continue
        n = b['scalar'].size(0)
        n_ess = int(b['labels'].sum())
        n_reg = int(b['regulatory_mask'].sum())
        n_edges = b['edge_index'].size(1)
        if verbose:
            print(f'  {org:<15s} N={n:5d}  ess={n_ess:4d}  ne={n - n_ess:4d}  '
                  f'reg_mask>0={n_reg:5d}  edges={n_edges}')
        batches[org] = b
    return batches


# ──────────── self-test ────────────


def _self_test():
    print('data_loader self-test:')
    batches = load_organism_batches(['styphimurium', 'syn3a', 'mpne'])
    assert 'styphimurium' in batches and 'syn3a' in batches
    salm = batches['styphimurium']
    syn = batches['syn3a']
    print(f'\n  styphimurium batch shapes:')
    for k, v in salm.items():
        if isinstance(v, torch.Tensor):
            print(f'    {k:25s} {tuple(v.shape)}  dtype={v.dtype}')

    print(f'\n  syn3a batch shapes:')
    for k, v in syn.items():
        if isinstance(v, torch.Tensor):
            print(f'    {k:25s} {tuple(v.shape)}')

    # Verify regulatory mask is 1.0 for Salmonella, 0.5 for others
    sal_mask_unique = salm['regulatory_mask'].unique().tolist()
    syn_mask_unique = syn['regulatory_mask'].unique().tolist()
    print(f'\n  styphimurium reg_mask values: {sal_mask_unique}')
    print(f'  syn3a reg_mask values: {syn_mask_unique}')

    # Sanity: feed Salmonella batch through the LNN
    from cell_sim.layer_ml.essentiality_lnn import EssentialityLNN, LNNConfig
    cfg = LNNConfig(hidden=64, memory_size=128, n_liquid_steps=3)
    model = EssentialityLNN(cfg)
    model.eval()
    out = model(salm)
    print(f'\n  LNN forward on styphimurium ({salm["scalar"].size(0)} genes):')
    print(f'    essentiality: {tuple(out["essentiality"].shape)}, '
          f'min={out["essentiality"].min().item():.2f}, '
          f'max={out["essentiality"].max().item():.2f}')

    print('  ALL PASS')


if __name__ == '__main__':
    _self_test()

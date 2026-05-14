"""Build per-node static feature tensor for the LGNN.

Six categories of static features (54 cols total when all enabled):

  1. Gene-class (19 cols, per-locus broadcast)
     14 keyword categories + 3 sequence summaries + 2 gene metadata.
     Source: memory_bank/data/syn3a_gene_class_features.csv (built from
     syn3A.gb via scripts/build_syn3a_gene_class_features.py).

  2. Spatial (12 cols, per-locus broadcast)
     gene/rna/protein/translation distance-to-membrane plus site-type
     fractions and rna-degradosome distance.
     Source: syn3a_spatial_features_all4.parquet (Step B output from
     MinCell_{1..4}.lm).

  3. Per-species role one-hot (8 cols)
     is_gene, is_mrna, is_mrna_decay, is_protein, is_membrane_protein,
     is_translation_complex, is_degradation, is_membrane_translation,
     plus is_cofactor (one-hot inferred from the row name's prefix and
     suffix).

  4. Promoter strength (1 col, per-locus broadcast)
     Per Equation 20 of Thornburg 2026: derived from P_L initial count
     divided by 180. Reflects per-gene transcription rate.

  5. Log initial count (1 col, per-species)
     log1p of |x0[species]| from t=0 of replicate 1. Gives the model a
     per-species scale reference.

  6. Proteomics (13 cols, per-locus broadcast)
     5 Primary Function one-hot + 6 Localization one-hot + 2 scalar
     (log Sim. Initial Ptn Cnt, log Exp. Ptn Cnt). Source:
     `initial_concentrations.xlsx` "Comparative Proteomics" sheet
     (Breuer 2019 + Cell 2022 annotations bundled with the simulator).
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

_LOCUS_RE = re.compile(r'_(\d{4})(?:_|$)')

GENE_CLASS_COLS = [
    # Keyword categories (14)
    'is_ribosomal', 'is_synthetase', 'is_trna', 'is_polymerase',
    'is_dna_machinery', 'is_translation', 'is_transcription',
    'is_membrane', 'is_atp_binding', 'is_kinase', 'is_phosphatase',
    'is_synthase', 'is_uncharacterized', 'is_metabolism',
    # Sequence summaries (3)
    'protein_length_aa_log', 'tm_helix_count', 'has_signal_pep',
    # Gene metadata (2)
    'has_gene_name', 'length_bp_log',
]

SPATIAL_COLS = [
    'gene_dist_to_membrane_mean', 'gene_frac_in_DNA',
    'rna_dist_to_membrane_mean', 'rna_dist_to_degradosome_mean',
    'rna_frac_in_cytoplasm', 'rna_frac_in_outer_cytoplasm',
    'protein_dist_to_membrane_mean', 'protein_frac_in_membrane',
    'protein_frac_in_cytoplasm', 'protein_frac_in_DNA',
    'translation_dist_to_membrane_mean', 'translation_frac_in_DNA',
]

ROLE_COLS = [
    'is_gene_row', 'is_mrna_row', 'is_mrna_decay_row', 'is_protein_row',
    'is_membrane_protein_row', 'is_translation_complex_row',
    'is_degradation_row', 'is_cofactor_or_flux_row',
]

EXTRA_COLS = ['promoter_strength_log', 'log_initial_count']

# 5 categories from Comparative Proteomics sheet's Primary Function column
PROTEOMICS_FUNCTION_VALUES = (
    'Genetic Information Processing',
    'Metabolism',
    'Cellular Processes',
    'Human Diseases',
    'Unclear',
)
# 6 categories from Localization column
PROTEOMICS_LOCALIZATION_VALUES = (
    'cytoplasm',
    'trans-membrane',
    'peripheral membrane',
    'lipoprotein',
    'unidentified',
    'extracellular',
)
PROTEOMICS_FUNCTION_COLS = [
    f'func_{v.replace(" ", "_").lower()}' for v in PROTEOMICS_FUNCTION_VALUES
]
PROTEOMICS_LOCALIZATION_COLS = [
    f'loc_{v.replace(" ", "_").replace("-", "_").lower()}'
    for v in PROTEOMICS_LOCALIZATION_VALUES
]
PROTEOMICS_COLS = (
    PROTEOMICS_FUNCTION_COLS
    + PROTEOMICS_LOCALIZATION_COLS
    + ['proteomics_log_sim_initial_ptn_cnt', 'proteomics_log_exp_ptn_cnt']
)

ALL_STATIC_COLS = (
    GENE_CLASS_COLS + SPATIAL_COLS + ROLE_COLS + EXTRA_COLS + PROTEOMICS_COLS
)


def _role_indicators(row_name: str) -> dict:
    n = row_name
    return {
        'is_gene_row':                int(n.startswith('G_')),
        'is_mrna_row':                int(n.startswith('R_') and not n.endswith('_d')),
        'is_mrna_decay_row':          int(n.startswith('R_') and n.endswith('_d')),
        'is_protein_row':             int(n.startswith('P_') and not n.startswith('PM_')),
        'is_membrane_protein_row':    int(n.startswith('PM_')),
        'is_translation_complex_row': int(n.startswith('RP_') or n.startswith('RPM_')),
        'is_degradation_row':         int(n.startswith('D_') or n.startswith('DM_')),
        'is_cofactor_or_flux_row':    int(_LOCUS_RE.search(n) is None),
    }


def _read_proteomics_sheet(xlsx_path: str | Path) -> pd.DataFrame:
    """Load 'Comparative Proteomics' from initial_concentrations.xlsx.

    Row 0 is a citation/source-attribution row, not data — drop it.
    """
    df = pd.read_excel(xlsx_path, sheet_name='Comparative Proteomics')
    df = df.iloc[1:].reset_index(drop=True)
    df = df.dropna(subset=['Locus Tag']).copy()
    df['locus_4d'] = df['Locus Tag'].str.replace('JCVISYN3A_', '', regex=False)
    return df


def load_breuer_labels(xlsx_path: str | Path) -> pd.DataFrame:
    """Return a DataFrame with locus_tag, locus_4d, essential bool from the
    Comparative Proteomics sheet's Essentiality column. 'Essential' -> True;
    'Quasiessential' and 'Nonessential' -> False (binary classification).
    """
    df = _read_proteomics_sheet(xlsx_path)
    out = pd.DataFrame({
        'locus_tag': df['Locus Tag'].astype(str),
        'locus_4d':  df['locus_4d'].astype(str),
        'essential': (df['Essentiality'].astype(str) == 'Essential'),
    })
    return out.reset_index(drop=True)


def build_static_node_features(
    row_names: Sequence[str],
    gene_class_csv: str | Path,
    spatial_parquet: Optional[str | Path] = None,
    initial_state_signed_log1p: Optional[np.ndarray] = None,
    proteomics_xlsx: Optional[str | Path] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Compose 54-column static feature tensor in row_names order.

    Args
    ----
    row_names                 : list[str], LGNN species order
    gene_class_csv            : per-locus gene_class features
    spatial_parquet           : optional per-locus spatial features
                                  (Step B output). If None, those 12 cols are
                                  zero-filled.
    initial_state_signed_log1p: optional (S,) array of t=0 signed-log1p
                                  counts from a reference replicate. Used
                                  to derive promoter_strength (from P_L
                                  cells) and log_initial_count (per-species).
                                  If None, those 2 cols are zero-filled.
    proteomics_xlsx           : optional path to initial_concentrations.xlsx;
                                  if given, populates the 13 PROTEOMICS_COLS
                                  from the Comparative Proteomics sheet.
                                  If None those cols are zero-filled.

    Returns
    -------
    torch.FloatTensor of shape (N, 54).
    """
    N = len(row_names)
    F_gc = len(GENE_CLASS_COLS)
    F_sp = len(SPATIAL_COLS)
    F_role = len(ROLE_COLS)
    F_extra = len(EXTRA_COLS)
    F_prot = len(PROTEOMICS_COLS)
    F = F_gc + F_sp + F_role + F_extra + F_prot

    out = np.zeros((N, F), dtype=np.float32)
    name_to_idx = {n: i for i, n in enumerate(row_names)}

    # ----- 1. Gene-class -----
    gc = pd.read_csv(gene_class_csv)
    gc['locus_4d'] = gc['locus_tag'].str.replace('JCVISYN3A_', '', regex=False)
    gc['protein_length_aa_log'] = np.log1p(
        gc['protein_length_aa'].fillna(0).astype(float)
    )
    gc['length_bp_log'] = np.log1p(gc['length_bp'].astype(float))
    gc_lookup = gc.set_index('locus_4d')[GENE_CLASS_COLS]
    matched_gc = 0
    for i, name in enumerate(row_names):
        m = _LOCUS_RE.search(name)
        if m is None: continue
        locus = m.group(1)
        if locus in gc_lookup.index:
            out[i, :F_gc] = gc_lookup.loc[locus].values.astype(np.float32)
            matched_gc += 1
    if verbose:
        print(f'  gene_class matched: {matched_gc}/{N} '
              f'({100*matched_gc/N:.1f}%)')

    # ----- 2. Spatial -----
    if spatial_parquet is not None and Path(spatial_parquet).exists():
        sp = pd.read_parquet(spatial_parquet)
        # Some spatial cols may be NaN where the category had no observations
        sp_lookup = sp.set_index('locus')[SPATIAL_COLS]
        # Replace NaN with column median so the model gets a sensible default
        sp_lookup = sp_lookup.fillna(sp_lookup.median(numeric_only=True))
        # Any remaining NaN (e.g., all-NaN column): zero
        sp_lookup = sp_lookup.fillna(0.0)
        matched_sp = 0
        for i, name in enumerate(row_names):
            m = _LOCUS_RE.search(name)
            if m is None: continue
            locus = m.group(1)
            if locus in sp_lookup.index:
                out[i, F_gc:F_gc + F_sp] = sp_lookup.loc[locus].values.astype(np.float32)
                matched_sp += 1
        if verbose:
            print(f'  spatial matched:    {matched_sp}/{N} '
                  f'({100*matched_sp/N:.1f}%)')
    else:
        if verbose:
            print(f'  spatial: SKIPPED (no parquet provided)')

    # ----- 3. Role indicators (one-hot) -----
    for i, name in enumerate(row_names):
        roles = _role_indicators(name)
        out[i, F_gc + F_sp:F_gc + F_sp + F_role] = [
            roles[c] for c in ROLE_COLS
        ]

    # ----- 4. Promoter strength + log_initial_count -----
    if initial_state_signed_log1p is not None:
        # signed_log1p inverts to: x_real = sign * (exp(|y|) - 1)
        y = np.asarray(initial_state_signed_log1p, dtype=np.float32)
        x_real = np.sign(y) * (np.exp(np.abs(y)) - 1.0)

        # log_initial_count (per species, just |x|)
        out[:, F_gc + F_sp + F_role + 1] = np.log1p(np.abs(x_real))

        # Promoter strength: for each locus L, look up P_L's initial count, /180.
        # Broadcast to all species in locus L.
        prom_per_locus: dict[str, float] = {}
        for i, name in enumerate(row_names):
            if name.startswith('P_') and not name.startswith('PM_'):
                m = _LOCUS_RE.search(name)
                if m is not None:
                    locus = m.group(1)
                    prom_per_locus[locus] = float(np.abs(x_real[i])) / 180.0
        # Broadcast
        matched_prom = 0
        for i, name in enumerate(row_names):
            m = _LOCUS_RE.search(name)
            if m is None: continue
            locus = m.group(1)
            if locus in prom_per_locus:
                out[i, F_gc + F_sp + F_role] = np.log1p(prom_per_locus[locus])
                matched_prom += 1
        if verbose:
            print(f'  promoter matched:   {matched_prom}/{N} '
                  f'({100*matched_prom/N:.1f}%)')
            print(f'  log_init_count:     all {N} species')
    else:
        if verbose:
            print(f'  promoter + log_init_count: SKIPPED (no initial state)')

    # ----- 6. Proteomics (function/localization/initial counts) -----
    proteomics_offset = F_gc + F_sp + F_role + F_extra
    if proteomics_xlsx is not None and Path(proteomics_xlsx).exists():
        prot = _read_proteomics_sheet(proteomics_xlsx)
        # function -> one-hot index
        func_to_idx = {v: i for i, v in enumerate(PROTEOMICS_FUNCTION_VALUES)}
        loc_to_idx  = {v: i for i, v in enumerate(PROTEOMICS_LOCALIZATION_VALUES)}
        F_func = len(PROTEOMICS_FUNCTION_VALUES)
        F_loc  = len(PROTEOMICS_LOCALIZATION_VALUES)
        prot_lookup: dict[str, np.ndarray] = {}
        for _, r in prot.iterrows():
            vec = np.zeros(F_prot, dtype=np.float32)
            f_str = str(r.get('Primary Function', '')).strip()
            l_str = str(r.get('Localization', '')).strip()
            if f_str in func_to_idx:
                vec[func_to_idx[f_str]] = 1.0
            if l_str in loc_to_idx:
                vec[F_func + loc_to_idx[l_str]] = 1.0
            try:
                sim_init = float(r.get('Sim. Initial Ptn Cnt', 0) or 0)
                vec[F_func + F_loc + 0] = float(np.log1p(max(sim_init, 0.0)))
            except (TypeError, ValueError):
                pass
            try:
                exp_cnt = float(r.get('Exp. Ptn Cnt', 0) or 0)
                vec[F_func + F_loc + 1] = float(np.log1p(max(exp_cnt, 0.0)))
            except (TypeError, ValueError):
                pass
            prot_lookup[r['locus_4d']] = vec

        matched_prot = 0
        for i, name in enumerate(row_names):
            m = _LOCUS_RE.search(name)
            if m is None:
                continue
            locus = m.group(1)
            if locus in prot_lookup:
                out[i, proteomics_offset:proteomics_offset + F_prot] = \
                    prot_lookup[locus]
                matched_prot += 1
        if verbose:
            print(f'  proteomics matched: {matched_prot}/{N} '
                  f'({100 * matched_prot / N:.1f}%)')
    else:
        if verbose:
            print(f'  proteomics: SKIPPED (no xlsx provided)')

    return torch.from_numpy(out)


def feature_summary(features: torch.Tensor) -> pd.DataFrame:
    arr = features.float().numpy()
    rows = []
    for i, name in enumerate(ALL_STATIC_COLS):
        col = arr[:, i]
        rows.append({
            'feature':      name,
            'min':          col.min(),
            'max':          col.max(),
            'mean':         col.mean(),
            'frac_nonzero': (col != 0).mean(),
        })
    return pd.DataFrame(rows)


# Back-compat alias: M4's train_m4.py imports STATIC_FEATURE_COLS
STATIC_FEATURE_COLS = ALL_STATIC_COLS

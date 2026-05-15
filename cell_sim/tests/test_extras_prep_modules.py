"""Smoke tests for the 6 prep extras modules.

Each module loads its file and produces tensors / dicts. Tests cover:
  - Correct shape / type on synthetic fixture
  - Graceful handling of missing file
  - Correct row-targeting (where applicable)
"""
from __future__ import annotations
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent.parent))

from cell_sim.lgnn.extras import (
    build_regulatory_features,
    build_ribosome_subunit_features,
    build_thermodynamics_features,
    describe_alternate_sbml,
    find_alternate_sbml_path,
    load_wcm_naming_map,
    apply_name_map,
    load_sim_metadata,
    summarize_sim_metadata,
)


def _toy_rows():
    return ['M_atp_c', 'M_adp_c',
            'G_0001', 'R_0001', 'P_0001', 'PM_0612',
            'RP_0234', 'F_R_PYK', 'F_R_ATPS']


# ----------------------------------------------------------------------
# regulatory_features
# ----------------------------------------------------------------------

def test_regulatory_features_returns_correct_shape():
    with tempfile.TemporaryDirectory() as d:
        df = pd.DataFrame({
            'protein_locus': ['JCVISYN3A_0001', 'JCVISYN3A_0612'],
            'metabolite':    ['M_atp_c',        'M_adp_c'],
        })
        p = Path(d) / 'pm.xlsx'
        df.to_excel(p, index=False)
        out = build_regulatory_features(p, _toy_rows(), verbose=False)
    assert out.shape == (len(_toy_rows()), 3)
    assert out.dtype == torch.float32


def test_regulatory_features_handles_missing_file():
    out = build_regulatory_features('/nonexistent/pm.xlsx', _toy_rows(), verbose=False)
    assert (out == 0).all()


def test_regulatory_features_marks_atp_as_regulated():
    with tempfile.TemporaryDirectory() as d:
        df = pd.DataFrame({
            'protein_locus': ['JCVISYN3A_0001'],
            'metabolite':    ['M_atp_c'],
        })
        p = Path(d) / 'pm.xlsx'
        df.to_excel(p, index=False)
        rows = _toy_rows()
        out = build_regulatory_features(p, rows, verbose=False)
    atp_idx = rows.index('M_atp_c')
    # is_regulated_metabolite is col 1
    assert out[atp_idx, 1].item() == 1.0


# ----------------------------------------------------------------------
# ribosome_subunit
# ----------------------------------------------------------------------

def test_ribosome_subunit_returns_correct_shape():
    with tempfile.TemporaryDirectory() as d:
        df = pd.DataFrame({
            'gene_locus': ['0001', '0234'],
            'order':      [1, 2],
        })
        p = Path(d) / 'ls.xlsx'
        df.to_excel(p, index=False)
        out = build_ribosome_subunit_features(p, _toy_rows(), verbose=False)
    assert out.shape == (len(_toy_rows()), 2)


def test_ribosome_subunit_handles_missing_file():
    out = build_ribosome_subunit_features('/nonexistent/ls.xlsx', _toy_rows(), verbose=False)
    assert (out == 0).all()


def test_ribosome_subunit_marks_listed_loci():
    with tempfile.TemporaryDirectory() as d:
        df = pd.DataFrame({
            'gene_locus': ['0001'],
            'order':      [1],
        })
        p = Path(d) / 'ls.xlsx'
        df.to_excel(p, index=False)
        rows = _toy_rows()
        out = build_ribosome_subunit_features(p, rows, verbose=False)
    p0001_idx = rows.index('P_0001')
    assert out[p0001_idx, 0].item() == 1.0
    # Metabolite row should remain zero
    atp_idx = rows.index('M_atp_c')
    assert out[atp_idx].sum().item() == 0


# ----------------------------------------------------------------------
# thermodynamics
# ----------------------------------------------------------------------

def test_thermodynamics_returns_correct_shape():
    with tempfile.TemporaryDirectory() as d:
        df = pd.DataFrame({
            'reaction_id':         ['R_PYK', 'R_ATPS'],
            'dG_prime_kJ_per_mol': [-12.0,   -25.0],   # forward-favorable
        })
        p = Path(d) / 'gibbs.csv'
        df.to_csv(p, index=False)
        out = build_thermodynamics_features(p, _toy_rows(), verbose=False)
    assert out.shape == (len(_toy_rows()), 2)


def test_thermodynamics_handles_missing_file():
    out = build_thermodynamics_features('/nonexistent/gibbs.csv',
                                          _toy_rows(), verbose=False)
    assert (out == 0).all()


def test_thermodynamics_sign_matches_forward_favorable():
    """For ΔG < 0 (forward-favorable), dG_sign should be +1."""
    with tempfile.TemporaryDirectory() as d:
        df = pd.DataFrame({
            'reaction_id':         ['R_PYK'],
            'dG_prime_kJ_per_mol': [-12.0],
        })
        p = Path(d) / 'gibbs.csv'
        df.to_csv(p, index=False)
        rows = _toy_rows()
        out = build_thermodynamics_features(p, rows, verbose=False)
    pyk_idx = rows.index('F_R_PYK')
    assert out[pyk_idx, 1].item() == 1.0
    # log_K_eq should be positive (forward-favorable)
    assert out[pyk_idx, 0].item() > 0


# ----------------------------------------------------------------------
# alternate_sbml
# ----------------------------------------------------------------------

def test_find_alternate_sbml_returns_none_when_absent():
    with tempfile.TemporaryDirectory() as d:
        out = find_alternate_sbml_path(d)
    assert out is None


def test_describe_alternate_sbml_handles_missing():
    out = describe_alternate_sbml('/nonexistent/some.xml')
    assert out['exists'] is False


def test_describe_alternate_sbml_reports_counts():
    """Build a tiny SBML file and verify the describer reports species/reaction counts."""
    sbml = '''<?xml version="1.0"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core">
  <model>
    <listOfSpecies>
      <species id="A"/>
      <species id="B"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="R1"/>
    </listOfReactions>
  </model>
</sbml>'''
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'mini.xml'
        p.write_text(sbml)
        info = describe_alternate_sbml(p)
    assert info['exists']
    assert info['n_species'] == 2
    assert info['n_reactions'] == 1


# ----------------------------------------------------------------------
# wcm_naming
# ----------------------------------------------------------------------

def test_wcm_naming_returns_dict():
    with tempfile.TemporaryDirectory() as d:
        df = pd.DataFrame({
            'WCM_name':      ['R_PYK', 'M_atp_c'],
            'standard_name': ['Pyruvate kinase', 'ATP cytoplasmic'],
        })
        p = Path(d) / 'naming.xlsx'
        df.to_excel(p, index=False)
        m = load_wcm_naming_map(p, verbose=False)
    assert isinstance(m, dict)
    assert m.get('R_PYK') == 'Pyruvate kinase'


def test_wcm_naming_handles_missing_file():
    m = load_wcm_naming_map('/nonexistent/naming.xlsx', verbose=False)
    assert m == {}


def test_apply_name_map_falls_back_to_original():
    m = {'R_PYK': 'Pyruvate kinase'}
    assert apply_name_map('R_PYK', m) == 'Pyruvate kinase'
    assert apply_name_map('R_UNKNOWN', m) == 'R_UNKNOWN'   # graceful fallback


# ----------------------------------------------------------------------
# sim_metadata
# ----------------------------------------------------------------------

def test_load_sim_metadata_handles_missing_file():
    out = load_sim_metadata('/nonexistent/meta.pkl', verbose=False)
    assert out is None


def test_load_sim_metadata_returns_dict():
    payload = {'rng_seed': 42, 'duration_s': 7200, 'replicates': 50}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'meta.pkl'
        with open(p, 'wb') as f:
            pickle.dump(payload, f)
        out = load_sim_metadata(p, verbose=False)
    assert isinstance(out, dict)
    assert out['rng_seed'] == 42


def test_summarize_sim_metadata_renders():
    s = summarize_sim_metadata({'a': 1, 'b': [1, 2, 3]})
    assert isinstance(s, str)
    assert 'a' in s and 'b' in s


def test_summarize_sim_metadata_handles_none():
    s = summarize_sim_metadata(None)
    assert 'no metadata' in s.lower()


# ----------------------------------------------------------------------
# Integration: all 6 prep extras can be combined with the existing
# extras through the combined builder
# ----------------------------------------------------------------------

def test_combined_features_with_all_prep_extras():
    """Build everything together. Demonstrate that all 9 extras (3 active
    + 6 prep) can be wired without colliding."""
    from cell_sim.lgnn.training.train_m7 import _build_combined_static_features

    with tempfile.TemporaryDirectory() as d:
        # Active extras
        kin_df = pd.DataFrame({
            'reaction_id': ['R_PYK'], 'kcat': [100.0], 'km': [0.5],
        })
        kin_df.to_excel(Path(d) / 'kin.xlsx', index=False)
        cx_df = pd.DataFrame({
            'complex_name': ['rib'], 'gene_loci': ['0001'],
        })
        cx_df.to_excel(Path(d) / 'cx.xlsx', index=False)
        med_df = pd.DataFrame({
            'metabolite_name': ['glc'], 'concentration': [10.0],
        })
        med_df.to_excel(Path(d) / 'med.xlsx', index=False)
        # Prep extras
        pm_df = pd.DataFrame({
            'protein_locus': ['JCVISYN3A_0001'], 'metabolite': ['M_atp_c'],
        })
        pm_df.to_excel(Path(d) / 'pm.xlsx', index=False)
        ls_df = pd.DataFrame({'gene_locus': ['0001'], 'order': [1]})
        ls_df.to_excel(Path(d) / 'ls.xlsx', index=False)
        thermo_df = pd.DataFrame({
            'reaction_id': ['R_PYK'], 'dG_prime_kJ_per_mol': [-12.0],
        })
        thermo_df.to_csv(Path(d) / 'gibbs.csv', index=False)

        rows = _toy_rows()
        out = _build_combined_static_features(
            row_names=rows,
            use_role=True,
            use_kinetic_priors=True,
            use_complex_constraints=True,
            use_medium_features=True,
            use_regulatory_features=True,
            use_ribosome_subunit=True,
            use_thermodynamics=True,
            kinetic_params_xlsx=str(Path(d)/'kin.xlsx'),
            complex_formation_xlsx=str(Path(d)/'cx.xlsx'),
            medium_xlsx=str(Path(d)/'med.xlsx'),
            protein_metabolites_xlsx=str(Path(d)/'pm.xlsx'),
            large_subunit_xlsx=str(Path(d)/'ls.xlsx'),
            gibbs_csv_path=str(Path(d)/'gibbs.csv'),
            verbose=False,
        )
    assert out is not None
    # role(8) + kin(2) + complex(>=1) + medium(4) + reg(3) + ribo(2) + thermo(2) = >= 22
    assert out.shape[1] >= 22


def test_combined_features_prep_extras_default_off():
    """When use_* flags for prep extras are False (the defaults), those
    columns are NOT in the tensor."""
    from cell_sim.lgnn.training.train_m7 import _build_combined_static_features
    rows = _toy_rows()
    out = _build_combined_static_features(
        row_names=rows,
        use_role=True,
        # All prep extras explicitly off
        use_regulatory_features=False,
        use_ribosome_subunit=False,
        use_thermodynamics=False,
        verbose=False,
    )
    # Should be just the 8 role columns
    assert out.shape == (len(rows), 8)

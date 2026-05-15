"""Smoke tests for Tier-2/3 extras modules.

Each module is INDEPENDENT - tests use only its own loader, no model
or cross-module dependencies. Tests cover:
  - Graceful handling of missing files (return zero tensor + warning)
  - Correct shape and dtype for the returned tensor
  - Non-zero entries only on relevant rows
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent.parent))

from cell_sim.lgnn.extras import (
    build_kinetic_prior_features,
    build_complex_membership_features,
    build_medium_features,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _toy_rows():
    return [
        'M_atp_c', 'M_adp_c',
        'G_0001', 'R_0001', 'P_0001', 'PM_0612',
        'RP_0234', 'F_R_PYK', 'F_R_ATPS',
    ]


def _toy_kinetic_xlsx(tmp_path: Path) -> Path:
    """Build a fake kinetic_params.xlsx with 2 reactions."""
    df = pd.DataFrame({
        'reaction_id': ['R_PYK', 'R_ATPS'],
        'kcat':         [100.0, 250.0],
        'km':           [0.5,   0.1],
    })
    p = tmp_path / 'kinetic.xlsx'
    df.to_excel(p, index=False, sheet_name='Sheet1')
    return p


def _toy_complex_xlsx(tmp_path: Path) -> Path:
    """Build a fake complex_formation.xlsx with 2 complexes."""
    df = pd.DataFrame({
        'complex_name': ['ribosome', 'rnap'],
        'gene_loci':    ['0001, 0612', '0234'],
    })
    p = tmp_path / 'complex.xlsx'
    df.to_excel(p, index=False, sheet_name='Sheet1')
    return p


def _toy_medium_xlsx(tmp_path: Path) -> Path:
    """Build a fake Medium.xlsx with a few metabolites."""
    df = pd.DataFrame({
        'metabolite_name': ['glc', 'nh4', 'ala', 'ile', 'pi'],
        'concentration':    [10.0, 1.0,  0.5,  0.3,  0.7],
    })
    p = tmp_path / 'medium.xlsx'
    df.to_excel(p, index=False, sheet_name='Sheet1')
    return p


# ----------------------------------------------------------------------
# kinetic_priors
# ----------------------------------------------------------------------

def test_kinetic_priors_returns_correct_shape():
    with tempfile.TemporaryDirectory() as d:
        xlsx = _toy_kinetic_xlsx(Path(d))
        rows = _toy_rows()
        out = build_kinetic_prior_features(xlsx, rows, verbose=False)
    assert out.shape == (len(rows), 2)
    assert out.dtype == torch.float32


def test_kinetic_priors_zero_outside_flux_rows():
    """Non-F_* rows should always have zero kinetic features."""
    with tempfile.TemporaryDirectory() as d:
        xlsx = _toy_kinetic_xlsx(Path(d))
        rows = _toy_rows()
        out = build_kinetic_prior_features(xlsx, rows, verbose=False)
    for i, name in enumerate(rows):
        if not name.startswith('F_'):
            assert out[i].sum().item() == 0.0, f'{name} got nonzero kinetics'


def test_kinetic_priors_handles_missing_file():
    """Missing file -> zero tensor with warning, not crash."""
    out = build_kinetic_prior_features(
        Path('/nonexistent/file.xlsx'),
        _toy_rows(),
        verbose=False,
    )
    assert out is not None
    assert out.shape[0] == len(_toy_rows())
    assert (out == 0).all()


def test_kinetic_priors_matched_flux_rows_get_values():
    with tempfile.TemporaryDirectory() as d:
        xlsx = _toy_kinetic_xlsx(Path(d))
        rows = _toy_rows()
        out = build_kinetic_prior_features(xlsx, rows, verbose=False)
    # F_R_PYK should have log1p(100) for kcat (column 0)
    pyk_idx = rows.index('F_R_PYK')
    assert out[pyk_idx, 0].item() > 0, 'kcat for R_PYK should be > 0'
    # F_R_ATPS likewise
    atps_idx = rows.index('F_R_ATPS')
    assert out[atps_idx, 0].item() > 0, 'kcat for R_ATPS should be > 0'


# ----------------------------------------------------------------------
# complex_constraints
# ----------------------------------------------------------------------

def test_complex_features_returns_correct_shape():
    with tempfile.TemporaryDirectory() as d:
        xlsx = _toy_complex_xlsx(Path(d))
        rows = _toy_rows()
        out = build_complex_membership_features(xlsx, rows, verbose=False)
    assert out.shape[0] == len(rows)
    assert out.shape[1] >= 1


def test_complex_features_locus_0001_in_ribosome():
    """Locus 0001 should be marked as belonging to the ribosome complex."""
    with tempfile.TemporaryDirectory() as d:
        xlsx = _toy_complex_xlsx(Path(d))
        rows = _toy_rows()
        out = build_complex_membership_features(xlsx, rows, verbose=False)
    # Locus 0001 appears in G_0001, R_0001, P_0001 - all should be hot for ribosome
    for prefix in ('G_', 'R_', 'P_'):
        idx = rows.index(f'{prefix}0001')
        assert out[idx].sum().item() > 0, f'{prefix}0001 should have a complex flag'


def test_complex_features_handles_missing_file():
    out = build_complex_membership_features(
        Path('/nonexistent/complex.xlsx'),
        _toy_rows(),
        verbose=False,
    )
    assert out is not None
    assert (out == 0).all()


def test_complex_features_unaffected_rows_are_zero():
    """A locus that's not in any listed complex (e.g., 0612 only in ribosome)
    + G_0001 in ribosome, but RP_0234 in rnap. Verify the F_* and M_* rows
    that aren't tied to any locus are all zero."""
    with tempfile.TemporaryDirectory() as d:
        xlsx = _toy_complex_xlsx(Path(d))
        rows = _toy_rows()
        out = build_complex_membership_features(xlsx, rows, verbose=False)
    # M_atp_c has no locus tag -> should be zero
    atp_idx = rows.index('M_atp_c')
    assert out[atp_idx].sum().item() == 0
    # F_R_PYK has no locus -> should be zero
    pyk_idx = rows.index('F_R_PYK')
    assert out[pyk_idx].sum().item() == 0


# ----------------------------------------------------------------------
# medium_features
# ----------------------------------------------------------------------

def test_medium_features_returns_correct_shape():
    with tempfile.TemporaryDirectory() as d:
        xlsx = _toy_medium_xlsx(Path(d))
        rows = _toy_rows()
        out = build_medium_features(xlsx, rows, n_features=4, verbose=False)
    assert out.shape == (len(rows), 4)


def test_medium_features_broadcasted_to_all_rows():
    """Every row should have the same medium-summary vector."""
    with tempfile.TemporaryDirectory() as d:
        xlsx = _toy_medium_xlsx(Path(d))
        rows = _toy_rows()
        out = build_medium_features(xlsx, rows, verbose=False)
    # All rows are identical
    first = out[0]
    for i in range(len(rows)):
        assert torch.allclose(out[i], first), \
            f'row {i} differs from row 0; medium should broadcast'


def test_medium_features_nonzero_when_metabolites_present():
    with tempfile.TemporaryDirectory() as d:
        xlsx = _toy_medium_xlsx(Path(d))
        rows = _toy_rows()
        out = build_medium_features(xlsx, rows, verbose=False)
    assert out.abs().sum().item() > 0


def test_medium_features_handles_missing_file():
    out = build_medium_features(
        Path('/nonexistent/medium.xlsx'),
        _toy_rows(),
        verbose=False,
    )
    assert out is not None
    assert (out == 0).all()

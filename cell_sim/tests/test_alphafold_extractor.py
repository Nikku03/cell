"""Sandbox-safe tests for :class:`AlphaFoldExtractor`.

No ``biopython`` or network access is required. The heavy biopython
import sits behind ``_ensure_loaded`` and the network fetch is only
called from non-empty ``extract``.
"""
from __future__ import annotations

import importlib
import sys

import pandas as pd

from cell_sim.features import FeatureRegistry, FeatureSource
from cell_sim.features.extractors.alphafold_extractor import (
    AlphaFoldExtractor,
)


_HEAVY_MODULES = {
    "torch", "transformers", "biopython", "Bio", "mace",
    "huggingface_hub", "rdkit", "ase", "e3nn", "requests",
}


def test_module_import_cheap():
    pre = set(sys.modules)
    importlib.import_module(
        "cell_sim.features.extractors.alphafold_extractor"
    )
    post = set(sys.modules)
    roots = {m.split(".")[0] for m in (post - pre)}
    offenders = roots & _HEAVY_MODULES
    assert not offenders, (
        f"alphafold_extractor module pulled heavy deps at import: "
        f"{offenders}"
    )


def test_extractor_metadata():
    ex = AlphaFoldExtractor()
    assert ex.name == "alphafold_db"
    assert ex.version == "0.1.0"
    # 9 structural descriptors.
    assert len(ex.feature_cols) == 9
    expected = {
        "af_plddt_mean",
        "af_plddt_std",
        "af_disorder_fraction",
        "af_helix_fraction",
        "af_sheet_fraction",
        "af_coil_fraction",
        "af_sequence_length",
        "af_radius_of_gyration_angstrom",
        "af_has_structure",
    }
    assert set(ex.feature_cols) == expected


def test_empty_input_returns_empty_frame():
    pre = set(sys.modules)
    ex = AlphaFoldExtractor()
    out = ex.extract(
        pd.DataFrame({"locus_tag": [], "uniprot_id": []})
    )
    post = set(sys.modules)
    roots = {m.split(".")[0] for m in (post - pre)}
    assert not (roots & _HEAVY_MODULES)
    assert list(out.columns) == ex.feature_cols
    assert out.index.name == "locus_tag"
    assert len(out) == 0


def test_extractor_instantiation_does_not_load_model():
    pre = set(sys.modules)
    ex = AlphaFoldExtractor()
    post = set(sys.modules)
    roots = {m.split(".")[0] for m in (post - pre)}
    assert not (roots & _HEAVY_MODULES)
    # biopython is not loaded until _ensure_loaded fires.
    assert ex._biopython_loaded is False


def test_feature_cols_match_contract():
    ex = AlphaFoldExtractor()
    out = ex.extract(
        pd.DataFrame({"locus_tag": [], "uniprot_id": []})
    )
    assert list(out.columns) == ex.feature_cols


def test_registered_with_registry(tmp_path):
    reg = FeatureRegistry(cache_dir=tmp_path)
    ex = AlphaFoldExtractor()
    reg.register(FeatureSource(
        name=ex.name,
        parquet_path=tmp_path / f"{ex.name}.parquet",
        expected_sha256=None,
        feature_cols=ex.feature_cols,
        version=ex.version,
    ))
    assert reg.list_sources() == [ex.name]
    assert reg.is_cached(ex.name) is False

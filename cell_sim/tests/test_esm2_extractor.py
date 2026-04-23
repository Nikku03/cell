"""Sandbox-safe tests for :class:`ESM2Extractor`.

None of these tests require ``torch`` or ``transformers`` to be
installed. The model is only loaded by :meth:`ESM2Extractor._ensure_loaded`,
which is only called inside :meth:`ESM2Extractor.extract` on a
non-empty input. Every test here exercises either metadata, empty
input, or registry wiring — never a forward pass.
"""
from __future__ import annotations

import importlib
import sys

import pandas as pd
import pytest

from cell_sim.features import FeatureRegistry, FeatureSource
from cell_sim.features.extractors.esm2_extractor import ESM2Extractor


_HEAVY_MODULES = {
    "torch", "transformers", "biopython", "Bio", "mace",
    "huggingface_hub", "rdkit", "ase", "e3nn", "requests",
}


def test_module_import_cheap():
    """Importing the extractor module must not pull heavy ML deps."""
    pre = set(sys.modules)
    importlib.import_module(
        "cell_sim.features.extractors.esm2_extractor"
    )
    post = set(sys.modules)
    roots = {m.split(".")[0] for m in (post - pre)}
    offenders = roots & _HEAVY_MODULES
    assert not offenders, (
        f"esm2_extractor module pulled heavy deps at import: "
        f"{offenders}"
    )


def test_extractor_metadata():
    ex = ESM2Extractor()
    assert ex.name == "esm2_650M"
    assert ex.version == "0.1.0"
    assert len(ex.feature_cols) == 1280
    # Column names are stable and ordered.
    assert ex.feature_cols[0] == "esm2_650M_dim_0"
    assert ex.feature_cols[-1] == "esm2_650M_dim_1279"
    assert ex._MODEL_ID == "facebook/esm2_t33_650M_UR50D"


def test_empty_input_returns_empty_frame():
    """Empty input must short-circuit BEFORE _ensure_loaded fires,
    so no heavy import is triggered."""
    pre = set(sys.modules)
    ex = ESM2Extractor()
    out = ex.extract(
        pd.DataFrame({"locus_tag": [], "sequence": []})
    )
    post = set(sys.modules)
    roots = {m.split(".")[0] for m in (post - pre)}
    assert not (roots & _HEAVY_MODULES)
    # The empty frame has the correct contract.
    assert list(out.columns) == ex.feature_cols
    assert out.index.name == "locus_tag"
    assert len(out) == 0


def test_extractor_instantiation_does_not_load_model():
    """Constructor must NOT touch torch / transformers."""
    pre = set(sys.modules)
    ex = ESM2Extractor()
    post = set(sys.modules)
    roots = {m.split(".")[0] for m in (post - pre)}
    assert not (roots & _HEAVY_MODULES)
    # Internal state indicates no model loaded.
    assert ex._model is None
    assert ex._tokenizer is None
    assert ex._resolved_device is None


def test_feature_cols_match_contract():
    """The empty-path output columns equal `feature_cols` exactly in
    order. The non-empty path follows the same path; any contract
    violation there would be caught by `BatchedFeatureExtractor.
    write_cache` before reaching disk."""
    ex = ESM2Extractor()
    out = ex.extract(
        pd.DataFrame({"locus_tag": [], "sequence": []})
    )
    assert list(out.columns) == ex.feature_cols


def test_registered_with_registry(tmp_path):
    """The extractor's metadata plugs into :class:`FeatureSource`
    and the registry reports `is_cached=False` because no parquet
    has been produced yet."""
    reg = FeatureRegistry(cache_dir=tmp_path)
    ex = ESM2Extractor()
    source = FeatureSource(
        name=ex.name,
        parquet_path=tmp_path / f"{ex.name}.parquet",
        expected_sha256=None,
        feature_cols=ex.feature_cols,
        version=ex.version,
    )
    reg.register(source)
    assert reg.list_sources() == [ex.name]
    assert reg.is_cached(ex.name) is False

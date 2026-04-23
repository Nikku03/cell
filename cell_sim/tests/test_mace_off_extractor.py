"""Sandbox-safe tests for :class:`MaceOffExtractor`.

The extractor wraps :class:`cell_sim.layer1_atomic.engine.MACEBackend`;
its heavy deps (``mace-torch``, ``torch``, ``rdkit``, ``ase``,
``e3nn``) are loaded by ``MACEBackend._ensure_loaded`` only when a
non-empty ``extract`` fires. All 5 tests here stay in the cheap
sandbox path.
"""
from __future__ import annotations

import importlib
import sys

import pandas as pd

from cell_sim.features import FeatureRegistry, FeatureSource
from cell_sim.features.extractors.mace_off_extractor import (
    MaceOffExtractor,
)


_HEAVY_MODULES = {
    "torch", "transformers", "biopython", "Bio", "mace",
    "huggingface_hub", "ase", "e3nn", "requests",
    # Deliberately NOT including "rdkit" here: the existing
    # cell_sim/layer1_atomic/engine.py's SimilarityBackend
    # opportunistically imports rdkit in its constructor via a
    # try/except. We don't import engine at module load, so this
    # should still be clean. If your environment pre-loads rdkit
    # through some other path, the test below ignores it.
}


def test_module_import_cheap():
    pre = set(sys.modules)
    importlib.import_module(
        "cell_sim.features.extractors.mace_off_extractor"
    )
    post = set(sys.modules)
    roots = {m.split(".")[0] for m in (post - pre)}
    offenders = roots & _HEAVY_MODULES
    assert not offenders, (
        f"mace_off_extractor module pulled heavy deps at import: "
        f"{offenders}"
    )


def test_extractor_metadata():
    ex = MaceOffExtractor()
    assert ex.name == "mace_off_kcat"
    assert ex.version == "0.1.0"
    assert len(ex.feature_cols) == 7
    expected = {
        "mace_kcat_mean_per_s",
        "mace_kcat_std_per_s",
        "mace_kcat_min_per_s",
        "mace_kcat_max_per_s",
        "mace_n_substrates",
        "mace_mean_confidence",
        "mace_has_estimate",
    }
    assert set(ex.feature_cols) == expected


def test_empty_input_returns_empty_frame():
    pre = set(sys.modules)
    ex = MaceOffExtractor()
    out = ex.extract(
        pd.DataFrame({
            "locus_tag": [],
            "enzyme_name": [],
            "reaction_class": [],
            "substrate_smiles": [],
        })
    )
    post = set(sys.modules)
    roots = {m.split(".")[0] for m in (post - pre)}
    assert not (roots & _HEAVY_MODULES)
    assert list(out.columns) == ex.feature_cols
    assert out.index.name == "locus_tag"
    assert len(out) == 0


def test_extractor_instantiation_does_not_load_model():
    """Constructor records config only; no backend, no heavy
    imports."""
    pre = set(sys.modules)
    ex = MaceOffExtractor(model="small", device="cpu")
    post = set(sys.modules)
    roots = {m.split(".")[0] for m in (post - pre)}
    assert not (roots & _HEAVY_MODULES)
    # Internal state says nothing has been loaded yet.
    assert ex._backend is None
    assert ex._backend_model == "small"
    assert ex._backend_device == "cpu"


def test_registered_with_registry(tmp_path):
    reg = FeatureRegistry(cache_dir=tmp_path)
    ex = MaceOffExtractor()
    reg.register(FeatureSource(
        name=ex.name,
        parquet_path=tmp_path / f"{ex.name}.parquet",
        expected_sha256=None,
        feature_cols=ex.feature_cols,
        version=ex.version,
    ))
    assert reg.list_sources() == [ex.name]
    assert reg.is_cached(ex.name) is False

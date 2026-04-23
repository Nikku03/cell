"""Tests for :class:`FeatureRegistry` — declarative feature loader with
SHA-256 integrity validation.

Tests build synthetic parquet files (3-5 rows, 2-3 columns) on a
``tmp_path`` fixture. No pretrained models are loaded. No torch or
transformers imports anywhere in this file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cell_sim.features.feature_registry import (
    FeatureRegistry, FeatureSource,
)
from cell_sim.features.cache_manifest import CachedFeatureManifest


# ---- helpers ----

def _make_features(path: Path, locus_tags: list[str],
                   cols: list[str], seed: int = 0) -> None:
    """Write a synthetic parquet indexed by locus_tag with float
    columns populated from a deterministic RNG."""
    rng = np.random.default_rng(seed)
    data = {c: rng.standard_normal(len(locus_tags)) for c in cols}
    df = pd.DataFrame(data, index=pd.Index(locus_tags, name="locus_tag"))
    df.to_parquet(path, index=True)


def _write_manifest_entry(cache_dir: Path, name: str, parquet: Path,
                           version: str) -> None:
    """Register ``parquet`` in ``cache_dir/manifest.json``."""
    m_path = cache_dir / "manifest.json"
    m = CachedFeatureManifest.load(m_path)
    m.add(name, parquet, version=version)
    m.save(m_path)


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """Fresh cache dir with an empty manifest."""
    cd = tmp_path / "cache"
    cd.mkdir()
    (cd / "manifest.json").write_text('{"sources": {}}\n')
    return cd


# ---- tests ----

def test_registry_empty_initial_state(cache: Path):
    reg = FeatureRegistry(cache_dir=cache)
    assert reg.list_sources() == []
    assert reg.is_cached("anything") is False


def test_register_source(cache: Path):
    reg = FeatureRegistry(cache_dir=cache)
    src = FeatureSource(
        name="dummy",
        parquet_path=cache / "dummy.parquet",
        expected_sha256=None,
        feature_cols=["a", "b"],
        version="0.1.0",
    )
    reg.register(src)
    assert reg.list_sources() == ["dummy"]
    assert reg.get_source("dummy") is src


def test_is_cached_missing_file(cache: Path):
    reg = FeatureRegistry(cache_dir=cache)
    reg.register(FeatureSource(
        name="missing", parquet_path=cache / "missing.parquet",
        expected_sha256=None, feature_cols=["a"], version="0.1.0",
    ))
    assert reg.is_cached("missing") is False


def test_is_cached_present_file(cache: Path):
    parquet = cache / "present.parquet"
    _make_features(parquet, ["JCVISYN3A_0001", "JCVISYN3A_0002"],
                    ["x", "y"])
    reg = FeatureRegistry(cache_dir=cache)
    reg.register(FeatureSource(
        name="present", parquet_path=parquet,
        expected_sha256=None, feature_cols=["x", "y"], version="0.1.0",
    ))
    assert reg.is_cached("present") is True


def test_load_missing_file_raises(cache: Path):
    reg = FeatureRegistry(cache_dir=cache)
    reg.register(FeatureSource(
        name="missing", parquet_path=cache / "missing.parquet",
        expected_sha256=None, feature_cols=["a"], version="0.1.0",
    ))
    with pytest.raises(FileNotFoundError):
        reg.load("missing")


def test_load_sha_mismatch_raises(cache: Path):
    parquet = cache / "tampered.parquet"
    _make_features(parquet, ["JCVISYN3A_0001", "JCVISYN3A_0002"],
                    ["x"])
    _write_manifest_entry(cache, "tampered", parquet, "0.1.0")
    # Tamper: rewrite the parquet with different data.
    _make_features(parquet, ["JCVISYN3A_0001", "JCVISYN3A_0002"],
                    ["x"], seed=99)
    reg = FeatureRegistry(cache_dir=cache)
    reg.register(FeatureSource(
        name="tampered", parquet_path=parquet,
        expected_sha256=None, feature_cols=["x"], version="0.1.0",
    ))
    with pytest.raises(ValueError, match="mismatch"):
        reg.load("tampered")


def test_load_no_manifest_entry_raises(cache: Path):
    """Parquet exists but nothing registered in manifest.json —
    we refuse to load untracked features."""
    parquet = cache / "ghost.parquet"
    _make_features(parquet, ["JCVISYN3A_0001"], ["x"])
    # Deliberately skip the manifest add.
    reg = FeatureRegistry(cache_dir=cache)
    reg.register(FeatureSource(
        name="ghost", parquet_path=parquet,
        expected_sha256=None, feature_cols=["x"], version="0.1.0",
    ))
    with pytest.raises(ValueError, match="no manifest entry"):
        reg.load("ghost")


def test_load_valid_returns_dataframe(cache: Path):
    parquet = cache / "good.parquet"
    locus_tags = [f"JCVISYN3A_{i:04d}" for i in range(4)]
    _make_features(parquet, locus_tags, ["dim0", "dim1", "dim2"])
    _write_manifest_entry(cache, "good", parquet, "0.1.0")

    reg = FeatureRegistry(cache_dir=cache)
    reg.register(FeatureSource(
        name="good", parquet_path=parquet,
        expected_sha256=None,
        feature_cols=["dim0", "dim1", "dim2"], version="0.1.0",
    ))
    df = reg.load("good")
    assert df.index.name == "locus_tag"
    assert list(df.index) == locus_tags
    assert list(df.columns) == ["dim0", "dim1", "dim2"]


def test_join_features_two_sources(cache: Path):
    """Two declared sources, one cached + one not — missing source's
    columns come back as NaN for every gene."""
    p1 = cache / "s1.parquet"
    _make_features(p1, ["JCVISYN3A_0001", "JCVISYN3A_0002"], ["a"])
    _write_manifest_entry(cache, "s1", p1, "0.1.0")

    reg = FeatureRegistry(cache_dir=cache)
    reg.register(FeatureSource(
        name="s1", parquet_path=p1,
        expected_sha256=None, feature_cols=["a"], version="0.1.0",
    ))
    reg.register(FeatureSource(
        name="s2", parquet_path=cache / "s2.parquet",
        expected_sha256=None, feature_cols=["b", "c"], version="0.1.0",
    ))

    joined = reg.join_features(
        ["JCVISYN3A_0001", "JCVISYN3A_0002", "JCVISYN3A_0099"]
    )
    assert list(joined.columns) == ["a", "b", "c"]
    assert list(joined.index) == [
        "JCVISYN3A_0001", "JCVISYN3A_0002", "JCVISYN3A_0099",
    ]
    # s2 never cached -> all NaN for b and c.
    assert joined["b"].isna().all()
    assert joined["c"].isna().all()
    # Unknown locus for s1 -> NaN.
    assert pd.isna(joined.loc["JCVISYN3A_0099", "a"])
    # Known locus -> finite.
    assert not pd.isna(joined.loc["JCVISYN3A_0001", "a"])


def test_join_features_preserves_order(cache: Path):
    """Input locus_tag order must be preserved in the output index
    even when the parquet stores them in a different order."""
    parquet = cache / "shuffled.parquet"
    # Parquet stores 0003, 0001, 0002.
    _make_features(parquet, ["JCVISYN3A_0003", "JCVISYN3A_0001",
                              "JCVISYN3A_0002"], ["v"])
    _write_manifest_entry(cache, "shuffled", parquet, "0.1.0")
    reg = FeatureRegistry(cache_dir=cache)
    reg.register(FeatureSource(
        name="shuffled", parquet_path=parquet,
        expected_sha256=None, feature_cols=["v"], version="0.1.0",
    ))
    # Ask for them in yet another order.
    order = ["JCVISYN3A_0001", "JCVISYN3A_0002", "JCVISYN3A_0003"]
    out = reg.join_features(order)
    assert list(out.index) == order


def test_join_features_unknown_source_raises(cache: Path):
    reg = FeatureRegistry(cache_dir=cache)
    with pytest.raises(KeyError, match="unknown source"):
        reg.join_features(
            ["JCVISYN3A_0001"], sources=["never_registered"],
        )


def test_join_features_no_sources_returns_empty_index_frame(cache: Path):
    reg = FeatureRegistry(cache_dir=cache)
    out = reg.join_features(["JCVISYN3A_0001", "JCVISYN3A_0002"])
    assert list(out.index) == ["JCVISYN3A_0001", "JCVISYN3A_0002"]
    assert list(out.columns) == []


def test_base_package_import_has_no_heavy_deps():
    """Regression guard: importing ``cell_sim.features`` must not
    drag in torch / transformers / esm / mace / e3nn / biopython /
    rdkit / huggingface_hub. The feature registry is a cheap read
    abstraction."""
    import importlib
    import sys
    pre = set(sys.modules)
    importlib.import_module("cell_sim.features")
    post = set(sys.modules)
    heavy = {"torch", "transformers", "esm", "mace",
             "e3nn", "biopython", "rdkit", "huggingface_hub"}
    pulled = {m.split(".")[0] for m in (post - pre)}
    offenders = pulled & heavy
    assert not offenders, (
        f"cell_sim.features should not import any heavy deps; "
        f"found: {offenders}"
    )

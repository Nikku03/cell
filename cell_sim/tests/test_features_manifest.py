"""Tests for :class:`CachedFeatureManifest` — on-disk manifest that
tracks the SHA-256 of every cached feature parquet.

Tests do not use any pretrained model — they exercise the manifest
against small byte-blobs on a ``tmp_path`` fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cell_sim.features.cache_manifest import CachedFeatureManifest


def _make_fake_parquet(path: Path, rows: int = 3) -> None:
    """Write a small parquet file for SHA-hashing tests.

    Contents are deterministic given ``rows`` so tests can recompute
    expected hashes if they ever need to.
    """
    df = pd.DataFrame({
        "locus_tag": [f"JCVISYN3A_{i:04d}" for i in range(rows)],
        "val_a": list(range(rows)),
        "val_b": [float(i) * 0.5 for i in range(rows)],
    }).set_index("locus_tag")
    df.to_parquet(path, index=True)


def test_manifest_empty_on_create():
    m = CachedFeatureManifest()
    assert m.sources == {}


def test_manifest_add_computes_sha(tmp_path: Path):
    parquet = tmp_path / "fake.parquet"
    _make_fake_parquet(parquet)
    m = CachedFeatureManifest()
    m.add("fake", parquet, version="0.1.0")
    assert "fake" in m.sources
    entry = m.sources["fake"]
    # SHA is a 64-char lowercase hex digest.
    assert isinstance(entry["sha256"], str)
    assert len(entry["sha256"]) == 64
    assert all(c in "0123456789abcdef" for c in entry["sha256"])
    # version + timestamp + rows populated.
    assert entry["version"] == "0.1.0"
    assert "created_at" in entry and entry["created_at"].endswith("Z")
    # rows should be 3 because pyarrow is installed in this repo.
    assert entry["rows"] == 3


def test_manifest_verify_matches(tmp_path: Path):
    parquet = tmp_path / "fake.parquet"
    _make_fake_parquet(parquet)
    m = CachedFeatureManifest()
    m.add("fake", parquet, version="0.1.0")
    assert m.verify("fake", parquet) is True


def test_manifest_verify_detects_tamper(tmp_path: Path):
    parquet = tmp_path / "fake.parquet"
    _make_fake_parquet(parquet)
    m = CachedFeatureManifest()
    m.add("fake", parquet, version="0.1.0")
    # Append a byte — the parquet is now corrupt AND its SHA changed.
    with open(parquet, "ab") as f:
        f.write(b"\x00")
    assert m.verify("fake", parquet) is False


def test_manifest_roundtrip(tmp_path: Path):
    parquet = tmp_path / "fake.parquet"
    _make_fake_parquet(parquet)
    path = tmp_path / "manifest.json"

    m1 = CachedFeatureManifest()
    m1.add("fake", parquet, version="0.1.0")
    m1.save(path)

    assert path.exists()
    raw = json.loads(path.read_text())
    assert "sources" in raw and "fake" in raw["sources"]

    m2 = CachedFeatureManifest.load(path)
    assert m2.sources == m1.sources
    assert m2.verify("fake", parquet) is True


def test_manifest_load_missing_file_returns_empty(tmp_path: Path):
    """Loading a non-existent manifest must NOT raise — a fresh
    checkout with no cached features should be usable."""
    path = tmp_path / "does_not_exist.json"
    assert not path.exists()
    m = CachedFeatureManifest.load(path)
    assert m.sources == {}


def test_manifest_load_malformed_raises(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text("{\"wrong_key\": 42}")
    with pytest.raises(ValueError, match="malformed"):
        CachedFeatureManifest.load(path)


def test_manifest_add_missing_parquet_raises(tmp_path: Path):
    m = CachedFeatureManifest()
    missing = tmp_path / "nope.parquet"
    with pytest.raises(FileNotFoundError):
        m.add("nope", missing, version="0.1.0")


def test_manifest_remove_noop_on_missing_key(tmp_path: Path):
    m = CachedFeatureManifest()
    # Should not raise.
    m.remove("nonexistent")
    assert m.sources == {}


def test_manifest_verify_missing_parquet_returns_false(tmp_path: Path):
    m = CachedFeatureManifest()
    m.sources["ghost"] = {
        "sha256": "0" * 64,
        "version": "0.0.0",
        "rows": 0,
        "created_at": "2000-01-01T00:00:00Z",
    }
    assert m.verify("ghost", tmp_path / "not_there.parquet") is False


def test_manifest_verify_no_entry_returns_false(tmp_path: Path):
    parquet = tmp_path / "fake.parquet"
    _make_fake_parquet(parquet)
    m = CachedFeatureManifest()
    # No add() call; verify should still return False (not raise).
    assert m.verify("fake", parquet) is False

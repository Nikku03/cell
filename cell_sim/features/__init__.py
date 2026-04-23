"""``cell_sim.features``: per-gene pretrained-model feature caching.

This package is pure *plumbing* for large-model feature extraction
— it provides a consistent on-disk format (parquet keyed by
``locus_tag``), integrity validation (SHA256 against a manifest),
and an abstract base for GPU-batched inference.

No pretrained model weights are loaded or downloaded by this
package. Subclasses of :class:`BatchedFeatureExtractor` are
expected to import their own heavy dependencies (``torch``,
``transformers``, ``mace``, ``biopython``, …) *inside* their
``extract`` method so that importing ``cell_sim.features`` remains
cheap and side-effect-free.

Basic usage:

    >>> from pathlib import Path
    >>> from cell_sim.features import (
    ...     FeatureRegistry, FeatureSource,
    ... )
    >>> reg = FeatureRegistry()
    >>> reg.register(FeatureSource(
    ...     name="esm2_650M",
    ...     parquet_path=Path("cell_sim/features/cache/esm2_650M.parquet"),
    ...     expected_sha256=None,        # filled in once computed
    ...     feature_cols=[f"esm2_650M_dim_{i}" for i in range(1280)],
    ...     version="0.1.0",
    ... ))
    >>> reg.is_cached("esm2_650M")         # False until populated
    False
    >>> # A later session populates the cache via a subclass of
    >>> # BatchedFeatureExtractor that writes the parquet and updates
    >>> # manifest.json. Once that happens, reg.load('esm2_650M')
    >>> # returns a DataFrame indexed by locus_tag.
"""
from __future__ import annotations

from .feature_registry import FeatureRegistry, FeatureSource
from .cache_manifest import CachedFeatureManifest
from .batched_inference import (
    BatchedFeatureExtractor,
    BatchedInferenceConfig,
)

__all__ = [
    "FeatureRegistry",
    "FeatureSource",
    "CachedFeatureManifest",
    "BatchedFeatureExtractor",
    "BatchedInferenceConfig",
]

__version__ = "0.1.0"

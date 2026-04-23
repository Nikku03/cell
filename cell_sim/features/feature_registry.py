"""In-process registry of cached per-gene feature tables.

The registry keeps a dict of :class:`FeatureSource` entries describing
parquet files that live under ``cell_sim/features/cache/``. Entries are
*declared* up-front (``register``); they are *not* loaded until the
caller asks for them via :meth:`FeatureRegistry.load` or
:meth:`FeatureRegistry.join_features`. Each load re-validates the
parquet's SHA-256 against the manifest on disk.

Design constraints baked in:

* **Purity of join**: :meth:`join_features` never triggers a feature
  computation. Missing sources are filled with ``NaN`` across their
  declared ``feature_cols``. If you want features, you invoke the
  extractor explicitly — the registry is just a read-side abstraction.

* **Dependency hygiene**: this module imports only ``pandas``,
  ``hashlib``, ``json``, ``pathlib``, ``dataclasses``, and stdlib.
  No ``torch``, no ``transformers``, no ``biopython``. Heavy imports
  belong inside extractor subclasses, never here.

* **Deterministic output**: the DataFrame returned by
  :meth:`join_features` is indexed by the input ``locus_tags`` in the
  order the caller supplied. Column order is
  ``[source1 cols..., source2 cols..., ...]`` in registration order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from cell_sim.features.cache_manifest import (
    CachedFeatureManifest,
    _sha256_of_file,
)


@dataclass
class FeatureSource:
    """Metadata for one parquet-cached feature table.

    ``parquet_path`` should be a relative path under the cache dir,
    e.g. ``cache/esm2_650M.parquet``. ``expected_sha256`` may be
    ``None`` until the source is first produced; once the producer
    runs and updates the manifest, the registry will pick up the
    hash from there. If both are non-None and disagree, the registry
    treats the source as corrupt.
    """
    name: str
    parquet_path: Path
    expected_sha256: Optional[str]
    feature_cols: list[str]
    version: str


class FeatureRegistry:
    """Declarative-then-on-demand loader for cached feature tables.

    The registry neither triggers extractions nor imports extractor
    code. It answers four questions:

      * Is source X declared?           ``list_sources``
      * Is source X materialised?       ``is_cached``
      * Give me the DataFrame.          ``load``
      * Join multiple sources by gene.  ``join_features``
    """

    DEFAULT_CACHE_DIR = Path("cell_sim/features/cache")

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self._sources: dict[str, FeatureSource] = {}
        # Manifest loaded lazily so a fresh registry on a clean checkout
        # doesn't fail on a missing manifest file — it just sees an
        # empty manifest (consistent with "nothing is cached yet").
        self._manifest_path = self.cache_dir / "manifest.json"

    # ---- registration ----

    def register(self, source: FeatureSource) -> None:
        """Record a :class:`FeatureSource`.

        Re-registering under the same ``name`` replaces the prior
        entry. The parquet need not exist yet — registration only
        says "I intend to read this file later", not "load it now".
        """
        self._sources[source.name] = source

    def unregister(self, name: str) -> None:
        """Forget a source. No-op if not registered."""
        self._sources.pop(name, None)

    def list_sources(self) -> list[str]:
        """Names of declared sources, in registration order."""
        return list(self._sources.keys())

    def get_source(self, name: str) -> FeatureSource:
        """Return the :class:`FeatureSource` declared under ``name``."""
        if name not in self._sources:
            raise KeyError(
                f"FeatureRegistry has no declared source {name!r}. "
                f"Call register(...) first. "
                f"Known sources: {self.list_sources()!r}"
            )
        return self._sources[name]

    # ---- state inspection ----

    def is_cached(self, name: str) -> bool:
        """True iff the parquet for ``name`` exists on disk.

        Does NOT verify the SHA or that the file is a valid parquet
        (that happens in :meth:`load`). This is the cheap check.
        """
        if name not in self._sources:
            return False
        return self._sources[name].parquet_path.exists()

    def _load_manifest(self) -> CachedFeatureManifest:
        return CachedFeatureManifest.load(self._manifest_path)

    # ---- loading ----

    def load(self, name: str) -> pd.DataFrame:
        """Load the parquet for ``name`` and return a ``DataFrame``
        indexed by ``locus_tag``.

        Validates integrity in two steps:

          1. ``FileNotFoundError`` if the parquet is missing.
          2. ``ValueError`` if the on-disk SHA-256 doesn't match the
             manifest's ``sources[name]["sha256"]``. (If the manifest
             has no entry for ``name``, that is also a ValueError —
             "parquet exists but wasn't produced by a tracked run".)

        In the second case, callers should either re-run the
        extractor or call
        :meth:`CachedFeatureManifest.add` via the extractor's
        ``write_cache`` method to re-register the fresh content.
        """
        source = self.get_source(name)
        if not source.parquet_path.exists():
            raise FileNotFoundError(
                f"feature source {name!r} is declared but not "
                f"cached: {source.parquet_path} does not exist. "
                f"Run the corresponding extractor first."
            )

        manifest = self._load_manifest()
        entry = manifest.sources.get(name)
        stored_sha = (entry or {}).get("sha256")
        actual_sha = _sha256_of_file(source.parquet_path)

        if stored_sha is None:
            raise ValueError(
                f"feature source {name!r} is cached at "
                f"{source.parquet_path} but has no manifest entry. "
                f"Either the manifest is stale or the parquet was "
                f"produced outside the tracked extractor. Refusing "
                f"to load untracked features."
            )
        if stored_sha != actual_sha:
            raise ValueError(
                f"feature source {name!r} SHA-256 mismatch: "
                f"manifest says {stored_sha}, file is {actual_sha}. "
                f"The parquet has been modified since registration, "
                f"or the manifest is stale. Re-run the extractor "
                f"or call CachedFeatureManifest.add() to refresh."
            )
        if source.expected_sha256 is not None \
                and source.expected_sha256 != actual_sha:
            raise ValueError(
                f"feature source {name!r} SHA-256 mismatch vs "
                f"FeatureSource.expected_sha256: declared "
                f"{source.expected_sha256}, file is {actual_sha}."
            )

        df = pd.read_parquet(source.parquet_path)
        if "locus_tag" in df.columns:
            df = df.set_index("locus_tag")
        elif df.index.name != "locus_tag":
            raise ValueError(
                f"feature source {name!r} parquet at "
                f"{source.parquet_path} has no 'locus_tag' index or "
                f"column. Every cached feature table must be keyed by "
                f"locus_tag."
            )
        return df

    # ---- joining ----

    def join_features(
        self,
        locus_tags: list[str],
        sources: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame indexed by ``locus_tags`` with columns
        drawn from every requested source.

        * ``sources=None`` means all declared sources.
        * Missing sources (declared but not cached on disk) produce
          a block of ``NaN`` columns — no exception, no extraction
          triggered.
        * Row order matches the input ``locus_tags`` list exactly.
        * Duplicate locus_tags in the input are preserved (same row
          repeated). This matches what the detector-sweep code
          expects when evaluating a gene multiple times under
          different seeds.
        """
        names = sources if sources is not None else self.list_sources()
        missing_decl = [n for n in names if n not in self._sources]
        if missing_decl:
            raise KeyError(
                f"join_features: unknown source(s) {missing_decl!r}. "
                f"Known: {self.list_sources()!r}"
            )

        frames: list[pd.DataFrame] = []
        for name in names:
            source = self._sources[name]
            if self.is_cached(name):
                try:
                    block = self.load(name)
                    # Keep only the declared feature_cols. If the
                    # parquet has extra columns (e.g. sidecar raw
                    # attention matrices), we ignore them here —
                    # the contract is feature_cols.
                    block = block.reindex(columns=source.feature_cols)
                except (FileNotFoundError, ValueError):
                    # Treat integrity failures the same as not-cached:
                    # the join is pure and must never raise. The full
                    # error surfaces if the caller invokes .load()
                    # directly.
                    block = _empty_block(locus_tags, source.feature_cols)
            else:
                block = _empty_block(locus_tags, source.feature_cols)
            # Reindex to the caller's ordering; unknown rows -> NaN.
            block = block.reindex(locus_tags)
            frames.append(block)

        if not frames:
            # No sources declared; return an empty DataFrame but keep
            # the caller's row ordering so downstream column-adds
            # align correctly.
            return pd.DataFrame(index=pd.Index(locus_tags, name="locus_tag"))
        out = pd.concat(frames, axis=1)
        out.index.name = "locus_tag"
        return out


def _empty_block(locus_tags: list[str], cols: list[str]) -> pd.DataFrame:
    """NaN-filled DataFrame with the declared columns."""
    empty = pd.DataFrame(
        {c: [float("nan")] * len(locus_tags) for c in cols},
    )
    empty.index = pd.Index(locus_tags, name="locus_tag")
    return empty


__all__ = ["FeatureRegistry", "FeatureSource"]

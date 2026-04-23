"""Abstract base for pretrained-model feature extractors.

A concrete subclass (e.g. ``ESM2Extractor``, ``AlphaFoldLookup``,
``MaceOffKcat``) takes per-gene input — typically a DataFrame with
``locus_tag`` plus one or more input columns such as ``sequence`` for
protein LMs or ``pdb_path`` for structure-derived features — and
returns a DataFrame of dense features indexed by ``locus_tag``.

Dependency hygiene (strict)
---------------------------
This base class is deliberately import-free beyond stdlib + pandas.
Subclasses MUST perform their heavy imports (``torch``,
``transformers``, ``mace``, ``biopython``, …) **inside** their
``extract`` method, or inside a private ``_ensure_loaded`` helper
that ``extract`` calls. The pattern exists precisely to keep
``import cell_sim.features`` cheap on machines that will never run
the extractor.

Pattern mirrors ``cell_sim/layer1_atomic/engine.py::MACEBackend``:

    class ESM2Extractor(BatchedFeatureExtractor):
        name = "esm2_650M"
        version = "0.1.0"
        feature_cols = [f"esm2_650M_dim_{i}" for i in range(1280)]

        _model = None

        def _ensure_loaded(self, config):
            if self._model is not None:
                return
            # Heavy imports happen HERE, not at module top level.
            import torch
            from transformers import AutoTokenizer, AutoModel
            ...

        def extract(self, inputs, config):
            self._ensure_loaded(config)
            ...

Contract
--------
``extract(inputs, config) -> DataFrame`` MUST return a DataFrame
indexed by ``locus_tag`` whose columns are exactly ``self.feature_cols``
(in order). Extra columns are a contract violation. Empty inputs
return an empty DataFrame with the correct columns.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class BatchedInferenceConfig:
    """Configuration passed to every ``extract`` call.

    Kept deliberately small — extractor-specific knobs live in the
    subclass constructor, not here. The fields in this dataclass are
    the ones that must be tunable at call time for throughput
    reasons.
    """
    batch_size: int = 32
    device: str = "auto"           # "auto" | "cuda" | "cpu"
    dtype: str = "float32"         # "float32" | "float16" | "bfloat16"
    max_seq_length: Optional[int] = None
    num_workers: int = 0           # DataLoader worker count if applicable
    progress: bool = False         # whether the extractor may print/tqdm

    def resolve_device(self) -> str:
        """Resolve ``"auto"`` to ``"cuda"`` if CUDA is visible,
        else ``"cpu"``. Performs the ``torch`` import lazily so a
        bare ``import BatchedInferenceConfig`` stays light.
        """
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


class BatchedFeatureExtractor(ABC):
    """Abstract base for pretrained-model feature extractors.

    Subclasses set three class attributes:

      * ``name`` — short, filesystem-safe identifier, used as the
        parquet filename and manifest key.
      * ``version`` — semver-ish tag; bumped whenever the feature
        semantics change (weights upgrade, tokenizer change, …).
      * ``feature_cols`` — ordered list of DataFrame column names
        produced by ``extract``.

    Subclasses implement one required method, ``extract``. They MAY
    override :meth:`write_cache` if they need custom writer logic,
    but the default implementation (``DataFrame.to_parquet`` +
    :class:`CachedFeatureManifest.add`) is expected to cover every
    standard case.
    """

    # Type stubs; subclasses override.
    name: str = ""
    version: str = "0.0.0"
    feature_cols: list[str] = []

    @abstractmethod
    def extract(
        self,
        inputs: pd.DataFrame,
        config: BatchedInferenceConfig,
    ) -> pd.DataFrame:
        """Compute features for the rows of ``inputs``.

        ``inputs`` MUST contain a ``locus_tag`` column. Any other
        columns are extractor-specific (e.g. ``sequence`` for ESM-2,
        ``pdb_path`` for AlphaFold-derived features). The return
        value is a DataFrame indexed by ``locus_tag`` with columns
        exactly ``self.feature_cols``.

        This method is the one place where subclasses are allowed
        to import ``torch`` / ``transformers`` / etc. Do the import
        inside the method body (or inside a private helper called
        from the method body) — never at module top level.
        """
        raise NotImplementedError

    # ---- cache writing ----

    def write_cache(
        self,
        features: pd.DataFrame,
        cache_dir: Path = Path("cell_sim/features/cache"),
    ) -> tuple[Path, str]:
        """Write ``features`` as parquet and update ``manifest.json``.

        Returns ``(parquet_path, sha256_hex)``. The parquet filename
        is ``<name>.parquet`` under ``cache_dir``; the manifest sits
        at ``cache_dir / manifest.json``.

        Validates:

          * ``features`` is indexed by ``locus_tag`` (raises ValueError
            if not).
          * ``features.columns`` is exactly ``self.feature_cols``
            (raises ValueError if extra / missing columns).

        These checks keep extractor subclasses honest and guarantee
        the registry can re-read the parquet without surprises.
        """
        # Lazy import so simply constructing an extractor has no
        # dependency on the manifest module.
        from cell_sim.features.cache_manifest import (
            CachedFeatureManifest,
        )

        if features.index.name != "locus_tag":
            raise ValueError(
                f"{type(self).__name__}.write_cache: features must be "
                f"indexed by locus_tag, got index.name="
                f"{features.index.name!r}"
            )
        declared = list(self.feature_cols)
        got = list(features.columns)
        if got != declared:
            raise ValueError(
                f"{type(self).__name__}.write_cache: features have "
                f"columns {got!r}, expected {declared!r} (order "
                f"must match). Fix the extractor's output to match "
                f"its declared feature_cols."
            )

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = cache_dir / f"{self.name}.parquet"
        # Writing with the index ensures locus_tag is persisted.
        features.to_parquet(parquet_path, index=True)

        manifest_path = cache_dir / "manifest.json"
        manifest = CachedFeatureManifest.load(manifest_path)
        manifest.add(self.name, parquet_path, version=self.version)
        manifest.save(manifest_path)

        sha = manifest.sources[self.name]["sha256"]
        return parquet_path, sha


__all__ = [
    "BatchedFeatureExtractor",
    "BatchedInferenceConfig",
]

"""Concrete :class:`BatchedFeatureExtractor` subclasses for the Tier-1
pretrained-model stack.

Three extractors ship here:

  * :class:`ESM2Extractor`       — facebook/esm2_t33_650M_UR50D
                                   per-protein 1280-dim embedding.
  * :class:`AlphaFoldExtractor`  — AlphaFold-DB per-protein structural
                                   descriptors (pLDDT, SS composition,
                                   radius of gyration).
  * :class:`MaceOffExtractor`    — BDE-derived effective k_cat per
                                   enzyme (aggregated across its
                                   substrates). Wraps the existing
                                   ``cell_sim.layer1_atomic.engine
                                   .MACEBackend``.

Every extractor performs all its heavy imports
(``torch`` / ``transformers`` / ``biopython`` / ``mace-torch`` /
``rdkit`` / ``ase`` / ``requests``) inside ``_ensure_loaded`` or inline
in ``extract`` — never at module top level. That means ``import
cell_sim.features.extractors`` is as cheap as ``import
cell_sim.features`` itself, verified by regression tests in
``cell_sim/tests/test_*_extractor.py::test_module_import_cheap``.

Populating the cache is deliberately a *separate* exercise from
importing the module: the Colab notebook
``notebooks/populate_tier1_cache.ipynb`` instantiates each extractor,
calls ``.extract(...)`` on the full Syn3A proteome, and writes the
three resulting parquets + a refreshed ``manifest.json``. After that
notebook has run and the parquets are in the cache directory, the
:class:`FeatureRegistry` picks them up transparently.
"""
from __future__ import annotations

from cell_sim.features.extractors.esm2_extractor import ESM2Extractor
from cell_sim.features.extractors.alphafold_extractor import (
    AlphaFoldExtractor,
)
from cell_sim.features.extractors.mace_off_extractor import (
    MaceOffExtractor,
)

__all__ = [
    "ESM2Extractor",
    "AlphaFoldExtractor",
    "MaceOffExtractor",
]

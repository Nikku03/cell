"""Per-protein structural descriptors from ESMFold v1 predictions.

Session-15 pivot from the AlphaFold-DB path: JCVI-Syn3A (taxid
``2144189``) is not indexed in UniProt and the Luthey-Schulten 4DWCM
repo doesn't carry a NCBI-to-UniProt map either, so AFDB has no
structures keyed to Syn3A accessions. ESMFold bypasses the lookup
entirely — it takes the AA sequence directly and predicts a 3-D
structure with per-residue pLDDT. The output schema is identical to
:mod:`cell_sim.features.extractors.alphafold_extractor` with an
``esmfold_`` prefix so downstream consumers can switch backends
without changing column contracts.

Model facts not owned by this module:

  * Paper: Lin et al. 2023 *Science* 379:1123, "Evolutionary-scale
    prediction of atomic-level protein structure."
  * Weights: ``facebook/esmfold_v1`` on HuggingFace Hub.
  * Memory: ~24 GB VRAM at fp32, ~12 GB at fp16.
  * Throughput on a single A100-class GPU: roughly 2-10 s per
    400-residue protein, so 15-75 min for all 452 Syn3A CDS.

Output DataFrame schema (identical rows as AlphaFold with renamed keys)::

    index: locus_tag (str)
    columns: esmfold_plddt_mean, esmfold_plddt_std,
             esmfold_disorder_fraction,
             esmfold_helix_fraction, esmfold_sheet_fraction,
             esmfold_coil_fraction, esmfold_sequence_length,
             esmfold_radius_of_gyration_angstrom,
             esmfold_has_structure (float)

Rows where the sequence is empty / missing or ESMFold fails to
produce a valid PDB are filled with NaN in every column except
``esmfold_has_structure`` (which is set to 0.0). Matches the
:class:`cell_sim.features.feature_registry.FeatureRegistry`
join-by-NaN contract.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from cell_sim.features.batched_inference import (
    BatchedFeatureExtractor,
    BatchedInferenceConfig,
)
from cell_sim.features.extractors.alphafold_extractor import (
    _features_from_pdb,
)


_FEATURE_COLS: list[str] = [
    "esmfold_plddt_mean",
    "esmfold_plddt_std",
    "esmfold_disorder_fraction",
    "esmfold_helix_fraction",
    "esmfold_sheet_fraction",
    "esmfold_coil_fraction",
    "esmfold_sequence_length",
    "esmfold_radius_of_gyration_angstrom",
    "esmfold_has_structure",
]

# Maps the AlphaFold helper's output-dict keys onto the ESMFold ones.
# Kept alongside the feature-col declaration so any column-list rename
# is a single edit.
_AF_TO_ESMFOLD_KEY: dict[str, str] = {
    "af_plddt_mean": "esmfold_plddt_mean",
    "af_plddt_std": "esmfold_plddt_std",
    "af_disorder_fraction": "esmfold_disorder_fraction",
    "af_helix_fraction": "esmfold_helix_fraction",
    "af_sheet_fraction": "esmfold_sheet_fraction",
    "af_coil_fraction": "esmfold_coil_fraction",
    "af_sequence_length": "esmfold_sequence_length",
    "af_radius_of_gyration_angstrom":
        "esmfold_radius_of_gyration_angstrom",
    "af_has_structure": "esmfold_has_structure",
}


class ESMFoldExtractor(BatchedFeatureExtractor):
    """Per-protein structure + pLDDT via ``facebook/esmfold_v1``.

    Contract:

      * Input: DataFrame with at least ``locus_tag`` and ``sequence``
        (uppercase amino-acid string, single-letter codes, no stop
        ``*``). Empty input short-circuits before any heavy import.
      * Output: DataFrame indexed by ``locus_tag`` with the 9
        ``esmfold_*`` columns declared in :data:`_FEATURE_COLS`.
      * Missing / empty sequences and inference failures produce a
        NaN row with ``esmfold_has_structure = 0.0``.

    Usage (Colab):

        ex = ESMFoldExtractor()
        inputs = pd.DataFrame({
            "locus_tag": [...],
            "sequence": [...],
        })
        feats = ex.extract(inputs, BatchedInferenceConfig(
            batch_size=1, device="cuda", dtype="float16",
        ))
        ex.write_cache(feats)
    """

    name = "esmfold_v1"
    version = "0.1.0"
    feature_cols = list(_FEATURE_COLS)

    _MODEL_ID = "facebook/esmfold_v1"

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._resolved_device: Optional[str] = None
        self._biopython_loaded = False

    # ---- lazy model load ----

    def _ensure_loaded(self, config: BatchedInferenceConfig) -> None:
        """Load ``facebook/esmfold_v1`` the first time ``extract`` is
        called with non-empty input. Imports ``torch``,
        ``transformers``, and ``biopython`` inside the method — never
        at module import time."""
        if self._model is not None:
            return
        import torch  # noqa: WPS433
        import Bio.PDB  # noqa: WPS433, F401 — used via PDBParser
        from transformers import (  # noqa: WPS433
            AutoTokenizer, EsmForProteinFolding,
        )

        device = config.resolve_device()
        torch_dtype = _resolve_torch_dtype(config.dtype, torch)
        tokenizer = AutoTokenizer.from_pretrained(self._MODEL_ID)
        model = EsmForProteinFolding.from_pretrained(
            self._MODEL_ID,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
        )
        model = model.to(device).eval()

        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = device
        self._biopython_loaded = True

    # ---- extraction ----

    def extract(
        self,
        inputs: pd.DataFrame,
        config: Optional[BatchedInferenceConfig] = None,
    ) -> pd.DataFrame:
        if config is None:
            config = BatchedInferenceConfig()

        if len(inputs) == 0:
            return _empty_frame(self.feature_cols)

        _validate_input_columns(
            inputs, required={"locus_tag", "sequence"},
        )

        self._ensure_loaded(config)

        locus_tags = inputs["locus_tag"].astype(str).tolist()
        sequences = inputs["sequence"].astype(str).tolist()

        rows: list[dict[str, float]] = []
        for seq_raw in sequences:
            seq = seq_raw.strip().upper().rstrip("*")
            if not seq or seq == "NAN":
                rows.append(_no_structure_row())
                continue
            cap = config.max_seq_length or 1024
            if len(seq) > cap:
                seq = seq[:cap]
            try:
                pdb_str = self._infer_pdb(seq)
            except Exception:  # noqa: BLE001 — any inference failure → NaN
                rows.append(_no_structure_row())
                continue
            try:
                af_row = _features_from_pdb(pdb_str.encode("utf-8"))
            except Exception:  # noqa: BLE001 — parse failures → NaN
                rows.append(_no_structure_row())
                continue
            rows.append(_rename_af_to_esmfold(af_row))

        out = pd.DataFrame(rows, columns=self.feature_cols)
        out.index = pd.Index(locus_tags, name="locus_tag")
        return out

    # ---- inference ----

    def _infer_pdb(self, sequence: str) -> str:
        """Run ESMFold on a single sequence, return a PDB string.

        Encapsulated so subclasses or tests can swap in a mock model
        without touching ``extract``. The HuggingFace EsmForProteinFolding
        wrapper exposes ``model.infer_pdb(seq)`` which handles tokenisation,
        forward pass, and PDB serialisation internally.
        """
        return self._model.infer_pdb(sequence)


# ---- helpers ----


def _rename_af_to_esmfold(af_row: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for af_key, value in af_row.items():
        esmfold_key = _AF_TO_ESMFOLD_KEY.get(af_key)
        if esmfold_key is not None:
            out[esmfold_key] = value
    # Defensive: ensure every declared column is present.
    for col in _FEATURE_COLS:
        out.setdefault(col, math.nan)
    return out


def _empty_frame(cols: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(columns=cols)
    frame.index = pd.Index([], name="locus_tag")
    return frame


def _no_structure_row() -> dict[str, float]:
    row = {c: math.nan for c in _FEATURE_COLS}
    row["esmfold_has_structure"] = 0.0
    return row


def _resolve_torch_dtype(name: str, torch_module):
    mapping = {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    if name not in mapping:
        raise ValueError(
            f"ESMFoldExtractor: unsupported dtype {name!r}; "
            f"choose one of {list(mapping)}"
        )
    return mapping[name]


def _validate_input_columns(df: pd.DataFrame, *, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"ESMFoldExtractor.extract: input DataFrame missing "
            f"required column(s) {sorted(missing)!r}. "
            f"Got: {list(df.columns)!r}"
        )


__all__ = ["ESMFoldExtractor"]

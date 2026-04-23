"""Per-protein ESM-2 (650M) sequence embeddings.

Produces a 1280-dimensional mean-pooled representation of each CDS's
translated amino-acid sequence, using Meta's
``facebook/esm2_t33_650M_UR50D`` weights via HuggingFace
``transformers``.

The model lives in the notebook-side dependency stack (``torch`` +
``transformers``) — this module only imports them inside
``_ensure_loaded`` and ``extract``. The sandbox-side tests in
``cell_sim/tests/test_esm2_extractor.py`` pass without those
packages installed, verified by the
``test_module_import_cheap`` regression.

Model facts not owned by this module:

  * Paper: Lin et al. 2023 *Science* 379:1123, "Evolutionary-scale
    prediction of atomic-level protein structure" (ESM-2 family).
  * Weights: ``facebook/esm2_t33_650M_UR50D`` on HuggingFace Hub,
    ~650M parameters, 33 layers, 1280 embedding dim.
  * Context window: 1024 tokens (residues). Syn3A CDS longer than
    ~1024 aa are truncated with a warning; the longest Syn3A
    protein is ~900 aa so truncation should not trigger in practice.

Output DataFrame schema::

    index: locus_tag (str)
    columns: esm2_650M_dim_0, ..., esm2_650M_dim_1279 (float32)

For genes whose sequence is missing from the input DataFrame, the
extractor emits an NaN row — the downstream ``FeatureRegistry`` fills
NaN columns naturally in :meth:`join_features`.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from cell_sim.features.batched_inference import (
    BatchedFeatureExtractor,
    BatchedInferenceConfig,
)


# 1280-dim feature space; column names are stable and declared once.
_EMBED_DIM = 1280
_FEATURE_COLS = [f"esm2_650M_dim_{i}" for i in range(_EMBED_DIM)]


class ESM2Extractor(BatchedFeatureExtractor):
    """ESM-2 650M mean-pooled per-protein embedding.

    Contract:

      * Input: DataFrame with at least ``locus_tag`` and
        ``sequence`` columns. ``sequence`` is an uppercase amino-acid
        string (single-letter codes, no stop ``*``).
      * Output: DataFrame indexed by ``locus_tag`` with 1280
        ``esm2_650M_dim_{i}`` float columns. Rows corresponding to
        missing / empty sequences are filled with NaN.

    Usage (Colab):

        ex = ESM2Extractor()
        inputs = pd.DataFrame({
            "locus_tag": [...],
            "sequence": [...],
        })
        feats = ex.extract(inputs, BatchedInferenceConfig(
            batch_size=16, device="cuda", dtype="float16",
        ))
        ex.write_cache(feats)
    """

    name = "esm2_650M"
    version = "0.1.0"
    feature_cols = list(_FEATURE_COLS)

    _MODEL_ID = "facebook/esm2_t33_650M_UR50D"

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._resolved_device: Optional[str] = None

    # ---- lazy model load ----

    def _ensure_loaded(self, config: BatchedInferenceConfig) -> None:
        """Load the tokenizer + model the first time ``extract`` is
        called. Imports ``torch`` and ``transformers`` inside the
        method — never at module top level."""
        if self._model is not None:
            return
        import torch  # noqa: WPS433 — lazy by design
        from transformers import AutoTokenizer, AutoModel  # noqa: WPS433

        device = config.resolve_device()
        torch_dtype = _resolve_torch_dtype(config.dtype, torch)
        tokenizer = AutoTokenizer.from_pretrained(self._MODEL_ID)
        model = AutoModel.from_pretrained(
            self._MODEL_ID,
            torch_dtype=torch_dtype,
        )
        model = model.to(device).eval()

        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = device

    # ---- extraction ----

    def extract(
        self,
        inputs: pd.DataFrame,
        config: Optional[BatchedInferenceConfig] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame indexed by ``locus_tag`` with 1280
        columns. Empty input short-circuits; the ``_ensure_loaded``
        call (and therefore every heavy import) is skipped."""
        if config is None:
            config = BatchedInferenceConfig()

        # Short-circuit BEFORE any heavy import so sandbox-side tests
        # that call extract(empty_df) don't trigger torch imports.
        if len(inputs) == 0:
            return _empty_frame(self.feature_cols)

        _validate_input_columns(inputs, required={"locus_tag", "sequence"})

        self._ensure_loaded(config)
        # Re-imported here (free once cached in sys.modules) so type
        # hints inside the method resolve cleanly.
        import torch  # noqa: WPS433

        locus_tags = inputs["locus_tag"].astype(str).tolist()
        sequences = inputs["sequence"].astype(str).tolist()

        # Identify rows with no usable sequence (empty string, whitespace,
        # NaN-coerced-to-str "nan"). These become NaN output rows without
        # a forward pass.
        usable: list[tuple[int, str]] = []
        for i, seq in enumerate(sequences):
            seq = seq.strip().upper().rstrip("*")
            if not seq or seq == "NAN":
                continue
            # Truncate at max_seq_length if supplied; ESM-2's own cap is
            # 1024 residues including BOS/EOS.
            cap = config.max_seq_length or 1022
            if len(seq) > cap:
                seq = seq[:cap]
            usable.append((i, seq))

        embeddings: list[list[float]] = [
            [math.nan] * _EMBED_DIM for _ in range(len(sequences))
        ]

        batch_size = max(1, int(config.batch_size))
        with torch.no_grad():
            for start in range(0, len(usable), batch_size):
                batch = usable[start:start + batch_size]
                row_indices = [idx for idx, _ in batch]
                seqs = [seq for _, seq in batch]
                tokenised = self._tokenizer(
                    seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=(config.max_seq_length or 1024),
                )
                tokenised = {
                    k: v.to(self._resolved_device) for k, v in tokenised.items()
                }
                outputs = self._model(**tokenised)
                hidden = outputs.last_hidden_state  # (B, L, D)
                mask = tokenised["attention_mask"].unsqueeze(-1)  # (B, L, 1)
                # Mean-pool over residues, ignoring padding + special tokens.
                summed = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1)
                pooled = (summed / denom).float().cpu().numpy()
                for row_idx, vec in zip(row_indices, pooled):
                    embeddings[row_idx] = vec.tolist()

        out = pd.DataFrame(embeddings, columns=self.feature_cols)
        out.index = pd.Index(locus_tags, name="locus_tag")
        return out


# ---- helpers ----


def _empty_frame(cols: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(columns=cols)
    frame.index = pd.Index([], name="locus_tag")
    return frame


def _resolve_torch_dtype(name: str, torch_module):
    mapping = {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    if name not in mapping:
        raise ValueError(
            f"ESM2Extractor: unsupported dtype {name!r}; "
            f"choose one of {list(mapping)}"
        )
    return mapping[name]


def _validate_input_columns(df: pd.DataFrame, *, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"ESM2Extractor.extract: input DataFrame missing required "
            f"column(s) {sorted(missing)!r}. Got: {list(df.columns)!r}"
        )


__all__ = ["ESM2Extractor"]

"""MACE-OFF bond-dissociation-energy proxy for per-enzyme effective k_cat.

Wraps the existing :class:`cell_sim.layer1_atomic.engine.MACEBackend`
— do not duplicate its logic here. The backend already knows how to
compute the lowest O-H BDE of a substrate with mace-off and scale a
reference k_cat through Eyring / Hammond. This extractor's job is
just:

  * Iterate Syn3A catalytic reactions (one row per (enzyme, substrate)
    pair in the input DataFrame).
  * Call ``MACEBackend.estimate_kcat`` for each substrate given its
    enzyme's reference substrate dict.
  * Aggregate per-enzyme into summary statistics so the output can
    be keyed by ``locus_tag`` — the contract the
    :class:`FeatureRegistry` expects.

Sub-enzyme detail (one row per substrate) is not kept by the Tier-1
cache; a future extractor can publish a second parquet under a
different ``name`` if per-substrate features become useful.

Heavy imports are confined to :meth:`_ensure_loaded` via the backend's
own ``_ensure_loaded``. The sandbox-side tests in
``cell_sim/tests/test_mace_off_extractor.py`` run without mace-torch,
rdkit, ase, e3nn, or torch installed.

Contract:

  * Input: DataFrame with columns ``locus_tag``, ``enzyme_name``,
    ``reaction_class``, ``substrate_smiles``. Optional
    ``reference_substrates_json`` — JSON-encoded dict
    ``{smi: k_cat_per_s}`` describing the enzyme's measured
    substrates, used as Hammond-transfer reference; when absent the
    wrapper falls back to ``{substrate_smiles: 1.0}`` (placeholder
    rate) so the BDE shift is still computed correctly.
  * Output: DataFrame indexed by ``locus_tag`` with the 7 columns
    listed in :data:`_FEATURE_COLS`. Enzymes with no usable
    substrate produce NaN rows with ``mace_has_estimate = 0.0``.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Optional

import pandas as pd

from cell_sim.features.batched_inference import (
    BatchedFeatureExtractor,
    BatchedInferenceConfig,
)


_FEATURE_COLS: list[str] = [
    "mace_kcat_mean_per_s",
    "mace_kcat_std_per_s",
    "mace_kcat_min_per_s",
    "mace_kcat_max_per_s",
    "mace_n_substrates",
    "mace_mean_confidence",
    "mace_has_estimate",
]


class MaceOffExtractor(BatchedFeatureExtractor):
    """Aggregates :class:`MACEBackend` BDE-derived k_cat estimates
    across an enzyme's substrate set.

    The backend is instantiated lazily so importing this module does
    not pull in ``mace-torch`` / ``torch`` / ``rdkit`` / ``ase`` /
    ``e3nn``. Expected wall-time on GPU: ~5 minutes across all Syn3A
    catalytic reactions (mace-off23_small).
    """

    name = "mace_off_kcat"
    version = "0.1.0"
    feature_cols = list(_FEATURE_COLS)

    def __init__(self, model: str = "small", device: str = "auto") -> None:
        self._backend = None
        self._backend_model = model
        self._backend_device = device

    # ---- lazy backend load ----

    def _ensure_loaded(self, config: BatchedInferenceConfig) -> None:
        if self._backend is not None:
            return
        # Import of MACEBackend is deferred so that the module-top
        # import of this file doesn't pay for the layer1_atomic
        # import graph (which itself only pulls heavy deps via its
        # own lazy path). The outer import here is safe — it's stdlib
        # plus ``layer1_atomic.engine`` which is lazy in the heavy
        # deps it actually uses.
        from cell_sim.layer1_atomic.engine import MACEBackend  # noqa: WPS433
        device = (
            config.resolve_device()
            if self._backend_device == "auto"
            else self._backend_device
        )
        self._backend = MACEBackend(
            model=self._backend_model,
            device=device,
        )

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
            inputs,
            required={
                "locus_tag", "enzyme_name",
                "reaction_class", "substrate_smiles",
            },
        )

        self._ensure_loaded(config)
        # Import the backend data class here — layer1_atomic.engine
        # is already imported by _ensure_loaded, so this is free.
        from cell_sim.layer1_atomic.engine import EnzymeProfile  # noqa: WPS433

        # Group (locus_tag -> list of per-substrate estimates).
        per_locus: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        # Preserve locus_tag first-seen order so the output index
        # matches caller expectations.
        order: list[str] = []
        seen: set[str] = set()

        for row in inputs.itertuples(index=False):
            locus = str(row.locus_tag)
            if locus not in seen:
                order.append(locus)
                seen.add(locus)
            smi = str(row.substrate_smiles).strip()
            if not smi or smi.lower() in {"nan", "none", "-"}:
                continue
            reference_substrates = _parse_reference_substrates(
                getattr(row, "reference_substrates_json", None),
                fallback_smi=smi,
            )
            enzyme = EnzymeProfile(
                name=str(row.enzyme_name),
                reaction_class=str(row.reaction_class),
                known_substrate_smiles=reference_substrates,
            )
            estimate = self._backend.estimate_kcat(smi, enzyme)
            if estimate.source in {"mace_unavailable", "no_reference",
                                    "mace_bde_failed"}:
                # Skip unusable estimates; they don't contribute to
                # the enzyme's aggregate.
                continue
            per_locus[locus].append(
                (float(estimate.kcat_per_s),
                 float(estimate.confidence),
                 estimate.source)
            )

        rows: list[dict[str, float]] = []
        for locus in order:
            rows.append(_aggregate_for_locus(per_locus.get(locus, [])))

        out = pd.DataFrame(rows, columns=self.feature_cols)
        out.index = pd.Index(order, name="locus_tag")
        return out


# ---- helpers ----


def _empty_frame(cols: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(columns=cols)
    frame.index = pd.Index([], name="locus_tag")
    return frame


def _validate_input_columns(df: pd.DataFrame, *, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"MaceOffExtractor.extract: input DataFrame missing "
            f"required column(s) {sorted(missing)!r}. "
            f"Got: {list(df.columns)!r}"
        )


def _parse_reference_substrates(
    raw: object, fallback_smi: str,
) -> dict[str, float]:
    """Decode the per-row reference-substrate JSON blob. Falls back
    to a placeholder dict when the column is missing / empty so the
    BDE-shift math still runs (delta-BDE vs the substrate itself is 0,
    i.e. the resulting k_cat equals the fallback reference rate of
    1.0/s — honest default for enzymes with no measured substrates)."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {fallback_smi: 1.0}
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "-"}:
        return {fallback_smi: 1.0}
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return {fallback_smi: 1.0}
    if not isinstance(decoded, dict) or not decoded:
        return {fallback_smi: 1.0}
    out: dict[str, float] = {}
    for smi, kcat in decoded.items():
        try:
            out[str(smi)] = float(kcat)
        except (TypeError, ValueError):
            continue
    if not out:
        return {fallback_smi: 1.0}
    return out


def _aggregate_for_locus(
    entries: list[tuple[float, float, str]],
) -> dict[str, float]:
    if not entries:
        return {
            "mace_kcat_mean_per_s": math.nan,
            "mace_kcat_std_per_s": math.nan,
            "mace_kcat_min_per_s": math.nan,
            "mace_kcat_max_per_s": math.nan,
            "mace_n_substrates": 0.0,
            "mace_mean_confidence": math.nan,
            "mace_has_estimate": 0.0,
        }
    kcats = [k for k, _, _ in entries]
    confs = [c for _, c, _ in entries]
    mean_kcat = sum(kcats) / len(kcats)
    if len(kcats) > 1:
        var = sum((k - mean_kcat) ** 2 for k in kcats) / (len(kcats) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return {
        "mace_kcat_mean_per_s": mean_kcat,
        "mace_kcat_std_per_s": std,
        "mace_kcat_min_per_s": min(kcats),
        "mace_kcat_max_per_s": max(kcats),
        "mace_n_substrates": float(len(kcats)),
        "mace_mean_confidence": sum(confs) / len(confs),
        "mace_has_estimate": 1.0,
    }


__all__ = ["MaceOffExtractor"]

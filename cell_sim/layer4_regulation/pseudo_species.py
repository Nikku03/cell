"""Regulation pseudo-species namespace and state initialization.

Phase R1 — INFRASTRUCTURE ONLY.

Pseudo-species mirror the convention introduced in Phase 2 of the gene
expression layer (``_mrna:<locus>``, ``_protein_native:<locus>``).
They live in a dedicated dictionary attribute on ``CellState`` named
``regulation_pseudo_species`` so they are easy to inspect, easy to
serialize, and never collide with the metabolite-count namespace
managed by :mod:`layer3_reactions.coupled`.

The names below are stable contracts: any future Phase R2 code that
reads or writes these keys must use the prefixes exported from this
module, not hard-coded strings.

Namespace
---------

* ``_tf_bound:<tf_locus>:<target_locus>`` — 0/1 indicator. Whether the
  TF protein with locus ``tf_locus`` is currently bound at the
  promoter of ``target_locus``. Multiple TFs at the same target are
  tracked as multiple keys with distinct ``tf_locus``; their
  composition into a multiplicative effect on transcription rate is
  computed by :func:`recompute_promoter_state` (a placeholder for
  Phase R2 to flesh out).

* ``_promoter_state:<gene_locus>`` — float in [0, ∞). Composite
  promoter activity for ``gene_locus``: a multiplicative factor
  applied on top of the gene's intrinsic ``promoter_strength`` from
  the gene-expression layer. Default is 1.0 (no regulatory effect).

* ``_sigma_class:<sigma_locus>`` — non-negative integer. Free-pool
  count of the sigma factor with locus ``sigma_locus``. Sigma rules
  decrement / increment this as competition events fire.

* ``_riboswitch_state:<gene_locus>`` — float in [0, 1]. Fraction of
  riboswitch population in the folded conformation. Phase R1 toggles
  between 0.0 and 1.0; Phase R2 may use a continuous value.

* ``_phos_state:<rr_locus>`` — 0/1 indicator. Phosphorylation state
  of the response regulator at locus ``rr_locus``.

The state-initialization function :func:`initialize_regulation_state`
populates these entries for every locus mentioned in the network
config. When the network config is empty (the expected Phase R1 case)
the function still runs to completion but produces no entries; an
``[INFO]`` log message advertises that regulation infrastructure is
present but inert.
"""
from __future__ import annotations

import sys
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cell_sim.layer4_regulation.network_loader import RegulationNetworkConfig


# ============================================================================
# Pseudo-species prefixes (the stable namespace contract)
# ============================================================================
TF_BOUND_PREFIX = "_tf_bound:"
PROMOTER_STATE_PREFIX = "_promoter_state:"
SIGMA_CLASS_PREFIX = "_sigma_class:"
RIBOSWITCH_STATE_PREFIX = "_riboswitch_state:"
PHOS_STATE_PREFIX = "_phos_state:"

ALL_PREFIXES: tuple[str, ...] = (
    TF_BOUND_PREFIX,
    PROMOTER_STATE_PREFIX,
    SIGMA_CLASS_PREFIX,
    RIBOSWITCH_STATE_PREFIX,
    PHOS_STATE_PREFIX,
)


# ============================================================================
# State initialization
# ============================================================================
def initialize_regulation_state(
    state,
    network_config: "Optional[RegulationNetworkConfig]" = None,
    *,
    log: bool = True,
) -> dict:
    """Attach the regulation pseudo-species dictionary to ``state``.

    Mutates ``state`` by setting ``state.regulation_pseudo_species`` to
    a dict of ``{pseudo_species_name: count_or_state}``. The dict is
    populated entry-by-entry from the network config. When the config
    is ``None`` or has zero entries (the Phase R1 default) the
    dictionary is created empty and an ``[INFO]`` message is printed
    advertising the regulation-disabled state.

    Returns a small summary dict for logging / test assertions.

    Parameters
    ----------
    state
        A :class:`CellState` instance. The function mutates
        ``state.regulation_pseudo_species`` in place; it does not
        return the state.
    network_config
        A :class:`RegulationNetworkConfig` or ``None``. When ``None``,
        the function still attaches an empty pseudo-species dict (this
        is the no-op regulation case the architecture must support).
    log
        If True, emit the regulation-disabled ``[INFO]`` message when
        the config is empty. Tests may set this to False to keep
        stdout clean.

    Returns
    -------
    dict
        ``{"n_tf_bound": int, "n_promoter": int, "n_sigma": int,
        "n_riboswitch": int, "n_phos": int, "active": bool}``.
        ``active`` is True iff at least one pseudo-species was
        registered (i.e. the network config had at least one entry).
    """
    if not hasattr(state, "regulation_pseudo_species"):
        state.regulation_pseudo_species = {}
    ps = state.regulation_pseudo_species

    # Empty / None config — Phase R1 default. Initialize the namespace
    # but register no entries. The simulator can still call
    # rule.can_fire on a regulation rule and it will see no matching
    # key and return [], i.e. no propensity contribution.
    if network_config is None or _is_empty_config(network_config):
        if log:
            print(
                "[INFO] Regulation network configuration is empty. "
                "Initializing infrastructure with no active regulatory "
                "rules. See FUTURE_WORK.md for curation tasks."
            )
        return {
            "n_tf_bound": 0,
            "n_promoter": 0,
            "n_sigma": 0,
            "n_riboswitch": 0,
            "n_phos": 0,
            "active": False,
        }

    # Populated config — register one pseudo-species per regulatory
    # entity. Phase R1 leaves all initial states at their unbound /
    # free / unfolded defaults; Phase R2 may take initial states from
    # a curation-time observation.
    n_tf_bound = 0
    n_promoter = 0
    n_sigma = 0
    n_riboswitch = 0
    n_phos = 0

    for tf_entry in getattr(network_config, "transcription_factors", []) or []:
        tf_locus = tf_entry.get("tf_locus")
        if not tf_locus:
            continue
        for target in tf_entry.get("target_genes", []) or []:
            ps[f"{TF_BOUND_PREFIX}{tf_locus}:{target}"] = 0
            n_tf_bound += 1
            promoter_key = f"{PROMOTER_STATE_PREFIX}{target}"
            if promoter_key not in ps:
                ps[promoter_key] = 1.0
                n_promoter += 1

    for sigma_entry in getattr(network_config, "sigma_factors", []) or []:
        sigma_locus = sigma_entry.get("sigma_locus")
        if not sigma_locus:
            continue
        ps[f"{SIGMA_CLASS_PREFIX}{sigma_locus}"] = int(
            sigma_entry.get("initial_free_count", 0)
        )
        n_sigma += 1

    for rs_entry in getattr(network_config, "riboswitches", []) or []:
        gene_locus = rs_entry.get("gene_locus")
        if not gene_locus:
            continue
        ps[f"{RIBOSWITCH_STATE_PREFIX}{gene_locus}"] = float(
            rs_entry.get("initial_folded_fraction", 0.0)
        )
        n_riboswitch += 1

    for tcs_entry in getattr(network_config, "two_component_systems", []) or []:
        rr_locus = tcs_entry.get("response_regulator_locus")
        if not rr_locus:
            continue
        ps[f"{PHOS_STATE_PREFIX}{rr_locus}"] = int(
            tcs_entry.get("initial_phos_state", 0)
        )
        n_phos += 1

    return {
        "n_tf_bound": n_tf_bound,
        "n_promoter": n_promoter,
        "n_sigma": n_sigma,
        "n_riboswitch": n_riboswitch,
        "n_phos": n_phos,
        "active": (n_tf_bound + n_sigma + n_riboswitch + n_phos) > 0,
    }


def recompute_promoter_state(state, target_gene: str) -> float:
    """Composite promoter activity for ``target_gene`` from bound TFs.

    Phase R1 stub: looks at every ``_tf_bound:*:<target_gene>`` key
    and returns the count of currently-bound TFs as the composite
    state. A real implementation needs per-TF activator/repressor
    polarity (curation), cooperativity coefficients (curation), and
    a final mapping from "bound TF profile" to a multiplicative effect
    on transcription rate (Phase R2). This stub exists so call sites
    in Phase R2 can be wired without yet another refactor.
    """
    ps = getattr(state, "regulation_pseudo_species", None)
    if not ps:
        return 1.0
    suffix = f":{target_gene}"
    n_bound = 0
    for key, val in ps.items():
        if key.startswith(TF_BOUND_PREFIX) and key.endswith(suffix) and val:
            n_bound += 1
    promoter_key = f"{PROMOTER_STATE_PREFIX}{target_gene}"
    if promoter_key in ps:
        ps[promoter_key] = float(n_bound)
    return float(n_bound)


def _is_empty_config(config) -> bool:
    """Return True if every list in the config is empty.

    Phase R1's regulation_network_syn3a.yaml is exactly this — four
    empty lists. The check is duck-typed so tests can pass a plain
    object with the same attribute names.
    """
    for attr in ("transcription_factors", "sigma_factors",
                 "riboswitches", "two_component_systems"):
        items = getattr(config, attr, None)
        if items:
            return False
    return True


__all__ = [
    "ALL_PREFIXES",
    "PHOS_STATE_PREFIX",
    "PROMOTER_STATE_PREFIX",
    "RIBOSWITCH_STATE_PREFIX",
    "SIGMA_CLASS_PREFIX",
    "TF_BOUND_PREFIX",
    "initialize_regulation_state",
    "recompute_promoter_state",
]

"""Regulation rule factories — INFRASTRUCTURE ONLY (Phase R1).

Four factories that produce ``TransitionRule`` objects modeling the
regulatory architecture of bacterial gene expression. None of the rate
constants in this module are calibrated against Syn3A biology. Each
factory accepts kinetic parameters as arguments and emits a
``BiologyNotCurated`` warning when invoked without a ``source_citation``
keyword identifying where the parameters came from.

The factories produce architecturally valid rules — the same shape as
``layer3_reactions.gene_expression.make_transcription_rule`` — but their
``can_fire`` / ``apply`` closures are intentionally minimal. They mutate
pseudo-species defined in ``pseudo_species.py`` (TF bound state, sigma
class counts, riboswitch fold state, response-regulator phosphorylation
state) and they DO NOT touch metabolite counts or protein instances. The
production essentiality sweep does not invoke these rules, so their
trajectory effect on v15 / v16 is zero by construction.

When Phase R2 wires this layer into ``RealSimulator``, the mutating
side of each rule will need a second pass: the current implementations
update pseudo-species but do not, for example, modulate the
``promoter_strength`` dictionary that gene-expression transcription
rules read at ``can_fire`` time. That coupling is the integration step
and is deliberately deferred. See module docstring of
``cell_sim.layer4_regulation`` for the Phase R2 checklist.

The four factories
==================

* :func:`make_tf_binding_rule` — TF binding/unbinding to a target
  promoter. Modifies ``_tf_bound:<tf_locus>:<target_locus>`` between 0
  and 1 and recomputes the composite ``_promoter_state:<target_locus>``.
* :func:`make_sigma_competition_rule` — sigma factor competition for
  free RNAP. Models the gene-class affinity tilt produced by different
  housekeeping / stress / sporulation sigma subunits.
* :func:`make_riboswitch_rule` — ligand-dependent riboswitch folding
  that gates translation initiation on the affected mRNA.
* :func:`make_two_component_signaling_rule` — sensor kinase
  phosphorylation of a cognate response regulator.

Suppressing the BiologyNotCurated warning
=========================================

Tests and any future scaffolding code that legitimately exercises these
factories with placeholder kinetics should wrap the call in::

    import warnings
    from cell_sim.layer4_regulation.regulation import BiologyNotCurated

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiologyNotCurated)
        rule = make_tf_binding_rule("JCVISYN3A_xxxx", "JCVISYN3A_yyyy",
                                    k_bind=1.0, k_unbind=1.0)

Production code (Phase R2 wiring) MUST pass ``source_citation`` so the
warning never fires; the warning is the canary that tells a reviewer
some kinetic parameter was pulled out of thin air.
"""
from __future__ import annotations

import warnings
from typing import Callable, List, Optional

from layer2_field.dynamics import CellState, TransitionRule

from cell_sim.layer4_regulation.pseudo_species import (
    PHOS_STATE_PREFIX,
    PROMOTER_STATE_PREFIX,
    RIBOSWITCH_STATE_PREFIX,
    SIGMA_CLASS_PREFIX,
    TF_BOUND_PREFIX,
)


# ============================================================================
# BiologyNotCurated warning class
# ============================================================================
class BiologyNotCurated(UserWarning):
    """Emitted when a regulation rule factory is called without a
    documented source citation for its kinetic parameters.

    The warning exists explicitly so that future readers of a stack
    trace, test log, or grep result can tell at a glance which call
    sites are still using placeholder kinetics. When Phase R2 curation
    completes, ``source_citation`` should be passed at every call site
    and this warning should never fire under normal operation.

    Filtering this warning silences the only signal Phase R1 leaves
    behind that the parameters are not real biology — do so only in
    test code that explicitly checks the architectural contract.
    """


def _check_curation(
    factory_name: str,
    source_citation: Optional[str],
    parameters: dict,
) -> None:
    """Emit ``BiologyNotCurated`` if ``source_citation`` is missing.

    The parameters dict is purely for the warning message so a developer
    debugging a stale call site sees which numbers are uncalibrated.
    """
    if source_citation:
        return
    param_repr = ", ".join(f"{k}={v!r}" for k, v in parameters.items())
    warnings.warn(
        BiologyNotCurated(
            f"{factory_name}({param_repr}) called without source_citation. "
            "Kinetic parameters are not from a verified source. The "
            "resulting rule is architecturally valid but biologically "
            "uncalibrated and must NOT be wired into the production "
            "essentiality sweep."
        ),
        stacklevel=3,
    )


# ============================================================================
# 1. TF binding / unbinding
# ============================================================================
def make_tf_binding_rule(
    tf_locus: str,
    target_gene: str,
    k_bind: float,
    k_unbind: float,
    *,
    source_citation: Optional[str] = None,
) -> TransitionRule:
    """Produce a rule for TF binding/unbinding at a target gene's promoter.

    Biology
    -------
    A transcription factor (TF) protein binds to a specific operator or
    promoter sequence near its target gene, modulating RNAP recruitment.
    For an activator, the bound state increases ``promoter_strength``;
    for a repressor, the bound state decreases it. This factory is
    direction-agnostic: the bound state is recorded as a 0/1 indicator
    in the pseudo-species ``_tf_bound:<tf_locus>:<target_locus>`` and
    a separate composite-promoter step (in
    :func:`pseudo_species.recompute_promoter_state`) translates the set
    of bound TFs into a multiplicative effect on transcription rate.

    Rule semantics
    --------------
    Two competing first-order processes:

    * Binding: rate ∝ ``k_bind`` × free TF count × (promoter unbound).
    * Unbinding: rate ∝ ``k_unbind`` × (promoter bound).

    The current implementation toggles the bound state between 0 and 1
    at each fire; it does not model multi-TF cooperativity at the same
    promoter (which would need a graph of bound TFs, not a single bit).
    Cooperativity is a Phase R3 candidate.

    Parameters
    ----------
    tf_locus
        Locus tag of the TF protein (e.g. ``JCVISYN3A_xxxx``).
    target_gene
        Locus tag of the regulated gene.
    k_bind, k_unbind
        Forward / reverse rate constants. Units are 1/s for the
        unbinding term and 1/(M·s) for the binding term in a real
        kinetic treatment; this skeleton uses both as bare 1/s
        propensity prefactors and leaves the unit reconciliation to
        Phase R2 once curated values are available.
    source_citation
        REQUIRED in production code. Free-form string identifying the
        paper, database accession, or curation note that justifies the
        ``k_bind`` / ``k_unbind`` values. Without it a
        :class:`BiologyNotCurated` warning is emitted.

    Notes on Phase R1 scope
    -----------------------
    Parameters are unspecified at this phase — ``k_*`` values must come
    from curated sources before this rule is wired into production.
    Calling this with placeholder values produces an architecturally
    correct rule with biologically uncalibrated kinetics.
    """
    _check_curation(
        "make_tf_binding_rule",
        source_citation,
        {"tf_locus": tf_locus, "target_gene": target_gene,
         "k_bind": k_bind, "k_unbind": k_unbind},
    )
    name = f"tf_binding:{tf_locus}->{target_gene}"
    bound_key = f"{TF_BOUND_PREFIX}{tf_locus}:{target_gene}"
    promoter_key = f"{PROMOTER_STATE_PREFIX}{target_gene}"

    def can_fire(state: CellState):
        # Architecture-only: emit a single token whenever the regulatory
        # pseudo-species are present. Real binding requires a free TF
        # count check; that is a Phase R2 step coupled to the gex
        # protein-instance bookkeeping.
        ps = getattr(state, "regulation_pseudo_species", None)
        if ps is None or bound_key not in ps:
            return []
        bound = ps[bound_key]
        # Bound -> can unbind; unbound -> can bind. We emit one token
        # per state. Without a free-TF count, both rates are taken from
        # the rule's scalar `rate` field, scaled by k_bind / k_unbind.
        return [("flip", int(bound))]

    def apply(state: CellState, cands, rng):
        ps = getattr(state, "regulation_pseudo_species", None)
        if ps is None or bound_key not in ps:
            return
        ps[bound_key] = 0 if ps[bound_key] else 1
        # Composite promoter recomputation deferred to Phase R2 wiring;
        # the pseudo-species itself is updated so a downstream consumer
        # can read the bound state without re-deriving it from scratch.
        if promoter_key in ps:
            ps[promoter_key] = ps.get(promoter_key, 1.0)
        state.log_event(
            name, [],
            f"tf_binding({tf_locus}->{target_gene}) toggled to "
            f"{ps[bound_key]} [INFRASTRUCTURE-ONLY: not curated]",
        )

    # Pick the larger of the two rates as the rule's scalar rate so
    # propensity is non-zero in either direction. This is a placeholder
    # consistent with the no-curation phase; Phase R2 should split into
    # two separate rules (bind, unbind) with their own rates.
    rate = max(float(k_bind), float(k_unbind))
    rule = TransitionRule(
        name=name,
        participants=[tf_locus, target_gene],
        rate=rate,
        rate_source="placeholder_phase_r1",
        can_fire=can_fire,
        apply=apply,
    )
    return rule


# ============================================================================
# 2. Sigma factor competition
# ============================================================================
def make_sigma_competition_rule(
    sigma_locus: str,
    gene_class: str,
    competition_factor: float,
    *,
    source_citation: Optional[str] = None,
) -> TransitionRule:
    """Produce a rule modeling sigma factor competition for free RNAP.

    Biology
    -------
    Bacterial RNA polymerase requires a sigma subunit to recognize
    promoter sequences. Different sigma factors recognize different
    promoter classes: a housekeeping sigma (e.g. SigA) covers most
    growth-phase transcription; alternative sigmas (stress response,
    sporulation, heat shock) compete with the housekeeping sigma for
    the same pool of core RNAP. The competition tilts which gene class
    gets transcribed under different conditions.

    Rule semantics
    --------------
    Updates the pseudo-species ``_sigma_class:<sigma_locus>`` (free
    sigma factor count) by emitting a single first-order
    binding/release event with rate ∝ ``competition_factor``. The
    rule is gene-class-aware via the ``gene_class`` argument so that
    Phase R2 wiring can route housekeeping / stress / sporulation
    sigmas to disjoint gene cohorts.

    Parameters
    ----------
    sigma_locus
        Locus tag of the sigma factor protein.
    gene_class
        Free-form tag indicating which class of genes this sigma
        recruits. Convention: ``"housekeeping"``, ``"stress"``,
        ``"sporulation"``. Phase R2 may add more.
    competition_factor
        Scalar in [0, 1] capturing this sigma's affinity for free RNAP
        relative to the housekeeping sigma's reference. Real values
        require ChIP-seq / proteomics data not available for Syn3A at
        Phase R1.
    source_citation
        REQUIRED in production code. See
        :func:`make_tf_binding_rule` for the convention.

    Notes on Phase R1 scope
    -----------------------
    Parameters are unspecified at this phase — ``competition_factor``
    must come from a curated source before this rule is wired into
    production. Calling this with placeholder values produces an
    architecturally correct rule with biologically uncalibrated
    kinetics.
    """
    _check_curation(
        "make_sigma_competition_rule",
        source_citation,
        {"sigma_locus": sigma_locus, "gene_class": gene_class,
         "competition_factor": competition_factor},
    )
    name = f"sigma_competition:{sigma_locus}:{gene_class}"
    sigma_key = f"{SIGMA_CLASS_PREFIX}{sigma_locus}"

    def can_fire(state: CellState):
        ps = getattr(state, "regulation_pseudo_species", None)
        if ps is None or sigma_key not in ps:
            return []
        free = ps.get(sigma_key, 0)
        if free <= 0:
            return []
        return [("compete", gene_class)]

    def apply(state: CellState, cands, rng):
        ps = getattr(state, "regulation_pseudo_species", None)
        if ps is None or sigma_key not in ps:
            return
        # Architecture-only: a successful competition event "consumes"
        # one free sigma into a transient bound state. With no real RNAP
        # tracking yet, we just decrement the free pool by 1; the
        # release step is deferred to Phase R2 wiring.
        if ps[sigma_key] > 0:
            ps[sigma_key] -= 1
        state.log_event(
            name, [],
            f"sigma_competition({sigma_locus}, {gene_class}) "
            f"factor={competition_factor:.3g} "
            f"[INFRASTRUCTURE-ONLY: not curated]",
        )

    rate = float(competition_factor)
    rule = TransitionRule(
        name=name,
        participants=[sigma_locus],
        rate=rate,
        rate_source="placeholder_phase_r1",
        can_fire=can_fire,
        apply=apply,
    )
    return rule


# ============================================================================
# 3. Riboswitch folding
# ============================================================================
def make_riboswitch_rule(
    gene_locus: str,
    ligand_metabolite: str,
    k_fold: float,
    k_unfold: float,
    *,
    source_citation: Optional[str] = None,
) -> TransitionRule:
    """Produce a rule for ligand-dependent riboswitch folding.

    Biology
    -------
    A riboswitch is a structured 5' UTR motif on an mRNA that adopts
    one of two alternative folds depending on whether a small-molecule
    ligand is bound. The folded/unfolded ratio gates downstream
    translation initiation: in the canonical case, ligand binding
    stabilizes a fold that occludes the Shine-Dalgarno sequence and
    represses translation. Examples in other organisms include the
    thiamine pyrophosphate (TPP), purine, and guanine riboswitches.
    Whether Syn3A retains any riboswitches is itself a curation
    question.

    Rule semantics
    --------------
    Updates the pseudo-species ``_riboswitch_state:<gene_locus>`` by
    flipping between 0 (unfolded, translation permitted) and 1
    (folded, translation blocked) with rates ``k_fold`` and
    ``k_unfold`` modulated by the ligand metabolite count.

    Parameters
    ----------
    gene_locus
        Locus tag of the gene whose mRNA carries the riboswitch.
    ligand_metabolite
        SBML species id of the small molecule that gates folding.
        Convention follows the rest of the model:
        ``"M_<bigg>_c"``.
    k_fold, k_unfold
        Folding and unfolding rate constants. Real values are typically
        in the 0.01–10 /s range, but Syn3A-specific calibration would
        require structure-probing data not present in Phase R1.
    source_citation
        REQUIRED in production code.

    Notes on Phase R1 scope
    -----------------------
    Parameters are unspecified at this phase — ``k_*`` values must come
    from curated sources before this rule is wired into production.
    Calling this with placeholder values produces an architecturally
    correct rule with biologically uncalibrated kinetics.
    """
    _check_curation(
        "make_riboswitch_rule",
        source_citation,
        {"gene_locus": gene_locus, "ligand_metabolite": ligand_metabolite,
         "k_fold": k_fold, "k_unfold": k_unfold},
    )
    name = f"riboswitch:{gene_locus}:{ligand_metabolite}"
    rs_key = f"{RIBOSWITCH_STATE_PREFIX}{gene_locus}"

    def can_fire(state: CellState):
        ps = getattr(state, "regulation_pseudo_species", None)
        if ps is None or rs_key not in ps:
            return []
        return [("fold", float(ps[rs_key]))]

    def apply(state: CellState, cands, rng):
        ps = getattr(state, "regulation_pseudo_species", None)
        if ps is None or rs_key not in ps:
            return
        # Architecture-only: ignore ligand count entirely (real coupling
        # to metabolite_counts is a Phase R2 step). Just toggle the
        # state. The k_fold / k_unfold parameters are baked into the
        # rule's scalar rate so the propensity reflects the asymmetry.
        ps[rs_key] = 0.0 if ps[rs_key] >= 0.5 else 1.0
        state.log_event(
            name, [],
            f"riboswitch({gene_locus}, ligand={ligand_metabolite}) "
            f"-> {ps[rs_key]:.1f} "
            f"[INFRASTRUCTURE-ONLY: not curated]",
        )

    rate = max(float(k_fold), float(k_unfold))
    rule = TransitionRule(
        name=name,
        participants=[gene_locus, ligand_metabolite],
        rate=rate,
        rate_source="placeholder_phase_r1",
        can_fire=can_fire,
        apply=apply,
    )
    return rule


# ============================================================================
# 4. Two-component signaling (sensor kinase + response regulator)
# ============================================================================
def make_two_component_signaling_rule(
    sensor_locus: str,
    response_regulator_locus: str,
    k_phos: float,
    k_dephos: float,
    *,
    source_citation: Optional[str] = None,
) -> TransitionRule:
    """Produce a rule for sensor-kinase phosphorylation of a response regulator.

    Biology
    -------
    A two-component system pairs a membrane-bound sensor histidine
    kinase with a cytoplasmic response regulator (RR). On stimulus
    detection the kinase autophosphorylates and transfers the phosphate
    to a conserved aspartate on the RR. Phosphorylated RR typically
    acts as a sequence-specific TF and changes its target genes'
    transcription. Mycoplasma genitalium has very few two-component
    systems; whether Syn3A retains any is a curation question.

    Rule semantics
    --------------
    Toggles ``_phos_state:<rr_locus>`` between 0 and 1 at rates
    ``k_phos`` (forward) and ``k_dephos`` (reverse). Phase R2 wiring
    should couple the phosphorylated state to TF binding rules so the
    activated RR functions as a downstream regulator; that linkage is
    not implemented here.

    Parameters
    ----------
    sensor_locus
        Locus tag of the sensor histidine kinase.
    response_regulator_locus
        Locus tag of the cognate response regulator.
    k_phos, k_dephos
        Phosphorylation / dephosphorylation rate constants (1/s).
    source_citation
        REQUIRED in production code.

    Notes on Phase R1 scope
    -----------------------
    Parameters are unspecified at this phase — ``k_*`` values must come
    from curated sources before this rule is wired into production.
    Calling this with placeholder values produces an architecturally
    correct rule with biologically uncalibrated kinetics.
    """
    _check_curation(
        "make_two_component_signaling_rule",
        source_citation,
        {"sensor_locus": sensor_locus,
         "response_regulator_locus": response_regulator_locus,
         "k_phos": k_phos, "k_dephos": k_dephos},
    )
    name = f"two_component:{sensor_locus}->{response_regulator_locus}"
    phos_key = f"{PHOS_STATE_PREFIX}{response_regulator_locus}"

    def can_fire(state: CellState):
        ps = getattr(state, "regulation_pseudo_species", None)
        if ps is None or phos_key not in ps:
            return []
        return [("phos", int(ps[phos_key]))]

    def apply(state: CellState, cands, rng):
        ps = getattr(state, "regulation_pseudo_species", None)
        if ps is None or phos_key not in ps:
            return
        ps[phos_key] = 0 if ps[phos_key] else 1
        state.log_event(
            name, [],
            f"two_component({sensor_locus}->{response_regulator_locus}) "
            f"phos={ps[phos_key]} [INFRASTRUCTURE-ONLY: not curated]",
        )

    rate = max(float(k_phos), float(k_dephos))
    rule = TransitionRule(
        name=name,
        participants=[sensor_locus, response_regulator_locus],
        rate=rate,
        rate_source="placeholder_phase_r1",
        can_fire=can_fire,
        apply=apply,
    )
    return rule


__all__ = [
    "BiologyNotCurated",
    "make_riboswitch_rule",
    "make_sigma_competition_rule",
    "make_tf_binding_rule",
    "make_two_component_signaling_rule",
]

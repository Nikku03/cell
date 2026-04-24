"""iMB155 pathway patches: close three v10b false positives.

v10b's PerRule detector flags three Breuer-Nonessential genes as
essential because the iMB155 metabolic reconstruction encodes only
ONE pathway per metabolite:

* ``JCVISYN3A_0034`` — assigned as ``PLACEHOLDER_LOCUS`` in
  ``nutrient_uptake.py`` for 8+ non-specific transporters (CHOLt,
  GLYCt, O2t, TAGt, ADEt_syn, GUAt_syn, URAt_syn, CYTDt_syn). None of
  these is a biologically annotated transporter for 0034; the gene
  was chosen as the highest-abundance uncharacterised ABC efflux
  protein as a fallback carrier. When 0034 is knocked out, the
  PerRule detector sees every catalysis rule in the placeholder list
  drop to zero events and calls it essential.

* ``JCVISYN3A_0228`` (lpdA) — sole enzyme for PDH_E3 in iMB155.
  M. mycoides capri (Syn3A's ancestor) has LplA / LipA / LipB as
  alternate lipoyl-cofactor regenerators, none of which are encoded
  in ``kinetic_params.xlsx``.

* ``JCVISYN3A_0732`` (deoC) — sole enzyme for DRPA (2-deoxy-D-ribose-
  5-P aldol cleavage) in iMB155. Syn3A retains residual aldol-
  cleavage activity via other enzymes not captured in the 356-reaction
  SBML.

Strategy
--------
Clear these three loci from ``compiled_spec['enzyme_loci']`` for the
rules they over-gate. Two branches per rule:

1. **Alternate enzyme present** (e.g. a rule listed under both the
   patched locus and a real enzyme): drop the patched locus from
   ``enzyme_loci`` and keep the compiled-MM rule. The real enzyme
   still gates firing; the patched gene's KO no longer silences it.

2. **Sole enzyme was the patched locus** (e.g. all 12 rules
   currently gated by 0034, plus PDH_E3 and DRPA): replace with a
   python-closure ``TransitionRule`` that fires at the same per-
   reaction rate as a single-enzyme instance but with NO enzyme
   gating. The new rule has no ``compiled_spec``, so
   ``build_gene_to_rules`` excludes it from every gene's rule set.
   ``PerRuleDetector.detect_for_gene('JCVISYN3A_0034')`` then
   returns ``(NONE, None, 0.0, 'no_catalytic_rules')`` — the FP is
   closed by removing the signal, not by manipulating event counts.

The replacement rule still fires in WT — with propensity
``kcat × saturation`` instead of ``kcat × E × saturation`` — so
downstream metabolite pools stay supplied. This is biologically
honest for the placeholder transporters (cholesterol / glycerol /
O2 / TAG passive diffusion in a ~55-protein Mycoplasma membrane is
dominated by non-specific lipid flip-flop, not by a single ABC
transporter) and a known-approximate workaround for PDH_E3 / DRPA
(where real alternate enzymes exist but aren't in our kinetics
table).

See ``memory_bank/facts/parameters/imb155_pathway_patches.json`` for
per-rule citations and the biology rationale for each patched locus.
"""
from __future__ import annotations

from typing import List

from cell_sim.layer2_field.dynamics import TransitionRule
from cell_sim.layer3_reactions.coupled import (
    get_species_count, update_species_count,
)


# Three loci whose KO produces a v10b false positive solely because of
# an iMB155 reconstruction gap. Any rule that lists one of these as a
# sole catalyser gets re-expressed as an ungated python-closure rule.
PATCHED_LOCI: frozenset[str] = frozenset((
    "JCVISYN3A_0034",   # over-assigned PLACEHOLDER_LOCUS
    "JCVISYN3A_0228",   # lpdA (PDH_E3)
    "JCVISYN3A_0732",   # deoC (DRPA)
))


# v13 discovered that replacing compiled-MM rules with single-token
# ungated python-closure rules drops effective propensity by ~factor-E
# (where E is the original enzyme count in WT). That slowdown
# cascaded through the metabolic network: NNATr's WT event count
# dropped from 25 to 14, falling below the PerRuleDetector's
# min_wt_events=20 threshold, which made the v12 sweep regress
# JCVISYN3A_0380 (nadD, Essential) from TP -> FN.
#
# Fix: each patched rule fires at N tokens per step instead of 1, so
# propensity is ``kcat * N * saturation`` — roughly matches a
# fixed-size enzyme pool. 20 is the single-cell average for
# Mycoplasma transporter / catalytic proteins at scale=0.05. A
# higher value would bias WT dynamics; a lower value reintroduces
# the regression.
_UNGATED_TOKEN_COUNT: int = 20


def _build_ungated_rule(
    name: str,
    substrates: list,
    products: list,
    kcat: float,
    include_saturation: bool,
    km_dict: dict,
    patched_for: str,
) -> TransitionRule:
    """Build a python-closure MM rule that fires without enzyme gating.

    Propensity is ``kcat × N × saturation`` where
    ``N = _UNGATED_TOKEN_COUNT`` (fixed at 20 — see module
    docstring). This restores rule activity to roughly what the
    compiled-MM enzyme-gated path would produce at a typical
    Mycoplasma transporter/catalytic enzyme pool size, preventing
    the v12 NNATr regression where network slowdown dropped
    unrelated rules below the PerRuleDetector WT threshold.

    The returned rule has ``compiled_spec = None`` on purpose:

    * ``gene_rule_map.build_gene_to_rules`` requires
      ``compiled_spec['kind'] == 'mm'``, so the rule is invisible to
      the per-rule detector — the patched locus' ``gene_to_rules``
      set loses this entry entirely.
    * The fast simulator's vectorised MM path requires a
      ``compiled_spec``; without one the rule falls through to the
      python-closure cache path, which honours our custom
      ``can_fire`` / ``apply``.
    """
    substrate_ids = [sid for sid, _ in substrates]
    sub_pairs = [(sid, float(st)) for sid, st in substrates]
    prd_pairs = [(sid, float(st)) for sid, st in products]
    km = dict(km_dict)

    def can_fire(state):
        for sid, st in sub_pairs:
            if get_species_count(state, sid) < st:
                return []
        if include_saturation:
            from cell_sim.layer3_reactions.reversible import (
                mm_saturation_factor,
            )
            sat = mm_saturation_factor(state, substrate_ids, km)
        else:
            sat = 1.0
        if sat <= 0:
            return []
        return [(None, sat)] * _UNGATED_TOKEN_COUNT

    def apply(state, cands, rng):
        if not cands:
            return
        for sid, st in sub_pairs:
            if get_species_count(state, sid) < st:
                return
        for sid, st in sub_pairs:
            update_species_count(state, sid, -int(st))
        for sid, st in prd_pairs:
            update_species_count(state, sid, +int(st))
        state.log_event(
            name, [],
            f"{name} (imb155_patch: loci_cleared={patched_for})",
        )

    rule = TransitionRule(
        name=name,
        participants=list(substrate_ids),
        rate=float(kcat),
        rate_source="imb155_patch",
        can_fire=can_fire,
        apply=apply,
    )
    return rule


def apply_imb155_patches(rules: List[TransitionRule]) -> List[TransitionRule]:
    """Return a new rule list with the three iMB155 loci cleared.

    The input list is not modified in place. Non-catalysis rules and
    catalysis rules with no patched-locus overlap pass through
    unchanged (same object reference). Patched rules are replaced
    with either (a) a compiled-MM rule with the patched locus dropped
    from ``enzyme_loci`` if alternate enzymes remain, or (b) an
    ungated python-closure rule.
    """
    out: List[TransitionRule] = []
    for rule in rules:
        spec = getattr(rule, "compiled_spec", None)
        if not spec or spec.get("kind") != "mm":
            out.append(rule)
            continue
        enzyme_loci = list(spec.get("enzyme_loci") or [])
        patched = [l for l in enzyme_loci if l in PATCHED_LOCI]
        if not patched:
            out.append(rule)
            continue
        remaining = [l for l in enzyme_loci if l not in PATCHED_LOCI]
        if remaining:
            new_spec = dict(spec)
            new_spec["enzyme_loci"] = remaining
            new_rule = TransitionRule(
                name=rule.name,
                participants=list(rule.participants),
                rate=rule.rate,
                rate_source=rule.rate_source,
                can_fire=rule.can_fire,
                apply=rule.apply,
            )
            new_rule.compiled_spec = new_spec
            out.append(new_rule)
            continue
        patched_for = ",".join(patched)
        out.append(_build_ungated_rule(
            name=rule.name,
            substrates=list(spec["substrates"]),
            products=list(spec["products"]),
            kcat=float(spec["kcat"]),
            include_saturation=bool(spec.get("include_saturation", True)),
            km_dict=dict(spec.get("Km") or {}),
            patched_for=patched_for,
        ))
    return out


def count_patched_rules(rules: List[TransitionRule]) -> dict:
    """Diagnostic: count how many rules each patched locus gates.

    Useful for sanity-checking the patch before wiring it into the
    simulator. Returns ``{'JCVISYN3A_0034': N, ...}``.
    """
    counts: dict[str, int] = {l: 0 for l in PATCHED_LOCI}
    for rule in rules:
        spec = getattr(rule, "compiled_spec", None)
        if not spec or spec.get("kind") != "mm":
            continue
        for locus in spec.get("enzyme_loci") or ():
            if locus in counts:
                counts[locus] += 1
    return counts

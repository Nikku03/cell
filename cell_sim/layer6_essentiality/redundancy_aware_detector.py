"""RedundancyAwareDetector — per-gene detector that checks *total
metabolite production*, not just per-rule event silencing.

Why this exists
---------------
``PerRuleDetector`` (Session 7) trips whenever a gene's catalysis rules
go silent in a KO. That is a direct causal signal — but it false-
positives on Breuer-Nonessential catalytic genes (v5: lpdA, deoC,
0034) because the simulator contains no biological pathway redundancy,
while Breuer's essentiality labels implicitly account for it.

The fix that actually works: check whether any metabolite produced by
the silenced rules still gets its total production rate maintained by
alternate catalysis rules. If yes → Breuer-style redundancy applies
→ abstain. If some product's total production collapses → trip.

This shifts the detection question from "did the gene's rules fire?"
to "did the system's capacity to produce the gene's products survive?"
which is semantically closer to the FBA logic Breuer's MCC target
embodies.

Honest scope
------------
The detector takes a ``metabolite_producers`` map at construction time
(from :func:`gene_rule_map.build_metabolite_producers`). The map lists
all catalysis rules that produce each metabolite in the simulator's
rule set. If the simulator is missing biological pathways that exist
in a real cell (incomplete iMB155 reconstruction), redundancy checks
can still be wrong — but that's a simulator-completeness bug, not a
detector bug.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cell_sim.layer6_essentiality.harness import (
    FailureMode, Trajectory,
)


@dataclass
class RedundancyAwareDetector:
    """Per-gene detector that trips only when a silenced gene's
    products actually lose production capacity in the KO.

    Parameters
    ----------
    wt : Trajectory
        Wild-type trajectory with ``event_counts_by_rule`` populated.
    gene_to_rules : dict[str, set[str]]
        From :func:`gene_rule_map.build_gene_to_rules`.
    metabolite_producers : dict[str, list[(rule_name, stoich)]]
        From :func:`gene_rule_map.build_metabolite_producers`.
    rule_products : dict[str, list[(metabolite_id, stoich)]]
        From :func:`gene_rule_map.build_rule_products`. Tells the
        detector which metabolites a silenced rule would have produced.
    min_wt_production : int
        A metabolite must have at least this many total production
        events (summed across all producer rules, weighted by
        stoichiometry) in WT before the detector considers its
        production "meaningful". Prevents trivial-zero-vs-zero
        positives on metabolites the simulator barely uses.
    drop_threshold : float
        Fractional drop in production needed to trip. 0.30 means
        "production in KO fell below 30 %% of WT".
    """

    wt: Trajectory
    gene_to_rules: dict[str, set[str]]
    metabolite_producers: dict[str, list]
    rule_products: dict[str, list]
    min_wt_production: int = 20
    drop_threshold: float = 0.30

    def detect_for_gene(
        self,
        locus_tag: str,
        ko: Trajectory,
    ) -> tuple[FailureMode, float | None, float, str]:
        """Return ``(mode, time_s, confidence, evidence)``.

        - ``NONE`` ``"no_catalytic_rules"`` if the gene has no rules in
          ``gene_to_rules``.
        - ``NONE`` ``"alternates_compensate"`` if all affected products
          maintain ``>= drop_threshold`` of their WT production in KO.
        - ``CATALYSIS_SILENCED`` if at least one affected product drops
          below ``drop_threshold`` in KO, with confidence =
          ``1 - ko/wt production ratio`` on the worst-hit product.
        - ``NONE`` ``"wt_under_threshold"`` if every affected product
          had ``< min_wt_production`` in WT (the test is uninformative).
        """
        rules = self.gene_to_rules.get(locus_tag) or set()
        if not rules:
            return FailureMode.NONE, None, 0.0, "no_catalytic_rules"

        wt_counts: dict[str, int] = (
            self.wt.samples[-1].event_counts_by_rule or {}
        )
        ko_counts: dict[str, int] = (
            ko.samples[-1].event_counts_by_rule or {}
        )
        t_end = self.wt.samples[-1].t_s

        # Aggregate affected products.
        affected: set[str] = set()
        for rn in rules:
            for p, _stoich in self.rule_products.get(rn) or ():
                affected.add(p)
        if not affected:
            return FailureMode.NONE, None, 0.0, "rules_have_no_products"

        worst_ratio = 1.0
        worst_evidence = ""
        any_meaningful = False
        for product in sorted(affected):
            producers = self.metabolite_producers.get(product) or []
            wt_prod = 0.0
            ko_prod = 0.0
            for (rn, stoich) in producers:
                wt_prod += float(wt_counts.get(rn, 0)) * stoich
                ko_prod += float(ko_counts.get(rn, 0)) * stoich
            if wt_prod < self.min_wt_production:
                continue
            any_meaningful = True
            ratio = ko_prod / wt_prod if wt_prod > 0 else 1.0
            if ratio < worst_ratio:
                worst_ratio = ratio
                worst_evidence = (
                    f"{product}: wt={wt_prod:.0f} ko={ko_prod:.0f} "
                    f"ratio={ratio:.3f}"
                )
        if not any_meaningful:
            return FailureMode.NONE, None, 0.0, "wt_under_threshold"

        if worst_ratio < self.drop_threshold:
            confidence = max(0.0, min(1.0, 1.0 - worst_ratio))
            return (
                FailureMode.CATALYSIS_SILENCED,
                t_end,
                confidence,
                f"production_collapse[{worst_evidence}]",
            )
        return FailureMode.NONE, None, 0.0, (
            f"alternates_compensate[worst={worst_evidence}]"
        )

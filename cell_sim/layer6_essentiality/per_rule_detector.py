"""Per-rule event-count detector.

Compares per-rule event counts in the KO trajectory against WT. A
knockout is flagged ``CATALYSIS_SILENCED`` iff **every** rule in
``gene_to_rules[locus_tag]`` has ``>= min_wt_events`` events in WT
but 0 events in KO.

Honest scope
------------
This detector is tautologically true for direct catalytic KOs: remove
the enzyme, its rule stops firing. The detector's **value** is that it
converts this direct causal fact into a uniform prediction across all
genes with a measured k_cat, without per-pool threshold tuning. It is
*not* magic and will not touch genes that don't appear in any
``catalysis:*`` rule (ribosomal proteins, tRNA synthetases, replication
genes) - those return ``NONE``. That is the intended behaviour; the
architectural fix for those classes is Path A (longer bio-time runs)
documented in ``memory_bank/concepts/essentiality/REPORT.md``, not this
detector.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cell_sim.layer6_essentiality.harness import (
    FailureMode, Trajectory,
)


@dataclass
class PerRuleDetector:
    """Gene-specific detector. Call :meth:`detect_for_gene` with the
    locus_tag being knocked out and the KO trajectory."""

    wt: Trajectory
    gene_to_rules: dict[str, set[str]]
    # Minimum event count in WT required for a rule to count as "should
    # be firing". Prevents false positives on rules that simply don't
    # activate in WT over the short window (e.g. reverse reactions that
    # haven't accumulated product yet).
    min_wt_events: int = 20

    def detect_for_gene(
        self,
        locus_tag: str,
        ko: Trajectory,
    ) -> tuple[FailureMode, float | None, float, str]:
        """Return ``(mode, time_s, confidence, evidence)``.

        - If the gene has no ``catalysis:*`` rules, returns
          ``(NONE, None, 0.0, "no_catalytic_rules")`` - no signal.
        - If any rule in the gene's set had < ``min_wt_events`` events
          in WT, returns ``(NONE, None, 0.0, "wt_under_threshold")`` -
          the detector refuses to call it because WT itself didn't
          exercise the rule enough to make a confident zero.
        - Otherwise if every rule dropped from >=``min_wt_events`` to 0
          in KO, returns ``(CATALYSIS_SILENCED, t_end, 1.0,
          "silenced: <rules>")``.
        - Otherwise ``(NONE, None, 0.0, "partial_silence: <rules>")``
          - the KO didn't fully silence the rule set (e.g. the gene is
          one of several enzymes catalysing the reaction).
        """
        rules = self.gene_to_rules.get(locus_tag)
        if not rules:
            return FailureMode.NONE, None, 0.0, "no_catalytic_rules"

        wt_last = self.wt.samples[-1].event_counts_by_rule or {}
        ko_last = ko.samples[-1].event_counts_by_rule or {}

        silenced: list[str] = []
        partial: list[str] = []
        under_thr: list[str] = []
        for rule_name in sorted(rules):
            wt_n = int(wt_last.get(rule_name, 0))
            ko_n = int(ko_last.get(rule_name, 0))
            if wt_n < self.min_wt_events:
                under_thr.append(f"{rule_name}[wt={wt_n}]")
                continue
            if ko_n == 0:
                silenced.append(f"{rule_name}[wt={wt_n}]")
            else:
                partial.append(f"{rule_name}[wt={wt_n},ko={ko_n}]")

        # Refuse to call if any checked rule was under the WT threshold
        # - safer to abstain than to count a zero-vs-zero as positive.
        if under_thr and not silenced and not partial:
            return (FailureMode.NONE, None, 0.0,
                    f"wt_under_threshold: {','.join(under_thr[:3])}")

        if partial:
            return (FailureMode.NONE, None, 0.0,
                    f"partial_silence: {','.join(partial[:3])}")

        if silenced and not partial:
            t_end = self.wt.samples[-1].t_s
            evidence = f"silenced: {','.join(silenced[:3])}"
            if len(silenced) > 3:
                evidence += f" (+{len(silenced)-3} more)"
            return FailureMode.CATALYSIS_SILENCED, t_end, 1.0, evidence

        return FailureMode.NONE, None, 0.0, ""

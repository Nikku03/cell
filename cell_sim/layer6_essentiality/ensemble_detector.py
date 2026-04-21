"""EnsembleDetector: combines PerRule + ShortWindow signals.

Session 7 diagnosed the two single-detector walls: `ShortWindowDetector`
catches only pgi-class central-glycolysis KOs; `PerRuleDetector`
catches the metabolic-enzyme subset but false-positives on
Breuer-nonessential catalytic genes because the simulator lacks the
pathway redundancy Breuer's labels account for.

The ensemble's hypothesis is that **true essential KOs perturb pools
visibly AND silence their catalysis rules**, while the Breuer-
nonessential catalytic FPs silence rules but don't meaningfully
perturb pools (the surviving alternate paths keep metabolism running
in a real cell - but in this simulator they don't exist, so pools do
often still drift...).

This is a measurement, not a proof. Ship and see.

Policy options:

``AND``
  Trip only if BOTH detectors fire. Highest precision. Expected to
  drop all v5 FPs; probably also drops some v5 TPs whose pool signal
  is weak (plsX-area, 0813, 0729). Predicted recall collapse -> likely
  lower MCC than v5, not higher. Worth measuring anyway.

``OR_HIGH_CONFIDENCE`` (default)
  Trip if either detector fires with ``confidence >= min_confidence``.
  ``PerRuleDetector`` always reports confidence 1.0 when it trips;
  ``ShortWindowDetector`` confidence is ``1 - min_ratio``.
  With ``min_confidence = 0.15`` the policy is effectively
  "PerRule always fires; ShortWindow only fires when the signal is
  large" - not obviously better than PerRule alone.

``PER_RULE_WITH_POOL_CONFIRM``
  Trip if PerRule fires AND the KO shows ANY watched pool deviation
  above a loose floor (``min_pool_dev = 0.02``). This is the policy
  most aligned with the diagnostic hypothesis: require that the
  knocked-out enzyme's absence actually perturb the simulator's
  metabolic state in addition to silencing the rule.

The ensemble reports the stricter failure mode when both fire
(``CATALYSIS_SILENCED`` if the per-rule signal is present), falling
back to the ShortWindow mode otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from cell_sim.layer6_essentiality.harness import (
    FailureMode, Trajectory,
)
from cell_sim.layer6_essentiality.per_rule_detector import PerRuleDetector
from cell_sim.layer6_essentiality.short_window_detector import (
    ShortWindowDetector,
)


class EnsemblePolicy(str, Enum):
    AND = "and"
    OR_HIGH_CONFIDENCE = "or_high_confidence"
    PER_RULE_WITH_POOL_CONFIRM = "per_rule_with_pool_confirm"


@dataclass
class EnsembleDetector:
    per_rule: PerRuleDetector
    short_window: ShortWindowDetector
    policy: EnsemblePolicy = EnsemblePolicy.PER_RULE_WITH_POOL_CONFIRM
    min_confidence: float = 0.15
    min_pool_dev: float = 0.02

    def detect_for_gene(
        self,
        locus_tag: str,
        ko: Trajectory,
    ) -> tuple[FailureMode, float | None, float, str]:
        pr_mode, pr_t, pr_conf, pr_ev = self.per_rule.detect_for_gene(
            locus_tag, ko,
        )
        sw_mode, sw_t, sw_conf, sw_ev = self.short_window.detect(ko)

        pr_fires = pr_mode != FailureMode.NONE
        sw_fires = sw_mode != FailureMode.NONE

        if self.policy is EnsemblePolicy.AND:
            if pr_fires and sw_fires:
                mode = pr_mode  # PerRule mode is the stricter label
                return mode, pr_t or sw_t, max(pr_conf, sw_conf), (
                    f"AND[pr:{pr_ev} | sw:{sw_ev}]"
                )
            return FailureMode.NONE, None, 0.0, (
                f"AND_no_agreement[pr:{pr_ev} | sw:{sw_ev}]"
            )

        if self.policy is EnsemblePolicy.OR_HIGH_CONFIDENCE:
            if pr_fires and pr_conf >= self.min_confidence:
                return pr_mode, pr_t, pr_conf, f"OR:pr[{pr_ev}]"
            if sw_fires and sw_conf >= self.min_confidence:
                return sw_mode, sw_t, sw_conf, f"OR:sw[{sw_ev}]"
            return FailureMode.NONE, None, 0.0, (
                f"OR_below_confidence[pr:{pr_conf:.2f},sw:{sw_conf:.2f}]"
            )

        # PER_RULE_WITH_POOL_CONFIRM
        if not pr_fires:
            return FailureMode.NONE, None, 0.0, f"pr_abstain[{pr_ev}]"
        # PerRule fires. Require any watched pool to deviate above
        # min_pool_dev at any sample.
        max_dev = _max_pool_deviation(self.short_window, ko)
        if max_dev >= self.min_pool_dev:
            return pr_mode, pr_t, min(1.0, pr_conf), (
                f"pr+pool_confirm[{pr_ev}; max_pool_dev={max_dev:.3f}]"
            )
        return FailureMode.NONE, None, 0.0, (
            f"pr_no_pool_confirm[{pr_ev}; max_pool_dev={max_dev:.3f}]"
        )


def _max_pool_deviation(sw: ShortWindowDetector, ko: Trajectory) -> float:
    """Max |ko/wt - 1| across the watched pool set at any time point.
    Used only for PER_RULE_WITH_POOL_CONFIRM - NOT a detection criterion
    of its own; purely a tie-break."""
    wt_samples = sw.wt.samples
    n = min(len(wt_samples), len(ko.samples))
    best = 0.0
    for i in range(n):
        ws = wt_samples[i].pools
        ks = ko.samples[i].pools
        for pool in sw.pools:
            w = ws.get(pool)
            k = ks.get(pool)
            if w is None or k is None or w <= 0:
                continue
            dev = abs((k - w) / w)
            if dev > best:
                best = dev
    return best

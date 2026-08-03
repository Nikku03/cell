"""ROBUSTNESS HARNESS: a contrast is not a result until it has been swept over the choices nobody deliberated.

WHY THIS EXISTS.  On 3 Aug a PPI block was reported as beating chan2a by +0.0140 / +0.0152, having survived three
separate attacks on its control -- degree preservation exact, 98.2% of edges rewired, three seeds stable, leakage
ratio 1.05x.  Every one of those checks passed and every one was necessary.  Then a sweep over the SVD rank showed
the effect existed in ONE of six (edge-set x rank) cells: the exact configuration that had been run first.
`SVD_K = 128` was a line written without deliberation, and it was carrying the entire finding.

The lesson generalises past that one bug.  Control checks all ask **"is the null valid?"**  None of them asks
**"is the effect robust to defaults I chose without thinking?"**  Those are different questions and only the
second catches a result that is really the argmax of an unswept space.  Three other corrections the same day had
the same root -- a stale verdict string, a twin count read off the wrong channel matrix, a hardcoded NPRED label
-- all of them values trusted rather than re-derived.

USAGE.  Accumulate paired per-item deltas for every (config, cohort) cell, then ask for the verdict:

    sw = Sweeper("chan2a+ppi - chan2a+ppi_rw", axes={"rank": [64, 128, 256]})
    for cfg in sw.configs():
        ...build arms at cfg["rank"]...
        for cohort, deltas in (("cohort1", d1), ("cohort2", d2)):
            sw.add(cfg, cohort, deltas)
    print(sw.report())
    verdict = sw.verdict()["verdict"]

VERDICT LADDER, fixed here so it cannot be renegotiated once the numbers are in:

    ROBUST        resolved in >= 2/3 of cells, all point estimates the same sign.  A result.
    DIRECTIONAL   sign-consistent across every cell but resolved in fewer than 2/3.  Something is there and it
                  is too weak or too noisy to bank -- the honest verdict for the PPI block.
    FRAGILE       resolved in at least one cell but the signs disagree, or resolved in exactly one cell out of
                  four or more.  This is what the argmax of an unswept space looks like and it must not be
                  reported as a finding.
    NULL          resolved nowhere.
    HARMFUL       resolved NEGATIVE in >= 2/3 of cells.

`Sweeper.add` refuses a config that is not in the declared grid, and `verdict()` raises on a single-cell sweep,
so "I only ran the default" cannot reach a report by accident.
"""
import itertools
import math

import numpy as np

BOOT = 10_000


class SingleConfigError(RuntimeError):
    """Raised when a verdict is requested from a sweep that never swept anything."""


class Sweeper:
    def __init__(self, name, axes, boot=BOOT, seed=0):
        if not axes or any(len(v) < 2 for v in axes.values()):
            raise SingleConfigError(
                f"{name}: every axis needs >= 2 values -- a sweep over one value is the bug this class exists "
                f"to prevent. Got {({k: len(v) for k, v in axes.items()})}.")
        self.name, self.axes, self.boot, self.seed = name, dict(axes), boot, seed
        self._grid = [dict(zip(self.axes, combo)) for combo in itertools.product(*self.axes.values())]
        self._cells = {}

    def configs(self):
        return list(self._grid)

    @staticmethod
    def _key(cfg):
        return tuple(sorted((k, str(v)) for k, v in cfg.items()))

    def add(self, cfg, cohort, deltas):
        """deltas: paired per-item differences (arm A minus arm B) for one cohort at one config."""
        if self._key(cfg) not in {self._key(c) for c in self._grid}:
            raise ValueError(f"{self.name}: config {cfg} is not in the declared grid {self.axes}")
        d = np.asarray(deltas, float)
        rng = np.random.default_rng(self.seed)
        bs = d[rng.integers(0, len(d), size=(self.boot, len(d)))].mean(1)
        self._cells[(self._key(cfg), cohort)] = {
            "config": dict(cfg), "cohort": cohort, "n": int(len(d)), "delta": float(d.mean()),
            "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}

    def verdict(self):
        cells = list(self._cells.values())
        if len(cells) < 2:
            raise SingleConfigError(
                f"{self.name}: {len(cells)} cell(s) accumulated. A verdict from a single configuration is "
                f"exactly the failure this harness prevents -- sweep the grid or do not report a verdict.")
        pos = [c for c in cells if c["lo"] > 0]
        neg = [c for c in cells if c["hi"] < 0]
        signs = {int(np.sign(c["delta"])) for c in cells if c["delta"] != 0}
        consistent = len(signs) <= 1
        need = math.ceil(2 * len(cells) / 3)
        if len(neg) >= need:
            v = "HARMFUL"
        elif len(pos) >= need and consistent:
            v = "ROBUST"
        elif len(pos) == 0 and len(neg) == 0:
            v = "NULL"
        elif consistent and len(pos) > 0:
            v = "DIRECTIONAL"
        else:
            v = "FRAGILE"
        if len(pos) == 1 and len(cells) >= 4 and v not in ("HARMFUL",):
            v = "FRAGILE"
        return {"name": self.name, "verdict": v, "n_cells": len(cells), "n_resolved_pos": len(pos),
                "n_resolved_neg": len(neg), "sign_consistent": bool(consistent),
                "cells_needed_for_robust": need,
                "passing_configs": [c["config"] for c in pos], "cells": cells}

    def report(self):
        r = self.verdict()
        ax = ",".join(self.axes)
        out = [f"\n  SWEEP {self.name}   axes[{ax}]  {r['n_cells']} cells"]
        out.append(f"  {'config':<28} {'cohort':<10} {'delta':>9} {'95% CI':>22}")
        for c in r["cells"]:
            cfg = " ".join(f"{k}={v}" for k, v in c["config"].items())
            mark = "RES+" if c["lo"] > 0 else ("RES-" if c["hi"] < 0 else "ns")
            out.append(f"  {cfg:<28} {c['cohort']:<10} {c['delta']:>+9.4f} "
                       f"[{c['lo']:>+8.4f},{c['hi']:>+8.4f}] {mark}")
        out.append(f"  -> {r['verdict']}   resolved +{r['n_resolved_pos']}/-{r['n_resolved_neg']} "
                   f"of {r['n_cells']} (>= {r['cells_needed_for_robust']} needed), "
                   f"sign-consistent {r['sign_consistent']}")
        if r["verdict"] == "FRAGILE":
            out.append("     FRAGILE means the effect lives in a corner of the grid -- the argmax of choices "
                       "nobody deliberated. Do not report it as a finding.")
        if r["verdict"] == "DIRECTIONAL":
            out.append("     DIRECTIONAL means every cell agrees on the sign but too few resolve. Real but "
                       "not bankable; say so rather than quoting the best cell.")
        return "\n".join(out)

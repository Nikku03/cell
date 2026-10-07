"""AUDITING THE SWAPPED-SOURCE CONTROL THAT TWO SHIPPED MODULES DEPEND ON.

WHY THIS EXISTS. `colab/impute_reg_sign.py` set out to impute the sign of 558,005 unsigned curated edges,
came back null, and on the way audited its own control and found it defective:

    matching the swapped source on perturbation-strength DECILE left a standardised mean difference of
    0.25 -- curated sources are TFs and sit at the TOP of whatever decile they land in, so the control
    was systematically WEAKER than the arm it controls. Tightening it moved that module's K562 increment
    -0.0074 (decile) -> +0.0144 (nearest-neighbour) -> -0.0137 (joint strength x library size).

The control's construction moved the answer by as much as the effect was worth. Two modules already in this
branch use the same decile-matched swapped-source control:

    colab/causal_reg.py       the +15.0-point sign-transfer headline (67.2% vs 52.2% swapped)
    colab/reliable_edges.py   committed today: incumbent increment +0.1520, power-weighted +0.1805

Their increments are an order of magnitude larger than the shift that decided impute_reg_sign, so the
expectation is that they survive. THAT IS AN EXPECTATION, NOT A MEASUREMENT, and a control defect found in
one module is not allowed to stay unmeasured in two others because the numbers there look comfortable.

WHAT IS MEASURED HERE.

  1  THE IMBALANCE ITSELF. For the incumbent |z| >= 5 edge set, the standardised mean difference in RPE1
     perturbation strength between the edges' own source perturbations and the decile-matched replacements.
     SMD, not a rank test: a rank test on the bin index is 1.0 by construction and would look like a passed
     check. |SMD| < 0.1 is the conventional "balanced"; the defect found upstream was 0.25.

  2  HOW MUCH THE ANSWER MOVES WHEN THE CONTROL IS TIGHTENED. The same increment recomputed under three
     constructions of the swapped source:
        DECILE       the incumbent: replacement drawn from the same strength decile
        NEAREST      replacement drawn from the k nearest perturbations in strength, which removes the
                     within-decile gradient that the decile arm leaves
        JOINT        nearest in strength AND matched on the replacement's own cell count, because
                     reliable_edges established today that RPE1 cell count is itself worth 39% of a gain
     If the increment is stable across all three, the headline stands and now stands measured. If it moves,
     the strictest arm governs, exactly as it did upstream.

  3  THE SAME AUDIT ON THE POWER-WEIGHTED RULE, because that rule was ADMITTED today on the strength of an
     increment measured with the decile control, and it selects perturbations with more cells -- which is
     precisely the axis the joint arm tightens.

WHAT THIS CANNOT DO. It audits the SOURCE side of the swap. The gene (target) side is held fixed by
construction in this control and is not at issue. And it does not re-audit the decile-matched MAGNITUDE
control used by the fast test elsewhere in the project; that is a different comparison with a different
matching and would need its own pass.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
N_SEEDS = 5
NEAR_K = 25             # neighbourhood size for the nearest-in-strength arm
SMD_BAR = 0.10          # conventional balance threshold


def smd(a, b):
    """Standardised mean difference. The statistic the upstream audit used, and the reason it works:
    it reads the RAW covariate, so a matching that is perfect on the bin index but skewed inside the bin
    still shows up."""
    sa, sb = np.std(a), np.std(b)
    p = np.sqrt(0.5 * (sa ** 2 + sb ** 2))
    return float((np.mean(a) - np.mean(b)) / max(p, 1e-9))


def main():
    import reliable_edges as RE
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("AUDIT: does the decile-matched swapped-source control hold up in causal_reg / reliable_edges?")
    report("=" * 100)
    report("  TRIGGER: impute_reg_sign found |SMD| 0.25 on perturbation strength inside its decile-matched")
    report("  swapped source, and tightening the control moved its increment by more than the effect.")

    X, psym, kg, ncell, gmean = RE.load_k562(report)
    Zk, kmed, kmad = RE.robust_z(X)
    Lr, rperts, rg, rn = RE.pseudobulk_rpe1(report)
    Zr, _m, _d = RE.robust_z(Lr)
    rgi, kgi_ = RE.declared_join(list(rg), list(kg), 0.80, "RPE1 genes -> K562 genes", report)
    rpi, kpi_ = RE.declared_join(list(rperts), list(psym), 0.30, "RPE1 perturbations -> K562", report)
    ZK = Zk[np.ix_(kpi_, kgi_)]
    ZR = Zr[np.ix_(rpi, rgi)]
    NC = ncell[kpi_]
    RN = rn[rpi].astype(np.float64)
    ok = np.isfinite(ZK) & np.isfinite(ZR)
    report(f"  shared grid {ZK.shape[0]:,} x {ZK.shape[1]:,}; finite in both {ok.sum():,}")

    strength = np.nanmean(np.abs(ZR), axis=1)          # RPE1 perturbation strength, the matched covariate
    q = np.nanquantile(strength, np.linspace(.1, .9, 9))
    rq = np.digitize(strength, q)
    order = np.argsort(strength)
    rank_of = np.empty(len(strength), int)
    rank_of[order] = np.arange(len(strength))

    def swap_alt(ei, seed, mode):
        r = np.random.default_rng(seed)
        alt = np.empty(len(ei), int)
        if mode == "decile":
            for k in np.unique(rq):
                m = rq[ei] == k
                if m.any():
                    alt[m] = r.choice(np.where(rq == k)[0], int(m.sum()))
        else:
            n = len(strength)
            for i, e in enumerate(ei):
                lo = max(0, rank_of[e] - NEAR_K)
                hi = min(n, rank_of[e] + NEAR_K + 1)
                pool = order[lo:hi]
                pool = pool[pool != e]
                if mode == "joint" and len(pool) > 3:
                    # among the nearest in strength, prefer the closest in RPE1 cell count too
                    d = np.abs(np.log1p(RN[pool]) - np.log1p(RN[e]))
                    pool = pool[np.argsort(d)[:max(3, len(pool) // 3)]]
                alt[i] = int(r.choice(pool)) if len(pool) else int(e)
        return alt

    def arm(mask, label):
        ei, ej = np.where(mask)
        agree = float((np.sign(ZK[ei, ej]) == np.sign(ZR[ei, ej])).mean())
        out = {"label": label, "n_edges": int(len(ei)), "agree": agree}
        report(f"\n  {label}: {len(ei):,} edges, raw sign agreement {agree:.4f}")
        for mode in ("decile", "nearest", "joint"):
            incs, smds, smdc = [], [], []
            for s in range(N_SEEDS):
                alt = swap_alt(ei, 100 + s, mode)
                zr = ZR[alt, ej]
                f = np.isfinite(zr)
                a = float((np.sign(ZK[ei, ej]) == np.sign(zr))[f].mean())
                incs.append(agree - a)
                smds.append(smd(strength[ei], strength[alt]))
                smdc.append(smd(np.log1p(RN[ei]), np.log1p(RN[alt])))
            m, sd = float(np.mean(incs)), float(np.std(incs))
            sm, sc = float(np.mean(smds)), float(np.mean(smdc))
            flag = "" if abs(sm) < SMD_BAR else f"  <-- |SMD| {abs(sm):.2f} EXCEEDS the {SMD_BAR:.2f} bar"
            report(f"    {mode:8s} increment {m:+.4f} (sd {sd:.4f})   "
                   f"SMD strength {sm:+.3f}  SMD log cells {sc:+.3f}{flag}")
            out[mode] = {"increment": m, "sd": sd, "smd_strength": sm, "smd_log_cells": sc}
        spread = max(out[k]["increment"] for k in ("decile", "nearest", "joint")) - \
            min(out[k]["increment"] for k in ("decile", "nearest", "joint"))
        out["spread_across_constructions"] = float(spread)
        out["fraction_of_headline_moved"] = float(spread / max(abs(out["decile"]["increment"]), 1e-9))
        report(f"    -> across the three constructions the increment spans {spread:.4f}, which is "
               f"{100*out['fraction_of_headline_moved']:.0f}% of the decile-matched value")
        return out

    inc_mask = ok & (np.abs(ZK) >= 5.0)
    N = int(inc_mask.sum())
    a_inc = arm(inc_mask, "INCUMBENT |z| >= 5 (the +15.0 headline's edge rule)")
    pw = np.abs(ZK) * np.sqrt(NC[:, None] / (NC[:, None] + np.nanmedian(NC)))
    a_pw = arm(RE.top_n_mask(pw, ok, N), "POWER-WEIGHTED z (admitted today on the decile control)")

    # ---- verdict ----
    worst = max(abs(a_inc[m]["smd_strength"]) for m in ("decile", "nearest", "joint"))
    dec_bad = abs(a_inc["decile"]["smd_strength"]) >= SMD_BAR
    survives = (min(a_inc[m]["increment"] for m in ("decile", "nearest", "joint")) > 0.05
                and min(a_pw[m]["increment"] for m in ("decile", "nearest", "joint")) > 0.05)
    verdict = (
        f"{'BOTH HEADLINES SURVIVE THE TIGHTER CONTROL' if survives else 'A HEADLINE MOVES WHEN THE CONTROL IS TIGHTENED'}. "
        f"The decile-matched swapped source leaves |SMD| {abs(a_inc['decile']['smd_strength']):.3f} on RPE1 "
        f"perturbation strength for the incumbent |z| >= 5 edge set"
        + (f", which EXCEEDS the {SMD_BAR:.2f} balance bar and confirms the defect impute_reg_sign found "
           f"upstream. " if dec_bad else
           f", inside the {SMD_BAR:.2f} balance bar, so the defect impute_reg_sign found on curated TF "
           f"sources does NOT reproduce on this edge set -- the two differ because curated sources are all "
           f"TFs and sit at the top of their decile, while a |z| >= 5 edge set spans the whole strength "
           f"range. ")
        + f"Recomputed under three constructions of the control, the incumbent increment reads "
          f"{a_inc['decile']['increment']:+.4f} (decile), {a_inc['nearest']['increment']:+.4f} (nearest in "
          f"strength), {a_inc['joint']['increment']:+.4f} (nearest + cell-count matched) -- a span of "
          f"{a_inc['spread_across_constructions']:.4f}, or "
          f"{100*a_inc['fraction_of_headline_moved']:.0f}% of the decile value. The power-weighted rule "
          f"reads {a_pw['decile']['increment']:+.4f} / {a_pw['nearest']['increment']:+.4f} / "
          f"{a_pw['joint']['increment']:+.4f}, a span of {a_pw['spread_across_constructions']:.4f}. "
        + (f"The ordering between the two rules is preserved under every construction, so today's admission "
           f"of the power-weighted rule does not depend on which control was used."
           if all(a_pw[m]["increment"] > a_inc[m]["increment"] for m in ("decile", "nearest", "joint"))
           else
           f"THE ORDERING BETWEEN THE TWO RULES IS NOT PRESERVED across constructions, which means today's "
           f"admission of the power-weighted rule DID depend on the choice of control and must be revised."))
    report("\n" + "=" * 100)
    report(f"  VERDICT: {verdict}")

    R = {"model": "audit-matched-control-v1",
         "trigger": "colab/impute_reg_sign.py found |SMD| 0.25 in its decile-matched swapped source and "
                    "the tightening moved its increment by more than the effect was worth",
         "audits": ["colab/causal_reg.py (+15.0 sign transfer)", "colab/reliable_edges.py (committed today)"],
         "smd_bar": SMD_BAR, "n_seeds": N_SEEDS, "near_k": NEAR_K,
         "incumbent": a_inc, "power_weighted": a_pw,
         "decile_control_imbalanced": bool(dec_bad), "worst_abs_smd": float(worst),
         "both_survive": bool(survives),
         "limits": [
             "this audits the SOURCE side of the swap only. The target gene is held fixed by construction "
             "in this control and is not at issue",
             "the nearest-in-strength arm uses a fixed neighbourhood of 25 ranks. A tighter neighbourhood "
             "balances better and shrinks the pool, and no attempt was made to tune that trade-off",
             "the joint arm matches RPE1 cell count among the nearest-in-strength candidates rather than "
             "jointly optimising both, so its balance on cell count is better than the nearest arm's but "
             "is not guaranteed to be under the bar",
             "it does not re-audit the decile-matched MAGNITUDE control used by the project's fast test, "
             "which is a different comparison with a different matching and needs its own pass",
             "5 seeds per construction. The seed spread is reported and the increments here are large "
             "relative to it, but 5 is few for a spread estimate"],
         "verdict": verdict, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "audit_matched_control.json", "w"), indent=1)
    report(f"\n  -> {OUT/'audit_matched_control.json'}")


if __name__ == "__main__":
    main()

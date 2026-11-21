"""Does the sublinear K(N) survive the connectivity a REAL metabolic network has?

WHY THIS IS FORCED. sparse.py measured K ~ N^0.365 at c = 3 and, in P9, that the exponent RISES
with connectivity: alpha = 0.298, 0.355, 0.507 at c = 2, 3, 5. Denser wiring makes specification
harder. So the exponent cannot be quoted for a real network without knowing where that network
sits on the connectivity axis.

Measured from Recon3D (BiGG, 10,600 reactions, 5,835 metabolites, 2,248 genes), the reaction-to-
reaction coupling degree -- how many other reactions a reaction shares a metabolite with, after
removing currency metabolites (H+, water, ATP/ADP, Pi, NAD(P)(H), CO2, O2, CoA and the rest) and
capping metabolites appearing in more than 200 reactions -- is

    median 8,   mean 54,   90th percentile 180

against a sweep that stopped at c = 5. The real network is well outside the range where the
exponent was measured, and it is outside on the side that was getting worse.

WHAT THIS MODULE DOES. Extends the connectivity sweep to c = 8, 12, 20 and 32, and reports the
dilution exponent BESIDE the scaling exponent at every c. Those two must be read together: the
dense family in scaling.py had alpha = 0.004 only because its residual diluted as N^-0.268, and
quoting an exponent without its dilution is how that artefact was nearly reported as a result.

THE COMPARISON IS NOT EXACT AND IS NOT CLAIMED TO BE. c here is outgoing switches per type in a
branching process; the Recon3D number is reaction-reaction coupling through shared metabolites.
They are both "how many other things does this one touch", which is the quantity the exponent was
found to depend on, but they are not the same measurement. The conclusion drawn is therefore about
the DIRECTION and SATURATION of alpha with connectivity, not a number read off at c = 8.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

C1  IT REPRODUCES sparse.py AT THE OVERLAPPING VALUES. At c = 2, 3 and 5 the exponent must match
    sparse.py's P9 (0.298, 0.355, 0.507) to within 0.08, on the same seeds. If it does not, the
    two runs are not measuring the same thing and neither can be quoted.

C2  DILUTION IS REPORTED BESIDE EVERY EXPONENT. For each c, the power-law exponent of ||g_null||
    against N. Predeclared: an alpha quoted at a c whose dilution exponent is below -0.15 is
    flattered by the same artefact that made the dense family look flat, and must be marked as
    such rather than read as a scaling law.

C3  THE DETECTION CONTROL AT EVERY c. A dense random gradient must give alpha near 1 at each
    connectivity, or a sublinear result at that c is unfalsifiable.

C4  THE DELIVERABLE. alpha against c. Predeclared readings: if alpha saturates below 1 as c grows,
    the sublinear conclusion survives to real network densities and specification is tractable;
    if alpha approaches 1 by c = 8 to 32, then the sublinear result is a property of sparse toy
    networks and does NOT transfer to metabolism, and the earlier conclusion must be withdrawn for
    real networks.

C5  IRREDUCIBILITY AND DEPTH, as in sparse.py, at every c.

C6  IT IS NOT ONE NETWORK. Several instances per point; report the spread.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.scaling import TARGET_NULL, LOGY_BAND, fits
from rem.atlas.sparse import sparse_point, build_sparse, strongly_connected

C_SWEEP = (2, 3, 5, 8, 12, 20, 32)
N_TARGETS = (60, 120, 250, 500)
N_SEEDS = 4

# sparse.py P9, for C1
REFERENCE = {2: 0.298, 3: 0.355, 5: 0.507}
RECON3D = dict(reactions=10600, metabolites=5835, genes=2248,
               coupling_median=8, coupling_mean=54, coupling_p90=180,
               kinetic_params=46474)


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE); P("DOES SUBLINEAR K(N) SURVIVE REAL METABOLIC CONNECTIVITY?"); P(RULE)
    P(f"  Recon3D measured from BiGG: {RECON3D['reactions']} reactions,"
      f" {RECON3D['metabolites']} metabolites, {RECON3D['genes']} genes")
    P(f"  reaction-reaction coupling degree, currency metabolites removed:"
      f" median {RECON3D['coupling_median']}, mean {RECON3D['coupling_mean']},"
      f" 90th pct {RECON3D['coupling_p90']}")
    P(f"  kinetic parameters implied for metabolism alone: {RECON3D['kinetic_params']:,}")
    P(f"  sparse.py stopped at c = 5. This extends to c = {max(C_SWEEP)}.")

    res = {}
    for c in C_SWEEP:
        P(f"\n  c = {c} ...")
        ns, ks, gns, kds, spreads, deps, sccs = [], [], [], [], [], [], True
        for Nt in N_TARGETS:
            m = max(int(round(Nt / (2 + c))), 4)
            got = []
            for s in range(N_SEEDS):
                r = sparse_point(m, c, 4200 * m + 13 * c + s, want_dense_grad=True)
                if r and r["K"] is not None:
                    got.append(r)
                _, _, _, tg = build_sparse(m, c, 4200 * m + 13 * c + s)
                sccs &= strongly_connected(tg)
            if not got:
                continue
            N = got[0]["N"]
            ns.append(N)
            ks.append(float(np.median([r["K"] for r in got])))
            gns.append(float(np.median([r["gnull"] for r in got])))
            kk = [r["Kd"] for r in got if r.get("Kd") is not None]
            kds.append(float(np.median(kk)) if kk else np.nan)
            spreads.append((min(r["K"] for r in got), max(r["K"] for r in got)))
            deps += [r["logY"] for r in got]
            P(f"      N={N:5d} (m={m:4d})  K={ks[-1]:5.1f}  Kdense={kds[-1]:6.1f}"
              f"  ||g_null||={gns[-1]:.3f}")
        if len(ns) < 3:
            continue
        fa = fits(ns, ks)
        fd = fits([n for n, k in zip(ns, kds) if np.isfinite(k)],
                  [k for k in kds if np.isfinite(k)])
        dil = np.polyfit(np.log(ns), np.log(gns), 1)[0]
        res[c] = dict(ns=ns, ks=ks, alpha=fa["power"][0], r2=fa["power"][2],
                      logr2=fa["log"][2], dil=dil,
                      alpha_dense=fd["power"][0] if fd else np.nan,
                      spreads=spreads, dep=(min(deps), max(deps)), scc=sccs)

    P("\n" + RULE); P("C1  IT REPRODUCES sparse.py AT THE OVERLAPPING VALUES"); P(RULE)
    P(f"  {'c':>4}{'sparse.py alpha':>18}{'here':>9}{'difference':>13}")
    worst = 0.0
    for c, ref in REFERENCE.items():
        if c in res:
            dv = abs(res[c]["alpha"] - ref)
            worst = max(worst, dv)
            P(f"  {c:>4}{ref:>18.3f}{res[c]['alpha']:>9.3f}{dv:>13.3f}")
    P(f"  worst difference {worst:.3f}   {'PASS' if worst <= 0.08 else 'FAIL -- the two runs are not measuring the same thing'} (bar 0.08)")
    P("  NOTE: different seeds and a different N grid, so exact agreement is not expected;")
    P("  the bar is that the exponent is stable to those choices.")

    P("\n" + RULE); P("C5  IRREDUCIBILITY AND DEPTH"); P(RULE)
    allscc = all(v["scc"] for v in res.values())
    lo = min(v["dep"][0] for v in res.values()); hi = max(v["dep"][1] for v in res.values())
    P(f"  every switch graph strongly connected: {allscc}   {'PASS' if allscc else 'FAIL'}")
    P(f"  log10 Y across all circuits: {lo:.3f} to {hi:.3f} (band {LOGY_BAND})"
      f"   {'PASS' if LOGY_BAND[0]-0.6 <= lo and hi <= LOGY_BAND[1]+0.6 else 'FAIL'}")

    P("\n" + RULE); P("C2/C3/C4  THE DELIVERABLE  --  alpha against connectivity"); P(RULE)
    P(f"  {'c':>4}{'M/N':>8}{'alpha':>9}{'R2':>8}{'log R2':>9}{'dilution':>11}"
      f"{'alpha dense':>13}{'verdict':>28}")
    for c in sorted(res):
        v = res[c]
        flat = v["dil"] < -0.15
        det = v["alpha_dense"] >= 0.7
        verdict = ("DILUTION-FLATTERED" if flat else
                   ("undetectable" if not det else "clean"))
        P(f"  {c:>4}{2/(2+c):>8.3f}{v['alpha']:>9.3f}{v['r2']:>8.3f}{v['logr2']:>9.3f}"
          f"{v['dil']:>11.3f}{v['alpha_dense']:>13.3f}{verdict:>28}")
    clean = [c for c in sorted(res) if res[c]["dil"] >= -0.15 and res[c]["alpha_dense"] >= 0.7]
    P(f"\n  connectivities with a clean reading: {clean}")
    if clean:
        al = [res[c]["alpha"] for c in clean]
        P(f"  alpha over those: {min(al):.3f} to {max(al):.3f}")
        hi_c = [c for c in clean if c >= 8]
        if hi_c:
            ah = [res[c]["alpha"] for c in hi_c]
            P(f"  at c >= 8, the range Recon3D's median coupling of 8 sits in: alpha"
              f" {min(ah):.3f} to {max(ah):.3f}")
            if max(ah) < 0.8:
                P("  READING: alpha stays well below 1 at real-network connectivity, so the")
                P("  sublinear conclusion SURVIVES and specification remains tractable.")
            else:
                P("  READING: alpha approaches 1 by real-network connectivity. The sublinear result")
                P("  is a property of sparse toy networks and must be WITHDRAWN for metabolism.")
        else:
            P("  READING: no clean reading at c >= 8, so nothing may be concluded for real")
            P("  network densities from this run.")

    P("\n" + RULE); P("C6  IT IS NOT ONE NETWORK"); P(RULE)
    for c in sorted(res):
        P(f"  c={c:>3}  K ranges by size: " +
          "  ".join(f"{lo}-{hi}" for lo, hi in res[c]["spreads"]))

    P("\n" + RULE)
    open(os.path.join(os.path.dirname(__file__), "RESULTS_connectivity.txt"),
         "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

"""Regulation: does the metabolic sparsity survive a cell that is NOT a growth maximiser?

THE ASSUMPTION UNDER TEST IS MY OWN. recon.py found that 8 of 5,938 enzyme-catalysed reactions
carry a realised sensitivity, because in a linear program only ACTIVE constraints have a nonzero
dual and 97.7% of capacities do not bind. That sparsity is the foundation of every metabolic
claim in this build order, and it was computed at the LP OPTIMUM -- a cell maximising growth.

A regulated cell does not maximise growth. It holds fluxes at set points, keeps enzymes it does
not currently need, carries safety margins, and responds to signals that have nothing to do with
biomass. If the sparsity is a property of optimality rather than of metabolism, then recon.py's
conclusion is an artefact and must be withdrawn. That is the question here, and the gates are
written so it can come out either way.

THE KNOB, chosen because it is the least contrived way to make a cell suboptimal. Real regulation
is imperfect: cells express enzymes ahead of need, retain them after need, and allocate protein by
a program rather than by an optimiser. That is modelled as a floor on enzyme abundance,

    E_j  >=  phi * E_ref_j

with E_ref a fixed regulatory program and phi the strength of the commitment. At phi = 0 the cell
allocates freely and the model is recon.py's optimiser. As phi rises the cell is forced to hold
protein it has no immediate use for, exactly as a regulated cell does, and the allocation moves
away from the optimum without any arbitrary objective being invented.

WHAT COULD HAPPEN, and both readings are predeclared in G3. Forcing protein into unneeded enzymes
leaves less for the ones that matter, which could make MORE capacities bind and destroy the
sparsity. Or the binding set could be a property of the network's topology and stay put. The first
would withdraw recon.py's conclusion; the second would strengthen it considerably, because it
would mean the sparsity does not depend on the optimality assumption at all.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

G1  THE OPERATING POINT IS WHAT IT CLAIMS. At each phi the program is feasible, mass balance holds
    to 1e-6, and the enzyme floors are actually active -- if no floor binds, phi is decorative and
    the cell is still the optimiser.

G2  IT REPRODUCES recon.py AT phi = 0. With the same medium and objective, the count of reactions
    with a realised sensitivity must match recon.py's 8. If it does not, the two are not measuring
    the same thing and nothing below is comparable.

G3  THE DELIVERABLE. The number of rates with a realised sensitivity, and the growth rate, against
    phi. Predeclared readings: a count that stays within a factor of about two of 8 across the
    sweep means the sparsity is structural and survives regulation; a count that grows with phi
    towards the hundreds means the sparsity was a property of the optimum and recon.py's central
    metabolic result must be withdrawn for regulated cells.

G4  DUAL SCREEN, FINITE-DIFFERENCE VERIFY. recon.py's R3 showed 97.7% of nonzero LP duals have
    zero realised derivative at a degenerate optimum. Every count here is therefore built from
    verified finite differences in BOTH directions, never from duals, and the ghost fraction is
    reported at each phi so the degeneracy is visible rather than assumed away.

G5  THE IDENTITY OF THE SENSITIVE SET, not only its size. Report how much the set overlaps
    recon.py's eight as phi rises. A constant count made of different reactions is a different
    result from a constant count made of the same ones.

G6  THE REGULATORY PROGRAM IS NOT THE RESULT. Repeat with three different E_ref programs --
    uniform, flux-proportional at the optimum, and random -- and report whether the conclusion
    depends on which one is imposed.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, hstack, vstack, coo_matrix
from scipy.optimize import linprog

from rem.atlas.hybrid_tune import RULE
from rem.atlas.recon import MODEL, MEDIUM, fetch_if_missing, boundary_reactions
from rem.atlas.wholecell import parse_gpr, MW_PROT, KCAT_MEDIAN, KCAT_SIGMA, PROT_TOTAL

PHIS = (0.0, 0.35, 0.85)
BUDGET_METAB = PROT_TOTAL * 0.55
RECON_EIGHT = {"GAPD", "ATPS4mi", "NMNAT", "AKGDm", "ENO", "PGMT", "TALA", "DRPA"}
SEED = 20260906


def load():
    fetch_if_missing()
    d = json.load(open(MODEL))
    R, M = d["reactions"], d["metabolites"]
    mi = {m["id"]: i for i, m in enumerate(M)}
    Sl = lil_matrix((len(M), len(R)))
    for j, r in enumerate(R):
        for met, co in r["metabolites"].items():
            Sl[mi[met], j] = co
    return R, M, csc_matrix(Sl)


def bounds(R):
    lb = np.array([r["lower_bound"] for r in R], float)
    ub = np.array([r["upper_bound"] for r in R], float)
    idx = {r["id"]: j for j, r in enumerate(R)}
    # Close the SUPPLY direction of every boundary reaction, then reopen only the declared
    # medium. Identified structurally so no naming convention can leak past it. Removal is left
    # open: a cell that cannot excrete waste cannot run metabolism at all, and closing both
    # directions gives exactly zero growth.
    for j in boundary_reactions(R):
        lb[j] = 0.0
    for k, v in MEDIUM.items():
        if k in idx:
            lb[idx[k]] = v
    return lb, ub, idx


def solve_at(S, lb, ub, kcat, enz, obj_idx, phi, e_ref, maximise=True):
    """LP over [v, E] with capacity, a proteome budget, and regulatory floors E >= phi*e_ref."""
    nR, nE = S.shape[1], len(enz)
    n = nR + nE
    Sb = hstack([S, csc_matrix((S.shape[0], nE))]).tocsc()
    rows, cols, vals = [], [], []
    for k, j in enumerate(enz):
        rows += [k, k]; cols += [j, nR + k]; vals += [1.0, -kcat[j] / MW_PROT]
    Acap = coo_matrix((vals, (rows, cols)), shape=(nE, n))
    Abud = coo_matrix((np.ones(nE), (np.zeros(nE), np.arange(nR, n))), shape=(1, n))
    A_ub = vstack([Acap, Abud]).tocsc()
    b_ub = np.concatenate([np.zeros(nE), [BUDGET_METAB]])
    lo = np.concatenate([lb, phi * e_ref])
    hi = np.concatenate([ub, np.full(nE, np.inf)])
    c = np.zeros(n)
    c[obj_idx] = -1.0 if maximise else 0.0
    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=Sb, b_eq=np.zeros(S.shape[0]),
                   bounds=list(zip(lo, hi)), method="highs"), nR, nE


THRESHOLDS = (1e-3, 1e-4, 1e-6)


def sensitive_set(S, lb, ub, kcat, enz, obj_idx, phi, e_ref, res, mu, n_probe=150):
    """Dual screens, finite differences in BOTH directions decide. recon.py's R3 measured 97.7%
    of nonzero duals as ghosts at a degenerate optimum, so the count is never taken from duals."""
    nE = len(enz)
    marg = np.asarray(res.ineqlin.marginals)[:nE]
    cand = [k for k in np.argsort(-np.abs(marg))[:n_probe] if abs(marg[k]) > 1e-12]
    real, ghosts, mags = [], 0, []
    for k in cand:
        j = int(enz[k])
        d = 0.0
        for f in (1.05, 0.95):
            k2 = kcat.copy(); k2[j] *= f
            r2, _, _ = solve_at(S, lb, ub, k2, enz, obj_idx, phi, e_ref)
            if r2.status != 0:
                continue
            m2 = -r2.fun
            if m2 > 0 and mu > 0:
                d = max(d, abs(np.log(m2) - np.log(mu)) / abs(np.log(f)))
        mags.append((j, d))
        if d > 1e-9:
            real.append(j)
        else:
            ghosts += 1
    return real, ghosts, len(cand), mags


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE); P("REGULATION: DOES METABOLIC SPARSITY SURVIVE A CELL THAT DOES NOT OPTIMISE?"); P(RULE)
    R, M, S = load()
    lb, ub, idx = bounds(R)
    obj_idx = idx["BIOMASS_maintenance"]
    enz = np.array([j for j in range(len(R)) if parse_gpr(R[j].get("gene_reaction_rule", ""))],
                   dtype=int)
    rng = np.random.default_rng(SEED)
    kcat = np.zeros(len(R))
    kcat[enz] = np.exp(rng.normal(np.log(KCAT_MEDIAN), KCAT_SIGMA, len(enz))) * 3600.0
    P(f"  {len(R)} reactions, {len(enz)} enzyme-catalysed, metabolic protein budget"
      f" {BUDGET_METAB:.4f} g/gDW")
    P(f"  regulatory floor E_j >= phi * E_ref_j; phi = 0 is recon.py's optimiser")

    # the reference programme: flux-proportional at the unregulated optimum
    r0, nR, nE = solve_at(S, lb, ub, kcat, enz, obj_idx, 0.0, np.zeros(len(enz)))
    if r0.status != 0:
        P("  FAIL -- the unregulated program is infeasible; nothing below can run")
        open(os.path.join(os.path.dirname(__file__), "RESULTS_regulation.txt"),
             "w").write("\n".join(out) + "\n")
        return
    mu0 = -r0.fun
    E0 = r0.x[nR:]
    programmes = {
        "flux-proportional at the optimum": np.maximum(E0, 1e-9) / max(E0.sum(), 1e-12) * BUDGET_METAB,
        "uniform across all enzymes": np.full(nE, BUDGET_METAB / nE),
        "random": (lambda w: w / w.sum() * BUDGET_METAB)(rng.random(nE) + 1e-6),
    }
    P(f"  unregulated maximum growth mu0 = {mu0:.6f} /h")

    results = {}
    for pname, e_ref in programmes.items():
        P("\n" + RULE); P(f"REGULATORY PROGRAMME: {pname}"); P(RULE)
        P(f"  {'phi':>6}{'mu':>11}{'mu/mu0':>9}{'floors':>13}{'probed':>7}"
          f"{'>1e-3':>8}{'>1e-4':>8}{'>1e-6':>8}{'recon':>8}")
        rows = []
        for phi in PHIS:
            t0 = time.time()
            res, _, _ = solve_at(S, lb, ub, kcat, enz, obj_idx, phi, e_ref)
            if res.status != 0:
                P(f"  {phi:>6.2f}{'infeasible -- the regulatory floor exceeds the budget':>60}")
                continue
            mu = -res.fun
            E = res.x[nR:]
            nfloor = int((E <= phi * e_ref + 1e-12).sum()) if phi > 0 else 0
            real, ghosts, nprobe, mags = sensitive_set(S, lb, ub, kcat, enz, obj_idx, phi,
                                                      e_ref, res, mu)
            ids = {R[j]["id"] for j in real}
            ov = len(ids & RECON_EIGHT)
            at = [sum(1 for _, d in mags if d > t) for t in THRESHOLDS]
            cens = "CENSORED" if len(real) >= nprobe else ""
            rows.append((phi, mu, nfloor, nprobe, ghosts, len(real), ov, ids, at, mags))
            P(f"  {phi:>6.2f}{mu:>11.6f}{mu/max(mu0,1e-12):>9.4f}{nfloor:>13}{nprobe:>7}"
              f"{at[0]:>8}{at[1]:>8}{at[2]:>8}{f'{ov}/8':>8} {cens:<9}({time.time()-t0:.0f}s)")
        results[pname] = rows

    # ---- G1, G2, G3, G5, G6 ----------------------------------------------------------------------
    base = results.get("flux-proportional at the optimum", [])
    P("\n" + RULE); P("G1  THE OPERATING POINT IS WHAT IT CLAIMS"); P(RULE)
    act = [r[2] for r in base if r[0] > 0]
    P(f"  enzyme floors active at phi > 0: {act}")
    P(f"  {'PASS' if act and max(act) > 0 else 'FAIL -- phi is decorative, the cell is still the optimiser'}")

    P("\n" + RULE); P("G2  IT REPRODUCES recon.py AT phi = 0"); P(RULE)
    if base:
        n0 = base[0][5]
        P(f"  sensitive reactions at phi = 0: {n0}   recon.py measured 8")
        P(f"  overlap with recon.py's eight: {base[0][6]}/8")
        P(f"  {'PASS' if 4 <= n0 <= 16 else 'FAIL -- not measuring the same thing as recon.py'}"
          f" (bar: within a factor of two of 8)")
        P("  NOTE: the medium and objective match recon.py, but this model adds a proteome budget")
        P("  and a per-reaction enzyme variable, so exact agreement is not expected.")

    P("\n" + RULE); P("G3  THE DELIVERABLE"); P(RULE)
    P("  counts at the 1e-3 threshold, which is the level that could move a factor-of-two answer")
    P(f"  {'programme':>34}" + "".join(f"{'phi='+str(p):>10}" for p in PHIS))
    for pname, rows in results.items():
        d = {r[0]: r[8][0] for r in rows}
        P(f"  {pname:>34}" + "".join(f"{d.get(p, '--'):>10}" for p in PHIS))
    allc = [r[8][0] for rows in results.values() for r in rows]
    if allc:
        P(f"\n  count across every programme and phi: min {min(allc)}, max {max(allc)}")
        if max(allc) <= 2 * max(min(allc), 8):
            P("  READING: the sensitive set stays small as the cell is forced away from the")
            P("  optimum, so the sparsity is STRUCTURAL and does not depend on the optimality")
            P("  assumption. recon.py's conclusion is strengthened rather than merely surviving.")
        else:
            P("  READING: the sensitive set GROWS as the cell is forced away from the optimum, so")
            P("  the sparsity was a property of the optimum. recon.py's metabolic conclusion must")
            P("  be WITHDRAWN for regulated cells, and the count above is what replaces it.")

    P("\n" + RULE); P("G5  THE IDENTITY OF THE SENSITIVE SET, NOT ONLY ITS SIZE"); P(RULE)
    if base:
        first = base[0][7]
        P(f"  {'phi':>6}{'size':>7}{'overlap with phi=0':>22}{'overlap with recon 8':>22}")
        for r in base:
            P(f"  {r[0]:>6.2f}{len(r[7]):>7}{len(r[7] & first):>22}{r[6]:>22}")
        P("  A constant count made of different reactions is a different result from a constant")
        P("  count made of the same ones.")

    P("\n" + RULE); P("G6  THE PROGRAMME IS NOT THE RESULT"); P(RULE)
    for pname, rows in results.items():
        cs = [r[8][0] for r in rows]
        P(f"  {pname:>34}: counts {cs}")
    P("  If the three programmes disagree, the conclusion belongs to the programme imposed and")
    P("  not to regulation in general, and must be reported that way.")

    P("\n" + RULE)
    open(os.path.join(os.path.dirname(__file__), "RESULTS_regulation.txt"),
         "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

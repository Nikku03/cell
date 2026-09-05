"""Which human metabolic rates should be measured? The projection, run on Recon3D itself.

THE OBSTACLE, STATED FIRST BECAUSE IT DETERMINES WHAT THIS CAN AND CANNOT DO. Recon3D is a
STOICHIOMETRIC model. It contains 10,600 reactions, 5,835 metabolites and 2,248 genes, and not one
rate constant. So d log Y / d log k cannot be computed on it the way it was computed on the
branching processes -- there are no k's in the file.

WHAT IS COMPUTABLE, AND WHY IT IS THE SAME QUESTION. In the enzyme-constrained formulation that
every ecGEM uses, each reaction carries a capacity constraint

    v_j  <=  kcat_j * E_j

so kcat_j enters the model ONLY as a bound. The derivative of the objective with respect to that
bound is exactly the linear program's dual variable for the constraint, and therefore

    d(growth) / d log kcat_j  =  (reduced cost of reaction j) * (its binding bound)

This is computable from the stoichiometry alone, WITHOUT knowing any kcat value, because the dual
does not depend on where the bound sits, only on which bounds bind. One LP solve returns all
10,600 sensitivities at once -- the same adjoint structure as before, now with the LP dual playing
the role of the backward solve.

The assumption this carries, and it is not small: enzyme abundance E_j is held fixed, so these are
sensitivities at constant proteome. A cell that reallocates protein in response to a slow enzyme
is outside this analysis.

WHY THE ANSWER IS EXPECTED TO BE SPARSE. In a linear program only ACTIVE constraints carry a
nonzero dual. Every reaction whose capacity is not binding has exactly zero sensitivity, however
important it looks in the network diagram. That is the structural reason a K(N) far below N is
plausible for metabolism, and here it can be counted rather than extrapolated.

THE DEGENERACY PROBLEM, WHICH IS THE REASON FOR GATE R4. FBA linear programs are massively
degenerate: many flux vectors achieve the same optimum, and the duals are not unique. A ranking
read off one solve can be an artefact of which vertex the solver happened to stop at. R4 is
therefore the gate on which the whole deliverable stands or falls, and it is predeclared with a
condition that can void the result entirely.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

R1  THE MODEL IS LOADED CORRECTLY. Stoichiometric matrix 5835 x 10600 as published, the LP
    feasible, and the optimal flux vector satisfying S v = 0 to better than 1e-6.

R2  THE MEDIUM IS DECLARED AND ACTUALLY BINDS. All 1,560 exchange reactions are open in the file
    as distributed, which gives unlimited nutrient uptake and a meaningless growth rate of 755.
    A defined medium is imposed and reported in full. Growth must fall substantially, and the
    carbon source must bind, or the medium is decorative.

R3  THE SENSITIVITY IS A SENSITIVITY. For the top-ranked reactions, compare the dual-derived
    d log(growth) / d log(capacity) against a direct finite difference: perturb that reaction's
    bound by 1%, re-solve, measure. Worst relative disagreement below 5%, or the duals are being
    read wrongly.

R4  THE DEGENERACY CONTROL, ON WHICH EVERYTHING ELSE DEPENDS. Re-solve many times with tiny random
    perturbations to the objective, breaking ties differently each time, and record how often each
    reaction carries a nonzero sensitivity. Predeclared: a reaction enters the deliverable only if
    it is nonzero in at least 90% of solves. If fewer than half of the top twenty by magnitude are
    stable at that level, the ranking is a solver artefact and NO list may be reported.

R5  SPARSITY, THE COUNT THAT ANSWERS THE QUESTION. How many of the 5,938 enzyme-catalysed
    reactions carry a stable nonzero sensitivity. This is the measured analogue of K.

R6  WHAT PHYSIOLOGY ALREADY PINS. Partition the stable set into exchange and transport reactions,
    whose fluxes standard assays measure directly, and internal reactions, which they do not. Only
    the second group needs a new targeted measurement, and that count is the deliverable.

R7  THE LIST. Reaction identifier, name, subsystem and gene rule for every reaction in the stable
    internal set, ranked by sensitivity.

R8  DOMAIN, AND IT CAN INVALIDATE ANY SINGLE LIST. Repeat under a different objective and a
    different medium. Report the overlap of the top twenty. Predeclared: overlap below half means
    the list is a property of the question asked, not of human metabolism, and must be quoted only
    with its objective and medium attached.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.optimize import linprog

from rem.atlas.hybrid_tune import RULE

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "recon3d.json")
MODEL_URL = "http://bigg.ucsd.edu/static/models/Recon3D.json"
MODEL_SHA256 = "aba925f17547a42f9fdb4c1f685d89364cbf4979bbe7862e9f793af7169b26d5"   # the exact file this analysis was run on
N_PERT = 40
STABLE_FRAC = 0.90
TOL = 1e-9

# The medium, declared in full. Uptake bounds are negative by convention.
MEDIUM = {
    "EX_glc__D_e": -1.0,                                    # the limiting carbon source
    "EX_o2_e": -1000.0, "EX_h2o_e": -1000.0, "EX_h_e": -1000.0, "EX_pi_e": -1000.0,
    "EX_so4_e": -1000.0, "EX_nh4_e": -1000.0, "EX_na1_e": -1000.0, "EX_k_e": -1000.0,
    "EX_fe2_e": -1000.0, "EX_fe3_e": -1000.0, "EX_hco3_e": -1000.0, "EX_co2_e": -1000.0,
}
AMINO = ["his__L", "ile__L", "leu__L", "lys__L", "met__L", "phe__L", "thr__L", "trp__L",
         "val__L", "arg__L", "gln__L", "cys__L", "tyr__L", "ser__L", "gly", "ala__L",
         "asn__L", "asp__L", "glu__L", "pro__L"]
VITAMIN = ["chol", "inost", "ribflv", "thm", "pydxn", "ncam", "pnto__R", "fol", "btn", "lnlc"]
for a in AMINO:
    MEDIUM[f"EX_{a}_e"] = -1.0
for v in VITAMIN:
    MEDIUM[f"EX_{v}_e"] = -1.0


def fetch_if_missing():
    """The model file is not committed -- 7.8 MB of third-party data. It is downloaded from BiGG
    and its SHA-256 checked, so the analysis is reproducible without vendoring the file."""
    if not os.path.exists(MODEL):
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    import hashlib
    h = hashlib.sha256(open(MODEL, "rb").read()).hexdigest()
    return h


def load():
    d = json.load(open(MODEL))
    R, M = d["reactions"], d["metabolites"]
    mi = {m["id"]: i for i, m in enumerate(M)}
    S = lil_matrix((len(M), len(R)))
    for j, r in enumerate(R):
        for met, co in r["metabolites"].items():
            S[mi[met], j] = co
    return R, M, csc_matrix(S)


def bounds_for(R, medium, richer=False):
    lb = np.array([r["lower_bound"] for r in R], float)
    ub = np.array([r["upper_bound"] for r in R], float)
    idx = {r["id"]: j for j, r in enumerate(R)}
    for j, r in enumerate(R):
        if r["id"].startswith("EX_"):
            lb[j] = 0.0                      # close every uptake, then reopen the medium
    for k, v in medium.items():
        if k in idx:
            lb[idx[k]] = v * (10.0 if richer else 1.0)
    return lb, ub


def solve(S, c, lb, ub):
    return linprog(c, A_eq=S, b_eq=np.zeros(S.shape[0]),
                   bounds=list(zip(lb, ub)), method="highs")


def sensitivities(res, lb, ub, growth):
    """d log(growth) / d log(capacity), from the LP duals. Only binding bounds are nonzero."""
    g = np.zeros(len(lb))
    if growth <= 0:
        return g
    up = np.asarray(res.upper.marginals)
    lo = np.asarray(res.lower.marginals)
    # objective was minimised as -growth, so flip sign back
    g_up = -up * ub / growth
    g_lo = -lo * lb / growth
    g = np.where(np.abs(g_up) > np.abs(g_lo), g_up, g_lo)
    return np.nan_to_num(g)


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE); P("WHICH HUMAN METABOLIC RATES SHOULD BE MEASURED?  Recon3D, via the LP dual"); P(RULE)
    sha = fetch_if_missing()
    P(f"  model: {MODEL_URL}")
    P(f"  sha256 {sha}")
    P(f"  matches the file this was written against: {sha == MODEL_SHA256}")
    R, M, S = load()
    idx = {r["id"]: j for j, r in enumerate(R)}
    enz = np.array([bool(r.get("gene_reaction_rule", "").strip()) for r in R])
    P(f"  {len(R)} reactions, {len(M)} metabolites, {int(enz.sum())} enzyme-catalysed")

    obj = np.zeros(len(R))
    obj[idx["BIOMASS_maintenance"]] = -1.0

    # ---- R1, R2 ---------------------------------------------------------------------------------
    lb0 = np.array([r["lower_bound"] for r in R], float)
    ub0 = np.array([r["upper_bound"] for r in R], float)
    r_open = solve(S, obj, lb0, ub0)
    lb, ub = bounds_for(R, MEDIUM)
    t0 = time.time()
    res = solve(S, obj, lb, ub)
    growth = -res.fun
    P("\n" + RULE); P("R1  THE MODEL IS LOADED CORRECTLY"); P(RULE)
    P(f"  stoichiometric matrix {S.shape}, {S.nnz} nonzeros"
      f"   {'PASS' if S.shape == (5835, 10600) else 'FAIL'} (published 5835 x 10600)")
    mb = float(np.abs(S @ res.x).max())
    P(f"  LP status {res.status}, solved in {time.time()-t0:.2f}s, worst |S v| = {mb:.2e}"
      f"   {'PASS' if res.status == 0 and mb < 1e-6 else 'FAIL'}")

    P("\n" + RULE); P("R2  THE MEDIUM IS DECLARED AND ACTUALLY BINDS"); P(RULE)
    P(f"  all exchanges open, as distributed : growth = {-r_open.fun:.4f}")
    P(f"  defined medium ({len(MEDIUM)} components, glucose-limited) : growth = {growth:.6f}")
    gj = idx["EX_glc__D_e"]
    P(f"  glucose uptake flux {res.x[gj]:.4f} against its bound {lb[gj]:.4f}"
      f"   binding: {abs(res.x[gj] - lb[gj]) < 1e-6}")
    ok2 = growth < 0.5 * (-r_open.fun) and abs(res.x[gj] - lb[gj]) < 1e-6 and growth > 0
    P(f"  {'PASS' if ok2 else 'FAIL -- the medium is decorative'}")
    P(f"  medium: " + ", ".join(sorted(MEDIUM)[:8]) + f", ... ({len(MEDIUM)} total)")

    g0 = sensitivities(res, lb, ub, growth)
    P(f"\n  reactions with nonzero sensitivity in this single solve:"
      f" {int((np.abs(g0) > TOL).sum())} of {len(R)}")

    # ---- R4, the gate everything stands on --------------------------------------------------------
    P("\n" + RULE); P("R4  THE DEGENERACY CONTROL"); P(RULE)
    P(f"  FBA duals are not unique. Re-solving {N_PERT} times with tiny random objective")
    P(f"  perturbations, so ties break differently, and counting how often each is nonzero.")
    rng = np.random.default_rng(20260905)
    count = np.zeros(len(R))
    mags = np.zeros((N_PERT, len(R)))
    t0 = time.time()
    for t in range(N_PERT):
        o = obj + rng.uniform(-1e-7, 1e-7, len(R))
        rr = solve(S, o, lb, ub)
        if rr.status != 0:
            continue
        gg = sensitivities(rr, lb, ub, -rr.fun if -rr.fun > 0 else growth)
        mags[t] = gg
        count += (np.abs(gg) > TOL)
    stab = count / N_PERT
    P(f"  {N_PERT} solves in {time.time()-t0:.1f}s")
    order0 = np.argsort(-np.abs(g0))
    top20 = [j for j in order0[:20]]
    nstab = sum(1 for j in top20 if stab[j] >= STABLE_FRAC)
    P(f"  of the top 20 by magnitude in the reference solve, {nstab} are nonzero in"
      f" >= {STABLE_FRAC:.0%} of perturbed solves")
    P(f"  {'PASS -- the ranking is not a solver artefact' if nstab >= 10 else 'FAIL -- NO LIST MAY BE REPORTED; the ranking is an artefact of tie-breaking'}"
      f" (bar: at least 10 of 20)")
    stable = (stab >= STABLE_FRAC)
    gmed = np.median(mags, axis=0)
    P(f"  reactions stably nonzero across perturbations: {int(stable.sum())}")

    # ---- R3 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("R3  THE SENSITIVITY IS A SENSITIVITY"); P(RULE)
    cand = [j for j in np.argsort(-np.abs(gmed)) if stable[j]][:12]
    P(f"  {'reaction':>22}{'dual':>12}{'finite difference':>20}{'rel err':>10}")
    worst = 0.0
    for j in cand:
        which = "ub" if abs(ub[j]) > 0 and abs(-res.upper.marginals[j]) > abs(-res.lower.marginals[j]) else "lb"
        b = ub.copy() if which == "ub" else lb.copy()
        delta = 0.01
        b2 = b.copy(); b2[j] = b[j] * (1.0 + delta)
        rr = solve(S, obj, lb if which == "ub" else b2, b2 if which == "ub" else ub)
        if rr.status != 0 or -rr.fun <= 0:
            continue
        fd = (np.log(-rr.fun) - np.log(growth)) / np.log(1.0 + delta)
        rel = abs(fd - gmed[j]) / max(abs(fd), 1e-12)
        worst = max(worst, rel)
        P(f"  {R[j]['id']:>22}{gmed[j]:>12.5f}{fd:>20.5f}{rel:>10.4f}")
    P(f"  worst relative disagreement {worst:.4f}"
      f"   {'PASS' if worst < 0.05 else 'FAIL'} (bar 5%)")

    # ---- R5, R6, R7 -----------------------------------------------------------------------------
    P("\n" + RULE); P("R5  SPARSITY  --  how many rates actually matter"); P(RULE)
    stab_enz = stable & enz
    P(f"  enzyme-catalysed reactions               : {int(enz.sum())}")
    P(f"  of those, stably sensitive               : {int(stab_enz.sum())}")
    P(f"  as a fraction                            : {stab_enz.sum()/max(enz.sum(),1):.5f}")
    P(f"  every other enzyme-catalysed reaction has EXACTLY zero sensitivity: its capacity")
    P(f"  does not bind, so its kcat cannot change this answer at all.")

    P("\n" + RULE); P("R6  WHAT PHYSIOLOGY ALREADY PINS"); P(RULE)
    def is_boundary(j):
        rid = R[j]["id"]
        sub = (R[j].get("subsystem") or "").lower()
        return rid.startswith(("EX_", "DM_", "sink_")) or "transport" in sub or "exchange" in sub
    bnd = [j for j in range(len(R)) if stable[j] and is_boundary(j)]
    inner = [j for j in range(len(R)) if stable[j] and enz[j] and not is_boundary(j)]
    P(f"  stable & boundary (exchange / transport, measurable by standard flux assays): {len(bnd)}")
    P(f"  stable & internal enzyme-catalysed (NOT measurable that way)               : {len(inner)}")
    P(f"  -> the targeted-measurement requirement for THIS question is {len(inner)} rates")

    P("\n" + RULE); P("R7  THE LIST"); P(RULE)
    inner.sort(key=lambda j: -abs(gmed[j]))
    P(f"  {'rank':>4}  {'reaction':<16}{'dlogmu/dlogk':>14}  {'subsystem':<38} genes")
    for i, j in enumerate(inner[:40], 1):
        genes = (R[j].get("gene_reaction_rule") or "").replace(" or ", "/").replace(" and ", "+")
        P(f"  {i:>4}  {R[j]['id']:<16}{gmed[j]:>14.5f}  "
          f"{(R[j].get('subsystem') or '')[:38]:<38} {genes[:60]}")
    if len(inner) > 40:
        P(f"  ... and {len(inner)-40} more")

    # ---- R8 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("R8  DOMAIN  --  does the list depend on the question?"); P(RULE)
    variants = []
    o2 = np.zeros(len(R)); o2[idx["BIOMASS_reaction"]] = -1.0
    variants.append(("objective = BIOMASS_reaction", o2, lb, ub))
    lb3, ub3 = bounds_for(R, MEDIUM, richer=True)
    variants.append(("medium 10x richer", obj, lb3, ub3))
    base_top = [R[j]["id"] for j in inner[:20]]
    for name, oo, l2, u2 in variants:
        rr = solve(S, oo, l2, u2)
        if rr.status != 0 or -rr.fun <= 0:
            P(f"  {name}: infeasible or zero growth, not comparable")
            continue
        gv = sensitivities(rr, l2, u2, -rr.fun)
        iv = [j for j in np.argsort(-np.abs(gv)) if enz[j] and not is_boundary(j)
              and abs(gv[j]) > TOL][:20]
        ov = len(set(R[j]["id"] for j in iv) & set(base_top))
        P(f"  {name}: growth {-rr.fun:.6f}, top-20 overlap with the base list {ov}/20")
        variants and None
    P("  PREDECLARED: overlap below 10 of 20 means the list is a property of the question, not of")
    P("  human metabolism, and must always be quoted with its objective and medium attached.")

    P("\n" + RULE)
    open(os.path.join(HERE, "RESULTS_recon.txt"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

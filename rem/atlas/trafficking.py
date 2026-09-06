"""Trafficking: transport between compartments, and whether transporters are what limit a cell.

WHY THIS IS NOT THE SAME QUESTION AS SPACE. spatial.py measured that diffusion is safe to ignore
-- the crossover sits at Da ~ 3.3 and every process modelled here runs at Da <= 0.1 -- but its S4
found that COMPARTMENTALISATION changes a tail by 0.12 orders at exactly matched mean, for reasons
that have nothing to do with diffusion coefficients. Trafficking is that second thing at genome
scale, and it is much larger than it looks: measured from Recon3D, 4,230 of 10,600 reactions --
39.9% -- span more than one compartment.

THE PARAMETER ASYMMETRY THAT MAKES THIS EXPENSIVE. spatial.py's S6 noted that diffusion is cheap
to parameterise because a diffusion coefficient follows from size and viscosity to within a factor
of a few. Transport is the opposite. It is ACTIVE: pumps, carriers, importins, vesicles. A
transport rate is a protein's turnover number and cannot be derived from physics at all, so every
one of those 4,230 reactions carries a rate that has to be measured exactly like a metabolic kcat.
Trafficking therefore adds to the rate problem in the way diffusion does not.

THE QUESTION WORTH ASKING, AND IT IS DECIDABLE WITH THE MACHINERY ALREADY BUILT. Transport
reactions are 39.9% of the network. Are they 39.9% of the reactions that MATTER, or are they
over-represented? If transporters are disproportionately rate-limiting then trafficking rates
deserve measurement priority over metabolic ones; if under-represented, the reverse. That is a
single number with a baseline to compare against, which is a better shape of question than most
things in this build order have had.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

T1  THE INVENTORY IS ARITHMETIC. Compartment counts, transport-reaction counts and the
    compartment-pair table read from the model file, so a reader can check them.

T2  THE DELIVERABLE: ARE TRANSPORTERS OVER-REPRESENTED AMONG THE RATES THAT MATTER? Compute the
    sensitive set in the enzyme-constrained model, split it into transport and internal, and
    compare against the 39.9% baseline with a binomial test. Predeclared readings: significantly
    ABOVE baseline means trafficking rates deserve priority over metabolic ones; significantly
    BELOW means metabolic enzymes dominate and transport can wait; indistinguishable means
    transport is neither special nor negligible and should be measured in proportion.

T3  VERIFIED, NOT SCREENED. Every membership in the sensitive set comes from a finite difference
    in BOTH directions at a threshold of 1e-3, the level that could move a factor-of-two answer.
    recon.py's R3 measured 97.7% of nonzero LP duals as ghosts, and regulation.py's first run was
    censored at its probe limit; neither mistake is repeated here, and the probe depth is reported
    beside every count.

T4  THE SECRETORY CHAIN. A protein transits ER -> Golgi -> surface, with a chance of loss at each
    stage. Does a multi-step chain change the failure tail against a single effective step matched
    on the same mean transit time? This is the residence-time question of residence.py asked of
    trafficking, and the matched-mean discipline is the one division.py had to learn.

T5  WHAT TRAFFICKING COSTS IN PARAMETERS, counted, and contrasted with diffusion.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import json
import time
import numpy as np
from scipy.stats import binomtest

from rem.atlas.hybrid_tune import RULE
from rem.atlas.recon import MODEL, MEDIUM, fetch_if_missing
from rem.atlas.wholecell import parse_gpr, MW_PROT, KCAT_MEDIAN, KCAT_SIGMA
from rem.atlas.regulation import load, bounds, solve_at, BUDGET_METAB

COMP_NAMES = {"c": "cytosol", "m": "mitochondrion", "e": "extracellular", "n": "nucleus",
              "r": "endoplasmic reticulum", "g": "Golgi", "l": "lysosome", "x": "peroxisome",
              "i": "inner mito membrane"}
THRESH = 1e-3
N_PROBE = 200
SEED = 20260906


def compartments_of(rxn):
    return {met.rsplit("_", 1)[-1] for met in rxn["metabolites"]}


def is_transport(rxn):
    return len(compartments_of(rxn)) > 1


def erlang_tail(n_steps, mean_total, t_cut):
    """P(transit time > t_cut) for a chain of n exponential steps with the given TOTAL mean."""
    from scipy.special import gammaincc
    rate = n_steps / mean_total
    return float(gammaincc(n_steps, rate * t_cut))


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE); P("TRAFFICKING: TRANSPORT BETWEEN COMPARTMENTS, AND WHETHER IT LIMITS THE CELL")
    P(RULE)

    # ---- T1 -------------------------------------------------------------------------------------
    fetch_if_missing()
    d = json.load(open(MODEL))
    Rj, Mj = d["reactions"], d["metabolites"]
    comp = collections.Counter(m["id"].rsplit("_", 1)[-1] for m in Mj)
    P("\n" + RULE); P("T1  THE INVENTORY"); P(RULE)
    P(f"  {'compartment':>26}{'metabolites':>13}")
    for k, v in comp.most_common():
        P(f"  {COMP_NAMES.get(k, k):>26}{v:>13}")
    tr_mask = np.array([is_transport(r) for r in Rj])
    ntr = int(tr_mask.sum())
    P(f"\n  reactions spanning more than one compartment: {ntr} of {len(Rj)}"
      f" ({100*ntr/len(Rj):.1f}%)")
    pairs = collections.Counter()
    for r in Rj:
        cs = compartments_of(r)
        if len(cs) > 1:
            pairs[tuple(sorted(cs))] += 1
    P("  most common compartment pairs:")
    for p, c in pairs.most_common(6):
        P(f"    {' <-> '.join(COMP_NAMES.get(x, x) for x in p):>52}{c:>8}")

    # ---- T2, T3 ---------------------------------------------------------------------------------
    P("\n" + RULE); P("T2/T3  ARE TRANSPORTERS OVER-REPRESENTED AMONG THE RATES THAT MATTER?")
    P(RULE)
    R, M, S = load()
    lb, ub, idx = bounds(R)
    obj_idx = idx["BIOMASS_maintenance"]
    enz = np.array([j for j in range(len(R)) if parse_gpr(R[j].get("gene_reaction_rule", ""))],
                   dtype=int)
    rng = np.random.default_rng(SEED)
    kcat = np.zeros(len(R))
    kcat[enz] = np.exp(rng.normal(np.log(KCAT_MEDIAN), KCAT_SIGMA, len(enz))) * 3600.0
    e_ref = np.zeros(len(enz))

    enz_tr = np.array([is_transport(R[j]) for j in enz])
    base_frac = float(enz_tr.mean())
    P(f"  enzyme-catalysed reactions: {len(enz)}, of which transport: {int(enz_tr.sum())}"
      f" ({100*base_frac:.1f}%)")
    P(f"  BASELINE: if transport mattered in proportion, {100*base_frac:.1f}% of the sensitive")
    P(f"  set would be transport reactions.")

    res, nR, nE = solve_at(S, lb, ub, kcat, enz, obj_idx, 0.0, e_ref)
    if res.status != 0:
        P("  FAIL -- the base program is infeasible")
        open(os.path.join(os.path.dirname(__file__), "RESULTS_trafficking.txt"),
             "w").write("\n".join(out) + "\n")
        return
    mu = -res.fun
    P(f"  base growth {mu:.6f} /h")
    marg = np.asarray(res.ineqlin.marginals)[:nE]
    cand = [k for k in np.argsort(-np.abs(marg))[:N_PROBE] if abs(marg[k]) > 1e-12]
    P(f"  probing the top {len(cand)} by dual, verifying each in BOTH directions at"
      f" threshold {THRESH:g}")
    t0 = time.time()
    sens, mags = [], []
    for k in cand:
        j = int(enz[k])
        dmax = 0.0
        for f in (1.05, 0.95):
            k2 = kcat.copy(); k2[j] *= f
            r2, _, _ = solve_at(S, lb, ub, k2, enz, obj_idx, 0.0, e_ref)
            if r2.status == 0 and -r2.fun > 0:
                dmax = max(dmax, abs(np.log(-r2.fun) - np.log(mu)) / abs(np.log(f)))
        mags.append((j, dmax))
        if dmax > THRESH:
            sens.append(j)
    P(f"  {2*len(cand)} verification solves in {time.time()-t0:.0f}s")
    ns = len(sens)
    ntr_s = sum(1 for j in sens if is_transport(R[j]))
    P(f"\n  sensitive reactions above {THRESH:g}: {ns}"
      f"   {'CENSORED -- every probe was sensitive' if ns >= len(cand) else 'not censored'}")
    if ns:
        obs = ntr_s / ns
        bt = binomtest(ntr_s, ns, base_frac)
        P(f"  of those, transport: {ntr_s} ({100*obs:.1f}%) against a {100*base_frac:.1f}% baseline")
        P(f"  binomial test p = {bt.pvalue:.4g}")
        if bt.pvalue > 0.05:
            P("  READING: indistinguishable from the baseline. Transport is neither special nor")
            P("  negligible, and trafficking rates should be measured in proportion to their")
            P("  share of the network -- which is 40%, so it is a large absolute burden.")
        elif obs > base_frac:
            P("  READING: transport is OVER-represented among the rates that matter, so")
            P("  trafficking rates deserve measurement priority over metabolic ones.")
        else:
            P("  READING: transport is UNDER-represented, so metabolic enzymes dominate and")
            P("  trafficking rates can wait despite being 40% of the network.")
        P(f"\n  the sensitive transport reactions:")
        for j in [j for j in sens if is_transport(R[j])][:14]:
            cs = "/".join(sorted(COMP_NAMES.get(c, c) for c in compartments_of(R[j])))
            P(f"    {R[j]['id']:<16}{cs}")

    # ---- T4 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("T4  THE SECRETORY CHAIN  (does multi-step transit change the tail?)")
    P(RULE)
    P("  A protein transits ER -> Golgi -> surface. Against a single effective step matched on")
    P("  the SAME mean transit time, does the number of stages change the late tail?")
    mean_total = 1.0
    P(f"  {'stages':>8}{'CV of transit':>15}" + "".join(f"{'P(t>'+str(c)+')':>14}"
                                                        for c in (2, 3, 5)))
    for n in (1, 2, 3, 5, 8):
        cv = 1.0 / np.sqrt(n)
        row = f"  {n:>8}{cv:>15.4f}"
        for c in (2, 3, 5):
            row += f"{erlang_tail(n, mean_total, float(c)):>14.4e}"
        P(row)
    r1 = erlang_tail(1, mean_total, 5.0)
    r3 = erlang_tail(3, mean_total, 5.0)
    P(f"\n  a three-stage chain against one step, at five times the mean transit:"
      f" {r1:.4e} against {r3:.4e}, a factor of {r1/max(r3,1e-300):.1f}")
    P("  Multi-step trafficking is a NOISE FILTER on transit time: the coefficient of variation")
    P("  falls as 1/sqrt(stages), and the late tail falls by orders. A model that collapses the")
    P("  secretory pathway into one effective step OVERSTATES how often a protein arrives late by")
    P("  that factor. This is residence.py's question asked of trafficking, and unlike there the")
    P("  answer is that the structure matters a great deal.")

    # ---- T5 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("T5  WHAT TRAFFICKING COSTS IN PARAMETERS"); P(RULE)
    P(f"  transport reactions in Recon3D                     {ntr}")
    P(f"  of which enzyme-catalysed, so carrying a kcat      {int(enz_tr.sum())}")
    P(f"  as a share of all enzyme-catalysed reactions       {100*base_frac:.1f}%")
    P("\n  The asymmetry that matters: a diffusion coefficient follows from size and viscosity to")
    P("  within a factor of a few, so space is nearly free to parameterise. A transport rate is a")
    P("  pump's or a carrier's turnover number -- active, protein-mediated, and derivable from no")
    P("  physics at all. Every one of these must be measured exactly like a metabolic kcat.")
    P("  So of the two things usually filed together under 'spatial biology', the one that is")
    P("  safe to ignore is cheap to parameterise, and the one that is not is expensive.")

    P("\n" + RULE)
    open(os.path.join(os.path.dirname(__file__), "RESULTS_trafficking.txt"),
         "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

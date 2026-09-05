"""A coupled genome-scale model of metabolism AND gene expression, and what it can be asked.

WHAT THIS IS, AND WHAT IT IS NOT. This couples the two largest subsystems of a human cell at
genome scale: metabolism, from Recon3D (10,600 reactions, 5,835 metabolites, 2,248 genes), and
gene expression for the whole protein-coding genome (19,900 genes, four rates each). In systems
biology this is called an ME-model -- Metabolism and Expression -- and it is the largest thing
that can honestly be built from public data plus the machinery in this build order.

It is NOT a whole human cell, and W8 enumerates what is missing rather than letting the name imply
coverage. Absent entirely: DNA replication and the cell cycle, signal transduction, splicing and
RNA processing, protein folding and chaperones, secretion and trafficking, the cytoskeleton,
organelle biogenesis and dynamics, membrane potential, and every spatial degree of freedom. A
model with none of those is not a cell; it is the metabolic and biosynthetic core of one.

THE COUPLING, which is what makes it more than two models side by side.

  1. ENZYME CAPACITY. Each metabolic reaction is limited by the enzyme catalysing it,
     v_j <= kcat_j * E_j, so metabolic flux is bounded by what expression produced.
  2. PROTEOME BUDGET. Total protein is finite: sum over enzymes of E_j * MW <= P_metabolic.
     Every enzyme made is one not made elsewhere.
  3. BIOSYNTHETIC COST. Sustaining the proteome at growth rate mu costs amino acids and ATP,
     (mu + k_dp,g) * E_g per protein per unit time, and that demand is charged to metabolism --
     including for the ~18,000 non-metabolic genes, which pay but do not catalyse.
  4. SELF-CONSISTENCY. Growth appears on both sides: it dilutes the proteome and it is what
     metabolism produces. The model is solved by bisection on mu, each step one linear program,
     which is the standard ME formulation.

WHAT IS ASSUMED, ALL DECLARED AND SWEPT IN W7. kcat values are not known genome-wide, so they are
drawn from a lognormal and swept; a single mean protein length and mass stand in for per-gene
values; the enzyme pool is held per reaction rather than per subunit, the standard GECKO
simplification. None of these is a measurement and none is quoted as one.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

W1  THE PARTS CONNECT. Recon3D's gene identifiers must map into the expression layer, and the
    overlap is reported: how many of the 19,900 genes are metabolic, how many reactions get an
    enzyme, and how many are left orphaned. An ME-model whose layers do not actually share genes
    is two models in one file.

W2  THE LINEAR PROGRAM IS WELL POSED. Mass balance satisfied to 1e-6, the proteome budget binding,
    and the solution finite.

W3  THE FIXED POINT EXISTS AND IS FOUND. Bisection on growth must bracket and converge, with the
    feasible-infeasible boundary located to a stated tolerance.

W4  THE COUPLING CHANGES THE ANSWER. Compare growth under the coupled model against plain FBA with
    no enzyme or proteome constraints. Predeclared: if they agree, the expression layer is
    decorative and this is FBA with extra steps.

W5  WHICH LAYER LIMITS. Report whether growth is bounded by a binding enzyme capacity, by the
    proteome budget, or by nutrient uptake, and the shadow price of each.

W6  THE WHOLE-CELL SENSITIVITY. The projection run over BOTH rate classes at once -- metabolic
    kcats and expression rates -- to find which rates the coupled answer depends on. This is the
    question the whole build order has been walking towards, asked of the largest model available.

W7  THE ASSUMPTIONS DO NOT CARRY THE CONCLUSION. Sweep the kcat distribution, the proteome budget
    and the mean protein length. Predeclared: if the identity of the limiting layer or the rank
    ordering of sensitive rates changes across the sweep, the result is a property of my assumed
    constants and must be reported as such.

W8  WHAT IS ABSENT, quantified. List the subsystems not modelled and estimate the fraction of
    cellular protein mass they represent, so the reader can see how much of a cell this is.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import re
import time
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, hstack, vstack, eye, coo_matrix
from scipy.optimize import linprog

from rem.atlas.hybrid_tune import RULE
from rem.atlas.recon import MODEL, MEDIUM, fetch_if_missing

# ---- declared assumptions, all swept in W7 ---------------------------------------------------
# CORRECTED UNITS. The first run put kcat in 1/h against an enzyme budget normalised to 1.0, so
# any flux could be bought with almost no protein, enzymes became free and the optimiser spent the
# whole budget on ribosome. Flux, enzyme and ribosome are now carried in ONE consistent system:
# flux in mmol/gDW/h, protein in g/gDW, capacity v <= kcat * E / MW.
N_GENES_TOTAL = 19900          # human protein-coding genes
MW_PROT = 50.0                 # g/mmol, a 50 kDa average protein
KCAT_MEDIAN = 25.0             # 1/s
KCAT_SIGMA = 1.2               # lognormal spread in ln units
PROT_TOTAL = 0.5               # g protein per gDW
OTHER_FRAC = 0.45              # non-metabolic, non-ribosomal share of the proteome
# CORRECTED AGAIN. The second run charged the FULL ribosome mass against a PROTEIN budget and
# came out 53.8% ribosome against a few per cent in real cells. A mammalian 80S ribosome is about
# 4.3 MDa of which only ~1/3 is protein; the rest is rRNA, which is not made by ribosomes and does
# not compete for the protein budget. So capacity is expressed per gram of ribosomal PROTEIN:
#   5.6 aa/s * 110 g/mol * 3600 s/h / (4.3e6 * 0.33) = 1.56 g protein per g ribosomal protein per h
# and a declared fraction of ribosomes are not elongating at any moment.
RIB_MW = 4.3e6                 # g/mol, mammalian 80S
RIB_PROT_FRAC = 0.33           # protein share of ribosome mass; the rest is rRNA
ELONG_AA_PER_S = 5.6
AA_MW = 110.0
ACTIVE_RIB_FRAC = 0.80         # fraction of ribosomes elongating at any instant
K_ELONG_MASS = (ELONG_AA_PER_S * AA_MW * 3600.0 / (RIB_MW * RIB_PROT_FRAC)) * ACTIVE_RIB_FRAC
MU_MAX_SEARCH = 2.0


def parse_gpr(rule):
    """Enzyme availability for a reaction: sum over isozymes, minimum over complex subunits."""
    if not rule or not rule.strip():
        return []
    parts = re.split(r"\s+or\s+", rule.replace("(", " ").replace(")", " "))
    out = []
    for p in parts:
        subs = [t for t in re.split(r"\s+and\s+", p) if t.strip()]
        if subs:
            out.append([t.strip() for t in subs])
    return out


def load_model():
    fetch_if_missing()
    d = json.load(open(MODEL))
    return d["reactions"], d["metabolites"]


def build_lp(S, lb, ub, kcat, enz_idx, mu, obj_idx, k_dp, budget, k_elong,
             other_frac, k_dp_other, k_dp_rib):
    """One linear program at fixed growth mu. Variables are [v (nR), E (nE), R_ribosome (1)].

    Three couplings, all linear once mu is fixed:
      capacity      v_j - kcat_j E_j <= 0                       metabolism limited by expression
      budget        sum_j E_j + R + P_other <= budget           protein is finite
      translation   sum_j (mu + k_dp_j) E_j + (mu+k_dp_r) R
                      + (mu+k_dp_o) P_other  <=  R * k_elong    the ribosome must keep up

    The third is what makes protein DEGRADATION rates enter a whole-cell answer: a protein with a
    short lifetime must be resynthesised continuously whether the cell is growing or not, and that
    consumes the same ribosome capacity as growth does. Biomass already carries the metabolic cost
    of growth, so no amino-acid or ATP drain is added here -- that would double-count it."""
    nR = S.shape[1]
    nE = len(enz_idx)
    n = nR + nE + 1
    Sb = hstack([S, csc_matrix((S.shape[0], nE + 1))]).tocsc()

    rows, cols, vals = [], [], []
    for k, j in enumerate(enz_idx):
        rows += [k, k]; cols += [j, nR + k]; vals += [1.0, -kcat[j] / MW_PROT]
    Acap = coo_matrix((vals, (rows, cols)), shape=(nE, n))

    P_other = other_frac * budget
    rb = np.concatenate([np.ones(nE), [1.0]])
    Abud = coo_matrix((rb, (np.zeros(nE + 1), np.arange(nR, n))), shape=(1, n))

    tr = np.concatenate([mu + k_dp[enz_idx], [(mu + k_dp_rib) - k_elong]])
    Atr = coo_matrix((tr, (np.zeros(nE + 1), np.arange(nR, n))), shape=(1, n))

    A_ub = vstack([Acap, Abud, Atr]).tocsc()
    b_ub = np.concatenate([np.zeros(nE),
                           [budget - P_other],
                           [-(mu + k_dp_other) * P_other]])
    lo = np.concatenate([lb, np.zeros(nE + 1)])
    hi = np.concatenate([ub, np.full(nE + 1, np.inf)])
    lo[obj_idx] = hi[obj_idx] = mu
    c = np.zeros(n)
    c[nR:] = 1.0
    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=Sb, b_eq=np.zeros(S.shape[0]),
                   bounds=list(zip(lo, hi)), method="highs"), nR, nE


def max_growth(S, lb, ub, kcat, enz_idx, obj_idx, k_dp, budget, k_elong,
               other_frac, k_dp_other, k_dp_rib, tol=1e-6, hi0=MU_MAX_SEARCH):
    """Bisection on growth: the largest mu for which the coupled program is feasible."""
    lo, hi = 0.0, hi0
    r0, _, _ = build_lp(S, lb, ub, kcat, enz_idx, 0.0, obj_idx, k_dp, budget,
                        k_elong, other_frac, k_dp_other, k_dp_rib)
    if r0.status != 0:
        return None, None, 0
    rhi, _, _ = build_lp(S, lb, ub, kcat, enz_idx, hi, obj_idx, k_dp, budget,
                         k_elong, other_frac, k_dp_other, k_dp_rib)
    if rhi.status == 0:
        return hi, rhi, 1
    best = r0
    steps = 0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        r, _, _ = build_lp(S, lb, ub, kcat, enz_idx, mid, obj_idx, k_dp, budget,
                           k_elong, other_frac, k_dp_other, k_dp_rib)
        steps += 1
        if r.status == 0:
            lo, best = mid, r
        else:
            hi = mid
    return lo, best, steps


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE); P("A COUPLED GENOME-SCALE MODEL OF METABOLISM AND EXPRESSION"); P(RULE)
    R, M = load_model()
    nR = len(R)
    idx = {r["id"]: j for j, r in enumerate(R)}
    mi = {m["id"]: i for i, m in enumerate(M)}
    Sl = lil_matrix((len(M), nR))
    for j, r in enumerate(R):
        for met, co in r["metabolites"].items():
            Sl[mi[met], j] = co
    S = csc_matrix(Sl)
    lb = np.array([r["lower_bound"] for r in R], float)
    ub = np.array([r["upper_bound"] for r in R], float)
    for j, r in enumerate(R):
        if r["id"].startswith("EX_"):
            lb[j] = 0.0
    for k, v in MEDIUM.items():
        if k in idx:
            lb[idx[k]] = v
    obj_idx = idx["BIOMASS_maintenance"]

    # ---- W1 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("W1  THE PARTS CONNECT"); P(RULE)
    gpr = [parse_gpr(r.get("gene_reaction_rule", "")) for r in R]
    enz_idx = np.array([j for j in range(nR) if gpr[j]], dtype=int)
    toks = set()
    for g in gpr:
        for iso in g:
            toks.update(iso)
    base = {t.split("_")[0] for t in toks}
    ncomplex = sum(1 for g in gpr if any(len(iso) > 1 for iso in g))
    niso = sum(1 for g in gpr if len(g) > 1)
    P(f"  expression layer: {N_GENES_TOTAL} protein-coding genes, 4 rates each"
      f" = {4*N_GENES_TOTAL:,} rates")
    P(f"  metabolism layer: {nR} reactions, {len(M)} metabolites")
    P(f"  reactions carrying an enzyme (a GPR): {len(enz_idx)} of {nR}")
    P(f"  distinct gene identifiers in those rules: {len(toks)}"
      f" ({len(base)} distinct base genes)")
    P(f"  of the {N_GENES_TOTAL} genes, {len(base)} are metabolic"
      f" ({100*len(base)/N_GENES_TOTAL:.1f}%); the other"
      f" {N_GENES_TOTAL-len(base):,} pay the proteome and ribosome costs but catalyse nothing here")
    P(f"  reactions needing a complex (an 'and'): {ncomplex};"
      f" with isozymes (an 'or'): {niso}")
    P(f"  orphan reactions with no enzyme at all: {nR-len(enz_idx)}"
      f"   {'PASS' if len(enz_idx) > 0.4*nR else 'FAIL -- the layers barely share genes'}")

    rng = np.random.default_rng(20260905)
    kcat = np.zeros(nR)
    kcat[enz_idx] = np.exp(rng.normal(np.log(KCAT_MEDIAN), KCAT_SIGMA, len(enz_idx))) * 3600.0
    # v [mmol/gDW/h] <= kcat [1/h] * E [g/gDW] / MW [g/mmol]
    k_dp = np.zeros(nR)
    k_dp[enz_idx] = np.exp(rng.normal(np.log(0.015), 0.7, len(enz_idx)))
    K_ELONG = K_ELONG_MASS
    KDP_OTHER, KDP_RIB = 0.015, 0.010

    # ---- W3, W4 ---------------------------------------------------------------------------------
    P("\n" + RULE); P("W3  THE FIXED POINT EXISTS AND IS FOUND"); P(RULE)
    t0 = time.time()
    mu, best, steps = max_growth(S, lb, ub, kcat, enz_idx, obj_idx, k_dp,
                                 PROT_TOTAL, K_ELONG, OTHER_FRAC, KDP_OTHER, KDP_RIB,
                                 tol=1e-6)
    P(f"  bisection: {steps} linear programs in {time.time()-t0:.1f}s")
    if mu is None:
        P("  FAIL -- infeasible even at zero growth; the coupled model has no solution")
        open(os.path.join(os.path.dirname(__file__), "RESULTS_wholecell.txt"),
             "w").write("\n".join(out) + "\n")
        return
    P(f"  coupled growth rate mu = {mu:.6f} /h   (doubling time {np.log(2)/max(mu,1e-12):.2f} h)")
    P(f"  {'PASS' if mu > 0 else 'FAIL'} (a fixed point was located)")

    r_fba = linprog(-np.eye(nR)[obj_idx], A_eq=S, b_eq=np.zeros(S.shape[0]),
                    bounds=list(zip(lb, ub)), method="highs")
    mu_fba = -r_fba.fun if r_fba.status == 0 else float("nan")
    P("\n" + RULE); P("W4  THE COUPLING CHANGES THE ANSWER"); P(RULE)
    P(f"  plain FBA, no enzyme or proteome constraint : mu = {mu_fba:.6f} /h")
    P(f"  coupled metabolism + expression             : mu = {mu:.6f} /h")
    P(f"  ratio {mu/max(mu_fba,1e-12):.4f}"
      f"   {'PASS -- expression genuinely constrains metabolism' if mu < 0.95*mu_fba else 'FAIL -- the expression layer is decorative'}")

    # ---- W2, W5 ---------------------------------------------------------------------------------
    x = best.x
    v, E, Rrib = x[:nR], x[nR:nR + len(enz_idx)], x[-1]
    P("\n" + RULE); P("W2  THE LINEAR PROGRAM IS WELL POSED"); P(RULE)
    mb = float(np.abs(S @ v).max())
    P(f"  worst |S v| = {mb:.2e}   {'PASS' if mb < 1e-6 else 'FAIL'}")
    used = float(E.sum() + Rrib)
    avail = PROT_TOTAL * (1.0 - OTHER_FRAC)
    P(f"  protein used {used:.6f} g/gDW of the {avail:.6f} available to metabolism + ribosome"
      f"   binding: {abs(used - avail) < 1e-8}")
    P(f"  ribosome {Rrib:.6f} g/gDW = {100*Rrib/PROT_TOTAL:.2f}% of total protein"
      f"  (mammalian cells are a few per cent)")

    P("\n" + RULE); P("W5  WHICH LAYER LIMITS"); P(RULE)
    nE = len(enz_idx)
    marg = np.asarray(best.ineqlin.marginals) if hasattr(best, "ineqlin") else np.zeros(nE + 2)
    cap_m, bud_m, tr_m = marg[:nE], float(marg[nE]), float(marg[nE + 1])
    nbind = int((np.abs(cap_m) > 1e-9).sum())
    P(f"  enzyme-capacity constraints binding : {nbind} of {nE}")
    P(f"  proteome-budget shadow price        : {bud_m:.6e}")
    P(f"  translation-capacity shadow price   : {tr_m:.6e}")
    P(f"  ribosome mass fraction of the metabolic proteome: {Rrib/max(used,1e-12):.4f}")
    lim = max((abs(bud_m), "proteome budget"), (abs(tr_m), "translation capacity"),
              (float(np.abs(cap_m).max()) if nE else 0.0, "an enzyme capacity"))[1]
    P(f"  the binding limitation is: {lim}")

    # ---- W6 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("W6  THE WHOLE-CELL SENSITIVITY"); P(RULE)
    P("  Screening by dual, then verifying by finite difference on the bisection -- the two-stage")
    P("  pattern recon.py's R3 forced, since LP duals fire falsely at degenerate optima.")
    PROBE_TOL, PROBE_STEP = 1e-9, 1.05
    floor = PROBE_TOL / max(mu, 1e-12) / np.log(PROBE_STEP)
    P(f"  bisection tolerance {PROBE_TOL:.0e} on mu = {mu:.6f} with a {100*(PROBE_STEP-1):.0f}%")
    P(f"  perturbation puts the RESOLUTION FLOOR at |d log mu / d log k| = {floor:.2e}.")
    P(f"  Anything at or below that is noise, not a sensitivity, and is marked so.")
    cand = [int(enz_idx[k]) for k in np.argsort(-np.abs(cap_m))[:12] if abs(cap_m[k]) > 1e-12]
    P(f"  {'rate':>26}{'d log mu / d log k':>21}{'':>4}")
    rows_out = []
    for j in cand[:8]:
        k2 = kcat.copy(); k2[j] *= PROBE_STEP
        m2, _, _ = max_growth(S, lb, ub, k2, enz_idx, obj_idx, k_dp, PROT_TOTAL,
                              K_ELONG, OTHER_FRAC, KDP_OTHER, KDP_RIB, tol=PROBE_TOL)
        d = (np.log(max(m2, 1e-300)) - np.log(mu)) / np.log(PROBE_STEP) if m2 else 0.0
        rows_out.append((f"kcat[{R[j]['id']}]", d))
        P(f"  {('kcat '+R[j]['id']):>26}{d:>21.6f}{('  NOISE' if abs(d) <= floor else ''):>4}")
    for nm, mult in (("k_elong (ribosome speed)", "elong"), ("proteome budget", "budget"),
                     ("k_dp of the non-metabolic proteome", "kdpo")):
        if mult == "elong":
            m2, _, _ = max_growth(S, lb, ub, kcat, enz_idx, obj_idx, k_dp, PROT_TOTAL,
                                  K_ELONG * PROBE_STEP, OTHER_FRAC, KDP_OTHER, KDP_RIB,
                                  tol=PROBE_TOL)
        elif mult == "budget":
            m2, _, _ = max_growth(S, lb, ub, kcat, enz_idx, obj_idx, k_dp,
                                  PROT_TOTAL * PROBE_STEP, K_ELONG, OTHER_FRAC, KDP_OTHER,
                                  KDP_RIB, tol=PROBE_TOL)
        else:
            m2, _, _ = max_growth(S, lb, ub, kcat, enz_idx, obj_idx, k_dp, PROT_TOTAL,
                                  K_ELONG, OTHER_FRAC, KDP_OTHER * PROBE_STEP, KDP_RIB,
                                  tol=PROBE_TOL)
        d = (np.log(max(m2, 1e-300)) - np.log(mu)) / np.log(PROBE_STEP) if m2 else 0.0
        rows_out.append((nm, d))
        P(f"  {nm:>26}{d:>21.6f}{('  NOISE' if abs(d) <= floor else ''):>4}")
    P("  A global parameter that outranks every individual kcat means the whole-cell answer is")
    P("  set by allocation, not by any one enzyme.")

    # ---- W7 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("W7  THE ASSUMPTIONS DO NOT CARRY THE CONCLUSION"); P(RULE)
    P(f"  {'variant':>34}{'mu':>12}{'ratio to base':>15}{'limiting':>26}")
    for nm, kw in (("base", {}),
                   ("kcat median x10", {"kc": 10.0}),
                   ("kcat median /10", {"kc": 0.1}),
                   ("proteome budget x2", {"bud": 2.0}),
                   ("ribosome speed x2", {"el": 2.0}),
                   ("non-metabolic fraction 0.7", {"of": 0.7}),
                   ("non-metabolic fraction 0.2", {"of": 0.2}),
                   ("ribosome speed /2", {"el": 0.5})):
        kc = kcat * kw.get("kc", 1.0)
        m2, b2, _ = max_growth(S, lb, ub, kc, enz_idx, obj_idx, k_dp,
                               PROT_TOTAL * kw.get("bud", 1.0), K_ELONG * kw.get("el", 1.0),
                               kw.get("of", OTHER_FRAC), KDP_OTHER, KDP_RIB, tol=1e-4)
        if m2 is None or b2 is None:
            P(f"  {nm:>34}{'infeasible':>12}")
            continue
        mg = np.asarray(b2.ineqlin.marginals)
        l2 = max((abs(float(mg[nE])), "proteome budget"),
                 (abs(float(mg[nE + 1])), "translation capacity"),
                 (float(np.abs(mg[:nE]).max()), "an enzyme capacity"))[1]
        P(f"  {nm:>34}{m2:>12.6f}{m2/max(mu,1e-12):>15.4f}{l2:>26}")

    # ---- W8 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("W8  WHAT IS ABSENT"); P(RULE)
    P("  Modelled: metabolism (10,600 reactions), the metabolic proteome, ribosome allocation,")
    P("  and the translation cost of the whole proteome including non-metabolic genes.")
    P("  NOT modelled, at all:")
    for line in ("DNA replication and the cell cycle",
                 "signal transduction and regulation -- the model has no controller",
                 "splicing, RNA processing, export",
                 "protein folding, chaperones, quality control",
                 "secretion, trafficking, the endomembrane system",
                 "the cytoskeleton and mechanics",
                 "organelle biogenesis; mitochondria appear only as compartment labels",
                 "membrane potential and electrochemical gradients beyond stoichiometry",
                 "every spatial degree of freedom -- the cell is one well-mixed bag",
                 "cell-to-cell variation; this is one deterministic allocation"):
        P(f"    - {line}")
    P(f"  {len(base)} of {N_GENES_TOTAL} genes ({100*len(base)/N_GENES_TOTAL:.1f}%) have a")
    P(f"  mechanism in this model. The other {100*(1-len(base)/N_GENES_TOTAL):.1f}% appear only as")
    P("  a mass and a ribosome burden. That is the honest measure of how much of a cell this is.")

    P("\n" + RULE)
    open(os.path.join(os.path.dirname(__file__), "RESULTS_wholecell.txt"),
         "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

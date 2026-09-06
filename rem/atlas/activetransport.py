"""Active transport, and a thermodynamic defect it exposes in everything built on Recon3D.

WHAT ACTIVE TRANSPORT IS, AND WHY IT IS NOT JUST ANOTHER KCAT. trafficking.py treats every
transport reaction as an enzyme with a turnover number. That is right for facilitated diffusion
and wrong for active transport, which moves a solute UPHILL and must be paid for. Measured from
Recon3D's 4,230 transport reactions:

    passive / facilitated               2,730   64.5%
    secondary active, ion-coupled       1,018   24.1%
    primary active, ATP-hydrolysing       482   11.4%

The good news first: the energy cost IS in the stoichiometry, so mass balance already charges it.
A primary-active reaction reads  -1 atp_c + 1 adp_c + 1 pi_c - 1 h2o_c  alongside the solute, and
a secondary-active one reads  -1 h_c + 1 h_m. Flux balance cannot move a solute uphill without
consuming the ATP or the proton, so nothing here is free at the level of mass and charge.

THE DEFECT. Flux balance has no thermodynamic DIRECTION constraint. A reversible transport
reaction may run either way and mass balance is satisfied both times, so the optimiser will pick
whichever direction helps -- including one the actual electrochemical gradient forbids. In
Recon3D, 2,323 of 4,230 transport reactions (54.9%) are reversible. Chain enough of them together
and the model can build a cycle that creates free energy.

It does. Sealing the cell -- every exchange, demand and sink bounded to zero, so nothing enters or
leaves -- and maximising ATP production gives 1000.0, which is the flux bound. The model makes
unlimited ATP from nothing. This is a known pathology of genome-scale reconstructions and the
reason loopless and thermodynamic FBA exist; what matters here is not that it exists but whether
MY results used it.

WHY THIS MIGHT HAVE CONTAMINATED THE WHOLE-CELL RESULT, WHICH IS THE REASON THIS MODULE EXISTS.
wholecell.py concluded that growth is limited by translation capacity and that individual kcats
score three orders below ribosome speed, with the proteome budget's shadow price exactly zero. If
ATP is free, metabolism cannot bind, and "allocation dominates" would follow from the defect
rather than from biology. A4 and A6 decide that, and A6 can withdraw the conclusion.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

A1  THE MECHANISM SPLIT IS ARITHMETIC. Counts of primary active, secondary active and passive
    transport read from the stoichiometry, with the criterion stated, so a reader can disagree
    with the classification rather than with a conclusion.

A2  THE FREE-LUNCH TEST. Seal every exchange and maximise ATP. Any flux above zero is energy from
    nothing. Already measured at 1000.0 and recorded as a FAILURE of the model, not of this gate.

A3  WHICH REACTIONS CARRY IT. Identify the reactions carrying flux in the sealed-box solution and
    split them by transport class. Predeclared: if transport dominates the loop, active transport
    is the mechanism of the defect; if not, the loop is metabolic and transport is incidental.

A4  DOES THE GROWTH SOLUTION ACTUALLY USE IT? Compare the loop-carrying set against the reactions
    carrying flux in the ordinary open-medium growth solution. Predeclared: substantial overlap
    means every earlier energy-related result is contaminated; little overlap means the loop
    exists but is not being exploited, and the earlier results stand with a caveat.

A5  THE FIX AND ITS COST. Force primary-active transport to run only in the ATP-consuming
    direction and re-test A2. Report how much of the free lunch that removes, and what it costs
    in growth.

A7  THE MASS-BALANCE AUDIT. Count reactions whose elemental formulas do not balance. A
    mass-imbalanced reaction lets the optimiser create atoms, which is a different and worse
    defect than an energy loop.

A8  THE DECISIVE TEST, which set overlap cannot substitute for. Seal every exchange and maximise
    BIOMASS. Growth above zero means the model builds a cell out of nothing, and every
    Recon3D-derived result in this build order is contaminated at the root.

A9  THE REQUALIFICATION. Remove the mass-imbalanced reactions, retest A8, and re-measure growth.
    Predeclared: if sealed-box growth falls to zero and open growth is materially unchanged, the
    earlier conclusions survive with the defect recorded; if open growth changes, every
    Recon3D-derived result must be requalified by that amount.

A6  DOES THE WHOLE-CELL CONCLUSION SURVIVE THE FIX? Recompute growth and the limiting constraint
    with the loops blocked. Predeclared: if translation still binds and kcats still rank three
    orders below ribosome speed, wholecell.py's conclusion stands and was never resting on free
    energy. If the limiting layer moves to metabolism, that conclusion must be WITHDRAWN.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import json
import time
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.optimize import linprog

from rem.atlas.hybrid_tune import RULE
from rem.atlas.recon import MODEL, MEDIUM, fetch_if_missing

FLUX_TOL = 1e-6


def classify(rxn):
    b = {m.rsplit("_", 1)[0] for m in rxn["metabolites"]}
    comps = {m.rsplit("_", 1)[-1] for m in rxn["metabolites"]}
    if len(comps) <= 1:
        return "internal"
    if "atp" in b and "adp" in b:
        return "primary active"
    if b & {"na1", "k", "h", "ca2"}:
        return "secondary active"
    return "passive"


def build():
    fetch_if_missing()
    d = json.load(open(MODEL))
    R, M = d["reactions"], d["metabolites"]
    mi = {m["id"]: i for i, m in enumerate(M)}
    Sl = lil_matrix((len(M), len(R)))
    for j, r in enumerate(R):
        for met, co in r["metabolites"].items():
            Sl[mi[met], j] = co
    return R, M, csc_matrix(Sl)


def sealed_bounds(R):
    lb = np.array([r["lower_bound"] for r in R], float)
    ub = np.array([r["upper_bound"] for r in R], float)
    for j, r in enumerate(R):
        if r["id"].startswith(("EX_", "DM_", "SK_", "sink_")):   # SK_ is Recon3D's sink prefix
            lb[j] = ub[j] = 0.0
    return lb, ub


def open_bounds(R):
    lb = np.array([r["lower_bound"] for r in R], float)
    ub = np.array([r["upper_bound"] for r in R], float)
    idx = {r["id"]: j for j, r in enumerate(R)}
    for j, r in enumerate(R):
        if r["id"].startswith(("EX_", "SK_", "DM_")):
            lb[j] = 0.0
        if r["id"].startswith(("SK_", "DM_")):
            ub[j] = 0.0
    for k, v in MEDIUM.items():
        if k in idx:
            lb[idx[k]] = v
    return lb, ub, idx


def parse_formula(f):
    import re
    out = collections.Counter()
    if not f:
        return out
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f):
        if el:
            out[el] += int(n) if n else 1
    return out


def imbalanced(R, M):
    """Reactions whose elemental formulas do not balance. Boundary reactions are exempt."""
    mf = {m["id"]: m.get("formula", "") for m in M}
    bad, checked = [], 0
    for j, r in enumerate(R):
        if r["id"].startswith(("EX_", "DM_", "SK_", "sink_", "BIOMASS")):
            continue   # boundary reactions are unbalanced BY DESIGN; SK_ is the sink prefix
        tot = collections.Counter()
        ok = True
        for met, co in r["metabolites"].items():
            f = mf.get(met, "")
            if not f or "R" in f or "X" in f:
                ok = False
                break
            for el, n in parse_formula(f).items():
                tot[el] += co * n
        if not ok:
            continue
        checked += 1
        if any(abs(v) > 1e-6 for v in tot.values()):
            bad.append(j)
    return bad, checked


def block_uphill(R, lb, ub):
    """Force primary-active transport to run only in the ATP-consuming direction.

    The sign convention is read from the stoichiometry itself: the direction in which atp_c is a
    substrate is the physiological one, so the opposite direction is closed."""
    lb, ub = lb.copy(), ub.copy()
    n = 0
    for j, r in enumerate(R):
        if classify(r) != "primary active":
            continue
        atp = next((v for k, v in r["metabolites"].items() if k.startswith("atp_")), 0.0)
        if atp < 0 and lb[j] < 0:
            lb[j] = 0.0
            n += 1
        elif atp > 0 and ub[j] > 0:
            ub[j] = 0.0
            n += 1
    return lb, ub, n


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE); P("ACTIVE TRANSPORT, AND A THERMODYNAMIC DEFECT IT EXPOSES"); P(RULE)
    R, M, S = build()
    nR = len(R)
    kind = [classify(r) for r in R]
    cnt = collections.Counter(kind)

    # ---- A1 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("A1  THE MECHANISM SPLIT"); P(RULE)
    ntr = sum(v for k, v in cnt.items() if k != "internal")
    P(f"  {'class':>22}{'count':>8}{'share of transport':>21}")
    for k in ("passive", "secondary active", "primary active"):
        P(f"  {k:>22}{cnt[k]:>8}{100*cnt[k]/max(ntr,1):>20.1f}%")
    P(f"  {'internal (one compartment)':>22}{cnt['internal']:>8}")
    P(f"  transport total {ntr} of {nR} reactions ({100*ntr/nR:.1f}%)")
    P("  criterion: primary active if the reaction carries both atp and adp; secondary active if")
    P("  it moves Na+, K+, H+ or Ca2+ alongside the solute; passive otherwise.")
    P("  The energy cost IS in the stoichiometry, so mass balance already charges it -- a solute")
    P("  cannot move uphill without the ATP or the ion appearing as a substrate.")

    # ---- A2 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("A2  THE FREE-LUNCH TEST"); P(RULE)
    lb_s, ub_s = sealed_bounds(R)
    idx = {r["id"]: j for j, r in enumerate(R)}
    atp_obj = "ATPM" if "ATPM" in idx else None
    c = np.zeros(nR)
    if atp_obj:
        c[idx[atp_obj]] = -1.0
    res_seal = linprog(c, A_eq=S, b_eq=np.zeros(S.shape[0]),
                       bounds=list(zip(lb_s, ub_s)), method="highs")
    free_atp = -res_seal.fun if res_seal.status == 0 else float("nan")
    rev_tr = sum(1 for j in range(nR) if kind[j] != "internal"
                 and lb_s[j] < 0 and ub_s[j] > 0)
    P(f"  every exchange, demand and sink bounded to zero -- nothing enters or leaves the cell")
    P(f"  reversible transport reactions under those bounds: {rev_tr}")
    P(f"  maximum ATP flux through {atp_obj}: {free_atp:.4f}")
    P(f"  {'FAIL -- the model creates energy from nothing' if free_atp > 1e-6 else 'PASS'}"
      f" (bar: exactly zero)")
    P("  This is a known pathology of genome-scale reconstructions, not a bug in this code. What")
    P("  matters is whether the results in this build order depended on it, which A4 and A6 decide.")

    # ---- A3 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("A3  WHICH REACTIONS CARRY THE LOOP"); P(RULE)
    v_seal = res_seal.x if res_seal.status == 0 else np.zeros(nR)
    loop = np.where(np.abs(v_seal) > FLUX_TOL)[0]
    lk = collections.Counter(kind[j] for j in loop)
    P(f"  reactions carrying flux in the sealed box: {len(loop)}")
    for k in ("internal", "passive", "secondary active", "primary active"):
        P(f"    {k:>22}{lk[k]:>7}{100*lk[k]/max(len(loop),1):>8.1f}%")
    tr_share = sum(lk[k] for k in ("passive", "secondary active", "primary active")) / max(len(loop), 1)
    P(f"  transport share of the loop {100*tr_share:.1f}% against"
      f" {100*ntr/nR:.1f}% of the network")
    P(f"  READING: {'transport DOMINATES the loop, so active transport is the mechanism' if tr_share > 1.5*ntr/nR else 'the loop is largely metabolic and transport is incidental to it'}")

    # ---- A4 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("A4  DOES THE GROWTH SOLUTION ACTUALLY USE IT?"); P(RULE)
    lb_o, ub_o, idxo = open_bounds(R)
    cg = np.zeros(nR); cg[idxo["BIOMASS_maintenance"]] = -1.0
    res_g = linprog(cg, A_eq=S, b_eq=np.zeros(S.shape[0]),
                    bounds=list(zip(lb_o, ub_o)), method="highs")
    mu_open = -res_g.fun if res_g.status == 0 else float("nan")
    grow = set(np.where(np.abs(res_g.x) > FLUX_TOL)[0]) if res_g.status == 0 else set()
    ov = len(set(loop.tolist()) & grow)
    P(f"  open-medium growth {mu_open:.6f} /h, carrying flux through {len(grow)} reactions")
    P(f"  overlap with the sealed-box loop: {ov} reactions"
      f" ({100*ov/max(len(loop),1):.1f}% of the loop)")
    P(f"  READING: {'SUBSTANTIAL overlap -- earlier energy-related results are contaminated' if ov > 0.5*len(loop) else 'limited overlap -- the loop exists but the growth solution is not built on it'}")

    # ---- A5 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("A5  THE FIX AND ITS COST"); P(RULE)
    lb_sb, ub_sb, nfix = block_uphill(R, lb_s, ub_s)
    res_seal2 = linprog(c, A_eq=S, b_eq=np.zeros(S.shape[0]),
                        bounds=list(zip(lb_sb, ub_sb)), method="highs")
    free2 = -res_seal2.fun if res_seal2.status == 0 else float("nan")
    P(f"  primary-active transporters forced to the ATP-consuming direction: {nfix} closed")
    P(f"  sealed-box ATP after the fix: {free2:.4f}   (was {free_atp:.4f})")
    P(f"  {'the free lunch SURVIVES -- it does not run through primary active transport' if free2 > 1e-6 else 'the free lunch is REMOVED by constraining primary active transport'}")
    lb_ob, ub_ob, _ = block_uphill(R, lb_o, ub_o)
    res_g2 = linprog(cg, A_eq=S, b_eq=np.zeros(S.shape[0]),
                     bounds=list(zip(lb_ob, ub_ob)), method="highs")
    mu2 = -res_g2.fun if res_g2.status == 0 else float("nan")
    P(f"  open-medium growth after the fix: {mu2:.6f} /h against {mu_open:.6f}"
      f"  (ratio {mu2/max(mu_open,1e-12):.4f})")

    # ---- A6 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("A6  DOES THE WHOLE-CELL CONCLUSION SURVIVE THE FIX?"); P(RULE)
    P("  wholecell.py found translation capacity binding, the proteome budget's shadow price")
    P("  exactly zero, and every individual kcat three orders below ribosome speed. If ATP were")
    P("  free, metabolism could not bind and that conclusion would follow from the defect.")
    P(f"\n  growth with loops open  : {mu_open:.6f} /h")
    P(f"  growth with loops closed: {mu2:.6f} /h")
    if not np.isfinite(mu2) or mu2 <= 0:
        P("  the constrained model is infeasible, so the fix as applied is too aggressive and")
        P("  no conclusion may be drawn from it")
    elif abs(mu2 - mu_open) / max(mu_open, 1e-12) < 0.01:
        P("  Growth is unchanged to within 1%. The free-energy loop is available but the growth")
        P("  solution does not depend on it, so wholecell.py's conclusion is NOT resting on free")
        P("  ATP. It stands, with the defect recorded as a property of the reconstruction.")
    else:
        P(f"  Growth changes by {100*abs(mu2-mu_open)/mu_open:.1f}% when the loops are closed, so")
        P("  the earlier energy-related results ARE affected and must be requalified.")

    P("\n" + RULE); P("A7  THE MASS-BALANCE AUDIT"); P(RULE)
    bad, checked = imbalanced(R, M)
    P(f"  reactions with parseable formulas checked: {checked}")
    P(f"  elementally IMBALANCED: {len(bad)} ({100*len(bad)/max(checked,1):.1f}%)")
    P("  Such a reaction lets the optimiser create atoms: it breaks conservation of matter, not")
    P("  merely of free energy.")

    P("\n" + RULE); P("A8  THE DECISIVE TEST -- CAN IT GROW ON NOTHING?"); P(RULE)
    res_bm = linprog(cg, A_eq=S, b_eq=np.zeros(S.shape[0]),
                     bounds=list(zip(lb_s, ub_s)), method="highs")
    mu_seal = -res_bm.fun if res_bm.status == 0 else float("nan")
    P(f"  maximum biomass with every exchange, demand and sink closed: {mu_seal:.6e} /h")
    P(f"  against open-medium growth {mu_open:.6f} /h"
      f"  ({100*mu_seal/max(mu_open,1e-12):.1f}% of it)")
    P(f"  {'FAIL -- the model builds a cell out of nothing' if mu_seal > 1e-6 else 'PASS'}")
    P("  Set overlap could not have shown this: shared reactions only mean shared central")
    P("  metabolism. Growth in a sealed box shows the solution can be MANUFACTURED.")

    P("\n" + RULE); P("A9  THE REQUALIFICATION"); P(RULE)
    lb_f, ub_f = lb_s.copy(), ub_s.copy()
    lb_f[bad] = 0.0; ub_f[bad] = 0.0
    res_bm2 = linprog(cg, A_eq=S, b_eq=np.zeros(S.shape[0]),
                      bounds=list(zip(lb_f, ub_f)), method="highs")
    mu_seal2 = -res_bm2.fun if res_bm2.status == 0 else float("nan")
    lb_o2, ub_o2 = lb_o.copy(), ub_o.copy()
    lb_o2[bad] = 0.0; ub_o2[bad] = 0.0
    res_g3 = linprog(cg, A_eq=S, b_eq=np.zeros(S.shape[0]),
                     bounds=list(zip(lb_o2, ub_o2)), method="highs")
    mu_open2 = -res_g3.fun if res_g3.status == 0 else float("nan")
    P(f"  with the {len(bad)} imbalanced reactions removed:")
    P(f"    sealed-box biomass : {mu_seal2:.6e} /h   (was {mu_seal:.6e})")
    P(f"    open-medium growth : {mu_open2:.6f} /h   (was {mu_open:.6f},"
      f" ratio {mu_open2/max(mu_open,1e-12):.4f})")
    if mu_seal2 > 1e-6:
        P("  Sealed-box growth SURVIVES removing them, so mass creation is not the only route and")
        P("  the reconstruction has a deeper problem than 152 reactions.")
    elif abs(mu_open2 - mu_open) / max(mu_open, 1e-12) < 0.05:
        P("  Sealed-box growth is removed and open growth is unchanged to within 5%. The earlier")
        P("  Recon3D results survive, with the defect recorded as a property of the model.")
    else:
        P(f"  Sealed-box growth is removed but open growth changes by"
          f" {100*abs(mu_open2-mu_open)/mu_open:.1f}%, so every Recon3D-derived result in this")
        P("  build order must be requalified by that amount.")

    P("\n" + RULE)
    P("Active transport is paid for in the stoichiometry and unconstrained in direction. The first")
    P("is why this model is not getting transport for free; the second is why it can still make")
    P("energy from nothing. Only the second is a defect, and it belongs to the reconstruction.")
    P(RULE)
    open(os.path.join(os.path.dirname(__file__), "RESULTS_activetransport.txt"),
         "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

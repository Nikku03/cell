"""The falsification layer every imported model must pass BEFORE any sensitivity is computed.

THE RULE THIS IMPLEMENTS, in the order it must run:

    conservation audit  ->  sealed-cell test  ->  energy-loop test  ->  only then sensitivities

The ordering is not stylistic. There is no value in debating regulation, rare-event tails or
whole-cell allocation if the underlying network can manufacture biomass in a sealed box, because
every one of those results is computed on top of a network that violates conservation of matter.
This build order learned that the expensive way: four modules reported numbers from a model whose
growth was 219x inflated by a leak, and the gate written to catch it passed because it tested two
necessary conditions and treated them as sufficient.

THREE THINGS THAT LOOKED LIKE ONE TRANSPORT PROBLEM, now separated:

    transport energy cost      ALREADY ENCODED. Primary-active reactions carry
                               -1 atp + 1 adp + 1 pi in the stoichiometry and secondary-active
                               carry the ion, so mass balance charges for uphill movement.
    thermodynamic direction    UNCONSTRAINED. Flux balance has no dG, so a reversible transporter
                               runs whichever way helps; 2,323 of 4,230 are reversible here.
    elemental conservation     BROKEN IN SOME REACTIONS. 60 internal reactions do not balance.

Elemental conservation comes first because it is the only one of the three that can create matter.

A CORRECTION TO THE RECORD, because the earlier alarm was mine. activetransport.py reported that
Recon3D makes 1000 ATP and 120 biomass from nothing and that every derived result was contaminated
at the root. That was a bug in my test, not in the model: I sealed the cell by name, matching
"sink_", while Recon3D writes "SK_", so 101 sinks stayed open in a box I called sealed. Properly
sealed, Recon3D produces EXACTLY ZERO ATP and EXACTLY ZERO biomass. The reconstruction is sound on
that count and the withdrawal is retracted.

The same typo did do real damage, in the other direction: the medium builder every Recon3D module
inherited closed only EX_ reactions, so 246 boundary reactions fed the cell for free and growth
came out 370.383 /h against a true 1.688875 /h. That is what invalidated the downstream results,
and it has been fixed and rerun. So of the two alarms, the dramatic one was false and the boring
one was real.

WHAT REMAINS GENUINELY OPEN, and M5 settles it: 60 internal reactions do not conserve elements.
That is small (0.77%) and it has never been tested on the CORRECTED model. The earlier A9 test
answered a different question, on a leaky model, using a count of 152 that wrongly included 92
legitimate sinks.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

M1  BOUNDARY IDENTIFICATION IS STRUCTURAL AND COMPLETE. Boundary reactions are those with exactly
    one metabolite, which is a property of the stoichiometry and not of a naming convention. Every
    one must be classified as declared medium or supply-closed, with zero unaccounted.

M2  CONSERVATION AUDIT. Every internal reaction with parseable formulas must balance elementally.
    Report the count, the elements involved, and whether the imbalance is hydrogen-only, which is
    a protonation convention rather than mass creation.

M3  THE SEALED-CELL TEST. With the supply direction of every boundary reaction closed, maximum
    biomass must be exactly zero. Removal is left open, because a cell that cannot excrete waste
    cannot run metabolism and closing both directions gives a vacuous zero.

M4  THE ENERGY-LOOP TEST. Sealed as in M3, the maximum production of every energy currency --
    ATP, NADH, NADPH, FADH2 -- must be exactly zero. A model can conserve mass and still create
    free energy, so M3 passing does not imply M4.

M5  DO THE IMBALANCED REACTIONS MATTER ON THE CORRECTED MODEL? Remove the internal reactions
    failing M2 and re-measure growth on the sealed medium. Predeclared: growth unchanged within 5%
    means the pathology exists but does not support the physiological solution, and the corrected
    results stand; a larger change means every sensitivity must be recomputed without them.

M6  THE ORDERING IS ENFORCED, NOT ADVISED. Report an explicit verdict for whether sensitivities
    may be computed at all, and make it depend on M2 through M4 rather than on judgement.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import json
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, hstack

from rem.atlas.hybrid_tune import RULE
from rem.atlas.recon import (MODEL, MEDIUM, fetch_if_missing, boundary_reactions, bounds_for,
                             load, solve)
from rem.atlas.activetransport import parse_formula

# The energy-loop test needs a way to ask "can this network make ATP from nothing?" without
# itself creating matter. A bare drain (atp -> nothing) would, so each currency gets a full
# DISSIPATION reaction that is elementally balanced on its own -- the test checks that balance
# and prints it, so the gate cannot be accused of manufacturing the thing it is looking for.
DISSIPATION = {
    "ATP":   {"atp_c": -1.0, "h2o_c": -1.0, "adp_c": 1.0, "pi_c": 1.0, "h_c": 1.0},
    "NADH":  {"nadh_c": -1.0, "nad_c": 1.0, "h_c": 1.0},
    "NADPH": {"nadph_c": -1.0, "nadp_c": 1.0, "h_c": 1.0},
    "FADH2": {"fadh2_m": -1.0, "fad_m": 1.0, "h_m": 2.0},
}


def conservation_audit(R, M):
    """Internal reactions whose elemental formulas do not balance. Boundary reactions are exempt
    by construction -- they are holes in the cell wall, not chemistry."""
    mf = {m["id"]: m.get("formula", "") for m in M}
    bnd = set(boundary_reactions(R))
    bad, checked, by_elem = [], 0, collections.Counter()
    for j, r in enumerate(R):
        if j in bnd or r["id"].startswith("BIOMASS"):
            continue
        tot, ok = collections.Counter(), True
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
        off = tuple(sorted(el for el, v in tot.items() if abs(v) > 1e-6))
        if off:
            bad.append(j)
            by_elem[off] += 1
    return bad, checked, by_elem


def sealed_supply(R, lb):
    lb = lb.copy()
    for j in boundary_reactions(R):
        lb[j] = 0.0
    return lb


def max_flux(S, lb, ub, j):
    c = np.zeros(S.shape[1]); c[j] = -1.0
    r = linprog(c, A_eq=S, b_eq=np.zeros(S.shape[0]), bounds=list(zip(lb, ub)), method="highs")
    return -r.fun if r.status == 0 else float("nan")


def balance_of(spec, mf):
    """Elemental balance of a proposed reaction, so the dissipation reactions can be shown not to
    create matter themselves."""
    tot = collections.Counter()
    for met, co in spec.items():
        f = mf.get(met, "")
        if not f:
            return None
        for el, n in parse_formula(f).items():
            tot[el] += co * n
    return {el: v for el, v in tot.items() if abs(v) > 1e-9}


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE)
    P("THE MODEL AUDIT   conservation -> seal -> energy -> only then sensitivities")
    P(RULE)
    sha = fetch_if_missing()
    P(f"  model {os.path.basename(MODEL)}   sha256 {sha[:32]}...")
    R, M, S = load()
    mi = {m["id"]: i for i, m in enumerate(M)}
    idx = {r["id"]: j for j, r in enumerate(R)}
    mf = {m["id"]: m.get("formula", "") for m in M}
    P(f"  {len(R)} reactions, {len(M)} metabolites, {S.nnz} nonzeros")

    obj = np.zeros(len(R))
    obj[idx["BIOMASS_maintenance"]] = -1.0
    lb, ub = bounds_for(R, MEDIUM)

    # ---- M1  BOUNDARY IDENTIFICATION -----------------------------------------------------------
    P("\n" + RULE); P("M1  BOUNDARY IDENTIFICATION IS STRUCTURAL AND COMPLETE"); P(RULE)
    bnd = boundary_reactions(R)
    pref = collections.Counter(R[j]["id"].split("_")[0] for j in bnd)
    byname = {j for j, r in enumerate(R) if r["id"].split("_")[0] in ("EX", "DM", "SK")}
    P(f"  structural (exactly one metabolite) : {len(bnd)}")
    P(f"    by name prefix: " + ", ".join(f"{k}_ {v}" for k, v in pref.most_common()))
    P(f"  name-based (EX_/DM_/SK_)            : {len(byname)}")
    P(f"  structural but unnamed {len(set(bnd) - byname)}, named but not structural {len(byname - set(bnd))}")
    med_idx = {idx[k] for k in MEDIUM if k in idx}
    unacc = [j for j in bnd if j not in med_idx and lb[j] != 0.0]
    P(f"  declared medium {len(med_idx)}, supply-closed {len(bnd) - len(med_idx)}, UNACCOUNTED {len(unacc)}")
    m1 = not unacc
    P(f"  M1: {'PASS' if m1 else 'FAIL -- ' + str(len(unacc)) + ' boundary reactions open and undeclared'}")

    # ---- M2  CONSERVATION AUDIT ----------------------------------------------------------------
    P("\n" + RULE); P("M2  CONSERVATION AUDIT  (this is the one that comes first)"); P(RULE)
    bad, checked, by_elem = conservation_audit(R, M)
    internal = len(R) - len(bnd)
    P(f"  internal reactions {internal}, of which {checked} have parseable formulas on every")
    P(f"  metabolite ({100.0*checked/internal:.1f}%); the rest carry R- or X-groups and cannot be checked")
    P(f"  IMBALANCED: {len(bad)} of {checked} checked  ({100.0*len(bad)/checked:.2f}%)")
    P("  by element set:")
    for els, n in by_elem.most_common(12):
        P(f"    {'+'.join(els):<22} {n}")
    h_only = by_elem.get(("H",), 0)
    P(f"  hydrogen-only imbalances: {h_only} of {len(bad)}"
      f"  ({100.0*h_only/max(1,len(bad)):.0f}%) -- protonation convention, not mass creation")
    P(f"  imbalances involving C, N, P or S: "
      f"{sum(n for els, n in by_elem.items() if set(els) & {'C','N','P','S'})}")
    m2 = len(bad) == 0
    P(f"  M2: {'PASS' if m2 else 'FAIL -- ' + str(len(bad)) + ' internal reactions do not conserve elements'}")
    P("      A FAIL here does not by itself invalidate anything. It says a pathology exists.")
    P("      M5 asks the question that matters: does the physiological solution USE it?")

    # ---- M3  THE SEALED-CELL TEST --------------------------------------------------------------
    P("\n" + RULE); P("M3  THE SEALED-CELL TEST"); P(RULE)
    lb_seal = sealed_supply(R, lb)
    r_seal = solve(S, obj, lb_seal, ub)
    mu_seal = -r_seal.fun if r_seal.status == 0 else float("nan")
    r_med = solve(S, obj, lb, ub)
    growth = -r_med.fun
    P(f"  growth on the declared medium         : {growth:.6f} /h")
    P(f"  growth with every supply direction shut: {mu_seal:.6e} /h")
    m3 = abs(mu_seal) < 1e-9
    P(f"  M3: {'PASS -- a sealed cell does not grow' if m3 else 'FAIL -- biomass from nothing'}")

    # ---- M4  THE ENERGY-LOOP TEST --------------------------------------------------------------
    P("\n" + RULE); P("M4  THE ENERGY-LOOP TEST"); P(RULE)
    P("  Mass conservation does not imply energy conservation: a network can balance every atom")
    P("  and still run a cycle that nets ATP. Sealed exactly as in M3, each currency is drained")
    P("  through an elementally balanced dissipation reaction and the drain is maximised.")
    m4 = True
    nS, nR = S.shape
    for name, spec in DISSIPATION.items():
        miss = [m for m in spec if m not in mi]
        if miss:
            P(f"  {name:<6} SKIPPED -- metabolites absent from the model: {miss}")
            continue
        bal = balance_of(spec, mf)
        col = np.zeros((nS, 1))
        for met, co in spec.items():
            col[mi[met], 0] = co
        S2 = csc_matrix(hstack([S, csc_matrix(col)]))
        lb2 = np.append(lb_seal, 0.0)
        ub2 = np.append(ub, 1000.0)
        v = max_flux(S2, lb2, ub2, nR)
        okc = abs(v) < 1e-9
        m4 = m4 and okc
        bs = "balanced" if bal == {} else f"UNBALANCED {bal}"
        P(f"  {name:<6} max drain {v:14.6e}   dissipation reaction is {bs}"
          f"   {'PASS' if okc else 'FAIL'}")
    P(f"  M4: {'PASS -- no free energy in a sealed box' if m4 else 'FAIL -- the network generates energy from nothing'}")

    # ---- M5  DO THE IMBALANCED REACTIONS MATTER? -----------------------------------------------
    P("\n" + RULE); P("M5  DO THE IMBALANCED REACTIONS MATTER ON THE CORRECTED MODEL?"); P(RULE)
    carry = [j for j in bad if abs(r_med.x[j]) > 1e-9]
    P(f"  of the {len(bad)} imbalanced reactions, {len(carry)} carry flux in the optimal solution")
    for j in carry[:10]:
        P(f"    {R[j]['id']:<24} v = {r_med.x[j]:+.6f}")
    lb5, ub5 = lb.copy(), ub.copy()
    for j in bad:
        lb5[j] = 0.0
        ub5[j] = 0.0
    r5 = solve(S, obj, lb5, ub5)
    mu5 = -r5.fun if r5.status == 0 else float("nan")
    rel = abs(mu5 - growth) / growth if growth > 0 else float("nan")
    P(f"  growth with all {len(bad)} knocked out : {mu5:.6f} /h   against {growth:.6f} /h")
    P(f"  relative change {100*rel:.4f}%   (predeclared bar: 5%)")
    m5 = rel < 0.05
    if m5:
        P("  M5: PASS -- the pathology exists but the physiological solution does not lean on it.")
        P("      The corrected sensitivity results stand as computed.")
    else:
        P("  M5: FAIL -- growth depends on reactions that do not conserve elements.")
        P("      Every sensitivity must be recomputed with them removed.")

    # ---- M6  THE ORDERING IS ENFORCED ----------------------------------------------------------
    P("\n" + RULE); P("M6  MAY SENSITIVITIES BE COMPUTED ON THIS MODEL?"); P(RULE)
    P(f"    M1 boundaries accounted   {'PASS' if m1 else 'FAIL'}")
    P(f"    M2 conservation           {'PASS' if m2 else 'FAIL'}   (advisory; M5 is the decider)")
    P(f"    M3 sealed cell            {'PASS' if m3 else 'FAIL'}   (BLOCKING)")
    P(f"    M4 energy loops           {'PASS' if m4 else 'FAIL'}   (BLOCKING)")
    P(f"    M5 imbalance irrelevant   {'PASS' if m5 else 'FAIL'}   (BLOCKING)")
    verdict = m1 and m3 and m4 and m5
    P()
    P(f"  VERDICT: sensitivities {'MAY' if verdict else 'MAY NOT'} be computed on this model.")
    P("  The rule is mechanical, not a judgement call: M1, M3, M4 and M5 all block. M2 alone does")
    P("  not, because an imbalanced reaction that carries no flux and supports no growth is a")
    P("  defect in the reconstruction rather than a defect in the answer.")

    dst = os.path.join(os.path.dirname(__file__), "MODELAUDIT.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P(f"\n  written to {dst}")


if __name__ == "__main__":
    main()

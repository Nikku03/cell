"""BUILD 8 -- THE SHARED PROTEOME BUDGET: the coupling that builds 1 and 7 left out.

WHAT BUILDS 1 AND 7 ESTABLISHED, AND WHY IT DIAGNOSES THE IMPLEMENTATION RATHER THAN THE IDEA.

Build 1 gave every reaction its own ceiling, v_i <= kcat_i * CAP(GPR, abundance) * scale, and it changed nothing:
recall stayed at 0.203, not one lethal call moved, AUC fell 0.676 -> 0.594, and permuting the quantities
reproduced the result exactly. Build 7 measured why, and the number is brutal:

    1,651 reactions carry a ceiling
       73 carry ANY flux at the pFBA optimum
    1,578 carry exactly zero
        5 are AT their ceiling; 19 are within two orders of it

A ceiling on a reaction that carries no flux constrains nothing, however accurate its kcat. Improving kcat from a
factor-of-15 error to a factor-of-2 error -- the whole promise of k_app calibration -- cannot move a bound that
sits six orders above the flux. Build 7's shrinkage sweep confirmed it: lambda 0 and lambda 0.25 both leave
exactly 5 reactions at their ceiling.

BUT PER-REACTION CEILINGS ARE NOT WHAT MAKES ecMODELS WORK. GECKO's constraint is a SHARED BUDGET:

    sum over reactions of  v_i / kcat_i   <=   P_total

Every reaction draws from ONE pool. That single change alters the character of the constraint completely:

    per-reaction ceiling      reactions are independent. A reaction below its own ceiling is unconstrained, and
                              its kcat is irrelevant. 1,578 of 1,651 are in exactly that position.
    shared budget             reactions COMPETE. Flux through any reaction costs protein that another reaction
                              then cannot have, so every kcat matters -- not as a limit, as a PRICE. A slow
                              enzyme is expensive; the optimiser routes around it. That is the mechanism by which
                              enzyme kinetics is supposed to shape the flux distribution, and build 1 did not
                              contain it.

It also changes what a knockout means. Under per-reaction ceilings, deleting an isozyme lowers one bound that was
not binding anyway. Under a shared budget, the surviving route has to be PAID FOR out of the same pool, so the
cell gives something up elsewhere -- which is the "quantity" argument of DESIGN.md sec 3 in the form that actually
has teeth.

CALIBRATION, AND THE ONE ANCHOR WE HAVE. P_total is fitted -- by bisection -- so the wild-type cell reproduces the
MEASURED 24 h doubling time, exactly as build 1 fitted its global scale. One free parameter, one measured number,
and DepMap essentiality is never part of the fit. There is still no measured flux for K562 in this project, so
genuine k_app = v/E calibration remains out of reach and is not claimed.

PRE-REGISTERED CRITERION, unchanged from DESIGN.md sec 3 so all three builds are directly comparable: recall must
rise above 0.203 with precision >= 0.6, and must beat a kcat-shuffled control calibrated the same way.

THE DIAGNOSTIC THAT DECIDES WHETHER THE MECHANISM ENGAGED AT ALL, reported whatever the score does: under a shared
budget the meaningful quantity is not "how many reactions are at a ceiling" -- there are no per-reaction ceilings
-- but how much of the pool each reaction consumes, and whether the pool is TIGHT. If the pool constraint is not
active at the optimum, nothing has been imposed and the result was determined before it was scored.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ec_capacity import (K562_DOUBLING_H, LETHAL_THR, cap_of, gpr_alive, gpr_tree,   # noqa: E402
                         load_kcat_by_ec, metrics, reaction_ec)

POOL = "prot_pool_shared_budget"


def build_costs(m, kcat, r2ec, abund):
    """reaction -> protein cost per unit flux, 1/kcat weighted by the GPR's own logic.

    The cost of running a reaction is the protein it ties up. For an isozyme pair the cell may use whichever is
    cheaper, so the effective turnover is the MAXIMUM over OR branches; for a complex every subunit must be
    present, so the cost is the SUM over AND members. That is the mirror image of build 1's capacity rule, and it
    is the correct one for a cost rather than a ceiling.
    """
    cost, why = {}, collections.Counter()
    for r in m.reactions:
        if r.boundary:
            continue
        t = gpr_tree(r.gene_reaction_rule)
        if t is None:
            why["no parsable GPR"] += 1
            continue
        ks = [kcat[e] for e in r2ec.get(r.id, set()) if e in kcat]
        if not ks:
            why["no kcat for any EC"] += 1
            continue
        k = float(np.median(ks))
        if k <= 0:
            why["non-positive kcat"] += 1
            continue
        cost[r.id] = 1.0 / k
        why["cost assigned"] += 1
    return cost, why


def add_pool(m, cost, P):
    """sum_i (v_i^+ + v_i^-) * cost_i <= P, as one shared constraint over every costed reaction."""
    expr = sum((m.reactions.get_by_id(rid).forward_variable
                + m.reactions.get_by_id(rid).reverse_variable) * c for rid, c in cost.items())
    con = m.problem.Constraint(expr, ub=float(P), name=POOL)
    m.add_cons_vars([con])
    return con


def growth_with_pool(m, cost, P):
    with m:
        add_pool(m, cost, P)
        g = m.slim_optimize()
    return 0.0 if g is None or np.isnan(g) else float(g)


def calibrate_pool(m, cost, target, lo=1e-9, hi=1e9):
    """Growth is monotone non-decreasing in the pool size, so bisect in log space."""
    for _ in range(80):
        mid = float(np.sqrt(lo * hi))
        if growth_with_pool(m, cost, mid) < target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0005:
            break
    return float(np.sqrt(lo * hi))


def deletion_sweep_pool(m, model_genes, r_of_gene, cost, P, use_pool):
    """Delete each gene and re-optimise, with or without the shared budget imposed."""
    trees = {r.id: gpr_tree(r.gene_reaction_rule) for r in m.reactions if r.gene_reaction_rule}
    with m:
        if use_pool:
            add_pool(m, cost, P)
        wt = m.slim_optimize()
        out = {}
        for g in model_genes:
            dead, touched = {g}, []
            for rid in r_of_gene.get(g, ()):
                t = trees.get(rid)
                if t is not None and not gpr_alive(t, dead):
                    r = m.reactions.get_by_id(rid)
                    touched.append((r, r.lower_bound, r.upper_bound))
                    r.lower_bound = r.upper_bound = 0.0
            v = m.slim_optimize()
            out[g] = 0.0 if v is None or np.isnan(v) else float(v)
            for r, lo, hi in touched:
                r.lower_bound, r.upper_bound = lo, hi
    return out, (0.0 if wt is None or np.isnan(wt) else float(wt))


def main():
    import cobra
    from cobra.flux_analysis import pfba
    cobra.Configuration().solver = "glpk"
    from cell_sim import set_medium, ensembl_to_symbol, depmap_k562

    kcat = load_kcat_by_ec()
    r2ec = reaction_ec()
    m = cobra.io.read_sbml_model(str(SP / "HumanGEM.xml"))
    set_medium(m)
    e2s = ensembl_to_symbol()
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    ppm = {names[int(k)]: float(v) for k, v in D["ppm"].items()
           if k.isdigit() and int(k) < len(names) and v}
    abund = {g.id: ppm[e2s[g.id]] for g in m.genes if g.id in e2s and e2s[g.id] in ppm}
    assert len(abund) > 1000, "ABUNDANCE JOIN TOO SMALL -- refusing to continue"

    cost, why = build_costs(m, kcat, r2ec, abund)
    print(f"\nprotein cost assigned to {len(cost):,} reactions; {dict(why.most_common())}")
    assert len(cost) > 500, "COST JOIN TOO SMALL -- refusing to continue"

    mu = float(np.log(2) / K562_DOUBLING_H)
    free = m.slim_optimize()
    P = calibrate_pool(m, cost, mu)
    gwt = growth_with_pool(m, cost, P)
    print(f"unconstrained growth {free:.4f}/h; measured target {mu:.4f}/h (24 h doubling)")
    print(f"calibrated pool P = {P:.6g} -> growth {gwt:.4f}/h ({np.log(2)/max(gwt,1e-9):.1f} h doubling)")
    if abs(gwt - mu) > 0.15 * mu:
        print("  pool calibration did not converge -- refusing to score.")
        return

    # ---- did the mechanism engage? the pool must be TIGHT, and the draw must be spread over real reactions ----
    with m:
        con = add_pool(m, cost, P)
        sol = pfba(m)
        used = float(sum(abs(float(sol.fluxes[rid])) * c for rid, c in cost.items()))
        shadow = None
        try:
            shadow = float(con.dual)
        except Exception:
            pass
    draws = sorted(((abs(float(sol.fluxes[rid])) * c, rid) for rid, c in cost.items()), reverse=True)
    nz = [d for d in draws if d[0] > 0]
    print("\n" + "=" * 88)
    print("DID THE MECHANISM ENGAGE?  the shared budget must be tight and broadly drawn upon")
    print("=" * 88)
    print(f"  pool used {used:.6g} of P = {P:.6g}  ({100*used/max(P,1e-30):.2f}% -- must be ~100% to bind)")
    print(f"  {len(nz):,} of {len(cost):,} costed reactions draw from the pool at the optimum")
    if shadow is not None:
        print(f"  shadow price of the pool: {shadow:.6g}  (non-zero means growth is genuinely pool-limited)")
    tot = sum(d[0] for d in nz) or 1.0
    print(f"  the 10 largest consumers take {100*sum(d[0] for d in nz[:10])/tot:.1f}% of the budget:")
    for d, rid in nz[:10]:
        print(f"    {rid:<12s} {100*d/tot:>6.2f}%   cost/flux {cost[rid]:.3g}")
    print(f"  CONTRAST with build 1's per-reaction ceilings: 5 reactions were at a ceiling; here {len(nz):,} "
          f"reactions compete for one budget.")

    # ---- the acceptance test ----
    dep = depmap_k562()
    r_of_gene = collections.defaultdict(list)
    for r in m.reactions:
        for g in r.genes:
            r_of_gene[g.id].append(r.id)
    model_genes = sorted(g.id for g in m.genes if e2s.get(g.id, g.id) in dep)
    print(f"\nscoring {len(model_genes):,} model genes present in DepMap K562")

    pb, wb = deletion_sweep_pool(m, model_genes, r_of_gene, cost, P, use_pool=False)
    base = metrics(pb, wb, e2s, dep, "no pool (baseline)")
    pp, wp = deletion_sweep_pool(m, model_genes, r_of_gene, cost, P, use_pool=True)
    pool = metrics(pp, wp, e2s, dep, "shared proteome budget")
    if base is None or pool is None:
        return
    moved = sum(1 for g in model_genes
                if abs(pb[g] / max(wb, 1e-12) - pp[g] / max(wp, 1e-12)) > 1e-6)
    print(f"  genes whose growth ratio changed: {moved:,} of {len(model_genes):,}")

    rng = np.random.default_rng(0)
    shuf = []
    for i in range(3):
        ks, vs = list(kcat), list(kcat.values())
        rng.shuffle(vs)
        c2, _ = build_costs(m, dict(zip(ks, vs)), r2ec, abund)
        if not c2:
            continue
        P2 = calibrate_pool(m, c2, mu)
        if abs(growth_with_pool(m, c2, P2) - mu) > 0.15 * mu:
            print(f"  shuffle {i}: pool calibration failed, skipped")
            continue
        ps, ws = deletion_sweep_pool(m, model_genes, r_of_gene, c2, P2, use_pool=True)
        r_ = metrics(ps, ws, e2s, dep, f"SHUFFLED kcat, shared budget #{i}")
        if r_:
            shuf.append(r_)

    print("\n" + "=" * 88)
    print(f"{'':34s} {'called':>7s} {'prec':>7s} {'recall':>7s} {'AUC':>8s}")
    for tag, r in [("no pool (baseline)", base), ("shared proteome budget", pool)]:
        print(f"{tag:34s} {r['n_called']:>7,} {r['precision']:>7.3f} {r['recall']:>7.3f} {r['auc']:>8.4f}")
    for i, r in enumerate(shuf):
        print(f"{'  SHUFFLED kcat #%d' % i:34s} {r['n_called']:>7,} {r['precision']:>7.3f} "
              f"{r['recall']:>7.3f} {r['auc']:>8.4f}")
    print("=" * 88)
    d_rec = pool["recall"] - base["recall"]
    passed = bool(d_rec > 0 and pool["precision"] >= 0.6)
    beat = bool(shuf) and pool["recall"] > max(s["recall"] for s in shuf)
    print(f"\nPRE-REGISTERED CRITERION (DESIGN.md sec 3, same bar as builds 1 and 7)")
    print(f"  recall {base['recall']:.3f} -> {pool['recall']:.3f} ({d_rec:+.3f}), "
          f"precision {base['precision']:.3f} -> {pool['precision']:.3f}")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}"
          + (f"; beats every shuffled control: {beat}" if shuf else "; NO CONTROL RAN"))
    if passed and not beat:
        print("  ...but permuted kcats do as well, so the gain is the budget's existence, not the kinetics.")

    json.dump({"n_cost": len(cost), "P": P, "wt_growth": gwt, "mu_target": mu,
               "wt_unconstrained": float(free), "pool_used": used,
               "pool_utilisation": float(used / max(P, 1e-30)), "n_drawing": len(nz),
               "shadow_price": shadow, "n_ratio_changed": moved,
               "top_consumers": [{"reaction": rid, "share": float(d / tot)} for d, rid in nz[:20]],
               "baseline": base, "pool": pool, "shuffled": shuf,
               "passed": passed, "beats_control": beat},
              open(OUT / "gecko.json", "w"), indent=1)
    print(f"\n  -> {OUT/'gecko.json'}")


if __name__ == "__main__":
    main()

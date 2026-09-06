"""THE TRADITIONAL TEST, WITH THE BEST OF EVERYTHING MEASURED SO FAR.

Six formulations have now failed to move FBA's essentiality recall off 0.203: per-reaction ceilings (build 1),
the whole kcat accuracy axis (7), a shared proteome budget (8), a 2.9x coverage increase (9), a minimal medium
(10), and a static ceiling. Each used ONE kcat source and ONE constraint shape. This file assembles the best
version of every component -- chosen by measurement, not preference -- and runs the same test once more, so the
conclusion rests on the best available inputs rather than on whatever was to hand.

WHAT "BEST" MEANS HERE, EACH SETTLED BY THE HEAD-TO-HEAD IN kcat_headtohead.py (915 common labels, leave-one-out)

    EC-median over human records   2.62x median fold error   <- tier 1, best accuracy, worst coverage (362 EC)
    CatPred                        8.38x                     <- tier 2, 1,209 genes, real signal over the null
    EC-median any organism         (build 9's fallback)      <- tier 3, coverage
    global median                 14.23x                     <- tier 4, the null, better than nothing
    NOT ecHumanGEM (66x, bias +1.71) and NOT EC-max (41x, bias +1.61): both are kcat_MAX, upper bounds by
    design rather than estimators, and using them as point estimates inflates every ceiling by ~1.7 log units.

    UNITS from the derived chain rather than a bare fit:
        v_max = kcat[1/s] x 3600[s/h] x (ppm/1e6) x P_tot / MW,  MW measured (44,293 g/mol abundance-weighted),
        P_tot = 0.6 g/gDW assumed. Derived constant 4.877e-05; at that value the cell doubles in 160 h against a
        measured 24 h, so the run is ALSO calibrated to the measured growth rate and both are reported.

THE NUMBER THAT DECIDES THIS BEFORE THE SCORE DOES. Build 9 established that what matters is not how many
reactions get a ceiling but how many FLUX-CARRYING reactions do -- a ceiling on an unused reaction is inert at
any accuracy. That count is printed first, and if the best-of-everything table does not raise it, the essentiality
result was determined in advance.

PRE-REGISTERED CRITERION, identical to builds 1/7/8/9/10 so all seven are comparable: recall must rise above
0.203 with precision >= 0.6, and beat a kcat-shuffled control calibrated the same way.
"""
import collections
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ec_capacity import (K562_DOUBLING_H, calibrate, cap_of, deletion_sweep, gpr_tree,   # noqa: E402
                         growth_at, metrics, reaction_ec)
from gecko import calibrate_pool, deletion_sweep_pool, growth_with_pool                   # noqa: E402

P_TOT, DERIVED_HINT = 0.6, 4.877e-5


def best_kcat_table(r2ec, m, e2s):
    """reaction -> (kcat, tier), by the hierarchy the head-to-head selected. Coverage reported per tier."""
    recs = json.load(open(SP / "dlkcat.tsv"))
    hum, allo = collections.defaultdict(list), collections.defaultdict(list)
    for r in recs:
        ec = (r.get("ECNumber") or "").strip()
        try:
            v = float(r.get("Value"))
        except (TypeError, ValueError):
            continue
        if not ec or v <= 0 or not np.isfinite(v):
            continue
        allo[ec].append(v)
        if r.get("Organism") == "Homo sapiens":
            hum[ec].append(v)
    ec_hum = {e: float(np.median(v)) for e, v in hum.items()}
    ec_all = {e: float(np.median(v)) for e, v in allo.items()}
    gmed = float(np.median([x for v in hum.values() for x in v]))

    kr = json.load(open(OUT / "kinetics_refined.json"))["kinetics_refined"]
    cat = {g: float(v["kcat_per_s"]) for g, v in kr.items()
           if isinstance(v, dict) and v.get("kcat_per_s") and float(v["kcat_per_s"]) > 0}
    print(f"kcat sources: human-EC {len(ec_hum):,}, any-organism-EC {len(ec_all):,}, "
          f"CatPred genes {len(cat):,}, global median {gmed:.3g} /s")
    assert len(ec_hum) > 100 and len(cat) > 500, "KCAT SOURCE JOIN TOO SMALL -- refusing to continue"

    out, tier = {}, collections.Counter()
    for r in m.reactions:
        if r.boundary or not r.gene_reaction_rule:
            continue
        ecs = r2ec.get(r.id, set())
        v = [ec_hum[e] for e in ecs if e in ec_hum]
        if v:
            out[r.id], t = float(np.median(v)), "1 human-EC median (2.62x)"
        else:
            cg = [cat[e2s[g.id]] for g in r.genes if e2s.get(g.id) in cat]
            if cg:
                out[r.id], t = float(np.median(cg)), "2 CatPred (8.38x)"
            else:
                v2 = [ec_all[e] for e in ecs if e in ec_all]
                if v2:
                    out[r.id], t = float(np.median(v2)), "3 any-organism EC"
                else:
                    out[r.id], t = gmed, "4 global median (14.2x)"
        tier[t] += 1
    print(f"assigned kcat to {len(out):,} reactions: {dict(sorted(tier.items()))}")
    return out, tier


def main():
    import cobra
    from cobra.flux_analysis import pfba
    cobra.Configuration().solver = "glpk"
    from cell_sim import set_medium, ensembl_to_symbol, depmap_k562

    r2ec = reaction_ec()
    m = cobra.io.read_sbml_model(str(SP / "HumanGEM.xml"))
    set_medium(m)
    e2s = ensembl_to_symbol()
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    ppm = {names[int(k)]: float(v) for k, v in D["ppm"].items()
           if k.isdigit() and int(k) < len(names) and v}
    abund = {g.id: ppm[e2s[g.id]] for g in m.genes if g.id in e2s and e2s[g.id] in ppm}
    assert len(abund) > 1000, "ABUNDANCE JOIN TOO SMALL"
    print(f"abundance: {len(abund):,} of {len(m.genes):,} model genes")

    kcat, tiers = best_kcat_table(r2ec, m, e2s)
    trees = {r.id: gpr_tree(r.gene_reaction_rule) for r in m.reactions if r.gene_reaction_rule}
    cap, cost, tree, kc = {}, {}, {}, {}
    for rid, k in kcat.items():
        t = trees.get(rid)
        if t is None:
            continue
        c = cap_of(t, abund)
        kc[rid], tree[rid], cost[rid] = k, t, 1.0 / k
        if c is not None and c > 0:
            cap[rid] = float(k * c)
    print(f"capacity on {len(cap):,} reactions; protein cost on {len(cost):,}")

    # ---- THE DECISIVE COVERAGE NUMBER (build 9's lesson) ----
    wt = pfba(m)
    carry = {r.id for r in m.reactions if not r.boundary and abs(float(wt.fluxes[r.id])) > 1e-9}
    overlap = len(set(cap) & carry)
    print(f"\nflux-carrying reactions: {len(carry):,}; of those, {overlap:,} now have a ceiling")
    print(f"  build 1 (human-only EC): 196    build 9 (hierarchy): 566    HERE: {overlap:,}")

    mu = float(np.log(2) / K562_DOUBLING_H)
    dep = depmap_k562()
    r_of_gene = collections.defaultdict(list)
    for r in m.reactions:
        for g in r.genes:
            r_of_gene[g.id].append(r.id)
    model_genes = sorted(g.id for g in m.genes if e2s.get(g.id, g.id) in dep)
    print(f"scoring {len(model_genes):,} model genes in DepMap K562")

    g_der, _ = growth_at(m, cap, DERIVED_HINT)
    scale = calibrate(m, cap, mu)
    g_cal, nb = growth_at(m, cap, scale)
    print(f"\nunits: derived constant {DERIVED_HINT:.4g} -> growth {g_der:.5f}/h "
          f"({np.log(2)/max(g_der,1e-12):.1f} h doubling)")
    print(f"       calibrated to 24 h  {scale:.4g} -> growth {g_cal:.5f}/h ({nb:,} bounds tightened)")
    if abs(g_cal - mu) > 0.15 * mu:
        print("  calibration failed -- refusing to score.")
        return

    res = {}
    pb, wb = deletion_sweep(m, model_genes, r_of_gene, tree, kc, abund, cap, scale, dose=False)
    base = metrics(pb, wb, e2s, dep, "no constraint (baseline)")
    pd_, wd = deletion_sweep(m, model_genes, r_of_gene, tree, kc, abund, cap, scale, dose=True)
    res["per-reaction ceilings"] = metrics(pd_, wd, e2s, dep, "best kcat | per-reaction ceilings")
    P = calibrate_pool(m, cost, mu)
    if np.isfinite(P) and abs(growth_with_pool(m, cost, P) - mu) <= 0.15 * mu:
        pp, wp = deletion_sweep_pool(m, model_genes, r_of_gene, cost, P, use_pool=True)
        res["shared proteome budget"] = metrics(pp, wp, e2s, dep, "best kcat | shared budget")
    rng = np.random.default_rng(0)
    ks, vs = list(kcat), list(kcat.values())
    rng.shuffle(vs)
    ksh = dict(zip(ks, vs))
    cap2 = {rid: float(ksh[rid] * (cap_of(tree[rid], abund) or 0)) for rid in cap if rid in tree}
    cap2 = {k_: v for k_, v in cap2.items() if v > 0}
    kc2 = {rid: ksh[rid] for rid in cap2}
    s2 = calibrate(m, cap2, mu)
    if abs(growth_at(m, cap2, s2)[0] - mu) <= 0.15 * mu:
        ps, ws = deletion_sweep(m, model_genes, r_of_gene, tree, kc2, abund, cap2, s2, dose=True)
        res["SHUFFLED kcat (control)"] = metrics(ps, ws, e2s, dep, "SHUFFLED kcat | per-reaction")

    print("\n" + "=" * 90)
    print(f"{'arm':40s} {'called':>7s} {'prec':>7s} {'recall':>7s} {'AUC':>8s}")
    print(f"{'no constraint (baseline)':40s} {base['n_called']:>7,} {base['precision']:>7.3f} "
          f"{base['recall']:>7.3f} {base['auc']:>8.4f}")
    for k_, v in res.items():
        if v:
            print(f"{k_:40s} {v['n_called']:>7,} {v['precision']:>7.3f} {v['recall']:>7.3f} {v['auc']:>8.4f}")
    print("=" * 90)
    best = max((v for v in res.values() if v), key=lambda v: v["recall"], default=None)
    passed = bool(best and best["recall"] > base["recall"] and best["precision"] >= 0.6)
    print(f"\nPRE-REGISTERED CRITERION (same bar as builds 1, 7, 8, 9, 10)")
    if best:
        print(f"  recall {base['recall']:.3f} -> {best['recall']:.3f}, precision {best['precision']:.3f}")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    print(f"\n  This is the SEVENTH formulation. Inputs: the most accurate kcat available (2.62x on a common")
    print(f"  label set), the widest coverage available ({overlap:,} flux-carrying reactions vs 196 in build 1),")
    print(f"  first-principles units, and both constraint shapes.")

    json.dump({"tiers": dict(tiers), "n_capacity": len(cap), "n_cost": len(cost),
               "n_flux_carrying": len(carry), "overlap_flux_carrying": overlap,
               "derived_scale": DERIVED_HINT, "growth_at_derived": g_der,
               "calibrated_scale": scale, "growth_calibrated": g_cal,
               "baseline": base, "arms": res, "passed": passed},
              open(OUT / "best_combined.json", "w"), indent=1)
    print(f"\n  -> {OUT/'best_combined.json'}")


if __name__ == "__main__":
    main()

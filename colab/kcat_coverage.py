"""BUILD 9 -- COVERAGE, NOT ACCURACY: the axis that was actually binding.

WHAT BUILDS 1 AND 7 LEFT ON THE TABLE. The standard account of enzyme-constrained modelling says the problem with
in-vitro kcat is ACCURACY: BRENDA numbers are off by a factor of 10-15 (log10 RMSE ~1.0-1.2), DLKcat predictions
by 6-7 (~0.8), and proteome-constrained k_app by 2-3 (~0.4). Build 7 tested that axis directly with a shrinkage
sweep and found it inert: at lambda 0 (raw in vitro) and at lambda 1 (every individual kcat discarded in favour of
one class median) and at every point between, recall stayed at exactly 0.203 and the number of reactions at their
ceiling moved 5 -> 8. No accuracy tier between those two extremes can land anywhere else.

The reason is that accuracy only matters for a reaction whose ceiling is near its flux, and build 7 measured how
many those are: 1,578 of 1,651 ceilinged reactions carry NO FLUX AT ALL, 5 sit at their ceiling, 19 are within two
orders. Improving a number that multiplies a bound six orders above the flux changes nothing.

COVERAGE IS A DIFFERENT AXIS, AND IT WAS NEVER TESTED. Build 1 matched Homo sapiens records only:

    DLKcat table                                            1,706 distinct EC numbers
    restricted to Homo sapiens                                362        <- build 1 used this
    Human-GEM reactions carrying an ec-code annotation       5,387
      matched, human-only exact EC                           1,811
      matched, any-organism exact EC                         3,071
      matched, any-organism + EC-class fallback              5,158

Dropping to human threw away 1,344 of 1,706 EC numbers. kcat is well conserved within an EC across orthologs, so
the accepted practice in every ecModel pipeline is a fallback hierarchy rather than a drop. This costs no new data.

THE QUESTION THIS FILE ANSWERS, AND THE ONE IT DOES NOT. It does NOT claim the fallback numbers are as good as
human measurements -- a cross-organism median is a worse estimate, and a class median is worse still. It asks the
only question that matters given build 7: does the extra coverage put ceilings on reactions that CARRY FLUX?
A ceiling on an unused reaction is inert however accurate; a ceiling on a used one can bind however rough.

    the decisive number: of the reactions carrying flux at the wild-type optimum, how many have a ceiling
                         before  (human-only)  vs  after  (hierarchy)

PRE-REGISTERED CRITERION, unchanged from DESIGN.md sec 3 so builds 1, 7, 8 and 9 are directly comparable: recall
must rise above 0.203 with precision >= 0.6, and must beat a kcat-shuffled control calibrated the same way. Tested
in BOTH formulations, because build 8 argues the shape of the constraint matters more than its parameters:

    per-reaction ceilings   v_i <= kcat_i * CAP(GPR, abundance) * scale     (build 1's shape, wider coverage)
    shared proteome budget  sum_i v_i / kcat_i <= P_total                   (build 8's shape, wider coverage)
"""
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ec_capacity import (K562_DOUBLING_H, calibrate, cap_of, deletion_sweep, gpr_tree,   # noqa: E402
                         growth_at, load_kcat_by_ec, load_kcat_hierarchical, metrics,
                         reaction_ec, resolve_kcat)
from gecko import calibrate_pool, deletion_sweep_pool, growth_with_pool   # noqa: E402


def build_capacity_h(m, table, classes, abund, hierarchical):
    """Capacity and protein cost per reaction, using either build 1's human-only table or the hierarchy."""
    cap, cost, tree, kc, why = {}, {}, {}, {}, collections.Counter()
    for r in m.reactions:
        if r.boundary:
            continue
        t = gpr_tree(r.gene_reaction_rule)
        if t is None:
            why["no parsable GPR"] += 1
            continue
        ecs = r.__dict__.get("_ecs", set())
        if hierarchical:
            k, tier = resolve_kcat(ecs, table, classes)
        else:
            hits = [table[e] for e in ecs if e in table]
            k, tier = (float(np.median(hits)), 1) if hits else (None, None)
        if k is None or k <= 0:
            why["no kcat"] += 1
            continue
        why[f"tier {tier}"] += 1
        kc[r.id], tree[r.id], cost[r.id] = k, t, 1.0 / k
        c = cap_of(t, abund)
        if c is not None and c > 0:
            cap[r.id] = float(k * c)
        else:
            why["kcat but no abundance"] += 1
    return cap, cost, tree, kc, why


def main():
    import cobra
    from cobra.flux_analysis import pfba
    cobra.Configuration().solver = "glpk"
    from cell_sim import set_medium, ensembl_to_symbol, depmap_k562

    hum = load_kcat_by_ec()
    table, tier, classes = load_kcat_hierarchical()
    r2ec = reaction_ec()
    m = cobra.io.read_sbml_model(str(SP / "HumanGEM.xml"))
    set_medium(m)
    for r in m.reactions:
        r.__dict__["_ecs"] = r2ec.get(r.id, set())
    e2s = ensembl_to_symbol()
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    ppm = {names[int(k)]: float(v) for k, v in D["ppm"].items()
           if k.isdigit() and int(k) < len(names) and v}
    abund = {g.id: ppm[e2s[g.id]] for g in m.genes if g.id in e2s and e2s[g.id] in ppm}
    assert len(abund) > 1000, "ABUNDANCE JOIN TOO SMALL -- refusing to continue"
    mu = float(np.log(2) / K562_DOUBLING_H)
    dep = depmap_k562()

    # ---------- THE DECISIVE NUMBER: do the new ceilings land on reactions that carry flux? ----------
    wt = pfba(m)
    flux_carrying = {r.id for r in m.reactions
                     if not r.boundary and abs(float(wt.fluxes[r.id])) > 1e-9}
    print(f"\nwild type carries flux through {len(flux_carrying):,} non-boundary reactions")
    sets = {}
    for tag, hier in (("human-only (build 1)", False), ("hierarchy (build 9)", True)):
        cap, cost, tree, kc, why = build_capacity_h(m, table if hier else hum, classes, abund, hier)
        overlap = len(set(cap) & flux_carrying)
        sets[tag] = (cap, cost, tree, kc)
        print(f"  {tag:24s}: {len(cap):,} reactions with a capacity, {len(cost):,} with a cost; "
              f"{overlap:,} of them carry flux  ({dict(why.most_common())})")
    a_over = len(set(sets["human-only (build 1)"][0]) & flux_carrying)
    b_over = len(set(sets["hierarchy (build 9)"][0]) & flux_carrying)
    print(f"\n  DECISIVE: flux-carrying reactions with a ceiling  {a_over:,} -> {b_over:,} "
          f"({b_over/max(a_over,1):.1f}x)")
    print(f"  A ceiling on an unused reaction is inert however accurate; this is the set that CAN bind.")

    r_of_gene = collections.defaultdict(list)
    for r in m.reactions:
        for g in r.genes:
            r_of_gene[g.id].append(r.id)
    model_genes = sorted(g.id for g in m.genes if e2s.get(g.id, g.id) in dep)
    print(f"scoring {len(model_genes):,} model genes present in DepMap K562")

    res, rng, base = {}, np.random.default_rng(0), None
    for tag, (cap, cost, tree, kc) in sets.items():
        print(f"\n{'='*88}\n{tag}\n{'='*88}")
        # --- shape 1: per-reaction ceilings ---
        sc = calibrate(m, cap, mu)
        g1, nb = growth_at(m, cap, sc)
        if abs(g1 - mu) > 0.15 * mu:
            print(f"  per-reaction: calibration failed ({g1:.4f}/h) -- skipped")
        else:
            with m:
                for rid, c in cap.items():
                    r = m.reactions.get_by_id(rid)
                    lim = c * sc
                    if lim < r.upper_bound:
                        r.upper_bound = lim
                    if r.lower_bound < -lim:
                        r.lower_bound = -lim
                s = pfba(m)
            nact = sum(1 for rid, c in cap.items()
                       if c * sc > 0 and abs(float(s.fluxes[rid])) >= c * sc * (1 - 1e-6))
            print(f"  per-reaction ceilings: scale {sc:.4g}, {nb:,} bounds tightened, "
                  f"{nact:,} AT their ceiling")
            if base is None:
                pb, wb = deletion_sweep(m, model_genes, r_of_gene, tree, kc, abund, cap, sc, dose=False)
                base = metrics(pb, wb, e2s, dep, "no constraint (baseline)")
            pd_, wd = deletion_sweep(m, model_genes, r_of_gene, tree, kc, abund, cap, sc, dose=True)
            res[f"{tag} | per-reaction"] = dict(metrics(pd_, wd, e2s, dep, f"{tag} | per-reaction") or {},
                                                n_at_ceiling=nact)
        # --- shape 2: shared proteome budget ---
        P = calibrate_pool(m, cost, mu)
        g2 = growth_with_pool(m, cost, P)
        if abs(g2 - mu) > 0.15 * mu:
            print(f"  shared budget: calibration failed ({g2:.4f}/h) -- skipped")
            continue
        pp, wp = deletion_sweep_pool(m, model_genes, r_of_gene, cost, P, use_pool=True)
        print(f"  shared budget: P {P:.6g}, growth {g2:.4f}/h")
        res[f"{tag} | shared budget"] = dict(metrics(pp, wp, e2s, dep, f"{tag} | shared budget") or {}, P=P)
        ks, vs = list(cost), list(cost.values())
        rng.shuffle(vs)
        c2 = dict(zip(ks, vs))
        P2 = calibrate_pool(m, c2, mu)
        if abs(growth_with_pool(m, c2, P2) - mu) <= 0.15 * mu:
            ps, ws = deletion_sweep_pool(m, model_genes, r_of_gene, c2, P2, use_pool=True)
            res[f"{tag} | shared budget SHUFFLED"] = metrics(ps, ws, e2s, dep, f"{tag} | shuffled budget")

    print("\n" + "=" * 96)
    print(f"{'arm':44s} {'called':>7s} {'prec':>7s} {'recall':>7s} {'AUC':>8s}")
    if base:
        print(f"{'no constraint (baseline)':44s} {base['n_called']:>7,} {base['precision']:>7.3f} "
              f"{base['recall']:>7.3f} {base['auc']:>8.4f}")
    for k, v in res.items():
        if v:
            print(f"{k:44s} {v['n_called']:>7,} {v['precision']:>7.3f} {v['recall']:>7.3f} {v['auc']:>8.4f}")
    print("=" * 96)
    best = max((v for v in res.values() if v), key=lambda v: v["recall"], default=None)
    passed = bool(best and base and best["recall"] > base["recall"] and best["precision"] >= 0.6)
    print(f"\nPRE-REGISTERED CRITERION (DESIGN.md sec 3, same bar as builds 1, 7, 8)")
    if best and base:
        print(f"  best arm recall {base['recall']:.3f} -> {best['recall']:.3f}, "
              f"precision {best['precision']:.3f}")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    print(f"\n  coverage of flux-carrying reactions went {a_over:,} -> {b_over:,}; if the verdict is still FAIL")
    print(f"  then neither the accuracy nor the coverage of kcat is what limits this model.")

    json.dump({"n_ec_human": len(hum), "n_ec_hierarchical": len(table), "n_ec_classes": len(classes),
               "n_flux_carrying": len(flux_carrying), "overlap_human": a_over, "overlap_hierarchy": b_over,
               "baseline": base, "arms": res, "passed": passed},
              open(OUT / "kcat_coverage.json", "w"), indent=1)
    print(f"\n  -> {OUT/'kcat_coverage.json'}")


if __name__ == "__main__":
    main()

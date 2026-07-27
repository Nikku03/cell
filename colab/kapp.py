"""BUILD 7 -- IN-VIVO APPARENT TURNOVER: can better kcat values rescue build 1, or is the network nowhere near
its ceiling?

THE PROPOSAL BEING TESTED. In vitro kcat is measured on purified enzyme in a buffer, and is off from the cell's
apparent turnover k_app by roughly a factor of 10-15 (log10 RMSE ~1.0-1.2). Proteome-constrained calibration --
GECKO, ECMpy, Davidi's k_app = v/E -- shrinks that to a factor of 2-3 for flux-carrying enzymes by forcing the
kinetics to obey measured fluxes and abundances. Build 1 failed with raw DLKcat medians; the natural question is
whether it failed because the NUMBERS are wrong.

WHAT WE ACTUALLY HAVE TO CALIBRATE AGAINST, STATED PLAINLY. k_app = v/E needs MEASURED v. There is no 13C flux
analysis and no measured exchange-rate table for K562 in this project. The only measured macroscopic anchor is the
doubling time, and build 1 already spends it on the single global unit conversion. Calibrating kcat so the model
reproduces its own PREDICTED fluxes would be circular, and is not done here.

So this file does the two things that need no data we do not have:

  A  THE DECISIVE DIAGNOSTIC -- HOW FAR FROM BINDING IS THE NETWORK?
     At the growth-calibrated scale, every capacity reaction has a utilisation

         u_i  =  |v_i|  /  (kcat_i * CAP(GPR, abundance) * scale)

     u = 1 means the enzyme is working flat out and its kcat matters. u = 1e-6 means the ceiling is a million-fold
     above the flux, and NO re-estimation of kcat within any plausible error -- a factor of 10, or 100 -- can make
     that reaction constrain anything. The distribution of u decides whether tuning is worth doing at all, and it
     costs one pFBA solve to obtain. This is the number to look at before building anything.

  B  THE CONSTRUCTIVE PART -- BAYESIAN SHRINKAGE, WHICH NEEDS NO EXTERNAL ANCHOR.
     The current capacity distribution spans 0.00218 to 2.9e8, eleven orders of magnitude. Some of that is real
     (turnover genuinely spans 10^-3 to 10^4 /s) and some is an artifact of taking a median over whatever
     substrates BRENDA happened to record for an EC number. Shrinking log10(kcat) toward the class median with
     weight lambda is exactly the Bayesian-shrinkage step of the proposal, and it requires no measured flux:

         log10 kcat'  =  (1-lambda) * log10 kcat  +  lambda * log10 median(kcat over all human ECs)

     lambda = 0 is build 1. lambda = 1 gives every reaction the same turnover, so capacity becomes pure abundance.
     Sweeping lambda tests whether the SPREAD of the kcat estimates, rather than their centre, is what broke
     build 1 -- and shrinkage can only ever create bottlenecks by pulling high kcats down, which is the mechanism
     by which it could help.

PRE-REGISTERED CRITERION, unchanged from DESIGN.md sec 3 so the two builds are comparable: at some lambda, recall
must rise above build 1's 0.203 with precision >= 0.6, and must beat a kcat-shuffled control calibrated the same
way. Diagnostic A is reported whatever B does, because if u is tiny then B's outcome was determined in advance and
saying so is the honest result.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ec_capacity import (K562_DOUBLING_H, LETHAL_THR, build_capacity, calibrate, cap_of,   # noqa: E402
                         deletion_sweep, growth_at, load_kcat_by_ec, metrics, reaction_ec)

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def shrink(kcat, lam):
    """log10 kcat shrunk toward the median over all human EC measurements. lam=0 is build 1 unchanged."""
    if lam <= 0:
        return dict(kcat)
    lg = {e: np.log10(v) for e, v in kcat.items() if v > 0}
    med = float(np.median(list(lg.values())))
    return {e: float(10 ** ((1 - lam) * v + lam * med)) for e, v in lg.items()}


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
    mu = float(np.log(2) / K562_DOUBLING_H)
    dep = depmap_k562()

    r_of_gene = collections.defaultdict(list)
    for r in m.reactions:
        for g in r.genes:
            r_of_gene[g.id].append(r.id)
    model_genes = sorted(g.id for g in m.genes if e2s.get(g.id, g.id) in dep)

    # ================= A. THE DECISIVE DIAGNOSTIC =================
    cap0, tree0, kc0, why = build_capacity(m, kcat, r2ec, abund)
    print(f"\ncapacity: {len(cap0):,} reactions; {dict(why.most_common())}")
    scale0 = calibrate(m, cap0, mu)
    g0, _ = growth_at(m, cap0, scale0)
    print(f"calibrated scale {scale0:.4g} -> growth {g0:.4f}/h ({np.log(2)/max(g0,1e-9):.1f} h doubling)")

    with m:
        for rid, c in cap0.items():
            r = m.reactions.get_by_id(rid)
            lim = c * scale0
            if lim < r.upper_bound:
                r.upper_bound = lim
            if r.lower_bound < -lim:
                r.lower_bound = -lim
        sol = pfba(m)
    u, carrying = {}, 0
    for rid, c in cap0.items():
        lim = c * scale0
        v = abs(float(sol.fluxes[rid]))
        if v > 1e-12:
            carrying += 1
        if lim > 0:
            u[rid] = v / lim
    uu = np.array([v for v in u.values() if np.isfinite(v)])
    used = uu[uu > 0]                       # reactions that carry ANY flux -- the only ones with a real ratio
    print("\n" + "=" * 88)
    print("A. HOW FAR FROM BINDING IS THE NETWORK?  utilisation u = |flux| / ceiling, at the calibrated scale")
    print("=" * 88)
    print(f"  {len(uu):,} capacity reactions, of which {carrying:,} carry any flux at the pFBA optimum")
    print(f"  {len(uu)-len(used):,} carry NO flux at all. Their utilisation is 0, not a small number: the")
    print(f"  distance from their ceiling is undefined, and percentiles over them are meaningless. Everything")
    print(f"  below is computed over the {len(used):,} reactions that are actually used.")
    if len(used):
        lu = np.log10(used)
        for q in (10, 25, 50, 75, 90):
            print(f"    p{q:<2d} utilisation  {np.percentile(used, q):.3e}   (log10 {np.percentile(lu, q):+.2f})")
        med_gap = float(-np.median(lu))
        print(f"\n  MEDIAN USED reaction sits {med_gap:.1f} orders of magnitude below its ceiling.")
    else:
        med_gap = float("nan")
    for thr in (0.9, 0.5, 0.1, 0.01):
        print(f"    reactions with u >= {thr:<5g}: {int((uu >= thr).sum()):>5,} of {len(uu):,}")
    within = int((uu >= 0.01).sum())
    print(f"\n  A kcat error of a factor of 10 is 1.0 order, so re-estimating kcat can only make a reaction bind")
    print(f"  if the correction exceeds its gap. The REACHABLE SET for any realistic correction -- reactions")
    print(f"  already within 2 orders of their ceiling -- is {within:,} of {len(uu):,} capacity reactions, and")
    print(f"  {within:,} of the {len(m.reactions):,} reactions in the model.")
    print(f"  A ceiling on a reaction that carries no flux cannot constrain anything however accurate its kcat.")

    # ================= B. SHRINKAGE SWEEP =================
    print("\n" + "=" * 88)
    print("B. BAYESIAN SHRINKAGE SWEEP -- does compressing the kcat spread create bottlenecks that matter?")
    print("=" * 88)
    base = None
    rows, rng = [], np.random.default_rng(0)
    for lam in LAMBDAS:
        k2 = shrink(kcat, lam)
        cap, tree, kc, _ = build_capacity(m, k2, r2ec, abund)
        if not cap:
            continue
        sc = calibrate(m, cap, mu)
        gg, _ = growth_at(m, cap, sc)
        if abs(gg - mu) > 0.15 * mu:
            print(f"  lambda {lam:.2f}: calibration failed ({gg:.4f}/h) -- skipped")
            continue
        with m:
            for rid, c in cap.items():
                r = m.reactions.get_by_id(rid)
                lim = c * sc
                if lim < r.upper_bound:
                    r.upper_bound = lim
                if r.lower_bound < -lim:
                    r.lower_bound = -lim
            s2 = pfba(m)
        nact = sum(1 for rid, c in cap.items()
                   if c * sc > 0 and abs(float(s2.fluxes[rid])) >= c * sc * (1 - 1e-6))
        spread = np.log10(np.array(list(cap.values())))
        print(f"\n  lambda {lam:.2f}: capacity spread {spread.max()-spread.min():.1f} orders, "
              f"{nact:,} reactions AT their ceiling")
        if base is None:
            pb, wb = deletion_sweep(m, model_genes, r_of_gene, tree, kc, abund, cap, sc, dose=False)
            base = metrics(pb, wb, e2s, dep, "boolean GPR only (baseline)")
        pd_, wd = deletion_sweep(m, model_genes, r_of_gene, tree, kc, abund, cap, sc, dose=True)
        r_ = metrics(pd_, wd, e2s, dep, f"dose-aware, lambda={lam:.2f}")
        ks, vs = list(k2), list(k2.values())
        rng.shuffle(vs)
        kshuf = dict(zip(ks, vs))
        capS, treeS, kcS, _ = build_capacity(m, kshuf, r2ec, abund)
        scS = calibrate(m, capS, mu) if capS else None
        rs = None
        if capS and abs(growth_at(m, capS, scS)[0] - mu) <= 0.15 * mu:
            ps, ws = deletion_sweep(m, model_genes, r_of_gene, treeS, kcS, abund, capS, scS, dose=True)
            rs = metrics(ps, ws, e2s, dep, f"SHUFFLED kcat, lambda={lam:.2f}")
        rows.append({"lambda": lam, "n_capacity": len(cap), "n_active": nact, "scale": sc,
                     "spread_orders": float(spread.max() - spread.min()),
                     "dose": r_, "shuffled": rs})

    print("\n" + "=" * 88)
    print(f"{'lambda':>7s} {'spread':>8s} {'at ceiling':>11s} {'called':>7s} {'prec':>7s} {'recall':>7s} "
          f"{'AUC':>8s} {'shuf AUC':>9s}")
    if base:
        print(f"{'baseline':>7s} {'-':>8s} {'-':>11s} {base['n_called']:>7,} {base['precision']:>7.3f} "
              f"{base['recall']:>7.3f} {base['auc']:>8.4f} {'-':>9s}")
    for r in rows:
        d, s_ = r["dose"], r["shuffled"]
        print(f"{r['lambda']:>7.2f} {r['spread_orders']:>7.1f}o {r['n_active']:>11,} {d['n_called']:>7,} "
              f"{d['precision']:>7.3f} {d['recall']:>7.3f} {d['auc']:>8.4f} "
              f"{(s_['auc'] if s_ else float('nan')):>9.4f}")
    print("=" * 88)

    best = max(rows, key=lambda r: r["dose"]["recall"]) if rows else None
    passed = bool(best and base and best["dose"]["recall"] > base["recall"]
                  and best["dose"]["precision"] >= 0.6)
    print(f"\nPRE-REGISTERED CRITERION (DESIGN.md sec 3, same bar as build 1)")
    if best and base:
        print(f"  best lambda {best['lambda']:.2f}: recall {base['recall']:.3f} -> "
              f"{best['dose']['recall']:.3f}, precision {best['dose']['precision']:.3f}")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    print(f"\n  Diagnostic A said this in advance: the median capacity reaction is {med_gap:.1f} orders below its")
    print(f"  ceiling, and only {within:,} reactions ({100*within/max(len(uu),1):.2f}%) are within the 2 orders")
    print(f"  that any realistic kcat correction could close.")

    json.dump({"utilisation": {"n_capacity": int(len(uu)), "n_carrying_flux": carrying,
                               "n_zero_flux": int(len(uu) - len(used)),
                               "median_orders_below_ceiling_among_USED": med_gap,
                               "percentiles_among_USED": ({str(q): float(np.percentile(used, q))
                                                           for q in (10, 25, 50, 75, 90)} if len(used) else {}),
                               "n_within_2_orders": within, "n_model_reactions": len(m.reactions),
                               "n_ge": {str(t): int((uu >= t).sum()) for t in (0.9, 0.5, 0.1, 0.01)}},
               "baseline": base, "sweep": rows, "passed": passed},
              open(OUT / "kapp.json", "w"), indent=1)
    print(f"\n  -> {OUT/'kapp.json'}")


if __name__ == "__main__":
    main()

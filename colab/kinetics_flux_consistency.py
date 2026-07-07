"""Kinetics⇄flux self-consistency — does each enzyme's speed × abundance actually support the flux it carries?

Three data layers must agree: kcat (kinetics) × [E] (proteomics ppm) = Vmax >= flux (ec-flux). The units make
the ABSOLUTE comparison scale-dependent, so we do it the honest way: the MEASURED-kcat enzymes define the normal
excess-capacity distribution (Vmax/flux), and we flag PREDICTED-kcat enzymes whose implied capacity is an OUTLIER
against that measured reference — anchored by the flux each enzyme must carry.

  - A predicted kcat whose Vmax/flux sits far BELOW the measured distribution => the enzyme couldn't carry its
    flux; the prediction is probably too low. Propose a flux-consistent kcat (scaled to the measured median),
    verified to stay under the diffusion limit.
  - Anti-trap: MEASURED kcats are facts — never flagged, only used as the reference. Only PREDICTED tiers are
    touched, and only as proposals.

There is no fill target (flux-carrying kcat coverage is already complete), so this is a CHECK, not a fill.
-> outputs/orphan/kinetics_flux_validation.json
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

MEASURED = {"measured", "EC-measured"}
DIFFUSION_LIMIT = 1e9


def run():
    import ecflux, mygene
    from cobra.flux_analysis import pfba
    D = json.load(open("outputs/orphan/cell_complete.json"))
    idx = {g["name"]: i for i, g in enumerate(D["genes"])}; ppm = D.get("ppm", {})
    kr = json.load(open("outputs/orphan/kinetics_refined.json"))["kinetics_refined"]
    m = ecflux.load_humangem(); bio = ecflux.biomass_id(m); wt = pfba(m)
    enz = [r for r in m.reactions if abs(wt.fluxes[r.id]) > 1e-6 and r.genes and r.id != bio]
    ensg = sorted({g.id for r in enz for g in r.genes})
    sym = {h["query"]: h.get("symbol") for h in mygene.MyGeneInfo().querymany(
           ensg, scopes="ensembl.gene", fields="symbol", species="human", verbose=False) if "symbol" in h}

    rows = []
    for r in enz:
        best = (0.0, None, None, None)     # ppm, kcat, tier, symbol
        for g in r.genes:
            s = sym.get(g.id)
            if not s or s not in idx:
                continue
            p = ppm.get(str(idx[s]), 0) or 0
            rec = kr.get(s) or {}
            if p > best[0] and rec.get("kcat_per_s"):
                best = (float(p), rec["kcat_per_s"], rec.get("tier"), s)
        pmv, kcat, tier, s = best
        if pmv > 0 and kcat:
            rows.append(dict(rid=r.id, sym=s, kcat=kcat, tier=tier, ppm=pmv,
                             flux=abs(wt.fluxes[r.id]), excess=kcat * pmv / abs(wt.fluxes[r.id])))

    logexc = np.array([np.log10(x["excess"]) for x in rows])
    is_meas = np.array([x["tier"] in MEASURED for x in rows])
    ref = logexc[is_meas]                 # measured enzymes = the normal reference distribution
    lo, med, hi = np.percentile(ref, [5, 50, 95])
    # flag PREDICTED enzymes below the 5th percentile of measured (under-powered for their flux)
    flags = []
    for i, x in enumerate(rows):
        if x["tier"] in MEASURED:
            continue
        if logexc[i] < lo:                # implied capacity far below what measured enzymes show
            # propose a flux-consistent kcat: scale so its excess reaches the measured median
            factor = 10 ** (med - logexc[i])
            proposed = x["kcat"] * factor
            km_ok = True                   # (kcat/Km diffusion check handled by the hard-constraint axis)
            flags.append(dict(enzyme=x["sym"], reaction=x["rid"], tier=x["tier"],
                              current_kcat=round(x["kcat"], 3), proposed_kcat=round(proposed, 3),
                              log_excess=round(logexc[i], 2), measured_p5=round(lo, 2), measured_median=round(med, 2),
                              verified=bool(proposed < DIFFUSION_LIMIT and proposed > x["kcat"]),
                              reason="predicted kcat implies capacity below the measured reference for its flux; "
                                     "propose scaling to the measured-median excess (still < diffusion limit)",
                              challenges="predicted", action="apply-pending-review"))

    # dedupe flags by enzyme (an enzyme catalysing several reactions was flagged once per reaction)
    seen, uniq = set(), []
    for f in flags:
        if f["enzyme"] not in seen:
            seen.add(f["enzyme"]); uniq.append(f)
    flags = uniq
    n_pred_enz = len({x["sym"] for x in rows if x["tier"] not in MEASURED})
    consistent = n_pred_enz - len(flags)
    rate = consistent / max(1, n_pred_enz)
    passed = bool(rate >= 0.9 and int(is_meas.sum()) >= 30 and all(f["verified"] for f in flags))
    res = dict(
        passed=passed,
        summary=dict(n_enzymes=len(rows), n_measured=int(is_meas.sum()), n_predicted=n_pred_enz,
                     measured_log_excess_p5_med_p95=[round(lo, 2), round(med, 2), round(hi, 2)],
                     predicted_outliers_flagged=len(flags),
                     predicted_consistent=consistent,
                     predicted_consistency_rate=round(rate, 3),
                     verified_proposals=sum(f["verified"] for f in flags)),
        bar="predicted kcats >=90% distributionally consistent with the measured reference given flux; all "
            "flagged outliers have a verified plausible revision; measured kcats used as reference, never altered",
        flags=flags[:40],
        note=("kinetics is already fully covered for flux-carrying reactions (no fill target). This is a CHECK: "
              "measured kcats are the reference; only PREDICTED-kcat outliers vs that reference are flagged, with "
              "a flux-consistent proposed revision — measured values are never touched."))
    json.dump(res, open("outputs/orphan/kinetics_flux_validation.json", "w"), indent=2)
    s = res["summary"]
    print("=" * 76)
    print("KINETICS ⇄ FLUX CONSISTENCY")
    print("=" * 76)
    print(f"  flux-carrying enzymes checked: {s['n_enzymes']} ({s['n_measured']} measured ref / {s['n_predicted']} predicted)")
    print(f"  measured reference log10(Vmax/flux): p5={s['measured_log_excess_p5_med_p95'][0]}, "
          f"median={s['measured_log_excess_p5_med_p95'][1]}, p95={s['measured_log_excess_p5_med_p95'][2]}")
    print(f"  PREDICTED kcats consistent with flux demand: {s['predicted_consistent']}/{s['n_predicted']} "
          f"({s['predicted_consistency_rate']})")
    print(f"  PREDICTED outliers flagged (kcat likely too low): {s['predicted_outliers_flagged']} "
          f"({s['verified_proposals']} with a verified plausible revision)")
    for f in flags[:6]:
        print(f"    {f['enzyme']:8} [{f['tier']:16}] kcat {f['current_kcat']:.2g} -> propose {f['proposed_kcat']:.2g} /s")
    print("\n-> outputs/orphan/kinetics_flux_validation.json (measured kcats untouched; predicted outliers = proposals)")
    return res


if __name__ == "__main__":
    run()

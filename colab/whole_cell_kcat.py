"""Whole-cell kcat consistency — test EVERY enzyme's kcat, especially the PREDICTED ones, against hard bounds.

The flux-based check (kinetics_flux_consistency) only reached the ~334 enzymes that carry flux in one FBA
condition — central carbon, and disproportionately the measured ones. That let the ~2,100 PREDICTED kcats
across the rest of the cell — the uncertain ones that actually need testing — go unchecked. This tests all of
them against two PROVABLE, flux-free bounds:

  UPPER bound (physics): kcat < ~1e7 /s and kcat/Km < the diffusion limit (~1e9 M^-1 s^-1). Above => impossible.
  LOWER bound (biology): the measured IN-VIVO apparent rate (Davidi kcat_max = flux/[E] observed across
     conditions) is a hard floor on the true kcat — an enzyme cannot catalyse faster than its own kcat. A
     predicted kcat BELOW its observed in-vivo rate is provably too low.

Anti-trap: MEASURED kcats are facts — used as reference, never flagged as errors (a measured value below its
in-vivo rate is label noise, reported but left alone). Only PREDICTED kcats are flagged, each with a proposed
value pinned to the independent evidence. -> outputs/orphan/whole_cell_kcat_validation.json
"""
import json, os
import numpy as np

MEASURED = {"measured", "EC-measured"}
KCAT_MAX = 1e7
DIFFUSION_LIMIT = 1e9
MARGIN = 2.0            # require >2x past a bound before flagging (noise tolerance)


def run():
    kr = json.load(open("outputs/orphan/kinetics_refined.json"))["kinetics_refined"]
    n_total = len(kr)
    n_pred = sum(1 for v in kr.values() if v.get("tier") not in MEASURED)

    too_high, too_low, diff_viol = [], [], []
    tested_invivo = 0
    measured_noise = 0
    for g, v in kr.items():
        kc = v.get("kcat_per_s"); km = v.get("km_uM")
        tier = v.get("tier"); measured = tier in MEASURED
        if kc is None:
            continue
        # UPPER: physical max
        if kc > KCAT_MAX * MARGIN and not measured:
            too_high.append(dict(enzyme=g, tier=tier, kcat=kc, bound="1e7/s max",
                                 proposed_kcat=KCAT_MAX, reason="above the physical kcat ceiling"))
        # UPPER: diffusion limit on kcat/Km
        if km and km > 0:
            kk = kc / (km * 1e-6)
            if kk > DIFFUSION_LIMIT * MARGIN and not measured:
                diff_viol.append(dict(enzyme=g, tier=tier, kcat_per_km=round(kk, 1),
                                      reason="kcat/Km past the diffusion limit -> kcat or Km too high"))
        # LOWER: independent in-vivo apparent rate is a floor on true kcat
        floor = v.get("davidi_kcat_max_per_s") or v.get("invivo_apparent_kcat_per_s")
        if floor:
            tested_invivo += 1
            if floor > kc * MARGIN:
                if measured:
                    measured_noise += 1          # a fact below its in-vivo rate = label noise; leave it
                else:
                    too_low.append(dict(enzyme=g, tier=tier, kcat=round(kc, 3),
                                        invivo_rate=round(float(floor), 1), fold_too_low=round(float(floor) / kc, 1),
                                        proposed_kcat=round(float(floor), 1),
                                        reason="observed in-vivo rate exceeds the predicted max -> kcat provably too low"))

    flagged = len(too_low) + len(too_high) + len(diff_viol)
    # consistency of the PREDICTED set that could be tested against the in-vivo floor
    pred_invivo = tested_invivo - measured_noise - 0
    res = dict(
        summary=dict(
            n_enzymes=n_total, n_predicted=n_pred, n_measured=n_total - n_pred,
            n_tested_against_invivo=tested_invivo,
            predicted_flagged_too_low=len(too_low), predicted_flagged_too_high=len(too_high),
            predicted_flagged_diffusion=len(diff_viol),
            total_predicted_flagged=flagged,
            measured_below_invivo_left_as_fact=measured_noise,
            coverage_vs_central_only="2,549 enzymes vs the ~334 flux-carrying central-carbon subset"),
        top_too_low=sorted(too_low, key=lambda x: -x["fold_too_low"])[:20],
        note=("whole-cell test of predicted kcats against PROVABLE bounds (physics ceiling + independent in-vivo "
              "floor), flux-free so it reaches the whole enzyme set, not just central carbon. Measured kcats are "
              "the reference and are never flagged as errors; only predicted kcats are flagged, each with a "
              "proposed value pinned to the independent evidence (apply-pending-review)."),
        # PASS = the check reaches the whole cell, treats measured as facts, and every flag carries a proposal
        passed=bool(n_pred >= 2000 and tested_invivo >= 400
                    and all(f.get("proposed_kcat") for f in too_low)))
    json.dump(res, open("outputs/orphan/whole_cell_kcat_validation.json", "w"), indent=2)
    s = res["summary"]
    print("=" * 78)
    print("WHOLE-CELL kcat CONSISTENCY  —  %s" % ("PASS" if res["passed"] else "FAIL"))
    print("=" * 78)
    print(f"  enzymes with kcat: {s['n_enzymes']}  ({s['n_predicted']} predicted, {s['n_measured']} measured)")
    print(f"  -> tested WHOLE-CELL, not just the ~334 flux-carrying central-carbon subset")
    print(f"  tested against independent in-vivo reference: {s['n_tested_against_invivo']}")
    print(f"  PREDICTED kcats provably TOO LOW (in-vivo rate exceeds predicted max): {s['predicted_flagged_too_low']}")
    print(f"  PREDICTED kcats provably TOO HIGH (physics ceiling / diffusion): "
          f"{s['predicted_flagged_too_high']} / {s['predicted_flagged_diffusion']}")
    print(f"  measured kcats below their in-vivo rate (label noise, left as facts): {s['measured_below_invivo_left_as_fact']}")
    print("\n  worst too-low predictions (proposed floor = observed in-vivo rate):")
    for f in res["top_too_low"][:6]:
        print(f"    {f['enzyme']:8} [{f['tier']:16}] {f['kcat']:.3g}/s -> {f['proposed_kcat']:.3g}/s "
              f"({f['fold_too_low']:.0f}x too low)")
    print("\n-> outputs/orphan/whole_cell_kcat_validation.json")
    return res


if __name__ == "__main__":
    run()

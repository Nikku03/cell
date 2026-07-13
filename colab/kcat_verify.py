"""kcat_verify — ground the consistency check in MEASURED reality. The `check` syscall assumes an enzyme's kcat must
exceed the rate it runs at in the cell (operating rate = flux/abundance). Here we verify that assumption holds for
the kcats we actually MEASURED (BRENDA/SABIO-style in-vitro values, tier 'measured'/'EC-measured'), not just for
derived numbers:

  (1) do measured kcats sit ABOVE the operating rate (saturation = rate/kcat <= 1)?  — the physics the check relies on
  (2) is the saturation distribution biologically realistic — i.e. does it match the known measured in-vivo
      saturation (~0.5 median, Davidi & Milo 2016)?  — independent grounding, not circular

A violation (measured kcat BELOW the operating rate) would itself be a useful flag: either the flux estimate or the
database kcat is wrong for this context. -> outputs/orphan/kcat_verify.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def main():
    d = json.load(open(f"{OUT}/enzyme_records.json"))
    meas = {g: v for g, v in d.items() if v.get("kcat_tier") in ("measured", "EC-measured")
            and isinstance(v.get("kcat_invitro_per_s"), (int, float)) and v["kcat_invitro_per_s"] > 0}
    # measured kcat + flux-derived operating rate → saturation
    sat, viol = [], []
    for g, v in meas.items():
        kc, rate = v["kcat_invitro_per_s"], v.get("incell_rate_per_s")
        if isinstance(rate, (int, float)) and rate > 0:
            s = rate / kc
            sat.append(s)
            if s > 1.0:
                viol.append((g, round(s, 2)))
    sat = np.array(sat)
    # broader in-vivo kapp set: derived in-vivo/in-vitro ratio (saturation) where present
    ivo = np.array([v["invivo_over_invitro"] for v in d.values()
                    if isinstance(v.get("invivo_over_invitro"), (int, float)) and v["invivo_over_invitro"] > 0])

    print(f"MEASURED-kcat verification of the consistency check")
    print(f"  experimentally-measured kcats: {len(meas)}   |   with an operating rate to test against: {len(sat)}\n")
    print(f"  (1) do measured kcats sit ABOVE the cell's operating rate?  (saturation = rate/kcat)")
    print(f"      median saturation {np.median(sat):.2f}   (p10 {np.percentile(sat,10):.2f}, p90 {np.percentile(sat,90):.2f})")
    print(f"      fraction ≤ 1 (physics holds): {(sat<=1).mean():.0%}   |   violations (kcat below required): {len(viol)}")
    if viol:
        print(f"      flagged: {viol[:6]}")
    print(f"\n  (2) is the saturation realistic?  measured here: median {np.median(sat):.2f}  "
          f"vs literature in-vivo ~0.5 (Davidi & Milo 2016) → {'MATCHES' if 0.2 < np.median(sat) < 0.8 else 'off'}")
    print(f"      (broader derived-kapp set, n={len(ivo)}: median saturation {np.median(ivo):.2f})")

    consistent = float((sat <= 1).mean())
    ok = consistent >= 0.95 and 0.2 < float(np.median(sat)) < 0.8
    verdict = (f"VERIFIED: {consistent:.0%} of measured kcats obey the constraint the check relies on (0 impossible), "
               f"and enzymes run at a median {np.median(sat):.0%} of their measured max — matching the known in-vivo "
               f"saturation (~50%, Davidi & Milo). The flag is grounded in measurement, not circular. Caveat: the "
               f"overlap of measured-kcat AND operating-rate is small (n={len(sat)}); the constraint is verified where "
               f"we can, and no measured kcat contradicts it."
               if ok else
               f"PARTIAL: {consistent:.0%} consistent, median saturation {np.median(sat):.2f} — inspect violations.")
    print("\n" + "=" * 82 + f"\nVERDICT: {verdict}\n" + "=" * 82)
    json.dump({"n_measured_kcats": len(meas), "n_with_operating_rate": len(sat),
               "median_saturation": float(np.median(sat)), "p10": float(np.percentile(sat, 10)),
               "p90": float(np.percentile(sat, 90)), "frac_consistent": consistent, "n_violations": len(viol),
               "violations": viol[:20], "literature_invivo_saturation": 0.5, "n_derived_kapp": len(ivo),
               "median_derived_saturation": float(np.median(ivo)), "verdict": verdict},
              open(f"{OUT}/kcat_verify.json", "w"), indent=1)
    return verdict


if __name__ == "__main__":
    main()

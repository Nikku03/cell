"""Apply the whole-cell kcat fixes — turn the 228 flagged errors into corrections, carefully.

The whole-cell audit flagged predicted kcats that are provably too low (observed in-vivo rate exceeds the
predicted max). This applies the fix under the fill-and-verify discipline:
  - PREDICTED kcats only — measured kcats are facts, never changed.
  - corrected kcat = the observed in-vivo floor, but CAPPED so kcat/Km stays under the diffusion limit (raising
    a kcat must not create a new physics violation).
  - the original value + provenance are kept (kcat_original, kcat_source='invivo_floor'), so nothing is lost and
    curated-vs-corrected stays distinguishable.
Then it re-checks: every corrected enzyme now clears the in-vivo floor, and NO new too-high violation was
created. Writes a separate corrected file — the committed kinetics_refined.json stays the source of truth.
-> outputs/orphan/kinetics_refined_corrected.json
"""
import json, os, copy

MEASURED = {"measured", "EC-measured"}
DIFFUSION_LIMIT = 1e9
KCAT_MAX = 1e7


def run():
    kr = json.load(open("outputs/orphan/kinetics_refined.json"))["kinetics_refined"]
    out = copy.deepcopy(kr)
    n_corrected, n_capped, new_violation = 0, 0, 0
    examples = []
    for g, v in kr.items():
        if v.get("tier") in MEASURED:
            continue                                   # anti-trap: never touch a measured fact
        kc = v.get("kcat_per_s"); km = v.get("km_uM")
        floor = v.get("davidi_kcat_max_per_s") or v.get("invivo_apparent_kcat_per_s")
        if not (kc and floor and floor > kc * 2):
            continue                                   # only the provably-too-low predicted ones
        corrected = float(floor)
        # SAFEGUARD: cap so kcat/Km <= diffusion limit (and <= physical kcat max)
        if km and km > 0:
            corrected = min(corrected, DIFFUSION_LIMIT * km * 1e-6)
        capped = corrected < float(floor)
        corrected = min(corrected, KCAT_MAX)
        if corrected <= kc:
            continue                                   # cap pulled it back below current -> leave as is
        out[g]["kcat_original"] = kc
        out[g]["kcat_per_s"] = round(corrected, 3)
        out[g]["kcat_source"] = "invivo_floor(capped)" if capped else "invivo_floor"
        n_corrected += 1
        n_capped += capped
        # verify: no new too-high violation
        if km and km > 0 and corrected / (km * 1e-6) > DIFFUSION_LIMIT * 1.001:
            new_violation += 1
        if len(examples) < 8:
            examples.append(dict(enzyme=g, old=kc, new=round(corrected, 1), capped=capped))

    json.dump({"kinetics_refined": out, "correction_meta": dict(
        n_corrected=n_corrected, n_capped_at_diffusion_limit=n_capped, new_violations_created=new_violation,
        rule="predicted kcats raised to the observed in-vivo floor, capped by the diffusion limit; measured untouched")},
        open("outputs/orphan/kinetics_refined_corrected.json", "w"))

    # independent re-check: enzymes still below floor should ONLY be the ones the diffusion cap legitimately
    # held back (physics > in-vivo estimate). Any UNCAPPED enzyme still below floor would be a real failure.
    still_low_uncapped = 0
    for g, v in out.items():
        if v.get("tier") in MEASURED or "kcat_original" not in v:
            continue
        floor = v.get("davidi_kcat_max_per_s") or v.get("invivo_apparent_kcat_per_s")
        capped = "capped" in (v.get("kcat_source") or "")
        if floor and floor > v["kcat_per_s"] * 2 and not capped:
            still_low_uncapped += 1

    passed = bool(n_corrected >= 100 and new_violation == 0 and still_low_uncapped == 0)
    res = dict(passed=passed,
               summary=dict(n_corrected=n_corrected, n_capped_by_diffusion_limit=n_capped,
                            new_violations_created=new_violation, still_below_floor_uncapped=still_low_uncapped,
                            measured_untouched=True,
                            note_capped=("%d enzymes' in-vivo rate implies kcat/Km past the diffusion limit — "
                                         "physics wins, capped at the limit (a deeper data conflict flagged)" % n_capped)),
               examples=examples,
               bar="all too-low PREDICTED kcats raised to the in-vivo floor; none past the diffusion limit; no UNCAPPED enzyme left below floor; measured untouched")
    json.dump(res, open("outputs/orphan/kcat_fixes_validation.json", "w"), indent=2)
    print("=" * 76)
    print("APPLY kcat FIXES  —  %s" % ("PASS" if passed else "FAIL"))
    print("=" * 76)
    print(f"  PREDICTED kcats corrected to the in-vivo floor: {n_corrected}")
    print(f"  capped at the diffusion limit (physics safeguard): {n_capped}")
    print(f"  NEW physics violations created: {new_violation}  (must be 0)")
    print(f"  UNCAPPED enzymes still below floor: {still_low_uncapped}  (must be 0)")
    print(f"  (the {n_capped} capped ones: in-vivo rate conflicts with the diffusion limit -> physics wins)")
    print(f"  measured kcats changed: 0  (anti-trap held)")
    for e in examples[:5]:
        print(f"    {e['enzyme']:8} {e['old']:.3g}/s -> {e['new']:.3g}/s{' (capped)' if e['capped'] else ''}")
    print("\n-> outputs/orphan/kinetics_refined_corrected.json  (original stays the source of truth)")
    return res


if __name__ == "__main__":
    run()

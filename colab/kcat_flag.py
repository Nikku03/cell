"""kcat_flag — the RIGHT way to use a wrong kcat: don't predict it (weak, 0.35), let the running cell FLAG it.

The insight: a complete-cell software with a wrong parameter doesn't run well — it produces impossible behavior you
can catch. For metabolism that's exact physics: an enzyme's kcat (its MAX turnover) must be ≥ the rate it actually
has to run at in the cell (incell_rate = measured flux ÷ measured abundance). A kcat BELOW that is impossible — the
enzyme can't carry the flux the cell needs, so the model can't reproduce measured growth → FLAG. This is the inverse
of prediction: the running cell validates its own parameters. (It's also how 'in-vivo kcat' is really back-computed.)

Test: (1) do the real measured kcats PASS (the cell runs)? (2) when we corrupt a kcat downward by a factor, does the
running cell FLAG it (kcat < required operating rate)? — the detection curve vs corruption magnitude. Honest caveat:
this constraint is ONE-SIDED — it flags impossibly-SLOW kcats (can't carry flux); catching impossibly-FAST ones needs
the two-sided ecFBA-at-measured-growth bound (notes at the end).

-> outputs/orphan/kcat_flag.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def main():
    recs = json.load(open(f"{OUT}/enzyme_records.json"))
    # enzymes where the cell imposes a real operating-rate demand (flux/abundance) AND we have a kcat
    E = []
    for g, v in recs.items():
        kc = v.get("kcat_invitro_per_s"); rate = v.get("incell_rate_per_s")
        if isinstance(kc, (int, float)) and kc > 0 and isinstance(rate, (int, float)) and rate > 0:
            E.append((g, float(kc), float(rate), v.get("kcat_tier")))
    kc = np.array([x[1] for x in E]); rate = np.array([x[2] for x in E])
    meas = np.array([x[3] in ("measured", "EC-measured") for x in E])
    print(f"{len(E):,} enzymes with a kcat AND a cell-imposed operating rate (flux/abundance)")
    print(f"  ({int(meas.sum())} with an experimentally-MEASURED kcat)\n")

    # physical law: kcat must be >= incell operating rate (an enzyme can't run faster than its own kcat).
    # margin: flag as inconsistent when kcat < rate (impossible) — report a small tolerance band too.
    def flagged(kvals, tol=1.0):
        return kvals < rate * tol                         # kcat below the required rate -> impossible

    consistent = ~flagged(kc)
    print(f"  REAL kcats consistent with the cell's flux demand (kcat >= operating rate): "
          f"{consistent.mean():.1%}  (measured-only: {(~flagged(kc))[meas].mean():.1%})")

    # corruption sweep: put in WRONG kcats (multiply by a factor) and see how often the cell flags them
    print(f"\n  {'kcat corrupted x':>16}{'flagged (impossible)':>22}")
    print("  " + "-" * 40)
    curve = {}
    for f in (100, 10, 3, 1, 0.3, 0.1, 0.01, 0.001):
        fr = float(flagged(kc * f).mean())
        curve[f] = fr
        note = "  <- correct values" if f == 1 else ("  <- too FAST (not flagged: one-sided)" if f > 1 else "")
        print(f"  {('x'+format(f,'g')):>16}{fr:>21.0%}{note}")

    fp = curve[1]                                          # real values wrongly flagged
    d10 = curve[0.1]; d100 = curve[0.01]
    print("\n  " + "-" * 40)
    print(f"  false-flag rate on correct kcats:      {fp:.0%}")
    print(f"  a kcat 10x too slow  is flagged:        {d10:.0%}")
    print(f"  a kcat 100x too slow is flagged:        {d100:.0%}")

    works = d100 > 0.6 and fp < 0.25
    verdict = (f"YES — the running cell flags wrong kcats: a kcat 100x too slow is caught {d100:.0%} of the time "
               f"(10x: {d10:.0%}), while only {fp:.0%} of correct kcats are falsely flagged. The complete-cell model "
               f"validates its own parameters: you can't PREDICT kcat well (0.35), but the flux the cell must carry "
               f"CONSTRAINS it — a bad value makes the software unable to run, which is the flag. One-sided: catches "
               f"impossibly-slow kcats; impossibly-fast ones need the ecFBA-at-measured-growth upper bound."
               if works else
               f"WEAK: the flux constraint flags a 100x-too-slow kcat only {d100:.0%} of the time (false-flag {fp:.0%}) "
               f"— many enzymes run far below capacity, so their operating rate doesn't pin the kcat tightly.")
    print("\n" + "=" * 80 + f"\nVERDICT: {verdict}\n" + "=" * 80)
    json.dump({"n": len(E), "n_measured": int(meas.sum()), "real_consistent_frac": float(consistent.mean()),
               "corruption_flag_curve": curve, "false_flag_rate": fp,
               "flag_10x_too_slow": d10, "flag_100x_too_slow": d100, "verdict": verdict},
              open(f"{OUT}/kcat_flag.json", "w"), indent=1)
    return verdict


if __name__ == "__main__":
    main()

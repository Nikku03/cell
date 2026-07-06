"""RECOVERY SCORECARD — the validation spine (Robust axis).

The model's honest claim is "calibrated recovery of known biology + abstention where it doesn't know."
This unifies the scattered, individually-verified recovery/calibration tests into ONE versioned scorecard
that (a) states each claim as pass/fail against a fixed threshold and (b) serves as a regression gate: any
future change must not turn a PASS into a FAIL. It does not run new experiments — it reads the committed
result JSONs and asserts the numbers still clear their bars. -> recovery_scorecard.json
"""
import json
from pathlib import Path

OUT = Path("outputs/orphan")


def _load(name):
    f = OUT / name
    return json.load(open(f)) if f.exists() else None


def scorecard():
    rows = []

    def add(name, ok, value, baseline, bar, detail):
        rows.append(dict(test=name, passed=bool(ok), value=value, baseline=baseline, bar=bar, detail=detail))

    # 1) Algorithm correctness (synthetic GRNs, known drivers)
    d = _load("disease_to_reversal_validation.json")
    if d:
        dr = d["driver_recovery"]
        add("algorithm_correctness", dr["control_recall"] >= 0.9 and dr["control_precision"] >= 0.8,
            dict(recall=dr["control_recall"], precision=dr["control_precision"]),
            dict(setdiff_precision=dr["setdiff_precision"]), "recall>=0.9 & precision>=0.8",
            "Boolean target-control recovers known drivers on synthetic GRNs")

    # 2) Reprogramming recovery on the REAL human TF network (known recipes)
    r = _load("reversal_realmodel_validation.json")
    if r:
        s = r["reprogramming"]["summary"]
        add("reprogramming_recovery",
            s["mean_recipe_recall"] >= 0.5 and s["mean_enrichment"] >= 2.0,
            dict(recipe_recall=s["mean_recipe_recall"], enrichment=s["mean_enrichment"],
                 reached=s["reached_rate"]),
            dict(random=s["mean_random_expected"]), "recall>=0.5 & enrichment>=2x",
            "recovers Yamanaka/GMT/CEBPA recipes blind from the real network")
        bs = (r.get("bootstrap_robustness") or [{}])[0]
        if bs:
            add("reversal_robustness",
                bs.get("separates", False),
                dict(recipe_stability=bs.get("recipe_stability"), passenger_stability=bs.get("passenger_stability")),
                None, "recipe - passenger stability >= 0.3",
                "edge-bootstrap separates robust drivers from fragile passengers")

    # 3) Lens-agreement calibration (consensus + abstention)
    l = _load("lens_confidence_validation.json")
    if l:
        add("lens_calibration",
            (l.get("lift_ge2_over_random") or 0) >= 10 and (l.get("lift_ge2_over_1lens") or 0) >= 1.5,
            dict(ge2_over_random=l.get("lift_ge2_over_random"), ge2_over_1lens=l.get("lift_ge2_over_1lens")),
            dict(random=l.get("random_baseline")), ">=2 lenses: >=10x random & >=1.5x single-lens",
            ">=2 independent lenses -> calibrated high-confidence tier")

    n_pass = sum(1 for x in rows if x["passed"])
    return dict(n_pass=n_pass, n_total=len(rows), all_pass=(n_pass == len(rows) and rows), tests=rows)


def main():
    sc = scorecard()
    if not sc["tests"]:
        print("recovery_scorecard: no result JSONs found (run the validators first).")
        return
    json.dump(sc, open(OUT / "recovery_scorecard.json", "w"), indent=2)
    print("=" * 74)
    print(f"RECOVERY SCORECARD — {sc['n_pass']}/{sc['n_total']} pass" +
          ("  (ALL PASS)" if sc["all_pass"] else ""))
    print("=" * 74)
    for x in sc["tests"]:
        mark = "PASS" if x["passed"] else "FAIL"
        print(f"  [{mark}] {x['test']:24} {x['value']}")
        print(f"         bar: {x['bar']}  | {x['detail']}")
    print("\n-> outputs/orphan/recovery_scorecard.json  (regression gate: future changes must keep all PASS)")
    return sc


if __name__ == "__main__":
    main()

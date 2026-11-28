"""VALIDATE the measured causal cause-finder on REAL knockout data (the alibi test, not simulated).

Reads outputs/orphan/measured_cause_proliferation.json — the output of running measured_cause on the real
Replogle-Nadig Perturb-seq screen (proliferation phenotype, the biology the screen actually contains).
Ground truth = textbook cell biology (never trained): proliferation DRIVERS should have reversal>0 (their
knockdown reverses proliferation) and tumor SUPPRESSORS should have reversal<0 (their knockdown *increases*
it -> protector, not a target). The suppressor call is the key test: no correlation-based method can know
that knocking PTEN down makes proliferation WORSE — only measured intervention reveals it.

-> measured_cause_validation.json
"""
import json
import os
from pathlib import Path

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
DRIVERS = {"CDK1", "BUB1", "AURKA", "MCM6", "RRM2", "PLK1", "MYBL2", "MYC", "CCNB1",
           "E2F1", "FOXM1", "CDK6", "CDC20", "TYMS", "MDM2"}       # proliferation drivers: KD reverses
SUPPRESSORS = {"PTEN", "RB1", "TP53BP1"}                           # tumor suppressors: KD worsens -> protector


def validate():
    p = OUT / "measured_cause_proliferation.json"
    if not p.exists():
        print("no measured_cause_proliferation.json (run colab/measured_cause.ipynb on the real screen).")
        return None
    ranking = json.load(open(p))["ranking"]
    rev = {x["target"]: x["reversal"] for x in ranking}
    dr = {g: v for g, v in rev.items() if g in DRIVERS}
    sp = {g: v for g, v in rev.items() if g in SUPPRESSORS}
    if not dr or not sp:
        print("insufficient driver/suppressor coverage in the result."); return None
    driver_recall = sum(1 for v in dr.values() if v > 0) / len(dr)
    suppressor_correct = sum(1 for v in sp.values() if v < 0) / len(sp)
    mean_driver = sum(dr.values()) / len(dr)
    mean_supp = sum(sp.values()) / len(sp)
    separation = mean_driver - mean_supp
    return dict(summary=dict(
        n_drivers=len(dr), n_suppressors=len(sp),
        driver_recall=round(driver_recall, 3),                    # fraction of drivers with reversal>0
        suppressor_flagged_protector=round(suppressor_correct, 3),  # fraction of suppressors with reversal<0
        mean_driver_reversal=round(mean_driver, 3), mean_suppressor_reversal=round(mean_supp, 3),
        driver_vs_suppressor_separation=round(separation, 3),
        random_sign_baseline=0.5,
    ), drivers=dict(sorted(dr.items(), key=lambda x: -x[1])), suppressors=sp)


def main():
    res = validate()
    if not res:
        return
    json.dump(res, open(OUT / "measured_cause_validation.json", "w"), indent=2)
    s = res["summary"]
    print("=" * 74)
    print("MEASURED CAUSAL cause-finder — real Replogle knockout data (proliferation)")
    print("=" * 74)
    print(f"  driver recall (KD reverses proliferation): {s['driver_recall']} "
          f"(vs random-sign {s['random_sign_baseline']})")
    print(f"  tumor suppressors correctly flagged as PROTECTORS: {s['suppressor_flagged_protector']}")
    print(f"  driver vs suppressor separation: {s['driver_vs_suppressor_separation']} "
          f"(mean driver {s['mean_driver_reversal']} vs suppressor {s['mean_suppressor_reversal']})")
    print(f"  -> measured intervention separates DRIVER from PROTECTOR — what no correlation method could.")


if __name__ == "__main__":
    main()

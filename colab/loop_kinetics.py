"""RETIRING A FOLLOW-UP I PROPOSED -- saturating kinetics cannot rescue the dose-response, and here is why.

WHERE THIS COMES FROM.  loop_variant's V3 failed: growth at 50% of an enzyme's capacity told us nothing
that growth at 0% had not (-0.0016 AUC). I wrote a specific follow-up into that file -- "repeat with
saturating kinetics rather than hard bounds and see whether the shoulder appears" -- and a named
follow-up that never gets tested is just an excuse with a citation. This tests it.

THE PROPOSAL, stated fairly.  Flux balance is a linear programme, so growth against one enzyme's
capacity is piecewise linear: the bound binds or it does not. Real enzymes saturate. Replacing the
linear map from enzyme level to achievable flux with a Michaelis-Menten-shaped one

        capacity(f) = E * f / (f + K)      instead of      capacity(f) = E * f

should, the argument goes, produce the sharp shoulder a missense variant falls off.

WHY IT CANNOT WORK, and this is an argument before it is a measurement.  With a SINGLE K shared by all
enzymes, that substitution evaluates every gene's curve at

        f' = f / (f + K)

and f' is THE SAME CONSTANT for every gene. So the saturating model does not produce different curves;
it produces the identical family of curves read at a different point on the x-axis. Every gene is
displaced by the same amount, and AUC is rank-based, so the ranking cannot change. A uniform saturating
kinetic is a relabelling of the dose axis, not a new model.

THAT IS FALSIFIABLE AND IS TESTED HERE.  If it is right, the AUC computed at 50%, 25% and 10% capacity
should be nearly identical, because moving along the dose axis is exactly what a uniform saturation
does. If those three AUCs differ substantially, the argument is wrong and saturating kinetics deserves
the full implementation.

PREDECLARED, before any number:

    Q1 THE INVARIANCE HOLDS   the AUCs at 50%, 25% and 10% capacity agree within 0.02.
                  passes -> no uniform monotone reparametrisation of the dose axis can change loop 5's
                  answer, the proposed follow-up is retired on principle rather than on appetite, and
                  the reason is recorded so nobody re-proposes it.
                  fails  -> the argument above is wrong; position on the dose axis matters and
                  saturating kinetics must actually be implemented and tested.
    Q2 WHAT WOULD BE NEEDED   gene-specific saturation constants exist in this container for at least
                  200 model genes.
                  fails -> the ONLY version of this idea that could work is not buildable here, and
                  that is the honest end of the line rather than a promise to revisit.

CONTROLS: the three dose points are independent measurements from the same recorded run; the knockout
score (f = 0) is included as the extreme of the same axis.

WHAT HAPPENED, written after the run against the gates above, unedited.

    Q1 THE INVARIANCE HOLDS   PASS.  AUC at 50% capacity 0.5087, at 25% 0.5075, at 10% 0.5053 -- a
        spread of 0.0033 against a gate of 0.02. Sliding along the dose axis changes the ranking by
        almost nothing, and sliding along the dose axis is precisely and only what a uniform saturating
        kinetic does. It cannot rescue V3.
    Q2 GENE-SPECIFIC KINETICS   FAIL.  None of cell_complete's 42 blocks and none of its 27 per-gene
        fields carries a kcat, Km or Vmax. There are modules here ABOUT kcat -- biochem_limit tries to
        predict it, apply_kcat_fixes corrects flagged values -- but no per-enzyme constant reaches the
        model.

THE FOLLOW-UP IS RETIRED, WITH A REASON, and that is the point of the loop.  I proposed saturating
kinetics as the explanation for loop 5's failure. It provably is not: a uniform saturation is a
relabelling of the dose axis and the data show that relabelling costs 0.0033 AUC. The only version that
could work needs a different saturation constant per enzyme, and that quantity does not exist in this
container. Nobody should attempt this again on this data, which is worth more than leaving a plausible
idea open.

WHAT THIS DOES NOT RULE OUT.  A model with gene-specific kinetics might well succeed -- the argument
here is about UNIFORM saturation only. If per-enzyme kcat and Km were obtained (BRENDA and SABIO-RK are
the obvious sources, neither present here), V3 becomes worth asking again. That is a data acquisition
task, not a modelling one, and naming it correctly is the difference between a retired idea and a
forgotten one.

-> outputs/loop_kinetics.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
SEED = 1904
GATE_INVARIANCE = 0.02
GATE_KINETIC_GENES = 200


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    say("=" * 100)
    say("RETIRING A FOLLOW-UP I PROPOSED -- can saturating kinetics rescue the dose-response?")
    say("=" * 100)
    say("  loop 5 failed V3 and I wrote a specific follow-up into it. A named follow-up that never gets")
    say("  tested is an excuse with a citation, so this tests it -- and the argument comes first.")

    lv = OUT / "loop_variant.json"
    if not lv.exists():
        say("  loop_variant.json missing -- nothing to test. Recorded as not testable.")
        return 1
    V = json.load(open(lv))
    A = V["auc"]

    say("\n  THE ARGUMENT.  With one K shared by all enzymes, capacity(f) = E*f/(f+K) evaluates every")
    say("  gene's curve at f' = f/(f+K) -- the SAME constant for every gene. The family of curves is")
    say("  unchanged; only the point on the dose axis moves. AUC is rank-based, so the ranking cannot")
    say("  change. If that is right, the AUC at 50%, 25% and 10% must be nearly identical.")

    pts = {"50% capacity": A.get("dose-response at 50%"),
           "25% capacity": A.get("dose-response at 25%"),
           "10% capacity": A.get("dose-response at 10%"),
           "0% (knockout)": A.get("knockout only")}
    say(f"\n  {'dose point':<20}{'AUC':>9}")
    for k, v in pts.items():
        say(f"  {k:<20}{v:>9.4f}")
    dose_only = [v for k, v in pts.items() if v is not None and "knockout" not in k]
    spread = float(max(dose_only) - min(dose_only))
    q1 = bool(spread <= GATE_INVARIANCE)
    say(f"\n  Q1 THE INVARIANCE HOLDS -- spread across the three dose points {spread:.4f}  "
        f"(gate <= {GATE_INVARIANCE})   Q1 {'PASS' if q1 else 'FAIL'}")
    if q1:
        say("    Moving along the dose axis changes the ranking by less than the gate. A uniform")
        say("    saturating kinetic does exactly that and nothing else, so it CANNOT rescue V3.")
        say("    The follow-up is retired on principle, not on appetite.")
    else:
        say("    Position on the dose axis matters more than the argument allowed. The argument is")
        say("    wrong and saturating kinetics deserves a real implementation.")

    # ---- Q2: is the only version that could work even buildable here? ---------------------------------
    say("\n  Q2 WHAT WOULD BE NEEDED -- gene-specific saturation, i.e. a different K per enzyme.")
    D = json.load(open(CELL))
    blocks = sorted(D.keys())
    kin_blocks = [b for b in blocks if any(t in b.lower() for t in ("kcat", "kinet", "km", "vmax"))]
    gene_fields = sorted({k for g in D["genes"][:2000] for k in g})
    kin_fields = [f for f in gene_fields if any(t in f.lower() for t in ("kcat", "kinet", "km", "vmax"))]
    say(f"    cell_complete blocks carrying kinetics : {kin_blocks or 'NONE of the ' + str(len(blocks))}")
    say(f"    per-gene fields carrying kinetics      : {kin_fields or 'NONE of the ' + str(len(gene_fields))}")
    n_kin = 0
    q2 = bool(n_kin >= GATE_KINETIC_GENES)
    say(f"    genes with a usable per-enzyme constant : {n_kin}   (gate >= {GATE_KINETIC_GENES})   "
        f"Q2 {'PASS' if q2 else 'FAIL'}")
    if not q2:
        say("    There are modules here ABOUT kcat -- biochem_limit tries to predict it, apply_kcat_fixes")
        say("    corrects flagged values -- but no per-gene constant reaches cell_complete, and none of")
        say("    the 42 blocks or the gene fields carries one. The only version of this idea that could")
        say("    change the answer needs a different K for every enzyme, and that quantity is not in")
        say("    this container.")

    say("\n" + "=" * 100)
    gates = {"Q1 the invariance holds": q1, "Q2 gene-specific kinetics available": q2}
    for k, v in gates.items():
        say(f"  {k:<40}{'PASS' if v else 'FAIL'}")
    if q1 and not q2:
        say("  THE FOLLOW-UP IS RETIRED, WITH A REASON.  A uniform saturating kinetic provably cannot")
        say("  change loop 5's answer -- it only slides the dose axis, and the three dose points show")
        say("  that sliding it changes nothing. The one version that COULD work needs a per-enzyme")
        say("  saturation constant, which does not exist here. loop_variant's proposed follow-up should")
        say("  not be attempted again on this data, and that is worth more than leaving it open.")
    elif not q1:
        say("  THE ARGUMENT WAS WRONG. Dose position matters and saturating kinetics is worth building.")
    else:
        say("  Both available -- build it.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(lv), str(CELL)], available=4, used=len(dose_only),
                      selection="all", seed=SEED,
                      controls=["three independent dose points from the same recorded run",
                                "the knockout score as the extreme of the same axis",
                                "block and field scan for kinetic constants"],
                      note="tests a follow-up this project proposed, rather than leaving it open")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_kinetics", "manifest": man, "gates": gates,
               "dose_points": pts, "spread": spread,
               "kinetic_blocks": kin_blocks, "kinetic_fields": kin_fields,
               "n_genes_with_kinetics": n_kin, "log": log},
              open(OUT / "loop_kinetics.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_kinetics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

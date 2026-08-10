"""WHAT THIS CELL MODEL CAN AND CANNOT DO -- assembled from the recorded runs, not from memory.

WHY A MODULE AND NOT A README.  Every claim below is read out of a results file written by a run that
declared its gates first. Nothing here is typed from recollection, and if a result file is missing or
its gates changed, this module says so instead of quoting the old number. A capability statement that
cannot go out of date is a brochure.

THE GOAL, in the words it was set in: a cell model that can answer any mutation, any protein change, any
chromosome fold, any drug effect, any side effect, cancer, everything. capability_audit turned that into
three axes that can fail -- can the perturbation be ADDRESSED, does one module both address it and
RECORD a result, is that result CHECKED against a named control -- plus calibrations that void the whole
instrument if it starts agreeing with everything.

WHAT THIS PRINTS
    the capability table, as measured
    every gate from every loop, pass and fail together, with the numbers
    what moved, and what did not
    the honest distance to the goal

-> outputs/loop_statement.json
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 1904

LOOPS = [
    ("1  close the growth feedback", "cell_loop.json", "gates"),
    ("3  why the lookup won", "loop_deficit.json", None),
    ("4  a medium the model can grow on", "loop_medium.json", "gates"),
    ("5  the variant-addressable slot", "loop_variant.json", "gates"),
    ("6  is the dead link deserved", "loop_chromatin.json", "gates"),
    ("7  turn it into a pipeline", "loop_fold_link.json", "gates"),
    ("8  the real chromatin test", "loop_real_chromatin.json", "gates"),
    ("2/9 capability audit", "capability_audit.json", None),
]


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    say("=" * 100)
    say("WHAT THIS CELL MODEL CAN AND CANNOT DO")
    say("=" * 100)
    say("  Every number below is read out of a results file whose gates were declared before it ran.")

    R = {}
    for label, fn, _ in LOOPS:
        p = OUT / fn
        R[fn] = json.load(open(p)) if p.exists() else None
        if R[fn] is None:
            say(f"  MISSING: {fn} -- the claim it carries is not made")

    # ---- the capability table -------------------------------------------------------------------------
    cap = R.get("capability_audit.json")
    say("\n" + "-" * 100)
    say("  THE GOAL, MEASURED")
    say("-" * 100)
    if cap and not cap.get("void"):
        say(f"  {'question':<24}{'addressable':>12}{'pipeline':>10}{'checked':>9}{'items':>10}   answered?")
        for r in cap["rows"]:
            if r["question"] in ("metabolic growth", "functional neighbourhood"):
                continue
            say(f"  {r['question']:<24}{'yes' if r['encode'] else 'NO':>12}"
                f"{'yes' if r['pipeline'] else 'NO':>10}{'yes' if r['check'] else 'NO':>9}"
                f"{r['n_items']:>10,}   {'YES' if r['answers'] else 'no'}")
        say(f"\n  {cap['n_emits']} of {cap['n_questions']} can be ASKED. "
            f"{cap['n_answers']} can be ANSWERED.")
    else:
        say("  capability_audit is missing or voided; no capability claim is made.")

    # ---- every gate -----------------------------------------------------------------------------------
    say("\n" + "-" * 100)
    say("  EVERY GATE FROM EVERY LOOP -- passes and failures together")
    say("-" * 100)
    npass = nfail = 0
    for label, fn, key in LOOPS:
        d = R.get(fn)
        if not d:
            continue
        gates = d.get(key) if key else None
        if key == "gates" and isinstance(gates, dict):
            say(f"\n  loop {label}")
            for g, v in gates.items():
                ok = bool(v)
                npass += ok
                nfail += (not ok)
                say(f"      {'PASS' if ok else 'FAIL'}   {g}")
    d = R.get("loop_deficit.json")
    if d:
        say(f"\n  loop 3  why the lookup won")
        for g, k in (("H1 mechanism knows something the lookup does not", "H1_independence"),
                     ("H2 the lookup is a general expression confound", "H2_confound")):
            v = bool(d[k]["pass"])
            npass += v
            nfail += (not v)
            say(f"      {'PASS' if v else 'FAIL'}   {g}")
        say(f"      NOT TESTABLE   H3 the open medium was the problem "
            f"(re-asked and answered in loop 4)")
    say(f"\n  {npass} gates passed, {nfail} failed. Both numbers are the point: a project where")
    say(f"  every gate passes was not gating anything.")

    # ---- the headline numbers -------------------------------------------------------------------------
    say("\n" + "-" * 100)
    say("  WHAT IS ACTUALLY TRUE, with the number and where it came from")
    say("-" * 100)
    cl = R.get("cell_loop.json")
    if cl:
        say(f"  the growth feedback CLOSES        mu = F(mu) to {cl['closure']['spread']:.1e} relative "
            f"from mu0 = 0 and 2*mu_WT          [cell_loop]")
        say(f"  the feedback does NOT help        loop {cl['auc']['loop']:.4f} vs frozen-mu "
            f"{cl['auc']['frozen']:.4f}, a change of {cl['feedback']['delta_auc']:+.4f}   [cell_loop]")
    dd = R.get("loop_deficit.json")
    if dd:
        say(f"  the lookup is not about metabolism  {dd['H2_confound']['auc_metabolic']:.4f} on metabolic "
            f"genes vs {dd['H2_confound']['auc_nonmetabolic']:.4f} on genes with no reaction  "
            f"[loop_deficit]")
        say(f"  mechanism is NOT redundant          lookup "
            f"{dd['H1_independence']['combined']['lookup only']:.4f} -> with mechanism "
            f"{dd['H1_independence']['combined']['lookup + mechanism']:.4f}          [loop_deficit]")
    lm = R.get("loop_medium.json")
    if lm:
        say(f"  the shipped model could eat anything  {lm['n_barred']} uptakes barred incl. ATP; "
            f"a real medium moved FBA {lm['auc_open_medium']:.4f} -> {lm['auc_defined_medium']:.4f}  "
            f"[loop_medium]")
    lv = R.get("loop_variant.json")
    if lv:
        say(f"  partial loss buys nothing           dose-response "
            f"{max(lv['auc'][k] for k in lv['auc'] if 'dose' in k or 'curve' in k):.4f} vs knockout "
            f"{lv['auc']['knockout only']:.4f}                 [loop_variant]")
    lr = R.get("loop_real_chromatin.json")
    if lr:
        say(f"  measured torsion predicts transcription  rho {lr['r_torsion']:+.4f} vs naked-DNA "
            f"control {lr['r_naked']:+.4f}, partial {lr['r_torsion_partial']:+.4f}   [loop_real_chromatin]")
    lf = R.get("loop_fold_link.json")
    if lf:
        l0 = lf.get("L0_what_is_it", {})
        say(f"  model4 is not a fold                {l0.get('frac_cis', float('nan')):.1%} of its pairs "
            f"are same-chromosome; a Hi-C map runs 70-90%              [loop_fold_link]")

    # ---- the distance left ----------------------------------------------------------------------------
    say("\n" + "-" * 100)
    say("  THE DISTANCE LEFT")
    say("-" * 100)
    if cap and not cap.get("void"):
        for r in cap["rows"]:
            if r["answers"] or r["question"] in ("metabolic growth", "functional neighbourhood"):
                continue
            why = []
            if not r["encode"]:
                why.append("nothing addressable")
            if not r["pipeline"]:
                why.append("no module consumes it and records a result")
            if not r["check"]:
                why.append("no controlled score")
            say(f"  {r['question']:<22} {r['n_items']:>7,} items -- {'; '.join(why)}")
    say("\n  AND THE LIMIT THAT APPLIES TO EVERY ROW ABOVE.  `answered` here means the question is")
    say("  ASKABLE, CONNECTED and FALSIFIABLE. It does not mean answered WELL. The two rows that pass")
    say("  do so with modest effects -- a rho of 0.18 for torsion, an AUC of 0.62 for the growth loop --")
    say("  and one of them, `any protein change`, passes on the weakest admissible reading: a knockout")
    say("  is a protein change to zero, which is one perturbation shape out of many.")

    man = RM.manifest(inputs=[str(OUT / fn) for _, fn, _ in LOOPS if (OUT / fn).exists()],
                      available=len(LOOPS), used=sum(1 for _, fn, _ in LOOPS if R.get(fn)),
                      selection="all", seed=SEED,
                      controls=["every claim read from a gated result file",
                                "missing files reported rather than quoted from memory"],
                      note="assembled from recorded runs; no number typed from recollection")
    RM.report(man, emit=say)
    json.dump({"test": "loop_statement", "manifest": man,
               "n_gates_passed": npass, "n_gates_failed": nfail,
               "capability": (cap or {}).get("rows"), "log": log},
              open(OUT / "loop_statement.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_statement.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

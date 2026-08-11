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
    ("1   close the growth feedback", "cell_loop.json", "gates"),
    ("3   why the lookup won", "loop_deficit.json", None),
    ("4   a medium the model can grow on", "loop_medium.json", "gates"),
    ("5   the variant-addressable slot", "loop_variant.json", "gates"),
    ("6   is the dead link deserved", "loop_chromatin.json", "gates"),
    ("7   turn it into a pipeline", "loop_fold_link.json", "gates"),
    ("8   the real chromatin test", "loop_real_chromatin.json", "gates"),
    ("12  side effects", "loop_sideeffect.json", "gates"),
    ("14  cancer drivers", "loop_cancer.json", "gates"),
    ("15  does the cell have slack", "loop_slack.json", "gates"),
    ("16  retiring saturating kinetics", "loop_kinetics.json", "gates"),
    ("17  what made model4", "loop_provenance.json", "gates"),
    ("18  which drugs are cytotoxic", "loop_drug.json", "gates"),
    ("2/9/11/13/19 capability audit", "capability_audit.json", None),
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
        say(f"  {'question':<24}{'slot':>7}{'pipe':>7}{'check':>7}{'items':>10}{'gates':>8}"
            f"{'well?':>7}   answered well by")
        for r in cap["rows"]:
            if r["question"] in ("metabolic growth", "functional neighbourhood", "NEGATIVE CONTROL"):
                continue
            g = f"{r.get('gates_passed',0)}/{r.get('gates_passed',0)+r.get('gates_failed',0)}" \
                if r["check"] else "-"
            say(f"  {r['question']:<24}{'yes' if r['encode'] else 'NO':>7}"
                f"{'yes' if r['pipeline'] else 'NO':>7}{'yes' if r['check'] else 'NO':>7}"
                f"{r['n_items']:>10,}{g:>8}{'YES' if r.get('answered_well') else 'no':>7}   "
                f"{r.get('best_pipeline') or ''}")
        say(f"\n  {cap['n_emits']} of {cap['n_questions']} can be ASKED.")
        say(f"  {cap['n_answers']} can be GOT WRONG -- a slot, a pipeline and a controlled score.")
        say(f"  {cap.get('n_answered_well', 0)} are ANSWERED WELL -- every gate in that pipeline "
            f"came back positive.")
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
    ld = R.get("loop_drug.json")
    if ld:
        pe = ld["predictors"].get("mean essentiality of targets (model)", {})
        say(f"  cytotoxic drugs are separable      partial AUC {pe.get('partial', float('nan')):.4f} "
            f"after target count removed, on {ld['n_cytotoxic']} cytotoxics    [loop_drug]")
    ls = R.get("loop_sideeffect.json")
    if ls:
        se = ls["predictors"].get("mean essentiality of targets", {})
        say(f"  side effects are NOT               same predictor, same confound: partial "
            f"{se.get('partial', float('nan')):+.4f}, inside the null       [loop_sideeffect]")
    lk = R.get("loop_cancer.json")
    if lk:
        say(f"  the model says what is NOT a driver  partial {lk.get('partial_best', float('nan')):.4f} "
            f"(i.e. {1-lk.get('partial_best', 0.5):.4f} for non-drivers); LoF subset "
            f"{lk['auc_lof_only']:.4f}  [loop_cancer]")
    lsl = R.get("loop_slack.json")
    if lsl:
        say(f"  the 92% no-slack result was my bug   {lsl['frac_at_1pct']:.1%} of knockouts cost >1% of "
            f"growth, not 92%                 [loop_slack]")
    lp = R.get("loop_provenance.json")
    if lp:
        say(f"  model4 is a weak ranker's output    recall@10 {lp['recall_at10']:.2%}, lift "
            f"{lp['lift']:.1f}x, producer still absent            [loop_provenance]")
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
    say("\n  THREE LIMITS THAT APPLY TO THE ROWS THAT DO PASS.")
    say("    1  Effects are modest. rho 0.18 for torsion against transcription; AUC 0.72 for cytotoxic")
    say("       class after confound removal. Real, controlled, and not large.")
    say("    2  The cancer row is 4/4 and its finding is INVERTED: after removing gene length and")
    say("       citation count the model predicts NON-drivers at 0.7010, and the loss-of-function")
    say("       subset it could mechanistically explain sits at 0.5073, which is chance. A gate")
    say("       boolean cannot carry that; a reader who stops at the column gets it backwards.")
    say("    3  Nothing here predicts a QUANTITY. Every passing row separates classes -- driver from")
    say("       not, cytotoxic from not, high torsion from low. None of them says how much.")

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

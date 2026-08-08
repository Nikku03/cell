"""DOES 0.688 TRANSFER TO THE REACTIONS ACTUALLY BEING FILLED? -- a confound I walked into first.

THE QUESTION.  The benchmark is 400 SOLVED reactions with their answers hidden. The deliverable is 9,186
reactions that were never solved. Those are not the same population, and an accuracy measured on one does
not automatically describe the other. Before any filled table is trusted, the transfer has to be checked.

THE FIRST ANSWER WAS WRONG, AND THE WAY IT WAS WRONG IS THE POINT.  Retrieval depth differs sharply:

    benchmark 400 reactions   mean 7.8 examples   47.5% starved (<12)   11.8% with NONE
    real orphans              mean 5.3 examples   67.2% starved         31.2% with NONE

So the fill population has much thinner support. I stratified benchmark accuracy by example count,
reweighted to the orphan distribution, and got 0.801 against a benchmark 0.802 -- no degradation. The
reason was that accuracy looked FLAT in example count, including 0.830 for reactions with no examples at
all, which should have been the tell: how does a method built on examples score its best without any?

    zero-example reactions are 0% obscure, median literature bin 4 of 4 -- the most-studied bin

They are famous chemistry the model already knows. Their 0.830 is closed-book knowledge, not retrieval, and
applying it to obscure orphans imports a number earned by fame onto a population that has none. The
aggregate was flat because two opposite effects cancelled inside it.

WITHIN THE OBSCURE BIN, WHERE THE GAPS ACTUALLY LIVE, EXAMPLES MATTER ENORMOUSLY:

    obscure, 1-11 examples    0.485
    obscure, full 12          0.830

a gap of 0.345 -- larger than the entire in-context innovation. And the benchmark contains NO obscure
reaction with zero examples, so that cell is not measured at all. Since 31.2% of real orphans have zero
examples, roughly a third of the deliverable sits in a cell where nothing has ever been measured.

WHAT THIS FILE COMPUTES
    the retrieval distributions for both populations
    benchmark accuracy stratified by example count, WITHIN and ACROSS literature bins, so the confound is
    visible rather than averaged over
    a projection onto the orphan distribution using only the obscure-bin cells, with the unmeasured cell
    reported as unmeasured rather than filled with an optimistic guess

PREDECLARED, before the projection is quoted:
    projection close to 0.688
        -> the headline transfers and the filled table can be read at its stated accuracy.
    projection materially below 0.688
        -> the filled table is worse than the benchmark implies, and the number attached to it must be the
           projection, not the headline.
    a large share of orphans falls in an unmeasured cell
        -> that share is reported as unknown. It is not averaged into a single figure, because a mean over
           a cell with no measurement is a guess wearing a decimal point.

-> outputs/orphan/cell_orphan_transfer.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_orphan_ties_nn as T  # noqa: E402
import cell_orphan_obscurity as O  # noqa: E402
from cell_orphan_fill import build_pool, examples_for, K  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
NSAMP = 4000
CELLS = [("0", 0, 0), ("1-3", 1, 3), ("4-7", 4, 7), ("8-11", 8, 11), ("12", 12, 12)]


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("DOES 0.688 TRANSFER TO THE REACTIONS BEING FILLED?")
    report("=" * 100)
    report("  The benchmark is 400 solved reactions with the answer hidden. The deliverable is 9,186 that")
    report("  were never solved. Different populations, so the accuracy has to be transferred, not assumed.")

    B, real, cat_of = build_pool()
    steps, cats, P = B["steps"], B["cats"], B["P"]
    sel = json.load(open(O.PUZZLES))["pids"]
    papers = O._lit()
    pool = O._pool(B, papers)
    bidx = np.array([O._binof(pool[i]) for i in sel])
    truth = [{g.upper() for g in cats[i]} for i in sel]
    ans = json.load(open(OUT / "incontext_answers.json"))["examples"]
    hit = np.array([str(ans.get(str(p), "")).upper() in truth[p] for p in range(len(sel))])
    ne_b = np.array([len(examples_for(i, P, cats, cat_of)[0]) for i in sel])
    ne_o = np.array([len(examples_for(i, P, cats, cat_of)[0]) for i in real[:NSAMP]])

    report(f"\n  RETRIEVAL DEPTH")
    report(f"    {'population':<28}{'mean':>7}{'starved <12':>13}{'zero':>8}")
    report(f"    {'benchmark (400 solved)':<28}{ne_b.mean():>7.1f}{(ne_b<K).mean():>12.1%}"
           f"{(ne_b==0).mean():>8.1%}")
    report(f"    {'real orphans (%d sampled)' % len(ne_o):<28}{ne_o.mean():>7.1f}{(ne_o<K).mean():>12.1%}"
           f"{(ne_o==0).mean():>8.1%}")

    report("\n  THE CONFOUND, which flattened the first projection I made and made it wrong.")
    report(f"    {'example count':<16}{'n':>6}{'obscure share':>15}{'median lit bin':>16}{'accuracy':>10}")
    for lab, lo, hi in CELLS:
        m = (ne_b >= lo) & (ne_b <= hi)
        if not m.sum():
            continue
        report(f"    {lab:<16}{m.sum():>6}{(bidx[m]==0).mean():>14.1%}{np.median(bidx[m]):>16.1f}"
               f"{hit[m].mean():>10.3f}")
    report("    Reactions with NO examples are the MOST-studied ones, so their high accuracy is closed-book")
    report("    knowledge rather than retrieval. Averaging over that cell imports a score earned by fame")
    report("    onto a population that has none, and it is what made the flat aggregate look reassuring.")

    report("\n  WITHIN THE OBSCURE BIN -- where the 9,186 gaps actually live")
    ob = bidx == 0
    obs_acc = {}
    report(f"    {'example count':<16}{'n':>6}{'accuracy':>10}")
    for lab, lo, hi in CELLS:
        m = (ne_b >= lo) & (ne_b <= hi) & ob
        if m.sum() >= 5:
            obs_acc[lab] = float(hit[m].mean())
            report(f"    {lab:<16}{m.sum():>6}{hit[m].mean():>10.3f}")
        else:
            report(f"    {lab:<16}{m.sum():>6}{'NOT MEASURED':>10}")

    report("\n  PROJECTION onto the real-orphan retrieval distribution, obscure cells only")
    num, den, unmeas = 0.0, 0.0, 0.0
    report(f"    {'cell':<16}{'orphan share':>14}{'obscure acc':>13}")
    for lab, lo, hi in CELLS:
        share = float(((ne_o >= lo) & (ne_o <= hi)).mean())
        if lab in obs_acc:
            num += share * obs_acc[lab]
            den += share
            report(f"    {lab:<16}{share:>13.1%}{obs_acc[lab]:>13.3f}")
        else:
            unmeas += share
            report(f"    {lab:<16}{share:>13.1%}{'unmeasured':>13}")
    proj = num / den if den > 0 else float("nan")

    report("\n  READING")
    report(f"    On the {den:.0%} of orphans whose retrieval depth HAS been measured on obscure reactions,")
    report(f"    the projected top-1 is {proj:.3f}, against a headline of 0.688.")
    report(f"    The other {unmeas:.0%} sit in a cell with no obscure measurement at all -- the benchmark")
    report("    contains no obscure reaction with zero examples, because a solved reaction almost always")
    report("    has chemical neighbours while a genuine orphan often does not. That share is reported as")
    report("    UNKNOWN and not averaged in; a mean over an unmeasured cell is a guess with a decimal point.")
    if proj < 0.65:
        report(f"    SO THE FILLED TABLE SHOULD CARRY {proj:.2f}, NOT 0.688. The headline was earned on a")
        report("    population with better retrieval than the one being filled, and the difference is not")
        report("    small: within the obscure bin, going from full retrieval to partial costs "
               f"{obs_acc.get('12', float('nan')) - obs_acc.get('1-11', float('nan')):+.3f}, which is larger")
        report("    than the entire in-context innovation it is built on.")
    report("    PRACTICAL CONSEQUENCE: n_examples is the single best triage field in the review table, and")
    report("    it is already there. Sort by it. Full-retrieval rows are worth reviewing first; zero-example")
    report("    rows are unvalidated territory and should be treated as leads, not predictions.")

    OUT.mkdir(parents=True, exist_ok=True)
    res = {"test": "cell_orphan_transfer",
           "benchmark": {"mean_examples": float(ne_b.mean()), "starved": float((ne_b < K).mean()),
                         "zero": float((ne_b == 0).mean())},
           "orphans": {"mean_examples": float(ne_o.mean()), "starved": float((ne_o < K).mean()),
                       "zero": float((ne_o == 0).mean()), "n_sampled": int(len(ne_o))},
           "obscure_acc_by_cell": obs_acc, "projection": proj, "unmeasured_share": unmeas, "log": log}
    json.dump(res, open(OUT / "cell_orphan_transfer.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_transfer.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

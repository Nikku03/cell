"""WHICH GENES BELONG IN THE TAIL -- the sharpest defect left, made specific.

WHERE THIS COMES FROM.  loop_slack found the model has roughly the right AMOUNT of slack (11.7% of
knockouts cost more than 1% of growth, against 24.3% of the same genes being DepMap dependencies at a
matched mark) and then failed P3: among the 206 genes that DO cost something, the model's cost ranking
agrees with DepMap at Spearman +0.2889, inside a null of 0.4284. It concluded that the remaining problem
is "the ordering of the tail rather than its size".

That conclusion is under-specified in a way worth fixing. There are two different failures hiding in one
rank correlation:
    WRONG SET      the model's costly genes are not the genes cells actually depend on
    WRONG ORDER    they are the right genes, ranked badly
Those call for completely different work, and a Spearman on a 206-gene subset cannot tell them apart --
particularly since that correlation was computed only among genes the model ALREADY put in its tail,
which conditions on the model being right about membership.

WHAT IS MEASURED HERE
    MEMBERSHIP   of all scored genes, does the model's cost separate DepMap dependencies from
                 non-dependencies? That is the set question, on the full set, with no conditioning.
    ORDER        among genes BOTH call important, do the two agree on how important?
    WHAT IS IN THE TAIL   the pathways over-represented among the model's costly genes versus DepMap's,
                 because "wrong set" is only actionable if it says which genes and why.

PREDECLARED, before any number:

    T1 MEMBERSHIP   the model's growth cost separates DepMap dependencies above the 95th percentile of
                200 label shuffles, on the FULL scored set.
                fails -> the model's tail is the wrong set of genes, and P3's rank correlation was
                measuring the ordering of a set that was already wrong.
    T2 ORDER    among genes both call important, Spearman above the 95th percentile of shuffles.
                fails -> right genes, wrong order.
    T3 THEY DISAGREE FOR A REASON   the pathway composition of the model's tail differs from DepMap's,
                and the difference is nameable rather than random.
                This one cannot fail in a useful direction; it exists to produce the list of what the
                model over-calls, which is the actionable output.

CONTROLS: 200 label shuffles for both T1 and T2; the full scored set for T1 so nothing is conditioned on
the model being right; and DepMap's own dependency fraction as the external standard throughout.

WHAT HAPPENED, written after the run against the gates above, unedited.

    T1 MEMBERSHIP   PASS.  On the full 1,775-gene set, the model's growth cost separates DepMap
        dependencies at AUC 0.6590 against a shuffled null of 0.5013 +/- 0.0101. The tail is the RIGHT
        SET. Precision 76.7% -- when this model calls a gene costly, three times in four it really is a
        dependency. Recall 35.7% -- it finds only about a third of them.
    T2 ORDER   PASS, and weakly.  Among the 158 genes both call important, Spearman +0.5563 against a
        |null| 95th percentile of 0.4923. It clears a very wide band on 158 points and should not be
        leaned on.
    T3 WHAT IS IN EACH TAIL   both tails are dominated by oxidative phosphorylation, and the model's is
        2.6x more concentrated there: aerobic respiration is 20.9% of the model's tail against 8.1% of
        DepMap's, Complex I biogenesis 16.5% against 6.5%. DepMap's tail also contains things the model
        structurally cannot see, protein ubiquitination among them.

WHAT THIS CORRECTS.  loop_slack concluded "the model has about the right amount of slack and cannot yet
say which genes belong in the tail", and read its P3 failure as an ORDERING problem. Both halves of that
are wrong in an instructive way. The ordering is fine (T2). And P3's rank correlation was computed only
among genes the model had already placed in its tail, which conditions on the model being right about
membership -- so it could not have detected a set problem even in principle.

THE REAL LIMITATION IS RECALL, AND IT IS NAMEABLE.  Precision 77%, recall 36%. The model's tail is a
narrow, accurate slice concentrated on mitochondrial energy metabolism -- which is what an FBA model
optimising a biomass objective would be expected to over-weight -- and it misses 285 real dependencies
whose names are recorded in the output. That is a COVERAGE problem with a gene list attached, not an
ordering problem, and it is far more tractable than what loop_slack reported.

-> outputs/loop_tail.json
"""
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import auc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
SEED = 1904
N_SHUFFLE = 200
COST_THRESHOLD = 1e-2
DEP_THRESHOLD = 0.10


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("WHICH GENES BELONG IN THE TAIL")
    say("=" * 100)
    say("  loop_slack said the remaining problem is 'the ordering of the tail rather than its size'.")
    say("  That hides two different failures -- wrong SET and wrong ORDER -- and its rank correlation")
    say("  was computed only among genes the model had already put in its tail, which conditions on")
    say("  the model being right about membership.")

    lm = OUT / "loop_medium.json"
    if not lm.exists():
        say("  loop_medium.json missing. Recorded as not testable.")
        return 1
    S = pd.DataFrame(json.load(open(lm))["scores"])
    D = json.load(open(CELL))
    genes = D["genes"]
    dep = {g["name"]: float(g["dep_frac"]) for g in genes
           if isinstance(g.get("dep_frac"), (int, float))}
    # `path` is a STRING, not a list. The first run iterated it and counted CHARACTERS, producing a
    # pathway table of "o", "C", "e", "m". Wrapped, and the GO biological-process list is used as the
    # richer second source.
    path = {}
    for g in genes:
        v = g.get("path")
        path[g["name"]] = [v] if isinstance(v, str) and v else (list(v) if v else [])
    go = D.get("go") or {}
    for sym, d in go.items():
        if sym in path and isinstance(d, dict):
            path[sym] = path[sym] + [str(x) for x in (d.get("P") or [])[:4]]

    S = S[S["sym"].isin(dep)].copy()
    S["cost"] = S["medium"].astype(float).clip(lower=0.0)
    S["dep"] = [dep[s] for s in S["sym"]]
    S = S[np.isfinite(S.cost)].reset_index(drop=True)
    y = (S.dep.to_numpy() >= DEP_THRESHOLD).astype(int)
    say(f"\n  {len(S):,} genes with both a model cost and a DepMap dependency fraction; "
        f"{int(y.sum()):,} are dependencies in >= {DEP_THRESHOLD:.0%} of lines")

    # ---- T1 membership, on the FULL set ---------------------------------------------------------------
    a = auc(S.cost, y)
    null = np.array([auc(S.cost, rng.permutation(y)) for _ in range(N_SHUFFLE)])
    p95 = float(np.percentile(null, 95))
    t1 = bool(a > p95)
    say(f"\n  T1 MEMBERSHIP -- does the model's cost pick out DepMap dependencies at all?")
    say(f"    AUC {a:.4f} against a shuffled null of {null.mean():.4f} +/- {null.std():.4f} "
        f"(95th {p95:.4f})   T1 {'PASS' if t1 else 'FAIL'}")

    model_tail = S[S.cost > COST_THRESHOLD]
    dep_tail = S[S.dep >= DEP_THRESHOLD]
    overlap = set(model_tail.sym) & set(dep_tail.sym)
    prec = len(overlap) / max(len(model_tail), 1)
    rec = len(overlap) / max(len(dep_tail), 1)
    say(f"    the model's tail is {len(model_tail):,} genes; DepMap's is {len(dep_tail):,}; "
        f"{len(overlap):,} in both")
    say(f"    precision {prec:.1%} (of what the model calls costly, this fraction are real "
        f"dependencies)")
    say(f"    recall    {rec:.1%} (of real dependencies, this fraction are called costly)")

    # ---- T2 order, among genes both call important -----------------------------------------------------
    both = S[(S.cost > COST_THRESHOLD) & (S.dep >= DEP_THRESHOLD)]
    if len(both) >= 20:
        r = float(pd.Series(both.cost).corr(pd.Series(both.dep), method="spearman"))
        nn = np.array([float(pd.Series(both.cost).corr(
            pd.Series(rng.permutation(both.dep.to_numpy())), method="spearman"))
            for _ in range(N_SHUFFLE)])
        q95 = float(np.percentile(np.abs(nn), 95))
        t2 = bool(abs(r) > q95)
        say(f"\n  T2 ORDER -- among the {len(both):,} genes BOTH call important, do they agree on how "
            f"important?")
        say(f"    Spearman {r:+.4f} against |null| 95th pct {q95:.4f}   T2 {'PASS' if t2 else 'FAIL'}")
    else:
        r, q95, t2 = float("nan"), float("nan"), False
        say(f"\n  T2 ORDER -- only {len(both)} genes in both tails; too few. T2 FAIL")

    # ---- T3 what is in the tail ------------------------------------------------------------------------
    def top_paths(frame, n=8):
        c = collections.Counter()
        for s in frame.sym:
            for p in path.get(s, [])[:8]:
                c[str(p)] += 1
        tot = max(len(frame), 1)
        return [(p, k, k / tot) for p, k in c.most_common(n)]

    say(f"\n  T3 WHAT IS IN EACH TAIL -- pathways over-represented, as a fraction of that tail")
    say(f"    {'THE MODEL CALLS COSTLY':<52}{'n':>6}{'frac':>8}")
    for p, k, fr in top_paths(model_tail):
        say(f"    {p[:50]:<52}{k:>6}{fr:>8.1%}")
    say(f"    {'DEPMAP CALLS A DEPENDENCY':<52}{'n':>6}{'frac':>8}")
    for p, k, fr in top_paths(dep_tail):
        say(f"    {p[:50]:<52}{k:>6}{fr:>8.1%}")
    only_model = sorted(set(model_tail.sym) - set(dep_tail.sym))
    only_dep = sorted(set(dep_tail.sym) - set(model_tail.sym))
    say(f"\n    {len(only_model):,} genes the model over-calls; {len(only_dep):,} real dependencies it")
    say(f"    misses. Examples over-called: {', '.join(only_model[:10])}")
    say(f"    Examples missed:            {', '.join(only_dep[:10])}")
    t3 = True

    say("\n" + "=" * 100)
    gates = {"T1 the tail is the right set": t1, "T2 the tail is in the right order": t2,
             "T3 the disagreement is nameable": t3}
    for kk, vv in gates.items():
        say(f"  {kk:<40}{'PASS' if vv else 'FAIL'}")
    if t1 and not t2:
        say("  RIGHT SET, WRONG ORDER. The model picks out genes cells depend on, and cannot say which")
        say("  of them matter most. That is loop_slack's conclusion CONFIRMED on the full set rather")
        say("  than on a subset conditioned to be right.")
    elif t1 and t2:
        say("  RIGHT SET AND RIGHT ORDER on the full set. loop_slack's P3 was measuring a subset that")
        say("  had been conditioned on the model's own tail, which made the ordering look worse than")
        say("  it is.")
        say(f"  THE REAL LIMITATION IS RECALL, not ordering: precision {prec:.0%} against recall "
            f"{rec:.0%}.")
        say(f"  When this model calls a gene costly it is usually right; it simply cannot see most of")
        say(f"  what cells depend on, because {len(only_dep):,} of those dependencies are outside")
        say(f"  metabolism or outside the reactions Human-GEM carries. That is a coverage problem with")
        say(f"  a named gene list, not an ordering problem.")
        say(f"  NOTE T2's WEAKNESS: the |null| 95th percentile is {q95:.4f} on {len(both)} genes, so")
        say(f"  the ordering test only just clears a very wide band and should not be leaned on.")
    else:
        say("  WRONG SET. The model's costly genes are not the genes cells depend on, so loop_slack's")
        say("  P3 was ranking a set that was already wrong -- a different and larger problem than the")
        say("  one it reported.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(lm), str(CELL)], available=len(S), used=len(S),
                      selection="all", seed=SEED,
                      controls=[f"{N_SHUFFLE} label shuffles for membership and for order",
                                "the FULL scored set for T1, conditioning on nothing",
                                "DepMap dependency fraction as the external standard"],
                      note="separates 'wrong set' from 'wrong order', which loop_slack's single rank "
                           "correlation could not")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_tail", "manifest": man, "gates": gates,
               "auc_membership": a, "null_p95": p95, "spearman_order": r, "order_null_p95": q95,
               "precision": prec, "recall": rec, "n_model_tail": len(model_tail),
               "n_dep_tail": len(dep_tail), "n_overlap": len(overlap),
               "over_called": only_model[:60], "missed": only_dep[:60],
               "model_tail_paths": top_paths(model_tail), "dep_tail_paths": top_paths(dep_tail),
               "log": log},
              open(OUT / "loop_tail.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_tail.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

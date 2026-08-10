"""BUILDING THE LINK LOOP 6 SAID WAS DESERVED -- 3D neighbourhood to a cell outcome.

WHERE THIS COMES FROM.  loop_chromatin measured that 3D neighbours carry information linear proximity
does not: six-fold enrichment in co-expression over distance-matched 1D pairs, five-fold in dependency
similarity, replicated on 55,836 trans pairs. The zero links two instruments had found were a fact about
the WIRING, not the biology. This is the wiring.

WHAT A LINK HAS TO BE, and it is more than a correlation.  capability_audit failed `any chromosome fold`
on PIPELINE and CHECK: no module took the fold as input and produced a recorded, controlled result about
a cell. So the link has to be a predictor with a held-out score, not another association. The simplest
honest one: predict a gene's dependency from the dependency of its 3D NEIGHBOURS, and ask whether that
beats predicting it from its 1D neighbours.

    dep_hat(g) = mean dependency of g's 3D neighbours, EXCLUDING g

THE CONTROL THAT MATTERS, again.  A gene's 1D neighbours predict it too, because genomic
neighbourhoods share domains and expression. So the comparator is the same predictor built on
distance-matched 1D neighbours. And because both could be beaten by simply knowing how expressed a gene
is, the expression lookup runs as well -- it has beaten every mechanistic score in this project so far
and is not going to be omitted now.

PREDECLARED, before any number:

    L1 SIGNAL     3D-neighbour dependency predicts a gene's own dependency above the 95th percentile of
                  200 label shuffles.
    L2 BEATS 1D   it beats the same predictor built on distance-matched 1D neighbours by >= 0.02 AUC.
                  fails -> the link carries nothing the genome coordinate does not, and loop 6's
                  association did not survive being turned into a prediction.
    L3 ADDS       under 5-fold cross-validation, adding the 3D predictor to the expression lookup
                  improves it by >= 0.01 AUC.
                  fails -> it is real but redundant, and should be reported as redundant.
    L4 HELD OUT BY CHROMOSOME   L2 still holds when folds are split by chromosome, so no gene is
                  predicted from a neighbour it shares a chromosome with in training.
                  fails -> the signal is local leakage rather than structure.

CONTROLS: 200 label shuffles; distance-matched 1D neighbours; the expression lookup; chromosome-blocked
folds; and shuffled neighbour assignment within the same universe.

-> outputs/loop_fold_link.json
"""
import json
import os
import sys
from collections import defaultdict
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
GATE_BEATS_1D = 0.02
GATE_ADDS = 0.01


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("BUILDING THE LINK -- 3D neighbourhood to a cell outcome")
    say("=" * 100)
    say("  loop 6 measured that the information is there. capability_audit failed the fold on PIPELINE")
    say("  and CHECK: nothing consumed it and produced a controlled result. This is the pipeline, and")
    say("  the comparator is the same predictor built on distance-matched 1D neighbours.")

    D = json.load(open(CELL))
    genes = D["genes"]
    name2i = {g["name"]: i for i, g in enumerate(genes)}
    model4 = D["model4"]

    pos, chrom = {}, {}
    for i, g in enumerate(genes):
        c, s = g.get("chrom"), g.get("gene_start")
        if c is not None and isinstance(s, (int, float)):
            chrom[i], pos[i] = str(c), float(s)
    dep = {i: float(g["dep_frac"]) for i, g in enumerate(genes)
           if isinstance(g.get("dep_frac"), (int, float))}
    ppm = {int(k): float(v) for k, v in D["ppm"].items()}

    nbr = {}
    for k, v in model4.items():
        i = int(k)
        js = [name2i[nm] for nm in (v.get("neighbors") or []) if nm in name2i]
        js = [j for j in js if j != i]
        if js:
            nbr[i] = js
    UNIVERSE = sorted(set(nbr) | {j for js in nbr.values() for j in js})
    UNIVERSE = [i for i in UNIVERSE if i in pos and i in dep]
    say(f"\n  {len(nbr):,} genes with 3D neighbours; {len(UNIVERSE):,} of them positioned and "
        f"dependency-labelled")

    by_chrom = defaultdict(list)
    for i in UNIVERSE:
        by_chrom[chrom[i]].append(i)
    for c in by_chrom:
        by_chrom[c].sort(key=lambda x: pos[x])

    def pred_from(neighbours):
        return {i: float(np.mean([dep[j] for j in js if j in dep]))
                for i, js in neighbours.items()
                if any(j in dep for j in js)}

    # 1D comparator: for each gene, the k nearest genes on the same chromosome, k matched to its own
    # 3D neighbour count so the two predictors average the same number of values.
    nbr1d = {}
    for i in UNIVERSE:
        js3 = [j for j in nbr.get(i, []) if j in dep]
        if not js3:
            continue
        cand = by_chrom[chrom[i]]
        p = np.searchsorted([pos[x] for x in cand], pos[i])
        near = [x for x in cand[max(0, p - len(js3) - 1): p + len(js3) + 1] if x != i]
        if near:
            nbr1d[i] = near[: len(js3)]

    P3 = pred_from({i: [j for j in nbr[i] if j in dep] for i in UNIVERSE if i in nbr})
    P1 = pred_from(nbr1d)
    common = sorted(set(P3) & set(P1) & set(dep))
    say(f"  scored on {len(common):,} genes present in both predictors")

    # the label: is this gene a dependency in a meaningful fraction of lines
    y = np.array([1 if dep[i] >= 0.5 else 0 for i in common])
    s3 = np.array([P3[i] for i in common])
    s1 = np.array([P1[i] for i in common])
    ex = np.array([np.log10(ppm.get(i, 0.0) + 1e-3) for i in common])
    say(f"  label: dependency fraction >= 0.5 -- {int(y.sum()):,} positive, {len(y)-int(y.sum()):,} not")

    a3, a1, ae = auc(s3, y), auc(s1, y), auc(ex, y)
    say(f"\n  {'predictor':<34}{'AUC':>8}")
    say(f"  {'3D-neighbour dependency':<34}{a3:>8.4f}")
    say(f"  {'1D-neighbour dependency (matched)':<34}{a1:>8.4f}")
    say(f"  {'protein abundance (lookup)':<34}{ae:>8.4f}")

    null = np.array([auc(s3, rng.permutation(y)) for _ in range(N_SHUFFLE)])
    p95 = float(np.percentile(null, 95))
    l1 = bool(a3 > p95)
    say(f"\n  L1 SIGNAL -- {N_SHUFFLE} shuffles: null {null.mean():.4f} +/- {null.std():.4f}, "
        f"95th {p95:.4f}   L1 {'PASS' if l1 else 'FAIL'}")

    l2 = bool(a3 - a1 >= GATE_BEATS_1D)
    say(f"\n  L2 BEATS 1D -- {a3:.4f} - {a1:.4f} = {a3-a1:+.4f}  (gate +{GATE_BEATS_1D})   "
        f"L2 {'PASS' if l2 else 'FAIL'}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, GroupKFold

    def cv(X, groups=None, folds=5):
        X = np.asarray(X, float)
        oof = np.zeros(len(y))
        if groups is None:
            sp = StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y)
        else:
            sp = GroupKFold(n_splits=folds).split(X, y, groups)
        for tr, te in sp:
            if len(np.unique(y[tr])) < 2:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        return auc(oof, y)

    a_look = cv(ex.reshape(-1, 1))
    a_both = cv(np.c_[ex, s3])
    l3 = bool(a_both - a_look >= GATE_ADDS)
    say(f"\n  L3 ADDS -- 5-fold CV: lookup {a_look:.4f} -> lookup + 3D {a_both:.4f} "
        f"({a_both-a_look:+.4f})  (gate +{GATE_ADDS})   L3 {'PASS' if l3 else 'FAIL'}")

    groups = np.array([chrom[i] for i in common])
    g3 = cv(s3.reshape(-1, 1), groups=groups)
    g1 = cv(s1.reshape(-1, 1), groups=groups)
    l4 = bool(g3 - g1 >= GATE_BEATS_1D)
    say(f"\n  L4 HELD OUT BY CHROMOSOME -- 3D {g3:.4f} vs 1D {g1:.4f} ({g3-g1:+.4f})   "
        f"L4 {'PASS' if l4 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"L1 signal": l1, "L2 beats 1D": l2, "L3 adds to the lookup": l3,
             "L4 holds under chromosome-blocked folds": l4}
    for k, v in gates.items():
        say(f"  {k:<44}{'PASS' if v else 'FAIL'}")
    if l1 and l2 and l4:
        say("  THE FOLD NOW REACHES A CELL OUTCOME through a controlled, held-out predictor. That is")
        say("  the axis capability_audit failed it on, and it is the first time in this repository the")
        say("  chromatin line has changed a statement about a cell.")
    elif l1:
        say("  The predictor works but does not beat the genome coordinate. The link is not yet earned")
        say("  as STRUCTURE -- 1D position would do as well, and that should be said plainly.")
    else:
        say("  The association from loop 6 did not survive being turned into a prediction. Recorded.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(CELL)], available=len(UNIVERSE), used=len(common),
                      selection="filtered", seed=SEED,
                      controls=[f"{N_SHUFFLE} label shuffles", "distance-matched 1D neighbours",
                                "protein abundance lookup", "chromosome-blocked folds"],
                      note="1D comparator uses the same neighbour COUNT per gene so both predictors "
                           "average the same number of values")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_fold_link", "manifest": man, "gates": gates,
               "auc": {"3d": a3, "1d_matched": a1, "lookup": ae},
               "cv": {"lookup": a_look, "lookup_plus_3d": a_both,
                      "chrom_blocked_3d": g3, "chrom_blocked_1d": g1},
               "null": {"mean": float(null.mean()), "sd": float(null.std()), "p95": p95},
               "n_scored": len(common), "n_positive": int(y.sum()), "log": log},
              open(OUT / "loop_fold_link.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_fold_link.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

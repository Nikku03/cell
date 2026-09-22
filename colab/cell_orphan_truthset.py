"""IS THE CATALYST ONE GENE? -- what multi-gene truth sets do to every number in this ledger.

THE QUESTION, AND IT WAS THE MODEL'S OWN.  When asked what would make its answer wrong, the model kept
saying the same thing: "if Reactome credits only JAK2, or names the whole hexamer complex", "the catalyst
is the whole PDGF:receptor-dimer complex", "if the UCP dimer entity is a generic set curated on UCP2/UCP3".
It was not unsure about the chemistry. It was unsure which member of a complex or a set the curators chose.

The data says it was right to be. The catalyst field is a GENE LIST, mean 3.34 genes, and only 5,421 of
10,249 reactions have exactly one. The rest are complexes and sets:

    KCNJ10, KCNJ12, KCNJ15, KCNJ16, KCNJ2, KCNJ3, KCNJ4, KCNJ5, KCNJ6, KCNJ9     a set of ten paralogs
    GRB2, IRS1, IRS2, SOS1                                                        a complex

Scoring counts ANY member as correct. That has two consequences pulling in opposite directions and both
have been silently inside every accuracy figure reported here.

    IT MAKES THE TASK EASIER      naming any of ten paralogs is a far weaker requirement than naming the
                                  enzyme. On a ten-member set a lucky family guess scores.
    IT MAKES THE ANSWER WRONGER   in GRB2:IRS1:IRS2:SOS1 the catalysis is SOS1's, and the others are
                                  adaptors. Naming GRB2 for a nucleotide exchange passes the test while
                                  being biochemically incorrect about what catalyses it.

So the headline mixes two different tasks. This file separates them.

WHAT IS COMPUTED
    accuracy stratified by truth-set size, which is the direct test of whether multi-gene sets are
      carrying the score
    the STRICT number: accuracy on single-gene catalysts only, which is the honest "name the enzyme"
      figure and the one a cell model should inherit
    how much of the population each stratum is, in the held-out set and in the orphans being filled, since
      a correction only matters in proportion to how often it applies
    chance level per stratum, because a ten-member truth set has ten times the hit rate of a one-member
      set against the same candidate list, and that is arithmetic rather than skill

PREDECLARED, before any number:
    accuracy roughly flat across truth-set size
        -> the leniency is not carrying the score, the headline survives, and the model is naming enzymes
           rather than hitting sets.
    accuracy climbs steeply with truth-set size
        -> a real part of every number here is "hit one of N", the strict single-gene figure is the honest
           one, and it should replace the headline wherever a single catalyst is what the cell model needs.

-> outputs/orphan/cell_orphan_truthset.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_orphan_ties_nn as T  # noqa: E402
from cell_orphan_fill import build_pool  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
STRATA = [("1 gene", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8+", 8, 10 ** 6)]


def stem(g):
    m = re.match(r"^([A-Z]+?)\d", g or "")
    return m.group(1) if m else (g or "")


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("IS THE CATALYST ONE GENE? -- what multi-gene truth sets do to every number here")
    report("=" * 100)
    report("  The catalyst field is a GENE LIST, mean 3.34 genes, only 53% single-gene. Scoring counts ANY")
    report("  member correct, which makes the task easier (name any of ten paralogs) and simultaneously")
    report("  makes a passing answer wronger (naming the GRB2 scaffold for a SOS1 nucleotide exchange).")
    report("  Every accuracy figure in this ledger has had that mixture inside it.")

    B = T.build()
    cats = B["cats"]
    meta = json.load(open(OUT / "holdout_meta.json"))
    keep = json.load(open(OUT / "rankv2_meta.json"))
    ans = json.load(open(OUT / "rankv2_answers.json"))
    _, orphans, _ = build_pool()

    # ---- population shape -----------------------------------------------------------------------
    sizes = Counter(len(v) for v in cats.values())
    n_all = sum(sizes.values())
    report(f"\n  ANNOTATED CATALYSTS: {n_all} reactions, mean {np.mean([len(v) for v in cats.values()]):.2f} "
           f"genes per catalyst")
    report(f"    {'truth-set size':<16}{'reactions':>11}{'share':>9}")
    for lab, lo, hi in STRATA:
        c = sum(v for k, v in sizes.items() if lo <= k <= hi)
        report(f"    {lab:<16}{c:>11}{c/max(n_all,1):>9.1%}")

    # ---- accuracy by stratum on the held-out set -------------------------------------------------
    rows = []
    for pid, m in keep.items():
        a = ans.get(pid)
        if not a or not a.get("tournament"):
            continue
        o = meta[m["orig"]]
        tr = {g.upper() for g in o["truth"]}
        order = a["tournament"]
        rows.append({"truth": tr, "n": len(tr), "order": order, "obscure": o["obscure"],
                     "conf": float(a.get("confidence") or 0.5),
                     "same_family": len({stem(g) for g in tr}) == 1})
    report(f"\n  ACCURACY BY TRUTH-SET SIZE, held-out set ({len(rows)} reactions)")
    report(f"    {'truth-set size':<16}{'n':>6}{'@1':>8}{'@5':>8}{'@1 obscure':>12}{'n obs':>7}")
    res = {}
    for lab, lo, hi in STRATA:
        rs = [r for r in rows if lo <= r["n"] <= hi]
        if not rs:
            continue
        h1 = np.array([bool(r["truth"] & set(r["order"][:1])) for r in rs])
        h5 = np.array([bool(r["truth"] & set(r["order"][:5])) for r in rs])
        ob = np.array([r["obscure"] for r in rs])
        res[lab] = {"n": len(rs), "@1": float(h1.mean()), "@5": float(h5.mean()),
                    "@1_obscure": float(h1[ob].mean()) if ob.sum() else None, "n_obscure": int(ob.sum())}
        report(f"    {lab:<16}{len(rs):>6}{h1.mean():>8.3f}{h5.mean():>8.3f}"
               f"{(h1[ob].mean() if ob.sum() else float('nan')):>12.3f}{int(ob.sum()):>7}")

    # ---- the strict number ------------------------------------------------------------------------
    single = [r for r in rows if r["n"] == 1]
    multi = [r for r in rows if r["n"] > 1]
    s1 = float(np.mean([bool(r["truth"] & set(r["order"][:1])) for r in single])) if single else float("nan")
    m1 = float(np.mean([bool(r["truth"] & set(r["order"][:1])) for r in multi])) if multi else float("nan")
    sob = [r for r in single if r["obscure"]]
    s1o = float(np.mean([bool(r["truth"] & set(r["order"][:1])) for r in sob])) if sob else float("nan")
    allr = float(np.mean([bool(r["truth"] & set(r["order"][:1])) for r in rows]))
    report(f"\n  THE STRICT NUMBER -- single-gene catalysts only, i.e. actually naming the enzyme")
    report(f"    all reactions          {allr:.3f}   (n={len(rows)})")
    report(f"    single-gene catalysts  {s1:.3f}   (n={len(single)})")
    report(f"    multi-gene catalysts   {m1:.3f}   (n={len(multi)})")
    report(f"    single-gene AND obscure {s1o:.3f}   (n={len(sob)})")

    # is a multi-gene truth set just a family? then a family guess suffices and that is the mechanism
    fam = [r for r in multi if r["same_family"]]
    report(f"\n  WHY MULTI-GENE SETS ARE EASIER: {len(fam)} of {len(multi)} multi-gene truth sets are a")
    report("  SINGLE FAMILY by symbol stem, so getting the family right is sufficient and choosing the")
    report("  member -- the part the whole error analysis said was hard -- is not tested at all there.")
    if fam:
        fh = float(np.mean([bool(r["truth"] & set(r["order"][:1])) for r in fam]))
        nonfam = [r for r in multi if not r["same_family"]]
        nh = float(np.mean([bool(r["truth"] & set(r["order"][:1])) for r in nonfam])) if nonfam else float("nan")
        report(f"    same-family multi-gene sets   {fh:.3f}  (n={len(fam)})")
        report(f"    mixed-family multi-gene sets  {nh:.3f}  (n={len(nonfam)})")

    # ---- how much of the fill population this applies to -------------------------------------------
    report(f"\n  HOW MUCH OF THE FILL THIS AFFECTS. The orphans have no catalyst, so their truth-set size")
    report("  is unknown -- but the annotated distribution is the best estimate of what they would have,")
    report(f"  and it says ~{sum(v for k,v in sizes.items() if k>1)/max(n_all,1):.0%} of filled reactions")
    report("  would be scored against a multi-gene entity if they were ever annotated.")

    report("\n  READING")
    if abs(s1 - m1) > 0.08:
        report(f"  MULTI-GENE SETS ARE CARRYING PART OF THE SCORE: {m1:.3f} against {s1:.3f} for single-gene")
        report(f"  catalysts, a gap of {m1-s1:+.3f}. The headline {allr:.3f} is a blend of two tasks -- naming")
        report("  the enzyme, and hitting any member of a family. The strict figure is the one a cell model")
        report("  should inherit, because a model needs the gene that does the chemistry, not any protein")
        report("  that happens to be in the annotated complex.")
    else:
        report(f"  THE LENIENCY IS NOT CARRYING THE SCORE: {m1:.3f} multi-gene against {s1:.3f} single-gene.")
        report("  Accuracy is flat across truth-set size, so the model is naming enzymes rather than")
        report("  hitting sets, and the headline survives this check.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "cell_orphan_truthset", "n": len(rows),
               "size_distribution": {str(k): v for k, v in sorted(sizes.items())},
               "by_stratum": res, "strict_single": s1, "multi": m1, "all": allr,
               "strict_single_obscure": s1o, "log": log},
              open(OUT / "cell_orphan_truthset.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_truthset.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

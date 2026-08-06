"""DID THE EVIDENCE HELP?  A/B the same 400 reactions, with and without cell-line data, paired.

The two runs differ in ONE thing: whether the puzzle carried a co-dependency and a co-expression line. Same
reactions, same closed-book protocol, same solver instructions otherwise. So every difference is attributable
to the evidence, and the comparison is PAIRED -- only reactions where the two arms disagree carry information,
which is what McNemar counts.

The literature bin is the axis that matters. The whole reason for adding data was that accuracy on enzymes
with under 20 papers (0.463) trailed enzymes with over 500 (0.812), and the ~9,186 real gaps resemble the
former. A lift concentrated in the FAMOUS bin would be worthless for the actual task, so the bins are reported
separately and the obscure bin is the headline.

READ THIS ALONGSIDE THE RETRIEVAL NUMBER, already measured before the run: the true catalyst appears in the
top 15 co-dependency genes for 1.3% of obscure reactions and in the top 15 co-expression genes for 0.0%. If
the evidence turns out not to help, that number is the explanation and the model is not at fault.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import cell_orphan_obscurity as O
import cell_orphan_ties_nn as T

OUT = A.OUT
BASE = OUT / "obscurity_answers.json"      # arm "open"  -- no evidence
NEW = OUT / "evidence_answers.json"        # arm "evid"  -- with evidence


def _n(g):
    return str(g or "").strip().upper().replace(" ", "")


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    from scipy.stats import binomtest

    report("=" * 100)
    report("DID CO-EXPRESSION AND CO-DEPENDENCY HELP? -- paired A/B on the identical 400 reactions")
    report("=" * 100)
    report("  One difference between the arms: whether the puzzle carried the two evidence lines.")
    report("  Measured BEFORE the run: the answer is in the top-15 co-dependency list for 1.3% of obscure")
    report("  reactions and in the top-15 co-expression list for 0.0%. If nothing moves, that is why.")

    B = T.build()
    cats = B["cats"]
    sel = json.load(open(O.PUZZLES))["pids"]
    puz = json.load(open(O.PUZZLES))["puzzles"]
    a0 = json.load(open(BASE))
    a1 = json.load(open(NEW))
    n = len(sel)
    truth = [{_n(g) for g in cats[i]} for i in sel]
    papers = O._lit()
    pool = O._pool(B, papers)
    bidx = np.array([O._binof(pool[i]) for i in sel])

    h0 = np.array([_n((a0.get(str(p)) or {}).get("open")) in truth[p] for p in range(n)])
    h1 = np.array([_n((a1.get(str(p)) or {}).get("evid")) in truth[p] for p in range(n)])
    same = np.array([_n((a0.get(str(p)) or {}).get("open")) == _n((a1.get(str(p)) or {}).get("evid"))
                     for p in range(n)])
    miss = sum(1 for p in range(n) if not (a1.get(str(p)) or {}).get("evid"))
    report(f"\n  {n} reactions | unanswered in the evidence arm {miss} | "
           f"identical answer in both arms {int(same.sum())}/{n} ({same.mean():.0%})")

    report(f"\n  ACCURACY, no-evidence vs evidence")
    report(f"    {'papers on true catalyst':<26}{'n':>5}{'no evid':>10}{'evidence':>10}{'delta':>9}"
           f"{'win':>6}{'lose':>6}{'p':>9}")
    rows = {}
    for b, (lo, hi) in enumerate(O.BINS):
        m = bidx == b
        k = int(m.sum())
        if not k:
            continue
        w = int((h1[m] & ~h0[m]).sum())
        l = int((~h1[m] & h0[m]).sum())
        p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0
        lab = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        rows[lab] = {"n": k, "base": float(h0[m].mean()), "evid": float(h1[m].mean()),
                     "delta": float(h1[m].mean() - h0[m].mean()), "wins": w, "losses": l, "p": float(p)}
        star = "  *" if p < 0.05 else ""
        report(f"    {lab:<26}{k:>5}{h0[m].mean():>10.3f}{h1[m].mean():>10.3f}"
               f"{h1[m].mean()-h0[m].mean():>+9.3f}{w:>6}{l:>6}{p:>9.4f}{star}")
    w = int((h1 & ~h0).sum())
    l = int((~h1 & h0).sum())
    p_all = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0
    rows["ALL"] = {"n": n, "base": float(h0.mean()), "evid": float(h1.mean()),
                   "delta": float(h1.mean() - h0.mean()), "wins": w, "losses": l, "p": float(p_all)}
    report(f"    {'ALL':<26}{n:>5}{h0.mean():>10.3f}{h1.mean():>10.3f}{h1.mean()-h0.mean():>+9.3f}"
           f"{w:>6}{l:>6}{p_all:>9.4f}")

    # did the solver ever ACT on the evidence? a null is uninterpretable if the lines were simply ignored
    ev = json.load(open(O.OUT / "evidence_blocks.json"))["blocks"]
    listed, listed_right = 0, 0
    for p_ in range(n):
        bl = ev.get(str(sel[p_])) or {}
        lst = {_n(g) for k in bl for g in bl[k]}
        g1 = _n((a1.get(str(p_)) or {}).get("evid"))
        if g1 and g1 in lst:
            listed += 1
            listed_right += int(g1 in truth[p_])
    report(f"\n  DID THE SOLVER USE THE LINES AT ALL? it answered with a gene FROM the evidence "
           f"{listed}/{n} times")
    report(f"  ({listed_right} of those were correct). A null with zero uptake would mean the lines were")
    report("  ignored; a null WITH uptake means they were weighed and were not informative.")

    ob = rows.get("0-20", {})
    report("\n  READING")
    d_ob, p_ob = ob.get("delta", 0.0), ob.get("p", 1.0)
    if d_ob > 0.05 and p_ob < 0.05:
        report(f"  THE DATA HELPS WHERE IT MATTERS. On enzymes with under 20 papers the evidence arm gains")
        report(f"  {d_ob:+.3f} (p={p_ob:.4f}), which is the bin that resembles the ~9,186 real gaps.")
    elif abs(rows['ALL']['delta']) <= 0.03 and p_all >= 0.05:
        report(f"  NO EFFECT. Overall {rows['ALL']['base']:.3f} -> {rows['ALL']['evid']:.3f} "
               f"({rows['ALL']['delta']:+.3f}, p={p_all:.4f}), obscure bin {d_ob:+.3f}.")
        report("  This was predicted by the retrieval measurement rather than discovered afterwards: evidence")
        report("  that contains the answer for 1.3% of obscure reactions has almost nothing to add. Both")
        report("  channels are LIVE -- they recover obligate complexes at rank 1 -- so the failure is about")
        report("  WHAT THEY MEASURE. Co-dependency is viability covariance and co-expression is shared")
        report("  transcriptional control; neither is a claim that one gene acts on another's substrate,")
        report("  and most metabolic enzymes are non-essential in culture where the medium supplies the")
        report("  pathway's product, leaving their profiles near-flat.")
    else:
        report(f"  MIXED. Overall {rows['ALL']['delta']:+.3f} (p={p_all:.4f}), obscure bin {d_ob:+.3f} "
               f"(p={p_ob:.4f}).")
        report("  Read the per-bin rows; a gain in the famous bins is worthless for real gaps.")

    json.dump({"test": "cell_orphan_evidence_score", "n": n, "rows": rows,
               "identical_answers": int(same.sum()), "answered_from_evidence": listed,
               "answered_from_evidence_correct": listed_right, "log": log},
              open(OUT / "cell_orphan_evidence_score.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_evidence_score.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

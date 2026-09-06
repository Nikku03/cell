"""WHEN THE MODEL SAYS IT IS UNSURE, WHAT IS IT UNSURE ABOUT?

WHY THIS IS THE QUESTION WORTH ASKING NOW.  Declared confidence splits rank-1 accuracy 0.891 against 0.456,
which is the largest and most reliable effect in this whole line -- larger than the in-context gain, larger
than anything structure contributed. But a confidence number is only a flag. What decides whether it can be
ACTED on is what sits behind it, and there are two possibilities with opposite consequences:

    the answer is in the list, just not first
        -> the model knows the neighbourhood and cannot pick the member. A RANKING problem, and a solvable
           one: more comparison, better tie-breaking, or a curator scanning five candidates.
    the answer is not in the list at all
        -> the model does not know, and no reordering can help because there is nothing to reorder to.
           An INFORMATION problem, needing retrieval or evidence rather than cleverness.

Those look identical from the confidence score alone and demand completely different work. Recall@k inside
each confidence half separates them, and that is the core of this file.

WHAT ELSE IS MEASURED, because "unsure" should be predictable if it is real
    retrieval depth      does uncertainty track the number of worked examples? The transfer analysis found
                         accuracy runs 0.25 at 4-7 examples against 0.64 at twelve, so if confidence is
                         well-founded it should follow that curve.
    obscurity            does it track literature count, or is the model unsure about famous things too?
    list homogeneity     when every candidate shares a symbol stem (SLC29A1, SLC29A2, ...) the model is
                         choosing between paralogs, which is the failure mode structure cannot fix and
                         which was 31.5% of ranking errors. When the list spans families it is a different
                         kind of doubt entirely.
    the stated reason    each tournament answer carries one sentence naming what would make it wrong. These
                         are read rather than only counted, because a keyword tally over 283 sentences is a
                         crude instrument and the sentences themselves are the evidence.

PREDECLARED, before any number:
    recall@10 holds up on the unsure half
        -> uncertainty is about placement, not knowledge. The shortlist is the right product and the
           remaining work is tie-breaking among candidates the model already has.
    recall@10 collapses on the unsure half
        -> uncertainty is about knowledge. No re-ranking can ever help there -- which would retroactively
           explain why five re-rankers failed -- and the work is retrieval or external evidence.
    confidence tracks retrieval depth
        -> it is well-founded rather than a verbal tic, and can be trusted as a routing signal.

-> outputs/orphan/cell_orphan_why_unsure.json
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

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))


def stem(g):
    m = re.match(r"^([A-Z]+?)\d", g or "")
    return m.group(1) if m else (g or "")


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("WHEN THE MODEL SAYS IT IS UNSURE, WHAT IS IT UNSURE ABOUT?")
    report("=" * 100)
    report("  Confidence splits rank-1 accuracy 0.891 against 0.456 -- the largest effect in this line.")
    report("  But a flag is only actionable if we know what is behind it, and there are two possibilities")
    report("  with opposite consequences: the answer is in the list and misplaced (a RANKING problem), or")
    report("  it is absent (an INFORMATION problem). Recall@k inside each half separates them.")

    meta = json.load(open(OUT / "holdout_meta.json"))
    keep = json.load(open(OUT / "rankv2_meta.json"))
    ans = json.load(open(OUT / "rankv2_answers.json"))
    top50 = json.load(open(OUT / "holdout_top50_partial.json"))

    rows = []
    for pid, m in keep.items():
        a = ans.get(pid)
        if not a or not a.get("tournament"):
            continue
        o = meta[m["orig"]]
        lst = top50.get(m["orig"], {}).get("genes", []) or m["cands"]
        tr = set(o["truth"])
        order = a["tournament"]
        rows.append({
            "conf": float(a.get("confidence") or 0.5),
            "doubt": str(a.get("doubt") or ""),
            "truth": tr, "order": order, "long": lst,
            "n_ex": int(o.get("n_examples", 0)), "obscure": bool(o["obscure"]),
            "stems": len({stem(g) for g in order[:5]}),
        })
    n = len(rows)
    conf = np.array([r["conf"] for r in rows])
    thr = float(np.median(conf))
    lo = conf < thr
    report(f"\n  {n} reactions | median confidence {thr:.2f} | {int(lo.sum())} below, {int((~lo).sum())} above")

    # ---- THE DECISIVE ONE: is the answer present but misplaced, or absent? --------------------------
    def rec(rs, k, field="order"):
        return float(np.mean([bool(r["truth"] & set(r[field][:k])) for r in rs])) if rs else float("nan")

    hi_rows = [r for r, m_ in zip(rows, lo) if not m_]
    lo_rows = [r for r, m_ in zip(rows, lo) if m_]
    report("\n  IS THE ANSWER PRESENT BUT MISPLACED, OR ABSENT? -- the distinction that decides the work")
    report(f"    {'half':<18}{'n':>5}{'@1':>8}{'@3':>8}{'@5':>8}{'@10':>8}{'@50':>8}")
    for lab, rs in (("confident", hi_rows), ("unsure", lo_rows)):
        report(f"    {lab:<18}{len(rs):>5}{rec(rs,1):>8.3f}{rec(rs,3):>8.3f}{rec(rs,5):>8.3f}"
               f"{rec(rs,10):>8.3f}{rec(rs,50,'long'):>8.3f}")

    # ---- is confidence well-founded? ----------------------------------------------------------------
    report("\n  IS THE UNCERTAINTY WELL-FOUNDED? -- what actually differs between the halves")
    report(f"    {'feature':<28}{'confident':>12}{'unsure':>10}")
    feats = [("examples retrieved (mean)", lambda r: r["n_ex"]),
             ("obscure share", lambda r: float(r["obscure"])),
             ("distinct families in top-5", lambda r: r["stems"])]
    fres = {}
    for lab, fn in feats:
        a_, b_ = np.mean([fn(r) for r in hi_rows]), np.mean([fn(r) for r in lo_rows])
        fres[lab] = [float(a_), float(b_)]
        report(f"    {lab:<28}{a_:>12.2f}{b_:>10.2f}")

    # accuracy by example count WITHIN each half: does confidence add over just counting examples?
    report("\n  DOES CONFIDENCE ADD ANYTHING OVER JUST COUNTING EXAMPLES? -- rank-1 accuracy")
    report(f"    {'examples':<14}{'confident':>12}{'n':>5}{'unsure':>10}{'n':>5}")
    for lab, f_ in (("0", lambda r: r["n_ex"] == 0), ("1-7", lambda r: 1 <= r["n_ex"] <= 7),
                    ("8-12", lambda r: r["n_ex"] >= 8)):
        h = [r for r in hi_rows if f_(r)]
        l = [r for r in lo_rows if f_(r)]
        report(f"    {lab:<14}{rec(h,1):>12.3f}{len(h):>5}{rec(l,1):>10.3f}{len(l):>5}")

    # ---- what the model SAYS it doubts --------------------------------------------------------------
    PAT = [("which family member / paralog", r"paralog|isoform|family member|another member|similar enzyme|"
                                             r"other (slc|cyp|aldh|ugt|mmp)|same family"),
           ("compartment or localisation", r"compartment|localis|localiz|membrane|mitochond|cytosol|lumen"),
           ("substrate specificity / chain length", r"chain length|specificit|substrate preference|"
                                                    r"acyl|position|regioselect|stereo"),
           ("wrong reaction class entirely", r"not (an )?enzyme|transporter rather|spontaneous|non-enzymatic|"
                                             r"different (class|mechanism|reaction)"),
           ("no good candidate / unknown", r"unknown|not (in|among)|none of|no candidate|uncharacteri")]
    report("\n  WHAT THE MODEL SAYS IT DOUBTS -- its own one-sentence reason, categorised")
    report(f"    {'category':<40}{'confident':>11}{'unsure':>9}")
    cats = {}
    for lab, pat in PAT:
        a_ = sum(1 for r in hi_rows if re.search(pat, r["doubt"], re.I))
        b_ = sum(1 for r in lo_rows if re.search(pat, r["doubt"], re.I))
        cats[lab] = [a_, b_]
        report(f"    {lab:<40}{a_/max(len(hi_rows),1):>10.1%}{b_/max(len(lo_rows),1):>9.1%}")
    report("    (keyword categories over 283 sentences -- a crude instrument, so examples follow)")
    report("\n  UNSURE CASES, IN THE MODEL'S OWN WORDS:")
    for r in lo_rows[:6]:
        report(f"    [conf {r['conf']:.2f}] {r['doubt'][:120]}")
    report("  CONFIDENT CASES:")
    for r in hi_rows[:4]:
        report(f"    [conf {r['conf']:.2f}] {r['doubt'][:120]}")

    report("\n  READING")
    r10_lo, r10_hi = rec(lo_rows, 10), rec(hi_rows, 10)
    r1_lo = rec(lo_rows, 1)
    if r10_lo > 0.75:
        report(f"  ON THE UNSURE HALF THE ANSWER IS STILL THERE: recall@10 is {r10_lo:.3f} against a rank-1")
        report(f"  of {r1_lo:.3f}. The model is not lost -- it has the right neighbourhood and cannot pick")
        report("  the member. That is a RANKING problem, it is what a curator scanning five candidates")
        report("  solves in seconds, and it means the shortlist is the correct product rather than a")
        report("  consolation prize.")
    else:
        report(f"  ON THE UNSURE HALF THE ANSWER IS OFTEN ABSENT: recall@10 falls to {r10_lo:.3f} against")
        report(f"  {r10_hi:.3f} on the confident half. No reordering can help where the gene is not in the")
        report("  list, which retroactively explains why five re-rankers failed: they were asked to fix")
        report("  cases that were unfixable by reordering. The work is retrieval or external evidence.")
    d = fres["examples retrieved (mean)"]
    if d[0] - d[1] > 1.0:
        report(f"  AND THE UNCERTAINTY IS WELL-FOUNDED: the unsure half retrieved {d[1]:.1f} examples on")
        report(f"  average against {d[0]:.1f} for the confident half, so the model's doubt tracks the one")
        report("  variable already measured to drive accuracy rather than being a verbal tic.")
    else:
        report(f"  The two halves retrieved similar numbers of examples ({d[0]:.1f} vs {d[1]:.1f}), so the")
        report("  confidence is NOT simply reading retrieval depth -- it is judging the chemistry, which")
        report("  makes it more useful than a count anyone could compute without a model.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "cell_orphan_why_unsure", "n": n, "threshold": thr,
               "recall": {"confident": {str(k): rec(hi_rows, k) for k in (1, 3, 5, 10)},
                          "unsure": {str(k): rec(lo_rows, k) for k in (1, 3, 5, 10)}},
               "features": fres, "doubt_categories": cats, "log": log},
              open(OUT / "cell_orphan_why_unsure.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_why_unsure.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

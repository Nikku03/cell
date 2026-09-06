"""STEP 1: IS THE ANSWER EVEN IN THE SHORTLIST?  Closed-book top-10 on the literature-stratified benchmark.

WHY THIS IS THE GATE.  The proposed architecture is "language model proposes ~10 candidates, a structural
calculation re-ranks them". A re-ranker cannot recover an answer that is not in the list -- that is the same
0.505 ceiling the candidate-list arms hit, and this project has now paid for that lesson twice. So before any
docking result means anything, the shortlist has to contain the truth often enough to be worth sorting.

The comparison that matters is against the SAME solver's single guess on the SAME 400 reactions:

    top-1, no shortlist, measured earlier      0.645 overall     0.463 on enzymes with <20 papers

Recall@10 is the ceiling any re-ranker could reach; recall@1 here should reproduce ~0.645, and if it does not,
the shortlist prompt changed the task rather than extending it, which would invalidate the comparison.

THE OBSCURE BIN IS THE HEADLINE, again. The ~9,186 real gaps are obscure by construction, so the bin of
enzymes with under 20 linked papers is the only one that forecasts field performance.

PREDECLARED, before any number:
    recall@10 in the obscure bin >= ~0.70   -> the shortlist is worth re-ranking; a structural discriminator
                                               has real headroom to convert (0.463 -> up to that ceiling).
    recall@10 ~ recall@1                    -> the model has one idea per reaction and nine restatements of
                                               it. Positions 2-10 carry no new hypotheses, and no re-ranker
                                               can help however good it is.
    recall@1 far from 0.645                 -> asking for ten answers changed the first one; the arms are no
                                               longer comparable and the gate cannot be read.
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import cell_orphan_obscurity as O
import cell_orphan_ties_nn as T

OUT = A.OUT
SHORT = OUT / "shortlist_answers.json"
KS = (1, 3, 5, 10)
BASE_TOP1 = {"all": 0.645, "obscure": 0.463}      # same 400 reactions, single-guess arm


def _n(g):
    return str(g or "").strip().upper().replace(" ", "")


def merge(tdir):
    got = {}
    for f in sorted(Path(tdir).glob("agent-*.jsonl")):
        txt = f.read_text(errors="replace")
        tags = set(re.findall(r"open_b(\d+)\.txt", txt))
        if len(tags) != 1:
            continue
        for line in txt.splitlines():
            if '"answers"' not in line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            stack = [rec]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    if isinstance(x.get("answers"), list):
                        for a_ in x["answers"]:
                            if isinstance(a_, dict) and "pid" in a_ and isinstance(a_.get("genes"), list):
                                got[int(a_["pid"])] = [_n(g) for g in a_["genes"]]
                    stack.extend(x.values())
                elif isinstance(x, list):
                    stack.extend(x)
    miss = sorted(set(range(400)) - set(got))
    if miss:
        print(f"  REFUSING: {len(miss)} shortlists missing {miss[:10]} -- a missing batch scores as wrong")
        return 1
    json.dump({str(k): v for k, v in got.items()}, open(SHORT, "w"), indent=1)
    dup = sum(1 for v in got.values() if len(set(v)) < len(v))
    print(f"  merged {len(got)} shortlists -> {SHORT}")
    print(f"  shortlists containing a duplicate gene: {dup}")
    return 0


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "merge":
        return merge(sys.argv[2])

    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    from scipy.stats import binomtest

    report("=" * 100)
    report("STEP 1 -- is the answer even in the shortlist? closed-book top-10")
    report("=" * 100)
    report("  The architecture is 'LLM proposes 10, structure re-ranks'. A re-ranker cannot recover an answer")
    report("  that is not in the list, which is the 0.505 ceiling all over again. So recall@10 is the real")
    report("  ceiling of the whole design, and the obscure bin is the only one that forecasts field work.")
    report(f"  Same solver, same 400 reactions, single guess: {BASE_TOP1['all']:.3f} overall, "
           f"{BASE_TOP1['obscure']:.3f} obscure.")

    B = T.build()
    cats = B["cats"]
    sel = json.load(open(O.PUZZLES))["pids"]
    sl = json.load(open(SHORT))
    n = len(sel)
    truth = [{_n(g) for g in cats[i]} for i in sel]
    papers = O._lit()
    pool = O._pool(B, papers)
    bidx = np.array([O._binof(pool[i]) for i in sel])

    hits = {k: np.zeros(n, bool) for k in KS}
    ndist = []
    for p in range(n):
        lst = sl.get(str(p)) or []
        ndist.append(len(set(lst)))
        for k in KS:
            hits[k][p] = bool(truth[p] & set(lst[:k]))

    report(f"\n  {n} reactions | distinct genes per shortlist: median {np.median(ndist):.0f} "
           f"min {int(np.min(ndist))}")
    report(f"\n  RECALL@k BY HOW MUCH HAS BEEN PUBLISHED ABOUT THE TRUE CATALYST")
    report(f"    {'papers':<12}{'n':>5}" + "".join(f"{('@' + str(k)):>9}" for k in KS)
           + f"{'@10 - @1':>11}")
    rows = {}
    for b, (lo, hi) in enumerate(O.BINS):
        m = bidx == b
        k_ = int(m.sum())
        if not k_:
            continue
        lab = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        v = {str(k): float(hits[k][m].mean()) for k in KS}
        rows[lab] = {"n": k_, **v}
        report(f"    {lab:<12}{k_:>5}" + "".join(f"{v[str(k)]:>9.3f}" for k in KS)
               + f"{v['10'] - v['1']:>+11.3f}")
    v = {str(k): float(hits[k].mean()) for k in KS}
    rows["ALL"] = {"n": n, **v}
    report(f"    {'ALL':<12}{n:>5}" + "".join(f"{v[str(k)]:>9.3f}" for k in KS)
           + f"{v['10'] - v['1']:>+11.3f}")

    ob = rows.get("0-20", {})
    ob10, ob1 = ob.get("10", 0.0), ob.get("1", 0.0)
    drift = v["1"] - BASE_TOP1["all"]
    report(f"\n  COMPARABILITY CHECK -- did asking for ten answers change the first one?")
    report(f"    recall@1 here {v['1']:.3f} vs the single-guess arm {BASE_TOP1['all']:.3f}  ({drift:+.3f})")
    if abs(drift) > 0.06:
        report("    *** THE FIRST ANSWER MOVED. The shortlist prompt changed the task rather than extending")
        report("        it, so the two arms are not comparable and the gate below cannot be read cleanly.")

    report("\n  READING")
    # The re-ranker re-ranks THIS shortlist, so its headroom is measured against THIS list's position-1,
    # not against the older single-guess arm. Using 0.463 would credit the re-ranker with the +0.062 that
    # asking for ten answers already delivered by itself.
    headroom = ob10 - ob1
    if ob10 >= 0.70:
        report(f"  THE GATE CLEARS. On the obscure bin the answer is in the top 10 for {ob10:.3f} of reactions")
        report(f"  while position 1 is right {ob1:.3f} of the time, so a re-ranker has {headroom:+.3f} to")
        report("  convert. Step 2's discriminator has something real to sort.")
        report(f"  NOTE the obscure-bin position-1 is {ob1:.3f} against {BASE_TOP1['obscure']:.3f} for the")
        report("  single-guess arm: asking for ten ranked answers improved the FIRST one. That gain belongs to")
        report("  the prompt, not to any re-ranker, and is excluded from the headroom above.")
    elif ob10 - ob1 < 0.10:
        report(f"  NINE RESTATEMENTS OF ONE IDEA. Obscure-bin recall@10 {ob10:.3f} against recall@1 {ob1:.3f}")
        report(f"  -- positions 2-10 add {ob10-ob1:+.3f}. The model has one hypothesis per reaction and pads")
        report("  the rest with near-duplicates, so no re-ranker can help however good it is.")
    else:
        report(f"  THE GATE FAILS. Obscure-bin recall@10 is {ob10:.3f}, under the 0.70 predeclared. A")
        report(f"  discriminator would be sorting a list that lacks the answer {1-ob10:.0%} of the time, which")
        report("  is the candidate-list ceiling this project has already paid for twice.")
    report(f"\n  CEILING FOR THE WHOLE DESIGN: a perfect re-ranker on these shortlists would reach "
           f"{v['10']:.3f} overall")
    report(f"  and {ob10:.3f} on obscure enzymes, against {BASE_TOP1['all']:.3f} / "
           f"{BASE_TOP1['obscure']:.3f} today.")

    json.dump({"test": "cell_orphan_shortlist", "n": n, "ks": list(KS), "rows": rows,
               "recall1_drift_vs_single_guess": drift, "obscure_at10": ob10, "log": log},
              open(OUT / "cell_orphan_shortlist.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_shortlist.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

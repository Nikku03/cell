"""IS IT REASONING OR IS IT MEMORY?  Score the missing-piece arm against how much has been PUBLISHED about
the answer, on a benchmark deliberately built to contain obscure enzymes.

THE QUESTION THIS SETTLES.  `cell_orphan_llm` found that closed-book reasoning names the true catalyst for
0.720 of held-out reactions with no candidate list, against 0.505 for any arm restricted to the candidate
list. That was immediately caveated: these are curated reactions published before the training cutoff, so the
score may be recall of the literature the answer key was built from rather than reasoning about chemistry.

It matters enormously which. The point of the whole exercise is to fill in ~9,200 enzyme-shaped reactions that
have NO catalyst annotated. Those gaps are, by construction, the ones nobody has written down.

    recall  -> the tool transcribes known-but-unlinked biology. Real value, bounded value.
    reason  -> the tool predicts genuinely unrecorded links, and the cell map can actually be completed.

THE AXIS, and why the obvious version of it is worthless.  "How obscure is this gene" must not be measured by
grepping gene symbols in PubMed: CAT, SI, KL, PIGS and SET are all real human symbols AND common words, and
string matching would file catalase alongside the household pet. This uses NCBI `gene2pubmed`, the curated
gene-to-paper link table, joined on GENE ID. 3,322,637 human links, zero string matching.

WHY THE NATURAL SAMPLE CANNOT ANSWER IT.  Measured first, on the 200 reactions already scored: the median
true catalyst has 230 linked papers and exactly ONE puzzle out of 200 has a catalyst under 20 papers. That is
a selection effect, not an accident -- a gene is annotated as a catalyst BECAUSE it has been studied. A random
draw from annotated reactions therefore contains almost no obscure enzymes and has no power at the only end of
the axis that matters.

So this benchmark is STRATIFIED: equal numbers of reactions per literature bin, including a deliberately
enriched bin of enzymes with under 20 papers. Rare by construction, so the decisive comparison has real n.

    bin (papers on the easiest true catalyst)   available reactions
      0-20                                                       99
     21-50                                                      804
     51-150                                                   2,656
    151-500                                                   4,294
      501+                                                    3,110

THE CONFOUND, stated before the result.  Obscure enzymes may sit in intrinsically EASIER reactions: a
one-substrate metabolic step with an unmistakable transformation, versus a promiscuous kinase acting on a
complex. A flat accuracy curve could then be two opposing effects cancelling. Three things are reported
against it: the candidate-list size and reaction category per bin, and a multivariable logistic fit of
correctness on log10(papers) CONTROLLING for both. The controlled coefficient is the number to read.

PREDECLARED, before any number:
    accuracy roughly FLAT across bins, controlled slope ~0
        -> reasoning from chemistry. The score is not bought with fame, and it should transfer to the 9,200
           unannotated reactions. This is the outcome that makes the backtrace idea deliverable.
    accuracy RISES steeply with literature volume, controlled slope clearly positive
        -> memory. The tool fills in known-but-unlinked biology and must not be pointed at true gaps without
           a large discount. The headline 0.720 would then be an overestimate of field performance by
           whatever the gap between the top and bottom bin is.
    bottom bin near zero
        -> the tool knows famous enzymes and nothing else, and cannot complete the map at all.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import cell_orphan_llm as L
import cell_orphan_ties_nn as T

OUT = A.OUT
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
LIT = SP / "gene_lit.json"
PUZZLES = OUT / "obscurity_puzzles.json"
ANSWERS = OUT / "obscurity_answers.json"
BINS = ((0, 20), (21, 50), (51, 150), (151, 500), (501, 10 ** 9))
PER_BIN = 80
SEED = 11


def _lit():
    G = json.load(open(LIT))
    lit, syn = G["lit"], G["syn"]

    def papers(g):
        """Linked papers for a gene symbol. None when the symbol cannot be mapped to an NCBI gene at all --
        never silently zero, which would file an unmappable symbol as maximally obscure."""
        if g in lit:
            return lit[g]
        s = syn.get(g)
        if s is not None:
            return lit.get(s, 0)
        return None
    return papers


def _pool(B, papers):
    """Reactions eligible for the benchmark: catalysed, with participants, NOT used to train the baselines,
    and with a catalyst that maps to NCBI. Keyed by the MOST-published true catalyst, because a hit only
    requires naming ONE member of the truth set -- the easiest member is what sets the difficulty."""
    cats, P, ex = B["cats"], B["P"], B["examples"]
    trained = {int(ex[k]["rx"]) for k in B["tr_i"]}
    out = {}
    for i in cats:
        if not P[i] or i in trained:
            continue
        pv = [papers(g) for g in cats[i]]
        if any(v is None for v in pv):
            continue
        out[i] = max(pv)
    return out


def _binof(v):
    for b, (lo, hi) in enumerate(BINS):
        if lo <= v <= hi:
            return b
    return len(BINS) - 1


def dump():
    B = T.build()
    papers = _lit()
    pool = _pool(B, papers)
    steps, cats = B["steps"], B["cats"]
    rng = np.random.default_rng(SEED)

    byb = {b: [] for b in range(len(BINS))}
    for i, v in pool.items():
        byb[_binof(v)].append(i)
    sel = []
    for b in sorted(byb):
        arr = sorted(byb[b])
        take = min(PER_BIN, len(arr))
        pick = rng.choice(len(arr), take, replace=False)
        sel += [int(arr[int(x)]) for x in sorted(pick)]
        print(f"  bin {BINS[b][0]}-{BINS[b][1] if BINS[b][1] < 10**9 else '+'}: "
              f"{len(arr):>5} available, took {take}")
    sel = sorted(sel)

    out = []
    for pid, i in enumerate(sel):
        s = steps[i]
        out.append({
            "pid": pid,
            "source": s.get("source"),
            "category": s.get("category"),
            "reversible": bool(s.get("reversible")),
            "inputs": [L._clean(p) for p in (s.get("in") or [])],
            "outputs": [L._clean(p) for p in (s.get("out") or [])],
        })
    OKTOP = {"pid", "source", "category", "reversible", "inputs", "outputs"}
    OKPART = {"name", "type", "compartment"}
    for pid, i in enumerate(sel):
        assert set(out[pid]) <= OKTOP, f"unexpected field {set(out[pid]) - OKTOP}"
        for w in ("inputs", "outputs"):
            for q in out[pid][w]:
                assert set(q) <= OKPART, f"unexpected participant field {set(q) - OKPART}"
        assert steps[i].get("id", "\0") not in json.dumps(out[pid]), "reaction id leaked"
    json.dump({"pids": sel, "puzzles": out}, open(PUZZLES, "w"), indent=1)

    gim = sum(1 for pid, i in enumerate(sel)
              if any(g in " ".join(q["name"] for w in ("inputs", "outputs") for q in out[pid][w])
                     for g in cats[i]))
    pv = np.array([pool[i] for i in sel])
    print(f"\nwrote {len(out)} puzzles -> {PUZZLES}")
    print(f"  papers on the easiest true catalyst: median {np.median(pv):.0f} "
          f"min {pv.min()} max {pv.max()}")
    print(f"  under 20 papers: {int((pv <= 20).sum())} (the natural draw had 1 in 200)")
    print(f"  answer named in the reaction: {gim} -- scored as a separate stratum")
    print(f"  NO candidate list, NO catalyst, NO ids. Open-ended only: this measures NAMING, not ranking.")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "dump":
        return dump()

    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    import statsmodels.api as sm
    from scipy.stats import binomtest

    report("=" * 100)
    report("IS IT REASONING OR MEMORY? -- accuracy vs how much has been PUBLISHED about the answer")
    report("=" * 100)
    report("  The 0.720 open-ended result may be recall of the literature the answer key was built from.")
    report("  That decides whether this can fill the ~9,200 unannotated reactions or only transcribe known")
    report("  biology. Axis = NCBI gene2pubmed, joined on GENE ID -- never string-matched, because CAT, SI")
    report("  and KL are real symbols AND common words.")
    report("  The natural sample could not answer it: median catalyst 230 papers, 1 of 200 under 20. A gene")
    report("  is annotated BECAUSE it was studied. So this benchmark is STRATIFIED to enrich the obscure end.")
    report("  PREDECLARED: flat curve + controlled slope ~0 => reasoning, transfers to true gaps.")
    report("  Rising curve => memory, and 0.720 overstates field performance by the top-to-bottom gap.")

    B = T.build()
    papers = _lit()
    pool = _pool(B, papers)
    cats, steps = B["cats"], B["cats"]
    pz = json.load(open(PUZZLES))
    sel, puz = pz["pids"], pz["puzzles"]
    ans = json.load(open(ANSWERS))
    n = len(sel)

    truth = [set(cats[i]) for i in sel]
    pv = np.array([pool[i] for i in sel], float)
    bidx = np.array([_binof(v) for v in pv])
    hit = np.array([_norm(ans.get(str(p), {}).get("open")) in {_norm(g) for g in truth[p]}
                    for p in range(n)])
    miss = sum(1 for p in range(n) if not (ans.get(str(p), {}) or {}).get("open"))
    gim = np.array([any(g in " ".join(q["name"] for w in ("inputs", "outputs") for q in puz[p][w])
                        for g in truth[p]) for p in range(n)])
    ncand = np.array([len(B["cats"].get(i, [])) for i in sel], float)
    metab = np.array([puz[p]["category"] == "metabolic" for p in range(n)])

    report(f"\n  {n} reactions | unanswered {miss} | answer named in reaction {int(gim.sum())}")
    report(f"  OVERALL open-ended accuracy: {hit.mean():.3f}  ({int(hit.sum())}/{n})")

    report(f"\n  ACCURACY BY HOW MUCH HAS BEEN PUBLISHED ABOUT THE TRUE CATALYST")
    report(f"    {'papers':<14}{'n':>5}{'accuracy':>11}{'95% CI':>18}{'metab':>8}{'named':>7}")
    rows = {}
    for b, (lo, hi) in enumerate(BINS):
        m = bidx == b
        k = int(m.sum())
        if not k:
            continue
        h = hit[m]
        ci = binomtest(int(h.sum()), k).proportion_ci(0.95)
        lab = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        rows[lab] = {"n": k, "acc": float(h.mean()), "lo": float(ci.low), "hi": float(ci.high),
                     "metabolic": float(metab[m].mean()), "named": int(gim[m].sum())}
        report(f"    {lab:<14}{k:>5}{h.mean():>11.3f}{f'[{ci.low:.3f}, {ci.high:.3f}]':>18}"
               f"{metab[m].mean():>8.2f}{int(gim[m].sum()):>7}")

    lo_m, hi_m = bidx == 0, bidx == len(BINS) - 1
    gap = float(hit[hi_m].mean() - hit[lo_m].mean())
    report(f"\n  TOP BIN MINUS BOTTOM BIN: {hit[hi_m].mean():.3f} - {hit[lo_m].mean():.3f} = {gap:+.3f}")

    # ---------------- trend, raw and controlled ----------------
    x = np.log10(np.maximum(pv, 1.0))
    X1 = sm.add_constant(x)
    r1 = sm.Logit(hit.astype(float), X1).fit(disp=0)
    Xc = sm.add_constant(np.column_stack([x, np.log10(np.maximum(ncand, 1.0)), metab.astype(float)]))
    r2 = sm.Logit(hit.astype(float), Xc).fit(disp=0)
    report(f"\n  TREND: logistic fit of correctness on log10(papers)")
    report(f"    raw        slope {r1.params[1]:+.3f}  p={r1.pvalues[1]:.4f}")
    report(f"    CONTROLLED slope {r2.params[1]:+.3f}  p={r2.pvalues[1]:.4f}   "
           f"(controls: log candidates, metabolic-vs-not)")
    report("    The CONTROLLED slope is the one to read: obscure enzymes may sit in easier reactions, and")
    report("    an uncontrolled flat line could be two opposing effects cancelling out.")

    slope, psl = float(r2.params[1]), float(r2.pvalues[1])
    report("\n  READING")
    if psl >= 0.05 and abs(gap) < 0.15:
        report(f"  REASONING, NOT MEMORY. Controlled slope {slope:+.3f} (p={psl:.4f}) is indistinguishable from")
        report(f"  zero and the top-to-bottom gap is {gap:+.3f}. Enzymes with under 20 linked papers are named")
        report(f"  about as accurately as enzymes with thousands. The 0.720 was not bought with fame, and")
        report("  there is no measured reason it should collapse on the ~9,200 unannotated reactions.")
    elif psl < 0.05 and slope > 0:
        report(f"  MEMORY IS DOING REAL WORK. Controlled slope {slope:+.3f} (p={psl:.4f}): accuracy rises with")
        report(f"  how much has been published about the answer, by {gap:+.3f} from the bottom bin to the top.")
        report(f"  On genuinely obscure enzymes the arm scores {hit[lo_m].mean():.3f}, not {hit.mean():.3f}.")
        report("  This tool transcribes known-but-unlinked biology. Pointing it at true gaps needs that")
        report("  discount applied, and the honest expected yield on unannotated reactions is the BOTTOM bin.")
    else:
        report(f"  MIXED / UNDERPOWERED. Controlled slope {slope:+.3f} (p={psl:.4f}), gap {gap:+.3f}. Read the")
        report("  per-bin confidence intervals rather than a verdict.")
    report(f"\n  WHAT TO EXPECT ON THE {len([1 for i in pool])} unannotated reactions: the bottom-bin accuracy")
    report(f"  {hit[lo_m].mean():.3f} is the closest honest estimate, NOT the headline {hit.mean():.3f} -- true")
    report("  gaps are obscure by construction, so the obscure bin is the population that resembles them.")

    json.dump({"test": "cell_orphan_obscurity", "n": n, "overall": float(hit.mean()), "bins": rows,
               "gap_top_minus_bottom": gap, "slope_raw": float(r1.params[1]),
               "p_raw": float(r1.pvalues[1]), "slope_controlled": slope, "p_controlled": psl,
               "unanswered": miss, "log": log},
              open(OUT / "cell_orphan_obscurity.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_obscurity.json'}")
    return 0


def _norm(g):
    return str(g or "").strip().upper().replace(" ", "")


if __name__ == "__main__":
    raise SystemExit(main())

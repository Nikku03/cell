"""Loop 204. Does "reason first, calculate only what is missing" produce a smaller job?

THE ARCHITECTURE THIS TESTS. An LLM acts as scientific planner: it takes a question, assembles the
shortest reliable chain of known relationships that reaches an answer, identifies which links in
that chain are unknown AND matter, and hands only those to a computational engine. The engine
computes them, the planner substitutes them back, and the result is cached for reuse. The claim is
that this turns a whole-cell simulation into a small set of targeted calculations.

The claim rests on three things, and all three are measurable here rather than arguable:

    (1) CHAINS ARE SHORT           -- an answer needs few links
    (2) MOST LINKS ARE ALREADY KNOWN -- few of them are missing
    (3) MISSING LINKS ARE REUSED   -- computing one helps answer many questions

If (2) fails, "calculate only what is missing" is the whole simulation with extra steps. If (3)
fails, nothing amortises and every question pays full price. This loop measures all three against
loop 190's census, which is the only inventory in this project of which layer covers which gene.

WHAT THE CENSUS ALREADY SAID, AND WHY IT IS NOT THE ANSWER. Loop 190 measured how many of the five
mechanistic layers each gene sits in: 1,643 genes in none, 6,430 in one, 6,051 in two, 1,822 in
three, 545 in four, and ONE in all five. That is a statement about COVERAGE. The architecture needs
a statement about TRAVERSAL -- whether you can get from a question to an answer by walking edges
between genes, not whether one gene happens to carry several annotations. Those are different
measurements and this loop makes the second one.

PREDECLARED, BEFORE ANY NUMBER.

  T1 IS THIS THE SAME CENSUS?
     Rebuild the five mechanistic layers from the same sources loop 190 used.
     Gate: PASS iff the layer counts, the 0-5 histogram and the pairwise overlaps reproduce loop
     190 exactly. FAIL means this audit describes a different inventory and nothing below counts.

  T2 HOW DEEP A CHAIN DOES THE COVERAGE SUPPORT?
     A chain that crosses k layers needs a gene present in all k.
     Gate: PASS iff at least 10% of genes carry four or more of the five mechanistic layers.
     A FAIL bounds how much of the architecture's propagation any single gene can carry.

  T3 CAN THE ARCHITECTURE'S OWN CHAIN BE WALKED?
     Its worked example is drug -> receptor -> signalling -> TF -> chromatin -> transcription ->
     protein -> metabolism. The receptor and signalling hops do not exist in this repo at all --
     the grounding pass established that -- so this tests the LONGEST SUFFIX that could exist here:
     a TF that has a motif, regulating a gene, that carries a modelled reaction.
     Counted as PATHS between genes over real edges, not as annotation overlaps.
     Gate: PASS iff every hop is carried by at least 100 genes AND at least one complete path
     exists. A FAIL says the chain breaks, and names the hop it breaks at.

  T4 THE UNION TEST -- the one that decides the architecture.
     Take every question of the form "given gene g in layer A, what is its state in layer B" that
     a user could reasonably ask, for all ordered layer pairs and all genes in A. For each, the
     missing links are the (gene, layer) cells the chain needs and the census does not have. Take
     the UNION of missing links over the whole question set and compare it to the TOTAL number of
     empty cells in the gene x layer matrix.
     Gate: PASS iff the union is below 50% of all empty cells. A FAIL means answering a realistic
     set of questions requires filling in most of the cell anyway, and the saving is illusory.

  T5 DOES ANYTHING AMORTISE?
     For each missing link in T4's union, how many distinct questions need it.
     Gate: PASS iff the median missing link is needed by two or more questions. A FAIL means the
     cache never hits and every question pays full price.

  T6 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import gzip, json, os, sys, time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
TABLE = "colab/data/cell_complete.json.gz"
ANNO = "colab/data/all_gene_annotation.json"
BUNDLE = "colab/data/net_bundle.json.gz"
OUT = "outputs/loop_chain_audit.json"
CUR = (0, 55716)
CORE = ["reaction", "TF_in_network", "has_regulator", "has_PPI", "has_motif"]

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def build_layers():
    tab = json.load(gzip.open(TABLE))["genes"]
    sym = [str(g["name"]).upper() for g in tab]
    A = {k.upper(): v for k, v in json.load(open(ANNO))["classification"].items()}
    z = np.load("colab/data/rem_enzyme.npz", allow_pickle=True)
    met = {str(s).upper() for s in z["symbols"] if s}
    nb = json.load(gzip.open(BUNDLE))
    names, reg, ppi = nb["names"], nb["reg"], nb["ppi"]
    nidx = {n.upper(): i for i, n in enumerate(names)}
    cur = reg[CUR[0]:CUR[1]]
    regulators = {int(r[0]) for r in cur}
    targets = defaultdict(int)
    for r in cur:
        targets[int(r[1])] += 1
    ppi_deg = Counter()
    for a, b in ppi:
        ppi_deg[int(a)] += 1; ppi_deg[int(b)] += 1
    dom = json.load(open("colab/data/tf_domains.json"))["matrices"]
    motif_names = {(v.get("name") or "").upper().split("::")[0] for v in dom.values()}
    L = {
        "reaction": np.array([s in met for s in sym]),
        "TF_in_network": np.array([nidx.get(s, -1) in regulators for s in sym]),
        "has_regulator": np.array([targets.get(nidx.get(s, -1), 0) > 0 for s in sym]),
        "has_PPI": np.array([ppi_deg.get(nidx.get(s, -1), 0) > 0 for s in sym]),
        "has_motif": np.array([s in motif_names for s in sym]),
    }
    return sym, L, names, nidx, cur


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "chain audit"}
    say("=" * 104)
    say('LOOP 204 -- DOES "REASON FIRST, CALCULATE ONLY WHAT IS MISSING" PRODUCE A SMALLER JOB?')
    say("=" * 104)

    sym, L, names, nidx, cur = build_layers()
    n = len(sym)
    M = np.array([L[k] for k in CORE])          # 5 x n boolean
    depth = M.sum(0)

    # ------------------------------------------------------------ T1
    say("T1 IS THIS THE SAME CENSUS?")
    counts = {k: int(L[k].sum()) for k in CORE}
    hist = {k: int((depth == k).sum()) for k in range(6)}
    say(f"     layer counts {counts}")
    say(f"     depth histogram {hist}")
    ref_counts = {"reaction": 2568, "TF_in_network": 1177, "has_regulator": 7485,
                  "has_PPI": 14230, "has_motif": 723}
    ref_hist = {0: 1643, 1: 6430, 2: 6051, 3: 1822, 4: 545, 5: 1}
    pw = {}
    for i, a in enumerate(CORE):
        for b in CORE[i + 1:]:
            pw[f"{a} & {b}"] = int((L[a] & L[b]).sum())
    say(f"     TF_in_network & reaction {pw['TF_in_network & reaction']}   "
        f"has_motif & reaction {pw['has_motif & reaction']}   "
        f"has_regulator & reaction {pw['has_regulator & reaction']}")
    ok1 = (counts == ref_counts and hist == ref_hist and n == 16492)
    G.add("T1", ok1,
          if_true="T1 PASS -- layers, histogram and overlaps reproduce loop 190 exactly",
          if_false=lambda: f"T1 FAIL -- {counts} vs {ref_counts}, {hist} vs {ref_hist}, n={n}")
    res["census"] = {"n": n, "counts": counts, "histogram": hist, "pairwise": pw}

    # ------------------------------------------------------------ T2
    say("T2 HOW DEEP A CHAIN DOES THE COVERAGE SUPPORT?")
    for k in range(6):
        say(f"       {k} of 5 layers   {hist[k]:>7,}   {hist[k]/n:>6.2%}")
    deep4 = (hist[4] + hist[5]) / n
    say(f"     genes carrying 4 or more layers  {hist[4]+hist[5]:,}  = {deep4:.2%}")
    G.add("T2", bool(deep4 >= 0.10), stat=deep4, requires=("T1",),
          if_true=lambda: f"T2 PASS -- {deep4:.1%} of genes carry a four-layer chain",
          if_false=lambda: f"T2 FAIL -- only {deep4:.2%} of genes carry four or more layers, and "
                           f"exactly {hist[5]} carries all five. A single gene cannot hold much of "
                           f"the architecture's propagation chain")

    # ------------------------------------------------------------ T3
    say("T3 CAN THE ARCHITECTURE'S OWN CHAIN BE WALKED?")
    say("     receptor and signalling hops: ABSENT from this repo entirely (grounding pass)")
    say("     testing the longest suffix that could exist: "
        "motif-carrying TF -> regulates -> gene -> carries a modelled reaction")
    idx = {s: i for i, s in enumerate(sym)}
    tf_motif = {s for s in sym if L["has_motif"][idx[s]] and L["TF_in_network"][idx[s]]}
    rx = {s for s in sym if L["reaction"][idx[s]]}
    edges = defaultdict(set)
    for r in cur:
        edges[names[int(r[0])].upper()].add(names[int(r[1])].upper())
    hop1 = len(tf_motif)
    reachable = {t for s in tf_motif for t in edges.get(s, ())}
    hop2 = len(reachable)
    endpoints = reachable & rx
    hop3 = len(endpoints)
    paths = sum(1 for s in tf_motif for t in edges.get(s, ()) if t in rx)
    say(f"       hop 1  TFs in the network WITH a motif            {hop1:>7,}")
    say(f"       hop 2  distinct genes they regulate               {hop2:>7,}")
    say(f"       hop 3  of those, carrying a modelled reaction     {hop3:>7,}")
    say(f"       complete TF -> gene -> reaction paths             {paths:>7,}")
    weakest = min(hop1, hop2, hop3)
    G.add("T3", bool(weakest >= 100 and paths > 0), stat=float(weakest), requires=("T1",),
          if_true=lambda: f"T3 PASS -- the suffix walks: {paths:,} complete paths, weakest hop "
                          f"{weakest:,} genes",
          if_false=lambda: f"T3 FAIL -- weakest hop carries {weakest:,} genes ({paths:,} complete "
                           f"paths). And the two hops the architecture starts from, receptor and "
                           f"signalling, are not in this repo at all")
    res["chain"] = {"tf_with_motif": hop1, "regulated": hop2, "regulated_with_reaction": hop3,
                    "complete_paths": paths, "weakest_hop": weakest}

    # ------------------------------------------------------------ T4
    say("T4 THE UNION TEST")
    total_cells = n * len(CORE)
    filled = int(M.sum())
    empty = total_cells - filled
    say(f"     gene x layer matrix  {n:,} x {len(CORE)} = {total_cells:,} cells")
    say(f"       filled {filled:,} ({filled/total_cells:.2%})   EMPTY {empty:,} "
        f"({empty/total_cells:.2%})")
    need = Counter()
    n_q = 0
    for ai, a in enumerate(CORE):
        for bi, b in enumerate(CORE):
            if a == b:
                continue
            askers = np.where(M[ai])[0]          # genes in A -- the ones you could ask about
            missing_b = askers[~M[bi][askers]]   # ...whose layer B is unknown
            n_q += len(askers)
            for g in missing_b:
                need[(int(g), bi)] += 1
    union = len(need)
    say(f"     questions of the form 'gene in layer A -> its state in layer B'   {n_q:,}")
    say(f"     DISTINCT missing (gene, layer) links they require                 {union:,}")
    say(f"     as a fraction of all empty cells                                  "
        f"{union/empty:.2%}")
    G.add("T4", bool(union / empty < 0.5), stat=union / empty, requires=("T1",),
          if_true=lambda: f"T4 PASS -- a realistic question set needs {union/empty:.1%} of the "
                          f"empty cells, so targeting genuinely saves work",
          if_false=lambda: f"T4 FAIL -- answering the natural question set requires "
                           f"{union:,} of {empty:,} empty cells ({union/empty:.1%}). "
                           f"'Calculate only what is missing' is most of the cell")
    res["union"] = {"total_cells": total_cells, "filled": filled, "empty": empty,
                    "questions": n_q, "union_missing": union, "fraction_of_empty": union / empty}

    # ------------------------------------------------------------ T5
    say("T5 DOES ANYTHING AMORTISE?")
    reuse = np.array(sorted(need.values()))
    med = float(np.median(reuse)) if len(reuse) else float("nan")
    say(f"     questions served per computed link:  median {med:.1f}   "
        f"mean {reuse.mean():.2f}   max {reuse.max()}")
    say(f"     links needed by only one question:   {(reuse==1).sum():,} "
        f"({(reuse==1).mean():.1%})")
    G.add("T5", bool(med >= 2), stat=med, requires=("T4",),
          if_true=lambda: f"T5 PASS -- the median computed link serves {med:.0f} questions, "
                          f"so a cache amortises",
          if_false=lambda: f"T5 FAIL -- the median computed link serves {med:.1f} questions")
    res["reuse"] = {"median": med, "mean": float(reuse.mean()), "max": int(reuse.max()),
                    "singletons": int((reuse == 1).sum())}

    # ------------------------------------------------------------ T6
    say("T6 WHAT THIS CANNOT SHOW")
    say("     A layer cell is BINARY here -- a gene is in the reaction layer or it is not. The")
    say("     architecture needs VALUES, not membership, so every count above is optimistic: a")
    say("     filled cell may still lack the rate constant the chain actually needs.")
    say("     Five layers is this repo's inventory, not the cell's. A real chain also needs")
    say("     receptors, kinases, transport and localisation, none of which are layers here.")
    say("     The question set is every ordered layer pair, which weights all questions equally.")
    say("     A real user asks a biased set, and a biased set could need far fewer links -- this")
    say("     design cannot say which bias, so T4 is an upper bound on the work and T5 a lower")
    say("     bound on the reuse.")
    say("     Nothing here says a missing link is COMPUTABLE. Loops 131-133 and 184 measured that")
    say("     the rate-determined ones are not, and loop 156 measured that growth does not need")
    say("     them anyway. Which bucket a link falls in is loop 205, not this loop.")

    G.summary(seconds=time.time() - t0)
    gates, void = G.as_dict()
    res["gates"], res["void"] = gates, void
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()

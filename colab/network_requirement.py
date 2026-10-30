"""WHAT WOULD THE NETWORK ACTUALLY NEED? -- a specification, derived rather than guessed.

The pivot tests killed the shortlist formulation: network AUC 0.5080 against exactly-matched decoys, and in
84.7% of shortlists the knocked-out gene had NO edge to any candidate. "The network is too sparse" is the
obvious diagnosis and it is not actionable. This turns it into a number: how many edges per gene, at what
precision, would be needed to reach the frequency baseline -- and is any of that attainable.

FOUR THINGS ARE SEPARABLE, and conflating them is why "add more edges" is not yet a plan:

  1 NODE COVERAGE      is the knocked-out gene in the network at all? is the mover?
  2 EDGE COVERAGE      of a knockout's true movers, how many are reachable by ANY edge?  <- the recall ceiling
  3 RANKING            among the genes it does reach, can it order them?                 <- measured separately
  4 PRECISION          of the genes it reaches, how many are actually movers?

If (2) is the binding constraint, no reweighting, embedding or architecture fixes it -- the answer simply is
not in the graph. This computes the ceiling under PERFECT ranking within the current edges, which upper-bounds
everything any model can do with this network, and then solves for the (degree x precision) a network would
need to match frequency.

THE CEILING IS THE POINT. A model can only rank what it can see.
"""
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
MIN_SPEC, NPICK = 5, 50
R = {}


def hdr(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


def main():
    import cellformer_af as CF
    d = CF.load_all()
    spec, nspec, tide, gi, kos, genes = d["spec"], d["nspec"], d["tide"], d["gi"], d["kos"], d["genes"]
    tr = np.array([int(hashlib.md5(k.encode()).hexdigest(), 16) % 2 == 0 for k in kos])
    test = [i for i in range(len(kos)) if not tr[i] and nspec[i] >= MIN_SPEC]
    freq = spec[tr].mean(0)

    adj = collections.defaultdict(set)
    adjc = [collections.defaultdict(set) for _ in range(5)]
    for (i, j), v in d["pair"].items():
        for c in range(5):
            if v[c]:
                adjc[c][i].add(j)
                adjc[c][j].add(i)
                adj[i].add(j)
                adj[j].add(i)

    # ---------------------------------------------------------------- 1. NODE COVERAGE
    hdr("1. NODE COVERAGE -- is the gene in the graph at all?")
    innet = set(adj)
    ko_in = [q for q in test if gi.get(kos[q]) is not None]
    ko_edge = [q for q in ko_in if gi[kos[q]] in innet]
    moved = np.where(spec.any(0))[0]
    moved_nt = [g for g in moved if not tide[g]]
    print(f"held-out scorable perturbations         {len(test):,}")
    print(f"  KO gene present in the readout        {len(ko_in):,} ({100*len(ko_in)/len(test):.1f}%)")
    print(f"  KO gene has >=1 network edge          {len(ko_edge):,} ({100*len(ko_edge)/len(test):.1f}%)")
    print(f"genes that ever move                    {len(moved_nt):,}")
    print(f"  of those, present in the graph        {sum(1 for g in moved_nt if g in innet):,} "
          f"({100*sum(1 for g in moved_nt if g in innet)/len(moved_nt):.1f}%)")
    R["node"] = {"test": len(test), "ko_in_readout": len(ko_in), "ko_has_edge": len(ko_edge),
                 "movers": len(moved_nt), "movers_in_graph": int(sum(1 for g in moved_nt if g in innet))}

    # ---------------------------------------------------------------- 2. EDGE COVERAGE = the ceiling
    hdr("2. EDGE COVERAGE -- what fraction of a knockout's true movers are REACHABLE?")
    cov, deg, prec, ceil = [], [], [], []
    percov = {c: [] for c in range(5)}
    CH = ["reaction", "complex", "PPI", "coexpr", "signed-reg"]
    for q in ko_edge:
        ki = gi[kos[q]]
        T = set(int(x) for x in np.where(spec[q])[0] if not tide[x])
        if not T:
            continue
        N = adj.get(ki, set())
        hit = len(T & N)
        cov.append(hit / len(T))
        deg.append(len(N))
        prec.append(hit / max(len(N), 1))
        # PERFECT ranking within the current edges: put every reachable true mover first, up to NPICK.
        ceil.append(min(hit, NPICK) / len(T))
        for c in range(5):
            percov[c].append(len(T & adjc[c].get(ki, set())) / len(T))
    cov, deg, prec, ceil = map(np.array, (cov, deg, prec, ceil))
    print(f"over {len(cov):,} held-out perturbations whose KO gene has edges:")
    print(f"  neighbours per KO gene      median {np.median(deg):.0f}, mean {deg.mean():.0f}, "
          f"p90 {np.percentile(deg,90):.0f}")
    print(f"  COVERAGE of true movers     mean {cov.mean():.4f}, median {np.median(cov):.4f}  "
          f"<- fraction of the answer the graph can even reach")
    print(f"  precision among neighbours  mean {prec.mean():.4f}")
    print(f"  perturbations with ZERO reachable movers: {100*(cov == 0).mean():.1f}%")
    print(f"\n  CEILING on network recall@{NPICK} under PERFECT ranking: {ceil.mean():.4f}")
    print(f"  frequency baseline for comparison:                 0.3502")
    print("  -> no model using this graph can exceed the ceiling. If it is below frequency, the graph is")
    print("     the binding constraint and architecture is irrelevant.")
    print("\n  coverage by channel (mean fraction of true movers reachable):")
    for c in range(5):
        v = np.array(percov[c])
        print(f"    {CH[c]:12s} {v.mean():.4f}")
    R["edge"] = {"median_degree": float(np.median(deg)), "coverage_mean": float(cov.mean()),
                 "precision_mean": float(prec.mean()), "zero_cov_frac": float((cov == 0).mean()),
                 "ceiling_recall": float(ceil.mean()),
                 "by_channel": {CH[c]: float(np.mean(percov[c])) for c in range(5)}}

    # ---------------------------------------------------------------- 3. RANKING vs REACH
    hdr("3. RANKING vs REACH -- which of the two is actually costing us?")
    # Actual network-only recall, versus the perfect-ranking ceiling above.
    act = []
    for q in ko_edge:
        T = set(int(x) for x in np.where(spec[q])[0] if not tide[x])
        if not T:
            continue
        ki = gi[kos[q]]
        sc = np.zeros(len(genes))
        for c in range(5):
            for j in adjc[c].get(ki, ()):
                sc[j] += 1.0
        sc[tide] = -1
        pick = set(int(x) for x in np.argsort(-sc)[:NPICK])
        act.append(len(pick & T) / len(T))
    act = np.array(act)
    print(f"  actual network-only recall@{NPICK}   {act.mean():.4f}")
    print(f"  ceiling with perfect ranking      {ceil.mean():.4f}")
    lost_rank = ceil.mean() - act.mean()
    lost_reach = 1.0 - ceil.mean()
    print(f"\n  lost to imperfect RANKING         {lost_rank:.4f}")
    print(f"  lost to unreachable movers        {lost_reach:.4f}   "
          f"({100*lost_reach/(lost_rank+lost_reach):.0f}% of the total shortfall)")
    R["split"] = {"actual": float(act.mean()), "ceiling": float(ceil.mean()),
                  "lost_ranking": float(lost_rank), "lost_reach": float(lost_reach)}

    # ---------------------------------------------------------------- 4. THE SPECIFICATION
    hdr("4. THE SPECIFICATION -- degree x precision needed to match frequency")
    TARGET = 0.3502
    med_T = float(np.median([nspec[q] for q in ko_edge]))
    p_obs = prec.mean()
    print(f"target                        recall@{NPICK} = {TARGET:.4f} (the frequency baseline)")
    print(f"median true movers per KO     {med_T:.0f}")
    print(f"observed precision            {p_obs:.4f}  (P(mover | neighbour), all channels pooled)")
    print(f"observed coverage             {cov.mean():.4f}  (fraction of true movers reachable at all)")

    # ONLY 50 GENES MAY BE REPORTED, and that cap changes the answer completely. A first version of this
    # section tabulated "edges per gene needed = TARGET x |T| / precision" and reported 322 edges at the
    # observed precision -- which silently assumed all 322 could be reported. They cannot. With 322 neighbours
    # at precision 0.0163 the pick of 50 contains 50/322 x 5.25 = 0.8 movers, a recall of 0.05, not 0.35.
    # ADDING EDGES AT CONSTANT PRECISION DOES NOT HELP; it dilutes. The binding requirement is precision
    # WITHIN the reported 50, and nothing else.
    need_hits = TARGET * med_T
    feasible_p = need_hits / NPICK
    print(f"\n  the report is capped at {NPICK} genes, so the binding requirement is precision INSIDE that 50:")
    print(f"    true movers needed in the pick   {need_hits:.1f} of {NPICK}")
    print(f"    required precision in the top 50 {feasible_p:.4f}")
    print(f"    observed precision               {p_obs:.4f}")
    print(f"    GAP                              {feasible_p/p_obs:.1f}x")
    print(f"\n  equivalently in coverage terms: the graph must REACH {TARGET:.2f} of each knockout's movers")
    print(f"  instead of {cov.mean():.4f} -- a {TARGET/max(cov.mean(),1e-9):.1f}x improvement in reach --")
    print("  and then rank them into the 50 slots.")
    print("\n  adding edges at the CURRENT precision does not help: the 50-gene cap means extra neighbours")
    print("  dilute the pick. Only higher precision, or genuinely new reach, moves this.")
    R["spec"] = {"target": TARGET, "median_movers": med_T, "observed_precision": float(p_obs),
                 "precision_needed_at_50_edges": float(feasible_p),
                 "fold_precision_gap": float(feasible_p / p_obs)}

    # ---------------------------------------------------------------- 5. IS IT ATTAINABLE?
    hdr("5. IS IT ATTAINABLE? -- the densest honest alternative")
    print("The graph's problem is REACH, and reach cannot be improved by reweighting what is already there.")
    print("Two sources could raise it, and they differ in whether they are legitimate here:\n")
    print("  co-response measured in K562 itself   DENSE, and CIRCULAR -- it is the readout being predicted")
    print("  co-response in the OTHER five lines   dense, and legitimate: different experiment, same biology")
    print("\nThe second is testable with files already on disk (RPE1, HepG2, Jurkat, HCT116, Melanoma).")
    print("It is the only remaining lever that changes REACH rather than ranking.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "network_requirement.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'network_requirement.json'}")


if __name__ == "__main__":
    main()

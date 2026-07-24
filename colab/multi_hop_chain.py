"""multi_hop_chain -- the user's refinement: don't just use DIRECT PPI; follow a CHAIN of PPIs (paths) from the knockout to each mover, since
the response is an end-state reached through intermediate steps.

The literal test: shortest-path hop distance from the knockout over the physical interactome (PPI + complex co-membership), then:
  A) COVERAGE: what fraction of specific movers are reachable within k hops -- for k = 1..4.
  B) THE CONTROL THAT MATTERS: what fraction of NON-movers are reachable in the same k hops. In a small-world network everything is ~3-4 hops
     apart, so coverage trivially -> 100%; the chain idea only has signal if movers are reachable in FEWER hops / are CLOSER than non-movers.
  C) PREDICTOR: rank genes by closeness (1/(1+dist)); does that predict the specific movers (recall@50 vs tide)?

Honest note on 'time flows / irreversible': physical PPI is UNDIRECTED, so a chain is symmetric and explodes; real directionality/time-ordering
lives in the signalling/regulatory layer (tested separately in mechanistic_propagate). This tests the PPI-chain idea as asked and measures whether
multi-hop physical reachability discriminates movers at all. Deterministic."""
import json, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness

OUT = Path("outputs/orphan")
MAXD = 4


def main():
    H = Harness("K562")
    D = json.load(open(OUT / "cell_complete.json")); names = [g["name"] for g in D["genes"]]; N = len(names)
    adj = collections.defaultdict(set)
    for a, b in D["ppi"]:
        if isinstance(a, int) and isinstance(b, int) and a < N and b < N:
            adj[names[a]].add(names[b]); adj[names[b]].add(names[a])
    for mem in D["complexes"].values():                       # complex co-membership = physical edges too
        mm = [names[x] for x in mem if isinstance(x, int) and x < N]
        if 2 <= len(mm) <= 200:
            for g in mm:
                adj[g].update(x for x in mm if x != g)
    print(f"physical interactome: {len(adj)} proteins, avg degree {np.mean([len(v) for v in adj.values()]):.1f}")

    def bfs(src):
        dist = {src: 0}; frontier = {src}
        for d in range(1, MAXD + 1):
            nxt = set()
            for u in frontier:
                for v in adj.get(u, ()):
                    if v not in dist:
                        dist[v] = d; nxt.add(v)
            frontier = nxt
            if not frontier:
                break
        return dist

    gi = H.gi
    cov_mov = {k: [] for k in range(1, MAXD + 1)}
    cov_non = {k: [] for k in range(1, MAXD + 1)}
    close_rec, tide_rec = [], []
    med_mov_d, med_non_d = [], []
    rng = np.random.RandomState(0)
    n_used = 0
    for ko in H.scorable:
        if ko not in adj:
            continue
        dist = bfs(ko)
        r = H.ki[ko]
        movers = [int(i) for i in np.where(H.mover[r])[0] if H.nontide[i]]
        if not movers:
            continue
        n_used += 1
        # distance (in hops) to each measured gene; unreachable within MAXD -> MAXD+1
        def dofg(gidx):
            nm = H.genes[gidx]; return dist.get(nm, MAXD + 1)
        mv_d = np.array([dofg(g) for g in movers])
        non_pool = rng.choice(H.nontide_idx, size=min(400, len(H.nontide_idx)), replace=False)
        non_d = np.array([dofg(int(g)) for g in non_pool if int(g) not in set(movers)])
        for k in range(1, MAXD + 1):
            cov_mov[k].append(float((mv_d <= k).mean()))
            cov_non[k].append(float((non_d <= k).mean()))
        med_mov_d.append(float(np.median(mv_d))); med_non_d.append(float(np.median(non_d)))
        # closeness predictor over nontide genes
        v = np.zeros(H.nG, np.float32)
        for nm, dd in dist.items():
            j = gi.get(nm)
            if j is not None and H.nontide[j]:
                v[j] = 1.0 / (1.0 + dd)
        spec = set(movers)
        close_rec.append(H._recall(v, spec, H.nontide_idx, 50))
        tide_rec.append(H._recall(H.mover_freq, spec, H.nontide_idx, 50))

    m = lambda a: float(np.mean(a)) if a else float("nan")
    print(f"\nknockouts used (in interactome, with movers): {n_used}\n")
    print(f"  {'within k hops':16s} {'% MOVERS':>9s} {'% NON-movers':>12s} {'enrichment':>11s}")
    for k in range(1, MAXD + 1):
        e = m(cov_mov[k]) / max(m(cov_non[k]), 1e-9)
        print(f"  {'<= ' + str(k) + ' hops':16s} {m(cov_mov[k])*100:8.1f}% {m(cov_non[k])*100:11.1f}% {e:10.2f}x")
    print(f"\n  median hop-distance to movers {m(med_mov_d):.2f}  vs to random non-movers {m(med_non_d):.2f}")
    print(f"  CLOSENESS predictor (rank by 1/(1+dist)) recall@50 {m(close_rec):.3f}  vs tide {m(tide_rec):.3f}")

    enr2 = m(cov_mov[2]) / max(m(cov_non[2]), 1e-9)
    closer = m(med_mov_d) < m(med_non_d) - 0.1
    beats = m(close_rec) > m(tide_rec) + 0.02
    verdict = (
        f"MULTI-HOP PPI CHAIN (multi_hop_chain.py): follow chains of physical interactions (not just direct PPI) from the knockout to each "
        f"mover. {n_used} knockouts, held-out K562. COVERAGE explodes with hops as expected in a small-world graph "
        f"({m(cov_mov[1])*100:.0f}% of movers within 1 hop -> {m(cov_mov[MAXD])*100:.0f}% within {MAXD}) -- BUT so does coverage of NON-movers "
        f"({m(cov_non[1])*100:.0f}% -> {m(cov_non[MAXD])*100:.0f}%), the small-world trap. The only thing that matters is DISCRIMINATION: are "
        f"movers CLOSER than random genes? Enrichment within 2 hops is {enr2:.2f}x; median hop-distance to movers {m(med_mov_d):.2f} vs "
        f"{m(med_non_d):.2f} to random; and ranking by chain-closeness recalls {m(close_rec):.3f} vs tide {m(tide_rec):.3f}. "
        + ("Movers ARE meaningfully closer in the interactome and chain-closeness beats the tide -- multi-hop physical chains carry real signal "
           "for the specific response." if (closer and beats) else
           "Movers are only marginally closer than random and chain-closeness does NOT beat the tide-null. So multi-hop PPI chaining does not "
           "rescue coverage: because the physical interactome is small-world and UNDIRECTED, within 2-3 hops nearly every gene is 'reachable', "
           "so a chain reaches the true movers and a huge number of non-movers alike -- reachability stops discriminating. The knockout->end-state "
           "cascade IS multi-step, but the missing ingredient is DIRECTION/TIME-ORDER (which physical PPI lacks), not more hops. Consistent with "
           "mechanistic_propagate (propagating to convergence over-diffused) and network_attribution (90% unexplained): adding hops trades a "
           "coverage gap for a specificity collapse.")
        + " Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"n_kos": n_used, "coverage_movers": {k: round(m(cov_mov[k]), 4) for k in cov_mov},
               "coverage_nonmovers": {k: round(m(cov_non[k]), 4) for k in cov_non},
               "median_dist_movers": round(m(med_mov_d), 3), "median_dist_nonmovers": round(m(med_non_d), 3),
               "closeness_recall": round(m(close_rec), 4), "tide_recall": round(m(tide_rec), 4),
               "verdict": verdict, "note": verdict}, open(OUT / "multi_hop_chain.json", "w"), indent=1)
    print("\n  -> outputs/orphan/multi_hop_chain.json")


if __name__ == "__main__":
    main()

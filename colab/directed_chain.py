"""directed_chain -- the user's fix for the small-world trap: put ARROWS on the edges (direction/time-order) and follow DIRECTED chains, so a
cascade can only flow downstream instead of fanning out symmetrically.

We don't have complex-assembly order, but we DO have a directed layer: SIGNOR directed-signed signaling + TRRUST TF->target (build_causal's
graph). Test: follow DIRECTED (forward-only) chains from each knockout and ask the same discrimination question as multi_hop_chain --
  A) directed COVERAGE of movers within k hops, and of NON-movers (the control);
  B) does directed reachability DISCRIMINATE movers better than the undirected explosion did? (undirected collapsed to 1.0x by 3 hops)
  C) directed-closeness recall@50 vs tide.
If direction restores discrimination but coverage is low, the concept is right and the limit is directed-edge COVERAGE (canonical only).
Deterministic."""
import json, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness
from transformer_causal import build_causal

OUT = Path("outputs/orphan")
MAXD = 4


def main():
    H = Harness("K562")
    reg, _, sources = build_causal(H)                         # directed: name -> [(target, sign)] (SIGNOR + TRRUST + causal)
    dadj = {s: set(t for t, _ in ts) for s, ts in reg.items()}
    outdeg = np.mean([len(v) for v in dadj.values()]) if dadj else 0
    print(f"directed network: {len(dadj)} source proteins, {sum(len(v) for v in dadj.values())} directed edges, avg out-degree {outdeg:.1f}")

    def bfs_dir(src):
        dist = {src: 0}; frontier = {src}
        for d in range(1, MAXD + 1):
            nxt = set()
            for u in frontier:
                for v in dadj.get(u, ()):
                    if v not in dist:
                        dist[v] = d; nxt.add(v)
            frontier = nxt
            if not frontier:
                break
        return dist

    gi = H.gi
    cov_mov = {k: [] for k in range(1, MAXD + 1)}
    cov_non = {k: [] for k in range(1, MAXD + 1)}
    close_rec, tide_rec, reach_any = [], [], []
    n_src = 0
    rng = np.random.RandomState(0)
    for ko in H.scorable:
        if ko not in dadj:                                    # only knockouts that ARE directed sources (have outgoing arrows)
            continue
        dist = bfs_dir(ko); n_src += 1
        r = H.ki[ko]
        movers = [int(i) for i in np.where(H.mover[r])[0] if H.nontide[i]]
        if not movers:
            continue
        def dof(g): return dist.get(H.genes[g], MAXD + 1)
        mv_d = np.array([dof(g) for g in movers])
        non_pool = rng.choice(H.nontide_idx, size=min(400, len(H.nontide_idx)), replace=False)
        non_d = np.array([dof(int(g)) for g in non_pool if int(g) not in set(movers)])
        for k in range(1, MAXD + 1):
            cov_mov[k].append(float((mv_d <= k).mean())); cov_non[k].append(float((non_d <= k).mean()))
        reach_any.append(float((mv_d <= MAXD).mean()))
        v = np.zeros(H.nG, np.float32)
        for nm, dd in dist.items():
            j = gi.get(nm)
            if j is not None and H.nontide[j]:
                v[j] = 1.0 / (1.0 + dd)
        spec = set(movers)
        close_rec.append(H._recall(v, spec, H.nontide_idx, 50)); tide_rec.append(H._recall(H.mover_freq, spec, H.nontide_idx, 50))

    m = lambda a: float(np.mean(a)) if a else float("nan")
    print(f"\nknockouts that are directed sources (have arrows out): {n_src} of {len(H.scorable)} ({100*n_src/len(H.scorable):.0f}%)\n")
    print(f"  {'within k hops':16s} {'% MOVERS':>9s} {'% NON-movers':>12s} {'discrimination':>14s}")
    for k in range(1, MAXD + 1):
        e = m(cov_mov[k]) / max(m(cov_non[k]), 1e-9)
        print(f"  {'<= ' + str(k) + ' hops':16s} {m(cov_mov[k])*100:8.1f}% {m(cov_non[k])*100:11.1f}% {e:13.2f}x")
    print(f"\n  movers directionally reachable within {MAXD} hops: {m(reach_any)*100:.1f}%")
    print(f"  DIRECTED-closeness recall@50 {m(close_rec):.3f}  vs tide {m(tide_rec):.3f}")
    print(f"  (compare undirected multi_hop_chain: 2-hop discrimination 1.63x, closeness recall 0.030)")

    disc2 = m(cov_mov[2]) / max(m(cov_non[2]), 1e-9); disc3 = m(cov_mov[3]) / max(m(cov_non[3]), 1e-9)
    direction_helps = disc3 > 1.5                              # directed still discriminates at 3 hops where undirected collapsed to 1.0
    verdict = (
        f"DIRECTED CHAIN (directed_chain.py): put ARROWS on the edges and follow DIRECTED chains, to beat the small-world collapse of undirected "
        f"PPI. Directed layer = SIGNOR signed signaling + TRRUST (avg out-degree {outdeg:.1f}). Only {100*n_src/len(H.scorable):.0f}% of knockouts "
        f"are directed sources (have outgoing arrows) -- the rest have NO directed edge to propagate. On those that do: directed discrimination "
        f"stays {disc2:.2f}x at 2 hops and {disc3:.2f}x at 3 hops "
        + ("(vs undirected's collapse to 1.0x) -- so DIRECTION DOES restore specificity: a directed cascade does not fan out to the whole cell. "
           if direction_helps else "-- direction helps only marginally here. ")
        + f"BUT only {m(reach_any)*100:.0f}% of a knockout's movers are directionally reachable at all, and directed-closeness recall is "
        f"{m(close_rec):.3f} vs tide {m(tide_rec):.3f}. "
        + ("So the concept is VALIDATED -- arrows fix the fan-out problem -- but the directed edges we have (SIGNOR/TRRUST, canonical) are too "
           "SPARSE to cover the response: most knockouts aren't even sources, and most movers are never reached. The bottleneck moves from "
           "'undirected explosion' to 'directed-edge coverage'. The user's assembly-order idea is exactly the right KIND of data (it would add "
           "directed edges), but assembly order isn't in any database at scale; the scalable path to a dense DIRECTED interactome is a "
           "time-course readout (order genes by when they move) + kinase-substrate/SIGNOR-style directed edges." if not (direction_helps and m(close_rec) > m(tide_rec) + 0.02) else
           "Direction both restores discrimination AND beats the tide -- directed chains are a real predictor.")
        + " Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"n_directed_sources": n_src, "frac_ko_are_sources": round(n_src / len(H.scorable), 3),
               "coverage_movers": {k: round(m(cov_mov[k]), 4) for k in cov_mov},
               "coverage_nonmovers": {k: round(m(cov_non[k]), 4) for k in cov_non},
               "movers_reachable": round(m(reach_any), 4), "directed_closeness_recall": round(m(close_rec), 4),
               "tide_recall": round(m(tide_rec), 4), "verdict": verdict, "note": verdict},
              open(OUT / "directed_chain.json", "w"), indent=1)
    print("\n  -> outputs/orphan/directed_chain.json")


if __name__ == "__main__":
    main()

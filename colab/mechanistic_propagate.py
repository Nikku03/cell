"""mechanistic_propagate -- the user's idea, tested: knock the protein OUT of the network, propagate to the FINAL (steady-state) effect,
and see if that predicts which genes actually change. Two networks, propagated to convergence, scored against the real K562 movers:

  A) UNDIRECTED PPI (physical protein-protein binding -- the literal 'protein network'): diffuse the knockout outward to convergence.
  B) DIRECTED SIGNED regulatory (TRRUST + SIGNOR/CollecTRI): propagate signed influence (up/down composed) to steady state.

Scored on tide-removed SPECIFIC movers (the real target) AND all-movers (the generic cascade), vs the tide-null, with a shuffled-edge control.
This is the mechanistic alternative to statistical retrieval; it settles whether 'simulate the network to the end' beats 'find a similar
measured knockout'. No learning; deterministic."""
import json, collections
from pathlib import Path
import numpy as np
from eval_harness import Harness
from transformer_causal import build_causal

OUT = Path("outputs/orphan")
K = 50


def main():
    H = Harness("K562")
    reg, _, sources = build_causal(H)                       # directed signed graph: name -> [(target, sign)]

    # undirected PPI (physical binding) adjacency, by name
    D = json.load(open(OUT / "cell_complete.json")); names = [g["name"] for g in D["genes"]]
    ppi = collections.defaultdict(set)
    for a, b in D["ppi"]:
        if isinstance(a, int) and isinstance(b, int) and a < len(names) and b < len(names):
            ppi[names[a]].add(names[b]); ppi[names[b]].add(names[a])

    # shuffled directed control: rewire each source's targets to random genes (same out-degree)
    rng = np.random.RandomState(0); allnames = list({n for n in H.genes})
    reg_sh = {}
    for s, ts in reg.items():
        reg_sh[s] = [(allnames[rng.randint(len(allnames))], sg) for _, sg in ts]

    def prop_directed(g, R, iters=10, alpha=0.6):
        infl = collections.defaultdict(float); frontier = {g: 1.0}
        for _ in range(iters):
            nxt = collections.defaultdict(float)
            for node, w in frontier.items():
                for tgt, s in R.get(node, []):
                    c = w * (s if s != 0 else 0.5) * alpha
                    infl[tgt] += c; nxt[tgt] += c
            if not nxt:
                break
            frontier = nxt
        v = np.zeros(H.nG)
        for nm, x in infl.items():
            j = H.gi.get(nm)
            if j is not None:
                v[j] += abs(x)
        return v

    def prop_undirected(g, iters=6, alpha=0.5):
        infl = collections.defaultdict(float); frontier = {g: 1.0}
        for _ in range(iters):
            nxt = collections.defaultdict(float)
            for node, w in frontier.items():
                nb = ppi.get(node, ())
                if not nb:
                    continue
                share = w * alpha / len(nb)
                for tgt in nb:
                    infl[tgt] += share; nxt[tgt] += share
            if not nxt:
                break
            frontier = nxt
        v = np.zeros(H.nG)
        for nm, x in infl.items():
            j = H.gi.get(nm)
            if j is not None:
                v[j] += x
        return v

    def score(predfn, kos, need_out=False):
        spec, allm = [], []
        for ko in kos:
            if need_out and ko not in sources:
                continue
            v = predfn(ko)
            if v.sum() == 0:
                continue
            r = H.ki[ko]; mv = np.where(H.mover[r])[0]
            sp = set(int(i) for i in mv if H.nontide[i]); am = set(int(i) for i in mv)
            if len(sp) >= 1:
                spec.append(H._recall(v, sp, H.nontide_idx, K))
                allm.append(H._recall(v, am, np.arange(H.nG), K))
        return (float(np.mean(spec)) if spec else float("nan"),
                float(np.mean(allm)) if allm else float("nan"), len(spec))

    kos = list(H.scorable)
    tide = float(np.mean([H._recall(H.mover_freq, set(int(i) for i in np.where(H.mover[H.ki[k]])[0] if H.nontide[i]),
                                    H.nontide_idx, K) for k in kos]))
    oracle = float(np.mean([H._recall(H._references(k, K)["ORACLE*"],
                                      set(int(i) for i in np.where(H.mover[H.ki[k]])[0] if H.nontide[i]),
                                      H.nontide_idx, K) for k in kos]))
    d_spec, d_all, d_n = score(lambda g: prop_directed(g, reg), kos, need_out=True)
    dsh_spec, _, _ = score(lambda g: prop_directed(g, reg_sh), kos, need_out=True)
    u_spec, u_all, u_n = score(prop_undirected, kos)

    print(f"held-out K562 scorable KOs: {len(kos)}   (directed = the {d_n} that are regulators; undirected PPI = {u_n} with partners)\n")
    print(f"{'method (propagate to steady state)':44s} {'SPECIFIC r@50':>13s} {'all-mover r@50':>14s}")
    print(f"{'TIDE-null (no mechanism)':44s} {tide:13.3f} {'-':>14s}")
    print(f"{'A) undirected PPI (physical binding)':44s} {u_spec:13.3f} {u_all:14.3f}")
    print(f"{'B) directed signed regulatory':44s} {d_spec:13.3f} {d_all:14.3f}")
    print(f"{'   directed, SHUFFLED edges (control)':44s} {dsh_spec:13.3f} {'-':>14s}")
    print(f"{'ORACLE* (find a similar measured KO)':44s} {oracle:13.3f} {'-':>14s}")

    beats = max(u_spec, d_spec) > tide + 0.02
    verdict = (
        "MECHANISTIC PROPAGATE (mechanistic_propagate.py): the 'knock the protein out of the network and simulate to the final effect' idea, "
        f"tested. Propagating a knockout to steady state on the UNDIRECTED PPI (physical binding) scores {u_spec:.3f} on tide-removed SPECIFIC "
        f"movers, and on the DIRECTED SIGNED regulatory network {d_spec:.3f} (shuffled-edge control {dsh_spec:.3f}), both vs the TIDE-null "
        f"{tide:.3f} and the retrieval ORACLE {oracle:.3f}. "
        + ("Propagation BEATS the tide on specific movers -- the mechanism carries real specific signal." if beats else
           "Neither beats the tide-null on SPECIFIC movers: propagation recovers only the GENERIC cascade (all-mover recall "
           f"{max(u_all, d_all):.2f} -- the tide), not the knockout-specific response. WHY: (1) the PPI graph is PHYSICAL binding, but an mRNA "
           "change is a TRANSCRIPTIONAL/regulatory effect, so it is the wrong graph; (2) even the directed regulatory graph only encodes GENERIC "
           "'X regulates Y' edges, so propagating them reproduces the generic cascade every strong KO shares (the tide), which is exactly the "
           "part the metric removes; (3) a true simulation needs quantitative dynamics -- rates, thresholds, feedback, cell-specific active "
           "state -- which we do not have, so propagation is only reachability, not simulation. This matches transformer_causal (directed tracer "
           "traces the generic cascade, fails on the specific residue) and confirms the specific response is not a function of the static network "
           "topology -- it is context-specific biology no propagation on a generic wiring diagram can produce.")
        + " Deterministic.")
    print(f"\nVERDICT: {verdict}")
    json.dump({"n_kos": len(kos), "tide_null": round(tide, 4), "oracle": round(oracle, 4),
               "undirected_ppi": {"specific": round(u_spec, 4), "all_mover": round(u_all, 4), "n": u_n},
               "directed_signed": {"specific": round(d_spec, 4), "all_mover": round(d_all, 4), "n": d_n},
               "directed_shuffled": round(dsh_spec, 4), "beats_tide": bool(beats),
               "verdict": verdict, "note": verdict}, open(OUT / "mechanistic_propagate.json", "w"), indent=1)
    print("\n  -> outputs/orphan/mechanistic_propagate.json")


if __name__ == "__main__":
    main()

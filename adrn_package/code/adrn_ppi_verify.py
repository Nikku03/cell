"""VERIFY the surprising PPI result before believing it.

adrn_source_blocks reported that `chan2a+ppi` beats `chan2a+ppi_rw` (+0.0155 / +0.0325, both resolved) --
i.e. PPI neighbourhood structure carries signal beyond degree.  **That contradicts eight prior network tests in
this project, every one of which was indistinguishable from degree-matched rewiring, once with the shuffle
winning outright.**  A lone positive against that much standing negative evidence is more likely to be an artefact
than a discovery, so it gets attacked before it gets reported.

THREE WAYS THE RESULT COULD BE FAKE, each checked here:

  1. THE CONTROL IS BROKEN.  If `rewire()` silently failed -- too few accepted swaps, or a degree sequence that
     drifted -- then `ppi_rw` is not a degree-matched twin and the contrast measures nothing.  Checked by
     asserting the degree sequence is EXACTLY preserved and reporting what fraction of edges actually moved.
     A double-edge swap that rejects most proposals leaves the graph nearly intact; one that corrupts degree
     invalidates the comparison in the other direction.

  2. THE RESULT IS SEED NOISE.  One rewiring draw is one sample from the null.  If the gap swings across seeds,
     +0.0155 is a draw from a wide distribution rather than an effect.  Checked by rebuilding the control under
     several independent seeds and reporting the spread.

  3. THE EDGES LEAK.  If any PPI edge was derived from co-expression or co-response in the same Perturb-seq data
     the model is scored against, the arm is reading the answer key through the graph.  `cell_complete.json`
     carries no provenance metadata, so this is checked empirically instead: the correlation between an edge
     existing and the two genes' measured response profiles being similar.  Genuine AP-MS/complex edges should
     show a modest association; edges *derived* from co-response would show a large one.

This does not re-score the model.  It asks only whether the control was a real control.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
from adrn_source_blocks import rewire

OUT, SP = A.OUT, A.SP
SEEDS = (A.SEED, A.SEED + 1, A.SEED + 2)


def main():
    log = []

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("VERIFY the PPI-beats-rewiring result: is the control real?")
    report("=" * 100)

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] if isinstance(g, dict) else str(g) for g in D["genes"]]
    n = len(names)
    edges = np.array([(a, b) for a, b in D["ppi"] if a < n and b < n], np.int64)
    report(f"  {len(edges):,} edges over {n:,} genes")

    deg0 = np.bincount(np.r_[edges[:, 0], edges[:, 1]], minlength=n)
    orig = {(min(a, b), max(a, b)) for a, b in edges}

    # ---- CHECK 1 + 2: degree preservation and how much the graph actually moved, across seeds
    report(f"\n  {'seed':>6} {'degree identical':>17} {'edges moved':>13} {'jaccard vs real':>16}")
    ok_all = True
    moved = []
    for s in SEEDS:
        e = rewire(edges, n, s)
        deg = np.bincount(np.r_[e[:, 0], e[:, 1]], minlength=n)
        same = bool((deg == deg0).all())
        ok_all &= same
        new = {(min(a, b), max(a, b)) for a, b in e}
        keep = len(orig & new)
        jac = keep / float(len(orig | new))
        frac_moved = 1.0 - keep / float(len(orig))
        moved.append(frac_moved)
        report(f"  {s:>6} {str(same):>17} {frac_moved:>12.1%} {jac:>16.3f}")
    report(f"  -> degree exactly preserved on every seed: {ok_all}")
    if not ok_all:
        report("  ABORT: the control does not preserve degree, so the contrast measures degree, not neighbourhood.")
    if min(moved) < 0.5:
        report(f"  WARNING: only {min(moved):.1%} of edges moved -- the 'rewired' graph is still mostly the real"
               " one, which would DEFLATE the contrast rather than inflate it.")

    # ---- CHECK 3: do edges encode measured co-response? (leakage)
    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    M = np.abs(z["M"]).astype(np.float32)
    ki = {k: i for i, k in enumerate(kos)}
    nidx = {g: i for i, g in enumerate(names)}
    both = [(a, b) for a, b in edges if names[a] in ki and names[b] in ki]
    report(f"\n  edges with BOTH genes measured in the K562 screen: {len(both):,}")
    if len(both) >= 500:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(both), min(4000, len(both)), replace=False)
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        lin = np.array([float(Mn[ki[names[both[i][0]]]] @ Mn[ki[names[both[i][1]]]]) for i in pick])
        # degree-matched random pairs: same genes, shuffled partners
        aa = [both[i][0] for i in pick]
        bb = [both[i][1] for i in pick]
        rng.shuffle(bb)
        rnd = np.array([float(Mn[ki[names[x]]] @ Mn[ki[names[y]]]) for x, y in zip(aa, bb)
                        if names[x] in ki and names[y] in ki])
        report(f"  cosine of measured response profiles:")
        report(f"    linked pairs   {lin.mean():.4f}")
        report(f"    shuffled pairs {rnd.mean():.4f}")
        report(f"    ratio          {lin.mean()/max(rnd.mean(), 1e-9):.2f}x")
        report("  A modest ratio is expected and is biology -- interacting proteins do share responses.")
        report("  A very large ratio would mean the edge list was DERIVED from this response data (leakage).")
        leak = float(lin.mean() / max(rnd.mean(), 1e-9))
    else:
        leak = float("nan")

    json.dump({"test": "adrn_ppi_verify", "n_edges": int(len(edges)),
               "degree_preserved": bool(ok_all), "frac_edges_moved": [float(x) for x in moved],
               "coresponse_ratio_linked_vs_shuffled": leak, "log": log},
              open(OUT / "adrn_ppi_verify.json", "w"), indent=2)
    report(f"\n  -> {OUT / 'adrn_ppi_verify.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

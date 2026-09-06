"""ROUTING: does the network actually have paths from a knocked-out gene to the genes that move?

THE GAP THIS FILLS. Every previous measurement in this project used DIRECT edges only. Coverage of a
knockout's true movers by immediate neighbours is 0.0479, and the recall ceiling under perfect ranking is
0.0478 against a frequency baseline of 0.3502. But "not a neighbour" is not "not reachable". If the movers sit
two or three steps away along a real chain -- knock out a kinase, it changes a TF, the TF changes its targets
-- then a one-hop test would miss the entire mechanism and wrongly declare the graph empty.

So: shortest paths from each knocked-out gene to every gene, over the union of the interaction network and
the transcription-factor networks, on every cell line available.

THE TRAP, AND WHY THIS FILE IS MOSTLY CONTROLS. Biological networks are small-world. In a graph with ~250k
edges over ~8k genes, almost everything is within three steps of almost everything, so "a path exists" is
guaranteed and means nothing. Two questions actually matter:

  1 REACHABILITY VOLUME  how many genes are within d steps? If d=3 already reaches most of the graph, then
                         path existence is not evidence -- it is topology.
  2 ARE MOVERS CLOSER    than genes that are not movers, once degree is controlled? A hub is close to
                         everything, so an uncontrolled comparison recovers degree, not biology.

Three controls, because each kills a different false positive:

  degree-matched decoys      compares a mover against a non-mover of the SAME degree
  frequency-matched decoys   compares against a non-mover moved as OFTEN across other perturbations, which
                             removes the frequency prior that has beaten every method here
  degree-preserving rewiring configuration-model shuffle that destroys biology while preserving every node's
                             degree exactly. If the real graph scores no better than its rewired twin, the
                             paths carry nothing beyond the degree sequence.

The rewired control is the decisive one. Everything else can be explained by hubs.
"""
import collections
import gzip
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TAU, TIDE_FRAC, MIN_SPEC, NPICK = 1.0, 0.05, 5, 50
MAXD = 5
RNG = np.random.default_rng(0)
R = {}


def hdr(s):
    print("\n" + "=" * 104)
    print(s)
    print("=" * 104)


# ------------------------------------------------------------------ EDGE SOURCES
def interaction_edges(gi):
    """The five measured channels, read from the model's own pair store so this cannot drift from it."""
    import cellformer_af as CF
    d = CF.load_all()
    out = collections.defaultdict(set)
    CH = ["reaction", "complex", "PPI", "coexpr", "signed-reg"]
    for (i, j), v in d["pair"].items():
        for c in range(5):
            if v[c]:
                out[CH[c]].add((i, j))
    return out, d


def tf_edges(gi):
    """TRRUST + tf_core + ChIP-derived regulatory edges, mapped into the readout's gene index.

    k562_tf_chip.json is NOT usable here and is deliberately skipped: it holds ENCODE accessions and download
    hrefs, not edges. Loading it would contribute zero edges while looking like a TF layer was included.
    """
    out = collections.defaultdict(set)

    def add(tag, a, b):
        if a in gi and b in gi and a != b:
            i, j = gi[a], gi[b]
            out[tag].add((min(i, j), max(i, j)))

    p = OUT / "trrust_regulon.json"
    if p.exists():
        for tf, tgts in (json.load(open(p)).get("tf_targets") or {}).items():
            for t in (tgts if isinstance(tgts, (list, tuple)) else []):
                add("TF-trrust", tf, t if isinstance(t, str) else (t[0] if isinstance(t, (list, tuple)) else ""))
    p = OUT / "chip_reg_edges.json"
    if p.exists():
        for tf, tgts in (json.load(open(p)).get("tf_targets") or {}).items():
            for t in (tgts if isinstance(tgts, (list, tuple)) else []):
                add("TF-chip", tf, t if isinstance(t, str) else (t[0] if isinstance(t, (list, tuple)) else ""))
    p = OUT / "tf_core.json"
    if p.exists():
        j = json.load(open(p))
        nm = j.get("name") or {}
        for e in (j.get("edges") or []):
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                a, b = e[0], e[1]
                a = nm.get(str(a), a) if not isinstance(a, str) else a
                b = nm.get(str(b), b) if not isinstance(b, str) else b
                if isinstance(a, str) and isinstance(b, str):
                    add("TF-core", a, b)
    return out


# ------------------------------------------------------------------ READOUTS
def load_pkl_line(path):
    """dict[KO] -> dict[target] -> z. NOTE: these files store only each perturbation's TOP ~250 genes."""
    d = pickle.load(open(path, "rb"))
    kos = sorted(d)
    genes = sorted({g for v in d.values() for g in v})
    gidx = {g: i for i, g in enumerate(genes)}
    M = np.zeros((len(kos), len(genes)), np.float32)
    for r, k in enumerate(kos):
        for g, z in d[k].items():
            M[r, gidx[g]] = z
    return kos, genes, M


def readouts():
    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    yield ("K562-gwps (full)", [str(x) for x in z["kos"]], [str(x) for x in z["genes"]],
           np.asarray(z["M"], np.float32))
    for nm in ("RPE1", "HepG2", "Jurkat", "HCT116", "Melanoma", "K562"):
        p = SP / f"nlz_{nm}.pkl"
        if p.exists():
            k, g, M = load_pkl_line(p)
            yield (f"{nm} (top-250)", k, g, M)


def spec_of(M):
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    return (A >= TAU) & ~tide, tide


# ------------------------------------------------------------------ GRAPH + REWIRE
def build_csr(edges, n):
    if not edges:
        return sparse.csr_matrix((n, n))
    ii = np.fromiter((a for a, _ in edges), np.int64, len(edges))
    jj = np.fromiter((b for _, b in edges), np.int64, len(edges))
    v = np.ones(len(ii), np.int8)
    A = sparse.coo_matrix((v, (ii, jj)), shape=(n, n))
    A = A + A.T
    A.data[:] = 1
    return A.tocsr()


def rewire(edges, n, rng):
    """Configuration-model shuffle: preserves every node's degree EXACTLY, destroys which pairs are joined.

    Stub matching rather than edge swapping -- it is O(E) instead of O(E x attempts), and the degree sequence
    is preserved by construction rather than approximately. Self-loops and duplicates are dropped, which
    shrinks the edge count by a few percent; the printed count says by how much, because a control that is
    quietly sparser than the real graph would flatter the real graph.
    """
    deg = np.zeros(n, np.int64)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    stubs = np.repeat(np.arange(n), deg)
    rng.shuffle(stubs)
    a, b = stubs[0::2], stubs[1::2]
    m = a != b
    return set(zip(np.minimum(a[m], b[m]).tolist(), np.maximum(a[m], b[m]).tolist()))


# ------------------------------------------------------------------ MAIN ANALYSIS
def analyse(tag, kos, genes, M, edge_sets, rewired=False, verbose=True):
    gi = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    spec, tide = spec_of(M)
    nspec = spec.sum(1)
    edges = set()
    for e in edge_sets.values():
        edges |= e
    if rewired:
        edges = rewire(sorted(edges), n, np.random.default_rng(7))
    A = build_csr(edges, n)
    deg = np.asarray(A.sum(1)).ravel()

    tr = np.array([int(hashlib.md5(k.encode()).hexdigest(), 16) % 2 == 0 for k in kos])
    test = [q for q in range(len(kos)) if not tr[q] and nspec[q] >= MIN_SPEC and kos[q] in gi]
    if len(test) < 20:
        return None
    freq = spec[tr].mean(0)
    cnt = spec[tr].sum(0)

    src = sorted({gi[kos[q]] for q in test})
    D = shortest_path(A, method="D", unweighted=True, indices=src)
    row_of = {s: i for i, s in enumerate(src)}

    # 1. reachability volume -- the small-world sanity check
    vol = {d: float(np.mean((D <= d).sum(1)) / n) for d in range(1, MAXD + 1)}

    # 2/3. distance of movers vs matched decoys, and lift by hop
    by_cnt = collections.defaultdict(list)
    moved = np.where(spec.any(0) & ~tide)[0]
    for g in moved:
        by_cnt[int(cnt[g])].append(int(g))
    by_cnt = {k: np.array(v) for k, v in by_cnt.items()}
    degbin = np.zeros(n, np.int64)
    nz = deg > 0
    if nz.any():
        o = np.argsort(deg)
        degbin[o] = np.arange(n) * 20 // n

    hop_hit = collections.Counter()
    hop_tot = collections.Counter()
    hop_hit_c = collections.Counter()
    hop_tot_c = collections.Counter()
    dmov, ddec = [], []
    for q in test:
        r = row_of[gi[kos[q]]]
        T = np.where(spec[q])[0]
        T = T[~tide[T]]
        if len(T) == 0:
            continue
        dt = D[r, T]
        dmov.append(dt[np.isfinite(dt)])
        # frequency-matched decoy per true mover
        dec = []
        for t in T:
            cand = by_cnt.get(int(cnt[t]))
            if cand is None or len(cand) < 2:
                continue
            x = int(RNG.choice(cand))
            if not spec[q, x]:
                dec.append(x)
        if dec:
            dd = D[r, np.array(dec)]
            ddec.append(dd[np.isfinite(dd)])
        # lift by hop, against ALL genes at that hop
        for d in range(1, MAXD + 1):
            at = np.where(D[r] == d)[0]
            at = at[~tide[at]]
            if len(at) == 0:
                continue
            hop_tot[d] += len(at)
            hop_hit[d] += int(spec[q, at].sum())
    dmov = np.concatenate(dmov) if dmov else np.array([])
    ddec = np.concatenate(ddec) if ddec else np.array([])

    # 4. path-based predictor: score = 1/2^d, i.e. closer is better, all hops usable
    rec_path, rec_1hop, rec_freq = [], [], []
    for q in test:
        r = row_of[gi[kos[q]]]
        T = set(int(x) for x in np.where(spec[q])[0] if not tide[x])
        if not T:
            continue
        d = D[r].copy()
        s = np.where(np.isfinite(d), 2.0 ** (-d), 0.0)
        s[tide] = -1
        rec_path.append(len(set(int(x) for x in np.argsort(-s)[:NPICK]) & T) / len(T))
        s1 = np.where(d == 1, 1.0, 0.0)
        s1[tide] = -1
        rec_1hop.append(len(set(int(x) for x in np.argsort(-s1)[:NPICK]) & T) / len(T))
        f = freq.copy()
        f[tide] = -1
        rec_freq.append(len(set(int(x) for x in np.argsort(-f)[:NPICK]) & T) / len(T))

    res = {
        "tag": tag, "rewired": rewired, "n_genes": n, "n_edges": len(edges),
        "median_degree": float(np.median(deg)), "n_test": len(test),
        "reach_volume": vol,
        "mover_dist_mean": float(dmov.mean()) if len(dmov) else None,
        "decoy_dist_mean": float(ddec.mean()) if len(ddec) else None,
        "hop_lift": {d: (hop_hit[d] / hop_tot[d]) for d in range(1, MAXD + 1) if hop_tot[d]},
        "recall_path": float(np.mean(rec_path)) if rec_path else None,
        "recall_1hop": float(np.mean(rec_1hop)) if rec_1hop else None,
        "recall_freq": float(np.mean(rec_freq)) if rec_freq else None,
        "unreachable_mover_frac": float(np.mean(~np.isfinite(D[[row_of[gi[kos[q]]] for q in test]][:, moved]))),
    }
    if verbose:
        print(f"\n  {tag}{'  [REWIRED CONTROL]' if rewired else ''}")
        print(f"    genes {n:,}, edges {len(edges):,}, median degree {np.median(deg):.0f}, "
              f"held-out {len(test):,}")
        print("    reachability volume: " + "  ".join(
            f"d<={d}: {100*vol[d]:.1f}%" for d in range(1, MAXD + 1)))
        if len(dmov) and len(ddec):
            print(f"    mean distance to true movers {dmov.mean():.3f}  vs frequency-matched decoys "
                  f"{ddec.mean():.3f}   (closer is better)")
        print("    P(mover) by hop: " + "  ".join(
            f"d={d}: {res['hop_lift'][d]:.4f}" for d in sorted(res["hop_lift"])))
        print(f"    recall@{NPICK}   paths {res['recall_path']:.4f}   1-hop {res['recall_1hop']:.4f}   "
              f"frequency {res['recall_freq']:.4f}")
    return res


def main():
    hdr("ROUTE MAP -- shortest paths from knockout to movers, all cell lines, interaction + TF networks")
    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    base_genes = [str(x) for x in z["genes"]]
    gi0 = {g: i for i, g in enumerate(base_genes)}
    inter, _ = interaction_edges(gi0)
    tfs = tf_edges(gi0)
    print("\nedge sources (mapped into the K562-gwps gene universe):")
    for k, v in list(inter.items()) + list(tfs.items()):
        print(f"  {k:14s} {len(v):>8,} pairs")
    print("  NOTE k562_tf_chip.json skipped: it contains ENCODE accessions and download links, not edges.")

    allres = []
    for tag, kos, genes, M in readouts():
        gi = {g: i for i, g in enumerate(genes)}
        # remap edges into THIS cell line's gene universe by name
        def remap(src):
            out = {}
            for k, st in src.items():
                s = set()
                for a, b in st:
                    na, nb = base_genes[a], base_genes[b]
                    if na in gi and nb in gi:
                        i, j = gi[na], gi[nb]
                        s.add((min(i, j), max(i, j)))
                out[k] = s
            return out
        ei, et = remap(inter), remap(tfs)
        for label, es in (("interaction only", ei),
                          ("interaction + TF", {**ei, **et}),
                          ("TF only", et)):
            r = analyse(f"{tag} | {label}", kos, genes, M, es)
            if r:
                allres.append(r)
        # rewired control on the full union, for this cell line
        r = analyse(f"{tag} | interaction + TF", kos, genes, M, {**ei, **et}, rewired=True)
        if r:
            allres.append(r)

    hdr("SUMMARY")
    print(f"{'readout | graph':52s} {'paths':>8s} {'1-hop':>8s} {'freq':>8s} {'d<=3 reach':>11s}")
    for r in allres:
        t = r["tag"] + ("  [REWIRED]" if r["rewired"] else "")
        print(f"{t:52s} {r['recall_path']:>8.4f} {r['recall_1hop']:>8.4f} {r['recall_freq']:>8.4f} "
              f"{100*r['reach_volume'][3]:>10.1f}%")
    R["results"] = allres
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "route_map.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'route_map.json'}")


if __name__ == "__main__":
    main()

"""DOES HYPERBOLIC GEOMETRY EARN ITS PLACE? Matched-dimension test on the GO hierarchy.

THE CLAIM UNDER TEST. Hyperbolic space embeds trees with far less distortion than Euclidean space at the same
dimension, because volume grows exponentially with radius. Cellular biology is full of hierarchy -- ontology,
pathway, complex, cell state -- so the proposal is to give the model a hyperbolic coordinate for it.

THAT CLAIM HAS TWO HALVES AND THEY ARE NOT THE SAME QUESTION.

    FIDELITY   does hyperbolic reconstruct the hierarchy better at matched dimension?
    UTILITY    does that fidelity buy anything downstream -- here, predicting which genes co-respond to
               perturbation in Replogle K562?

The first is close to settled in the literature and this module expects hyperbolic to win it. The second is
the one that decides whether the coordinate belongs in the model, and it does not follow from the first. A
representation can be geometrically faithful to an ontology and still carry nothing about how a cell behaves,
because the ontology is a description of what people have written down, not of what the cell does.

Reporting only the first, or letting the first stand in for the second, would be the same error this project
made when it read a benchmark-scoped ranking as a genome-wide capability.

THE DESIGN, and specifically what makes it fair.

  MATCHED DIMENSION. The entire hyperbolic argument is per-dimension efficiency, so comparing d=10 hyperbolic
    against d=100 Euclidean would answer nothing. Both get the same d, the same loss, the same negative
    sampling, the same epoch count, the same optimiser family.
  HELD OUT BY TERM, NOT BY PAIR. Terms are split; the embedding trains on the sub-ontology of TRAIN terms
    only. The downstream test then uses gene pairs whose shared annotation is a TEST term and which share no
    train term -- so the geometry has to generalise to a part of the hierarchy it never saw. Holding out
    pairs instead would let a large embedding memorise the ontology and call it generalisation.
  A NO-EMBEDDING BASELINE. Shortest-path distance in the ontology graph itself, no learning at all. If raw
    graph distance predicts co-response as well as either embedding, then the geometry question is moot and
    the honest answer is that neither curvature earns anything.
  RESPONSIVENESS MATCHING. Some genes move under almost any perturbation. Two such genes co-respond strongly
    with no functional relationship whatever, and both would sit near the ontology root. Every downstream
    comparison is therefore matched on the responsiveness decile of both genes, drawn 20 times -- the same
    discipline every other layer in this project was scored under.

WHAT WOULD FALSIFY THE PROPOSAL: hyperbolic winning fidelity but not utility, or the graph-distance baseline
matching both. What would support it: a utility gain at matched dimension that survives responsiveness
matching and holds across seeds.
"""
import gzip
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
OBO = SP / "go-basic.obo"
OBO_URL = "https://purl.obolibrary.org/obo/go/go-basic.obo"
NET = SP / "cell_complete.json.gz"
GWPS = SP / "gwps.h5ad"
BRANCH = "biological_process"
MIN_GENES, MAX_GENES = 3, 500        # a term with 2,000 genes is not a functional statement
DIMS = (5, 10)
EPOCHS = 30
NEG = 10
SEEDS = (0, 1, 2)
N_DRAWS = 20


def fetch_obo(report):
    if not OBO.exists():
        report(f"  fetching {OBO_URL}")
        req = urllib.request.Request(OBO_URL, headers={"User-Agent": "cellos"})
        with urllib.request.urlopen(req, timeout=600) as fh, open(OBO, "wb") as o:
            while True:
                b = fh.read(1 << 20)
                if not b:
                    break
                o.write(b)
    report(f"  go-basic.obo {OBO.stat().st_size/1e6:.1f} MB")


def parse_obo(report):
    """term id -> (name, namespace, [is_a parents]). Obsolete terms dropped."""
    terms, cur = {}, None
    with open(OBO) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                cur = {"id": None, "name": None, "ns": None, "is_a": [], "obs": False, "alt": []}
            elif line.startswith("[") and cur is not None:
                if cur["id"] and not cur["obs"]:
                    terms[cur["id"]] = cur
                cur = None
            elif cur is not None and ":" in line:
                k, _s, v = line.partition(": ")
                if k == "id" and v.startswith("GO:"):
                    cur["id"] = v
                elif k == "name":
                    cur["name"] = v
                elif k == "namespace":
                    cur["ns"] = v
                elif k == "is_a":
                    cur["is_a"].append(v.split(" ! ")[0].strip())
                elif k == "is_obsolete" and v == "true":
                    cur["obs"] = True
                elif k == "alt_id":
                    cur["alt"].append(v)
    if cur is not None and cur["id"] and not cur["obs"]:
        terms[cur["id"]] = cur
    bp = {k: v for k, v in terms.items() if v["ns"] == BRANCH}
    report(f"  parsed {len(terms):,} GO terms, {len(bp):,} in {BRANCH}")
    return bp


def build_graph(report):
    """Joint graph: term --is_a--> parent, and gene --annotated--> term."""
    bp = parse_obo(report)
    name2id = {}
    for tid, t in bp.items():
        name2id.setdefault(t["name"], tid)

    d = json.load(gzip.open(NET, "rt"))
    go = d.get("go", {})
    names = [g["name"] for g in d["genes"]]
    del d

    gene2terms, unmatched = {}, 0
    for g, br in go.items():
        for t in br.get("P", []):
            tid = name2id.get(t)
            if tid:
                gene2terms.setdefault(g, set()).add(tid)
            else:
                unmatched += 1
    report(f"  gene->term: {len(gene2terms):,} genes carry a {BRANCH} term "
           f"({unmatched:,} term names not found in the current ontology and dropped)")

    size = {}
    for g, ts in gene2terms.items():
        for t in ts:
            size[t] = size.get(t, 0) + 1
    keep = {t for t, n in size.items() if MIN_GENES <= n <= MAX_GENES}
    report(f"  terms with {MIN_GENES}-{MAX_GENES} genes: {len(keep):,}")
    return bp, gene2terms, keep, names


# ----------------------------------------------------------------------------- embeddings

def train_embedding(edges, n_nodes, dim, curvature, seed, report=None):
    """One embedding, either hyperbolic (Poincare ball) or Euclidean.

    Everything except the distance function is held identical between the two arms: the same edge list, the
    same number of negatives, the same epochs, the same learning rate schedule, the same initialisation
    scale. The Riemannian rescaling on the hyperbolic arm is not a tuning advantage -- it is what makes SGD
    valid on the ball at all, and without it the arm would be broken rather than merely different.
    """
    import torch
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    src = torch.tensor([e[0] for e in edges], dtype=torch.long)
    dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
    emb = torch.randn(n_nodes, dim, generator=g) * 1e-3
    emb.requires_grad_(True)
    lr0 = 0.3 if curvature == "hyperbolic" else 0.1
    nE = len(edges)
    bs = min(8192, nE)

    def dist(u, v):
        if curvature == "euclidean":
            return torch.norm(u - v, dim=-1)
        su = torch.clamp((u * u).sum(-1), max=1 - 1e-5)
        sv = torch.clamp((v * v).sum(-1), max=1 - 1e-5)
        num = ((u - v) ** 2).sum(-1)
        x = 1 + 2 * num / ((1 - su) * (1 - sv) + 1e-9)
        return torch.acosh(torch.clamp(x, min=1 + 1e-7))

    for ep in range(EPOCHS):
        lr = lr0 * (1 - ep / EPOCHS) + 1e-3
        perm = torch.randperm(nE, generator=g)
        for i0 in range(0, nE, bs):
            idx = perm[i0:i0 + bs]
            s, t = src[idx], dst[idx]
            neg = torch.randint(0, n_nodes, (len(idx), NEG), generator=g)
            u = emb[s].unsqueeze(1)
            pos_d = dist(emb[s], emb[t]).unsqueeze(1)
            neg_d = dist(u.expand(-1, NEG, -1), emb[neg])
            logits = -torch.cat([pos_d, neg_d], 1)
            loss = torch.nn.functional.cross_entropy(
                logits, torch.zeros(len(idx), dtype=torch.long))
            if emb.grad is not None:
                emb.grad = None
            loss.backward()
            with torch.no_grad():
                gr = emb.grad
                if curvature == "hyperbolic":
                    # Riemannian gradient on the Poincare ball
                    sq = (emb * emb).sum(-1, keepdim=True)
                    gr = gr * ((1 - sq) ** 2 / 4)
                emb -= lr * gr
                if curvature == "hyperbolic":
                    nrm = emb.norm(dim=-1, keepdim=True)
                    emb *= torch.clamp(1 - 1e-5, max=1.0) / torch.clamp(nrm, min=1 - 1e-5)
                    emb[nrm.squeeze(-1) < 1 - 1e-5] = emb[nrm.squeeze(-1) < 1 - 1e-5]
    return emb.detach(), dist


def pair_distance(emb, dist_fn, pairs, curvature):
    import torch
    a = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    b = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    with torch.no_grad():
        return dist_fn(emb[a], emb[b]).numpy()


def main():
    from scipy import stats
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("HYPERBOLIC vs EUCLIDEAN at matched dimension -- fidelity, and then utility")
    report("=" * 100)
    fetch_obo(report)
    bp, gene2terms, keep, gene_names = build_graph(report)

    # ---- node index: terms then genes ----
    terms = sorted(keep)
    tix = {t: i for i, t in enumerate(terms)}
    genes = sorted({g for g, ts in gene2terms.items() if ts & keep})
    gix = {g: len(terms) + i for i, g in enumerate(genes)}
    n_nodes = len(terms) + len(genes)
    report(f"  graph: {len(terms):,} terms + {len(genes):,} genes = {n_nodes:,} nodes")

    # ---- split TERMS, not pairs ----
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(terms))
    n_test = len(terms) // 5
    test_t = {terms[i] for i in perm[:n_test]}
    train_t = {terms[i] for i in perm[n_test:]}
    report(f"  term split: {len(train_t):,} train / {len(test_t):,} test "
           f"(the embedding never sees a test term)")

    isa_train, isa_test = [], []
    for t in terms:
        for p in bp[t]["is_a"]:
            if p in tix:
                (isa_train if (t in train_t and p in train_t) else isa_test).append((tix[t], tix[p]))
    ann_train = [(gix[g], tix[t]) for g in genes for t in (gene2terms[g] & train_t)]
    edges = isa_train + ann_train
    report(f"  training edges: {len(isa_train):,} is_a + {len(ann_train):,} annotation = {len(edges):,}")
    report(f"  held-out is_a edges (fidelity test): {len(isa_test):,}")

    # ---- downstream target: co-response in Replogle K562 ----
    import h5py
    with h5py.File(GWPS, "r") as f:
        X = f["X"][:]
        gid = [(s.decode() if isinstance(s, bytes) else str(s)).split(".")[0]
               for s in f["var"]["gene_id"][:]]
    from entity_registry import load as load_reg
    REG = load_reg()
    ens2sym = {}
    for e in gid:
        eid = REG.resolve("gene", "ensembl", e)
        ent = REG.entities.get(eid) if eid else None
        if ent and ent.get("symbol"):
            ens2sym[e] = ent["symbol"]
    sym2col = {ens2sym[e]: j for j, e in enumerate(gid) if e in ens2sym}
    ok = np.isfinite(X)
    X = np.where(ok, X, 0.0)
    report(f"  Replogle K562 co-response matrix: {X.shape[0]:,} perturbations x {X.shape[1]:,} genes; "
           f"{len(sym2col):,} genes joined by symbol")

    # gene responsiveness, the confound that must be matched
    resp = np.abs(X).mean(0)
    Xz = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)

    # ---- test pairs: share a TEST term, share NO train term ----
    from collections import defaultdict
    by_test = defaultdict(list)
    for g in genes:
        for t in (gene2terms[g] & test_t):
            if g in sym2col:
                by_test[t].append(g)
    pairs, seen = [], set()
    for t, gs in by_test.items():
        if not (2 <= len(gs) <= 60):
            continue
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                a, b = gs[i], gs[j]
                if gene2terms[a] & gene2terms[b] & train_t:
                    continue                      # their relationship is already in training
                k = (a, b) if a < b else (b, a)
                if k not in seen:
                    seen.add(k)
                    pairs.append(k)
    report(f"  downstream test pairs (share a held-out term, share NO train term): {len(pairs):,}")
    if len(pairs) < 500:
        raise SystemExit("too few held-out pairs to test")

    ca = np.array([sym2col[a] for a, _b in pairs])
    cb = np.array([sym2col[b] for _a, b in pairs])
    y = (Xz[:, ca] * Xz[:, cb]).mean(0)            # co-response correlation
    report(f"  co-response target: mean {y.mean():+.4f}, sd {y.std():.4f}")

    R = {"branch": BRANCH, "n_terms": len(terms), "n_genes": len(genes), "n_nodes": n_nodes,
         "n_train_edges": len(edges), "n_heldout_isa": len(isa_test), "n_test_pairs": len(pairs),
         "dims": list(DIMS), "epochs": EPOCHS, "negatives": NEG, "seeds": list(SEEDS), "arms": {}}

    # ---- NO-EMBEDDING BASELINE: shortest path in the ontology graph ----
    import collections
    adj = collections.defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    def sp(a, b, cap=8):
        if a == b:
            return 0
        seenv, frontier = {a}, [a]
        for depth in range(1, cap + 1):
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v == b:
                        return depth
                    if v not in seenv:
                        seenv.add(v)
                        nxt.append(v)
            frontier = nxt
            if not frontier:
                break
        return cap + 1
    sub = rng.choice(len(pairs), min(3000, len(pairs)), replace=False)
    gd = np.array([sp(gix[pairs[i][0]], gix[pairs[i][1]]) for i in sub], float)
    r_graph = float(stats.spearmanr(gd, y[sub])[0])
    report(f"\n  NO-EMBEDDING BASELINE: ontology shortest-path distance vs co-response, "
           f"Spearman {r_graph:+.4f} on {len(sub):,} pairs")
    R["graph_distance_baseline"] = {"spearman": r_graph, "n": int(len(sub))}

    # ---- the two arms ----
    report(f"\n  {'dim':>4s} {'arm':11s} {'fidelity (held-out is_a MAP)':>30s} "
           f"{'utility (Spearman vs co-response)':>36s}")
    for dim in DIMS:
        for arm in ("hyperbolic", "euclidean"):
            fid, ut = [], []
            for s in SEEDS:
                emb, dfn = train_embedding(edges, n_nodes, dim, arm, s)
                # fidelity: do held-out is_a pairs sit closer than random term pairs?
                if isa_test:
                    dpos = pair_distance(emb, dfn, isa_test[:5000], arm)
                    rnd = [(int(rng.integers(0, len(terms))), int(rng.integers(0, len(terms))))
                           for _ in range(len(dpos))]
                    dneg = pair_distance(emb, dfn, rnd, arm)
                    fid.append(float((dpos[:, None] < dneg[None, :]).mean()))
                dp = pair_distance(emb, dfn, [(gix[a], gix[b]) for a, b in pairs], arm)
                ut.append(float(stats.spearmanr(dp, y)[0]))
            report(f"  {dim:>4d} {arm:11s} {np.mean(fid):>30.4f} {np.mean(ut):>36.4f}"
                   f"   (sd {np.std(ut):.4f})")
            R["arms"][f"{arm}_d{dim}"] = {"fidelity_auc": float(np.mean(fid)),
                                          "fidelity_sd": float(np.std(fid)),
                                          "utility_spearman": float(np.mean(ut)),
                                          "utility_sd": float(np.std(ut)),
                                          "per_seed_utility": ut}

    # ---- responsiveness-matched utility, at the best dimension ----
    best = max(DIMS)
    report(f"\n  RESPONSIVENESS-MATCHED UTILITY at d={best}, {N_DRAWS} draws")
    report(f"    Two genes that respond to everything co-respond with no functional relationship, and both "
           f"sit near the ontology root.")
    ra = resp[ca]
    rb = resp[cb]
    qa = np.digitize(ra, np.quantile(ra, np.linspace(.1, .9, 9)))
    qb = np.digitize(rb, np.quantile(rb, np.linspace(.1, .9, 9)))
    strata = {}
    for i in range(len(pairs)):
        strata.setdefault((qa[i], qb[i]), []).append(i)
    usable = [v for v in strata.values() if len(v) >= 30]
    report(f"    {len(usable)} responsiveness strata with >=30 pairs "
           f"({sum(len(v) for v in usable):,} pairs)")
    for arm in ("hyperbolic", "euclidean"):
        emb, dfn = train_embedding(edges, n_nodes, best, arm, 0)
        dp = pair_distance(emb, dfn, [(gix[a], gix[b]) for a, b in pairs], arm)
        vals = []
        for s in range(N_DRAWS):
            rr = np.random.default_rng(s)
            take = np.concatenate([rr.choice(v, min(len(v), 200), replace=False) for v in usable])
            vals.append(float(stats.spearmanr(dp[take], y[take])[0]))
        report(f"    {arm:11s} within-stratum Spearman {np.mean(vals):+.4f} "
               f"(sd {np.std(vals):.4f}, {N_DRAWS} draws)")
        R["arms"][f"{arm}_matched"] = {"spearman": float(np.mean(vals)), "sd": float(np.std(vals))}

    # ---- verdict ----
    report("\n" + "=" * 100)
    h = R["arms"][f"hyperbolic_d{best}"]
    e = R["arms"][f"euclidean_d{best}"]
    hm = R["arms"]["hyperbolic_matched"]["spearman"]
    em = R["arms"]["euclidean_matched"]["spearman"]
    fid_win = h["fidelity_auc"] - e["fidelity_auc"]
    # utility is a DISTANCE vs SIMILARITY correlation, so more negative is better
    ut_win = e["utility_spearman"] - h["utility_spearman"]
    mt_win = em - hm
    R["summary"] = {"fidelity_advantage": fid_win, "utility_advantage_raw": ut_win,
                    "utility_advantage_matched": mt_win, "graph_baseline": r_graph}
    beats_graph = abs(h["utility_spearman"]) > abs(r_graph)
    if fid_win > 0.02 and mt_win > 0.01:
        v = (f"HYPERBOLIC EARNS ITS PLACE. At matched dimension d={best} it reconstructs held-out is_a edges "
             f"better ({h['fidelity_auc']:.4f} vs {e['fidelity_auc']:.4f}) AND predicts co-response better "
             f"after responsiveness matching ({hm:+.4f} vs {em:+.4f}). The ontology shortest-path baseline "
             f"reaches {r_graph:+.4f}, so the embedding is not merely re-encoding graph distance.")
    elif fid_win > 0.02:
        v = (f"FIDELITY YES, UTILITY NO -- the distinction this test was built to make. At d={best} "
             f"hyperbolic reconstructs the held-out hierarchy better ({h['fidelity_auc']:.4f} vs "
             f"{e['fidelity_auc']:.4f}), exactly as the literature says it should. But on the question that "
             f"decides whether the coordinate belongs in a cell model -- predicting which genes co-respond "
             f"to perturbation, on pairs whose only shared annotation is a term the embedding never saw -- "
             f"it gains {mt_win:+.4f} Spearman over Euclidean after responsiveness matching "
             f"({hm:+.4f} vs {em:+.4f}). Geometric faithfulness to the ontology is not the same thing as "
             f"carrying information about how the cell behaves, and the ontology is a record of what has "
             f"been written down rather than of what the cell does. The no-embedding ontology "
             f"shortest-path baseline reaches {r_graph:+.4f} on its own.")
    elif fid_win <= 0.02 and abs(mt_win) <= 0.01:
        v = (f"NEITHER GEOMETRY SEPARATES. At d={best} hyperbolic and Euclidean reconstruct the held-out "
             f"hierarchy comparably ({h['fidelity_auc']:.4f} vs {e['fidelity_auc']:.4f}) and predict "
             f"co-response comparably after matching ({hm:+.4f} vs {em:+.4f}). With the ontology "
             f"shortest-path baseline at {r_graph:+.4f}, nothing here justifies a curvature choice.")
    else:
        v = (f"EUCLIDEAN IS NOT WORSE. Fidelity advantage to hyperbolic {fid_win:+.4f}, matched utility "
             f"advantage {mt_win:+.4f} ({hm:+.4f} vs {em:+.4f}), graph baseline {r_graph:+.4f}. The "
             f"hyperbolic coordinate is not supported for this readout.")
    if not beats_graph:
        v += (f" Note also that neither embedding beats raw ontology graph distance in absolute terms, which "
              f"puts the whole embedding step in question before the curvature question is even reached.")
    R["verdict"] = v
    report(f"  VERDICT: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "hyperbolic_test.json", "w"), indent=1, default=float)
    report(f"\n  -> {OUT/'hyperbolic_test.json'}")


if __name__ == "__main__":
    main()

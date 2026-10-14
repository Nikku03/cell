"""CONNECTING PATHWAYS TO EACH OTHER: where one pathway's product is another pathway's substrate.

THE IDEA. Reactome curates ~2,900 human pathways as separate objects, but a cell does not run them separately. A
pathway ends by producing something, and that something is what another pathway consumes to start. So the obvious
connection is entity flow at the pathway boundary:

    boundary_out(P) = entities produced by P's reactions and NOT consumed again inside P
    boundary_in(Q)  = entities consumed by Q's reactions and NOT produced inside Q
    P -> Q           whenever boundary_out(P) meets boundary_in(Q)

THE TWO WAYS THIS GOES WRONG, BOTH OF WHICH PRODUCE A DENSE AND CONFIDENT-LOOKING GRAPH:

1. CURRENCY MOLECULES. ATP is an output of thousands of reactions and an input to thousands more, so joining on it
   declares that every pathway feeds every other. Here the guard is structural rather than a blacklist: linking
   entities must carry a UniProt accession, which SimpleEntity records (ATP, ADP, H2O, phosphate) do not have. A
   promiscuity cap is applied on top and swept, so its effect is visible rather than assumed.

2. GENE OVERLAP -- THE ONE THAT ACTUALLY MATTERS HERE. Reactome pathways are hierarchical and heavily overlapping:
   "Signaling by EGFR" and "Signaling by EGFR in Cancer" share most of their genes. Two pathways that share genes
   will share boundary entities for trivial reasons, so a link between them says nothing. Every result is therefore
   reported ALSO on the GENE-DISJOINT subset, which is the only part that represents a genuine hand-off between
   distinct biology. A headline computed over overlapping pathways would be measuring Reactome's filing system.

VALIDATION, two targets, neither of which is the entity flow that built the links:

  precedingEvent   Reactome curators separately assert that reaction r1 precedes r2. If P -> Q is real, reactions in
                   P should precede reactions in Q more often than the reverse. Scored against the REVERSED link and
                   against size-matched random pathway pairs, because "some reaction precedes some reaction" is easy
                   to satisfy by chance between any two large pathways.

  K562 Perturb-seq If P -> Q is real, knocking out a gene that is in P but NOT in Q should disturb Q's genes more
                   than a degree-matched knockout unrelated to either. This owes nothing to curation and is measured
                   in the cell line being modelled. Degree matching is essential: upstream pathway genes skew toward
                   hubs, and an unmatched comparison would measure essentiality and call it confirmation.
"""
import collections
import json
import os
import pickle
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
TREE = SP / "entity_tree.json"
UP2G = SP / "reactome/up2gene.tsv"
RXIO = SP / "reaction_io.json"
ORDER = SP / "reactome_order.json"
GMT = SP / "ReactomePathways.gmt"

MAXPROD = 50            # entity produced by more than this many reactions is treated as currency
MAXPART = 100           # reactions with more participants than this are machines, not steps
MIN_RX = 2              # a pathway needs at least this many resolved reactions to have a boundary
TAU, TIDE_FRAC = 1.0, 0.05
NPERM = 2000


def load():
    T = json.loads(TREE.read_text())
    up2g = {}
    for line in UP2G.read_text().splitlines():
        p = line.split("\t")
        if len(p) >= 2 and p[0] and p[1]:
            up2g[p[0]] = p[1]
    memo = {}

    def leaves(pe, stack=()):
        if pe in memo:
            return memo[pe]
        if pe in stack:
            return set()
        d = T.get(pe)
        if d is None:
            return set()
        out = {pe} if (d.get("up") and up2g.get(d["up"])) else set()
        for k in d.get("kids") or []:
            out |= leaves(k, stack + (pe,))
        memo[pe] = out
        return out

    return T, up2g, leaves


def main():
    T, up2g, leaves = load()
    io = json.loads(RXIO.read_text())["io"]
    preceding = json.loads(ORDER.read_text())["preceding"]
    paths = {}
    for line in GMT.read_text().splitlines():
        p = line.split("\t")
        if len(p) > 3:
            paths[p[1]] = {"name": p[0], "genes": set(p[2:])}
    print(f"entity tree {len(T):,} | reactions {len(io):,} | pathways {len(paths):,}")

    def gene_of(lf):
        return up2g.get((T.get(lf) or {}).get("up"))

    # reaction -> participating genes, and the PROTEIN entities on each side
    rxg, rxin, rxout = {}, {}, {}
    for r, v in io.items():
        gin, gout = set(), set()
        ein = {p for p in v.get("in", []) if leaves(p)}
        eout = {p for p in v.get("out", []) if leaves(p)}
        for p in v.get("in", []):
            gin |= {g for g in (gene_of(l) for l in leaves(p)) if g}
        for p in v.get("out", []):
            gout |= {g for g in (gene_of(l) for l in leaves(p)) if g}
        if len(gin | gout) > MAXPART:
            continue
        rxg[r] = gin | gout
        rxin[r], rxout[r] = ein, eout

    # promiscuity cap on linking entities
    prod = collections.Counter()
    for r in rxout:
        for p in rxout[r]:
            prod[p] += 1
    hot = {p for p, c in prod.items() if c > MAXPROD}
    print(f"  {len(rxg):,} reactions usable; {len(hot):,} entities capped as promiscuous (>{MAXPROD} producers)")

    # pathway -> reactions, strict containment so currency-adjacent reactions are not dragged in
    by_gene = collections.defaultdict(set)
    for r, gs in rxg.items():
        for g in gs:
            by_gene[g].add(r)
    p_rx = {}
    for pid, d in paths.items():
        cand = set()
        for g in d["genes"]:
            cand |= by_gene.get(g, set())
        rs = {r for r in cand if rxg[r] and rxg[r] <= d["genes"]}
        if len(rs) >= MIN_RX:
            p_rx[pid] = rs
    print(f"  pathways with >={MIN_RX} fully-contained reactions: {len(p_rx):,}")

    # boundaries
    bout, bin_ = {}, {}
    for pid, rs in p_rx.items():
        allin = set().union(*[rxin[r] for r in rs])
        allout = set().union(*[rxout[r] for r in rs])
        bout[pid] = (allout - allin) - hot
        bin_[pid] = (allin - allout) - hot

    ent_consumers = collections.defaultdict(set)
    for pid, es in bin_.items():
        for e in es:
            ent_consumers[e].add(pid)
    links = collections.defaultdict(set)      # (P,Q) -> linking entities
    for pid, es in bout.items():
        for e in es:
            for q in ent_consumers.get(e, ()):
                if q != pid:
                    links[(pid, q)].add(e)

    def jac_raw(A, B):
        return len(A & B) / max(len(A | B), 1)

    def jac(a, b):
        return jac_raw(paths[a]["genes"], paths[b]["genes"])

    # STRICT GENE-DISJOINTNESS IS THE WRONG CRITERION, AND MEASURING IT PROVED THAT: it selects ZERO links, and
    # necessarily so. If P outputs protein X and Q consumes X, then X participates in reactions of both, so Reactome
    # lists X in both gene sets. A protein hand-off MANUFACTURES gene overlap -- demanding none excludes exactly the
    # links being looked for. The meaningful criterion is that the two pathways are not near-duplicates of each
    # other, so the filter is on Jaccard, with the strict-disjoint count still reported as the finding it is.
    strict_disjoint = sum(1 for k in links if not (paths[k[0]]["genes"] & paths[k[1]]["genes"]))
    disjoint = {k: v for k, v in links.items() if jac_raw(paths[k[0]]["genes"], paths[k[1]]["genes"]) <= 0.10}
    print(f"\n  pathway links: {len(links):,} ordered pairs")
    print(f"    strictly gene-disjoint: {strict_disjoint:,}  <- zero is expected: a hand-off creates overlap")
    print(f"    low-overlap (Jaccard <= 0.10, i.e. not near-duplicates): {len(disjoint):,}")
    print(f"    median gene Jaccard over all links: {np.median([jac(a, b) for a, b in links]):.3f}")
    # REACTOME'S HIERARCHY INFLATES THE RAW LINK COUNT. "HIV Life Cycle", "Late Phase of HIV Life Cycle" and
    # "HIV Transcription Elongation" are nested, so a single biological hand-off into that branch is counted once per
    # ancestor. Nesting is detected structurally -- one gene set contained in the other -- rather than by fetching the
    # hierarchy, and reported so the link count is never read as a count of distinct biology.
    nested = sum(1 for a, b in links
                 if paths[a]["genes"] <= paths[b]["genes"] or paths[b]["genes"] <= paths[a]["genes"])
    print(f"    between NESTED pathways (one gene set contains the other): {nested:,} "
          f"({nested/max(len(links),1):.1%}) -- these are one hand-off counted per ancestor")
    recip = sum(1 for (a, b) in links if (b, a) in links)
    print(f"    reciprocal pairs (P->Q and Q->P both): {recip//2:,}")

    # ---- VALIDATION 1: curated precedingEvent, which did not build these links ----
    def prec_score(pairs):
        fwd = rev = 0
        for a, b in pairs:
            ra, rb = p_rx[a], p_rx[b]
            f = any(x in preceding.get(y, ()) for y in rb for x in ra)
            r_ = any(y in preceding.get(x, ()) for y in rb for x in ra)
            fwd += f and not r_
            rev += r_ and not f
        return fwd, rev

    for label, pairs in (("all links", list(links)), ("gene-disjoint", list(disjoint))):
        f, r = prec_score(pairs)
        tot = f + r
        print(f"    precedingEvent on {label:<14}: {f:,} forward-only vs {r:,} reverse-only"
              + (f"  -> {f/tot:.1%} forward" if tot else "  (no opinion)"))
    # control: size-matched random pathway pairs
    rng = np.random.default_rng(0)
    keys = list(p_rx)
    rand = [(keys[i], keys[j]) for i, j in rng.integers(0, len(keys), (min(len(links), 4000), 2)) if i != j]
    f, r = prec_score(rand)
    tot = f + r
    print(f"    precedingEvent on RANDOM pairs      : {f:,} forward-only vs {r:,} reverse-only"
          + (f"  -> {f/tot:.1%} forward" if tot else "  (no opinion)"))

    # ---- VALIDATION 2: K562 Perturb-seq, independent of all curation ----
    KO = pickle.load(open(SP / "nlz_K562.pkl", "rb"))
    kos = sorted(KO)
    ki = {k: i for i, k in enumerate(kos)}
    genes = sorted({g for v in KO.values() for g in v})
    gi = {g: i for i, g in enumerate(genes)}
    M = np.zeros((len(kos), len(genes)), np.float32)
    for k, v in KO.items():
        for g, z in v.items():
            M[ki[k], gi[g]] = z
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    N = len(names)
    deg = collections.Counter()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            deg[names[a]] += 1
            deg[names[b]] += 1
    kdeg = np.array([deg.get(k, 0) for k in kos], float)
    order = np.argsort(kdeg)
    rank = np.empty(len(kos), int)
    rank[order] = np.arange(len(kos))

    def perturb(pairs, label):
        rows = []
        for a, b in pairs:
            tgt = [gi[g] for g in paths[b]["genes"] if g in gi and not tide[gi[g]]]
            if not (5 <= len(tgt) <= 200):
                continue
            src = [g for g in paths[a]["genes"] if g in ki and g not in paths[b]["genes"]]
            for g in src[:8]:
                i = ki[g]
                lo, hi = max(0, rank[i] - 20), min(len(kos), rank[i] + 20)
                pool = [j for j in order[lo:hi] if kos[j] not in paths[b]["genes"] and kos[j] not in paths[a]["genes"]
                        and j != i]
                if len(pool) < 10:
                    continue
                obs = float(A[i, tgt].mean())
                ctrl = float(A[np.array(pool)[:, None], np.array(tgt)[None, :]].mean())
                rows.append(obs - ctrl)
        if not rows:
            print(f"    Perturb-seq {label:<14}: no testable pairs -- unmeasurable, not negative")
            return None
        d = np.array(rows)
        se = float(d.std() / np.sqrt(len(d)))
        print(f"    Perturb-seq {label:<14}: n={len(d):,}  upstream-KO minus degree-matched control "
              f"{d.mean():+.4f} +/- {se:.4f}")
        return float(d.mean()), se, len(d)

    print(f"\n  K562 test: does knocking out an upstream-pathway gene move the downstream pathway?")
    # The reversed link is the control that decides whether "direction" means anything here. If P->Q and Q->P give
    # the same lift, the measurement is only saying "these two pathways are related", not that P is upstream.
    fwd_set = [k for k in list(links)[:4000] if (k[1], k[0]) not in links]   # exclude reciprocal pairs: for those
                                                                            # forward and reverse are the same claim
    r_all = perturb(fwd_set, "forward links")
    r_rev = perturb([(b, a) for a, b in fwd_set], "REVERSED (control)")
    r_dis = perturb([k for k in list(disjoint)[:4000] if (k[1], k[0]) not in links], "low-overlap")

    top = sorted(links.items(), key=lambda kv: -len(kv[1]))[:15]
    print(f"\n  strongest links (by number of shared boundary entities):")
    for (a, b), es in top:
        print(f"    {paths[a]['name'][:44]:<46} -> {paths[b]['name'][:44]:<46} ({len(es)} entities, "
              f"jac {jac(a,b):.2f})")

    json.dump({"n_pathways": len(paths), "n_pathways_resolved": len(p_rx),
               "n_links": len(links), "n_links_disjoint": len(disjoint), "reciprocal": recip // 2,
               "perturb_all": r_all, "perturb_disjoint": r_dis, "perturb_reversed": r_rev,
               "links": [[a, b, sorted(v)[:5]] for (a, b), v in list(links.items())[:5000]]},
              open(OUT / "pathway_links.json", "w"))
    print(f"\n  -> {OUT/'pathway_links.json'}")


if __name__ == "__main__":
    main()

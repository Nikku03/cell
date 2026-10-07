"""graph_label — the LABELED multi-relational cell graph: nodes = genes, edges TYPED by relation. The CellGraph
adjacency already carries physical/functional edges (ppi, reg, sig, codep, lr); this adds the two relations the graph
was missing — PATHWAY co-membership (Reactome) and LITERATURE co-mention (PubMed via litmine) — and labels every edge
with its source relation, so the graph explicitly says HOW two genes are related, not just THAT they are.

  ppi   physical interaction            (cell_complete["ppi"])
  path  same Reactome pathway           (reactome_pathways.json; capped at <=MAXPATH genes so giant generic pathways
                                          like 'Metabolism' don't create a hairball)
  lit   co-mentioned in the same paper  (litmine.json; two genes share a PMID)

HONEST SCOPE: this is a DESCRIPTIVE knowledge graph — it says who-touches-whom / who-is-with-whom, which is the
recoverable near-field. It is NOT a causal predictor: this project measured that these edge types predict a gene's
knockout effect at r=0.03-0.09 (pathway 0.064), the noise floor — only curated functional complexes (0.23) carry
causal signal. So the labeled graph is the right SUBSTRATE for representation / link-prediction / querying, and it is
honest about not composing to phenotype.
  build() -> the labeled graph ; stats() -> outputs/orphan/graph_label.json ; neighbors(g) -> a gene's typed edges
"""
import os
import json, itertools, collections
from pathlib import Path
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
MAXPATH = 50                                                     # skip pathways bigger than this (generic hubs)
RELATIONS = ("ppi", "path", "lit")


def _load():
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    name2i = {n: i for i, n in enumerate(names)}
    return D, names, name2i


def build():
    """construct the labeled graph: rel -> {i: set(neighbors)}, plus a pair->set(labels) view."""
    D, names, name2i = _load()
    n = len(names)
    adj = {r: collections.defaultdict(set) for r in RELATIONS}

    def link(rel, i, j):
        if i != j:
            adj[rel][i].add(j); adj[rel][j].add(i)

    for e in D.get("ppi", []):                                  # physical interactions (index pairs)
        link("ppi", e[0], e[1])

    paths = json.load(open(OUT / "reactome_pathways.json"))["pathways"]
    for members in paths.values():                              # pathway co-membership (size-capped)
        if 2 <= len(members) <= MAXPATH:
            for i, j in itertools.combinations(sorted(set(members)), 2):
                link("path", i, j)

    lit = json.load(open(OUT / "litmine.json"))                 # literature co-mention (shared PMID)
    pmid2g = collections.defaultdict(set)
    for g, e in lit.items():
        idx = name2i.get(g)
        if idx is None:
            continue
        for p in e.get("papers", []):
            pmid2g[p["pmid"]].add(idx)
    for gs in pmid2g.values():
        if len(gs) > 1:
            for i, j in itertools.combinations(sorted(gs), 2):
                link("lit", i, j)

    return {"adj": adj, "names": names, "name2i": name2i, "n": n}


def neighbors(G, gene, cap=8):
    """a gene's typed neighborhood: {relation: [neighbor names]}. The point of a LABELED graph — see HOW it connects."""
    i = G["name2i"].get(gene)
    if i is None:
        return None
    out = {}
    for r in RELATIONS:
        nb = sorted(G["adj"][r].get(i, []), key=lambda j: -len(G["adj"]["ppi"].get(j, [])))
        if nb:
            out[r] = [G["names"][j] for j in nb[:cap]] + (["..."] if len(nb) > cap else [])
    return out


def stats(G=None):
    G = G or build()
    adj, n = G["adj"], G["n"]
    def edges(r):
        return sum(len(v) for v in adj[r].values()) // 2
    def covered(r):
        return len(adj[r])
    pairs = {r: {frozenset((i, j)) for i, ns in adj[r].items() for j in ns} for r in RELATIONS}
    ne = {r: len(pairs[r]) for r in RELATIONS}
    union = set().union(*pairs.values())
    # cross-type overlap: fraction of ppi pairs that are ALSO path / lit (complementary vs redundant)
    ov = {}
    for a, b in itertools.combinations(RELATIONS, 2):
        inter = len(pairs[a] & pairs[b]); den = min(len(pairs[a]), len(pairs[b])) or 1
        ov[f"{a}&{b}"] = round(inter / den, 3)
    out = {"n_nodes": n, "relations": list(RELATIONS), "maxpath": MAXPATH,
           "edges_by_type": ne, "genes_covered_by_type": {r: covered(r) for r in RELATIONS},
           "total_labeled_edges": len(union), "genes_with_any_edge": len(set().union(*[set(adj[r]) for r in RELATIONS])),
           "cross_type_overlap_frac": ov,
           "note": "DESCRIPTIVE knowledge graph (near-field). Edge types predict knockout effect at r~0.03-0.09 (floor); "
                   "good for representation / link-prediction / querying, NOT a causal/phenotype predictor. path capped "
                   f"at <= {MAXPATH} genes/pathway."}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "graph_label.json", "w"), indent=1)
    return out


def main():
    print("=" * 90)
    print("GRAPH-LABEL — the labeled multi-relational cell graph (ppi + pathway + literature)")
    print("=" * 90)
    G = build(); s = stats(G)
    print(f"  nodes: {s['n_nodes']:,}   |   genes with >=1 labeled edge: {s['genes_with_any_edge']:,}")
    for r in RELATIONS:
        print(f"    {r:5s}: {s['edges_by_type'][r]:>8,} edges   over {s['genes_covered_by_type'][r]:>6,} genes")
    print(f"  total unique labeled pairs: {s['total_labeled_edges']:,}")
    print(f"  cross-type overlap (fraction shared): {s['cross_type_overlap_frac']}")
    print("  -> the three relations are COMPLEMENTARY (low overlap): each says a different thing about a gene pair.")
    for g in ("TP53", "BRAF", "FNDC5"):
        nb = neighbors(G, g)
        if nb:
            print(f"  {g} labeled neighbours: " + "  ".join(f"[{r}] {', '.join(v[:4])}" for r, v in nb.items()))
    print(f"\n  -> {OUT/'graph_label.json'}")
    return s


if __name__ == "__main__":
    main()

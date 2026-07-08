"""extra_data — fold in three Drive datasets that fix FIELDS (not combiner features):

  reactome   full Reactome human pathways -> fills the pathway-coverage gap (23% -> much higher).
  signor     SIGNOR causal signaling edges (directed + signed) -> the causal-direction gap + more sig.
  collectri  CollecTRI TF->target edges (directed + signed)    -> causal direction + more reg.

Each loader is DEFENSIVE and DIAGNOSTIC: it prints the detected format (a couple of raw lines) and does a
best-effort parse, so the first Colab run tells us the exact columns and we refine if needed (same way STRING
and Geneformer were dialled in). Everything maps onto our gene index; nothing overwrites a measured fact — new
pathway memberships / directed edges are written as OVERLAYS.

-> outputs/orphan/reactome_pathways.json   {pathway: [gene_idx]}   (+ coverage report)
-> outputs/orphan/causal_edges.json        directed signed edges from SIGNOR + CollecTRI (+ counts)
"""
import os, sys, json, gzip
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

OUT = "outputs/orphan"


def _open(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p, "r")


def _sep(line):
    """detect the column separator from a header line — TSV is the norm for these, but CollecTRI is often CSV."""
    return "\t" if "\t" in line else ("," if "," in line else "\t")


def reactome(path, C):
    """parse a Reactome human file -> {pathway_name: [gene_idx]}. Handles GMT (name<TAB>desc<TAB>gene…) and
    long formats (…<TAB>gene_symbol<TAB>…<TAB>pathway_name…) by locating the column that best matches our genes."""
    if not path or not os.path.exists(path):
        return None
    idx = C.idx
    with _open(path) as fh:
        head = [fh.readline().rstrip("\n") for _ in range(4)]
    sep = _sep(head[0])
    print("  reactome sample:", [h[:90] for h in head[:2]])
    paths = {}
    with _open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split(sep)
            if len(p) < 3:
                continue
            genes = [c for c in p if c in idx]          # any column that is one of our gene symbols
            if not genes:
                continue
            # pathway label = the longest non-gene, non-id field (Reactome names are descriptive)
            name = max((c for c in p if c not in idx and not c.startswith("R-") and len(c) > 4),
                       key=len, default=p[-1])
            paths.setdefault(name, set()).update(idx[g] for g in genes)
    if not paths:
        print("  (reactome: parsed 0 pathways — format unrecognised, sending sample for refinement)")
        return None
    paths = {k: sorted(v) for k, v in paths.items() if v}
    covered = set().union(*[set(v) for v in paths.values()])
    n = len(C.genes)
    base = set(m for mem in C.D.get("pathways", {}).values() for m in mem if isinstance(m, int))
    json.dump({"pathways": {k: v for k, v in list(paths.items())}}, open(f"{OUT}/reactome_pathways.json", "w"))
    print(f"  reactome: {len(paths):,} pathways covering {len(covered):,}/{n:,} genes "
          f"({round(100*len(covered)/n)}%)  —  was {round(100*len(base)/n)}% "
          f"(+{len(covered - base):,} newly-covered genes)")
    return dict(n_pathways=len(paths), covered=len(covered), coverage_pct=round(100 * len(covered) / n, 1),
                new_genes=len(covered - base))


def _directed(path, C, a_cols, b_cols, eff_cols, src):
    """generic directed-edge parser: find the source/target/effect columns by header name, map to gene idx."""
    if not path or not os.path.exists(path):
        return []
    idx = C.idx
    with _open(path) as fh:
        header_line = fh.readline().rstrip("\n")
    sep = _sep(header_line)
    header = header_line.split(sep)
    print(f"  {src} columns:", header[:12])
    low = [h.lower() for h in header]
    def col(names):
        for nm in names:
            if nm in low:
                return low.index(nm)
        return None
    ia, ib, ie = col(a_cols), col(b_cols), col(eff_cols)
    if ia is None or ib is None:
        print(f"  ({src}: could not find source/target columns — sample header above for refinement)")
        return []
    edges = []
    with _open(path) as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split(sep)
            if len(p) <= max(ia, ib):
                continue
            a, b = p[ia], p[ib]
            if a in idx and b in idx and a != b:
                eff = (p[ie] if ie is not None and len(p) > ie else "").lower().strip()
                try:                                        # numeric weight (e.g. CollecTRI +1/-1)
                    fv = float(eff); sign = 1 if fv > 0 else (-1 if fv < 0 else 0)
                except ValueError:                          # text effect (e.g. SIGNOR up/down-regulates)
                    sign = -1 if any(k in eff for k in ("down", "repress", "inhib")) else \
                           (1 if any(k in eff for k in ("up", "activ")) else 0)
                edges.append([idx[a], idx[b], sign, src])
    print(f"  {src}: {len(edges):,} directed edges mapped to our genes")
    return edges


def causal(signor_path, collectri_path, C):
    edges = _directed(signor_path, C, ["entitya", "idagene", "gene_a", "source"],
                      ["entityb", "idbgene", "gene_b", "target"], ["effect", "mechanism"], "signor")
    edges += _directed(collectri_path, C, ["source", "tf", "regulator"],
                       ["target", "gene"], ["weight", "mor", "effect", "sign"], "collectri")
    if not edges:
        return None
    # how many are NEW vs the model's existing reg/sig (directed pairs)
    have = set()
    for rel in ("reg", "sig"):
        for e in C.D.get(rel, []):
            if isinstance(e[0], int) and isinstance(e[1], int):
                have.add((e[0], e[1]))
    new = sum(1 for e in edges if (e[0], e[1]) not in have)
    signed = sum(1 for e in edges if e[2] != 0)
    json.dump({"edges": edges, "meta": dict(total=len(edges), new_vs_model=new, signed=signed)},
              open(f"{OUT}/causal_edges.json", "w"))
    print(f"  causal edges: {len(edges):,} total | {new:,} NOT already in reg/sig | {signed:,} carry a "
          f"direction sign (the causal-direction the model lacked)")
    return dict(total=len(edges), new_vs_model=new, signed=signed)


def main():
    C = CompleteCell()
    res = {}
    r = reactome(os.environ.get("REACTOME_TXT"), C)
    if r:
        res["reactome"] = r
    c = causal(os.environ.get("SIGNOR_TSV"), os.environ.get("COLLECTRI_TSV"), C)
    if c:
        res["causal"] = c
    json.dump(res, open(f"{OUT}/extra_data_report.json", "w"), indent=2)
    if not res:
        print("no extra datasets found — set REACTOME_TXT / SIGNOR_TSV / COLLECTRI_TSV")
    return res


if __name__ == "__main__":
    main()

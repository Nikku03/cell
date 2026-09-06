"""chip_edges — MEASURED TF->target edges from ENCODE ChIP-seq (HepG2), the result of the ChIP-in-ML experiment.

What this session's experiment found (chip_reg_edges.json, 12 liver TFs in HepG2):
  - 68,233 measured TF->target edges; **97% are NEW** (not in our curated regulatory layer of ~57k). So ChIP-seq
    nearly DOUBLES the regulatory map from 12 TFs alone -- a real, measured COVERAGE gain.
  - Blind-holdout on these edges: overall common-neighbour AUC 0.824, but **96% of held-out edges are EASY**
    (endpoints share a train neighbour) because TFs bind thousands of genes -> the graph is hub-dense and
    interpolation is trivial. On the genuine HARD/discovery set (4%), **AUC 0.15 (below chance)** -- ChIP-seq
    edges do NOT let the model discover novel regulation; they make interpolation denser.
  - Honest caveats baked into the artifact: this is measured BINDING, not validated regulation; HepG2-specific;
    TSS-proximity (+-5kb) assignment over-calls targets.
=> Use as a MEASURED-BINDING prior for the HepG2 regulatory layer, flagged as such. Not universal, not functional-
   validated, and not a discovery signal.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

OUT = "outputs/orphan"


def load(path=f"{OUT}/chip_reg_edges.json"):
    return json.load(open(path)) if os.path.exists(path) else None


def signed_edges(C, d=None):
    """[(tf_idx, target_idx, +1, meta)] mapped to model indices. Sign +1 = 'TF binds near target' (activation is
    the null; ChIP alone cannot give direction -- flag as binding, not signed regulation)."""
    d = d or load()
    if not d:
        return []
    out = []
    for tf, targets in d.get("tf_targets", {}).items():
        i = C.idx.get(tf)
        if i is None:
            continue
        for t in targets:
            j = C.idx.get(t)
            if j is not None and j != i:
                out.append((i, j, 1, "chip:HepG2:binding-not-regulation"))
    return out


def coverage_vs_curated(C):
    """how many ChIP edges are NEW vs the curated causal/regulatory layer already in the model."""
    d = load()
    if not d:
        return None
    cur = set()
    for u, lst in (getattr(C, "causal_out", {}) or {}).items():
        for it in lst:
            cur.add((u, it[0]))
    edges = [(C.idx[tf], C.idx[t]) for tf, ts in d["tf_targets"].items() if tf in C.idx
             for t in ts if t in C.idx]
    new = [e for e in edges if e not in cur and (e[1], e[0]) not in cur]
    return {"chip_edges": len(edges), "curated": len(cur), "new": len(new),
            "cell_type": d.get("cell_type"), "caveat": d.get("caveat")}


if __name__ == "__main__":
    sys.path.insert(0, ".")
    from complete_cell import CompleteCell
    C = CompleteCell()
    print(coverage_vs_curated(C))
    print("signed edges:", len(signed_edges(C)))

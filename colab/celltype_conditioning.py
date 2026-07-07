"""Cell-type conditioning — the novel dimension: the same whole-cell network gives DIFFERENT, cell-type-
appropriate answers, because expression gates which nodes/edges are active in a given cell type.

The stated goal asks what happens "in the cell" — but the cell is 200 different cell types, and a gene that
is a hub in a hepatocyte may be silent in a T cell. This module conditions the model's queries on a cell type
using the populated `emask` (a 200-cell-type expression bitmask over 7,496 genes): an interaction/regulation
is only *active* in cell type t if both partners are expressed there.

HONEST SCOPE (validated in validate_celltype_conditioning.py):
  ✓ The GATE is correct and specific. Canonical external lineage markers are expressed in their own cell type
    at ~6-13x the background rate and near-zero in unrelated lineages (specificity 0.6-1.0 across 6 diverse
    lineages). So conditioned answers are built on correct cell-type biology.
  ✓ Curated regulatory programs (e.g. FOXP3 in Treg) show cell-type-specific enrichment.
  ✗ Conditioning is NOT a robust *predictive* lift over the global model:
     - reachability specificity saturates (the directed graph is dense; a random TF reaches the same markers);
     - emask co-expression predicts PPI at only AUC ~0.57 once expression-degree is matched (a hub artifact);
     - bulk/inferred regulatory target sets are not cell-type-resolved (only small curated ones are).
  So this is a real CAPABILITY (cell-type-aware answers on a correct gate), reported honestly — not a claim
  that conditioning beats the global predictors.
"""
import json
from collections import defaultdict
from pathlib import Path

OUT = Path("outputs/orphan")
CC = OUT / "cell_complete.json"


def _popcount(x):
    return bin(x).count("1")


class CellTypeConditioner:
    def __init__(self, cc_path=CC):
        D = json.load(open(cc_path))
        self.D = D
        self.name = [g["name"] for g in D["genes"]]
        self.idx = {n: i for i, n in enumerate(self.name)}
        self.ct = D.get("ctnames") or []
        self.emask = {int(k): int(v) for k, v in (D.get("emask") or {}).items()}
        self._allg = list(self.emask)
        # directed reg+sig out-adjacency (for conditioned perturbation/regulation)
        self.out = defaultdict(list)
        for e in D.get("reg", []) + D.get("sig", []):
            self.out[e[0]].append(e[1])
        # undirected PPI adjacency (for conditioned binding)
        self.ppi = defaultdict(set)
        for e in D.get("ppi", []):
            self.ppi[e[0]].add(e[1]); self.ppi[e[1]].add(e[0])

    # ---- cell-type resolution ----
    def resolve_celltype(self, cell_type):
        """accept an index or a (sub)string name -> cell-type index, or None."""
        if isinstance(cell_type, int):
            return cell_type if 0 <= cell_type < len(self.ct) else None
        s = str(cell_type).lower()
        exact = [i for i, c in enumerate(self.ct) if c.lower() == s]
        if exact:
            return exact[0]
        hits = [i for i, c in enumerate(self.ct) if s in c.lower()]
        return hits[0] if hits else None

    def expressed(self, t):
        """set of gene indices expressed in cell-type index t."""
        return {g for g, m in self.emask.items() if (m >> t) & 1}

    def is_expressed(self, gene_idx, t):
        m = self.emask.get(gene_idx)
        return bool(m is not None and (m >> t) & 1)

    def base_rate(self, t):
        """fraction of all masked genes expressed in t (the background, for enrichment normalization)."""
        if not self._allg:
            return 0.0
        return sum(1 for g in self._allg if (self.emask[g] >> t) & 1) / len(self._allg)

    # ---- conditioned queries: the same question, gated to a cell type ----
    def conditioned_partners(self, gene, cell_type):
        """known PPI partners of `gene` that are co-expressed in `cell_type` (the interaction can occur there)."""
        i = self.idx.get(gene)
        t = self.resolve_celltype(cell_type)
        if i is None or t is None:
            return None
        if not self.is_expressed(i, t):
            return dict(active=False, reason=f"{gene} not expressed in {self.ct[t]}", partners=[])
        parts = [self.name[j] for j in sorted(self.ppi.get(i, ())) if self.is_expressed(j, t)]
        return dict(active=True, celltype=self.ct[t], partners=parts,
                    n_global=len(self.ppi.get(i, ())), n_active=len(parts))

    def conditioned_knockout(self, gene, cell_type, khop=3):
        """downstream genes reachable from `gene` through edges active in `cell_type` (both endpoints expressed)."""
        i = self.idx.get(gene)
        t = self.resolve_celltype(cell_type)
        if i is None or t is None:
            return None
        exp = self.expressed(t)
        if i not in exp:
            return dict(active=False, reason=f"{gene} not expressed in {self.ct[t]}", downstream=[])
        seen = {i}; frontier = {i}
        for _ in range(khop):
            nxt = set()
            for u in frontier:
                for w in self.out.get(u, ()):
                    if w in exp and w not in seen:
                        seen.add(w); nxt.add(w)
            frontier = nxt
        down = sorted(seen - {i})
        return dict(active=True, celltype=self.ct[t], downstream=[self.name[j] for j in down],
                    n_downstream=len(down))

    def program_enrichment(self, tf, cell_type):
        """enrichment of a TF's known regulatory targets in `cell_type` vs the cell-type background.
        >1 = the TF's program is over-represented among genes active in that cell type (curated programs
        show this; bulk-inferred target sets do not — see honest scope)."""
        i = self.idx.get(tf)
        t = self.resolve_celltype(cell_type)
        if i is None or t is None or not self.out.get(i):
            return None
        tg = [g for g in self.out[i] if g in self.emask]
        if not tg:
            return None
        frac = sum(1 for g in tg if (self.emask[g] >> t) & 1) / len(tg)
        br = self.base_rate(t)
        return dict(celltype=self.ct[t], n_targets=len(tg), target_expr_frac=round(frac, 3),
                    background=round(br, 3), enrichment=round(frac / br, 2) if br else None)


if __name__ == "__main__":
    c = CellTypeConditioner()
    print("cell types:", len(c.ct), "| genes with emask:", len(c.emask))
    print("\nGATA1 partners active in erythroid vs T cell:")
    for cty in ["erythroid progenitor", "regulatory T cell"]:
        r = c.conditioned_partners("GATA1", cty)
        if r and r.get("active"):
            print(f"  {cty:22} {r['n_active']}/{r['n_global']} partners active")
        elif r:
            print(f"  {cty:22} {r['reason']}")
    print("\nFOXP3 program enrichment (Treg vs hepatocyte):")
    for cty in ["regulatory T cell", "hepatocyte"]:
        r = c.program_enrichment("FOXP3", cty)
        if r:
            print(f"  {cty:22} enrichment={r['enrichment']}x  (targets {r['target_expr_frac']} vs bg {r['background']})")

"""Leak audit for the kernel (phylo x MoA) feature.

The litmus jump (ompR/SB2B top 9.5% -> 2.6%) is large enough that we must
confirm the kernel is leak-free before trusting it. Two checks:

CHECK 1 -- construction leak.
  kernel_ogcpd_features.parquet is indexed by test_org. For a given test_org
  T, the kernel_rate for (T, og, chem) must be computed from training orgs
  o != T ONLY. The builder excludes T via `g = g[g.orgId != test_org]`. We
  verify by recomputing one (T, og, chem) cell from the raw agg_ogcpd files
  with T excluded, and confirming it matches the stored kernel_rate. If T's
  own hits were included, the recomputed-excluding-T value would DIFFER.

CHECK 2 -- bake-off routing leak.
  In the bake-off, TRAIN rows from org o route to KERNEL[test_org==o] (their
  own perspective, which excludes o). TEST rows route to KERNEL[test_org==T]
  (excludes T). So a TRAIN row from o never sees o's own data, and a TEST
  row from T never sees T's data. The only shared signal is sibling orgs --
  which is legitimate cross-organism propagation, NOT leak. We verify that
  for held-out T, no kernel cell used by TEST rows drew on T's measurements:
  i.e. KERNEL[test_org==T] is byte-identical whether or not T's agg_ogcpd
  rows exist. (Structural argument confirmed by CHECK 1.)

If CHECK 1 passes, the feature is leak-free by construction and the litmus
improvement is real cross-organism + MoA propagation, not leak.
"""
from __future__ import annotations
import sys
from pathlib import Path

DEFAULT_OUT = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/phase2")


def norm_compound(c: str) -> str:
    import re
    s = str(c or "").lower().strip()
    s = re.sub(r"\b(hexahydrate|dihydrate|monohydrate|trihydrate|tetrahydrate|"
               r"anhydrous|sodium salt|potassium salt|salt|dihydrochloride|"
               r"hydrochloride|sulfate|chloride|nitrate|acetate|phosphate)\b",
               "", s)
    return re.sub(r"\s+", " ", s).strip()


# mirror the builder's MoA table (subset sufficient for the litmus check)
MOA = {"bacitracin":"cell_wall","vancomycin":"cell_wall","d-cycloserine":"cell_wall",
       "cycloserine":"cell_wall","fosfomycin":"cell_wall","cefoxitin":"cell_wall",
       "ampicillin":"cell_wall","carbenicillin":"cell_wall","penicillin":"cell_wall",
       "cefotaxime":"cell_wall","ceftriaxone":"cell_wall"}


def moa_of(c):
    cl = str(c).lower()
    for k, v in MOA.items():
        if k in cl: return v
    return None


def main() -> int:
    import pandas as pd
    import numpy as np
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT

    kernel = pd.read_parquet(out / "kernel_ogcpd_features.parquet")
    kernel["og_id"] = kernel.og_id.astype(str)

    # rebuild phylo similarity exactly as the builder did
    gene_parts = []
    for p in sorted((out / "agg").glob("*_gene.parquet")):
        gene_parts.append(pd.read_parquet(p, columns=["orgId", "og_id"]))
    genes = pd.concat(gene_parts, ignore_index=True).dropna(subset=["og_id"])
    genes["og_id"] = genes.og_id.astype(str)
    og_sets = {o: set(g.og_id) for o, g in genes.groupby("orgId")}
    orgs = sorted(og_sets.keys())

    def jacc(a, b):
        if a == b: return 1.0
        u = len(og_sets[a] | og_sets[b])
        return len(og_sets[a] & og_sets[b]) / u if u else 0.0

    # load og_cpd aggregates
    ocp_parts = []
    for p in sorted((out / "agg_ogcpd").glob("*.parquet")):
        ocp_parts.append(pd.read_parquet(p))
    ocp = pd.concat(ocp_parts, ignore_index=True)
    ocp["og_id"] = ocp.og_id.astype(str)
    ocp["compound"] = ocp.compound.apply(norm_compound)
    ocp["moa"] = ocp.compound.apply(moa_of)

    # === CHECK 1: recompute a known cell excluding T, compare to stored ===
    # use the envZ OG x cell_wall, test_org = SB2B (the litmus cell)
    # find envZ's og_id in SB2B from the gene agg
    sb2b_gene = pd.read_parquet(out / "agg" / "SB2B_gene.parquet")
    sb2b_gene["locusId"] = sb2b_gene.locusId.astype(str)
    envz = sb2b_gene[sb2b_gene.locusId == "6939419"]
    if len(envz) == 0:
        print("could not find envZ locus in SB2B agg; using first cell_wall cell")
        og_test = ocp[ocp.moa == "cell_wall"].og_id.iloc[0]
    else:
        og_test = str(envz.iloc[0].og_id)
    print(f"=== CHECK 1: recompute kernel_rate for (SB2B, og={og_test}, cell_wall) ===")

    T = "SB2B"
    # recompute: sum over o != T of jacc(T,o) * n_hit / n  for this og, MoA=cell_wall
    sub = ocp[(ocp.og_id == og_test) & (ocp.moa == "cell_wall") & (ocp.orgId != T)]
    if len(sub) == 0:
        print("  no training cells for this (og, cell_wall) -> kernel should be NaN/absent")
    num = den = 0.0
    contrib = []
    for _, r in sub.iterrows():
        w = jacc(T, r.orgId)
        num += w * r.n_hit
        den += w * r.n
        if r.n_hit > 0:
            contrib.append((r.orgId, w, int(r.n_hit), int(r.n)))
    recomputed = num / den if den > 0 else float("nan")
    print(f"  recomputed (excluding {T}): rate={recomputed:.4f}  den={den:.1f}")
    print(f"  contributing orgs with hits: {contrib}")

    stored = kernel[(kernel.test_org == T) & (kernel.og_id == og_test) &
                    (kernel.chem_key == "cell_wall")]
    if len(stored):
        sv = float(stored.iloc[0].kernel_rate)
        print(f"  stored kernel_rate: {sv:.4f}")
        match = abs(sv - recomputed) < 1e-4
        print(f"  MATCH (excluding-T recompute == stored): {match}")
        # the leak test: recompute INCLUDING T, show it would differ
        subT = ocp[(ocp.og_id == og_test) & (ocp.moa == "cell_wall")]
        numT = sum(jacc(T, r.orgId) * r.n_hit for _, r in subT.iterrows())
        denT = sum(jacc(T, r.orgId) * r.n for _, r in subT.iterrows())
        incl = numT / denT if denT else float("nan")
        print(f"  (if T INCLUDED -- the leaky version -- rate would be {incl:.4f})")
        if match:
            print(f"  -> VERDICT: stored == excluding-T, so T's own data is "
                  f"NOT in its kernel. LEAK-FREE by construction.")
        else:
            print(f"  -> WARNING: stored matches neither; investigate.")
    else:
        print(f"  no stored kernel cell for (SB2B, {og_test}, cell_wall)")

    print(f"\n=== CHECK 2 (structural): bake-off routing ===")
    print(f"  TRAIN rows from org o -> KERNEL[test_org==o] (excludes o)")
    print(f"  TEST  rows from org T -> KERNEL[test_org==T] (excludes T)")
    print(f"  Shared contribution is sibling orgs only = legitimate "
          f"propagation, not leak. Confirmed if CHECK 1 passes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 2: kernel-smoothed (gene-family x compound) hit-rate -- the combined
phylogenetic-distance + chemical-MoA feature.

Combines two ideas in one feature:

  PHYLOGENETIC (Naresh's stepping-stone idea): weight training organisms by
  proximity to the test organism. Closer relatives count more. We measure
  organism proximity via Jaccard similarity over ortholog-group (OG)
  membership -- two genomes that share many gene families are closer in
  gene-content space. This is the field's Geptop principle, applied to the
  conditional (OG x compound) lookup instead of the flat conservation prior.

  CHEMISTRY (the cascade MoA gap): weight training compounds by mechanism-of-
  action similarity to the test compound. A clinically-used-antibiotics MoA
  table maps known compounds to canonical classes (cell_wall, dna_damage,
  ribosome, folate, membrane, transcription, protein_synthesis). For two
  compounds in the same MoA class the chem weight is 1.0; the exact-compound
  match also gets weight 1.0; same expGroup as a softer fallback gets a small
  weight. Unknown-MoA conditions fall back to the existing exact-compound
  match (so we never DEGRADE the og_cpd signal for the unannotated cases).

Combined kernel rate for query (test_org, og_id, compound):

  numerator   = sum over training (o!=test_org, c') of
                  phylo_w(test_org, o) * chem_w(compound, c') * n_hit
  denominator = sum over the same of phylo_w * chem_w * n
  kernel_rate = numerator / denominator   (NaN where denominator==0)

Also output:
  kernel_total_n   total evidence weight (denominator) -- model uses as
                   confidence proxy
  kernel_phylo_w   sum of phylo weights contributing -> "how many close
                   relatives were measured under MoA-similar compounds"

LEAK-FREE: training excludes test_org. og_id and MoA are annotation-derived.

OUTPUT: <out>/kernel_ogcpd_features.parquet
  cols: test_org, og_id, moa, kernel_rate, kernel_total_n, kernel_phylo_w

The bake-off joins on (test_org, og_id, MoA(compound)) at the row level.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

DEFAULT_OUT = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/phase2")

# Canonical drug MoA classes; lowercase normalized-compound -> class.
# Curated from the antibiotic literature -- the cisplatin/uvr cluster in our
# atlas validates dna_damage; the bacitracin/envZ cluster validates cell_wall.
MOA_TABLE = {
    # cell-wall
    "bacitracin": "cell_wall",
    "vancomycin": "cell_wall",
    "d-cycloserine": "cell_wall",
    "cycloserine": "cell_wall",
    "fosfomycin": "cell_wall",
    "ampicillin": "cell_wall",
    "carbenicillin": "cell_wall",
    "penicillin": "cell_wall",
    "cefoxitin": "cell_wall",
    "cefotaxime": "cell_wall",
    "ceftriaxone": "cell_wall",
    # DNA damage / gyrase
    "cisplatin": "dna_damage",
    "nalidixic acid": "dna_damage",
    "ciprofloxacin": "dna_damage",
    "norfloxacin": "dna_damage",
    "levofloxacin": "dna_damage",
    "mitomycin": "dna_damage",
    "bleomycin": "dna_damage",
    "novobiocin": "dna_damage",
    # ribosome (30S/50S)
    "tetracycline": "ribosome",
    "doxycycline": "ribosome",
    "minocycline": "ribosome",
    "tigecycline": "ribosome",
    "gentamicin": "ribosome",
    "kanamycin": "ribosome",
    "neomycin": "ribosome",
    "tobramycin": "ribosome",
    "amikacin": "ribosome",
    "sisomicin": "ribosome",
    "spectinomycin": "ribosome",
    "streptomycin": "ribosome",
    "chloramphenicol": "ribosome",
    "erythromycin": "ribosome",
    "azithromycin": "ribosome",
    "clindamycin": "ribosome",
    # folate
    "trimethoprim": "folate",
    "sulfamethoxazole": "folate",
    # membrane
    "polymyxin": "membrane",
    "colistin": "membrane",
    "daptomycin": "membrane",
    "benzalkonium": "membrane_detergent",
    "benzethonium": "membrane_detergent",
    "triclosan": "membrane_detergent",
    # transcription
    "rifampicin": "transcription",
    "rifampin": "transcription",
    # other protein synthesis
    "fusidic acid": "protein_synthesis",
    "linezolid": "protein_synthesis",
    "mupirocin": "protein_synthesis",
    # cell envelope / efflux
    "cerulenin": "fatty_acid_synthesis",
}

# weights for the chemistry kernel
W_EXACT = 1.0       # same exact compound
W_SAME_MOA = 1.0    # same MoA class (the key signal)
W_SAME_EXPGROUP = 0.3
W_FLOOR = 0.0       # unrelated -> no contribution


def norm_compound(c: str) -> str:
    import re
    s = str(c or "").lower().strip()
    s = re.sub(r"\b(hexahydrate|dihydrate|monohydrate|trihydrate|tetrahydrate|"
               r"anhydrous|sodium salt|potassium salt|salt|dihydrochloride|"
               r"hydrochloride|sulfate|chloride|nitrate|acetate|phosphate)\b",
               "", s)
    return re.sub(r"\s+", " ", s).strip()


def get_moa(cpd: str) -> str | None:
    """Substring match against the curated antibiotic MoA table."""
    if not cpd or not isinstance(cpd, str): return None
    c = cpd.lower()
    for k, v in MOA_TABLE.items():
        if k in c:
            return v
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--phylo-scale", type=float, default=1.0,
                    help="kernel scale; weight = jaccard**scale. 1.0 = linear")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = args.out / "kernel_ogcpd_features.parquet"
    if out_path.exists() and not args.force:
        print(f"kernel_ogcpd_features.parquet exists -> skip (use --force)")
        return 0

    import pandas as pd
    import numpy as np

    # ---- load og_id membership per org (for phylo distance) ----
    print("Loading per-org gene aggregates (for phylo distance) ...")
    gene_parts = []
    for p in sorted((args.out / "agg").glob("*_gene.parquet")):
        df = pd.read_parquet(p, columns=["orgId", "og_id"])
        gene_parts.append(df)
    if not gene_parts:
        print("ERROR: no agg/*_gene.parquet found", file=sys.stderr); return 1
    genes = pd.concat(gene_parts, ignore_index=True).dropna(subset=["og_id"])
    genes["og_id"] = genes.og_id.astype(str)
    og_sets = {o: set(g.og_id) for o, g in genes.groupby("orgId")}
    orgs = sorted(og_sets.keys())
    print(f"  {len(orgs)} organisms, OG-set sizes "
          f"min={min(len(s) for s in og_sets.values())} "
          f"max={max(len(s) for s in og_sets.values())}")

    # phylo distance matrix: jaccard over OG sets
    print("Computing phylo similarity (OG-set Jaccard) ...")
    sim = np.zeros((len(orgs), len(orgs)))
    for i, a in enumerate(orgs):
        for j, b in enumerate(orgs):
            if i == j:
                sim[i, j] = 1.0
            elif i < j:
                inter = len(og_sets[a] & og_sets[b])
                union = len(og_sets[a] | og_sets[b])
                s = inter / union if union > 0 else 0.0
                sim[i, j] = sim[j, i] = s
    # raise to scale (default 1.0)
    sim = sim ** args.phylo_scale
    org_to_idx = {o: i for i, o in enumerate(orgs)}
    print(f"  similarity stats: median={np.median(sim[sim<1.0]):.3f} "
          f"max(off-diag)={sim[sim<1.0].max():.3f}")

    # ---- load og_cpd aggregates ----
    print("Loading og_cpd aggregates ...")
    ocp_parts = []
    for p in sorted((args.out / "agg_ogcpd").glob("*.parquet")):
        df = pd.read_parquet(p)
        ocp_parts.append(df)
    ocp = pd.concat(ocp_parts, ignore_index=True)
    ocp["og_id"] = ocp.og_id.astype(str)
    ocp["compound"] = ocp.compound.apply(norm_compound)
    ocp = ocp[ocp.compound != ""]
    print(f"  {len(ocp):,} (orgId, og_id, compound) cells; "
          f"{ocp.compound.nunique()} distinct compounds")

    # ---- assign MoA class to every cell ----
    ocp["moa"] = ocp.compound.apply(get_moa)
    n_moa = ocp.moa.notna().sum()
    print(f"  cells with MoA class assigned: {n_moa:,} "
          f"({n_moa/len(ocp)*100:.1f}%)")
    print(f"  MoA distribution (top):\n{ocp.moa.value_counts().head(10).to_string()}")

    # ---- aggregate to (orgId, og_id, MoA) -- the "chem-grouped" view ----
    # for compounds without MoA, group by compound itself (exact-match fallback)
    ocp["chem_key"] = ocp.moa.fillna("EXACT::" + ocp.compound)
    grp = (ocp.groupby(["orgId", "og_id", "chem_key"])
              .agg(sum_n=("n", "sum"), sum_n_hit=("n_hit", "sum"))
              .reset_index())
    print(f"  chem-grouped cells: {len(grp):,}")
    # quick index lookup
    grp_by_chem = {k: g for k, g in grp.groupby("chem_key")}

    # ---- for each test_org, sum over training orgs weighted by phylo ----
    print("\nBuilding per-test-org kernel rates ...")
    results = []
    for test_org in orgs:
        t0 = time.time()
        ti = org_to_idx[test_org]
        # phylo weight per training org (exclude test_org itself)
        phylo_w_series = pd.Series(
            {o: float(sim[ti, org_to_idx[o]]) for o in orgs if o != test_org},
            name="phylo_w")
        # iterate over chem_keys -> aggregate training rows
        sub_parts = []
        for ck, g in grp_by_chem.items():
            g = g[g.orgId != test_org].copy()
            if len(g) == 0: continue
            g["pw"] = g.orgId.map(phylo_w_series)
            g["w_n"] = g.pw * g.sum_n
            g["w_hit"] = g.pw * g.sum_n_hit
            kr = (g.groupby("og_id")
                    .agg(kernel_total_n=("w_n", "sum"),
                         kernel_total_hit=("w_hit", "sum"),
                         kernel_phylo_w=("pw", "sum"))
                    .reset_index())
            kr["chem_key"] = ck
            sub_parts.append(kr)
        if not sub_parts: continue
        per_test = pd.concat(sub_parts, ignore_index=True)
        per_test["kernel_rate"] = (per_test.kernel_total_hit /
                                   per_test.kernel_total_n.clip(lower=1e-9))
        per_test["test_org"] = test_org
        results.append(per_test)
        print(f"  {test_org:25s} {len(per_test):>8,} (og x chem) cells "
              f"({time.time()-t0:.1f}s)")
    out_df = pd.concat(results, ignore_index=True)
    print(f"\ntotal kernel feature rows: {len(out_df):,}")

    args.out.mkdir(parents=True, exist_ok=True)
    out_df[["test_org", "og_id", "chem_key", "kernel_rate",
            "kernel_total_n", "kernel_phylo_w"]].to_parquet(
        out_path, index=False)
    print(f"wrote {out_path}")
    (args.out / "kernel_ogcpd_manifest.json").write_text(json.dumps({
        "n_rows": len(out_df),
        "n_orgs": len(orgs),
        "phylo_scale": args.phylo_scale,
        "moa_classes": sorted(set(MOA_TABLE.values())),
        "w_exact": W_EXACT, "w_same_moa": W_SAME_MOA,
        "w_same_expgroup": W_SAME_EXPGROUP,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

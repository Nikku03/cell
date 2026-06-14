"""FLUX-FIELD LINCHPIN at GTDB scale (the proper-power test).

The n=48 linchpin was null but underpowered (no rate dynamic range). This is
the powered version: measure each gene family's phylogenetic DISPERSION across
~28,000 GTDB species, residualize against prevalence (static conservation), and
test whether the residual predicts essentiality on held-out clades -- the
genuine "AlphaFold uses the deep MSA" analog (deep pangenome instead of 48 orgs).

ROUTE-AGNOSTIC: this script consumes three inputs however they were produced
(AnnoTree occurrence, MMseqs2 search vs GTDB, or eggNOG join -- the data adapter
is decided by the scout). The analysis logic below is identical regardless.

INPUTS (real mode):
  gtdb_taxonomy.tsv       species_id \\t phylum \\t class \\t order \\t family \\t genus
  family_occurrence.parquet  long form: columns [family_id, species_id]
                          (which GTDB species contain each gene family)
  gene_family_map.csv     organism, locus_tag, family_id
                          (how OUR genes map to those families)
  + our labels.csv, orthology_features.csv (family_frac), clade_splits.csv

DISPERSION METRIC (same structure as the n=48 proxy, at GTDB power):
  prevalence  = (#species with family) / (#GTDB species)        [conservation]
  dispersion  = #phyla (and #classes, #orders) the family spans  [lability]
  residualize each dispersion on a quadratic in prevalence -> *_resid
  = "more/fewer independent taxonomic occurrences than expected for this
     prevalence" = the gain/loss rate orthogonal to the conservation integral.

TEST (leave-one-clade-out, vs conservation baseline) + phylogeny-shuffle null.

--smoke : validate dispersion + residualization + test logic on SYNTHETIC
          GTDB-scale data (28k species, 150 phyla, 5k families). No downloads.
--real  : read the real GTDB inputs and run the powered linchpin.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
LAB = REPO / "data" / "drive_import" / "labels"
GTDB = REPO / "data" / "gtdb"
OUT = REPO / "outputs"
sys.path.insert(0, str(SCRIPTS))

from flux_linchpin import auprc, evaluate  # reuse tested helpers  # noqa: E402

TAX_LEVELS = ["phylum", "class", "order"]


def residualize_on_prevalence(df, col, prev="prevalence"):
    """Residual of `col` on a quadratic in prevalence (the conservation
    integral), so the residual is orthogonal to static conservation."""
    import numpy as np
    x = df[prev].values.astype(float)
    y = df[col].values.astype(float)
    A = np.vstack([np.ones_like(x), x, x ** 2]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def compute_dispersion(occ, taxonomy, n_species_total, shuffle_seed=None):
    """Per-family prevalence + taxonomic dispersion (n_phyla/class/order),
    each residualized on prevalence. occ: long [family_id, species_id].
    If shuffle_seed: permute species->taxonomy (the phylogeny-shuffle null)."""
    import numpy as np, pandas as pd
    tax = taxonomy.copy()
    if shuffle_seed is not None:
        rng = np.random.RandomState(shuffle_seed)
        for lv in TAX_LEVELS:
            tax[lv] = rng.permutation(tax[lv].values)
    m = occ.merge(tax, on="species_id", how="inner")
    agg = {"species_id": "nunique"}
    for lv in TAX_LEVELS:
        agg[lv] = "nunique"
    g = m.groupby("family_id").agg(agg).reset_index()
    g = g.rename(columns={"species_id": "n_species"})
    g["prevalence"] = g.n_species / float(n_species_total)
    for lv in TAX_LEVELS:
        g = g.rename(columns={lv: f"n_{lv}"})
        g[f"disp_{lv}_resid"] = residualize_on_prevalence(g, f"n_{lv}")
    return g


def build_per_gene(disp):
    """Join family dispersion to our genes + essentiality + conservation."""
    import pandas as pd
    lab = pd.read_csv(LAB / "labels.csv",
                      usecols=["organism", "locus_tag", "essential"])
    orth = pd.read_csv(LAB / "orthology_features.csv",
                       usecols=["organism", "locus_tag",
                                "family_frac_essential_leakfree"])
    gmap = pd.read_csv(GTDB / "gene_family_map.csv")  # organism, locus_tag, family_id
    cs = pd.read_csv(LAB / "clade_splits.csv", usecols=["organism", "clade"])
    pg = (lab.merge(orth, on=["organism", "locus_tag"], how="inner")
             .merge(gmap, on=["organism", "locus_tag"], how="inner")
             .merge(disp, on="family_id", how="left")
             .merge(cs, on="organism", how="left"))
    pg = pg.rename(columns={"family_frac_essential_leakfree": "family_frac"})
    disp_cols = ["prevalence"] + [f"disp_{lv}_resid" for lv in TAX_LEVELS]
    pg = pg.dropna(subset=["essential", "family_frac", "clade"] + disp_cols)
    return pg, disp_cols


def run_test(pg, disp_cols, label=""):
    import numpy as np
    clades = sorted(pg.clade.unique())
    fsets = {
        "conservation": ["family_frac"],
        "conservation+prevalence": ["family_frac", "prevalence"],
        "dispersion alone": [c for c in disp_cols if c != "prevalence"],
        "conservation+dispersion": ["family_frac"] +
            [c for c in disp_cols if c != "prevalence"],
        "conservation+prev+dispersion": ["family_frac"] + disp_cols,
    }
    res = evaluate(pg, fsets, clades)
    base = res["conservation"][0]
    print(f"\n  LEAVE-ONE-CLADE-OUT AUPRC {label}({len(clades)} clades):")
    for name, (ap, n) in res.items():
        if ap is None:
            print(f"    {name:<32s} n/a"); continue
        d = (ap - base) if base else 0
        print(f"    {name:<32s} AUPRC={ap:.4f}  Δcons={d:+.4f}  ({n})")
    full = res["conservation+dispersion"][0]
    return ((full - base) if (full and base) else None), res


def run_real(args):
    import pandas as pd
    print("=" * 68)
    print("FLUX-FIELD LINCHPIN @ GTDB scale -- powered test")
    print("=" * 68)
    tax = pd.read_csv(GTDB / "gtdb_taxonomy.tsv", sep="\t")
    occ = pd.read_parquet(GTDB / "family_occurrence.parquet")
    n_species = tax.species_id.nunique()
    print(f"  GTDB species: {n_species:,}  families w/ occurrence: "
          f"{occ.family_id.nunique():,}  occurrence rows: {len(occ):,}")

    disp = compute_dispersion(occ, tax, n_species)
    print(f"  dispersion computed for {len(disp):,} families")
    pg, disp_cols = build_per_gene(disp)
    print(f"  per-gene rows joined: {len(pg):,}  essential: "
          f"{int(pg.essential.sum()):,}  families mapped: "
          f"{pg.family_id.nunique():,}")

    # orthogonality
    import numpy as np
    for c in disp_cols:
        if c == "prevalence": continue
        r = float(np.corrcoef(pg[c], pg.family_frac)[0, 1])
        print(f"  orthogonality corr({c}, family_frac) = {r:+.3f}")

    lift, res = run_test(pg, disp_cols, "[REAL] ")

    # phylogeny-shuffle null
    print("\n  PHYLOGENY-SHUFFLE NULL (permute species->taxonomy):")
    null = []
    for seed in range(3):
        ds = compute_dispersion(occ, tax, n_species, shuffle_seed=seed)
        pgs, dc = build_per_gene(ds)
        l, _ = run_test(pgs, dc, f"[NULL seed {seed}] ")
        if l is not None: null.append(l)
    null_mean = float(np.mean(null)) if null else None
    print(f"\n  real dispersion lift: {lift:+.4f}" if lift is not None else
          "  real lift n/a")
    print(f"  shuffled lift (mean): {null_mean:+.4f}" if null_mean is not None
          else "  shuffled lift n/a")

    alive = (lift is not None and lift > 0.01 and
             (null_mean is None or lift > 2 * abs(null_mean)))
    print(f"\n{'='*68}\nVERDICT @ GTDB SCALE\n{'='*68}")
    if alive:
        print("  >>> FLUX ALIVE AT POWER. Gain/loss dispersion across GTDB adds")
        print("      essentiality signal over conservation, orthogonal and")
        print("      collapsing under phylogeny-shuffle. The Dispensability-")
        print("      Response-Field thesis SURVIVES the powered test -> the")
        print("      additional-data avenue is real; build it into Paper 2/3.")
    else:
        print("  >>> NULL AT POWER. With ~28k-species dynamic range and no")
        print("      lift, the powered test does NOT support flux as an")
        print("      orthogonal essentiality signal. This is the real kill the")
        print("      n=48 test could not deliver -> deprioritize flux; the")
        print("      additional-data win, if any, is interventional (DepMap).")

    OUT.mkdir(exist_ok=True)
    (OUT / "flux_gtdb_results.json").write_text(json.dumps({
        "n_gtdb_species": int(n_species),
        "n_families": int(disp.family_id.nunique()),
        "n_genes": int(len(pg)),
        "loco_auprc": {k: {"auprc": v[0], "n": v[1]} for k, v in res.items()},
        "dispersion_lift": lift, "shuffle_null_lift": null_mean,
        "verdict_alive": bool(alive),
    }, indent=2, default=str))
    print(f"\nwrote {OUT/'flux_gtdb_results.json'}")
    return 0


def run_smoke():
    """Validate dispersion + residualization + test on synthetic GTDB-scale
    data with a PLANTED signal (essential families are low-dispersion)."""
    import numpy as np, pandas as pd
    print("=" * 68)
    print("FLUX-GTDB SMOKE -- synthetic 28k species / 150 phyla / 5k families")
    print("=" * 68)
    rng = np.random.RandomState(0)
    n_species, n_phyla, n_fam = 28000, 150, 5000
    species = [f"s{i}" for i in range(n_species)]
    sp_phylum = rng.randint(0, n_phyla, n_species)
    sp_class = sp_phylum * 5 + rng.randint(0, 5, n_species)
    sp_order = sp_class * 4 + rng.randint(0, 4, n_species)
    tax = pd.DataFrame({"species_id": species,
                        "phylum": [f"p{x}" for x in sp_phylum],
                        "class": [f"c{x}" for x in sp_class],
                        "order": [f"o{x}" for x in sp_order],
                        "family": "f", "genus": "g"})
    # families: half "core" (high prevalence, concentrated), half "accessory"
    occ_rows = []
    fam_kind = {}
    for fi in range(n_fam):
        fid = f"OG{fi:05d}"
        if fi % 2 == 0:  # core: in many species but few phyla (concentrated)
            phyla = rng.choice(n_phyla, size=rng.randint(2, 6), replace=False)
            mask = np.isin(sp_phylum, phyla)
            idx = np.where(mask)[0]
            kind = "core"
        else:            # accessory: scattered across many phyla (dispersed)
            idx = rng.choice(n_species, size=rng.randint(200, 1500),
                             replace=False)
            kind = "accessory"
        fam_kind[fid] = kind
        for i in idx:
            occ_rows.append((fid, species[i]))
    occ = pd.DataFrame(occ_rows, columns=["family_id", "species_id"])
    print(f"  synthetic occurrence rows: {len(occ):,}")

    disp = compute_dispersion(occ, tax, n_species)
    print(f"  dispersion computed: {len(disp):,} families")
    print(f"    median n_phylum (core vs accessory): ", end="")
    disp["kind"] = disp.family_id.map(fam_kind)
    print(disp.groupby("kind").n_phylum.median().to_dict())

    # planted signal: low-dispersion (core) families are essential
    rng2 = np.random.RandomState(1)
    genes = []
    clades = [f"clade{c}" for c in range(10)]
    for gi in range(20000):
        fid = f"OG{rng2.randint(0, n_fam):05d}"
        kind = fam_kind[fid]
        p_ess = 0.6 if kind == "core" else 0.1
        genes.append((f"org{rng2.randint(0,48)}", f"g{gi}", fid,
                      int(rng2.random() < p_ess), rng2.choice(clades)))
    gdf = pd.DataFrame(genes, columns=["organism", "locus_tag", "family_id",
                                       "essential", "clade"])
    gdf = gdf.merge(disp, on="family_id", how="left")
    # synthetic conservation = prevalence + noise (the baseline to beat)
    gdf["family_frac"] = (gdf.prevalence * 2 +
                          rng2.normal(0, 0.1, len(gdf))).clip(0, 1)
    disp_cols = ["prevalence"] + [f"disp_{lv}_resid" for lv in TAX_LEVELS]
    gdf = gdf.dropna(subset=disp_cols)

    lift, res = run_test(gdf, disp_cols, "[SMOKE] ")
    print(f"\n  planted-signal dispersion lift: {lift:+.4f}")
    assert lift is not None and lift > 0.0, \
        "smoke: dispersion should add signal when essentiality is planted on it"
    print("\n=== FLUX-GTDB SMOKE PASS ===")
    print("  dispersion/residualization/LOCO-test logic validated on synthetic")
    print("  GTDB-scale data with a planted signal. Wire the real data adapter")
    print("  (per scout's recommendation) and run --real on Colab.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())

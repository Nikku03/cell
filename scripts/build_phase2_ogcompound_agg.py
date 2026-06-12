"""Phase 2: per-(organism, OG, compound) hit aggregates -> the conditional
analog of family_frac.

WHY: option B showed gene-function (domain) features lift the general task
but NOT the bacitracin litmus, because envZ/ompR -> bacitracin is a Shewanella
-specific circuit, not a transferable "histidine kinase -> bacitracin" rule.
The only leak-free feature that can carry it is cross-organism reproducibility
of the SPECIFIC (ortholog-group x compound) pair: "fraction of this gene's
orthologs that are strong hits under THIS compound, in OTHER organisms."

That is the conditional analog of family_frac_essential (the dominant phase-0
feature). This script produces the sufficient statistics for it:

  agg_ogcpd/<org>.parquet : (orgId, og_id, compound, n, n_hit)

computed from the FULL GeneFitness (not the downsampled shards), per organism,
resumably. The bake-off then computes, per held-out fold T:
  og_cpd_hit_rate(og, c) = sum_{o != T} n_hit / sum_{o != T} n
and maps it onto rows by (og_id, compound) -- leak-free for T.

PERFORMANCE: copy feba.db to local disk first (--db /content/feba.db).
"""
from __future__ import annotations
import argparse, re, sqlite3, sys, time
from pathlib import Path

DEFAULT_DB = Path("/content/feba.db")
DEFAULT_OUT = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/phase2")
HIT_FIT = -3.0
HIT_T = 3.0


def norm_compound(c: str) -> str:
    s = str(c or "").lower().strip()
    s = re.sub(r"\b(hexahydrate|dihydrate|monohydrate|trihydrate|tetrahydrate|"
               r"anhydrous|sodium salt|potassium salt|salt|dihydrochloride|"
               r"hydrochloride|sulfate|chloride|nitrate|acetate|phosphate)\b",
               "", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--hit-fit", type=float, default=HIT_FIT)
    ap.add_argument("--hit-t", type=float, default=HIT_T)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.db.exists():
        alt = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/"
                   "fitness_browser/feba.db")
        if alt.exists(): args.db = alt
        else:
            print(f"ERROR: feba.db not found ({args.db})", file=sys.stderr); return 1

    import pandas as pd
    out_dir = args.out / "agg_ogcpd"
    out_dir.mkdir(parents=True, exist_ok=True)

    # locus -> og_id map comes from the gene_agg files already on Drive
    agg_dir = args.out / "agg"
    gene_aggs = sorted(agg_dir.glob("*_gene.parquet"))
    if not gene_aggs:
        print(f"ERROR: no gene_agg files in {agg_dir}; run frame builder first",
              file=sys.stderr); return 1

    con = sqlite3.connect(str(args.db))
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")

    # experiment -> compound map
    exps = pd.read_sql("SELECT orgId, expName, condition_1 FROM Experiment", con)
    exps["compound"] = exps.condition_1.apply(norm_compound)

    orgs = [p.name.replace("_gene.parquet", "") for p in gene_aggs]
    print(f"organisms: {len(orgs)}")
    built = skipped = 0
    for org in orgs:
        out_path = out_dir / f"{org}.parquet"
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        t0 = time.time()
        # locus -> og_id for this org
        ga = pd.read_parquet(agg_dir / f"{org}_gene.parquet",
                             columns=["locusId", "og_id"])
        ga["locusId"] = ga.locusId.astype(str)
        og_map = ga.dropna(subset=["og_id"]).set_index("locusId").og_id

        gf = pd.read_sql("SELECT locusId, expName, fit, t FROM GeneFitness "
                         "WHERE orgId=?", con, params=(org,))
        if gf.empty:
            print(f"  {org:25s} no GeneFitness; skip"); continue
        gf["locusId"] = gf.locusId.astype(str)
        gf["og_id"] = gf.locusId.map(og_map)
        e = exps[exps.orgId == org][["expName", "compound"]]
        gf = gf.merge(e, on="expName", how="left")
        gf = gf[gf.og_id.notna() & (gf.compound != "")]
        gf["strong_hit"] = ((gf.fit < args.hit_fit) &
                            (gf.t.abs() >= args.hit_t)).astype(int)
        agg = gf.groupby(["og_id", "compound"]).agg(
            n=("fit", "size"), n_hit=("strong_hit", "sum")).reset_index()
        agg.insert(0, "orgId", org)
        agg.to_parquet(out_path, index=False)
        built += 1
        print(f"  {org:25s} {len(agg):>7,} (og,cpd) cells  "
              f"({time.time()-t0:.0f}s)")
    con.close()
    print(f"\nbuilt {built}, skipped {skipped} -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

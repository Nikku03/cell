"""Phase 2 frame builder (memory-safe, sharded) — assemble the (gene, condition)
training table without ever holding the whole frame in RAM.

Per PHASE2_DESIGN.md (feasibility-corrected): target is the REPRODUCIBLE
signal, strong conditional vulnerability (binary), NOT the continuous fit
(tail replicate corr only ~0.34). Continuous fit kept as a secondary column.

DESIGN (why this version exists): the naive "concat all ~30M rows then write"
OOM-killed Colab. This version processes ONE organism at a time and writes a
parquet shard per organism, holding at most one organism (~<=1.8M rows) in
memory. Two further moves keep it small and correct:

  1. DOWNSAMPLE the neutral mass per organism. strong_hit is ~1.5% of rows;
     we keep ALL positives + a capped multiple (--neg-ratio, default 10) of
     randomly sampled negatives. ~30M -> ~5M training rows. This is also what
     the bake-off wants for imbalanced classification.

  2. Store FULL-DATA sufficient statistics separately so downsampling does NOT
     bias the leak-free additive main-effects the bake-off computes per fold:
        gene_agg : (orgId, locusId, og_id, sum_fit, n, n_hit)
        cpd_agg  : (orgId, compound, sum_fit, n, n_hit)
     The bake-off derives leak-free gene-mean-fit / compound-mean-fit by
     combining these across folds (subtract the held-out group's stats),
     never touching raw rows.

Outputs:
  outputs/phase2_frame/<org>.parquet   downsampled training rows (shards)
  outputs/phase2_gene_agg.parquet      full-data per-(org,gene) sufficient stats
  outputs/phase2_cpd_agg.parquet       full-data per-(org,compound) suff. stats
  + stats to stdout

PERFORMANCE: copy feba.db to local disk first (--db /content/feba.db).
"""
from __future__ import annotations
import argparse, re, sqlite3, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DB = Path("/content/feba.db")
DEFAULT_FEATS = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/"
                     "orthology_features.csv")

HIT_FIT = -3.0
HIT_T   = 3.0


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
    ap.add_argument("--feats", type=Path, default=DEFAULT_FEATS)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs")
    ap.add_argument("--hit-fit", type=float, default=HIT_FIT)
    ap.add_argument("--hit-t", type=float, default=HIT_T)
    ap.add_argument("--neg-ratio", type=float, default=10.0,
                    help="negatives kept per positive, per organism")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.db.exists():
        alt = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/"
                   "fitness_browser/feba.db")
        if alt.exists(): args.db = alt
        else:
            print(f"ERROR: feba.db not found ({args.db} / {alt})", file=sys.stderr)
            return 1
    if not args.feats.exists():
        print(f"ERROR: gene features not found at {args.feats}", file=sys.stderr)
        return 1

    import pandas as pd
    import numpy as np
    rng = np.random.RandomState(args.seed)

    # ---- gene features ----
    print(f"Loading gene features {args.feats} ...")
    feats = pd.read_csv(args.feats)
    feats["fb_org"] = feats.organism.str.replace("^beril_", "", regex=True)
    feats["locusId"] = feats.locus_tag.astype(str)
    ff_cols = [c for c in feats.columns
               if c.startswith("family_frac_essential_fold")]
    gene_feat_cols = ff_cols + ["family_n_organisms", "n_paralogs_in_genome",
                                "is_orphan", "og_id"]
    gene_feat_cols = [c for c in gene_feat_cols if c in feats.columns]
    feats_small = feats[["fb_org", "locusId"] + gene_feat_cols].copy()
    if "og_id" in feats_small.columns:   # str for matched, <NA> for unmatched
        feats_small["og_id"] = feats_small["og_id"].astype("string")
    orgs_with_feats = set(feats_small.fb_org.unique())
    print(f"  features for {len(orgs_with_feats)} organisms; cols={gene_feat_cols}")

    con = sqlite3.connect(str(args.db))
    print("Loading Experiment + Compounds ...")
    exps = pd.read_sql(
        "SELECT orgId, expName, expGroup, condition_1, units_1, "
        "concentration_1, media, pH, temperature, aerobic FROM Experiment", con)
    exps["compound"] = exps.condition_1.apply(norm_compound)
    # FB stores blanks as '' not NaN -> coerce numeric cols or pyarrow chokes
    for c in ["concentration_1", "pH", "temperature"]:
        exps[c] = pd.to_numeric(exps[c], errors="coerce")
    comp = pd.read_sql("SELECT compound, MW, CAS FROM Compounds", con)
    comp["_n"] = comp.compound.apply(norm_compound)
    comp["MW"] = pd.to_numeric(comp["MW"], errors="coerce")
    mw_map = comp.dropna(subset=["MW"]).groupby("_n").MW.first()
    cas_map = comp[comp.CAS.astype(str).str.strip() != ""].groupby("_n").CAS.first()
    exps["MW"] = exps.compound.map(mw_map)
    exps["CAS"] = exps.compound.map(cas_map).astype("string")

    fb_orgs = sorted(orgs_with_feats & set(exps.orgId.unique()))
    print(f"  organisms in BOTH fitness + features: {len(fb_orgs)}\n  {fb_orgs}")

    shard_dir = args.out_dir / "phase2_frame"
    shard_dir.mkdir(parents=True, exist_ok=True)
    # clear stale shards
    for old in shard_dir.glob("*.parquet"):
        old.unlink()

    cond_cols = ["expGroup", "compound", "MW", "CAS", "concentration_1",
                 "pH", "temperature", "aerobic", "media"]
    keep_cols = (["orgId", "locusId", "expName", "fit", "t", "strong_hit"]
                 + cond_cols + gene_feat_cols)

    gene_aggs, cpd_aggs = [], []
    tot_rows = tot_pos = tot_kept = 0
    per_org_stats = []

    for org in fb_orgs:
        e = exps[exps.orgId == org]
        if e.empty: continue
        gf = pd.read_sql(
            "SELECT orgId, locusId, expName, fit, t FROM GeneFitness "
            "WHERE orgId=?", con, params=(org,))
        if gf.empty: continue
        gf["locusId"] = gf.locusId.astype(str)
        gf = gf.merge(e[["expName"] + cond_cols], on="expName", how="left")
        gfe = feats_small[feats_small.fb_org == org].drop(columns=["fb_org"])
        gf = gf.merge(gfe, on="locusId", how="left")
        gf["strong_hit"] = ((gf.fit < args.hit_fit) &
                            (gf.t.abs() >= args.hit_t)).astype(int)

        n = len(gf); npos = int(gf.strong_hit.sum())
        tot_rows += n; tot_pos += npos

        # FULL-DATA sufficient statistics (before downsampling)
        ga = gf.groupby("locusId").agg(
            og_id=("og_id", "first"), sum_fit=("fit", "sum"),
            n=("fit", "size"), n_hit=("strong_hit", "sum")).reset_index()
        ga.insert(0, "orgId", org); gene_aggs.append(ga)
        ca = gf.groupby("compound").agg(
            sum_fit=("fit", "sum"), n=("fit", "size"),
            n_hit=("strong_hit", "sum")).reset_index()
        ca.insert(0, "orgId", org); cpd_aggs.append(ca)

        # DOWNSAMPLE negatives
        pos = gf[gf.strong_hit == 1]
        neg = gf[gf.strong_hit == 0]
        keep_neg = min(len(neg), int(args.neg_ratio * max(npos, 1)))
        if keep_neg < len(neg):
            neg = neg.iloc[rng.choice(len(neg), keep_neg, replace=False)]
        shard = pd.concat([pos, neg], ignore_index=True)[keep_cols]
        shard.to_parquet(shard_dir / f"{org}.parquet", index=False)
        tot_kept += len(shard)
        per_org_stats.append((org, n, npos, len(shard)))
        print(f"  {org:8s} rows={n:>8,} pos={npos:>6,} "
              f"kept={len(shard):>8,}  feat={gf.og_id.notna().mean()*100:3.0f}%")
        del gf, pos, neg, shard

    con.close()

    pd.concat(gene_aggs, ignore_index=True).to_parquet(
        args.out_dir / "phase2_gene_agg.parquet", index=False)
    pd.concat(cpd_aggs, ignore_index=True).to_parquet(
        args.out_dir / "phase2_cpd_agg.parquet", index=False)

    print(f"\n=== SUMMARY ===")
    print(f"  organisms: {len(per_org_stats)}")
    print(f"  full rows: {tot_rows:,}   positives: {tot_pos:,} "
          f"(base rate {tot_pos/max(tot_rows,1):.4f})")
    print(f"  kept (downsampled) rows: {tot_kept:,} "
          f"(neg-ratio {args.neg_ratio})")
    print(f"  shards -> {shard_dir}/<org>.parquet")
    print(f"  gene_agg -> phase2_gene_agg.parquet  "
          f"cpd_agg -> phase2_cpd_agg.parquet")
    # litmus orgs present?
    present = {s[0] for s in per_org_stats}
    for o in ("MR1", "PV4", "SB2B"):
        print(f"  litmus {o}: {'present' if o in present else 'MISSING'}")
    print("\nNext: scripts/train_phase2_bakeoff.py reads the shard dir + aggs, "
          "computes per-fold leak-free main-effects, runs the bake-off.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

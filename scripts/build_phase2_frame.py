"""Phase 2 frame builder (memory-safe, sharded, RESUMABLE to Drive).

Per PHASE2_DESIGN.md (feasibility-corrected): target is the REPRODUCIBLE
signal, strong conditional vulnerability (binary), NOT the continuous fit
(tail replicate corr only ~0.34). Continuous fit kept as a secondary column.

DESIGN: processes ONE organism at a time (<=1.8M rows in RAM) and writes a
parquet shard per organism directly to the output dir (defaults to Drive) so
a disconnect costs only the partial organism. Three guarantees:

  1. DOWNSAMPLE the neutral mass per organism. strong_hit is ~1.5% of rows;
     keep ALL positives + a capped multiple (--neg-ratio, default 10) of
     randomly sampled negatives. ~30M -> ~5M training rows.

  2. PER-ORG full-data sufficient statistics (agg/<org>_gene.parquet,
     agg/<org>_cpd.parquet) stored alongside, so downsampling does NOT bias
     the leak-free additive main-effects the bake-off computes per fold.

  3. RESUMABLE. If frame/<org>.parquet AND agg/<org>_gene.parquet AND
     agg/<org>_cpd.parquet all exist, the organism is skipped. --force
     recomputes.

Output layout (defaults to Drive PHASE2_DIR):
  <out>/frame/<org>.parquet         downsampled training rows
  <out>/agg/<org>_gene.parquet      full-data per-(org,gene) suff stats
  <out>/agg/<org>_cpd.parquet       full-data per-(org,compound) suff stats
  <out>/_build_manifest.json        organisms successfully built + base rate

PERFORMANCE: copy feba.db to local disk first (--db /content/feba.db).
"""
from __future__ import annotations
import argparse, json, re, sqlite3, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DB = Path("/content/feba.db")
DEFAULT_FEATS = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/"
                     "orthology_features.csv")
DEFAULT_OUT = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/phase2")

HIT_FIT = -3.0
HIT_T   = 3.0


def norm_compound(c: str) -> str:
    s = str(c or "").lower().strip()
    s = re.sub(r"\b(hexahydrate|dihydrate|monohydrate|trihydrate|tetrahydrate|"
               r"anhydrous|sodium salt|potassium salt|salt|dihydrochloride|"
               r"hydrochloride|sulfate|chloride|nitrate|acetate|phosphate)\b",
               "", s)
    return re.sub(r"\s+", " ", s).strip()


def all_outputs_present(out: Path, org: str) -> bool:
    return ((out / "frame" / f"{org}.parquet").exists() and
            (out / "agg" / f"{org}_gene.parquet").exists() and
            (out / "agg" / f"{org}_cpd.parquet").exists())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--feats", type=Path, default=DEFAULT_FEATS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output root (default: Drive phase2 dir)")
    ap.add_argument("--hit-fit", type=float, default=HIT_FIT)
    ap.add_argument("--hit-t", type=float, default=HIT_T)
    ap.add_argument("--neg-ratio", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="recompute even if shards already exist")
    ap.add_argument("--only", nargs="*", default=None,
                    help="only build these orgIds (subset)")
    args = ap.parse_args()

    if not args.db.exists():
        alt = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/"
                   "fitness_browser/feba.db")
        if alt.exists():
            print(f"  --db not found; falling back to Drive copy {alt}")
            print(f"  (TIP: cp it to /content/feba.db for much faster queries)")
            args.db = alt
        else:
            print(f"ERROR: feba.db not found ({args.db} / {alt})", file=sys.stderr)
            return 1
    if not args.feats.exists():
        print(f"ERROR: gene features not found at {args.feats}", file=sys.stderr)
        return 1

    import pandas as pd
    import numpy as np
    rng = np.random.RandomState(args.seed)

    (args.out / "frame").mkdir(parents=True, exist_ok=True)
    (args.out / "agg").mkdir(parents=True, exist_ok=True)

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
    if "og_id" in feats_small.columns:
        feats_small["og_id"] = feats_small["og_id"].astype("string")
    orgs_with_feats = set(feats_small.fb_org.unique())
    print(f"  features for {len(orgs_with_feats)} organisms; cols={gene_feat_cols}")

    con = sqlite3.connect(str(args.db))
    print("Loading Experiment + Compounds ...")
    exps = pd.read_sql(
        "SELECT orgId, expName, expGroup, condition_1, units_1, "
        "concentration_1, media, pH, temperature, aerobic FROM Experiment", con)
    exps["compound"] = exps.condition_1.apply(norm_compound)
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
    if args.only:
        fb_orgs = [o for o in fb_orgs if o in set(args.only)]
    print(f"  organisms in BOTH fitness + features: {len(fb_orgs)}")
    print(f"  {fb_orgs}")

    cond_cols = ["expGroup", "compound", "MW", "CAS", "concentration_1",
                 "pH", "temperature", "aerobic", "media"]
    keep_cols = (["orgId", "locusId", "expName", "fit", "t", "strong_hit"]
                 + cond_cols + gene_feat_cols)

    per_org_stats = []
    tot_rows = tot_pos = tot_kept = 0
    skipped = []

    for org in fb_orgs:
        if not args.force and all_outputs_present(args.out, org):
            shard = pd.read_parquet(args.out / "frame" / f"{org}.parquet",
                                    columns=["strong_hit"])
            ga = pd.read_parquet(args.out / "agg" / f"{org}_gene.parquet",
                                 columns=["n", "n_hit"])
            n_full = int(ga.n.sum()); npos_full = int(ga.n_hit.sum())
            tot_rows += n_full; tot_pos += npos_full; tot_kept += len(shard)
            skipped.append(org)
            per_org_stats.append((org, n_full, npos_full, len(shard)))
            print(f"  {org:25s} SKIP (exists)   rows={n_full:>8,} "
                  f"pos={npos_full:>6,} kept={len(shard):>8,}")
            continue

        e = exps[exps.orgId == org]
        if e.empty:
            print(f"  {org:25s} SKIP (no experiments)"); continue

        t0 = time.time()
        gf = pd.read_sql(
            "SELECT orgId, locusId, expName, fit, t FROM GeneFitness "
            "WHERE orgId=?", con, params=(org,))
        if gf.empty:
            print(f"  {org:25s} SKIP (no GeneFitness rows)"); continue
        gf["locusId"] = gf.locusId.astype(str)
        gf = gf.merge(e[["expName"] + cond_cols], on="expName", how="left")
        gfe = feats_small[feats_small.fb_org == org].drop(columns=["fb_org"])
        gf = gf.merge(gfe, on="locusId", how="left")
        gf["strong_hit"] = ((gf.fit < args.hit_fit) &
                            (gf.t.abs() >= args.hit_t)).astype(int)

        n = len(gf); npos = int(gf.strong_hit.sum())
        tot_rows += n; tot_pos += npos

        # full-data sufficient statistics BEFORE downsampling
        ga = gf.groupby("locusId").agg(
            og_id=("og_id", "first"), sum_fit=("fit", "sum"),
            n=("fit", "size"), n_hit=("strong_hit", "sum")).reset_index()
        ga.insert(0, "orgId", org)
        ga.to_parquet(args.out / "agg" / f"{org}_gene.parquet", index=False)

        ca = gf.groupby("compound").agg(
            sum_fit=("fit", "sum"), n=("fit", "size"),
            n_hit=("strong_hit", "sum")).reset_index()
        ca.insert(0, "orgId", org)
        ca.to_parquet(args.out / "agg" / f"{org}_cpd.parquet", index=False)

        # downsampled shard
        pos = gf[gf.strong_hit == 1]
        neg = gf[gf.strong_hit == 0]
        keep_neg = min(len(neg), int(args.neg_ratio * max(npos, 1)))
        if keep_neg < len(neg):
            neg = neg.iloc[rng.choice(len(neg), keep_neg, replace=False)]
        shard = pd.concat([pos, neg], ignore_index=True)[keep_cols]
        shard.to_parquet(args.out / "frame" / f"{org}.parquet", index=False)
        tot_kept += len(shard)
        dt = time.time() - t0
        per_org_stats.append((org, n, npos, len(shard)))
        print(f"  {org:25s} built  rows={n:>8,} pos={npos:>6,} "
              f"kept={len(shard):>8,}  "
              f"feat={gf.og_id.notna().mean()*100:3.0f}%  ({dt:.0f}s)")
        del gf, pos, neg, shard, ga, ca

    con.close()

    manifest = {
        "out": str(args.out),
        "hit_fit": args.hit_fit, "hit_t": args.hit_t,
        "neg_ratio": args.neg_ratio,
        "n_organisms": len(per_org_stats), "skipped": skipped,
        "total_rows_full": tot_rows, "total_positives_full": tot_pos,
        "total_kept_downsampled": tot_kept,
        "base_rate": tot_pos / max(tot_rows, 1),
        "per_org": [{"org": o, "n_full": n, "n_pos": p, "n_kept": k}
                    for (o, n, p, k) in per_org_stats],
        "feature_cols_gene": gene_feat_cols,
        "cond_cols": cond_cols,
    }
    with open(args.out / "_build_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(f"  organisms: {len(per_org_stats)} (skipped existing: {len(skipped)})")
    print(f"  full rows: {tot_rows:,}   positives: {tot_pos:,} "
          f"(base rate {tot_pos/max(tot_rows,1):.4f})")
    print(f"  kept (downsampled) rows: {tot_kept:,} "
          f"(neg-ratio {args.neg_ratio})")
    print(f"  outputs -> {args.out}")
    present = {s[0] for s in per_org_stats}
    for o in ("MR1", "PV4", "SB2B"):
        print(f"  litmus {o}: {'present' if o in present else 'MISSING'}")
    print(f"  manifest -> {args.out / '_build_manifest.json'}")
    print("\nNext: scripts/train_phase2_bakeoff.py (uses these shards + per-fold "
          "leak-free main-effects from the agg files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

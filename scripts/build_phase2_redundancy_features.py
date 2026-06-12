"""Phase 2 functional-redundancy features — per-gene backup-pathway count.

WHY (the genome-evolution argument): essentiality is a property of (gene x
genome's current redundancy), not the gene alone. family_frac_essential
captures cross-organism conservation but is BLIND to whether THIS organism
has a non-homologous backup doing the same job. The classic case is
non-orthologous gene displacement (NOGD): two organisms both need function
F, but use unrelated genes to do it. family_frac sees the homologs and says
"often non-essential elsewhere", missing that the alternative was lost here.

Builds per-(orgId, locusId):

  n_same_ko_in_genome     other genes in this organism with same KEGG group
                          (broad functional similarity)
  n_same_ec_in_genome     other genes with same EC number (true functional
                          redundancy; EC is function-based, not orthology)
  n_same_ec_diff_og       same EC but DIFFERENT ortholog group (the pure
                          NOGD case -- "function present, but via a different
                          gene family")
  n_same_ko_diff_og       same KO, different OG (analogous)

The key signal the bake-off will look for is the INTERACTION:
  family_frac HIGH (function needed across life)
  AND redundancy LOW (no backup here)
  -> essential here.

XGBoost should find that interaction natively if we feed the raw cols.

LEAK-FREE: all annotations are sequence-based (BestHitKEGG by BLAST against
KEGG, ECInfo by KEGG group), never reference fitness or essentiality labels.

Output: <out>/redundancy_features.parquet  (small; ~170K genes)
        + <out>/redundancy_features_manifest.json

PERFORMANCE: ~3-5 min on CPU. Needs feba.db (use Drive copy directly --
the script does sqlite reads only, no heavy join, so Drive FUSE is fine).
"""
from __future__ import annotations
import argparse, json, sqlite3, sys
from pathlib import Path

DEFAULT_DB = Path("/content/feba.db")
DEFAULT_OUT = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/phase2")


def pick_col(cols, *candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.db.exists():
        alt = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/"
                   "fitness_browser/feba.db")
        if alt.exists():
            print(f"  --db not at {args.db}; using Drive copy {alt}")
            args.db = alt
        else:
            print(f"ERROR: feba.db not found", file=sys.stderr); return 1

    out_path = args.out / "redundancy_features.parquet"
    if out_path.exists() and not args.force:
        print(f"redundancy_features.parquet already exists -> skip (use --force)")
        return 0

    import pandas as pd
    con = sqlite3.connect(str(args.db))
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    cur = con.cursor()
    tabs = {r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    # ---- BestHitKEGG: locus -> KEGG gene id  THEN KEGGMember: keggId -> kgroup ----
    # BestHitKEGG stores the best matching KEGG gene id (e.g. eco:b0001), NOT the
    # KO group directly. To get the KO (kgroup) we join through KEGGMember,
    # which is the (kgroup, keggOrg, keggId) membership table.
    if "BestHitKEGG" not in tabs:
        print("ERROR: BestHitKEGG table absent", file=sys.stderr); return 1
    bhk_cols = [r[1] for r in cur.execute(
        "PRAGMA table_info(BestHitKEGG)").fetchall()]
    print(f"BestHitKEGG cols: {bhk_cols}")
    loc = pick_col(bhk_cols, "locusId", "locus")
    org = pick_col(bhk_cols, "orgId", "orgid")
    kid = pick_col(bhk_cols, "keggId", "keggid")
    korg = pick_col(bhk_cols, "keggOrg", "keggorg")
    print(f"  using org={org} locus={loc} keggOrg={korg} keggId={kid}")
    bhk = pd.read_sql(
        f"SELECT {org} AS orgId, {loc} AS locusId, "
        f"{korg} AS keggOrg, {kid} AS keggId FROM BestHitKEGG "
        f"WHERE {kid} IS NOT NULL AND {kid} != ''", con)
    bhk["locusId"] = bhk.locusId.astype(str)
    print(f"  BestHitKEGG: {len(bhk):,} hits")

    if "KEGGMember" not in tabs:
        print("ERROR: KEGGMember table absent (needed to map keggId -> kgroup)",
              file=sys.stderr); return 1
    km_cols = [r[1] for r in cur.execute(
        "PRAGMA table_info(KEGGMember)").fetchall()]
    print(f"KEGGMember cols: {km_cols}")
    kg = pick_col(km_cols, "kgroup", "ko", "keggGroup")
    mkid = pick_col(km_cols, "keggId", "keggid")
    mkorg = pick_col(km_cols, "keggOrg", "keggorg")
    join_keys = ["keggId"] + (["keggOrg"] if mkorg and korg else [])
    print(f"  using kgroup={kg} keggOrg={mkorg} keggId={mkid}  join={join_keys}")
    cols_select = f"{kg} AS ko, {mkid} AS keggId"
    if mkorg and korg:
        cols_select += f", {mkorg} AS keggOrg"
    km = pd.read_sql(
        f"SELECT {cols_select} FROM KEGGMember "
        f"WHERE {kg} IS NOT NULL AND {kg} != ''", con)
    print(f"  KEGGMember: {len(km):,} (kgroup, keggId) rows; "
          f"{km.ko.nunique():,} distinct KOs")

    # join BestHitKEGG -> KEGGMember -> ko
    bhk = bhk.merge(km, on=join_keys, how="inner")
    print(f"  after join -> {len(bhk):,} (orgId, locusId, ko) rows over "
          f"{bhk.groupby(['orgId','locusId']).ngroups:,} genes")

    # ---- KgroupEC: kgroup -> EC number ----
    ec_map = None
    if "KgroupEC" in tabs:
        ke_cols = [r[1] for r in cur.execute(
            "PRAGMA table_info(KgroupEC)").fetchall()]
        kkg = pick_col(ke_cols, "kgroup", "ko")
        kec = pick_col(ke_cols, "ec", "ecNumber")
        if kkg and kec:
            ke = pd.read_sql(f"SELECT {kkg} AS ko, {kec} AS ec FROM KgroupEC "
                             f"WHERE {kec} IS NOT NULL AND {kec} != ''", con)
            ec_map = ke
            print(f"  KgroupEC: {len(ke):,} (ko -> ec) rows")

    # ---- locus -> OG id (for "different OG" NOGD test) ----
    # Pull from the per-org gene_agg files written by build_phase2_frame.py.
    og_map = None
    agg_dir = args.out / "agg"
    gene_aggs = sorted(agg_dir.glob("*_gene.parquet")) if agg_dir.exists() else []
    if gene_aggs:
        parts = []
        for p in gene_aggs:
            df = pd.read_parquet(p, columns=["orgId", "locusId", "og_id"])
            df["locusId"] = df.locusId.astype(str)
            parts.append(df)
        og_map = pd.concat(parts, ignore_index=True).dropna(subset=["og_id"])
        og_map["og_id"] = og_map.og_id.astype(str)
        print(f"  loaded og_id map: {len(og_map):,} rows from {len(gene_aggs)} orgs")

    # ---- KO-level redundancy ----
    # For each (orgId, ko): how many distinct loci in this organism?
    # Each gene's n_same_ko_in_genome = that count - 1 (exclude self).
    ko_per_org = (bhk.drop_duplicates(["orgId", "locusId", "ko"])
                     .groupby(["orgId", "ko"]).size().rename("ko_genome_n"))
    bhk2 = bhk.drop_duplicates(["orgId", "locusId", "ko"]).join(
        ko_per_org, on=["orgId", "ko"])
    bhk2["n_same_ko_in_genome"] = (bhk2.ko_genome_n - 1).clip(lower=0)

    # NOGD-pure for KO: same KO, different OG
    if og_map is not None:
        # join og_id by (orgId, locusId)
        bhk3 = bhk2.merge(og_map.drop_duplicates(["orgId", "locusId"]),
                          on=["orgId", "locusId"], how="left")
        # for each (orgId, ko, og_id) count loci; "different OG" = total in
        # genome minus genes-with-same-(ko, og_id)
        same_og = (bhk3.dropna(subset=["og_id"])
                       .groupby(["orgId", "ko", "og_id"]).size()
                       .rename("ko_genome_n_same_og").reset_index())
        bhk3 = bhk3.merge(same_og, on=["orgId", "ko", "og_id"], how="left")
        bhk3["n_same_ko_diff_og"] = (
            bhk3.ko_genome_n.fillna(0) - bhk3.ko_genome_n_same_og.fillna(0)
        ).clip(lower=0)
    else:
        bhk3 = bhk2.copy()
        bhk3["n_same_ko_diff_og"] = pd.NA

    # collapse to one row per (orgId, locusId) by max over KOs
    # (a gene's redundancy = the MOST redundant of its functions; using max
    # is the right "is there ANY backup" signal)
    ko_feat = (bhk3.groupby(["orgId", "locusId"]).agg(
        n_same_ko_in_genome=("n_same_ko_in_genome", "max"),
        n_same_ko_diff_og=("n_same_ko_diff_og", "max"),
        n_ko_annotations=("ko", "nunique"),
    ).reset_index())

    # ---- EC-level redundancy (the true functional-redundancy / NOGD signal) ----
    ec_feat = None
    if ec_map is not None:
        # gene -> ec via KO
        ge_ec = bhk[["orgId", "locusId", "ko"]].merge(ec_map, on="ko", how="inner")
        ge_ec = ge_ec.drop_duplicates(["orgId", "locusId", "ec"])
        if len(ge_ec) > 0:
            ec_per_org = (ge_ec.groupby(["orgId", "ec"]).size()
                              .rename("ec_genome_n"))
            ge_ec = ge_ec.join(ec_per_org, on=["orgId", "ec"])
            ge_ec["n_same_ec_in_genome"] = (
                ge_ec.ec_genome_n - 1).clip(lower=0)
            if og_map is not None:
                ge_ec = ge_ec.merge(og_map.drop_duplicates(["orgId", "locusId"]),
                                    on=["orgId", "locusId"], how="left")
                same_og_ec = (ge_ec.dropna(subset=["og_id"])
                                  .groupby(["orgId", "ec", "og_id"]).size()
                                  .rename("ec_genome_n_same_og").reset_index())
                ge_ec = ge_ec.merge(same_og_ec,
                                    on=["orgId", "ec", "og_id"], how="left")
                ge_ec["n_same_ec_diff_og"] = (
                    ge_ec.ec_genome_n.fillna(0) -
                    ge_ec.ec_genome_n_same_og.fillna(0)).clip(lower=0)
            else:
                ge_ec["n_same_ec_diff_og"] = pd.NA
            ec_feat = (ge_ec.groupby(["orgId", "locusId"]).agg(
                n_same_ec_in_genome=("n_same_ec_in_genome", "max"),
                n_same_ec_diff_og=("n_same_ec_diff_og", "max"),
                n_ec_annotations=("ec", "nunique"),
            ).reset_index())
            print(f"  EC-level redundancy over {len(ec_feat):,} genes")

    # ---- merge KO + EC features ----
    feat = ko_feat
    if ec_feat is not None:
        feat = feat.merge(ec_feat, on=["orgId", "locusId"], how="outer")
    # NaN -> 0 for "no annotation found" (be honest: the FEATURE is informative
    # only where annotated; XGBoost handles NaN, so leave as NaN for downstream
    # but fill the redundancy counts with 0 where there IS at least a KO
    # annotation but no EC).
    for c in ["n_same_ko_in_genome", "n_same_ko_diff_og",
              "n_same_ec_in_genome", "n_same_ec_diff_og",
              "n_ko_annotations", "n_ec_annotations"]:
        if c in feat.columns:
            feat[c] = pd.to_numeric(feat[c], errors="coerce").astype("float32")

    args.out.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out_path, index=False)
    con.close()

    manifest = {"n_genes": len(feat),
                "redundancy_cols": [c for c in feat.columns
                                    if c not in ("orgId", "locusId")],
                "has_ec_redundancy": ec_feat is not None,
                "has_og_split": og_map is not None}
    (args.out / "redundancy_features_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"\nwrote {out_path}  ({len(feat):,} genes, "
          f"{len(manifest['redundancy_cols'])} feature cols)")
    print(f"wrote {args.out / 'redundancy_features_manifest.json'}")
    # quick sanity: what fraction of genes have a backup of any kind?
    if "n_same_ko_in_genome" in feat.columns:
        with_backup = (feat.n_same_ko_in_genome.fillna(0) > 0).mean()
        print(f"  fraction of annotated genes with at least one KO-backup: "
              f"{with_backup:.2f}")
    if "n_same_ec_diff_og" in feat.columns:
        nogd = (feat.n_same_ec_diff_og.fillna(0) > 0).mean()
        print(f"  fraction with a NOGD-style (same-EC, different-OG) backup: "
              f"{nogd:.3f}  <-- the pure non-homologous redundancy signal")
    return 0


if __name__ == "__main__":
    sys.exit(main())

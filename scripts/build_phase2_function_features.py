"""Phase 2 functional features — per-gene Pfam/TIGRFam domain content + SEED.

Option B in the bake-off plan: the model currently has conservation + genome
context + coarse condition class, but NO notion of "what kind of gene is
this". So it ranks our envZ/ompR x bacitracin lead at top ~8-12% instead of
top 0.1% — it defaults to "bacitracin kills cell-wall genes" because it can't
make the envZ -> two-component-system -> porin-regulation -> OM-exclusion
inference.

Fix (leak-free, transferable): give each gene its DOMAIN CONTENT. envZ carries
HisKA + HATPase_c in EVERY organism; if training organisms teach the model
"HisKA-domain genes are bacitracin-sensitive", that transfers to held-out
Shewanella. Domains are annotation-derived (sequence), not fitness-derived ->
no leakage of the held-out organism's labels.

Builds a per-(orgId, locusId) table:
  dom_<DOMAIN> multi-hot over the top-K most common domains (default 150)
  seed_class   ordinal SEED subsystem class (if SEEDAnnotation present)
  n_domains    count of annotated domains (proxy for multi-domain proteins)

Output (small, ~170K genes): <out>/func_features.parquet
        + <out>/func_features_manifest.json (the domain column list)

Resumable: skip if func_features.parquet exists (unless --force).
"""
from __future__ import annotations
import argparse, json, re, sqlite3, sys
from pathlib import Path

DEFAULT_DB = Path("/content/feba.db")
DEFAULT_OUT = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/phase2")


def pick_col(cols, *candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def sanitize(name: str) -> str:
    return "dom_" + re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")[:40]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--top-k", type=int, default=150,
                    help="number of most-common domains to one-hot")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.db.exists():
        alt = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/"
                   "fitness_browser/feba.db")
        if alt.exists(): args.db = alt
        else:
            print(f"ERROR: feba.db not found ({args.db})", file=sys.stderr); return 1

    out_path = args.out / "func_features.parquet"
    if out_path.exists() and not args.force:
        print(f"func_features.parquet already exists -> skip (use --force)")
        import pandas as pd
        man = json.loads((args.out / "func_features_manifest.json").read_text())
        print(f"  {man['n_genes']} genes, {len(man['domain_cols'])} domain cols")
        return 0

    import pandas as pd
    import numpy as np
    con = sqlite3.connect(str(args.db))
    cur = con.cursor()
    tabs = {r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    # ---- domains ----
    if "GeneDomain" not in tabs:
        print("ERROR: GeneDomain table absent", file=sys.stderr); return 1
    gd_cols = [r[1] for r in cur.execute("PRAGMA table_info(GeneDomain)").fetchall()]
    print(f"GeneDomain cols: {gd_cols}")
    loc = pick_col(gd_cols, "locusId", "locus")
    org = pick_col(gd_cols, "orgId", "orgid")
    dname = pick_col(gd_cols, "domainName", "domainId", "domainDb")
    print(f"  using org={org} locus={loc} domain={dname}")
    gd = pd.read_sql(f"SELECT {org} AS orgId, {loc} AS locusId, "
                     f"{dname} AS domain FROM GeneDomain", con)
    gd["locusId"] = gd.locusId.astype(str)
    gd = gd[gd.domain.notna() & (gd.domain.astype(str).str.strip() != "")]
    print(f"  {len(gd):,} domain annotations over "
          f"{gd.groupby(['orgId','locusId']).ngroups:,} genes")

    top = gd.domain.value_counts().head(args.top_k)
    print(f"  top-{args.top_k} domains cover "
          f"{top.sum()/len(gd)*100:.0f}% of annotations")
    # show the litmus-relevant ones if present
    litmus_dom = [d for d in top.index if any(
        k in str(d).lower() for k in
        ["hisk", "hatpase", "response_reg", "trans_reg", "ompr", "porin",
         "sigma", "psp"])]
    print(f"  litmus-relevant domains in top-{args.top_k}: {litmus_dom}")

    keep_doms = list(top.index)
    gd_top = gd[gd.domain.isin(keep_doms)].copy()
    gd_top["col"] = gd_top.domain.map(sanitize)
    # multi-hot
    multi = (gd_top.assign(v=1)
             .pivot_table(index=["orgId", "locusId"], columns="col",
                          values="v", aggfunc="max", fill_value=0))
    multi = multi.reset_index()
    # n_domains per gene (all domains, not just top-K)
    ndom = gd.groupby(["orgId", "locusId"]).size().rename("n_domains").reset_index()
    ndom["locusId"] = ndom.locusId.astype(str)
    feat = multi.merge(ndom, on=["orgId", "locusId"], how="left")

    domain_cols = [c for c in feat.columns if c.startswith("dom_")]

    # ---- SEED class (optional) ----
    if "SEEDAnnotation" in tabs:
        sa_cols = [r[1] for r in cur.execute(
            "PRAGMA table_info(SEEDAnnotation)").fetchall()]
        sloc = pick_col(sa_cols, "locusId", "locus")
        sorg = pick_col(sa_cols, "orgId")
        sdesc = pick_col(sa_cols, "seed_desc", "seedClass", "class", "desc")
        if sloc and sorg and sdesc:
            sa = pd.read_sql(f"SELECT {sorg} AS orgId, {sloc} AS locusId, "
                             f"{sdesc} AS seed FROM SEEDAnnotation", con)
            sa["locusId"] = sa.locusId.astype(str)
            sa = sa.dropna(subset=["seed"]).drop_duplicates(["orgId", "locusId"])
            # ordinal encode by frequency rank (stable, low-card-friendly)
            codes = {v: i for i, v in enumerate(
                sa.seed.value_counts().index)}
            sa["seed_class"] = sa.seed.map(codes).astype("float32")
            feat = feat.merge(sa[["orgId", "locusId", "seed_class"]],
                              on=["orgId", "locusId"], how="left")
            print(f"  SEED: {len(codes)} classes over {len(sa):,} genes")

    feat["locusId"] = feat.locusId.astype(str)
    # downcast domain cols to uint8 to keep the join cheap
    for c in domain_cols:
        feat[c] = feat[c].fillna(0).astype("uint8")
    args.out.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out_path, index=False)
    con.close()

    manifest = {"n_genes": len(feat), "top_k": args.top_k,
                "domain_cols": domain_cols,
                "has_seed": "seed_class" in feat.columns,
                "litmus_domains_present": litmus_dom}
    (args.out / "func_features_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"\nwrote {out_path}  ({len(feat):,} genes, "
          f"{len(domain_cols)} domain cols, "
          f"seed={'yes' if 'seed_class' in feat.columns else 'no'})")
    print(f"wrote {args.out / 'func_features_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

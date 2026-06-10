"""Phase 2 frame builder — assemble the (gene, condition) training table.

Per PHASE2_DESIGN.md (feasibility-corrected): the target is the REPRODUCIBLE
signal, strong conditional vulnerability (binary), NOT the continuous fit
(which reproduces at only ~0.34 in the tail). We keep the continuous fit as a
secondary column but the headline label is strong_hit.

Assembles, from feba.db + the on-Drive gene feature tables:
  rows  = every (organism, gene, experiment) fitness measurement, for
          organisms that have BOTH fitness data and gene features
  label = strong_hit  := (fit < HIT_FIT) & (|t| >= HIT_T)         [primary]
          fit         := continuous (secondary head / ranking)
  gene features (static, leak-free-capable):
          family_frac_essential_fold0..4, family_n_organisms,
          n_paralogs_in_genome, is_orphan
  condition features (static):
          expGroup (coarse MoA), MW (from Compounds), concentration_1,
          pH, temperature, aerobic, media
  keys for leak-free per-fold main-effects + grouping:
          orgId, locusId, compound (normalized), expName, og_id

NOTE on leak-free additive main-effects: they are NOT precomputed here (they
depend on the held-out fold). The bake-off computes gene-mean-fit and
compound-mean-fit per fold, excluding the held-out org/compound. This file
only carries the raw keys needed for that.

Output: outputs/phase2_frame.parquet  (+ stats to stdout)

PERFORMANCE: copy feba.db to local disk first (--db /content/feba.db).
The GeneFitness scan is the heavy part; we pull per-organism to bound memory.
"""
from __future__ import annotations
import argparse, re, sqlite3, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DB = Path("/content/feba.db")
# gene features live on Drive (built earlier over the multiorg "beril_" set)
DEFAULT_FEATS = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/"
                     "orthology_features.csv")

HIT_FIT = -3.0   # strong-vulnerability fitness threshold (matches the atlas)
HIT_T   = 3.0    # min |t| so the label is a confident measurement, not noise


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
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs" / "phase2_frame.parquet")
    ap.add_argument("--hit-fit", type=float, default=HIT_FIT)
    ap.add_argument("--hit-t", type=float, default=HIT_T)
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

    # ---- gene features (keyed by beril_<org>, numeric locus) ----
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
    orgs_with_feats = set(feats_small.fb_org.unique())
    print(f"  gene features for {len(orgs_with_feats)} organisms, "
          f"{len(feats_small):,} loci; cols={gene_feat_cols}")

    con = sqlite3.connect(str(args.db))

    # ---- experiment metadata + compound MW ----
    print("Loading Experiment + Compounds ...")
    exps = pd.read_sql(
        "SELECT orgId, expName, expGroup, condition_1, units_1, "
        "concentration_1, media, pH, temperature, aerobic FROM Experiment", con)
    exps["compound"] = exps.condition_1.apply(norm_compound)
    comp = pd.read_sql("SELECT compound, MW, CAS FROM Compounds", con)
    comp["_n"] = comp.compound.apply(norm_compound)
    mw_map = comp.dropna(subset=["MW"]).groupby("_n").MW.first()
    cas_map = comp.dropna(subset=["CAS"]).groupby("_n").CAS.first()
    exps["MW"] = exps.compound.map(mw_map)
    exps["CAS"] = exps.compound.map(cas_map)   # carried for v2 chem features

    fb_orgs = sorted(orgs_with_feats & set(exps.orgId.unique()))
    print(f"  organisms in BOTH fitness + features: {len(fb_orgs)}")
    print(f"  -> {fb_orgs}")

    # ---- per-organism fitness pull + join ----
    frames = []
    for org in fb_orgs:
        e = exps[exps.orgId == org]
        if e.empty: continue
        gf = pd.read_sql(
            "SELECT orgId, locusId, expName, fit, t FROM GeneFitness "
            "WHERE orgId=?", con, params=(org,))
        if gf.empty: continue
        gf["locusId"] = gf.locusId.astype(str)
        gf = gf.merge(e[["expName", "expGroup", "compound", "MW", "CAS",
                         "concentration_1", "pH", "temperature", "aerobic",
                         "media"]], on="expName", how="left")
        gfe = feats_small[feats_small.fb_org == org].drop(columns=["fb_org"])
        gf = gf.merge(gfe, on="locusId", how="left")
        frames.append(gf)
        print(f"  {org:8s} {len(gf):>8,} rows  "
              f"(feat-matched {gf.og_id.notna().mean()*100:.0f}%)")
    con.close()

    df = pd.concat(frames, ignore_index=True)
    print(f"\nassembled {len(df):,} rows, {df.orgId.nunique()} organisms")

    # ---- label ----
    df["strong_hit"] = ((df.fit < args.hit_fit) &
                        (df.t.abs() >= args.hit_t)).astype(int)
    base = df.strong_hit.mean()
    print(f"  strong_hit base rate (fit<{args.hit_fit} & |t|>={args.hit_t}): "
          f"{base:.4f}  ({int(df.strong_hit.sum()):,} positives)")
    print(f"  per-organism positive counts:")
    print(df.groupby("orgId").strong_hit.agg(["mean", "sum"]).to_string())

    # ---- coverage sanity ----
    print(f"\n  feature coverage:")
    for c in ["MW", "og_id", "family_n_organisms", "concentration_1", "pH"]:
        if c in df.columns:
            print(f"    {c:<22s} non-null {df[c].notna().mean()*100:5.1f}%")
    print(f"  distinct compounds: {df.compound.nunique()}   "
          f"compounds with MW: {df[df.MW.notna()].compound.nunique()}   "
          f"with CAS: {df[df.CAS.notna()].compound.nunique()}")

    # ---- compound-level fold structure (for leak-free LOCO splits) ----
    cpd_org = df.groupby("compound").orgId.nunique()
    print(f"\n  LOCO-ready compounds (>=3 orgs): {(cpd_org>=3).sum()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(args.out, index=False)
        print(f"\nwrote {args.out}  ({args.out.stat().st_size/1e6:.0f} MB)")
    except Exception as e:
        alt = args.out.with_suffix(".csv.gz")
        print(f"  parquet failed ({e}); writing {alt}")
        df.to_csv(alt, index=False, compression="gzip")
        print(f"wrote {alt}")
    print("\nNext: scripts/train_phase2_bakeoff.py (additive / XGBoost-concat / "
          "FiLM) with per-fold leak-free main-effects + tail metrics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

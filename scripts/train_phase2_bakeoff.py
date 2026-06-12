"""Phase 2 bake-off harness — additive vs XGBoost-concat, with leak-free
splits, tail-focused metrics, and the bacitracin litmus.

PER PHASE2_DESIGN.md / feasibility:
- target = binary strong_hit (continuous fit doesn't reproduce)
- ceiling = 0.62-0.78 strong-hit reproducibility; report relative to it
- floor   = additive tail-recall ~0.155
- split-by-COMPOUND, never by experiment (replicate leakage)
- per-fold leak-free family_frac column (fold k for held-out org's clade)
- additive main-effects (gene-mean-fit, cpd-mean-fit) computed from full-data
  per-org aggregates by SUBTRACTING the held-out group, never from raw rows

MODES (--mode):
  loo-org-fast   (default) one fit per arch; held-out test orgs chosen to
                  cover clades + the Shewanella litmus. Fast model selection.
  loo-org-full   every eligible org as a held-out fold (40+ retrains).
                  Publishable headline LOO-organism number.
  loco           leave-one-compound-out. MUST include bacitracin so the
                  litmus runs on a never-seen compound.
  double-blind   held-out = Shewanella ∪ bacitracin. THE adjuvant-discovery
                  setting.

ARCHS (--arch):
  additive   mu + alpha_gene_OG + beta_cond (per held-out org/cpd)
  xgb        gradient-boosted trees on concat features  (GPU if available)
  both       run both, save predictions for each (default)

RESUMABLE: per (mode, arch, fold) outputs to Drive. Skip if predictions
parquet already exists. --force overrides.

OUTPUTS (default DRIVE/phase2/):
  eval/<mode>/preds_<arch>__<fold_label>.parquet
  eval/<mode>/summary.json                (one combined summary per mode)
  models/<mode>/<arch>__<fold_label>.json (xgb model dumps)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/phase2")
DEFAULT_CLADES = (REPO / "data" / "drive_import" / "labels" / "clade_splits.csv")

# Fast bake-off held-out test set: covers clades + carries the litmus.
DEFAULT_FAST_TEST_ORGS = ["SB2B", "Keio", "Phaeo", "pseudo5_N2C3_1"]

# LOCO default: include bacitracin (the litmus) + a few representative drugs.
DEFAULT_LOCO_CPDS = ["bacitracin", "cisplatin", "gentamicin",
                     "nalidixic acid", "d-cycloserine"]

# pos-count floor for a fold to be eligible as a held-out test (avoids
# degenerate test sets with <100 positives)
MIN_POSITIVES_FOR_TEST = 100

NUMERIC_GENE_COLS = ["family_n_organisms", "n_paralogs_in_genome", "is_orphan"]
NUMERIC_COND_COLS = ["MW", "concentration_1", "pH", "temperature", "aerobic"]

# Optional functional features (domain content + SEED), loaded once into these
# module globals from <out>/func_features.parquet if it exists. Leak-free:
# annotation-derived, keyed by (orgId, locusId), so the held-out organism's
# domains are known without ever seeing its fitness.
FUNC_FEATS = None     # DataFrame: orgId, locusId, dom_*, n_domains, seed_class
FUNC_COLS = []        # list of functional feature column names


def load_func_features(out_dir):
    """Populate FUNC_FEATS / FUNC_COLS from func_features.parquet if present.

    Also merges in redundancy_features.parquet (per-genome functional-redundancy
    counts: same-KO/EC genes in this genome, with the NOGD 'same function via
    different OG' variant) if it exists. The two are merged on (orgId, locusId)
    so they look like one feature table to the rest of the harness."""
    global FUNC_FEATS, FUNC_COLS
    import pandas as pd
    p = out_dir / "func_features.parquet"
    pr = out_dir / "redundancy_features.parquet"
    have_func = p.exists()
    have_red = pr.exists()
    if not have_func and not have_red:
        print("  (no func or redundancy features -> running WITHOUT them)")
        return False
    parts = []
    if have_func:
        df = pd.read_parquet(p)
        df["locusId"] = df.locusId.astype(str)
        parts.append(df)
        print(f"  loaded functional features: {len(df):,} genes, "
              f"{len([c for c in df.columns if c.startswith('dom_')])} domain cols")
    if have_red:
        rdf = pd.read_parquet(pr)
        rdf["locusId"] = rdf.locusId.astype(str)
        parts.append(rdf)
        red_cols = [c for c in rdf.columns if c not in ("orgId", "locusId")]
        print(f"  loaded redundancy features: {len(rdf):,} genes, "
              f"cols={red_cols}")
    if len(parts) == 1:
        FUNC_FEATS = parts[0]
    else:
        FUNC_FEATS = parts[0].merge(parts[1], on=["orgId", "locusId"],
                                     how="outer")
    FUNC_COLS = [c for c in FUNC_FEATS.columns
                 if c not in ("orgId", "locusId")]
    return True


def merge_func(df):
    """Left-join functional features onto a frame keyed by (orgId, locusId)."""
    if FUNC_FEATS is None:
        return df
    import pandas as pd
    return df.merge(FUNC_FEATS, on=["orgId", "locusId"], how="left")


# Conditional-conservation prior: per-(orgId, og_id, compound) hit aggregates.
# Loaded once; the per-fold rate EXCLUDES the held-out org -> leak-free.
OGCPD = None          # DataFrame: orgId, og_id, compound, n, n_hit


def load_ogcpd(out_dir):
    """Populate OGCPD from agg_ogcpd/*.parquet if present."""
    global OGCPD
    import pandas as pd
    d = out_dir / "agg_ogcpd"
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    if not files:
        print("  (no agg_ogcpd/ -> running WITHOUT og_cpd_hit_rate)")
        return False
    OGCPD = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    OGCPD["og_id"] = OGCPD.og_id.astype(str)
    print(f"  loaded og_cpd aggregates: {len(OGCPD):,} (org,og,cpd) cells "
          f"from {len(files)} orgs")
    return True


def add_ogcpd_rate(df, held_out_org):
    """Add og_cpd_hit_rate + og_cpd_n -- the conditional analog of family_frac.

    LEAVE-OWN-ORG-OUT + exclude held-out org: for every row from org o, the
    rate is computed over the aggregates of all orgs EXCEPT both the held-out
    test org T AND o itself. This makes the feature leak-free in BOTH train
    (never sees its own org) and test (never sees T), with a consistent
    distribution across the two -- fixing the train-leak that an
    'exclude-T-only' rate causes (train rows would include their own org,
    test rows would not, and the model over-trusts the feature)."""
    if OGCPD is None:
        return df
    import pandas as pd
    df = df.copy()
    df["og_id"] = df.og_id.astype(str)
    sub = OGCPD[OGCPD.orgId != held_out_org]
    tot = sub.groupby(["og_id", "compound"]).agg(
        tot_n=("n", "sum"), tot_hit=("n_hit", "sum")).reset_index()
    own = sub.rename(columns={"orgId": "_o", "n": "own_n", "n_hit": "own_hit"})
    df = df.merge(tot, on=["og_id", "compound"], how="left")
    df = df.merge(own[["_o", "og_id", "compound", "own_n", "own_hit"]],
                  left_on=["orgId", "og_id", "compound"],
                  right_on=["_o", "og_id", "compound"], how="left")
    own_n = df.own_n.fillna(0.0); own_hit = df.own_hit.fillna(0.0)
    den = df.tot_n.fillna(0.0) - own_n          # measurements, others' orgs
    num = df.tot_hit.fillna(0.0) - own_hit      # hits, others' orgs
    df["og_cpd_hit_rate"] = (num / den.where(den > 0)).astype("float32")
    df["og_cpd_n"] = den.clip(lower=0).astype("float32")
    return df.drop(columns=["_o", "own_n", "own_hit", "tot_n", "tot_hit"],
                   errors="ignore")


def load_clade_map(path: Path) -> dict[str, int]:
    """org (without 'beril_') -> fold (0..4) for leak-free family_frac pick."""
    import pandas as pd
    cs = pd.read_csv(path)
    out = {}
    for _, r in cs.iterrows():
        o = str(r.organism)
        bare = o.replace("beril_", "", 1) if o.startswith("beril_") else o
        out[bare] = int(r.fold)
    return out


def ff_col_for_org(org: str, clade_map: dict[str, int]) -> str:
    """Pick the family_frac fold column whose held-out clade excludes org.

    When held-out org X belongs to fold k, fold-k family_frac was computed
    excluding X's clade -> leak-free for X. We use that same column for
    BOTH train and test rows in this fold, which keeps the model honest.
    """
    f = clade_map.get(org, 0)
    return f"family_frac_essential_fold{f}"


def load_shards(out_dir: Path, orgs: list[str], cols: list[str],
                subdir: str = "frame"):
    """Load and concat shards for the given orgs.

    subdir='frame'      -> downsampled shards (for TRAINING)
    subdir='frame_full' -> full un-downsampled shards (for TESTING at true
                           prevalence; recall-at-fixed-precision is only
                           honest on the real negative base rate)
    """
    import pandas as pd
    parts = []
    for o in orgs:
        p = out_dir / subdir / f"{o}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=cols))
    if not parts:
        raise RuntimeError(f"no shards loaded from {subdir}/ for {orgs}")
    return pd.concat(parts, ignore_index=True)


def load_agg(out_dir: Path, orgs: list[str], kind: str):
    """Load per-org sufficient statistics (kind in {'gene','cpd'})."""
    import pandas as pd
    parts = []
    for o in orgs:
        p = out_dir / "agg" / f"{o}_{kind}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    return pd.concat(parts, ignore_index=True) if parts else None


def compute_additive_effects(out_dir, train_orgs, held_out_cpds=None):
    """Leak-free main-effects from per-org sufficient statistics.

    gene effect: mean fit per (orgId, og_id) over train_orgs only (we're
      conditioning on the gene's own OG signature). For test rows we use
      og_id; if the test org is held out and we want a gene effect for it,
      we look up the same og_id's effect across train_orgs.

    cpd effect: mean fit per compound across train_orgs, EXCLUDING any
      held-out compounds (so LOCO is honest).
    """
    import pandas as pd
    ga = load_agg(out_dir, train_orgs, "gene")
    ca = load_agg(out_dir, train_orgs, "cpd")
    mu = float(ga.sum_fit.sum() / ga.n.sum()) if ga is not None and ga.n.sum() else 0.0

    # gene effect by og_id across all train_orgs
    g_by_og = (ga.dropna(subset=["og_id"])
                  .groupby("og_id").agg(sum=("sum_fit","sum"), n=("n","sum")))
    g_by_og["alpha_gene"] = g_by_og["sum"] / g_by_og["n"] - mu
    alpha_gene_map = g_by_og["alpha_gene"].to_dict()  # og_id -> alpha

    # cpd effect across train_orgs; drop held-out compounds
    if held_out_cpds:
        ca = ca[~ca.compound.isin(set(held_out_cpds))]
    c_by_cpd = ca.groupby("compound").agg(sum=("sum_fit","sum"), n=("n","sum"))
    c_by_cpd["beta_cond"] = c_by_cpd["sum"] / c_by_cpd["n"] - mu
    beta_cond_map = c_by_cpd["beta_cond"].to_dict()

    return mu, alpha_gene_map, beta_cond_map


def tail_metrics(y, p, threshold_precisions=(0.30, 0.50)):
    """Tail-focused metrics. Returns dict with AUPRC, calibration, and
    recall at each requested precision."""
    import numpy as np
    from sklearn.metrics import average_precision_score, precision_recall_curve
    if y.sum() == 0:
        return {"auprc": None, "base_rate": float(y.mean())}
    auprc = float(average_precision_score(y, p))
    pr, rc, th = precision_recall_curve(y, p)
    out = {"auprc": auprc, "base_rate": float(y.mean()),
           "n": int(len(y)), "n_pos": int(y.sum())}
    # recall at fixed precision (highest threshold meeting it)
    for tp in threshold_precisions:
        mask = pr[:-1] >= tp
        if mask.any():
            # pick threshold with max recall meeting the precision floor
            idx = int(np.argmax(rc[:-1] * mask))
            out[f"recall_at_p{int(tp*100)}"] = float(rc[idx])
            out[f"thresh_at_p{int(tp*100)}"] = float(th[idx])
        else:
            out[f"recall_at_p{int(tp*100)}"] = 0.0
    return out


def fit_additive(train, mu, alpha_gene_map, beta_cond_map):
    """Additive prediction for a frame: mu + alpha[og_id] + beta[compound]."""
    a = train.og_id.map(alpha_gene_map).fillna(0.0)
    b = train.compound.map(beta_cond_map).fillna(0.0)
    return mu + a + b


def fit_xgb(train_X, train_y, val_X=None, val_y=None, gpu=True):
    """Train one XGBoost classifier with class-balanced weight."""
    import xgboost as xgb
    spw = (train_y == 0).sum() / max((train_y == 1).sum(), 1)
    clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        scale_pos_weight=spw, objective="binary:logistic",
        tree_method="hist", device=("cuda" if gpu else "cpu"),
        eval_metric="aucpr", verbosity=0, random_state=0,
        early_stopping_rounds=30 if val_X is not None else None,
    )
    if val_X is not None:
        clf.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
    else:
        clf.fit(train_X, train_y, verbose=False)
    return clf


def featurize(df, ff_col):
    """Build the feature matrix the model will see. expGroup is encoded by
    ordinal hash; high-cardinality str cols (compound, media, og_id, CAS)
    are excluded from the model — we already extracted their signal into
    leak-free additive main-effects and family_frac columns."""
    import pandas as pd
    import numpy as np
    cols = [ff_col] + NUMERIC_GENE_COLS + NUMERIC_COND_COLS
    cols += [c for c in FUNC_COLS if c in df.columns]   # functional features
    cols += [c for c in ("og_cpd_hit_rate", "og_cpd_n")  # conditional prior
             if c in df.columns]
    X = df[cols].copy()
    # aerobic is stored as a string in feba.db -> map to {1.0, 0.0, NaN}
    if "aerobic" in X.columns:
        X["aerobic"] = (X["aerobic"].astype("string").str.lower()
                        .map({"aerobic": 1.0, "anaerobic": 0.0}))
    # expGroup ordinal (stable hash, fits XGBoost which handles unseen levels)
    if "expGroup" in df.columns:
        X["expGroup_h"] = (df.expGroup.fillna("").astype(str)
                           .apply(lambda s: hash(s) % 10000))
    # add main-effects as columns
    if "additive_pred" in df.columns:
        X["additive_pred"] = df["additive_pred"]
    # force everything numeric; any residual object column becomes NaN
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.astype("float32")


def run_loo_org_fold(out_dir, all_orgs, test_org, archs, gpu, force):
    """Train architectures on (all_orgs \\ test_org), predict on test_org.

    All predictions go to eval/loo-org/ regardless of mode; loo-org-fast and
    loo-org-full just exercise different subsets of folds, so the unified
    dir lets the fast pass be reused by the full pass.
    """
    import pandas as pd
    import numpy as np
    eval_dir = out_dir / "eval" / "loo-org"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fold_results = {}
    train_orgs = [o for o in all_orgs if o != test_org]

    ff_col = ff_col_for_org(test_org, CLADE_MAP)
    # need ff_col in the loaded columns
    base_cols = ["orgId", "locusId", "expName", "fit", "t", "strong_hit",
                 "compound", "expGroup", "og_id"] + NUMERIC_GENE_COLS \
                + NUMERIC_COND_COLS + [ff_col]

    # only do the expensive shard loads if any arch needs to run
    needs_run = any(not (eval_dir / f"preds_{a}__{test_org}.parquet").exists()
                    or force for a in archs)
    train_df = test_df = None
    if needs_run:
        mu, ag, bc = compute_additive_effects(out_dir, train_orgs)
        # TRAIN on downsampled shards; TEST on FULL (true-prevalence) shards
        train_df = merge_func(load_shards(out_dir, train_orgs, base_cols,
                                          subdir="frame"))
        test_df  = merge_func(load_shards(out_dir, [test_org], base_cols,
                                          subdir="frame_full"))
        # conditional-conservation prior, leak-free (excludes held-out org)
        train_df = add_ogcpd_rate(train_df, test_org)
        test_df  = add_ogcpd_rate(test_df,  test_org)
        train_df["additive_pred"] = fit_additive(train_df, mu, ag, bc)
        test_df["additive_pred"]  = fit_additive(test_df,  mu, ag, bc)
        print(f"  [{test_org}] train={len(train_df):,}  test={len(test_df):,}  "
              f"test_pos={int(test_df.strong_hit.sum()):,}  ff_col={ff_col}")

    for arch in archs:
        out_path = eval_dir / f"preds_{arch}__{test_org}.parquet"
        if out_path.exists() and not force:
            # recompute metrics from the saved predictions so the summary is
            # complete even on resumed runs
            pred_df = pd.read_parquet(out_path)
            y = pred_df.strong_hit.values.astype(int)
            p = pred_df.pred.values
            m = tail_metrics(y, p)
            m["arch"] = arch; m["test_org"] = test_org; m["status"] = "cached"
            # de-novo slice from cached og_cpd_n column (added in 86156c0)
            if "og_cpd_n" in pred_df.columns:
                novel = (pred_df.og_cpd_n.fillna(0).values == 0)
                if novel.sum() > 0 and y[novel].sum() > 0:
                    mn = tail_metrics(y[novel], p[novel])
                    m["novel_n"] = int(novel.sum())
                    m["novel_pos"] = int(y[novel].sum())
                    m["novel_auprc"] = mn["auprc"]
                    m["novel_recall_at_p30"] = mn.get("recall_at_p30", 0.0)
            fold_results[arch] = m
            print(f"    {arch:8s} CACHED  AUPRC={m['auprc']:.3f}  "
                  f"R@P30={m.get('recall_at_p30',0):.3f}  "
                  f"R@P50={m.get('recall_at_p50',0):.3f}  "
                  f"base={m['base_rate']:.4f}  "
                  f"novelR@P30={m.get('novel_recall_at_p30','-')}")
            continue
        t0 = time.time()
        if arch == "additive":
            score = -test_df.additive_pred.values
        elif arch == "xgb":
            X_tr = featurize(train_df, ff_col)
            X_te = featurize(test_df,  ff_col)
            y_tr = train_df.strong_hit.values.astype(int)
            clf = fit_xgb(X_tr, y_tr, gpu=gpu)
            score = clf.predict_proba(X_te)[:, 1]
            (out_dir / "models" / "loo-org").mkdir(parents=True, exist_ok=True)
            clf.save_model(str(out_dir / "models" / "loo-org"
                               / f"xgb__{test_org}.json"))
        else:
            continue
        m = tail_metrics(test_df.strong_hit.values.astype(int), score)
        m["arch"] = arch; m["test_org"] = test_org
        m["seconds"] = round(time.time() - t0, 1); m["status"] = "fresh"
        # novel-interaction slice: rows whose (OG x compound) was NEVER seen in
        # any training org (og_cpd_n==0) -> true de-novo generalization, no
        # nearest-neighbour lookup possible. This is the honest hard number.
        if "og_cpd_n" in test_df.columns:
            novel = (test_df.og_cpd_n.fillna(0).values == 0)
            if novel.sum() > 0 and test_df.strong_hit.values[novel].sum() > 0:
                mn = tail_metrics(test_df.strong_hit.values[novel].astype(int),
                                  score[novel])
                m["novel_n"] = int(novel.sum())
                m["novel_pos"] = int(test_df.strong_hit.values[novel].sum())
                m["novel_auprc"] = mn["auprc"]
                m["novel_recall_at_p30"] = mn.get("recall_at_p30", 0.0)
        fold_results[arch] = m
        keep = ["orgId","locusId","expName","compound","expGroup",
                "fit","t","strong_hit"]
        if "og_cpd_n" in test_df.columns: keep.append("og_cpd_n")
        if "og_cpd_hit_rate" in test_df.columns: keep.append("og_cpd_hit_rate")
        pred_df = test_df[keep].copy()
        pred_df["pred"] = score
        pred_df.to_parquet(out_path, index=False)
        print(f"    {arch:8s} AUPRC={m['auprc']:.3f}  "
              f"R@P30={m.get('recall_at_p30',0):.3f}  "
              f"R@P50={m.get('recall_at_p50',0):.3f}  "
              f"base={m['base_rate']:.4f}  "
              f"novelR@P30={m.get('novel_recall_at_p30','-')}  "
              f"({m['seconds']}s)")
    return fold_results


def bacitracin_litmus(out_dir):
    """Read the test_org predictions and check whether envZ/ompR/pspB ×
    bacitracin rows ranked high in their test_org. Always reads from
    eval/loo-org/ (the unified predictions dir). Also reports the top
    bacitracin-experiment predictions per (org, arch) so we can see what
    the model thinks the strongest hits ARE, not just our cluster."""
    import pandas as pd
    cluster = [("envZ","SB2B","6939419"), ("ompR","SB2B","6939418"),
               ("pspB","SB2B","6937067"), ("envZ","PV4","5211222"),
               ("ompR","PV4","5211221"), ("pspB","MR1","200970")]
    eval_dir = out_dir / "eval" / "loo-org"
    rows_out, top_out = [], []
    for gene, org, locus in cluster:
        for arch in ("additive", "xgb"):
            p = eval_dir / f"preds_{arch}__{org}.parquet"
            if not p.exists(): continue
            df = pd.read_parquet(p)
            df["locusId"] = df.locusId.astype(str)
            df = df.sort_values("pred", ascending=False).reset_index(drop=True)
            df["_rank"] = range(1, len(df) + 1)
            # cluster rank lookup
            rows = df[(df.locusId == str(locus)) &
                      (df.compound.fillna("").str.contains("bacitracin",
                                                           case=False, na=False))]
            for _, r in rows.iterrows():
                rows_out.append({
                    "arch": arch, "gene": gene, "org": org,
                    "compound": str(r.compound), "fit": float(r.fit),
                    "pred": float(r.pred), "rank": int(r._rank),
                    "pctile": float(r._rank / len(df))})
    # top-K bacitracin predictions per (org, arch) -- one summary per pair
    seen = set()
    for gene, org, _ in cluster:
        for arch in ("additive", "xgb"):
            key = (org, arch)
            if key in seen: continue
            seen.add(key)
            p = eval_dir / f"preds_{arch}__{org}.parquet"
            if not p.exists(): continue
            df = pd.read_parquet(p)
            df["locusId"] = df.locusId.astype(str)
            baci = df[df.compound.fillna("").str.contains("bacitracin",
                                                          case=False, na=False)]
            if baci.empty: continue
            baci = baci.sort_values("pred", ascending=False).head(10)
            top_out.append({"org": org, "arch": arch, "n_baci_rows": int(len(df[
                df.compound.fillna("").str.contains("bacitracin", case=False,
                                                    na=False)])),
                "top10": baci[["locusId","compound","fit","pred",
                               "strong_hit"]].to_dict("records")})
    return {"cluster_ranks": rows_out, "top_baci": top_out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--clades", type=Path, default=DEFAULT_CLADES)
    ap.add_argument("--mode", choices=["loo-org-fast","loo-org-full"],
                    default="loo-org-fast")
    ap.add_argument("--arch", choices=["additive","xgb","both"], default="both")
    ap.add_argument("--test-orgs", nargs="*", default=None,
                    help="explicit held-out test orgs (loo-org-fast)")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--no-func", action="store_true",
                    help="ignore func_features.parquet (ablation)")
    ap.add_argument("--no-ogcpd", action="store_true",
                    help="ignore og_cpd conditional-conservation prior (ablation)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    global CLADE_MAP
    CLADE_MAP = load_clade_map(args.clades)
    if not args.no_func:
        load_func_features(args.out)
    if not args.no_ogcpd:
        load_ogcpd(args.out)
    manifest = json.loads((args.out / "_build_manifest.json").read_text())
    all_orgs = [r["org"] for r in manifest["per_org"]]
    pos_by_org = {r["org"]: r["n_pos"] for r in manifest["per_org"]}

    eligible = [o for o in all_orgs
                if pos_by_org[o] >= MIN_POSITIVES_FOR_TEST and o in CLADE_MAP]
    skipped_small = [o for o in all_orgs if pos_by_org[o] < MIN_POSITIVES_FOR_TEST]
    no_clade      = [o for o in all_orgs if o not in CLADE_MAP]
    print(f"=== orgs ===")
    print(f"  total: {len(all_orgs)}   eligible test folds: {len(eligible)}")
    print(f"  skipped (too few positives <{MIN_POSITIVES_FOR_TEST}): "
          f"{skipped_small}")
    if no_clade:
        print(f"  skipped (no clade fold assignment): {no_clade}")

    archs = ["additive","xgb"] if args.arch == "both" else [args.arch]

    if args.mode == "loo-org-fast":
        if args.test_orgs:
            test_orgs = args.test_orgs
        else:
            # default test set: SB2B (litmus, big), Keio (canonical), Phaeo,
            # pseudo5_N2C3_1 -- if any aren't eligible, drop them.
            test_orgs = [o for o in DEFAULT_FAST_TEST_ORGS if o in eligible]
        print(f"\n=== loo-org-fast | held-out test orgs: {test_orgs} ===")
        results = {}
        for to in test_orgs:
            print(f"\n-- fold: hold out {to} --")
            results[to] = run_loo_org_fold(
                args.out, all_orgs, to, archs, gpu=not args.no_gpu,
                force=args.force)

    elif args.mode == "loo-org-full":
        print(f"\n=== loo-org-full | {len(eligible)} folds ===")
        results = {}
        for i, to in enumerate(eligible, 1):
            print(f"\n-- fold {i}/{len(eligible)}: hold out {to} --")
            results[to] = run_loo_org_fold(
                args.out, all_orgs, to, archs, gpu=not args.no_gpu,
                force=args.force)

    # ---- bacitracin litmus ----
    print(f"\n=== bacitracin litmus (from eval/loo-org/) ===")
    lit = bacitracin_litmus(args.out)
    if lit["cluster_ranks"]:
        print(f"  cluster ranks (envZ/ompR/pspB x bacitracin):")
        for r in lit["cluster_ranks"]:
            print(f"    {r['arch']:8s} {r['gene']:5s} {r['org']:5s} "
                  f"fit={r['fit']:+5.2f}  pred={r['pred']:+.3f}  "
                  f"rank={r['rank']:>6d}  top {r['pctile']*100:6.3f}%")
    else:
        print(f"  (no cluster rows matched -- shard/compound mismatch?)")
    if lit["top_baci"]:
        print(f"\n  top-10 bacitracin predictions per held-out org x arch:")
        for entry in lit["top_baci"]:
            print(f"    --- {entry['org']:5s} {entry['arch']:8s}  "
                  f"({entry['n_baci_rows']} bacitracin rows in test) ---")
            for r in entry["top10"]:
                print(f"      locus={str(r['locusId']):<10s} "
                      f"fit={r['fit']:+5.2f} pred={r['pred']:+.3f} "
                      f"hit={int(r['strong_hit'])}  cpd={r['compound']}")

    # ---- summary ----
    summary = {"mode": args.mode, "archs": archs,
               "test_orgs": list(results.keys()),
               "per_fold": results, "bacitracin_litmus": lit,
               "note": ("TRAIN on 10:1 downsampled shards; TEST on FULL "
                        "un-downsampled shards (frame_full/) so recall@P is "
                        "at TRUE prevalence. novel_* = slice where (OG x "
                        "compound) was never seen in any training org "
                        "(og_cpd_n==0): the de-novo generalization number.")}
    sp = args.out / "eval" / "loo-org" / f"summary_{args.mode}.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nwrote {sp}")

    # ---- macro table: true-prevalence recall@P + de-novo novel slice ----
    print(f"\n=== macro across folds (TEST at true prevalence) ===")
    import numpy as np
    for arch in archs:
        rs = [r[arch] for r in results.values()
              if arch in r and r[arch].get("auprc") is not None]
        if not rs: continue
        aps = [x["auprc"] for x in rs]
        r30 = [x.get("recall_at_p30", 0.0) for x in rs]
        base = [x.get("base_rate", float("nan")) for x in rs]
        nov = [x["novel_recall_at_p30"] for x in rs
               if x.get("novel_recall_at_p30") is not None]
        print(f"  {arch:8s}  AUPRC mean={np.mean(aps):.3f}   "
              f"recall@P30 mean={np.mean(r30):.3f}   "
              f"test_base_rate~{np.nanmean(base):.4f}")
        if nov:
            print(f"           de-novo (og_cpd_n==0) recall@P30 "
                  f"mean={np.mean(nov):.3f}  over {len(nov)} folds")
    return 0


if __name__ == "__main__":
    sys.exit(main())

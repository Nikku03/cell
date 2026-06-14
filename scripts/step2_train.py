"""STEP 2b: the condition-conditional fitness model + the road-to-0.77 readout.

Trains gradient-boosted regressors on the CES consensus_fitness label and asks
the question that matters: how close to the matched-quality ceiling (rho~0.77)
do we get on held-out clades, and does the CONDITION axis add over conservation?

Three nested models, leave-one-clade-out, compared on Spearman(pred, consensus):
  M_cons  : family_frac only            (the conservation baseline / the bar)
  M_gene  : all gene features           (conservation + cooccur + regulator + ...)
  M_full  : gene + CONDITION features    (the condition-conditional model)

CONDITION features are built LEAK-FREE per fold: from the TRAINING clades only,
the per-condition mean consensus_fitness and essential-rate (target encoding),
plus aerobic flag and condition size. Held-out clade conditions get the
training encoding; unseen conditions get the global train mean.

ABSTENTION (conservative bias): every metric is reported on (a) ALL held-out
cells and (b) the high-confidence subset (top ~85% by total_weight) -- the
regime where the matched-quality ceiling actually applies. We abstain on the
bottom ~15% by total_weight, not by mean_cor12 (CES averaging means a multi-
screen cell with modest per-screen quality is still reliable).

Reference lines: raw floor rho~0.38, matched-quality ceiling rho~0.77.

--smoke : validate train / LOCO / leak-free encoding / eval on synthetic data
--real  : train on outputs/step2_training_frame.parquet (Colab)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs"
FRAME_CANDIDATES = [
    OUT / "step2_training_frame.parquet",
    Path("/content/drive/MyDrive/step2_training_frame.parquet"),
]
FLOOR_RHO = 0.38      # raw cross-experiment median (Paper 1)
CEILING_RHO = 0.77    # matched-quality ceiling (Paper 1 decomposition)

GENE_FEATS = ["family_frac", "prevalence_og", "n_paralogs_in_genome",
              "is_orphan", "cooccur_max_jaccard", "cooccur_n_neighbors_50",
              "cooccur_essential_neighbor_frac", "is_regulator",
              "is_transporter", "is_signaling", "is_conditional"]
COND_FEATS = ["cond_aerobic", "cond_size", "cond_mean_fit", "cond_ess_rate"]


def spearman(a, b):
    import numpy as np
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    if len(a) < 30:
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = ra.std() * rb.std() * len(a)
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def auprc(y, s):
    import numpy as np
    y = np.asarray(y, int); s = np.asarray(s, float)
    m = ~np.isnan(s); y, s = y[m], s[m]
    if y.sum() < 5 or len(np.unique(y)) < 2:
        return float("nan")
    o = np.argsort(-s); ys = y[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(y.sum(), 1)
    return float(np.trapezoid(prec, rec))


def add_condition_cols(df):
    """Parse aerobic flag + size from condition_cluster. (mean_fit/ess_rate are
    target-encoded leak-free inside the LOCO loop.)"""
    aer = df.condition_cluster.str.split("|").str[1].fillna("-")
    df["cond_aerobic"] = (aer.str.lower().str.startswith("aerobic")).astype(int)
    df["cond_size"] = df.groupby("condition_cluster").locus_tag.transform("size")
    return df


def _encode_condition(train, test):
    """Leak-free per-condition target encoding from TRAIN rows only."""
    import numpy as np
    gmean = train.consensus_fitness.mean()
    gess = (train.consensus_essential >= 0.5).mean()
    cm = train.groupby("condition_cluster").consensus_fitness.mean()
    ce = (train.assign(e=(train.consensus_essential >= 0.5).astype(int))
              .groupby("condition_cluster").e.mean())
    for d in (train, test):
        d["cond_mean_fit"] = d.condition_cluster.map(cm).fillna(gmean)
        d["cond_ess_rate"] = d.condition_cluster.map(ce).fillna(gess)
    return train, test


_XGB_DEVICE = None  # cached: "cuda" if a usable GPU is present, else "cpu"


def _xgb_device():
    """Detect once whether xgboost can use a GPU. GPU hist is numerically
    near-identical to CPU hist (same algorithm, different float reduction
    order) -- fine for a pass/fail kill-gate, and ~10x faster on the
    654-feature ESM model."""
    global _XGB_DEVICE
    if _XGB_DEVICE is not None:
        return _XGB_DEVICE
    _XGB_DEVICE = "cpu"
    try:
        import xgboost as xgb, numpy as np
        Xt = np.random.rand(64, 4).astype("float32")
        yt = np.random.rand(64).astype("float32")
        xgb.XGBRegressor(tree_method="hist", device="cuda",
                         n_estimators=2).fit(Xt, yt)
        _XGB_DEVICE = "cuda"
    except Exception as e:
        print(f"  [xgb] GPU not usable ({type(e).__name__}); using CPU")
    return _XGB_DEVICE


def _fit_predict(Xtr, ytr, Xte, fast=True):
    import numpy as np
    try:
        import xgboost as xgb
        m = xgb.XGBRegressor(
            tree_method="hist", device=_xgb_device(),
            n_estimators=300 if not fast else 150,
            max_depth=7, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, max_bin=128)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    except ImportError:
        # numpy fallback: ridge regression (smoke only)
        Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
        Xtr1 = np.hstack([np.ones((len(Xtr), 1)), Xtr])
        Xte1 = np.hstack([np.ones((len(Xte), 1)), Xte])
        A = Xtr1.T @ Xtr1 + 1e-2 * np.eye(Xtr1.shape[1])
        w = np.linalg.solve(A, Xtr1.T @ ytr)
        return Xte1 @ w


def evaluate(df, fast=True, subsample=None, esm_cols=None, esm_df=None):
    import numpy as np, pandas as pd
    df = add_condition_cols(df)
    clades = sorted(df.clade.dropna().unique())
    models = {"M_cons": ["family_frac"],
              "M_gene": GENE_FEATS,
              "M_full": GENE_FEATS + COND_FEATS}
    if esm_cols:
        models["M_full_esm"] = GENE_FEATS + COND_FEATS + esm_cols
    res = {k: {"all": [], "hiconf": []} for k in models}
    res_ap = {k: [] for k in models}
    rng = np.random.RandomState(0)
    for held in clades:
        tr = df[df.clade != held]
        te = df[df.clade == held]
        if len(te) < 100 or (te.consensus_essential >= 0.5).sum() < 10:
            continue
        if subsample and len(tr) > subsample:
            tr = tr.sample(subsample, random_state=0)
        # ESM merge is per-fold (after subsample) to keep peak RAM bounded by
        # subsample x 640, not full-frame x 640. Drop rows with no embedding.
        if esm_df is not None:
            tr = tr.merge(esm_df, on=["organism", "locus_tag"], how="left")
            te = te.merge(esm_df, on=["organism", "locus_tag"], how="left")
            tr = tr.dropna(subset=[esm_cols[0]])
            te = te.dropna(subset=[esm_cols[0]])
        tr, te = _encode_condition(tr.copy(), te.copy())
        yte = te.consensus_fitness.values
        yte_ess = (te.consensus_essential >= 0.5).astype(int).values
        # high-confidence subset of the held-out clade (top 85% total_weight)
        thr = te.total_weight.quantile(0.15)
        hi = te.total_weight.values >= thr
        for name, cols in models.items():
            pred = _fit_predict(tr[cols].values, tr.consensus_fitness.values,
                                te[cols].values, fast=fast)
            # predicted fitness is low for essential -> Spearman naturally signed
            res[name]["all"].append(spearman(pred, yte))
            if hi.sum() > 30:
                res[name]["hiconf"].append(spearman(pred[hi], yte[hi]))
            # essential AUPRC: more-negative predicted fitness = more essential
            res_ap[name].append(auprc(yte_ess, -pred))
    out = {}
    for name in models:
        a = [x for x in res[name]["all"] if not np.isnan(x)]
        h = [x for x in res[name]["hiconf"] if not np.isnan(x)]
        ap = [x for x in res_ap[name] if not np.isnan(x)]
        out[name] = {
            "spearman_all": round(float(np.mean(a)), 4) if a else None,
            "spearman_hiconf": round(float(np.mean(h)), 4) if h else None,
            "auprc_essential": round(float(np.mean(ap)), 4) if ap else None,
            "n_clades": len(a)}
    return out


def _report(out):
    print(f"\n  reference: raw floor rho={FLOOR_RHO}  "
          f"matched-quality ceiling rho={CEILING_RHO}")
    print(f"\n  {'model':<8} {'Spearman(all)':>14} {'Spearman(hi-conf)':>18} "
          f"{'AUPRC-ess':>10}  clades")
    for name, d in out.items():
        print(f"  {name:<8} {str(d['spearman_all']):>14} "
              f"{str(d['spearman_hiconf']):>18} "
              f"{str(d['auprc_essential']):>10}  {d['n_clades']}")
    # AUPRC is the product metric for conditional essentiality
    cons = out.get("M_cons", {})
    full = out.get("M_full", {})
    esm = out.get("M_full_esm", {})
    def ap(m): return m.get("auprc_essential")
    print(f"\n  AUPRC-essential (the product metric, base rate ~0.015):")
    if ap(cons) is not None:
        print(f"    conservation:        {ap(cons):.4f}")
    if ap(full) is not None:
        print(f"    + condition:         {ap(full):.4f}  "
              f"({ap(full)-ap(cons):+.4f} vs conservation)")
    if ap(esm) is not None:
        print(f"    + condition + ESM:   {ap(esm):.4f}  "
              f"({ap(esm)-ap(full):+.4f} vs no-ESM)  "
              f"<<< Stage 2 lift")
        print(f"\n  STAGE-2 KILL-GATE: ESM must add >= +0.02 AUPRC over M_full")
        print(f"    -> ESM lift = {ap(esm)-ap(full):+.4f}  "
              f"{'PASS' if ap(esm)-ap(full) >= 0.02 else 'FAIL/marginal'}")


def run_real(args):
    import pandas as pd
    frame = next((p for p in ([Path(args.frame)] if args.frame
                              else FRAME_CANDIDATES) if p.exists()), None)
    if frame is None:
        raise FileNotFoundError(f"training frame not found in {FRAME_CANDIDATES}")
    print("=" * 70)
    print("STEP 2b -- condition-conditional model: road to the 0.77 ceiling")
    print("=" * 70)
    print(f"  frame: {frame}")
    df = pd.read_parquet(frame)
    print(f"  rows: {len(df):,}  orgs: {df.organism.nunique()}  "
          f"conditions: {df.condition_cluster.nunique()}  "
          f"clades: {df.clade.nunique()}")
    # optional ESM-2 features (Stage 2). Keep ESM as a side table; merge
    # per-fold inside evaluate() so the global frame doesn't get inflated
    # by 640 columns x 8.3M rows (~21 GB, kills Colab RAM).
    esm_cols = None
    esm_df = None
    if args.esm:
        epath = next((p for p in [Path(args.esm),
                                  OUT / "esm_embeddings.parquet",
                                  Path("/content/drive/MyDrive/esm_embeddings.parquet")]
                      if p.exists()), None)
        if epath is None:
            raise FileNotFoundError(f"--esm given but no embeddings at {args.esm}")
        esm_df = pd.read_parquet(epath)
        esm_cols = [c for c in esm_df.columns if c.startswith("e")
                    and c not in ("essential",)]
        # gene-level coverage (one embedding per (organism, locus_tag))
        gkeys = df[["organism", "locus_tag"]].drop_duplicates()
        cov = gkeys.merge(esm_df[["organism", "locus_tag"]],
                          on=["organism", "locus_tag"], how="left",
                          indicator=True)._merge.eq("both").mean()
        print(f"  ESM loaded: {len(esm_df):,} genes x {len(esm_cols)} dims, "
              f"{cov:.1%} of frame genes covered  ({epath})")
    print(f"  xgboost device: {_xgb_device()}")
    out = evaluate(df, fast=args.fast, subsample=args.subsample,
                   esm_cols=esm_cols, esm_df=esm_df)
    _report(out)
    OUT.mkdir(exist_ok=True)
    (OUT / "step2_model_results.json").write_text(json.dumps({
        "models": out, "floor_rho": FLOOR_RHO, "ceiling_rho": CEILING_RHO,
        "subsample": args.subsample, "fast": args.fast}, indent=2))
    print(f"\nwrote {OUT/'step2_model_results.json'}")
    return 0


def run_smoke():
    import numpy as np, pandas as pd
    print("=" * 70)
    print("STEP 2b SMOKE -- train / LOCO / leak-free encoding on synthetic data")
    print("=" * 70)
    rng = np.random.RandomState(0)
    n = 40000
    clades = [f"c{i}" for i in range(6)]
    ff = rng.beta(2, 2, n)                       # conservation
    cond = rng.randint(0, 20, n)                 # 20 conditions
    cond_effect = rng.normal(0, 1, 20)[cond]     # condition shifts fitness
    # true fitness: essential (low fit) when family_frac high AND condition-on
    fit = -3*ff - 0.5*cond_effect*ff + rng.normal(0, 0.5, n)
    df = pd.DataFrame({
        "organism": "o", "locus_tag": [str(i) for i in range(n)],
        "condition_cluster": [f"m|aerobic|c{c}" for c in cond],
        "consensus_fitness": fit,
        "consensus_essential": (fit < -2).astype(float),
        "family_frac": ff, "prevalence_og": ff*0.9+rng.normal(0,.05,n),
        "n_paralogs_in_genome": 0, "is_orphan": 0,
        "cooccur_max_jaccard": rng.random(n),
        "cooccur_n_neighbors_50": rng.randint(0,10,n),
        "cooccur_essential_neighbor_frac": rng.random(n),
        "is_regulator": 0, "is_transporter": 0, "is_signaling": 0,
        "is_conditional": 0,
        "total_weight": rng.random(n)*2,
        "clade": rng.choice(clades, n)})
    out = evaluate(df, fast=True, subsample=None)
    _report(out)
    assert out["M_cons"]["spearman_all"] is not None
    assert out["M_full"]["spearman_all"] is not None
    print("\n=== STEP 2b SMOKE PASS ===")
    print("  LOCO loop, leak-free condition encoding, 3-model comparison,")
    print("  hi-conf abstention subset, and ceiling readout all run. "
          "Run --real on Colab (pip install xgboost).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--frame", default=None)
    ap.add_argument("--esm", default=None,
                    help="path to esm_embeddings.parquet (adds M_full_esm model)")
    ap.add_argument("--subsample", type=int, default=1_500_000,
                    help="max train rows per fold (speed); 0 = full")
    ap.add_argument("--fast", action="store_true", default=True)
    ap.add_argument("--full", dest="fast", action="store_false")
    args = ap.parse_args()
    if args.subsample == 0:
        args.subsample = None
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())

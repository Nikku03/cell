"""PATH-ORPHAN, STEPS 6+8 (Phase C) -- combined model + permutation null + GATE.

Assembles every available feature on top of conservation and asks the only
question that matters: in the ROGUE ZONE (essential, conservation<0.1), where
conservation's R@P30 is 0.000, do the new features produce ANY P>=0.30 head, and
does it beat a label-permutation null?

Feature-agnostic: merges whichever of these exist (by locus_tag), so it runs as
soon as step 3 is done and gets stronger as 4/5/B land:
  dnds_<org>.parquet    (dnds, dn, ds, pct_identity)         [step 3]
  foldhit_<org>.parquet (fold_hit_named, fold_best_tm, ...)  [step 4]
  cofit_<org>.parquet   (cofit_ess, cofit_n)                 [step 5]
  esm_<org>.parquet     (esm_0..esm_k PCs)                   [phase B]

Eval: 5-fold CV (out-of-fold) logistic regression (pure NumPy; Colab can swap
xgboost). conservation is leak-free already; the new features are label-free
intrinsic/structural/cofitness signals. The permutation null (shuffle y, refit)
controls for small-n optimism -- the real rogue-zone R@P30 must exceed it.

OUTPUT:
  outputs/orphan/model_<org>.json   per-feature + combined rogue R@P30, null, gate

--smoke : synthetic data where one feature truly carries signal -> gate PASS;
          and a pure-noise feature -> gate FAIL (both directions verified)
--real  : run on the org's assembled features
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "orphan"
ROGUE_CONS = 0.10
PREC = 0.30
FEATURE_FILES = ["dnds", "foldhit", "cofit", "esm"]


def recall_at_precision(score, y, target=PREC):
    import numpy as np
    score = np.asarray(score, float); y = np.asarray(y, int)
    m = ~np.isnan(score); score, y = score[m], y[m]
    if y.sum() == 0:
        return 0.0
    order = np.argsort(-score); ys = y[order]
    tp = np.cumsum(ys); ks = np.arange(1, len(ys) + 1); prec = tp / ks
    ok = np.where(prec >= target)[0]
    return float(tp[int(ok.max())] / y.sum()) if len(ok) else 0.0


def perm_null(X, y, perms, seed=1):
    """Correct permutation null: shuffle labels, refit CV, score on the SAME
    shuffled labels. Returns array of null rogue-zone R@P30."""
    import numpy as np
    rng = np.random.RandomState(seed)
    out = []
    for i in range(perms):
        yp = rng.permutation(y)
        op = _logreg_cv_oof(X, yp, seed=i + 1)
        out.append(recall_at_precision(op, yp))
    return np.array(out)


def _logreg_cv_oof(X, y, folds=5, iters=400, lr=0.5, l2=1.0, seed=0):
    """Pure-NumPy L2 logistic regression, return out-of-fold probabilities."""
    import numpy as np
    rng = np.random.RandomState(seed)
    n = len(y); idx = rng.permutation(n); oof = np.zeros(n)
    fold = np.array_split(idx, folds)
    for f in range(folds):
        te = fold[f]; tr = np.concatenate([fold[j] for j in range(folds) if j != f])
        mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd == 0] = 1.0
        Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
        Xtr = np.c_[np.ones(len(tr)), Xtr]; Xte = np.c_[np.ones(len(te)), Xte]
        w = np.zeros(Xtr.shape[1])
        ytr = y[tr].astype(float)
        # class weight to counter imbalance
        pos = ytr.mean() or 1e-6
        cw = np.where(ytr == 1, 0.5 / pos, 0.5 / (1 - pos))
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-Xtr @ w))
            g = Xtr.T @ ((p - ytr) * cw) / len(tr)
            g[1:] += l2 * w[1:] / len(tr)
            w -= lr * g
        oof[te] = 1.0 / (1.0 + np.exp(-Xte @ w))
    return oof


def _assemble(org):
    import pandas as pd, numpy as np
    bridge = OUT / f"bridge_{org}.parquet"
    df = pd.read_parquet(bridge)
    feats = {}
    for name in FEATURE_FILES:
        p = OUT / f"{name}_{org}.parquet"
        if p.exists():
            fd = pd.read_parquet(p)
            df = df.merge(fd, on="locus_tag", how="left", suffixes=("", f"_{name}"))
            feats[name] = [c for c in fd.columns if c != "locus_tag"]
    return df, feats


def _numeric_matrix(df, cols):
    import numpy as np, pandas as pd
    mats = []; used = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        if np.all(np.isnan(s)):
            continue
        miss = np.isnan(s).astype(float)
        med = np.nanmedian(s) if not np.all(np.isnan(s)) else 0.0
        s = np.where(np.isnan(s), med, s)
        mats.append(s); used.append(c)
        if miss.any():
            mats.append(miss); used.append(c + "_missing")
    if not mats:
        return np.zeros((len(df), 0)), []
    return np.column_stack(mats), used


def run_real(args):
    import pandas as pd, numpy as np
    if not (OUT / f"bridge_{args.org}.parquet").exists():
        print("ERROR: run orphan_bridge.py --real first"); return 2
    df, feats = _assemble(args.org)
    cand = []
    for name, cols in feats.items():
        cand += cols
    # numeric feature columns (skip ids/strings); dnds sign: low omega -> essential
    if "dnds" in df.columns:
        df["neg_dnds"] = -pd.to_numeric(df["dnds"], errors="coerce")
        cand = ["neg_dnds" if c == "dnds" else c for c in cand]
    numeric_cols = [c for c in cand if c in df.columns]

    rogue = df["conservation"].to_numpy() < ROGUE_CONS
    y = df["essential"].to_numpy().astype(int)
    print("=" * 70)
    print(f"STEP 6+8 model+gate -- {args.org}")
    print(f"  features available: {list(feats) or '(none beyond conservation)'}")
    print(f"  rogue zone: {int(rogue.sum())} genes, {int(y[rogue].sum())} "
          f"essential (base {y[rogue].mean():.3f})")
    print("=" * 70)

    # per-feature univariate rogue-zone R@P30
    print("\n  per-feature rogue-zone R@P30 (univariate):")
    uni = {}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        r = recall_at_precision(s[rogue], y[rogue])
        uni[c] = round(r, 4)
        print(f"    {c:<24} {r:.3f}")

    # combined model (CV OOF) on the rogue zone
    X, used = _numeric_matrix(df, numeric_cols)
    combined_r = null_med = null_p = float("nan")
    if X.shape[1] and rogue.sum() > 20 and y[rogue].sum() >= 3:
        Xr, yr = X[rogue], y[rogue]
        oof = _logreg_cv_oof(Xr, yr, seed=0)
        combined_r = recall_at_precision(oof, yr)
        nulls = perm_null(Xr, yr, args.null_perms)
        null_med = float(np.median(nulls))
        null_p = float((nulls >= combined_r).mean())
        print(f"\n  COMBINED model rogue-zone R@P30 = {combined_r:.3f}")
        print(f"  permutation null ({args.null_perms}x): median {null_med:.3f}, "
              f"p(null>=real) = {null_p:.3f}")
    else:
        print("\n  (combined model skipped: not enough features/positives)")

    gate = (combined_r > 0 and not np.isnan(null_p) and null_p < 0.05)
    verdict = ("PASS -- new features detect rogue essentials conservation cannot"
               if gate else
               "FAIL -- no feature beats the zero bar above the null (so far)")
    print(f"\n  GATE: {verdict}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"model_{args.org}.json").write_text(json.dumps({
        "org": args.org, "features": list(feats),
        "rogue_n": int(rogue.sum()), "rogue_essential": int(y[rogue].sum()),
        "univariate_rogue_RatP30": uni,
        "combined_rogue_RatP30": None if np.isnan(combined_r) else round(combined_r, 4),
        "null_median": None if np.isnan(null_med) else round(null_med, 4),
        "null_p": None if np.isnan(null_p) else round(null_p, 4),
        "bar": 0.0, "gate_pass": bool(gate)}, indent=2))
    print(f"\n  wrote model_{args.org}.json")
    return 0


def run_smoke():
    import numpy as np
    print("=" * 70)
    print("STEP 6+8 model SMOKE -- gate fires on signal, not on noise")
    print("=" * 70)
    rng = np.random.RandomState(0)
    n = 600; y = (rng.rand(n) < 0.1).astype(int)
    # CASE A: a feature that genuinely separates -> gate should PASS
    sig = y * 1.6 + rng.normal(0, 1, n)
    X = sig.reshape(-1, 1)
    oof = _logreg_cv_oof(X, y, seed=0)
    rA = recall_at_precision(oof, y)
    nulls = perm_null(X, y, 30)
    pA = float(np.mean(nulls >= rA))
    print(f"  signal feature : R@P30={rA:.3f}  null_med={np.median(nulls):.3f}  "
          f"p={pA:.3f} -> {'PASS' if rA>0 and pA<0.05 else 'FAIL'}")
    assert rA > 0 and pA < 0.05, "gate must PASS on a real signal"
    # CASE B: pure noise -> gate should FAIL
    Xn = rng.normal(0, 1, n).reshape(-1, 1)
    oofn = _logreg_cv_oof(Xn, y, seed=0)
    rB = recall_at_precision(oofn, y)
    nullsB = perm_null(Xn, y, 30)
    pB = float(np.mean(nullsB >= rB))
    print(f"  noise feature  : R@P30={rB:.3f}  null_med={np.median(nullsB):.3f}  "
          f"p={pB:.3f} -> {'PASS' if rB>0 and pB<0.05 else 'FAIL'}")
    assert not (rB > 0 and pB < 0.05), "gate must FAIL on pure noise"
    # recall_at_precision sanity
    yp = np.zeros(100, int); yp[:10] = 1
    assert recall_at_precision(np.linspace(1, 0, 100), yp) == 1.0
    print("\n=== STEP 6+8 SMOKE PASS ===")
    print("  logistic CV-OOF, rogue R@P30, permutation null, and the gate")
    print("  decision (PASS on signal, FAIL on noise) all validated.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    ap.add_argument("--null_perms", type=int, default=50)
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())

"""Multi-organism essentiality predictor — leave-one-organism-out CV.

Loads the curated multi-organism dataset:

  * ``memory_bank/data/multiorg_essentiality/labels.csv`` — the
    ``(organism, locus_tag, gene_name, essential, source)`` table
    produced by :file:`notebooks/curate_multiorg_session20.ipynb`.
  * ``cell_sim/features/cache/esm2_650M_multiorg.parquet`` — ESM-2
    650M embeddings indexed by the composite key
    ``"{organism}|{locus_tag}"``, produced by the (still-pending)
    embed notebook.

Then runs leave-one-organism-out cross-validation: for each of the
five organisms (E. coli, B. subtilis, M. pneumoniae, M. genitalium,
Syn3A) it trains an XGBoost classifier on the four other organisms'
genes and predicts on the held-out one. The headline metric is
**Syn3A held-out MCC**, compared to the v15 keyword-prior baseline
of 0.5372 (``mcc_against_breuer_v15_round2_priors``).

Decision rule (per ``NEXT_SESSION.md`` Session 20 plan):

  * Syn3A held-out MCC ≥ 0.55  → integrate as a new Layer-6 detector
  * 0.45 ≤ MCC < 0.55          → useful as one input to an ensemble
  * MCC < 0.45                 → falsified at multi-organism scale,
                                  look elsewhere

Synthetic-data path: the trainer accepts ``--synthetic`` to generate
a fake (X, y) matrix of realistic shape and run the LOOCV machinery
end-to-end. That makes the script testable in the sandbox before
real embeddings exist, and keeps the LOOCV split logic + MCC math
unit-testable separately from the heavy real-data path.

Usage::

    # synthetic smoke test (sandbox-runnable today)
    python scripts/train_multiorg_xgboost.py --synthetic \\
        --out outputs/multiorg_xgboost_synthetic.json

    # real run (after the embed notebook lands the parquet)
    python scripts/train_multiorg_xgboost.py \\
        --out outputs/multiorg_xgboost_loocv.json
"""
from __future__ import annotations

import argparse
import json
import time
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_CSV = REPO_ROOT / "memory_bank/data/multiorg_essentiality/labels.csv"
EMBED_PARQUET = (
    REPO_ROOT
    / "cell_sim/features/cache/esm2_650M_multiorg.parquet"
)

ORGANISMS = ("ecoli", "bsub", "mpne", "mgen", "syn3a")
SEED = 42

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_weight=5,
    reg_lambda=1.0,
    objective="binary:logistic",
    tree_method="hist",
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=4,
)

# v15 keyword-prior baseline against Breuer 2019 — the bar.
V15_SYN3A_MCC = 0.5372


# ------------------------------------------------------------------
# data loading
# ------------------------------------------------------------------
def _composite_key(df: pd.DataFrame) -> pd.Series:
    """Build the ``"{organism}|{locus_tag}"`` key. Used both to
    index the embedding parquet and to join labels."""
    return df["organism"].astype(str) + "|" + df["locus_tag"].astype(str)


def load_real_data(
    labels_csv: Path = LABELS_CSV,
    embed_parquet: Path = EMBED_PARQUET,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Load + join the curated labels with the ESM-2 embeddings.

    Returns ``(X, y, org_col, feature_cols)`` where ``X`` is a dense
    DataFrame with one row per gene + one column per embedding
    dim, ``y`` is the binary essentiality label, and ``org_col``
    is the per-row organism string used for LOOCV splits.

    Both files must exist; missing files raise FileNotFoundError
    with an actionable message.
    """
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"missing {labels_csv} — run "
            f"notebooks/curate_multiorg_session20.ipynb first to "
            f"populate the multi-organism dataset"
        )
    if not embed_parquet.exists():
        raise FileNotFoundError(
            f"missing {embed_parquet} — run the multi-organism "
            f"embed Colab notebook to populate ESM-2 features"
        )

    labels = pd.read_csv(labels_csv)
    labels["_key"] = _composite_key(labels)

    embeds = pd.read_parquet(embed_parquet)
    # The embed parquet may be saved with a MultiIndex
    # (organism, locus_tag) or with the composite string already
    # in the index. Normalise to composite string indexed.
    if isinstance(embeds.index, pd.MultiIndex):
        embeds.index = pd.Index(
            [f"{a}|{b}" for a, b in embeds.index],
            name="_key",
        )
    elif embeds.index.name != "_key":
        embeds.index = embeds.index.astype(str)
        embeds.index.name = "_key"

    df = labels.set_index("_key").join(embeds, how="inner")
    feature_cols = [c for c in embeds.columns]
    X = df[feature_cols]
    y = df["essential"].astype(int)
    org = df["organism"].astype(str)
    return X, y, org, feature_cols


def make_synthetic_data(
    n_features: int = 1280, seed: int = SEED,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Generate a fake dataset with the realistic per-organism row
    counts and essentiality rates from the Session 20 README, plus
    a planted weak signal that XGBoost should learn (positive class
    has slightly higher mean in the first 8 features).

    Used by ``--synthetic`` and by ``test_train_multiorg.py`` to
    exercise the LOOCV split + metric math without needing the
    real ESM-2 parquet on disk.
    """
    rng = np.random.default_rng(seed)
    spec = [
        # (organism, n_genes, essential_rate)
        ("ecoli", 4300,  0.07),
        ("bsub",  4200,  0.06),
        ("mpne",   700,  0.50),
        ("mgen",   480,  0.80),
        ("syn3a",  455,  0.84),
    ]
    rows = []
    keys = []
    org_per_row = []
    y_per_row = []
    for org, n, ess_rate in spec:
        n_pos = int(round(n * ess_rate))
        labels = np.array([1] * n_pos + [0] * (n - n_pos))
        rng.shuffle(labels)
        for i, lab in enumerate(labels):
            keys.append(f"{org}|locus_{org}_{i:05d}")
            org_per_row.append(org)
            y_per_row.append(int(lab))
        # Planted signal: positives have +0.4 sigma in dims 0-7;
        # rest is iid standard normal. Keeps XGBoost honest — without
        # any signal it would just learn the per-organism prior.
        block = rng.standard_normal((n, n_features), dtype=np.float32)
        block[labels == 1, :8] += np.float32(0.4)
        rows.append(block)
    X_arr = np.concatenate(rows, axis=0)
    feature_cols = [f"esm2_650M_dim_{i}" for i in range(n_features)]
    X = pd.DataFrame(X_arr, columns=feature_cols, index=pd.Index(keys, name="_key"))
    y = pd.Series(y_per_row, index=X.index, name="essential")
    org = pd.Series(org_per_row, index=X.index, name="organism")
    return X, y, org, feature_cols


# ------------------------------------------------------------------
# LOOCV
# ------------------------------------------------------------------
def _mcc_from_confusion(tp: int, fp: int, tn: int, fn: int) -> float:
    num = tp * tn - fp * fn
    den_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den_sq <= 0:
        return 0.0
    return float(num / sqrt(den_sq))


def loocv(
    X: pd.DataFrame,
    y: pd.Series,
    org: pd.Series,
    organisms: tuple[str, ...] = ORGANISMS,
    params: dict | None = None,
    threshold: float = 0.5,
) -> dict:
    """Leave-one-organism-out cross-validation.

    For each organism present in ``org``, train XGBoost on every
    other organism's rows and predict on the held-out organism.
    Returns a dict keyed by organism name with per-organism
    (mcc, confusion, train_size, wall_s, top_features).
    """
    if params is None:
        params = XGB_PARAMS

    results: dict = {}
    for held in organisms:
        held_mask = (org == held)
        n_held = int(held_mask.sum())
        if n_held == 0:
            continue
        n_train_total = int((~held_mask).sum())
        if n_train_total == 0:
            continue

        X_tr = X[~held_mask]
        X_te = X[held_mask]
        y_tr = y[~held_mask]
        y_te = y[held_mask]
        n_pos_tr = int((y_tr == 1).sum())
        n_neg_tr = int((y_tr == 0).sum())
        if n_pos_tr == 0 or n_neg_tr == 0:
            results[held] = {
                "skipped_reason": (
                    f"degenerate training set: pos={n_pos_tr} "
                    f"neg={n_neg_tr}"
                ),
                "n_train": n_train_total,
                "n_test": n_held,
            }
            continue

        spw = float(n_neg_tr) / max(1, n_pos_tr)
        clf = XGBClassifier(scale_pos_weight=spw, **params)
        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr, verbose=False)
        proba = clf.predict_proba(X_te)[:, 1]
        wall = time.perf_counter() - t0
        y_hat = (proba >= threshold).astype(int)

        mcc = float(matthews_corrcoef(y_te, y_hat)) if len(np.unique(y_te)) > 1 else 0.0
        tn, fp, fn, tp = confusion_matrix(
            y_te, y_hat, labels=[0, 1],
        ).ravel()

        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        top_features = [
            (str(X.columns[i]), float(importances[i])) for i in top_idx
        ]

        results[held] = {
            "n_train": n_train_total,
            "n_test": n_held,
            "n_test_positive": int((y_te == 1).sum()),
            "n_test_negative": int((y_te == 0).sum()),
            "scale_pos_weight": spw,
            "mcc": mcc,
            "pooled_mcc_check": _mcc_from_confusion(
                int(tp), int(fp), int(tn), int(fn),
            ),
            "confusion": {
                "tp": int(tp), "fp": int(fp),
                "tn": int(tn), "fn": int(fn),
            },
            "precision": (
                float(tp / max(1, tp + fp))
            ),
            "recall": float(tp / max(1, tp + fn)),
            "wall_s": wall,
            "top_features": top_features,
        }
    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def _decision_band(syn3a_mcc: float) -> str:
    if syn3a_mcc >= 0.55:
        return "INTEGRATE_AS_DETECTOR"
    if syn3a_mcc >= 0.45:
        return "ENSEMBLE_INPUT_ONLY"
    return "FALSIFIED"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "outputs/multiorg_xgboost_loocv.json",
    )
    ap.add_argument(
        "--synthetic", action="store_true",
        help="use synthetic data (skip the parquet/CSV load)",
    )
    ap.add_argument(
        "--n-features", type=int, default=1280,
        help="for --synthetic only: feature dim (default 1280)",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    if args.synthetic:
        print("[synthetic] generating fake (X, y) with planted signal")
        X, y, org, feature_cols = make_synthetic_data(args.n_features)
    else:
        print("[real] loading labels.csv + esm2_650M_multiorg.parquet")
        X, y, org, feature_cols = load_real_data()

    org_summary = (
        pd.DataFrame({"org": org.values, "y": y.values})
          .groupby("org")["y"]
          .agg(["count", "sum", "mean"])
          .reset_index()
          .rename(columns={
              "count": "n_total",
              "sum": "n_essential",
              "mean": "ess_rate",
          })
    )
    print(f"\nfeatures: {len(feature_cols)}  total rows: {len(X)}")
    print(org_summary.to_string(index=False))

    print("\n=== LOOCV ===")
    res = loocv(X, y, org)

    syn3a = res.get("syn3a", {})
    syn3a_mcc = syn3a.get("mcc", float("nan"))
    syn3a_delta = syn3a_mcc - V15_SYN3A_MCC if not np.isnan(syn3a_mcc) else None
    decision = _decision_band(syn3a_mcc) if not np.isnan(syn3a_mcc) else "NO_SYN3A_DATA"

    for org_name, r in res.items():
        if "skipped_reason" in r:
            print(f"  {org_name:6s}  SKIPPED ({r['skipped_reason']})")
            continue
        print(f"  {org_name:6s}  mcc={r['mcc']:+.4f}  "
              f"prec={r['precision']:.3f}  rec={r['recall']:.3f}  "
              f"tp={r['confusion']['tp']:<5d}fp={r['confusion']['fp']:<5d}"
              f"tn={r['confusion']['tn']:<5d}fn={r['confusion']['fn']:<5d}  "
              f"wall={r['wall_s']:.1f}s  "
              f"n_train={r['n_train']}  n_test={r['n_test']}")

    print(f"\n=== headline ===")
    print(f"  Syn3A held-out MCC : {syn3a_mcc:.4f}")
    print(f"  v15 baseline       : {V15_SYN3A_MCC:.4f}")
    if syn3a_delta is not None:
        print(f"  delta              : {syn3a_delta:+.4f}")
    print(f"  decision band      : {decision}")

    out = {
        "synthetic": bool(args.synthetic),
        "n_features": len(feature_cols),
        "n_total_rows": int(len(X)),
        "organism_summary": org_summary.to_dict(orient="records"),
        "v15_syn3a_mcc": V15_SYN3A_MCC,
        "syn3a_held_out_mcc": float(syn3a_mcc) if not np.isnan(syn3a_mcc) else None,
        "syn3a_delta_over_v15": (
            float(syn3a_delta) if syn3a_delta is not None else None
        ),
        "decision_band": decision,
        "threshold": args.threshold,
        "xgb_params": params_for_json(XGB_PARAMS),
        "per_organism": res,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {args.out}")
    return 0


def params_for_json(p: dict) -> dict:
    return {k: v for k, v in p.items() if isinstance(v, (str, int, float, bool))}


if __name__ == "__main__":
    raise SystemExit(main())

"""Post-hoc calibration + decision-threshold optimization.

Why: analyze_loo_errors.py showed 16,464 of 33,987 FPs have prob > 0.9 --
the XGBoost is over-confident in the family_frac_essential 0.4-0.8
borderline zone. Recalibrating the probability scale and raising the
decision threshold can move many of these from "confident wrong" to
"uncertain wrong" / "non-essential", lifting precision without burning
recall.

No retraining required: works on the existing loo_predictions.csv.

Approach (leakage-free):
  For each fold k, fit a calibrator on the predictions from all OTHER
  folds (k' != k), then apply to fold k. The held-out fold is never
  used to fit the mapping that scores it.

Methods:
  - Isotonic regression (monotonic, distribution-free)
  - Platt scaling (sigmoid via LogisticRegression on raw prob)

Threshold sweep:
  Sweep t in [0.05, 0.95] step 0.01, recompute MCC on the full set,
  pick the t that maximises MCC. Reports the decomposed effect of
  calibration alone (t=0.5 on calibrated probs) vs. calibration + best
  threshold.

Outputs:
  calibrated_loo_predictions.csv  (raw rows + calibrated_prob + pred_optimal)
  calibration_summary.json        (best t, MCC, deltas, full sweep)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality"
PRED_CSV  = DATA_DIR / "loo_predictions.csv"
OUT_CSV   = DATA_DIR / "calibrated_loo_predictions.csv"
OUT_JSON  = DATA_DIR / "calibration_summary.json"


def confusion(y_true, pred):
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": tp / max(tp + fp, 1),
            "recall":    tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-csv", type=Path, default=PRED_CSV)
    p.add_argument("--out-csv",  type=Path, default=OUT_CSV)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--method",   choices=["isotonic", "platt", "both"],
                   default="both")
    args = p.parse_args()

    try:
        import pandas as pd
        import numpy as np
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import matthews_corrcoef
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1

    print(f"Loading {args.pred_csv} ...")
    df = pd.read_csv(args.pred_csv)
    print(f"  {len(df)} rows  {df.fold.nunique()} folds")

    y = df.y_true.values
    base_pred = (df.prob > 0.5).astype(int).values
    base_mcc  = float(matthews_corrcoef(y, base_pred))
    base_cm   = confusion(y, base_pred)
    print(f"\n=== uncalibrated baseline (t=0.5) ===")
    print(f"  TP={base_cm['TP']} FP={base_cm['FP']} TN={base_cm['TN']} FN={base_cm['FN']}")
    print(f"  prec={base_cm['precision']:.3f}  rec={base_cm['recall']:.3f}  MCC={base_mcc:+.4f}")

    folds = sorted(df.fold.unique())

    def calibrate_isotonic():
        out = np.zeros(len(df))
        for k in folds:
            tr = df.fold != k; te = df.fold == k
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            iso.fit(df.loc[tr, "prob"].values, df.loc[tr, "y_true"].values)
            out[te.values] = iso.transform(df.loc[te, "prob"].values)
        return out

    def calibrate_platt():
        out = np.zeros(len(df))
        for k in folds:
            tr = df.fold != k; te = df.fold == k
            lr = LogisticRegression(max_iter=500)
            lr.fit(df.loc[tr, "prob"].values.reshape(-1, 1),
                   df.loc[tr, "y_true"].values)
            out[te.values] = lr.predict_proba(
                df.loc[te, "prob"].values.reshape(-1, 1))[:, 1]
        return out

    calibs = {}
    if args.method in ("isotonic", "both"):
        print("\nFitting cross-fold isotonic ..."); calibs["isotonic"] = calibrate_isotonic()
    if args.method in ("platt", "both"):
        print("Fitting cross-fold Platt ...");     calibs["platt"] = calibrate_platt()

    def sweep(probs):
        ts = np.arange(0.05, 0.96, 0.01)
        best = {"t": 0.5, "MCC": -1.0, "cm": None}
        rows = []
        for t in ts:
            pred = (probs > t).astype(int)
            mcc  = float(matthews_corrcoef(y, pred))
            cm   = confusion(y, pred)
            rows.append({"t": float(t), "MCC": mcc, **cm})
            if mcc > best["MCC"]:
                best = {"t": float(t), "MCC": mcc, "cm": cm}
        return best, rows

    out = {
        "uncalibrated_baseline_t0.5": {"MCC": base_mcc, **base_cm},
        "calibrations": {},
    }

    print("\n=== uncalibrated, threshold sweep ===")
    b_u, rows_u = sweep(df.prob.values)
    cm = b_u["cm"]
    print(f"  best t={b_u['t']:.2f}  MCC={b_u['MCC']:+.4f}  "
          f"TP={cm['TP']} FP={cm['FP']} TN={cm['TN']} FN={cm['FN']}  "
          f"prec={cm['precision']:.3f}  rec={cm['recall']:.3f}")
    out["uncalibrated_sweep"] = {"best": b_u, "full_sweep": rows_u}

    for name, probs in calibs.items():
        print(f"\n=== {name} calibration, threshold sweep ===")
        b, rows = sweep(probs)
        cm = b["cm"]
        # decomposed: calibration alone at t=0.5
        p05 = (probs > 0.5).astype(int)
        m05 = float(matthews_corrcoef(y, p05))
        cm05 = confusion(y, p05)
        print(f"  calibration only (t=0.5): MCC={m05:+.4f}  "
              f"TP={cm05['TP']} FP={cm05['FP']} TN={cm05['TN']} FN={cm05['FN']}  "
              f"prec={cm05['precision']:.3f}  rec={cm05['recall']:.3f}")
        print(f"  + best threshold t={b['t']:.2f}:  MCC={b['MCC']:+.4f}  "
              f"TP={cm['TP']} FP={cm['FP']} TN={cm['TN']} FN={cm['FN']}  "
              f"prec={cm['precision']:.3f}  rec={cm['recall']:.3f}")
        print(f"  delta vs baseline: MCC {base_mcc:+.4f} -> {b['MCC']:+.4f}  "
              f"({b['MCC']-base_mcc:+.4f})")
        out["calibrations"][name] = {
            "calibration_only_t0.5": {"MCC": m05, **cm05},
            "best_with_threshold":   {"t": b["t"], "MCC": b["MCC"], **cm},
            "delta_MCC_vs_baseline": b["MCC"] - base_mcc,
            "full_sweep": rows,
        }

    # ---- pick winner; default isotonic if both ----
    pick = "isotonic" if "isotonic" in calibs else "platt"
    best_t = out["calibrations"][pick]["best_with_threshold"]["t"]
    final = calibs[pick]
    df["calibrated_prob"] = final
    df["pred_optimal"]    = (final > best_t).astype(int)
    df["calibration"]     = pick
    df["threshold"]       = best_t

    # ---- confident-FP reduction ----
    conf_fp_before = int(((df.prob > 0.9) & (df.y_true == 0)).sum())
    conf_fp_after  = int(((df.calibrated_prob > 0.9) & (df.y_true == 0)).sum())
    print(f"\n=== confident FP (prob > 0.9) reduction ({pick}) ===")
    print(f"  before: {conf_fp_before}  after: {conf_fp_after}  "
          f"reduction: {conf_fp_before - conf_fp_after}  "
          f"({100*(conf_fp_before-conf_fp_after)/max(conf_fp_before,1):.1f}%)")
    out["confident_fp_reduction"] = {
        "method": pick,
        "before": conf_fp_before, "after": conf_fp_after,
        "reduction": conf_fp_before - conf_fp_after,
    }

    # ---- FP/FN by family_frac bin, before vs after ----
    import pandas as _pd
    df["ff_bin"] = _pd.cut(df.family_frac_essential_leakfree,
        bins=[-0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["~0","0-0.2","0.2-0.4","0.4-0.6","0.6-0.8","0.8-1"])
    print(f"\n=== FP / FN by family_frac bin: before vs after ({pick}, t={best_t:.2f}) ===")
    print(f"{'bin':<10s} {'n':>7s} {'FP_b':>6s} {'FP_a':>6s} {'dFP':>6s}"
          f" {'FN_b':>6s} {'FN_a':>6s} {'dFN':>6s}")
    by_bin = {}
    for b in ["~0","0-0.2","0.2-0.4","0.4-0.6","0.6-0.8","0.8-1"]:
        sub = df[df.ff_bin == b]
        if len(sub) == 0: continue
        fp_b = int(((sub.prob > 0.5) & (sub.y_true == 0)).sum())
        fp_a = int(((sub.pred_optimal == 1) & (sub.y_true == 0)).sum())
        fn_b = int(((sub.prob <= 0.5) & (sub.y_true == 1)).sum())
        fn_a = int(((sub.pred_optimal == 0) & (sub.y_true == 1)).sum())
        print(f"  {b:<8s} {len(sub):>7d} {fp_b:>6d} {fp_a:>6d} {fp_a-fp_b:>+6d}"
              f" {fn_b:>6d} {fn_a:>6d} {fn_a-fn_b:>+6d}")
        by_bin[b] = {"n": len(sub), "FP_before": fp_b, "FP_after": fp_a,
                     "FN_before": fn_b, "FN_after": fn_a}
    out["bin_deltas"] = by_bin

    df.drop(columns=["ff_bin"]).to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

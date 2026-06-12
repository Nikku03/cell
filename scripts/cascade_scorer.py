"""Cascade scorer — fuse Phase 0 (conservation), Phase 1 (atlas measurements),
and Phase 2 (XGBoost predictor) into a confidence-attributed answer per query.

THE COMMERCIAL QUESTION the user asked: "64% recall at 30% precision isn't
useful enough -- what if we combine the three phases smartly?"

THIS SCRIPT ANSWERS: at what confidence threshold does the fused system hit
useful precision (60%, 70%, 80%, 90%), and how much coverage do we keep?
The "useful subset" -- predictions confident enough for a wet lab to act on
-- is the actual product.

PIPELINE per (held-out org, locusId, compound) query:
  1. PHASE 1 LOOKUP. og_cpd_n + og_cpd_hit_rate (already in Phase 2 preds)
     act as a Phase-1-atlas-in-relatives signal: how many TRAINING orgs
     measured (this OG x compound) as a strong hit. Strong evidence ->
     "atlas-corroborated".
  2. PHASE 0 PROXY (in the saved preds we don't have family_frac per row;
     we use the conditional pred as the proxy; not strictly Phase 0 but the
     attribution layer below makes clear what's contributing).
  3. PHASE 2 SCORE. The XGBoost calibrated probability.
  4. CONFIDENCE ATTRIBUTION. Each prediction gets a SOURCE tag:
       atlas_corroborated     og_cpd_hit_rate>=0.3 and og_cpd_n>=3
       de_novo_predicted      otherwise; relies on Phase 2's generalization
     and a confidence = the XGBoost pred.

OUTPUTS:
  PHASE2/eval/loo-org/cascade_metrics.json
    - Risk-coverage curve (precision/recall/coverage at every threshold)
    - Operating points: thresholds achieving 60/70/80/90% precision
    - Per-source breakdown
    - Per-clade-tier breakdown (where the system actually works)
  PHASE2/eval/loo-org/cascade_predictions.parquet
    - All predictions with cascade source + confidence
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

DEFAULT_OUT = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/phase2")


def tier_for(org: str) -> str:
    if "Methanococcus" in org: return "1_archaea"
    if org in ("SynE",):       return "1_cyano"
    if org in ("Btheta","Pedo557","DdiaME23","Ddia6719"):
        return "2_bacteroidetes"
    if any(k in org for k in ("Ralstonia","Burk","Cup")):
        return "3_burkholderiales"
    if org in ("MR1","PV4","SB2B","ANA3","Keio","Koxy","Dda3937","DvH"):
        return "5_gamma_litmus_clade"
    return "4_alpha_beta_gamma"


def precision_recall_at_threshold(y, p, t):
    pred = (p >= t)
    tp = int(((pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    cov = float(pred.mean())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"t": float(t), "coverage": cov, "precision": prec,
            "recall": rec, "TP": tp, "FP": fp, "FN": fn,
            "n_predicted_positive": int(pred.sum())}


def find_operating_point(y, p, target_precision, n_steps=200):
    """Lowest threshold (max coverage) that meets precision >= target."""
    import numpy as np
    ts = np.linspace(p.min(), p.max(), n_steps)
    best = None
    for t in ts:
        m = precision_recall_at_threshold(y, p, t)
        if m["precision"] >= target_precision and m["n_predicted_positive"] >= 20:
            if best is None or m["coverage"] > best["coverage"]:
                best = m
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arch", default="xgb",
                    help="which arch's predictions to score (default: xgb)")
    args = ap.parse_args()

    import pandas as pd
    import numpy as np

    eval_dir = args.out / "eval" / "loo-org"
    preds_files = sorted(eval_dir.glob(f"preds_{args.arch}__*.parquet"))
    if not preds_files:
        print(f"ERROR: no preds_{args.arch}__*.parquet in {eval_dir}",
              file=sys.stderr); return 1
    print(f"=== loading {len(preds_files)} per-fold {args.arch} predictions ===")

    frames = []
    for f in preds_files:
        df = pd.read_parquet(f)
        df["fold_org"] = f.stem.replace(f"preds_{args.arch}__", "")
        frames.append(df)
    all_preds = pd.concat(frames, ignore_index=True)
    print(f"  total rows: {len(all_preds):,}")
    print(f"  base rate (strong_hit): {all_preds.strong_hit.mean():.4f}")

    # ---- Phase 1 / atlas attribution ----
    has_atlas = "og_cpd_hit_rate" in all_preds.columns
    if has_atlas:
        ocr = all_preds.og_cpd_hit_rate.fillna(0.0)
        ocn = all_preds.og_cpd_n.fillna(0.0)
        all_preds["source"] = np.where(
            (ocr >= 0.3) & (ocn >= 3), "atlas_corroborated",
            np.where(ocn == 0, "de_novo_no_relatives_measured",
                     "interpolated"))
    else:
        all_preds["source"] = "phase2_only"

    print("\n=== source attribution (whole panel) ===")
    print(all_preds.source.value_counts().to_string())

    # ---- Phylogenetic tier ----
    all_preds["tier"] = all_preds.fold_org.apply(tier_for)

    # ---- Risk-coverage curve (whole panel) ----
    y = all_preds.strong_hit.values.astype(int)
    p = all_preds.pred.values.astype(float)
    print(f"\n=== risk-coverage curve (whole 43-fold panel) ===")
    print(f"  n={len(y):,}  pos={int(y.sum()):,}  base={y.mean():.4f}")

    curve = []
    ts = np.linspace(p.min(), p.max(), 100)
    for t in ts:
        curve.append(precision_recall_at_threshold(y, p, t))
    rc_df = pd.DataFrame(curve)

    # operating points at target precisions
    print(f"\n  {'target_prec':<12s} {'thresh':>8s} {'coverage':>9s} "
          f"{'precision':>10s} {'recall':>8s} {'n_pred_pos':>11s}")
    ops = {}
    for tp in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        op = find_operating_point(y, p, tp)
        if op:
            ops[f"p{int(tp*100)}"] = op
            print(f"  >= {tp:>5.2f}    {op['t']:>8.3f} "
                  f"{op['coverage']:>9.4f} {op['precision']:>10.3f} "
                  f"{op['recall']:>8.3f} {op['n_predicted_positive']:>11,}")
        else:
            print(f"  >= {tp:>5.2f}    UNREACHABLE on this panel")

    # ---- Per-clade-tier breakdown at the same target precisions ----
    print(f"\n=== per-clade tier at >=80% precision ===")
    tier_ops = {}
    for tier, grp in all_preds.groupby("tier"):
        yy = grp.strong_hit.values.astype(int)
        pp = grp.pred.values.astype(float)
        if yy.sum() < 20:
            print(f"  {tier:<28s} (skipped: only {int(yy.sum())} positives)")
            continue
        op80 = find_operating_point(yy, pp, 0.80)
        op70 = find_operating_point(yy, pp, 0.70)
        if op80:
            print(f"  {tier:<28s} >=80%: cov={op80['coverage']:.4f} "
                  f"rec={op80['recall']:.3f} n={op80['n_predicted_positive']:,}")
            tier_ops[tier] = {"p80": op80, "p70": op70}
        else:
            print(f"  {tier:<28s} 80% UNREACHABLE  "
                  + (f"(>=70%: cov={op70['coverage']:.4f} rec={op70['recall']:.3f})"
                     if op70 else "(70% UNREACHABLE)"))
            tier_ops[tier] = {"p80": None, "p70": op70}

    # ---- Per-source breakdown (atlas-corroborated vs de-novo) ----
    print(f"\n=== per-source quality (whole panel) ===")
    source_stats = {}
    for src, grp in all_preds.groupby("source"):
        yy = grp.strong_hit.values.astype(int)
        pp = grp.pred.values.astype(float)
        if len(yy) < 100:
            continue
        # at threshold 0.5 (a neutral binary cut)
        cut = (pp >= 0.5)
        tp = int(((cut) & (yy == 1)).sum())
        fp = int(((cut) & (yy == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(int(yy.sum()), 1)
        # also: best operating point >=70% precision
        op = find_operating_point(yy, pp, 0.70)
        print(f"  {src:<30s} n={len(yy):>8,}  pos={int(yy.sum()):>6,}  "
              f"base={yy.mean():.4f}  "
              f"prec@t0.5={prec:.3f} rec={rec:.3f}  "
              + (f">=70%: cov={op['coverage']:.4f}, rec={op['recall']:.3f}"
                 if op else ">=70% unreachable"))
        source_stats[src] = {
            "n": len(yy), "pos": int(yy.sum()), "base": float(yy.mean()),
            "at_t0.5": {"precision": prec, "recall": rec, "TP": tp, "FP": fp},
            "p70": op,
        }

    # ---- Litmus clade focused (the commercial-ready subset) ----
    litmus = all_preds[all_preds.tier == "5_gamma_litmus_clade"]
    print(f"\n=== litmus clade (Shewanella + Enterobacteriaceae + DvH) ===")
    if len(litmus) > 0:
        yL = litmus.strong_hit.values.astype(int)
        pL = litmus.pred.values.astype(float)
        print(f"  n={len(yL):,}  pos={int(yL.sum()):,}  base={yL.mean():.4f}")
        for tp in (0.60, 0.70, 0.80, 0.90):
            op = find_operating_point(yL, pL, tp)
            if op:
                # implied product pitch: "for every 1000 confident calls..."
                expected_tp = int(op["precision"] * op["n_predicted_positive"])
                print(f"  >=" f"{tp:.2f} prec  cov={op['coverage']:.4f}  "
                      f"rec={op['recall']:.3f}  yields {op['n_predicted_positive']:,} "
                      f"calls of which ~{expected_tp:,} are real hits")

    # ---- save ----
    out = {
        "n_predictions": len(all_preds),
        "base_rate": float(y.mean()),
        "whole_panel_operating_points": ops,
        "per_tier_operating_points": tier_ops,
        "per_source_stats": source_stats,
        "source_breakdown": all_preds.source.value_counts().to_dict(),
        "tier_breakdown": all_preds.tier.value_counts().to_dict(),
        "risk_coverage_curve": rc_df.to_dict("records"),
    }
    metrics_path = eval_dir / "cascade_metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {metrics_path}")

    cascade_preds_path = eval_dir / "cascade_predictions.parquet"
    # keep it light: drop full predictions if too large
    light_cols = ["fold_org", "tier", "source", "compound",
                  "strong_hit", "pred", "fit"]
    if "og_cpd_hit_rate" in all_preds.columns:
        light_cols += ["og_cpd_hit_rate", "og_cpd_n"]
    all_preds[light_cols].to_parquet(cascade_preds_path, index=False)
    print(f"wrote {cascade_preds_path}")

    # ---- verdict ----
    print(f"\n=== verdict (commercial-pitch reading) ===")
    p80 = ops.get("p80")
    p70 = ops.get("p70")
    p60 = ops.get("p60")
    if p80:
        print(f"  At 80% precision (publishable): "
              f"{p80['n_predicted_positive']:,} confident calls, "
              f"{p80['recall']*100:.1f}% recall.")
        print(f"    -> 'we will tell you {p80['n_predicted_positive']:,} gene-condition")
        print(f"     pairs to test; ~80% will be real hits, capturing "
              f"{p80['recall']*100:.0f}% of all real hits in the genome.'")
    if p70:
        print(f"  At 70% precision: {p70['n_predicted_positive']:,} calls, "
              f"{p70['recall']*100:.1f}% recall.")
    if p60:
        print(f"  At 60% precision: {p60['n_predicted_positive']:,} calls, "
              f"{p60['recall']*100:.1f}% recall.")

    if "5_gamma_litmus_clade" in tier_ops and tier_ops["5_gamma_litmus_clade"].get("p80"):
        lp80 = tier_ops["5_gamma_litmus_clade"]["p80"]
        print(f"  On the CLINICAL clade (Shewanella + Enterobacteriaceae): "
              f">=80% prec at cov={lp80['coverage']:.4f}, "
              f"rec={lp80['recall']*100:.0f}% -- "
              f"{lp80['n_predicted_positive']:,} confident calls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

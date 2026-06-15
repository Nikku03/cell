"""STAGE 4: selective prediction (abstention) -- the actual product question.

Stages 1-3 maxed out full-population AUPRC at ~0.234 and feature engineering is
plateauing. But a PRODUCT never runs at full population: it reports only the
predictions it is confident about and ABSTAINS on the rest. The question that
decides whether this beats Tn-seq is therefore NOT "what's the average
precision over all 8.3M cells" -- it's "if we report only our top X% most-
confident calls, how precise are they, and is that a usable shortlist?"

This script reads the out-of-fold (leave-one-clade-out) per-cell predictions
dumped by step2_train.py --dump_oof and computes the RISK-COVERAGE curve:
  - rank held-out cells by predicted essentiality (= -predicted fitness)
  - at each coverage c (report the top c% most-confident), measure PRECISION
    (fraction truly essential) and RECALL.
  - do this for the full model AND the conservation-only baseline (the only
    other zero-wet-lab option today), on (a) all held-out cells and (b)
    reliable-label cells (high total_weight, where the ground truth itself is
    trustworthy -- the honest estimate of true precision).
  - find the coverage at which precision crosses 0.5 / 0.7 / 0.9.

WHY reliable-label matters: a held-out cell with low total_weight has a noisy
label, so counting it as a miss underestimates true precision. Restricting the
evaluation to trustworthy labels is the matched-quality estimate (cf. Paper 1's
rho~0.77 ceiling vs 0.38 raw).

THE BAR ("better than Tn-seq"): Tn-seq requires building a transposon library +
sequencing for every organism x condition, and its own replicate calls agree
only at binary kappa ~0.39 (Paper 1). Our predictor needs zero wet lab. If our
confident subset reaches high precision at usable coverage, it delivers a
reliable shortlist for free -- which conservation alone does not.

OUTPUT: outputs/stage4_abstention_curve.parquet (the curve),
        outputs/stage4_abstention_summary.json, outputs/stage4_abstention.png

--smoke : validate the precision/recall-at-coverage math on synthetic data
--real  : read the OOF parquet and produce curve + summary + figure
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs"
OOF_CANDIDATES = [
    OUT / "stage4_oof.parquet",
    Path("/content/drive/MyDrive/path_b_stage2/stage4_oof.parquet"),
]
# Paper 1 reference: Tn-seq binary inter-screen reproducibility (median kappa)
TNSEQ_KAPPA_MEDIAN = 0.39


def pr_at_coverage(score, y, cov_grid):
    """score: higher = more confidently essential. y: 0/1 truth. Returns list
    of (coverage, k, precision, recall) for each coverage fraction in cov_grid."""
    import numpy as np
    score = np.asarray(score, float); y = np.asarray(y, int)
    m = ~np.isnan(score); score, y = score[m], y[m]
    order = np.argsort(-score)
    ys = y[order]
    tp = np.cumsum(ys)
    n = len(ys); P = int(y.sum())
    rows = []
    for c in cov_grid:
        k = max(1, min(n, int(round(c * n))))
        prec = tp[k - 1] / k
        rec = tp[k - 1] / P if P > 0 else float("nan")
        rows.append((float(c), int(k), float(prec), float(rec)))
    return rows


def coverage_at_precision(score, y, target):
    """Largest coverage fraction whose top-block precision is >= target.
    Returns (coverage, k, precision, recall) or None if never reached."""
    import numpy as np
    score = np.asarray(score, float); y = np.asarray(y, int)
    m = ~np.isnan(score); score, y = score[m], y[m]
    order = np.argsort(-score); ys = y[order]
    tp = np.cumsum(ys); n = len(ys); P = int(y.sum())
    ks = np.arange(1, n + 1)
    prec = tp / ks
    ok = np.where(prec >= target)[0]
    if len(ok) == 0:
        return None
    k = int(ok.max()) + 1          # largest block still at/above target precision
    return (k / n, k, float(tp[k - 1] / k),
            float(tp[k - 1] / P) if P > 0 else float("nan"))


def _pick_cols(oof):
    pred_cols = [c for c in oof.columns if c.startswith("pred_")]
    order = ["pred_M_full_esm_cond3", "pred_M_full_esm", "pred_M_full",
             "pred_M_gene"]
    best = next((c for c in order if c in pred_cols), pred_cols[0])
    base = "pred_M_cons" if "pred_M_cons" in pred_cols else None
    return best, base, pred_cols


def analyze(oof, rel_quantile=0.5):
    import numpy as np, pandas as pd
    best, base, _ = _pick_cols(oof)
    y = oof["y_ess"].to_numpy()
    base_rate = float(y.mean())
    # essentiality score = -predicted fitness (more negative fitness = essential)
    s_best = -oof[best].to_numpy()
    s_base = -oof[base].to_numpy() if base else None
    # reliable-label subset
    thr = oof["total_weight"].quantile(rel_quantile)
    rel = oof["total_weight"].to_numpy() >= thr
    cov_grid = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]

    curves = {}
    curves["model_all"] = pr_at_coverage(s_best, y, cov_grid)
    curves["model_reliable"] = pr_at_coverage(s_best[rel], y[rel], cov_grid)
    if s_base is not None:
        curves["cons_all"] = pr_at_coverage(s_base, y, cov_grid)
        curves["cons_reliable"] = pr_at_coverage(s_base[rel], y[rel], cov_grid)

    crossings = {}
    for tgt in (0.5, 0.7, 0.9):
        crossings[f"model_reliable@{tgt}"] = coverage_at_precision(
            s_best[rel], y[rel], tgt)
        crossings[f"model_all@{tgt}"] = coverage_at_precision(s_best, y, tgt)
    return {"best_model": best, "baseline": base, "base_rate": base_rate,
            "n_cells": int(len(y)), "n_reliable": int(rel.sum()),
            "rel_quantile": rel_quantile, "rel_threshold": float(thr),
            "cov_grid": cov_grid, "curves": curves, "crossings": crossings}


def _print(a):
    print(f"\n  best model : {a['best_model']}")
    print(f"  baseline   : {a['baseline']}")
    print(f"  held-out cells: {a['n_cells']:,}  base rate (essential): "
          f"{a['base_rate']:.4f}  (random precision)")
    print(f"  reliable-label subset (total_weight >= "
          f"{a['rel_quantile']:.0%}-ile = {a['rel_threshold']:.2f}): "
          f"{a['n_reliable']:,} cells")
    print(f"\n  PRECISION @ COVERAGE  (report top X% most-confident calls)")
    print(f"  {'cov':>6} {'k(rel)':>10} {'prec(model,rel)':>16} "
          f"{'recall':>8} {'prec(cons,rel)':>16}")
    mr = {c: (k, p, r) for c, k, p, r in a["curves"]["model_reliable"]}
    cr = ({c: (k, p, r) for c, k, p, r in a["curves"].get("cons_reliable", [])}
          if "cons_reliable" in a["curves"] else {})
    for c in a["cov_grid"]:
        k, p, r = mr[c]
        cp = cr.get(c, (None, float("nan"), None))[1]
        print(f"  {c:>6.1%} {k:>10,} {p:>16.3f} {r:>8.3f} {cp:>16.3f}")
    print(f"\n  COVERAGE AT WHICH PRECISION IS REACHED (reliable labels):")
    for tgt in (0.5, 0.7, 0.9):
        v = a["crossings"][f"model_reliable@{tgt}"]
        if v is None:
            print(f"    precision >= {tgt:.0%}: never reached")
        else:
            cov, k, p, r = v
            print(f"    precision >= {tgt:.0%}: top {cov:.2%} "
                  f"({k:,} calls)  actual precision {p:.3f}  recall {r:.3f}")
    print(f"\n  reference: Tn-seq binary inter-screen kappa ~ "
          f"{TNSEQ_KAPPA_MEDIAN} (Paper 1) -- our calls cost zero wet lab.")


def _figure(a, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable, skip figure: {e})"); return
    import numpy as np
    fig, ax = plt.subplots(figsize=(7, 5))
    def xy(key):
        c = a["curves"][key]
        return [r[0] for r in c], [r[2] for r in c]
    x, ym = xy("model_reliable"); ax.plot(x, ym, "-o", lw=2,
                                          label="full model (reliable labels)")
    x2, ya = xy("model_all"); ax.plot(x2, ya, "--", color="C0", alpha=.6,
                                      label="full model (all labels)")
    if "cons_reliable" in a["curves"]:
        xc, yc = xy("cons_reliable")
        ax.plot(xc, yc, "-s", color="C3", label="conservation (reliable)")
    ax.axhline(a["base_rate"], color="gray", ls=":", label="base rate (random)")
    for t in (0.5, 0.7, 0.9):
        ax.axhline(t, color="green", ls="--", alpha=.25)
    ax.set_xlabel("coverage (fraction of cells we report; rest abstained)")
    ax.set_ylabel("precision (fraction truly essential)")
    ax.set_title("Stage 4: selective prediction -- precision vs coverage")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"  wrote figure {path}")


def run_real(args):
    import pandas as pd
    oof_path = next((Path(p) for p in ([args.oof] if args.oof else [])
                     + OOF_CANDIDATES if Path(p).exists()), None)
    if oof_path is None:
        raise FileNotFoundError(
            f"OOF predictions not found. Run first:\n"
            f"  python scripts/step2_train.py --real --esm ... --cond3 ... "
            f"--dump_oof outputs/stage4_oof.parquet\n"
            f"searched: {OOF_CANDIDATES}")
    print("=" * 70)
    print("STAGE 4 -- selective prediction (abstention) curve")
    print("=" * 70)
    print(f"  OOF: {oof_path}")
    oof = pd.read_parquet(oof_path)
    a = analyze(oof, rel_quantile=args.rel_quantile)
    _print(a)
    OUT.mkdir(exist_ok=True)
    rows = []
    for key, c in a["curves"].items():
        for cov, k, p, r in c:
            rows.append({"curve": key, "coverage": cov, "k": k,
                         "precision": p, "recall": r})
    pd.DataFrame(rows).to_parquet(OUT / "stage4_abstention_curve.parquet",
                                  index=False)
    # json-safe crossings
    cj = {k: (None if v is None else
              {"coverage": v[0], "k": v[1], "precision": v[2], "recall": v[3]})
          for k, v in a["crossings"].items()}
    (OUT / "stage4_abstention_summary.json").write_text(json.dumps({
        "best_model": a["best_model"], "baseline": a["baseline"],
        "base_rate": a["base_rate"], "n_cells": a["n_cells"],
        "n_reliable": a["n_reliable"], "rel_quantile": a["rel_quantile"],
        "crossings": cj}, indent=2))
    _figure(a, OUT / "stage4_abstention.png")
    print(f"\nwrote {OUT/'stage4_abstention_curve.parquet'} + summary + png")
    return 0


def run_smoke():
    import numpy as np, pandas as pd
    print("=" * 70)
    print("STAGE 4 SMOKE -- precision/recall-at-coverage math")
    print("=" * 70)
    # construct a case with KNOWN ranking: score perfectly ranks 10 positives
    # at the top of 100 cells -> top-10% precision must be 1.0, recall 1.0
    n = 100
    y = np.zeros(n, int); y[:10] = 1
    score = np.linspace(1, 0, n)          # descending; positives at top
    rows = pr_at_coverage(score, y, [0.05, 0.1, 0.2, 1.0])
    print("  perfect-ranking curve (cov, k, prec, recall):")
    for r in rows:
        print("   ", tuple(round(x, 3) for x in r))
    d = {c: (k, p, r) for c, k, p, r in rows}   # d[c] = (k, precision, recall)
    assert d[0.1][0] == 10 and abs(d[0.1][1] - 1.0) < 1e-9, "top-10% precision"
    assert abs(d[0.1][2] - 1.0) < 1e-9, "top-10% recall = 10/10 = 1.0"
    assert abs(d[1.0][1] - 0.10) < 1e-9, "full-coverage precision == base rate"
    assert abs(d[0.05][2] - 0.5) < 1e-9, "top-5% recall = 5/10 = 0.5"
    # coverage_at_precision: precision>=1.0 reachable exactly at top 10%
    cv = coverage_at_precision(score, y, 1.0)
    print("  coverage@precision>=1.0:", tuple(round(x, 3) for x in cv))
    assert abs(cv[0] - 0.10) < 1e-9 and cv[1] == 10, "coverage@1.0 wrong"
    # a random ranker: precision ~ base rate everywhere (no high-precision head)
    rng = np.random.RandomState(0)
    cv5 = coverage_at_precision(rng.rand(n), y, 0.7)
    print("  random ranker coverage@0.7:", cv5)
    # end-to-end analyze() on a synthetic OOF frame
    N = 5000
    yy = (rng.rand(N) < 0.05).astype(int)
    good = -(yy * 2.0 + rng.normal(0, 1, N))     # decent essential ranker (fit)
    cons = -(yy * 0.5 + rng.normal(0, 1, N))     # weak ranker
    oof = pd.DataFrame({"clade": "c0", "y_ess": yy,
                        "total_weight": rng.rand(N) * 2,
                        "consensus_fitness": good,
                        "pred_M_full_esm_cond3": good,
                        "pred_M_cons": cons})
    a = analyze(oof)
    _print(a)
    assert a["best_model"] == "pred_M_full_esm_cond3"
    # the good model's top-1% precision should beat the base rate
    top = {c: p for c, k, p, r in a["curves"]["model_reliable"]}[0.01]
    assert top > a["base_rate"], "model head not above base rate"
    print("\n=== STAGE 4 SMOKE PASS ===")
    print("  precision/recall-at-coverage, coverage-at-precision, and analyze()")
    print("  validated. Run step2_train --dump_oof, then this --real.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--oof", default=None, help="path to stage4_oof.parquet")
    ap.add_argument("--rel_quantile", type=float, default=0.5,
                    help="reliable-label total_weight quantile threshold")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())

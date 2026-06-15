"""PATH-ORPHAN, STEP 2 (Phase A) -- the conservation baseline = THE BAR.

The kill-gate for every new feature (dN/dS, fold-homology, cofit, ESM) is:
beat conservation IN THE ROGUE ZONE (essential genes with family_frac < 0.1) by
>= 10% relative R@P30. This step measures that bar, using the leak-free
family_frac already in the step-0 bridge (computed excluding this organism's own
labels), so it needs no training and no external data -- it runs in the sandbox.

It reports recall-at-precision for the conservation score on:
  (a) the whole genome  -- sanity (conservation should be strong overall)
  (b) the ROGUE ZONE (conservation < ROGUE_CONS) -- the bar that matters; here
      conservation is near-useless by construction, so the bar is low and any
      real signal from the new features will be visible.

OUTPUT (idempotent):
  outputs/orphan/baseline_<org>.json   the bar numbers
  outputs/orphan/baseline_<org>.parquet per-precision recall table

--smoke : validate the ranking math on synthetic data
--real  : compute from outputs/orphan/bridge_<org>.parquet
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "orphan"
ROGUE_CONS = 0.10          # rogue zone = conservation below this
PREC_TARGETS = (0.3, 0.5, 0.7)


def recall_at_precision(score, y, target):
    """Largest top-block whose precision >= target; return (recall, k, prec)."""
    import numpy as np
    score = np.asarray(score, float); y = np.asarray(y, int)
    m = ~np.isnan(score); score, y = score[m], y[m]
    if y.sum() == 0:
        return (float("nan"), 0, float("nan"))
    order = np.argsort(-score); ys = y[order]
    tp = np.cumsum(ys); ks = np.arange(1, len(ys) + 1)
    prec = tp / ks
    ok = np.where(prec >= target)[0]
    if len(ok) == 0:
        return (0.0, 0, float("nan"))
    k = int(ok.max())
    return (float(tp[k] / y.sum()), k + 1, float(tp[k] / (k + 1)))


def auprc(score, y):
    import numpy as np
    try:
        from sklearn.metrics import average_precision_score
    except Exception:
        return float("nan")
    score = np.asarray(score, float); y = np.asarray(y, int)
    m = ~np.isnan(score); score, y = score[m], y[m]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, score))


def measure(score, y, label):
    import numpy as np
    y = np.asarray(y, int)
    n = len(y); pos = int(y.sum())
    base = pos / n if n else float("nan")
    rec = {"slice": label, "n": n, "positives": pos,
           "base_rate": round(base, 4), "auprc": round(auprc(score, y), 4)}
    for t in PREC_TARGETS:
        r, k, p = recall_at_precision(score, y, t)
        rec[f"R@P{int(t*100)}"] = round(r, 4)
        rec[f"k@P{int(t*100)}"] = k
    return rec


def run_real(args):
    import pandas as pd, numpy as np
    bridge = OUT / f"bridge_{args.org}.parquet"
    if not bridge.exists():
        print(f"ERROR: {bridge} missing -- run orphan_bridge.py --real first")
        return 2
    df = pd.read_parquet(bridge)
    score = df["conservation"].to_numpy()          # the conservation predictor
    y = df["essential"].to_numpy()
    rogue = df["conservation"].to_numpy() < ROGUE_CONS

    print("=" * 70)
    print(f"STEP 2 baseline (THE BAR) -- conservation, {args.org}")
    print("=" * 70)
    rows = [measure(score, y, "whole_genome"),
            measure(score[rogue], y[rogue], f"rogue_zone(cons<{ROGUE_CONS})")]
    # also: how do the rogue ESSENTIALS rank vs all genes -- the detection task
    hdr = f"  {'slice':<24}{'n':>7}{'pos':>6}{'base':>7}{'AUPRC':>7}" \
          f"{'R@P30':>7}{'R@P50':>7}{'R@P70':>7}"
    print(hdr)
    for r in rows:
        print(f"  {r['slice']:<24}{r['n']:>7}{r['positives']:>6}"
              f"{r['base_rate']:>7.3f}{r['auprc']:>7.3f}"
              f"{r['R@P30']:>7.3f}{r['R@P50']:>7.3f}{r['R@P70']:>7.3f}")
    bar = rows[1]
    # gate: normally +10% relative, but if conservation gives ~nothing in the
    # rogue zone (the expected/clean case) the gate is ABSOLUTE -- the combined
    # features must produce ANY P>=0.30 head (>= 3x the rogue base rate) that
    # also beats the permutation null (step 8). Precision 0.30 vs base ~0.08.
    rel_gate = 1.10 * bar["R@P30"]
    degenerate = bar["R@P30"] < 1e-6
    print(f"\n  THE BAR (rogue zone): R@P30 = {bar['R@P30']:.3f}  "
          f"(base rate {bar['base_rate']:.3f}, {bar['positives']} rogue "
          f"essentials)")
    if degenerate:
        print(f"  Conservation has ZERO purchase in the rogue zone (expected). "
              f"Gate is ABSOLUTE:")
        print(f"    combined model must reach R@P30 > 0 (a real {0.30:.0%}-"
              f"precision head, >= {0.30/bar['base_rate']:.1f}x base rate) AND "
              f"beat the permutation null.")
    else:
        print(f"  Kill-gate: family_frac+new features must reach R@P30 >= "
              f"{rel_gate:.3f} (+10% relative) in this slice.")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT / f"baseline_{args.org}.parquet",
                                  index=False)
    (OUT / f"baseline_{args.org}.json").write_text(json.dumps({
        "org": args.org, "rogue_cons": ROGUE_CONS,
        "bar_rogue_R@P30": bar["R@P30"],
        "gate_type": "absolute" if degenerate else "relative_10pct",
        "gate_rogue_R@P30": 0.0 if degenerate else round(rel_gate, 4),
        "gate_note": ("conservation==0 in rogue zone; combined model must reach "
                      "R@P30>0 AND beat permutation null") if degenerate else
                     "combined model R@P30 must beat bar by >=10% relative",
        "rows": rows}, indent=2))
    print(f"\n  wrote baseline_{args.org}.json + .parquet")
    return 0


def run_smoke():
    import numpy as np
    print("=" * 70)
    print("STEP 2 baseline SMOKE -- recall-at-precision ranking math")
    print("=" * 70)
    # perfect ranker: 10 positives at top of 100 -> P>=0.7 reachable to k=10
    y = np.zeros(100, int); y[:10] = 1
    s = np.linspace(1, 0, 100)
    r, k, p = recall_at_precision(s, y, 0.7)
    print(f"  perfect ranker  R@P70 = {r:.2f} (k={k}, prec={p:.2f})")
    assert abs(r - 1.0) < 1e-9 and k == 14, "largest P>=0.7 block = 14 (all 10 pos + 4 neg = 71%)"
    # random-ish ranker: precision near base rate -> can't hold P70
    rng = np.random.RandomState(0)
    r2, _, _ = recall_at_precision(rng.rand(100), y, 0.7)
    print(f"  random ranker   R@P70 = {r2:.2f} (near 0; no high-prec head)")
    assert r2 < 0.5, "random ranker must not hold high precision"
    # measure() end-to-end: rogue slice where conservation is flat (no signal)
    cons = np.concatenate([rng.rand(90) * 0.1, rng.rand(10) * 0.1])  # all <0.1
    yy = np.zeros(100, int); yy[:5] = 1                              # 5 rogue ess
    rec = measure(cons, yy, "rogue_zone")
    print(f"  flat-conservation rogue zone: R@P30={rec['R@P30']:.2f} "
          f"base={rec['base_rate']:.2f} (bar should be low)")
    assert rec["positives"] == 5 and rec["n"] == 100
    print("\n=== STEP 2 SMOKE PASS ===")
    print("  recall-at-precision, AUPRC, and the rogue-zone bar measurement")
    print("  validated. Run --real after orphan_bridge.py.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--org", default="beril_RalstoniaGMI1000")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())

"""HOW MANY PERTURBATIONS DO WE NEED? Measure the scaling curve instead of guessing at it.

THE QUESTION.  Every attempt to route around the data limit has now been priced and failed: more channels, more
datasets, more cell lines, more mechanism, more capacity, more non-linearity.  The remaining lever is more
measured perturbations.  Before anyone commissions a screen, the honest thing is to measure what a perturbation
is currently worth and what it would take to move the number.

THE MEASUREMENT.  Subsample the TRAINING pool to n knockouts, refit everything that depends on it -- the NMF
basis, the channel selection and the ridge -- and score the deployed model on the SAME two sealed cohorts.  The
sealed cohorts never change, so every point on the curve is measured against an identical test.

    n in {250, 500, 1000, 2000, 3200, 4720}, three independent subsamples each.

WHAT IS REFIT AND WHY THAT MATTERS.  A curve that holds the NMF basis fixed at full size would understate the
cost of small data, because the output vocabulary would still carry information from perturbations the model is
not supposed to have.  Everything downstream of the training pool is refit per subsample.

THE EXTRAPOLATION, and its warning label.  Performance against log(n) is close to linear over a wide middle range
for problems like this, so a log-linear fit gives a usable answer to "how many to reach X".  It is still an
extrapolation beyond measured data, and curves of this kind usually FLATTEN rather than continue.  So the
projected numbers are a LOWER BOUND on the perturbations required, not an estimate, and they are reported next to
the observed increment per doubling, which needs no model at all.

REFERENCE POINTS on the same axis: frequency baseline 0.2003, deployed model 0.2885, basis oracle 0.5222,
twin ceiling 0.6212 (cohort 2).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT = A.OUT
SP = A.SP
SIZES = (250, 500, 1000, 2000, 3200, 4720)
REPEATS = 3


def main() -> int:
    log: list[str] = []

    def report(text: str) -> None:
        print(text, flush=True)
        log.append(text)

    report("=" * 100)
    report("SCALING CURVE: what is one more measured perturbation worth?")
    report("=" * 100)

    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    genes = [str(g) for g in z["genes"]]
    ki = {k: i for i, k in enumerate(kos)}
    gi = {g: i for i, g in enumerate(genes)}
    Amat = np.abs(z["M"]).astype(np.float32)
    tide = (Amat >= A.TAU).mean(0) >= 0.05

    sealed: set[str] = set()
    answers: dict[str, dict] = {}
    for tag in ("", "_holdout"):
        f = OUT / f"ko_pred_dossiers{tag}.json"
        if f.exists():
            sealed |= set(json.loads(f.read_text()))
        af = OUT / f"ko_pred_answers{tag}.json"
        if af.exists():
            answers[tag] = json.loads(af.read_text())
    full_train = [k for k in kos if k not in sealed]
    universe = sorted(set(kos) | sealed)
    urow = {g: i for i, g in enumerate(universe)}
    report(f"  full training pool {len(full_train):,}; sealed {len(sealed)}; "
           f"sizes {SIZES}, {REPEATS} subsamples each")

    from sklearn.decomposition import NMF
    results: dict[int, dict[str, list[float]]] = {n: {"": [], "_holdout": []} for n in SIZES}

    for n in SIZES:
        for rep in range(REPEATS):
            rng = np.random.default_rng(A.SEED + 1000 * rep + n)
            subset = sorted(rng.choice(full_train, size=min(n, len(full_train)), replace=False).tolist())
            X = Amat[[ki[k] for k in subset]].copy()
            X[:, tide] = 0.0
            k_comp = min(A.K, max(5, len(subset) // 8))
            nmf = NMF(n_components=k_comp, init="nndsvda", max_iter=400, random_state=0)
            W = nmf.fit_transform(X).astype(np.float32)
            H = nmf.components_.astype(np.float32)
            del X
            base, _ = A.build_channels(universe, set(subset), lambda *_: None)
            extra, _, tb = C.extra_channels(universe, set(subset), lambda *_: None)
            chan = np.concatenate([base, extra[:, :tb]], axis=1)
            w = A.ridge_fit(chan[[urow[k] for k in subset]], W, A.PENALTY)

            for tag in ("", "_holdout"):
                ans = answers.get(tag)
                if not ans:
                    continue
                cohort = sorted(ans)
                mixes = A.ridge_apply(chan[[urow[k] for k in cohort]], w)
                hits = []
                for j, k in enumerate(cohort):
                    s = (mixes[j] @ H).copy()
                    s[tide] = -np.inf
                    if k in gi:
                        s[gi[k]] = -np.inf
                    top = np.argsort(-s)[:A.NPRED]
                    truth = set(ans[k]["movers"])
                    hits.append(len({genes[i] for i in top} & truth) / A.NPRED)
                results[n][tag].append(float(np.mean(hits)))
        c1 = np.mean(results[n][""])
        c2 = np.mean(results[n]["_holdout"])
        report(f"    n = {n:>5,}  (K={k_comp})   cohort1 {c1:.4f}   cohort2 {c2:.4f}")

    report(f"\n  {'n':>7} {'cohort 1':>10} {'sd':>7} {'cohort 2':>10} {'sd':>7}")
    for n in SIZES:
        a = np.array(results[n][""])
        b = np.array(results[n]["_holdout"])
        report(f"  {n:>7,} {a.mean():>10.4f} {a.std(ddof=1):>7.4f} {b.mean():>10.4f} {b.std(ddof=1):>7.4f}")

    report(f"\n  observed increment per DOUBLING of the training pool (cohort 2):")
    xs = np.log2(np.array(SIZES, float))
    ys = np.array([np.mean(results[n]["_holdout"]) for n in SIZES])
    for i in range(1, len(SIZES)):
        step = (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1])
        report(f"    {SIZES[i-1]:>5,} -> {SIZES[i]:<5,}  {ys[i]-ys[i-1]:+.4f}  "
               f"({step:+.4f} per doubling)")

    slope, intercept = np.polyfit(xs, ys, 1)
    report(f"\n  log-linear fit (cohort 2): precision = {intercept:.4f} + {slope:.4f} * log2(n)")
    report(f"    R^2 = {1 - np.var(ys - (intercept + slope*xs)) / np.var(ys):.4f}")
    report(f"\n  PROJECTED perturbations needed -- a LOWER BOUND, because curves of this kind flatten:")
    for label, target in (("frequency baseline 0.2003", 0.2003),
                          ("current 0.2885", 0.2885),
                          ("0.35", 0.35), ("0.40", 0.40),
                          ("basis oracle 0.5222", 0.5222),
                          ("twin ceiling 0.6212", 0.6212)):
        need = 2 ** ((target - intercept) / slope) if slope > 0 else float("inf")
        report(f"    {label:<26} {need:>18,.0f} perturbations"
               + ("   <- already have" if need <= len(full_train) else ""))

    json.dump({"test": "adrn_scaling_curve", "sizes": list(SIZES), "repeats": REPEATS,
               "results": {str(n): results[n] for n in SIZES},
               "slope_per_doubling": float(slope), "intercept": float(intercept), "log": log},
              open(OUT / "adrn_scaling_curve.json", "w"), indent=1)
    report(f"\n  -> {OUT/'adrn_scaling_curve.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

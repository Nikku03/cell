"""DEPTH TEST: is the model limited by ITS OWN QUALITY, or by how precisely the answer key was measured?

THE PREDICTION THIS FALSIFIES, which I committed to in UPDATES.md before running anything:

    "subsample cells per perturbation and both the split-half ceiling and the model score should fall
     together. If they do not, this reading is wrong."

WHY IT MATTERS.  The absolute ceiling on the sealed K562 task is 0.633 -- re-measuring the SAME knockout at the
same depth agrees with itself only 12.7 of 20. About 37% of the answer key is measurement noise, so every score
in this project has been computed against a target that is partly random. If depth is the binding constraint,
the lever is MORE CELLS PER PERTURBATION, not more perturbations -- and the scaling curve already said more
perturbations buys nothing (-0.0009 on its last doubling).

THE DESIGN.  At each depth d, for every (line, perturbation) group with at least 2d cells:

    ceiling(d)  two DISJOINT samples of d cells each -> top-20 agreement between them. This is how well the
                measurement agrees with itself at that depth, and it is the cap any model could reach.
    score(d)    a model trained at FULL depth, scored against an answer key built from d cells. The model is
                held constant so the only thing varying is the key's precision.

THE DISCRIMINATING QUANTITY is neither curve alone but their RATIO:

    ratio flat in d      the model tracks the measurement -- it is MEASUREMENT-LIMITED, and buying depth raises
                         the ceiling and the score together. Depth is the lever.
    ratio FALLS with d   the ceiling climbs faster than the model follows -- the MODEL is the limit, extra depth
                         exposes signal the model cannot use, and my reading in UPDATES.md is wrong.

PREDECLARED: judged on the ratio's trend across depths, with the Sweeper carrying depth x seed so a single lucky
subsample cannot decide it. Both curves are reported in full because the ratio alone hides which term moved.

DATA: the 1,000-perturbation cohort (421,610 cells, RPE1/Jurkat/HepG2). K562 has no per-cell data on this disk,
so this measures the principle on the lines that do, not the sealed K562 number itself.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
DEPTHS = (10, 20, 35, 60)
SEEDS = (0, 1, 2)
N_HOLDOUT, NPRED, PENALTIES = 200, 20, (1.0, 10.0, 100.0, 1000.0)


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("DEPTH TEST -- is the limit the model, or the precision of the answer key?")
    report("=" * 100)
    report("  PREDECLARED: ratio flat in depth => measurement-limited. ratio falls => model-limited.")

    z = np.load(SP / "sc_cohort_cells1000.npz", allow_pickle=True)
    X = z["X"].astype(np.float32)
    line = np.array([str(x) for x in z["line"]])
    pert = np.array([str(x) for x in z["pert"]])
    batch, ntc = z["batch"], z["is_ntc"]
    perts = sorted(set(pert[~ntc]))
    lines = sorted(set(line))
    report(f"  {X.shape[0]:,} cells, {len(perts)} perturbations, {X.shape[1]:,} genes")

    zk = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in zk["kos"]]
    universe = sorted(set(kos) | set(perts))
    base, _ = A.build_channels(universe, set(kos), report)
    extra, _, tb = C.extra_channels(universe, set(kos), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}

    ctrl = {}
    for ln in lines:
        m = (line == ln) & ntc
        if m.sum() >= 50:
            ctrl[ln] = X[m].mean(0)
    report(f"  control means for {len(ctrl)} lines")

    rng0 = np.random.default_rng(A.SEED)
    hold = set(np.array(perts)[rng0.choice(len(perts), N_HOLDOUT, replace=False)].tolist())
    train_p = [p for p in perts if p not in hold]

    # index cells by (line, pert) once
    cells = {}
    for ln in lines:
        for p in perts:
            idx = np.where((line == ln) & (pert == p) & ~ntc)[0]
            if len(idx):
                cells[(ln, p)] = idx
    sizes = np.array([len(v) for v in cells.values()])
    report(f"  {len(cells):,} (line,pert) groups; cells per group median {int(np.median(sizes))}, "
           f"range {sizes.min()}-{sizes.max()}")

    # FULL-DEPTH training targets, so the model is identical at every evaluated depth
    tr_keys = [(ln, p) for (ln, p) in cells if p in set(train_p) and ln in ctrl and len(cells[(ln, p)]) >= 30]
    Ytr = np.array([X[cells[k]].mean(0) - ctrl[k[0]] for k in tr_keys], np.float32)
    Ftr = np.concatenate([np.ones((len(tr_keys), 1), np.float32),
                          chan[[urow[p] for (_, p) in tr_keys]],
                          np.eye(len(lines), dtype=np.float32)[[lines.index(l) for (l, _) in tr_keys]]], 1)
    rv = np.random.default_rng(A.SEED + 1).permutation(len(tr_keys))
    nv = max(50, int(0.2 * len(rv)))
    best, bp = -1e18, PENALTIES[0]
    for pen in PENALTIES:
        w = A.ridge_fit(Ftr[rv[nv:]], Ytr[rv[nv:]], pen)
        e = -float(np.mean((A.ridge_apply(Ftr[rv[:nv]], w) - Ytr[rv[:nv]]) ** 2))
        if e > best:
            best, bp = e, pen
    W = A.ridge_fit(Ftr, Ytr, bp)
    report(f"  model fitted on {len(tr_keys):,} full-depth training groups (penalty {bp})")

    def top(v):
        return set(np.argsort(-np.abs(v))[:NPRED].tolist())

    sw = Sweeper("score - ceiling", axes={"depth": list(DEPTHS), "seed": list(SEEDS)})
    curves = {}
    for d in DEPTHS:
        for seed in SEEDS:
            rng = np.random.default_rng(A.SEED + 31 * seed + d)
            sc, ce = [], []
            for (ln, p), idx in cells.items():
                if p not in hold or ln not in ctrl or len(idx) < 2 * d:
                    continue
                pick = rng.choice(idx, 2 * d, replace=False)
                a, b = pick[:d], pick[d:]
                ya = X[a].mean(0) - ctrl[ln]
                yb = X[b].mean(0) - ctrl[ln]
                f = np.concatenate([[1.0], chan[urow[p]], np.eye(len(lines))[lines.index(ln)]])
                pred = A.ridge_apply(f[None, :].astype(np.float32), W)[0]
                sc.append(len(top(pred) & top(ya)) / float(NPRED))
                ce.append(len(top(yb) & top(ya)) / float(NPRED))
            if len(sc) < 30:
                continue
            sc, ce = np.array(sc), np.array(ce)
            sw.add({"depth": d, "seed": seed}, f"d{d}|s{seed}", sc - ce)
            curves[f"d={d}|seed={seed}"] = {"n": len(sc), "score": float(sc.mean()),
                                            "ceiling": float(ce.mean()),
                                            "ratio": float(sc.mean() / max(ce.mean(), 1e-9))}
            report(f"  depth {d:>3} seed{seed}: n={len(sc):>4}  score {sc.mean():.4f}  "
                   f"ceiling {ce.mean():.4f}  ratio {sc.mean()/max(ce.mean(),1e-9):.3f}")

    report(f"\n  {'depth':>6} {'score':>9} {'ceiling':>9} {'ratio':>8}")
    trend = []
    for d in DEPTHS:
        v = [c for k, c in curves.items() if k.startswith(f"d={d}|")]
        if not v:
            continue
        s, c, r = (np.mean([x["score"] for x in v]), np.mean([x["ceiling"] for x in v]),
                   np.mean([x["ratio"] for x in v]))
        trend.append((d, s, c, r))
        report(f"  {d:>6} {s:>9.4f} {c:>9.4f} {r:>8.3f}")

    report(sw.report())
    v = sw.verdict()
    if len(trend) >= 3:
        ds = np.log2([t[0] for t in trend])
        sl_s = float(np.polyfit(ds, [t[1] for t in trend], 1)[0])
        sl_c = float(np.polyfit(ds, [t[2] for t in trend], 1)[0])
        sl_r = float(np.polyfit(ds, [t[3] for t in trend], 1)[0])
        report(f"\n  per doubling of depth:  score {sl_s:+.4f}   ceiling {sl_c:+.4f}   ratio {sl_r:+.4f}")
        report("\n  READING")
        if abs(sl_r) < 0.02:
            report("  The ratio is FLAT in depth: the model tracks the measurement. MEASUREMENT-LIMITED --")
            report("  buying cells per perturbation raises the ceiling and the score together.")
        elif sl_r < 0:
            report("  The ratio FALLS with depth: the ceiling climbs faster than the model follows. The MODEL")
            report("  is the limit, extra depth exposes signal it cannot use, and my UPDATES.md reading is wrong.")
        else:
            report("  The ratio RISES with depth -- the model benefits more than the measurement does, which is")
            report("  not something either hypothesis predicted and needs explaining before it is used.")
    else:
        sl_s = sl_c = sl_r = None
        report("\n  too few depths resolved to fit a trend")

    json.dump({"test": "sc_depth", "depths": list(DEPTHS), "seeds": list(SEEDS),
               "curves": curves, "trend": [{"depth": t[0], "score": t[1], "ceiling": t[2], "ratio": t[3]}
                                           for t in trend],
               "slope_per_doubling": {"score": sl_s, "ceiling": sl_c, "ratio": sl_r},
               "sweep": v, "log": log}, open(OUT / "sc_depth.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'sc_depth.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""IS THE 0.846 DATA-LIMITED OR MODEL-LIMITED?  The scaling curve nobody ran.

THE TARGET.  Interface discrimination is at AUC 0.846 and the bar is 0.90. Three levers could close it -- more
complexes, more surface points per complex, a bigger network -- and picking one without evidence is how the
whole of today went wrong on the perturbation side.

THE FREE OBSERVATION.  `dmasif.main(subset=70)` caps at 70 complexes. **345 are already in the cache.** The cap
was for runtime, not for any principled reason, so 5x the data is sitting unused on disk.

WHAT THIS RUNS.  The identical dMaSIF -- same geodesic convolution, same APBS features, same GroupKFold by
complex so no complex appears in both train and test -- at increasing N, and reports AUC against N.

    AUC still rising at the largest N   -> DATA-limited. Fetch more complexes; the fetcher already exists and
                                           was verified to scale. 0.90 is a matter of running it.
    AUC flat                            -> MODEL-limited. More complexes will not help and the lever is points
                                           per complex, network depth, or exact MSMS surface instead of the
                                           marching-cubes approximation.

This is the same question the depth test asked about perturbation data, where the answer was "the model is the
limit". Here it has never been asked, and the two possible answers point at completely different work.

THE CONFOUND THE FIRST RUN HAD, and it is the same one the depth test had.  The first version drew folds
WITHIN each subset, so the test fold was ~12 complexes at N=35 and ~87 at N=260. The evaluation set changed
composition along with the training size, and if the complexes added at larger N happen to be easier the curve
rises for that reason rather than from data. It reported 0.7481 / 0.7783 / 0.8343 / 0.8813 and that number is
kept, but it does not isolate what it claims to.

Corrected here: a SINGLE held-out test set of N_HOLD complexes, never trained on at any N, with nested training
subsets drawn from the rest. Every point on the curve is scored on the identical complexes, so training size is
the only thing that varies.

PREDECLARED, before any number:
    a rising curve is only evidence if the LAST step still rises. A curve that flattens between the last two
    points is flat, however steep it looked earlier -- that is exactly the mistake the perturbation scaling
    curve was built to avoid.
    and the fixed test set must be LARGE enough that its AUC is stable: the per-seed spread across repeats is
    reported, and a step smaller than that spread is not a step.

COST.  Training is 5-fold x 60 epochs in the original. Folds and epochs are reduced here so the largest N is
reachable on CPU, and BOTH are held constant across N -- otherwise the curve would measure training budget
rather than data.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import dmasif as D

OUT = A.OUT
SIZES = (25, 50, 100, 200)       # TRAINING sizes; the test set is fixed and disjoint from all of them
N_HOLD = 60                      # held-out complexes, identical at every N
REPEATS, EPOCHS = 3, 30          # repeats replace folds: same held-out set, different init/shuffle


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("dMaSIF SCALING -- is interface discrimination data-limited or model-limited?")
    report("=" * 100)
    report("  PREDECLARED: judged on whether the LAST step still rises. A curve flat between the final two")
    report("  points is flat, however steep it looked earlier.")

    have = sorted(os.path.splitext(f)[0] for f in os.listdir(D.PDBDIR) if f.endswith(".pdb"))
    report(f"\n  {len(have):,} PDB files in the cache; the published 0.846 used 70")

    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(4)

    # ---- build every usable complex ONCE, then subsample for each N so the curve is nested ----
    report("\n  building surface-point clouds + APBS + KNN graphs (once, reused at every N)...")
    data = {}
    tb = time.time()
    for pdb in have:
        if len(data) >= max(SIZES) + N_HOLD:
            break
        try:
            pc = D.point_cloud(pdb)
            if pc is None or len(set(pc["ch"])) < 2:
                continue
            g = D.precompute_graph(pc)
            pos, neg = D.interface_point_pairs(pc)
            if len(pos) < 12:
                continue
            data[pdb] = {"pc": pc, "g": g, "pos": pos, "neg": neg}
            if len(data) % 25 == 0:
                report(f"    {len(data)} usable ({time.time()-tb:.0f}s)")
        except Exception:
            continue
    allp = sorted(data)
    report(f"    {len(allp)} usable complexes built in {time.time()-tb:.0f}s")
    FD = data[allp[0]]["pc"]["feat"].shape[1]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(allp))
    # THE FIXED HELD-OUT SET. Carved off first and never trained on at any N, so every point on the curve is
    # scored on the identical complexes. The first version drew folds inside each subset, which let the
    # evaluation set change composition along with the training size.
    HOLD = [allp[i] for i in order[:N_HOLD]]
    POOL = [allp[i] for i in order[N_HOLD:]]
    sizes = [n for n in SIZES if n <= len(POOL)]
    if len(POOL) > max(sizes):
        sizes.append(len(POOL))
    report(f"    {len(HOLD)} complexes held out and never trained on; {len(POOL)} available for training")
    report(f"    scaling curve at TRAINING N = {sizes}, every point scored on the same {len(HOLD)}")

    def run(n):
        """identical model, identical budget, identical test set -- only the TRAINING size changes"""
        aucs = []
        for rep in range(REPEATS):
            test = set(HOLD)
            train = list(POOL[:n])
            torch.manual_seed(rep)
            # dmasif defines Net inside main(); rebuild the identical architecture here
            class GeoConv(nn.Module):
                def __init__(s, din, dout):
                    super().__init__()
                    s.edge = nn.Sequential(nn.Linear(4 + din, dout), nn.ReLU())
                    s.logsig = nn.Parameter(torch.tensor(1.0))
                    s.self = nn.Sequential(nn.Linear(din + dout, dout), nn.ReLU())

                def forward(s, feat, idx, lc, dist):
                    fn = feat[idx]
                    e = s.edge(torch.cat([lc, dist.unsqueeze(-1), fn], -1))
                    w = torch.exp(-(dist ** 2) / (torch.exp(s.logsig) ** 2 + 1e-6)).unsqueeze(-1)
                    agg = (w * e).sum(1) / (w.sum(1) + 1e-6)
                    return s.self(torch.cat([feat, agg], -1))

            class Net2(nn.Module):
                def __init__(s, fd):
                    super().__init__()
                    s.c1 = GeoConv(fd, 32)
                    s.c2 = GeoConv(32, 32)
                    s.out = nn.Linear(32, 32)

                def forward(s, feat, idx, lc, dist):
                    h = s.c1(feat, idx, lc, dist)
                    h = s.c2(h, idx, lc, dist)
                    e = s.out(h)
                    return e / (e.norm(dim=1, keepdim=True) + 1e-9)

            model = Net2(FD)
            scale = nn.Parameter(torch.tensor(4.0))
            opt = torch.optim.Adam(list(model.parameters()) + [scale], lr=3e-3)
            lf = nn.BCEWithLogitsLoss()

            def embed(d):
                pc, g = d["pc"], d["g"]
                return model(torch.tensor(pc["feat"]), torch.tensor(g["idx"]),
                             torch.tensor(g["lc"]), torch.tensor(g["dist"]))

            np.random.seed(rep)
            for ep in range(EPOCHS):
                np.random.shuffle(train)
                model.train()
                for pdb in train:
                    d = data[pdb]
                    if not len(d["pos"]) or not len(d["neg"]):
                        continue
                    opt.zero_grad()
                    e = embed(d)
                    p = np.array(d["pos"])
                    q = np.array(d["neg"])
                    sp = (e[p[:, 0]] * e[p[:, 1]]).sum(1) * scale
                    sn = (e[q[:, 0]] * e[q[:, 1]]).sum(1) * scale
                    loss = lf(torch.cat([sp, sn]),
                              torch.cat([torch.ones(len(sp)), torch.zeros(len(sn))]))
                    loss.backward()
                    opt.step()
            model.eval()
            ys, ss = [], []
            with torch.no_grad():
                for pdb in test:
                    d = data[pdb]
                    if not len(d["pos"]) or not len(d["neg"]):
                        continue
                    e = embed(d)
                    p = np.array(d["pos"])
                    q = np.array(d["neg"])
                    sp = (e[p[:, 0]] * e[p[:, 1]]).sum(1) * scale
                    sn = (e[q[:, 0]] * e[q[:, 1]]).sum(1) * scale
                    ss += sp.tolist() + sn.tolist()
                    ys += [1] * len(sp) + [0] * len(sn)
            if len(set(ys)) > 1:
                aucs.append(float(roc_auc_score(ys, ss)))
        return float(np.mean(aucs)) if aucs else float("nan"), aucs

    curve = []
    for n in sizes:
        t1 = time.time()
        auc, per = run(n)
        sd = float(np.std(per)) if len(per) > 1 else float("nan")
        curve.append({"n": n, "auc": auc, "repeats": per, "sd": sd, "secs": time.time() - t1})
        report(f"    train N={n:>4}  AUC {auc:.4f} +/- {sd:.4f}  (repeats {[round(x,3) for x in per]})  "
               f"{time.time()-t1:.0f}s")

    report(f"\n  {'train N':>8} {'AUC':>9} {'sd':>8} {'delta':>9}")
    for i, c in enumerate(curve):
        d = "" if i == 0 else f"{c['auc']-curve[i-1]['auc']:+.4f}"
        report(f"  {c['n']:>8} {c['auc']:>9.4f} {c['sd']:>8.4f} {d:>9}")
    noise = float(np.mean([c["sd"] for c in curve]))
    report(f"  mean across-repeat sd {noise:.4f} -- a step smaller than this is not a step")

    report("\n  READING")
    if len(curve) >= 2:
        last = curve[-1]["auc"] - curve[-2]["auc"]
        # the noise floor is ENFORCED here, not merely printed: a step inside the across-repeat spread is
        # indistinguishable from re-running the same configuration twice.
        floor = max(0.01, noise)
        report(f"  last step: {last:+.4f} going from train N={curve[-2]['n']} to {curve[-1]['n']}, "
               f"against a floor of {floor:.4f} (max of the 0.01 threshold and the repeat spread)")
        if last > floor:
            report("  STILL RISING. Interface discrimination is DATA-limited: more complexes buy AUC, the")
            report("  fetcher already exists, and 0.90 is a matter of running it rather than redesigning.")
        elif last > 0.0:
            report("  Barely rising. More complexes buy little; reaching 0.90 this way would need a large")
            report("  multiple of the current set, and points-per-complex or network depth are better bets.")
        else:
            report("  FLAT or falling. MODEL-limited: more complexes will not reach 0.90. The levers are surface")
            report("  points per complex (currently ~1,500 subsampled), network depth (currently 2 layers), and")
            report("  exact MSMS surface instead of the marching-cubes approximation.")
    report(f"  target 0.90 | best here {max(c['auc'] for c in curve):.4f}")

    json.dump({"test": "dmasif_scaling", "sizes": sizes, "n_hold": N_HOLD, "repeats": REPEATS,
               "held_out": HOLD, "epochs": EPOCHS, "fixed_test_set": True,
               "n_available": len(have), "n_usable": len(allp), "curve": curve,
               "best_auc": max(c["auc"] for c in curve), "target": 0.90, "log": log},
              open(OUT / "dmasif_scaling.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'dmasif_scaling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

PREDECLARED, before any number:
    a rising curve is only evidence if the LAST step still rises. A curve that flattens between the last two
    points is flat, however steep it looked earlier -- that is exactly the mistake the perturbation scaling
    curve was built to avoid.

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
SIZES = (35, 70, 140, 260)
FOLDS, EPOCHS = 3, 30


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
        if len(data) >= max(SIZES):
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
    sizes = [n for n in SIZES if n <= len(allp)]
    if len(allp) > max(sizes):
        sizes.append(len(allp))
    report(f"    scaling curve at N = {sizes}")

    FD = data[allp[0]]["pc"]["feat"].shape[1]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(allp))

    def run(n):
        """identical model, identical budget, only N changes -- nested subsets so the curve is monotone in data"""
        pdbs = [allp[i] for i in order[:n]]
        folds = [set(np.array(pdbs)[np.arange(len(pdbs))[k::FOLDS]]) for k in range(FOLDS)]
        aucs = []
        for fi in range(FOLDS):
            test = folds[fi]
            train = [p for p in pdbs if p not in test]
            torch.manual_seed(0)
            net = D.__dict__.get("Net")
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
        curve.append({"n": n, "auc": auc, "folds": per, "secs": time.time() - t1})
        report(f"    N={n:>4}  AUC {auc:.4f}   (folds {[round(x,3) for x in per]})  {time.time()-t1:.0f}s")

    report(f"\n  {'N':>6} {'AUC':>9} {'delta':>9}")
    for i, c in enumerate(curve):
        d = "" if i == 0 else f"{c['auc']-curve[i-1]['auc']:+.4f}"
        report(f"  {c['n']:>6} {c['auc']:>9.4f} {d:>9}")

    report("\n  READING")
    if len(curve) >= 2:
        last = curve[-1]["auc"] - curve[-2]["auc"]
        report(f"  last step: {last:+.4f} going from N={curve[-2]['n']} to N={curve[-1]['n']}")
        if last > 0.01:
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

    json.dump({"test": "dmasif_scaling", "sizes": sizes, "folds": FOLDS, "epochs": EPOCHS,
               "n_available": len(have), "n_usable": len(allp), "curve": curve,
               "best_auc": max(c["auc"] for c in curve), "target": 0.90, "log": log},
              open(OUT / "dmasif_scaling.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'dmasif_scaling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

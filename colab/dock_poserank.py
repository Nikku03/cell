"""A POSE RANKER TRAINED ON THE RIGHT NEGATIVES -- decoy-pose contacts, which the FFT can now make in bulk.

WHAT THIS FIXES.  `dock_rescore` re-ranked FFT poses with the interface model and every arm came out 0/6 in the
top ten, no better than an UNTRAINED network. The reason was in the training data, not the architecture:

    the old model's negatives were FAR-APART pairs inside the true complex
    so it learned "are these two patches near each other?"  -- and at rank time EVERY pose has touching pairs
    it had never once seen a WRONG pose, so it had no basis for calling a contact spurious

Scoring the contacts of a given pose and choosing among poses are different problems. This trains for the
second one, with the negatives that problem actually needs:

    POSITIVES   contacting surface-point pairs at the NATIVE pose        (real interfaces)
    NEGATIVES   contacting surface-point pairs at DECOY poses            (ligand RMSD > DECOY_MIN A)

The decoys are not synthetic: they are poses the FFT search itself produced and scored highly, which makes them
exactly the confusions this ranker must resolve at test time. 2,400 per complex with known RMSD.

UNBOUND THROUGHOUT.  Each chain is surfaced from its OWN atoms with its own KNN graph. The complex surface
buries the interface and its graph reaches across it -- measured at 0.0843 AUC of difference in dock_rescore --
and neither is available when docking.

APBS IS OPTIONAL HERE, AND THAT IS A STATED DEVIATION.  The electrostatics feature needs the pdb2pqr and apbs
binaries. If they are absent the feature is filled with zeros and every arm loses it equally -- the comparison
between arms stays valid because they differ only in NEGATIVES, but the absolute numbers are not comparable to
dock_rescore's. Which mode ran is recorded in the output as `apbs_available`.

READOUTS.  Top-k is nearly powerless here: only 2-4 of 2,400 poses per complex are near-native, so chance at
top-10 is under 2% and six testable complexes cannot resolve a modest effect. Given equal weight alongside it:

    Spearman(score, RMSD)   over all ~2,400 poses. NEGATIVE is correct (better score, lower RMSD). With 2,400
                            poses this has real power where top-k has none.
    top-100 mean RMSD       versus the pose set's overall mean. Enrichment, not a needle-in-a-haystack hit.

ARMS AND CONTROLS:
    shape             the FFT's own score, 0/6 top-10 and 2/6 top-100
    old_negatives     the same architecture trained the OLD way (far-apart negatives) -- the thing being fixed
    new_negatives     trained on decoy-pose contacts. THE ARM UNDER TEST.
    UNTRAINED         random init, identical everything else
    chance            exact 1-(1-f)^k from each pose set's known near-native fraction

PREDECLARED, before any number:
    new beats old AND untrained on Spearman AND top-k   -> the negatives were the problem and this fixes it.
    new beats old on Spearman but not top-k             -> real signal, too weak to surface a 2-in-2400 needle.
                                                           Honest partial result: the lever is sampling, not score.
    new ~ old ~ untrained                               -> the negatives were NOT the problem, and the failure
                                                           is the representation or the task itself.

TRAIN/TEST.  Poses for training come only from TRAIN complexes; the 12 docking test complexes are never used to
fit anything. Homology between splits is NOT controlled (subtilisin/eglin variants recur), so the learned arms
are flattered -- which is why UNTRAINED is the control that matters: leakage cannot help unfitted weights.
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
import dock_fft as DF
import dock_rescore as RS
import flex_physics as fp

OUT = A.OUT
CACHE = Path(fp.PDBDIR).parent / "surf_cache"
N_TRAIN = 40
EPOCHS = 20
CONTACT = 4.5
DECOY_MIN = 10.0            # a pose this far from native is an unambiguous negative
DECOYS_PER_COMPLEX = 24
MIN_PAIRS = 10
TOPK = (1, 10, 100)
NEAR_NATIVE = DF.NEAR_NATIVE


def cached_surface(tag):
    """point cloud + KNN graph for one chain, cached to disk. The previous run spent 924s rebuilding these."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{tag}.npz"
    if f.exists():
        z = np.load(f)
        return ({"pts": z["pts"], "nrm": z["nrm"], "feat": z["feat"], "ch": z["ch"]},
                {"idx": z["idx"], "lc": z["lc"], "dist": z["dist"]})
    try:
        pc = D.point_cloud(tag)
    except Exception:
        return None
    if pc is None or len(pc["pts"]) < 100:
        return None
    g = D.precompute_graph(pc)
    np.savez_compressed(f, pts=pc["pts"], nrm=pc["nrm"], feat=pc["feat"], ch=pc["ch"],
                        idx=g["idx"], lc=g["lc"], dist=g["dist"])
    return pc, g


def fft_poses(p):
    """the identical blind pose set dock_fft produces, with ligand RMSD attached to each"""
    rots = DF.rotations(DF.N_ROT)
    centre, L, lcen = p["centre"], p["L"], p["lcen"]
    Lc = L - lcen
    FA = p["F_skin"] - p["_cw"] * p["F_core"]
    out = []
    for ri, Rm in enumerate(rots):
        g = DF.rasterise(Lc @ Rm.T + centre, centre, p["offs_L"])
        if g is None:
            continue
        sc = np.fft.irfftn(FA * np.conj(np.fft.rfftn(g.astype(np.float32))), p["shape"])
        for pk in DF.nms_peaks(sc, DF.PEAKS_PER_ROT, DF.NMS_CELLS):
            if sc[pk] <= 0:
                continue
            sh = DF.shift_of(pk)
            rm = float(np.sqrt((((Lc @ Rm.T + centre + sh) - L) ** 2).sum(1).mean()))
            out.append((float(sc[pk]), ri, sh, rm))
    return out, rots


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    import torch
    import torch.nn as nn
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    torch.set_num_threads(4)

    import shutil
    apbs_ok = bool(shutil.which("apbs") and shutil.which("pdb2pqr"))

    report("=" * 100)
    report("A POSE RANKER TRAINED ON DECOY-POSE CONTACTS -- the negatives the last model never saw")
    report("=" * 100)
    report(f"  positives = contacts at the NATIVE pose | negatives = contacts at poses > {DECOY_MIN:.0f} A RMSD,")
    report(f"  from the FFT search's own output ({DECOYS_PER_COMPLEX}/complex), so they are the real confusions")
    report(f"  APBS/pdb2pqr available: {apbs_ok}"
           + ("" if apbs_ok else "  <- electrostatics feature is ZERO for every arm; arms still differ only"
                                 " in negatives, but absolute values are not comparable to dock_rescore"))
    report("  PREDECLARED: new beats old AND untrained on Spearman AND top-k => the negatives were the problem.")
    report("  new beats old on Spearman only => real but too weak for a 2-in-2400 needle. new ~ old ~ untrained")
    report("  => the negatives were NOT the problem and the failure is the representation or the task.")

    dockj = json.load(open(OUT / "dock_fft.json"))
    CW = dockj["clash_w_chosen"]
    test_pdbs = [r["pdb"] for r in dockj["complexes"]]
    have = sorted(f[:-4] for f in os.listdir(fp.PDBDIR) if f.endswith(".pdb") and "__" not in f)
    report(f"\n  {len(have):,} structures on disk | docking test set: {test_pdbs}")

    report(f"\n  building/loading UNBOUND surfaces (cached to {CACHE.name}/)...")
    pool, tb = {}, time.time()
    for pdb in test_pdbs + [q for q in have if q not in test_pdbs]:
        if len(pool) >= N_TRAIN + len(test_pdbs):
            break
        p = DF.prepare(pdb)
        if p is None or not p["sane"]:
            continue
        chs, ok = {}, True
        for ch in (p["ca"], p["cb"]):
            tag = RS.chain_pdb(pdb, ch)
            r = cached_surface(tag) if tag else None
            if r is None:
                ok = False
                break
            chs[ch] = {"pc": r[0], "g": r[1]}
        if not ok:
            continue
        p["_cw"] = CW
        pool[pdb] = {"prep": p, "ub": chs}
        if len(pool) % 10 == 0:
            report(f"    {len(pool)} ready ({time.time()-tb:.0f}s)")
    tst = [q for q in test_pdbs if q in pool]
    tr = [q for q in sorted(pool) if q not in test_pdbs][:N_TRAIN]
    report(f"    {len(pool)} complexes in {time.time()-tb:.0f}s | train {len(tr)} | test {len(tst)}")
    if len(tr) < 10 or len(tst) < 5:
        report("  not enough complexes")
        return 1

    report("\n  generating FFT pose sets (train + test)...")
    tp = time.time()
    for i, pdb in enumerate(tr + tst):
        c = pool[pdb]
        c["poses"], c["rots"] = fft_poses(c["prep"])
        if (i + 1) % 10 == 0:
            report(f"    {i+1}/{len(tr)+len(tst)} pose sets ({time.time()-tp:.0f}s)")
    report(f"    done in {time.time()-tp:.0f}s")

    # ---------------- architecture: identical to dock_rescore, so ONLY the data differs ----------------
    class GeoConv(nn.Module):
        def __init__(s, din, dout):
            super().__init__()
            s.edge = nn.Sequential(nn.Linear(4 + din, dout), nn.ReLU())
            s.logsig = nn.Parameter(torch.tensor(1.0))
            s.self = nn.Sequential(nn.Linear(din + dout, dout), nn.ReLU())

        def forward(s, feat, idx, lc, dist):
            e = s.edge(torch.cat([lc, dist.unsqueeze(-1), feat[idx]], -1))
            w = torch.exp(-(dist ** 2) / (torch.exp(s.logsig) ** 2 + 1e-6)).unsqueeze(-1)
            return s.self(torch.cat([feat, (w * e).sum(1) / (w.sum(1) + 1e-6)], -1))

    class Net(nn.Module):
        def __init__(s, fd):
            super().__init__()
            s.c1, s.c2, s.out = GeoConv(fd, 32), GeoConv(32, 32), nn.Linear(32, 32)

        def forward(s, feat, idx, lc, dist):
            e = s.out(s.c2(s.c1(feat, idx, lc, dist), idx, lc, dist))
            return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    def emb(model, d):
        pc, g = d["pc"], d["g"]
        return model(torch.tensor(pc["feat"]), torch.tensor(g["idx"]),
                     torch.tensor(g["lc"]), torch.tensor(g["dist"]))

    def contacts(tree, lig_pts):
        nb = tree.query_ball_point(lig_pts, CONTACT)
        ii, jj = [], []
        for j, lst in enumerate(nb):
            for i in lst:
                ii.append(i); jj.append(j)
        return np.array(ii, int), np.array(jj, int)

    report(f"\n  extracting contact pairs: native (positive) + {DECOYS_PER_COMPLEX} decoys > {DECOY_MIN:.0f} A")
    rng = np.random.default_rng(0)
    for pdb in tr + tst:
        c = pool[pdb]; p = c["prep"]
        Pr = c["ub"][p["ca"]]["pc"]["pts"]
        Pl0 = c["ub"][p["cb"]]["pc"]["pts"] - p["lcen"]
        tree = cKDTree(Pr)
        c["_tree"], c["_Pl0"] = tree, Pl0
        c["pos_pairs"] = contacts(tree, Pl0 + p["lcen"])
        dec = [q for q in c["poses"] if q[3] > DECOY_MIN]
        rng.shuffle(dec)
        negs = []
        for _v, ri, sh, _rm in dec[:DECOYS_PER_COMPLEX]:
            ii, jj = contacts(tree, Pl0 @ c["rots"][ri].T + p["centre"] + sh)
            if len(ii) >= MIN_PAIRS:
                negs.append((ii, jj))
        c["neg_pairs"] = negs
    npos = sum(len(pool[q]["pos_pairs"][0]) for q in tr)
    nneg = sum(sum(len(a) for a, _ in pool[q]["neg_pairs"]) for q in tr)
    report(f"    training pairs: {npos:,} native-pose contacts / {nneg:,} decoy-pose contacts")
    assert npos > 500 and nneg > 500, "not enough training pairs"

    FD = pool[tr[0]]["ub"][pool[tr[0]]["prep"]["ca"]]["pc"]["feat"].shape[1]

    def train(mode, seed=0):
        """'new' = decoy-pose negatives; 'old' = far-apart-pair negatives (the previous recipe)"""
        torch.manual_seed(seed)
        model = Net(FD)
        scale = nn.Parameter(torch.tensor(4.0))
        opt = torch.optim.Adam(list(model.parameters()) + [scale], lr=3e-3)
        lf = nn.BCEWithLogitsLoss()
        order = list(tr)
        for _ in range(EPOCHS):
            np.random.shuffle(order)
            model.train()
            for pdb in order:
                c = pool[pdb]; p = c["prep"]
                pi, pj = c["pos_pairs"]
                if len(pi) < MIN_PAIRS:
                    continue
                if mode == "new":
                    if not c["neg_pairs"]:
                        continue
                    ni, nj = c["neg_pairs"][np.random.randint(len(c["neg_pairs"]))]
                else:
                    _, uneg = RS.cross_pairs(c["ub"][p["ca"]]["pc"]["pts"], c["ub"][p["cb"]]["pc"]["pts"])
                    if len(uneg) < MIN_PAIRS:
                        continue
                    ni = np.array([a for a, _ in uneg]); nj = np.array([b for _, b in uneg])
                opt.zero_grad()
                ea = emb(model, c["ub"][p["ca"]]); eb = emb(model, c["ub"][p["cb"]])
                sp = (ea[pi] * eb[pj]).sum(1) * scale
                sn = (ea[ni] * eb[nj]).sum(1) * scale
                loss = lf(torch.cat([sp, sn]),
                          torch.cat([torch.ones(len(sp)), torch.zeros(len(sn))]))
                loss.backward()
                opt.step()
        return model, scale

    report(f"\n  TRAINING on {len(tr)} complexes, {EPOCHS} epochs, identical architecture -- only NEGATIVES differ")
    t1 = time.time(); mnew, snew = train("new"); report(f"    new_negatives (decoy-pose)  {time.time()-t1:.0f}s")
    t1 = time.time(); mold, sold = train("old"); report(f"    old_negatives (far-apart)   {time.time()-t1:.0f}s")
    torch.manual_seed(1234); mun, sun = Net(FD), torch.tensor(4.0)

    report(f"\n  RANKING {len(tst)} held-out pose sets")
    rows = []
    for pdb in tst:
        c = pool[pdb]; p = c["prep"]
        poses, rots, tree, Pl0 = c["poses"], c["rots"], c["_tree"], c["_Pl0"]
        if not poses:
            continue
        with torch.no_grad():
            E = {k: (emb(m, c["ub"][p["ca"]]).numpy(), emb(m, c["ub"][p["cb"]]).numpy())
                 for k, m in (("new", mnew), ("old", mold), ("untrained", mun))}
        S = {k: [] for k in ("shape", "new", "old", "untrained")}
        rmsd = []
        for v, ri, sh, rm in poses:
            ii, jj = contacts(tree, Pl0 @ rots[ri].T + p["centre"] + sh)
            S["shape"].append(v)
            for k, sc_ in (("new", snew), ("old", sold), ("untrained", sun)):
                if len(ii) < MIN_PAIRS:
                    S[k].append(-1e9)
                else:
                    ea, eb = E[k]
                    S[k].append(float((1 / (1 + np.exp(-(ea[ii] * eb[jj]).sum(1) * float(sc_)))).mean()))
            rmsd.append(rm)
        rm = np.array(rmsd)
        frac = float((rm <= NEAR_NATIVE).mean())
        res = {}
        for k, v in S.items():
            v = np.array(v)
            o = np.argsort(-v)
            live = float(np.std(v)) > 1e-12
            res[k] = {"hits": {str(t): bool((rm[o][:t] <= NEAR_NATIVE).any()) for t in TOPK},
                      "spearman": float(spearmanr(v, rm).correlation) if live else None,
                      "top100_mean_rmsd": float(rm[o][:100].mean()), "live": live}
        rows.append({"pdb": pdb, "n_poses": len(rm), "near_native_frac": frac,
                     "best_rmsd": float(rm.min()), "mean_rmsd": float(rm.mean()), "arms": res})
        report(f"    {pdb}: {len(rm)} poses, frac {frac:.4f} | Spearman(score,RMSD) "
               + "  ".join(f"{k} {res[k]['spearman']:+.3f}" if res[k]["spearman"] is not None else f"{k} DEAD"
                           for k in ("shape", "new", "old", "untrained")))

    if not rows:
        report("\n  nothing ranked")
        return 1
    dead = sorted({k for r in rows for k in r["arms"] if not r["arms"][k]["live"]})
    if dead:
        report(f"\n    LIVENESS: {dead} constant in at least one complex -- those rows mean nothing.")
    tt = [r for r in rows if r["near_native_frac"] > 0]
    report(f"\n  {len(rows)} ranked | {len(tt)} have a near-native pose at all: {[r['pdb'] for r in tt]}")

    setmean = float(np.mean([r["mean_rmsd"] for r in rows]))
    report(f"\n  {'arm':<14}{'top-1':>8}{'top-10':>8}{'top-100':>9}{'Spearman':>11}{'top100 RMSD':>13}{'set mean':>10}")
    tab = {}
    for k in ("shape", "new", "old", "untrained"):
        sp = [r["arms"][k]["spearman"] for r in rows if r["arms"][k]["spearman"] is not None]
        tab[k] = {"top": {str(t): sum(r["arms"][k]["hits"][str(t)] for r in tt) for t in TOPK},
                  "spearman_median": float(np.median(sp)) if sp else None,
                  "top100_mean_rmsd": float(np.mean([r["arms"][k]["top100_mean_rmsd"] for r in rows]))}
        report(f"  {k:<14}" + "".join(f"{tab[k]['top'][str(t)]:>8}" for t in TOPK)
               + f"{tab[k]['spearman_median']:>11.3f}{tab[k]['top100_mean_rmsd']:>13.1f}{setmean:>10.1f}")
    ch = {str(t): float(np.mean([1 - (1 - r["near_native_frac"]) ** t for r in tt])) * len(tt) for t in TOPK}
    report(f"  {'chance':<14}" + "".join(f"{ch[str(t)]:>8.2f}" for t in TOPK)
           + f"{0.0:>11.3f}{setmean:>13.1f}{setmean:>10.1f}")
    report(f"  (top-k denominator is the {len(tt)} complexes where a near-native pose exists; "
           f"Spearman uses all {len(rows)})")

    report("\n  READING")
    sn_, so_, su_ = (tab["new"]["spearman_median"], tab["old"]["spearman_median"],
                     tab["untrained"]["spearman_median"])
    hn, ho, hu = tab["new"]["top"]["10"], tab["old"]["top"]["10"], tab["untrained"]["top"]["10"]
    better_sp = sn_ < so_ - 0.02 and sn_ < su_ - 0.02
    if better_sp and hn > max(ho, hu):
        report(f"  THE NEGATIVES WERE THE PROBLEM. Decoy-pose training moves Spearman to {sn_:+.3f} against")
        report(f"  {so_:+.3f} for the old recipe and {su_:+.3f} untrained, and lifts top-10 to {hn}/{len(tt)}.")
    elif better_sp:
        report(f"  REAL BUT WEAK. Spearman {sn_:+.3f} vs old {so_:+.3f} vs untrained {su_:+.3f}: the decoy")
        report(f"  negatives genuinely teach pose quality. But top-10 is {hn}/{len(tt)} -- with 2-4 near-native")
        report("  poses per 2,400 that is nowhere near enough to surface the needle. The lever is sampling")
        report("  density, not the score.")
    else:
        report(f"  THE NEGATIVES WERE NOT THE PROBLEM. Spearman {sn_:+.3f} vs old {so_:+.3f} vs untrained")
        report(f"  {su_:+.3f} -- retraining on exactly the confusions the search produces changes nothing.")
        report("  The limit is the surface representation or the task, not the choice of negatives.")

    json.dump({"test": "dock_poserank", "apbs_available": apbs_ok, "n_train": len(tr), "epochs": EPOCHS,
               "decoy_min_A": DECOY_MIN, "decoys_per_complex": DECOYS_PER_COMPLEX,
               "n_pos_pairs": npos, "n_neg_pairs": nneg, "arms": tab, "chance": ch,
               "complexes": rows, "n_testable": len(tt), "log": log},
              open(OUT / "dock_poserank.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'dock_poserank.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

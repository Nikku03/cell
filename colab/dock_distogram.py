"""DISTOGRAM OF EVERY RANKING ARM: does a summary statistic hide a local signal?

WHY THIS IS NOT JUST A PLOT.  Every arm so far has been judged by Spearman(score, RMSD) over the whole pose set
and by top-k hits. Both can miss the same thing:

    Spearman is GLOBAL and MONOTONIC. An arm that gives near-native poses a distinctive score but behaves
    non-monotonically over the rest of the range scores ~0 -- while being exactly the discriminator wanted.
    top-k needs the correct pose to out-rank 2,400 others, which 2-4 near-native poses almost never do.

So this asks a targeted question instead: **can the arm separate near-native poses from decoys at all**, as a
distribution rather than a correlation. Same pose sets, same scores, different question.

    AUC_far     near-native (<= 5 A) vs FAR decoys (> 10 A). The easy separation.
    AUC_close   near-native (<= 5 A) vs CLOSE-WRONG (5-10 A). The hard one, and the one that matters: a docking
                funnel needs to distinguish right from nearly-right, not right from absurd.
    distogram   z-scored value per RMSD bin, so the SHAPE of the relationship is visible rather than compressed
                into one number. A U-shape, a step, or a plateau all read as Spearman ~0.

ARMS, every one previously measured, recomputed on identical pose sets so the distograms explain the recorded
numbers rather than describing some other run:

    shape                      the FFT score              (recorded Spearman -0.059)
    global descriptors         elec/phobic complementarity, contiguity, bsa, packing  (recorded ~0)
    sphere energies            vdw/elec/induc/desolv x sum/median_nz/frac_fav          (best recorded -0.142)

BUDGET.  600 rotations, 4 peaks -- IDENTICAL to every previous ranking run, deliberately. The coverage sweep
showed 2,400 rotations is needed for 12/12 complexes to contain a correct pose, but using it here would produce
distograms that explain a run nobody did. Coverage at this budget is 6/12, so the near-native analyses are
restricted to those 6 and that is stated in every table rather than averaged away.

PREDECLARED, before any number:
    an arm with AUC_close >= 0.65 but Spearman ~0     -> the summary statistic was hiding a usable local
                                                         signal, and the arm should be re-read as a
                                                         near-native detector rather than a global ranker.
    AUC_close ~ 0.5 for every arm                     -> no arm distinguishes right from nearly-right, and the
                                                         nulls stand as recorded. The distogram then explains
                                                         WHY rather than rescuing anything.
    AUC_far high but AUC_close ~ 0.5                  -> the arms only reject absurd poses, which the FFT clash
                                                         filter already does. That would explain every result
                                                         to date in one sentence.

Per-pose values are written to an npz so the figure can be rebuilt without recomputing 22 minutes of energies.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import dock_energy as DE
import dock_fft as DF
import dock_global as DG
import dock_poserank as PRK
import dock_rescore as RS
import flex_physics as fp
import nexus_pairs as NP

OUT = A.OUT
NPZ = Path(fp.PDBDIR).parent / "distogram_poses.npz"
NEAR, FAR = 5.0, 10.0
BINS = [0, 5, 10, 15, 20, 30, 45, 60, 1e9]


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    report("=" * 100)
    report("DISTOGRAM OF EVERY RANKING ARM -- does a summary statistic hide a local signal?")
    report("=" * 100)
    report("  Spearman is global and monotonic; top-k needs the answer to beat 2,400 rivals. Both can miss an")
    report("  arm that separates near-native from decoys non-monotonically. This asks that directly, as AUC.")
    report(f"  AUC_far   = near-native (<={NEAR:.0f} A) vs far (>{FAR:.0f} A)   -- the easy separation")
    report(f"  AUC_close = near-native (<={NEAR:.0f} A) vs close-wrong ({NEAR:.0f}-{FAR:.0f} A) -- the one a "
           f"funnel needs")
    report("  PREDECLARED: AUC_close >= 0.65 with Spearman ~0 => the statistic hid a usable detector.")
    report("  AUC_close ~ 0.5 everywhere => the nulls stand and the distogram explains why.")
    report("  AUC_far high but AUC_close ~ 0.5 => the arms only reject absurd poses, which the clash filter")
    report("  already does -- that would explain every result to date in one sentence.")

    dockj = json.load(open(OUT / "dock_fft.json"))
    CW = dockj["clash_w_chosen"]
    test_pdbs = [r["pdb"] for r in dockj["complexes"]]
    report(f"\n  600 rotations x 4 peaks, IDENTICAL to every previous ranking run | {len(test_pdbs)} complexes")

    store, rows = {}, []
    for pdb in test_pdbs:
        p = DF.prepare(pdb)
        if p is None or not p["sane"]:
            continue
        p["_cw"] = CW
        t = fp.table(pdb)
        ca, cb = p["ca"], p["cb"]
        rn_map = NP.resnames(pdb)
        bmask = t["ch"] == cb
        scm = (t["ch"] == ca) & ~np.isin(t["nm"], list(fp.BB) + ["CB"])
        res_pool = [(int(x), np.where(scm & (t["rn"] == x))[0])
                    for x in sorted(set(t["rn"][scm].tolist()))]
        res_pool = [(a, b) for a, b in res_pool if len(b)]
        # unbound surfaces for the global descriptors
        try:
            Ar = PRK.cached_surface(RS.chain_pdb(pdb, ca))
            Bl = PRK.cached_surface(RS.chain_pdb(pdb, cb))
        except Exception:
            Ar = Bl = None
        if Ar is None or Bl is None or len(res_pool) < 5:
            continue
        Pr, Fr = Ar[0]["pts"], Ar[0]["feat"]
        Pl0, Fl = Bl[0]["pts"] - p["lcen"], Bl[0]["feat"]
        tree_s = cKDTree(Pr)

        poses, rots = PRK.fft_poses(p)
        if not poses:
            continue
        L0 = t["co"][bmask] - p["lcen"]
        t1 = time.time()
        recs, rmsd = [], []
        for v, ri, sh, rm in poses:
            lig_s = Pl0 @ rots[ri].T + p["centre"] + sh
            d = DG.descriptors(Pr, Fr, Pl0, Fl, tree_s, lig_s)
            if d is None:
                continue
            co = t["co"].copy()
            co[bmask] = L0 @ rots[ri].T + p["centre"] + sh
            e = DE.sphere_energies(t, co, ca, cb, res_pool, rn_map, cKDTree(co[bmask]), soft=True)
            if e is None:
                continue
            row = {"shape": v}
            row.update({f"g_{k}": val for k, val in d.items()})
            for term, vals in e.items():
                s = DE.summarise(vals)
                for st in ("sum", "median_nz", "frac_fav"):
                    row[f"e_{term}_{st}"] = s[st]
            recs.append(row)
            rmsd.append(rm)
        if len(recs) < 200:
            continue
        keys = sorted(recs[0])
        M = np.array([[r[k] for k in keys] for r in recs], float)
        R = np.array(rmsd)
        store[pdb] = (M, R, keys)
        rows.append({"pdb": pdb, "n": int(len(R)), "n_near": int((R <= NEAR).sum()),
                     "best": float(R.min()), "secs": time.time() - t1})
        report(f"    {pdb}: {len(R)} poses in {time.time()-t1:.0f}s | near-native {int((R<=NEAR).sum())} | "
               f"best {R.min():.1f} A")

    if not rows:
        report("\n  nothing computed")
        return 1
    keys = store[rows[0]["pdb"]][2]
    np.savez_compressed(NPZ, keys=np.array(keys),
                        **{f"M_{k}": v[0] for k, v in store.items()},
                        **{f"R_{k}": v[1] for k, v in store.items()})
    report(f"\n  per-pose values -> {NPZ.name} ({NPZ.stat().st_size/1e6:.0f} MB)")

    have_near = [r["pdb"] for r in rows if r["n_near"] > 0]
    report(f"  {len(rows)} complexes | {len(have_near)} contain a near-native pose: {have_near}")
    report("  (AUC_far and AUC_close are computed ONLY on those; the rest have no positive class)")

    # ---------------- targeted AUCs and Spearman, pooled per arm ----------------
    res = {}
    for ki, k in enumerate(keys):
        af, ac, sp = [], [], []
        for pdb, (M, R, _) in store.items():
            v = M[:, ki]
            if np.std(v) < 1e-12 or not np.isfinite(v).all():
                continue
            sp.append(spearmanr(v, R).correlation)
            near = R <= NEAR
            far = R > FAR
            close = (R > NEAR) & (R <= FAR)
            if near.sum() and far.sum():
                af.append(roc_auc_score(near[near | far], v[near | far]))
            if near.sum() and close.sum():
                ac.append(roc_auc_score(near[near | close], v[near | close]))
        if not sp:
            continue
        res[k] = {"spearman_median": float(np.median(sp)),
                  "auc_far": float(np.median(af)) if af else None,
                  "auc_close": float(np.median(ac)) if ac else None,
                  "n_far": len(af), "n_close": len(ac)}

    order = sorted(res, key=lambda k: -(res[k]["auc_close"] or 0))
    report(f"\n  {'arm':<22}{'Spearman':>10}{'AUC_far':>10}{'AUC_close':>12}   (median over complexes)")
    for k in order:
        r = res[k]
        report(f"  {k:<22}{r['spearman_median']:>+10.3f}"
               + (f"{r['auc_far']:>10.3f}" if r["auc_far"] is not None else f"{'--':>10}")
               + (f"{r['auc_close']:>12.3f}" if r["auc_close"] is not None else f"{'--':>12}"))
    report(f"  {'chance':<22}{0.0:>+10.3f}{0.5:>10.3f}{0.5:>12.3f}")

    # ---------------- the distogram itself: z-scored value per RMSD bin ----------------
    report(f"\n  DISTOGRAM -- median z-scored value per RMSD bin (shape of the relationship, not one number)")
    labels = [f"{BINS[i]:.0f}-{BINS[i+1]:.0f}" if BINS[i+1] < 1e8 else f">{BINS[i]:.0f}"
              for i in range(len(BINS) - 1)]
    report("    " + f"{'arm':<22}" + "".join(f"{l:>9}" for l in labels))
    disto = {}
    for k in order[:12]:
        ki = keys.index(k)
        cells = []
        for bi in range(len(BINS) - 1):
            vals = []
            for pdb, (M, R, _) in store.items():
                v = M[:, ki]
                s = np.std(v)
                if s < 1e-12:
                    continue
                m = (R >= BINS[bi]) & (R < BINS[bi + 1])
                if m.sum() >= 3:
                    vals.append(float(np.median((v[m] - v.mean()) / s)))
            cells.append(float(np.median(vals)) if vals else float("nan"))
        disto[k] = cells
        report("    " + f"{k:<22}" + "".join(f"{c:>9.2f}" if np.isfinite(c) else f"{'--':>9}" for c in cells))

    report("\n  READING")
    best = order[0]
    bc = res[best]["auc_close"] or 0.5
    bf = max((res[k]["auc_far"] or 0.5) for k in res)
    if bc >= 0.65:
        report(f"  A HIDDEN DETECTOR. '{best}' separates near-native from CLOSE-WRONG at AUC {bc:.3f} while its")
        report(f"  Spearman is {res[best]['spearman_median']:+.3f}. The summary statistic was hiding it: the arm")
        report("  is a near-native detector, not a global ranker, and should be used as one.")
    elif bf >= 0.65:
        report(f"  THE ARMS ONLY REJECT ABSURD POSES. Best AUC_far is {bf:.3f} but the best AUC_close is only")
        report(f"  {bc:.3f}. Every arm can tell a correct pose from a grossly wrong one and none can tell it")
        report("  from a nearly-right one -- and the FFT's clash filter already removes the grossly wrong. That")
        report("  is one sentence explaining every ranking null recorded today.")
    else:
        report(f"  NOTHING HIDDEN. Best AUC_close {bc:.3f}, best AUC_far {bf:.3f}, both at chance. The nulls")
        report("  stand exactly as recorded, and no non-monotonic structure was being missed by Spearman.")

    json.dump({"test": "dock_distogram", "near_A": NEAR, "far_A": FAR, "bins": BINS, "bin_labels": labels,
               "arms": res, "distogram": disto, "complexes": rows, "n_with_near": len(have_near),
               "npz": NPZ.name, "log": log}, open(OUT / "dock_distogram.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'dock_distogram.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

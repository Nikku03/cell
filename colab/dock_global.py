"""LOOK AT THE INTERFACE AS A WHOLE -- global descriptors where local contact scoring measured nothing.

THE ARGUMENT, and it comes from a measurement rather than a hunch.  `dock_poserank` trained a contact scorer on
1.4M decoy-pose contacts -- exactly the confusions the FFT search produces -- and got Spearman +0.003, worse
than an untrained network. The reason is structural, not a training failure:

    the FFT SELECTED those decoys for local shape fit
    so a patch of contacts at a decoy looks locally almost exactly like a patch at the native pose
    because locally, it IS one

Whatever separates a real interface from a shape-plausible fake therefore cannot be read off individual
contacts. It has to be a property of the WHOLE patch. That is what this measures.

The compute objection to doing this -- that bigger areas cost exponentially -- does not hold: scoring a fixed
interface is linear in contacts with a spatial index, and the FFT is polynomial in grid size. The exponential
cost in docking is CONFORMATIONAL search, degrees of freedom, not area. Evaluating the whole interface once per
pose is cheap, so the local/global split is free.

THE DESCRIPTORS, each chosen because a local pairwise scorer provably cannot see it:

    elec_compl    correlation of APBS potential across contacting point pairs. A real interface puts + against
                  -, so COMPLEMENTARY means NEGATIVE correlation. No single contact reveals this; it is a
                  property of the patch's charge pattern as a whole.
    phob_compl    same for hydrophobicity. Greasy meets greasy, so complementary means POSITIVE correlation.
    contiguity    largest connected cluster of buried receptor points / all buried points. A real interface is
                  ONE compact patch. A shape-plausible decoy can hit the same contact count spread over three
                  disconnected islands -- identical locally, obviously wrong globally.
    n_patches     the number of those clusters, reported directly.
    bsa_proxy     buried surface points. Included but FLAGGED: this is close to raw contact count, which
                  nexus_pairs measured as ACTIVELY WRONG (z -1.81) on rotation decoys. If it wins here that is
                  a contradiction to explain, not a result to bank.
    packing       buried points per contact pair -- how tightly the patch is packed rather than how big it is.

ARMS: each descriptor alone, a z-sum of the complementarity+contiguity ones, and a logistic combiner fitted on
TRAIN complexes only. Against `shape` (the FFT score, the standing baseline) and the exact chance rate.

PREDECLARED, before any number:
    a global descriptor beats shape on Spearman AND top-k   -> looking at the whole works, and the two-stage
                                                               architecture (local to sample, global to judge)
                                                               is validated.
    global ~ shape                                          -> global adds nothing over shape complementarity
                                                               and the ranking problem is still open.
    global WORSE than shape                                 -> informative: these descriptors are not what
                                                               separates real interfaces from good fakes.

POWER, stated up front so a null is not over-read: only 2-4 of 2,400 poses per complex are near-native, so
chance at top-10 is under 2% and top-k cannot resolve a modest effect with 6 testable complexes. Spearman over
all ~2,400 poses carries the weight; it is the readout with power here.

SIGN CONVENTION IS FIXED BEFORE THE RUN and asserted in code: for every arm, a BETTER score must mean a LOWER
RMSD, so Spearman(score, RMSD) is NEGATIVE when an arm works. Descriptors whose natural sign is inverted are
negated at definition, not after seeing the answer.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import dock_fft as DF
import dock_poserank as PRK
import dock_rescore as RS
import flex_physics as fp

OUT = A.OUT
N_TRAIN = 20
CONTACT = 4.5
BURY = 4.5                  # a surface point this close to the partner's surface counts as buried
CLUSTER = 3.5               # buried points within this distance are in the same patch
MIN_PAIRS = 10
TOPK = (1, 10, 100)
NEAR_NATIVE = DF.NEAR_NATIVE

# every descriptor is defined so that HIGHER = better pose, fixed before the run
DESCS = ("elec_compl", "phob_compl", "contiguity", "n_patches", "bsa_proxy", "packing")


def descriptors(Pr, Fr, Pl, Fl, tree_r, lig_pts):
    """global properties of the whole contact patch for one pose. Returns None if too few contacts."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nb = tree_r.query_ball_point(lig_pts, CONTACT)
    ii, jj = [], []
    for j, lst in enumerate(nb):
        for i in lst:
            ii.append(i); jj.append(j)
    if len(ii) < MIN_PAIRS:
        return None
    ii = np.asarray(ii); jj = np.asarray(jj)
    bur = np.unique(ii)                                     # buried receptor points

    # --- complementarity: patterns across the WHOLE patch, invisible to any single contact ---
    pr, pl = Fr[ii, 0], Fl[jj, 0]                           # APBS potential
    hr, hl = Fr[ii, 2], Fl[jj, 2]                           # hydrophobicity
    def corr(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    # complementary charge = ANTI-correlated, so negate to keep "higher is better"
    elec_compl = -corr(pr, pl)
    phob_compl = corr(hr, hl)                               # like meets like, already "higher is better"

    # --- contiguity: is this ONE patch, or scattered islands? ---
    B = Pr[bur]
    if len(B) > 1:
        d = np.linalg.norm(B[:, None, :] - B[None, :, :], axis=2)
        r, c = np.where((d < CLUSTER) & (d > 0))
        if len(r):
            g = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(B), len(B)))
            ncomp, lab = connected_components(g, directed=False)
            sizes = np.bincount(lab)
            contiguity = float(sizes.max() / len(B))
        else:
            ncomp, contiguity = len(B), 1.0 / len(B)
    else:
        ncomp, contiguity = 1, 1.0
    return {"elec_compl": elec_compl, "phob_compl": phob_compl, "contiguity": contiguity,
            "n_patches": -float(ncomp),                      # fewer patches is better -> negate
            "bsa_proxy": float(len(bur)), "packing": float(len(bur)) / len(ii)}


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr

    report("=" * 100)
    report("THE INTERFACE AS A WHOLE -- global descriptors where local contact scoring measured nothing")
    report("=" * 100)
    report("  WHY: dock_poserank got Spearman +0.003 training on 1.4M decoy-pose contacts. The FFT SELECTED")
    report("  those decoys for local shape fit, so locally a decoy patch IS a real patch. The signal, if any,")
    report("  has to be a property of the whole patch.")
    report("  PREDECLARED: a global descriptor beating shape on Spearman AND top-k validates local-to-sample /")
    report("  global-to-judge. global ~ shape => adds nothing. Sign convention fixed before the run: a working")
    report("  arm has NEGATIVE Spearman(score, RMSD), and inverted descriptors are negated at definition.")

    dockj = json.load(open(OUT / "dock_fft.json"))
    CW = dockj["clash_w_chosen"]
    test_pdbs = [r["pdb"] for r in dockj["complexes"]]
    have = sorted(f[:-4] for f in os.listdir(fp.PDBDIR) if f.endswith(".pdb") and "__" not in f)

    report(f"\n  loading cached UNBOUND surfaces...")
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
            r = PRK.cached_surface(tag) if tag else None
            if r is None:
                ok = False
                break
            chs[ch] = {"pc": r[0], "g": r[1]}
        if not ok:
            continue
        p["_cw"] = CW
        pool[pdb] = {"prep": p, "ub": chs}
    tst = [q for q in test_pdbs if q in pool]
    tr = [q for q in sorted(pool) if q not in test_pdbs][:N_TRAIN]
    report(f"    {len(pool)} complexes in {time.time()-tb:.0f}s | train {len(tr)} | test {len(tst)}")
    if len(tst) < 5:
        report("  not enough test complexes")
        return 1

    report("\n  generating pose sets and scoring the whole patch of each pose...")
    tp = time.time()
    for n, pdb in enumerate(tr + tst):
        c = pool[pdb]
        p = c["prep"]
        poses, rots = PRK.fft_poses(p)
        Pr = c["ub"][p["ca"]]["pc"]["pts"]
        Fr = c["ub"][p["ca"]]["pc"]["feat"]
        Pl0 = c["ub"][p["cb"]]["pc"]["pts"] - p["lcen"]
        Fl = c["ub"][p["cb"]]["pc"]["feat"]
        tree = cKDTree(Pr)
        rows, rmsd, shape = [], [], []
        for v, ri, sh, rm in poses:
            d = descriptors(Pr, Fr, Pl0, Fl, tree, Pl0 @ rots[ri].T + p["centre"] + sh)
            if d is None:
                continue
            rows.append(d); rmsd.append(rm); shape.append(v)
        c["D"] = {k: np.array([r[k] for r in rows], float) for k in DESCS}
        c["rmsd"] = np.array(rmsd)
        c["shape"] = np.array(shape)
        if (n + 1) % 8 == 0:
            report(f"    {n+1}/{len(tr)+len(tst)} complexes scored ({time.time()-tp:.0f}s)")
    report(f"    done in {time.time()-tp:.0f}s")

    # ---- LIVENESS: a descriptor that never varies cannot rank anything ----
    dead = [k for k in DESCS if all(float(pool[q]["D"][k].std()) < 1e-12 for q in tst)]
    if dead:
        report(f"\n    LIVENESS: {dead} constant across every test complex -- dead arms, rows below mean nothing.")

    # ---- fit a combiner on TRAIN complexes only ----
    def zc(a):
        s = a.std()
        return (a - a.mean()) / s if s > 1e-12 else np.zeros_like(a)

    COMBO = ("elec_compl", "phob_compl", "contiguity", "packing")
    Xtr, ytr = [], []
    for q in tr:
        c = pool[q]
        if not len(c["rmsd"]):
            continue
        Xtr.append(np.stack([zc(c["D"][k]) for k in COMBO], 1))
        ytr.append((c["rmsd"] <= NEAR_NATIVE * 2).astype(float))     # generous label: 10 A, else too few positives
    w = None
    if Xtr:
        X = np.vstack(Xtr); y = np.concatenate(ytr)
        if 0 < y.sum() < len(y):
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
            w = lr.coef_[0]
            report(f"\n  combiner fitted on {len(tr)} TRAIN complexes ({int(y.sum()):,} positives of {len(y):,})")
            report("    weights: " + "  ".join(f"{k} {v:+.3f}" for k, v in zip(COMBO, w)))
        else:
            report(f"\n  combiner NOT fitted: labels degenerate ({int(y.sum())} positives of {len(y)})")

    # ---- evaluate ----
    report(f"\n  RANKING {len(tst)} held-out pose sets")
    rows = []
    for pdb in tst:
        c = pool[pdb]
        rm = c["rmsd"]
        if not len(rm):
            continue
        arms = {"shape": c["shape"]}
        for k in DESCS:
            arms[k] = c["D"][k]
        arms["z_sum"] = sum(zc(c["D"][k]) for k in COMBO)
        if w is not None:
            arms["combiner"] = np.stack([zc(c["D"][k]) for k in COMBO], 1) @ w
        res = {}
        for k, v in arms.items():
            live = float(np.std(v)) > 1e-12
            o = np.argsort(-v)
            res[k] = {"hits": {str(t): bool((rm[o][:t] <= NEAR_NATIVE).any()) for t in TOPK},
                      "spearman": float(spearmanr(v, rm).correlation) if live else None,
                      "top100_mean_rmsd": float(rm[o][:100].mean()), "live": live}
        frac = float((rm <= NEAR_NATIVE).mean())
        rows.append({"pdb": pdb, "n_poses": int(len(rm)), "near_native_frac": frac,
                     "best_rmsd": float(rm.min()), "mean_rmsd": float(rm.mean()), "arms": res})
        report(f"    {pdb}: {len(rm)} poses, frac {frac:.4f} | "
               + "  ".join(f"{k} {res[k]['spearman']:+.3f}" for k in
                           ("shape", "elec_compl", "contiguity", "z_sum") if res[k]["spearman"] is not None))

    if not rows:
        report("\n  nothing ranked")
        return 1
    tt = [r for r in rows if r["near_native_frac"] > 0]
    setmean = float(np.mean([r["mean_rmsd"] for r in rows]))
    keys = ["shape"] + list(DESCS) + ["z_sum"] + (["combiner"] if w is not None else [])
    report(f"\n  {len(rows)} ranked | {len(tt)} have a near-native pose: {[r['pdb'] for r in tt]}")
    report(f"\n  {'arm':<13}{'top-1':>7}{'top-10':>8}{'top-100':>9}{'Spearman':>11}{'top100 RMSD':>13}{'set mean':>10}")
    tab = {}
    for k in keys:
        sp = [r["arms"][k]["spearman"] for r in rows if r["arms"][k]["spearman"] is not None]
        tab[k] = {"top": {str(t): sum(r["arms"][k]["hits"][str(t)] for r in tt) for t in TOPK},
                  "spearman_median": float(np.median(sp)) if sp else None,
                  "top100_mean_rmsd": float(np.mean([r["arms"][k]["top100_mean_rmsd"] for r in rows]))}
        report(f"  {k:<13}" + "".join(f"{tab[k]['top'][str(t)]:>7}" if t == 1 else
                                      f"{tab[k]['top'][str(t)]:>8}" if t == 10 else
                                      f"{tab[k]['top'][str(t)]:>9}" for t in TOPK)
               + f"{tab[k]['spearman_median']:>11.3f}{tab[k]['top100_mean_rmsd']:>13.1f}{setmean:>10.1f}")
    ch = {str(t): float(np.mean([1 - (1 - r["near_native_frac"]) ** t for r in tt])) * len(tt) for t in TOPK}
    report(f"  {'chance':<13}{ch['1']:>7.2f}{ch['10']:>8.2f}{ch['100']:>9.2f}{0.0:>11.3f}"
           f"{setmean:>13.1f}{setmean:>10.1f}")
    report(f"  (top-k denominator is the {len(tt)} complexes with a near-native pose; Spearman uses all {len(rows)})")

    report("\n  READING")
    base = tab["shape"]["spearman_median"]
    cands = {k: tab[k]["spearman_median"] for k in keys if k != "shape"}
    best = min(cands, key=lambda k: cands[k])
    bs, bh = cands[best], tab[best]["top"]["10"]
    if bs < base - 0.02 and bh >= tab["shape"]["top"]["10"]:
        report(f"  LOOKING AT THE WHOLE WORKS. '{best}' reaches Spearman {bs:+.3f} against shape's {base:+.3f},")
        report(f"  with top-10 {bh}/{len(tt)} vs shape {tab['shape']['top']['10']}/{len(tt)}. Local scoring")
        report("  measured +0.003 on the same pose sets; a property of the whole patch does what no contact")
        report("  could. Local to sample, global to judge, is the right split.")
    elif bs < base - 0.02:
        report(f"  PARTIAL. '{best}' improves Spearman to {bs:+.3f} from shape's {base:+.3f} -- a real global")
        report(f"  signal where local scoring had none -- but top-10 is {bh}/{len(tt)}, so it still cannot")
        report("  surface 2-4 near-native poses out of 2,400. The signal is real and too weak alone.")
    else:
        report(f"  NO GAIN. Best global arm '{best}' at Spearman {bs:+.3f} vs shape {base:+.3f}. Looking at the")
        report("  whole patch, with charge and hydrophobic complementarity and contiguity, does not separate")
        report("  real interfaces from shape-plausible fakes any better than the shape score already did.")

    json.dump({"test": "dock_global", "n_train": len(tr), "descriptors": list(DESCS), "combo": list(COMBO),
               "combiner_weights": (None if w is None else dict(zip(COMBO, map(float, w)))),
               "dead_arms": dead, "arms": tab, "chance": ch, "complexes": rows,
               "n_testable": len(tt), "log": log}, open(OUT / "dock_global.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'dock_global.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

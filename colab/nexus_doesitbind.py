"""DO THESE TWO PROTEINS INTERACT AT ALL?  The question that does NOT require solving pose ranking.

THE REFORMULATION, and why it is not the thing that already failed.  NEXUS scores an interface that EXISTS.
For two novel proteins there is no complex, so the obvious move is dock-then-score -- and that is blocked: the
top-ranked pose sits at rank ~1,791 of 9,600, so dock-then-score is scoring a wrong structure.

But "do they interact?" needs only WHETHER, not WHERE. That can be read from the DISTRIBUTION of what the
search produces without ever identifying the correct pose. A pair that genuinely binds may leave a different
signature across 2,400 poses than a pair that does not belong together, even when neither pair's best pose is
findable. Ranking and detection are different problems, and only one of them is known to be broken.

THE DESIGN THAT KILLS THE OBVIOUS ARTEFACT.  Bigger proteins make more contacts and score higher, so a
classifier fed raw scores would learn SIZE and look like it works. Two defences, both structural rather than
statistical:

    PAIRED    every receptor appears ONCE as a positive (with its true crystal partner) and ONCE as a negative
              (with a foreign chain from a different complex, size-matched). Any per-protein effect -- size,
              surface roughness, charge, how well it grids -- is present in BOTH classes and cancels.
    SIZE ARM  a classifier given ONLY the two chain sizes and nothing else. If docking features cannot beat
              that, they carry no information about binding, whatever their absolute AUC.

FEATURES, all read off the pose-score distribution and none requiring the right pose to be known:
    best / top10 / top100 mean shape score      how good the best fits are
    score spread, skew                          is there a distinct high-scoring population, or a smooth blob
    n_poses above a z-threshold                 how many poses stand out from that pair's own distribution
    contact-count stats at the top poses        interface size the search can achieve
    all z-scored WITHIN each pair before use, so absolute magnitude (which tracks size) is removed and only
    the SHAPE of the distribution survives

CONTROLS:
    size_only      the two chain sizes. The artefact baseline.
    label_perm     labels shuffled, same features, same splitter. The floor.
    per-pair z     already applied, so the headline features cannot express raw magnitude at all.

PREDECLARED, before any number:
    docking features beat size_only AND beat label_perm     -> the pose-score DISTRIBUTION carries binding
                                                               information, and detection works where ranking
                                                               does not. This would be a genuinely new
                                                               capability, not a rescue of the old one.
    docking ~ size_only                                     -> it learned protein size. No binding signal.
    docking ~ label_perm                                    -> nothing at all.

HONEST LIMITS, up front. Negatives are chains from different solved complexes, which are near-certainly
non-interacting but are NOT verified non-binders -- there is no gold-standard negative set for PPIs. Both
partners are bound-form coordinates. And a pair of proteins that fails to dock well may simply be two proteins
that were never going to fit in this grid representation, which is a property of the method, not of biology.
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
import flex_physics as fp

OUT = A.OUT
N_PAIRS = 40                  # positives; an equal number of paired negatives is built from the same receptors
N_ROT = 600
SIZE_TOL = 0.35               # a foreign partner must be within this fractional size of the true one
SEED = 0


def features(scores, rmsdless=True):
    """distribution shape only. z-scored within the pair, so absolute magnitude -- which tracks protein size --
    cannot survive into the classifier."""
    s = np.asarray(scores, float)
    if len(s) < 50 or np.std(s) < 1e-12:
        return None
    z = (s - s.mean()) / s.std()
    zs = np.sort(z)[::-1]
    return {"z_best": float(zs[0]), "z_top10": float(zs[:10].mean()), "z_top100": float(zs[:100].mean()),
            "skew": float(((z ** 3).mean())), "kurt": float(((z ** 4).mean())),
            "n_z3": float((z > 3).sum()) / len(z), "n_z4": float((z > 4).sum()) / len(z),
            "gap_1_10": float(zs[0] - zs[9]), "gap_10_100": float(zs[9] - zs[99]) if len(zs) > 99 else 0.0,
            "p99_p50": float(np.percentile(z, 99) - np.percentile(z, 50))}


def dock_scores(R, rR, L, rL):
    """all pose scores for a receptor/ligand pair, using dock_fft's own grids and weight. No RMSD: for a
    non-interacting pair there is no correct answer to measure against, which is the point."""
    centre = R.mean(0)
    box = DF.GRID * DF.SPACING
    dR = float(np.linalg.norm(R - centre, axis=1).max() * 2)
    dL = float(np.linalg.norm(L - L.mean(0), axis=1).max() * 2)
    if dR + dL > box:
        return None
    solid_R = DF.rasterise(R, centre, DF.offsets_for(rR))
    if solid_R is None:
        return None
    skin = DF._morph(solid_R, DF.SKIN, True) & ~solid_R
    core = DF._morph(solid_R, DF.CORE_ERODE, False)
    FA = np.fft.rfftn(skin.astype(np.float32)) - 50.0 * np.fft.rfftn(core.astype(np.float32))
    offs = DF.offsets_for(rL)
    Lc = L - L.mean(0)
    out = []
    for Rm in DF.rotations(N_ROT):
        g = DF.rasterise(Lc @ Rm.T + centre, centre, offs)
        if g is None:
            continue
        sc = np.fft.irfftn(FA * np.conj(np.fft.rfftn(g.astype(np.float32))), solid_R.shape)
        for pk in DF.nms_peaks(sc, DF.PEAKS_PER_ROT, DF.NMS_CELLS):
            if sc[pk] > 0:
                out.append(float(sc[pk]))
    return out or None


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    report("=" * 100)
    report("DO THESE TWO PROTEINS INTERACT AT ALL? -- the question that does not need pose ranking solved")
    report("=" * 100)
    report("  Dock-then-score is blocked: the top pose is rank ~1,791 of 9,600, so it scores a wrong structure.")
    report("  But WHETHER they bind can be read from the pose-score DISTRIBUTION without knowing WHERE.")
    report("  PAIRED design: every receptor appears once with its true partner and once with a size-matched")
    report("  foreign chain, so per-protein effects cancel. Features are z-scored WITHIN each pair, so raw")
    report("  magnitude (which tracks size) cannot reach the classifier.")
    report("  PREDECLARED: beats size_only AND label_perm => detection works where ranking does not.")
    report("  ~ size_only => it learned protein size. ~ label_perm => nothing at all.")

    have = sorted(f[:-4] for f in os.listdir(fp.PDBDIR) if f.endswith(".pdb") and "__" not in f)
    rng = np.random.default_rng(SEED)

    # ---- collect true pairs (receptor + its crystal partner) ----
    pool = []
    for pdb in have:
        if len(pool) >= N_PAIRS * 2:
            break
        p = DF.prepare(pdb)
        if p is None or not p["sane"]:
            continue
        t = fp.table(pdb)
        pool.append({"pdb": pdb, "ca": p["ca"], "cb": p["cb"],
                     "R": p["R"], "rR": t["rad"][t["ch"] == p["ca"]],
                     "L": p["L"], "rL": t["rad"][t["ch"] == p["cb"]]})
    report(f"\n  {len(pool)} usable complexes; building {min(N_PAIRS, len(pool))} paired examples")
    pool = pool[:N_PAIRS]

    # ---- for each receptor, find a size-matched FOREIGN partner from a different complex ----
    rows = []
    for i, a in enumerate(pool):
        n_true = len(a["L"])
        cands = [b for j, b in enumerate(pool)
                 if j != i and abs(len(b["L"]) - n_true) / max(n_true, 1) <= SIZE_TOL]
        if not cands:
            continue
        b = cands[int(rng.integers(len(cands)))]
        for lab, part, tag in ((1, a, "true"), (0, b, f"foreign:{b['pdb']}")):
            t1 = time.time()
            sc = dock_scores(a["R"], a["rR"], part["L"], part["rL"])
            f = features(sc) if sc else None
            if f is None:
                continue
            f.update({"n_R": len(a["R"]), "n_L": len(part["L"])})
            rows.append({"receptor": a["pdb"], "group": a["pdb"], "label": lab, "partner": tag,
                         "n_poses": len(sc), "feat": f, "secs": time.time() - t1})
        if len(rows) % 10 == 0 and rows:
            report(f"    {len(rows)} docked ({time.time()-t0:.0f}s)")

    pos = sum(r["label"] for r in rows)
    report(f"\n  {len(rows)} docked runs | {pos} positive, {len(rows)-pos} negative | "
           f"{len({r['group'] for r in rows})} distinct receptors")
    paired = [g for g in {r["group"] for r in rows}
              if len({r["label"] for r in rows if r["group"] == g}) == 2]
    report(f"  receptors appearing in BOTH classes (the paired design): {len(paired)}")
    if len(paired) < 10:
        report("  too few paired receptors to run the design")
        return 1
    rows = [r for r in rows if r["group"] in set(paired)]

    fk = [k for k in rows[0]["feat"] if k not in ("n_R", "n_L")]
    X = np.array([[r["feat"][k] for k in fk] for r in rows], float)
    Xs = np.array([[r["feat"]["n_R"], r["feat"]["n_L"]] for r in rows], float)
    y = np.array([r["label"] for r in rows])
    g = np.array([r["group"] for r in rows])

    def cv_auc(Xm, yy, seed=None):
        rr = np.random.default_rng(seed) if seed is not None else None
        yy2 = yy.copy()
        if rr is not None:                       # permute WITHIN the group structure, preserving pairing
            for grp in set(g):
                m = g == grp
                yy2[m] = rr.permutation(yy2[m])
        pred = np.zeros(len(yy2), float)
        gkf = GroupKFold(n_splits=min(5, len(set(g))))
        for tr, te in gkf.split(Xm, yy2, groups=g):
            if len(set(yy2[tr])) < 2:
                continue
            mu, sd = Xm[tr].mean(0), Xm[tr].std(0) + 1e-9
            lr = LogisticRegression(max_iter=2000).fit((Xm[tr] - mu) / sd, yy2[tr])
            pred[te] = lr.predict_proba((Xm[te] - mu) / sd)[:, 1]
        return float(roc_auc_score(yy2, pred)) if len(set(yy2)) > 1 else float("nan")

    auc_dock = cv_auc(X, y)
    auc_size = cv_auc(Xs, y)
    perm = [cv_auc(X, y, seed=s) for s in range(200)]
    perm = np.array([p for p in perm if np.isfinite(p)])
    pval = float((perm >= auc_dock).mean())

    report(f"\n  {'arm':<18}{'AUC':>8}   (GroupKFold by receptor, so no receptor spans train and test)")
    report(f"  {'docking dist':<18}{auc_dock:>8.3f}")
    report(f"  {'size_only':<18}{auc_size:>8.3f}   <- the artefact baseline")
    report(f"  {'label_perm':<18}{np.median(perm):>8.3f}   median of 200, 95th {np.percentile(perm,95):.3f}")
    report(f"  {'chance':<18}{0.5:>8.3f}")
    report(f"  permutation p = {pval:.4f}")

    report("\n  READING")
    if auc_dock > auc_size + 0.05 and pval < 0.05:
        report(f"  DETECTION WORKS WHERE RANKING DOES NOT. The pose-score distribution separates true partners")
        report(f"  from foreign ones at AUC {auc_dock:.3f}, against {auc_size:.3f} for size alone and a")
        report(f"  permutation null of {np.median(perm):.3f} (p={pval:.4f}). Knowing WHETHER two proteins bind")
        report("  does not require knowing WHERE -- which is a capability the ranking failure does not block.")
    elif auc_dock > auc_size + 0.05:
        report(f"  Beats size ({auc_dock:.3f} vs {auc_size:.3f}) but permutation p={pval:.4f} does not clear")
        report("  0.05. Suggestive on this sample size, not established.")
    elif auc_size >= auc_dock - 0.05:
        report(f"  IT LEARNED SIZE. docking {auc_dock:.3f} vs size_only {auc_size:.3f} -- the distribution")
        report("  features add nothing over how big the two chains are, despite per-pair z-scoring. The paired")
        report("  design and the size arm are exactly what make that visible rather than flattering.")
    else:
        report(f"  NOTHING. docking {auc_dock:.3f}, size {auc_size:.3f}, permutation {np.median(perm):.3f}.")
        report("  The pose-score distribution does not distinguish real partners from foreign chains.")

    json.dump({"test": "nexus_doesitbind", "n_rows": len(rows), "n_paired_receptors": len(paired),
               "n_rot": N_ROT, "auc_docking": auc_dock, "auc_size_only": auc_size,
               "perm_median": float(np.median(perm)), "perm_p95": float(np.percentile(perm, 95)),
               "p_value": pval, "features": fk,
               "rows": [{k: v for k, v in r.items() if k != "feat"} for r in rows], "log": log},
              open(OUT / "nexus_doesitbind.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'nexus_doesitbind.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

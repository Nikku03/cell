"""CLUSTER, DON'T SCORE: rank regions of pose space by POPULATION, the way ClusPro does.

THE IDEA, and it is the only lever left that is not a per-pose score.  Six scoring approaches have now failed
to rank FFT poses -- shape (-0.059), learned embeddings (+0.003), embeddings retrained on decoy contacts
(+0.003), global descriptors (~0), sphere energies (-0.142 but 8/12 sign-consistent), and the electrostatic
detector (real at p=0.0035 but worth ~10 rank places out of 1,800). Every one asked "which POSE scores best?"

ClusPro, which wins CAPRI, never asks that. It generates poses, CLUSTERS them, and ranks clusters by MEMBER
COUNT. The logic is that the native basin is an ATTRACTOR: many independent rotations land near the true
answer, while a spurious high-scoring pose is usually alone. Population is information no per-pose score
contains, and it has never been used here.

WHY IT ONLY BECAME TESTABLE TODAY.  Cluster population cannot work on a basin of one. At the 600-rotation
budget every previous ranking run used, the median complex had ONE near-native pose and half had none. The
coverage sweep raised that to 4-16 members at 2,400 rotations. That is the difference between a question with
an answer and a question without one.

THE COMPUTATION IS FREE, which matters because clustering 1,000 poses is 500,000 pairwise RMSDs.  For a rigid
ligand, the RMSD between two poses does not need the atoms at all. With the ligand centred and C = (1/N) sum
x_i x_i^T its 3x3 covariance:

    RMSD^2 = Tr( dR C dR^T ) + |dt|^2       dR = R1 - R2,  dt = t1 - t2

Verified exact to 7e-15 over 36 pose pairs before this was relied on. So each pairwise distance is a 3x3 trace
rather than a thousand-atom loop.

THE COMPARISON THAT DECIDES IT, and it is not against chance.  Any clustering of top-scoring poses inherits
whatever the score already knew. The question is whether POPULATION adds anything over SCORE:

    cluster_by_population   rank clusters by member count        <- the ClusPro claim
    cluster_by_bestscore    rank the SAME clusters by their best member's shape score
    pose_by_score           no clustering at all; the existing 0/12 baseline
    random_clusters         cluster a RANDOM subset of poses, not the top-scoring ones. If population ranking
                            works here too, it is not reading anything the score contributed.

PRE-SPECIFIED CONFIGURATION, fixed before the run because today has shown repeatedly what happens otherwise:
ClusPro's own parameters, adapted -- the top 1,000 poses by shape score, clustered at a 9 A ligand-RMSD radius,
greedy by descending score. A sweep over both parameters is reported afterwards and is EXPLORATORY: if some
other setting looks better it is a hypothesis, not this run's result.

PREDECLARED, before any number:
    population puts a near-native pose in the top-1 or top-3 clusters for a majority, AND beats bestscore
                        -> clustering works where scoring does not, and the fix for ranking was never a better
                           score. This is the ClusPro insight reproducing.
    population ~ bestscore
                        -> clusters help only insofar as the score already ranked them; population per se adds
                           nothing.
    neither works       -> the native basin is not an attractor at this sampling density, and the 4-16 members
                           the coverage sweep bought are still too few to stand out.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import dock_fft as DF
import flex_physics as fp

OUT = A.OUT
N_ROT = 2400                 # coverage-validated: 12/12 complexes contain a correct pose
PEAKS = 4
NEAR = DF.NEAR_NATIVE
N_TOP = 1000                 # PRE-SPECIFIED: poses clustered, ClusPro-style
RADIUS = 9.0                 # PRE-SPECIFIED: ligand-RMSD cluster radius, Angstrom
TOPC = (1, 3, 10)            # cluster ranks checked
SWEEP_TOP = (300, 1000, 3000)
SWEEP_RAD = (6.0, 9.0, 12.0)


def pose_rmsd_matrix(rots_idx, shifts, rots, C):
    """pairwise ligand RMSD from transformations alone: RMSD^2 = Tr(dR C dR^T) + |dt|^2, exact to 7e-15."""
    n = len(rots_idx)
    Rm = np.array([rots[i] for i in rots_idx])          # (n,3,3)
    T = np.asarray(shifts)                              # (n,3)
    # Tr(dR C dR^T) = Tr(R1 C R1^T) + Tr(R2 C R2^T) - 2 Tr(R1 C R2^T); first two are trace(C) for any rotation
    trC = float(np.trace(C))
    RC = np.einsum("nij,jk->nik", Rm, C)                # (n,3,3) = R1 C
    # Tr(R1 C R2^T) = sum_ij (R1 C)_ij (R2)_ij -- an ELEMENTWISE product summed, not a contraction over one
    # index. The first version wrote "nik,mjk->nm", summing i and j independently, which gave a max error of
    # 18 A against direct atom RMSD and a non-zero diagonal. Silently wrong clusters, caught by verifying the
    # vectorised matrix against the atom loop it replaces.
    cross = np.einsum("nij,mij->nm", RC, Rm)
    d2 = 2 * trC - 2 * cross
    dt = ((T[:, None, :] - T[None, :, :]) ** 2).sum(2)
    return np.sqrt(np.maximum(d2 + dt, 0.0))


def greedy_cluster(D, order, radius):
    """ClusPro-style: take the highest-ranked unassigned pose as a centre, absorb everything within radius."""
    n = D.shape[0]
    unassigned = np.ones(n, bool)
    clusters = []
    for i in order:
        if not unassigned[i]:
            continue
        mem = np.where(unassigned & (D[i] <= radius))[0]
        clusters.append((int(i), mem))
        unassigned[mem] = False
        if not unassigned.any():
            break
    return clusters


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("CLUSTER, DON'T SCORE -- rank regions of pose space by POPULATION")
    report("=" * 100)
    report("  Six per-pose scores have failed to rank these poses. ClusPro never ranks poses: it clusters them")
    report("  and ranks clusters by MEMBER COUNT, because the native basin is an attractor and a spurious")
    report("  high-scoring pose is usually alone. Population is information no per-pose score contains.")
    report(f"  PRE-SPECIFIED: top {N_TOP} poses, {RADIUS:.0f} A radius, {N_ROT} rotations. A parameter sweep")
    report("  follows and is EXPLORATORY -- a better setting there is a hypothesis, not this run's result.")
    report("  PREDECLARED: population beats bestscore AND lands near-native in top-1/top-3 for a majority")
    report("  => clustering works where scoring does not. population ~ bestscore => population adds nothing.")

    dockj = json.load(open(OUT / "dock_fft.json"))
    CW = dockj["clash_w_chosen"]
    test_pdbs = [r["pdb"] for r in dockj["complexes"]]
    rots = DF.rotations(N_ROT)
    rng = np.random.default_rng(0)

    rows = []
    for pdb in test_pdbs:
        p = DF.prepare(pdb)
        if p is None or not p["sane"]:
            continue
        centre, L, lcen = p["centre"], p["L"], p["lcen"]
        Lc = L - lcen
        C = (Lc.T @ Lc) / len(Lc)
        FA = p["F_skin"] - CW * p["F_core"]
        t1 = time.time()
        sc_l, ri_l, sh_l, rm_l = [], [], [], []
        for ri, Rm in enumerate(rots):
            g = DF.rasterise(Lc @ Rm.T + centre, centre, p["offs_L"])
            if g is None:
                continue
            f = np.fft.irfftn(FA * np.conj(np.fft.rfftn(g.astype(np.float32))), p["shape"])
            for pk in DF.nms_peaks(f, PEAKS, DF.NMS_CELLS):
                if f[pk] <= 0:
                    continue
                sh = DF.shift_of(pk)
                sc_l.append(float(f[pk])); ri_l.append(ri); sh_l.append(sh)
                rm_l.append(float(np.sqrt((((Lc @ Rm.T + centre + sh) - L) ** 2).sum(1).mean())))
        if len(sc_l) < N_TOP:
            continue
        S = np.array(sc_l); RI = np.array(ri_l); SH = np.array(sh_l); RM = np.array(rm_l)
        if (RM <= NEAR).sum() == 0:
            rows.append({"pdb": pdb, "n_poses": int(len(S)), "n_near": 0, "skipped": "no near-native"})
            report(f"    {pdb}: {len(S)} poses, 0 near-native -- SKIPPED")
            continue

        def run(n_top, radius, random_subset=False):
            if random_subset:
                sel = rng.choice(len(S), min(n_top, len(S)), replace=False)
            else:
                sel = np.argsort(-S)[:n_top]
            D = pose_rmsd_matrix(RI[sel], SH[sel], rots, C)
            order = np.argsort(-S[sel])
            cl = greedy_cluster(D, order, radius)
            rm_sel = RM[sel]
            by_pop = sorted(range(len(cl)), key=lambda i: -len(cl[i][1]))
            by_best = sorted(range(len(cl)), key=lambda i: -S[sel][cl[i][1]].max())
            def hits(rank_order):
                return {str(k): bool(any((rm_sel[cl[i][1]] <= NEAR).any() for i in rank_order[:k]))
                        for k in TOPC}
            return {"n_clusters": len(cl), "pop": hits(by_pop), "best": hits(by_best),
                    "top_cluster_size": int(len(cl[by_pop[0]][1])),
                    "n_near_in_sel": int((rm_sel <= NEAR).sum())}

        main_res = run(N_TOP, RADIUS)
        rand_res = run(N_TOP, RADIUS, random_subset=True)
        sweep = {f"{nt}x{int(rr)}": run(nt, rr) for nt in SWEEP_TOP for rr in SWEEP_RAD
                 if nt <= len(S)}
        # pose-level baseline: does the top-scoring POSE list contain near-native in top k
        o = np.argsort(-S)
        pose_base = {str(k): bool((RM[o][:k] <= NEAR).any()) for k in TOPC}
        rows.append({"pdb": pdb, "n_poses": int(len(S)), "n_near": int((RM <= NEAR).sum()),
                     "main": main_res, "random_subset": rand_res, "sweep": sweep,
                     "pose_baseline": pose_base, "secs": time.time() - t1})
        report(f"    {pdb}: {len(S)} poses, {int((RM<=NEAR).sum())} near-native, "
               f"{main_res['n_clusters']} clusters (top has {main_res['top_cluster_size']}) in "
               f"{time.time()-t1:.0f}s | pop top1 {main_res['pop']['1']}  top3 {main_res['pop']['3']}  "
               f"| best top1 {main_res['best']['1']}")

    ok = [r for r in rows if "main" in r]
    if len(ok) < 5:
        report(f"\n  only {len(ok)} usable complexes")
        return 1

    report(f"\n  {len(ok)} complexes with a near-native pose and enough poses to cluster")
    report(f"  median clusters {int(np.median([r['main']['n_clusters'] for r in ok]))} | "
           f"median top-cluster size {int(np.median([r['main']['top_cluster_size'] for r in ok]))} | "
           f"median near-native inside the clustered top-{N_TOP}: "
           f"{int(np.median([r['main']['n_near_in_sel'] for r in ok]))}")

    report(f"\n  PRE-SPECIFIED CONFIG (top {N_TOP}, {RADIUS:.0f} A)")
    report(f"    {'arm':<26}" + "".join(f"{'top-' + str(k):>9}" for k in TOPC))
    tab = {}
    for nm, key in (("cluster by POPULATION", "pop"), ("cluster by best score", "best")):
        tab[key] = {str(k): sum(r["main"][key][str(k)] for r in ok) for k in TOPC}
        report(f"    {nm:<26}" + "".join(f"{str(tab[key][str(k)]) + '/' + str(len(ok)):>9}" for k in TOPC))
    tab["rand_pop"] = {str(k): sum(r["random_subset"]["pop"][str(k)] for r in ok) for k in TOPC}
    report(f"    {'random subset, by pop':<26}"
           + "".join(f"{str(tab['rand_pop'][str(k)]) + '/' + str(len(ok)):>9}" for k in TOPC))
    tab["pose"] = {str(k): sum(r["pose_baseline"][str(k)] for r in ok) for k in TOPC}
    report(f"    {'POSE by score (baseline)':<26}"
           + "".join(f"{str(tab['pose'][str(k)]) + '/' + str(len(ok)):>9}" for k in TOPC))

    report(f"\n  SWEEP (EXPLORATORY -- a better setting here is a hypothesis, not this run's result)")
    report(f"    {'config':<12}{'pop top-1':>11}{'pop top-3':>11}{'best top-1':>12}")
    sw = {}
    for key in sorted({k for r in ok for k in r["sweep"]}):
        sub = [r["sweep"][key] for r in ok if key in r["sweep"]]
        if not sub:
            continue
        sw[key] = {"pop1": sum(s["pop"]["1"] for s in sub), "pop3": sum(s["pop"]["3"] for s in sub),
                   "best1": sum(s["best"]["1"] for s in sub), "n": len(sub)}
        report(f"    {key:<12}{str(sw[key]['pop1']) + '/' + str(sw[key]['n']):>11}"
               f"{str(sw[key]['pop3']) + '/' + str(sw[key]['n']):>11}"
               f"{str(sw[key]['best1']) + '/' + str(sw[key]['n']):>12}")

    report("\n  READING")
    p1, p3 = tab["pop"]["1"], tab["pop"]["3"]
    b1 = tab["best"]["1"]
    n = len(ok)
    if (p1 > n / 2 or p3 > n / 2) and p1 >= b1:
        report(f"  CLUSTERING WORKS. Ranking clusters by POPULATION puts a near-native pose in the top cluster")
        report(f"  for {p1}/{n} complexes and the top 3 for {p3}/{n}, against {b1}/{n} for ranking the same")
        report(f"  clusters by best score and {tab['pose']['1']}/{n} for the pose baseline. Six per-pose scores")
        report("  failed because the question was wrong: the answer is where poses PILE UP, not where one wins.")
    elif p1 >= b1 + 2 or p3 >= tab["best"]["3"] + 2:
        report(f"  PARTIAL. Population ({p1}/{n} top-1, {p3}/{n} top-3) beats best-score ({b1}/{n}) but does")
        report("  not reach a majority. The attractor is real and weak at this sampling density.")
    else:
        report(f"  CLUSTERING DOES NOT HELP. population {p1}/{n} top-1 vs best-score {b1}/{n} vs pose baseline")
        report(f"  {tab['pose']['1']}/{n}. The native basin does not stand out by member count at {N_ROT}")
        report("  rotations -- with 4-16 members against clusters built from the top 1,000 poses, the true")
        report("  region is not the most populated one.")

    json.dump({"test": "dock_cluster", "n_rot": N_ROT, "n_top": N_TOP, "radius": RADIUS,
               "prespecified": tab, "sweep_exploratory": sw, "complexes": rows,
               "n_usable": len(ok), "log": log}, open(OUT / "dock_cluster.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'dock_cluster.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

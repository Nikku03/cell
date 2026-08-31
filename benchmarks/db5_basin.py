"""Does basin breadth predict nativeness, and does free-energy reranking beat min-energy?

THREE QUESTIONS, ANSWERED IN ORDER, WITH THE BARS FIXED HERE BEFORE ANY NUMBER IS RUN.
Selection code is NOT to be touched until Q1 is answered.

  Q1  BASIN BREADTH vs NATIVENESS. For every pose compute the exact partition function over
      side-chain states (Algorithm 4, already verified against enumeration at 7e-15) and
      record basin breadth T*S_conf = E_min - F alongside interface RMSD to native.
      HYPOTHESIS: a near-native pose has a broader basin -- more side-chain configurations
      of comparable energy -- so T*S_conf should rise as I-RMSD falls, i.e. the correlation
      should be NEGATIVE.
      GATE: pooled Spearman(T*S_conf, I_rmsd) <= -0.10 with p < 1e-3, AND the median
      per-complex Spearman also negative. ANYTHING ELSE CLOSES THE LINE -- if breadth does
      not track nativeness, no reranking by a quantity built from it can help, and the
      honest conclusion is that this route is dead. That is predeclared, not decided after.

  Q2  RERANK BY FREE ENERGY. Rank each shortlist by F instead of E_min and report the
      rank-1 CAPRI success rate. GATE: to count as signal rather than noise the F ranking
      must reach >= 3 successes; 0-2 is a null and closes the line.
      THE BASELINE MUST BE RECOMPUTED, NOT QUOTED. The original 0/58 came from shortlists
      whose per-case seed was abs(hash(cid)), and Python randomises string hashing per
      process, so those exact pose sets are unreproducible -- three runs give 1170193992,
      1557393814, 344117794 for the same input. This script uses zlib.crc32 instead, and
      scores the E_min baseline and the F ranking on the SAME new shortlists so the
      comparison is internally valid rather than against a number from a lost run.

  Q3  ARE THE FIVE SCORERS' ERRORS CORRELATED? Per complex, rank the poses by each score and
      by true I-RMSD; the error of a pose under a score is its rank difference. Correlate
      those error vectors across score pairs.
      INTERPRETATION, fixed in advance: mean off-diagonal |rho| > 0.5 means the scorers share
      a bias, which a better ensemble treatment of the SAME energy function cannot remove;
      < 0.3 means the errors are largely independent roughness, which averaging or an
      ensemble can help with. Between 0.3 and 0.5 is reported as inconclusive.

RESULT, 58 complexes x 20 poses = 1160 exact partition functions, run after the bars above
were committed. Reproduce with `PYTHONPATH=. python benchmarks/basin_analysis.py`.

  Q1  PASS, and it survives the control the bar did not ask for.
        pooled Spearman(T*S_conf, I_rmsd)  -0.4498   n=1160, p=6.4e-53   (bar: <= -0.10)
        per-complex median                 -0.3962   negative on 52/58   (bar: < 0)
      The sign is the predicted one: broader basins sit closer to native. The obvious
      confound is size -- a bigger interface has more side chains, so more entropy -- and
      it is NOT what this is. T*S_conf is essentially uncorrelated with the contact count
      (Spearman -0.047), and controlling for it leaves the effect untouched at -0.4674
      (per-complex median -0.4084, negative on 51/58). Controlling instead for interface
      treewidth, which is itself a nativeness signal (-0.4349), weakens it to -0.3059 --
      still past the bar. Basin breadth carries real information about nativeness.

  Q2  VOID, NOT NULL. See ledger defect N. F scored 0/58, which reads as the predeclared
      null, but the CEILING was also 0/58: no shortlist contained a CAPRI-acceptable pose,
      so no ranking could have scored above zero and the >= 3 bar was unreachable. The
      best pose present anywhere had I_rmsd 3.70 A (median across complexes 11.80 A) and
      median best f_nat 0.010, below the 0.1 acceptable needs. The gate measured nothing.

      THE LINE STILL CLOSES, on a measurement that does not route through the ceiling:
        within-complex Spearman(E_min ranking, F ranking)  +0.9966
        complexes where F picks the SAME rank-1 pose        57/58
        within-complex E_min spread, median              107.952 kcal/mol
        within-complex T*S_conf spread, median             1.297 kcal/mol   (1.0% of it)
      Free energy is minimised energy plus a 1% perturbation, so reranking by it cannot
      change a selection outcome on these shortlists whatever the ceiling had been. The
      route the question proposed is dead for the stated reason, not for the gate's reason.

      WHAT Q1 AND Q2 TOGETHER SAY, and this was measured after seeing the data, so it is an
      observation and not a gate: the entropy term carries MORE nativeness signal than the
      energy it is added to. Mean within-complex Spearman against I_rmsd is +0.0924 for
      E_min and +0.0984 for F -- both near useless -- while T*S_conf ALONE reaches -0.3763.
      Burying a -0.38 signal inside a +0.09 one at a weight of 1% is why Q2 could never have
      moved. Whether ranking on breadth alone helps is UNTESTED and untestable here: with a
      ceiling of 0/58 it cannot be scored, and it needs a search that puts near-native poses
      in the shortlist first. No selection code has been changed.

  Q3  SHARED BIAS. Mean off-diagonal |rho| of the rank-error vectors is 0.676 over all five
      scorers, but that number is inflated -- ve, greedy and F are the same energy function
      (pairwise |rho| ~ 0.997), so five columns hold three distinct scorers. Deflated to
      grid/pair/ve the pairwise values are 0.538, 0.489, 0.587, mean 0.538. Still above the
      0.5 line, so the verdict stands on the honest count: the scorers fail together. A
      better ensemble treatment of the same energy cannot fix a bias all of them share.
"""
from __future__ import annotations

import argparse, glob, json, sys, time, zlib
sys.path.insert(0, ".")
import numpy as np

from rem.docking import capri, score
from rem.docking.data import Structure, load_case
from rem.docking.repack import build_from_case
from rem.docking.freeenergy import free_energy
from rem.docking.rigid import rotation_set
from benchmarks.db5_dock import run_arm, _as_struct

ROTATIONS, TOP_PER_ROT, N_POSES = 2000, 3, 20
REPACK_RES, N_CHI1, N_CHI2 = 6, 3, 2


def subset_ids():
    ids = []
    for f in sorted(glob.glob("benchmarks/shards/db5_dock_w*.json")):
        for r in json.load(open(f))["results"]:
            if "rescore" in r and "error" not in r.get("rescore", {}):
                ids.append((r["id"], r["class"]))
    return sorted(set(ids))


def poses_for(cid: str, rots) -> tuple:
    case = load_case(cid)
    rec, lig = case["r_u"], case["l_u"]
    seed = zlib.crc32(cid.encode()) & 0x7FFFFFFF        # deterministic, unlike hash()
    arm = run_arm(rec, lig, rots, seed, spacing=1.5, keep=N_POSES,
                  top_per_rot=TOP_PER_ROT)
    return rec, lig, arm


def score_pose(rec: Structure, lig_at_pose: Structure, grid_score: float) -> dict:
    rq = score.charges(rec.res_names, rec.atom_names)
    pair = score.pair_energy(rec.coords, rec.elements, rq, lig_at_pose.coords,
                             lig_at_pose.elements,
                             score.charges(lig_at_pose.res_names,
                                           lig_at_pose.atom_names))["total"]
    prob = build_from_case({"r_b": rec, "l_b": lig_at_pose}, side="r", bound=True,
                           max_residues=REPACK_RES, n_chi1=N_CHI1, n_chi2=N_CHI2)
    if len(prob.res_keys) < 2:
        return {"grid": -grid_score, "pair": pair, "ve": pair, "greedy": pair,
                "F": pair, "TS": 0.0, "treewidth": 0, "degenerate": True}
    g, _e = prob.to_factorgraph()
    ex = prob.solve_exact(g)
    gr = prob.solve_greedy(g, restarts=20)
    fe = free_energy(prob, energy_graph=g)
    return {"grid": -grid_score, "pair": pair, "ve": ex["energy"],
            "greedy": gr["energy"], "F": fe["F"], "TS": fe["TS_conf"],
            "treewidth": int(ex["treewidth"]), "degenerate": False}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="benchmarks/basin_w0.json")
    a = ap.parse_args(argv)

    ids = subset_ids()
    if a.limit:
        ids = ids[:a.limit]
    if a.nworkers > 1:
        ids = [x for i, x in enumerate(ids) if i % a.nworkers == a.worker]
    rots = rotation_set(ROTATIONS, seed=1)
    print(f"  {len(ids)} complexes, {N_POSES} poses each, {ROTATIONS} rotations")
    print(f"  {'case':6s} {'class':10s} {'poses':>5s} {'bestI':>7s} {'sec':>7s}")

    out, t0 = [], time.perf_counter()
    for n, (cid, cls) in enumerate(ids, 1):
        try:
            rec, lig, arm = poses_for(cid, rots)
        except Exception as e:                                   # noqa: BLE001
            print(f"  {cid:6s} ERROR {type(e).__name__}: {str(e)[:50]}")
            continue
        recs = []
        for p in arm["_poses"][:N_POSES]:
            lg = _as_struct(lig, p["coords"])
            try:
                sc = score_pose(rec, lg, p["grid_score"])
            except Exception as e:                               # noqa: BLE001
                continue
            m = p["metrics"]
            recs.append({**sc, "I_rmsd": m["I_rmsd"], "L_rmsd": m["L_rmsd"],
                         "f_nat": m["f_nat"], "quality": m["quality"]})
        if len(recs) < 5:
            continue
        out.append({"id": cid, "class": cls, "poses": recs})
        bi = min(r["I_rmsd"] for r in recs)
        print(f"  {cid:6s} {cls:10s} {len(recs):5d} {bi:7.2f} "
              f"{time.perf_counter()-t0:7.0f}", flush=True)
        if n % 5 == 0:
            json.dump(out, open(a.out, "w"), indent=1, default=float)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\n  wrote {a.out}: {len(out)} complexes, "
          f"{sum(len(c['poses']) for c in out)} poses, "
          f"{time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The real Docking Benchmark 5 run: per-complex CAPRI accuracy by difficulty class,
with search error separated from scoring error.

WHAT IS MEASURED, and the arms that make each number interpretable.

  BB  bound-to-bound. Dock r_b against l_b. The two components have the conformations they
      have in the complex, so there is NO conformational change to get wrong. Whatever
      fails here is SEARCH or SCORING, never induced fit. This is the ceiling.
  UU  unbound-to-unbound. Dock r_u against l_u. The real problem. The gap between UU and BB
      is the cost of conformational change, and it is the quantity the difficulty classes
      were defined to predict.

  In BOTH arms the ligand is passed through randomize_pose() first. DB5 ships the unbound
  structures pre-superimposed onto the bound complex, so a search started from the shipped
  coordinates starts at the answer; without the randomization every number here would be
  measuring "did the method fail to move", which is not docking.

THREE ERRORS, REPORTED SEPARATELY, because they have different fixes.
  discretization  the rotation set's floor for this instance: the best RMSD ANY pose built
                  from these rotations could reach. Measured per case, not assumed.
  search          the best pose among EVERY candidate the search generated -- all
                  (rotations x top translations), NOT the top-K after ranking. This
                  distinction is the whole point: a "best in list" taken from a
                  score-ordered short list is already contaminated by the score, and would
                  charge scoring failures to the search. So every candidate is evaluated.
  scoring         rank-1 minus best-among-all-candidates. If this is large, the answer was
                  generated and the score put something else first.

RESCORING ABLATIONS, on the top-N poses of the UU arm. Every arm reranks the SAME pose
list, so the search is held fixed and only the score changes -- the control moves one thing.
  grid    the Katchalski-Katzir shape score that generated the list (the baseline)
  pair    Lennard-Jones + Coulomb at the deposited rotamers, one conformation
  ve      Algorithm 2: exact side-chain repacking by variable elimination
  greedy  the same repacking by best-of-20 restarts, to show what the guarantee buys
  Z       Algorithm 4: -RT ln Z over the whole rotamer ensemble instead of its minimum

FLEXIBLE REFINEMENT, Algorithm 3, applied to the rank-1 UU pose on medium and difficult
cases -- the classes where side chains and backbone actually move.

SCOPE, stated so it cannot be mistaken for more than it is. The search runs on all usable
complexes. The rescoring ablations run on a STRATIFIED SUBSET, because each one rebuilds a
repacking problem per pose; the subset size is printed with the results and the cases are
drawn evenly across classes. A number from this file is a statement about this pipeline --
a shape-complementarity search, a fixed rotamer library, and a Lennard-Jones plus Coulomb
energy with no solvation term. It is not a claim about what a production docking program
would do.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rem.docking import capri, score
from rem.docking.data import Structure, load_case, rmsd
from rem.docking.repack import RepackProblem, build_from_case
from rem.docking.rigid import (RigidSearch, apply_pose, randomize_pose, rotation_set,
                               rotation_set_spacing)

ROTATIONS = 500
SPACING = 1.5
TOP_PER_ROT = 5
KEEP = 50
RESCORE_N = 20
REPACK_RES = 6
N_CHI1, N_CHI2 = 3, 2
OUT = "benchmarks/db5_dock_results.json"


def instance_floor(rotations: np.ndarray, moved: np.ndarray,
                   native: np.ndarray) -> Tuple[float, int]:
    """Best FULL-LIGAND RMSD reachable for this instance: the ligand starts at `moved` and
    the search may apply any rotation in the set plus any translation. The optimal
    translation is the centroid offset, so the floor is a pure rotation comparison."""
    M = moved - moved.mean(axis=0)
    N = native - native.mean(axis=0)
    best, bi = np.inf, -1
    for i, R in enumerate(rotations):
        r = float(np.sqrt(((M @ R.T - N) ** 2).sum(1).mean()))
        if r < best:
            best, bi = r, i
    return best, bi


def centroid_aligned_irmsd(rotations: np.ndarray, moved: np.ndarray, native: np.ndarray,
                           rec_iface: np.ndarray, lmask: np.ndarray,
                           Q: np.ndarray) -> Tuple[float, int]:
    """DIAGNOSTIC ONLY -- this is NOT a lower bound on I-RMSD, and saying so is the point.

    The first version of this function was used as the I-RMSD floor and the pilot caught it
    immediately: it reported a floor of 7.65 A on a case where the search achieved 4.75 A,
    which is impossible for a bound. The error was placing each rotation by its
    RMSD-OPTIMAL translation -- optimal for full-ligand RMSD, which is the centroid offset
    -- and then measuring I-RMSD, a different quantity that a different translation
    minimizes. A quantity computed at the argmin of the WRONG objective is not a bound on
    the right one; it is just one arbitrary pose's score.

    A genuine I-RMSD floor would need a minimization over translations per rotation, which
    costs as much as the docking search itself. So the reported floor is the L-RMSD floor
    from instance_floor(), which IS a valid bound (for a fixed rotation the centroid offset
    minimizes RMSD, and translation is free up to the voxel grid), and search error is
    defined in that same metric. This function is kept only to report the number and to
    keep the reasoning visible.
    """
    from rem.docking.data import superimposed_rmsd
    cn = native.mean(axis=0)
    M = moved - moved.mean(axis=0)
    best, bi = np.inf, -1
    for i, R in enumerate(rotations):
        placed = M @ R.T + cn
        r = float(superimposed_rmsd(np.vstack([rec_iface, placed[lmask]]), Q))
        if r < best:
            best, bi = r, i
    return best, bi


def _as_struct(template: Structure, coords: np.ndarray) -> Structure:
    return Structure(coords, template.atom_names, template.res_ids,
                     template.res_names, template.elements)


def run_arm(rec: Structure, lig: Structure, rotations: np.ndarray, seed: int,
            spacing: float = SPACING, keep: int = KEEP,
            top_per_rot: int = TOP_PER_ROT) -> dict:
    """One docking arm: randomize, search, evaluate EVERY candidate against native.

    I-RMSD is computed for all (rotations x top_per_rot) candidates, not just the ones that
    survive ranking, so the search error is measured on what the search actually generated.
    Only the interface backbone atoms are transformed for that evaluation -- a few hundred
    atoms rather than the whole ligand -- which is what makes evaluating every candidate
    affordable.
    """
    from rem.docking.data import superimposed_rmsd
    from rem import fftcorr

    native = lig.coords.copy()
    masks = capri.interface_mask(rec, lig, native)
    rmask, lmask = masks
    moved, R_app, t_app = randomize_pose(native, seed=seed, max_shift=20.0)
    floor, floor_i = instance_floor(rotations, moved, native)

    srch = RigidSearch(rec, _as_struct(lig, moved), spacing=spacing)
    rec_if = rec.coords[rmask]
    Q = np.vstack([rec_if, native[lmask]])
    diag_irmsd, _fi2 = centroid_aligned_irmsd(rotations, moved, native, rec_if, lmask, Q)
    lig_bb = np.isin(lig.atom_names, capri.BACKBONE)
    nat_bb = native[lig_bb]
    sub_bb = moved[lig_bb]
    sub0 = moved[lmask]                      # interface backbone only
    centre = srch.lig_centre

    t0 = time.perf_counter()
    scores, irms_all, lrms_all, meta = [], [], [], []
    for ri, R in enumerate(rotations):
        S = srch.score_rotation(R)
        for shift, sc in fftcorr.top_translations(S, k=top_per_rot, signed=True):
            t = fftcorr.shift_to_world(shift, spacing)
            sub = (sub0 - centre) @ R.T + centre + t
            bb = (sub_bb - centre) @ R.T + centre + t
            scores.append(sc)
            irms_all.append(superimposed_rmsd(np.vstack([rec_if, sub]), Q))
            lrms_all.append(float(np.sqrt(((bb - nat_bb) ** 2).sum(1).mean())))
            meta.append((ri, np.asarray(shift, dtype=int)))
    secs = time.perf_counter() - t0
    scores = np.asarray(scores)
    irms_all = np.asarray(irms_all); lrms_all = np.asarray(lrms_all)

    def full(idx: int) -> dict:
        ri, shift = meta[idx]
        t = fftcorr.shift_to_world(shift, spacing)
        c = apply_pose(moved, rotations[ri], t, centre=centre)
        return {"coords": c, "rot_index": int(ri), "t": t,
                "grid_score": float(scores[idx]),
                "metrics": capri.capri_metrics(rec, lig, native, c, masks)}

    order = np.argsort(-scores)              # rank by grid score, best first
    rank1_idx = int(order[0])
    best_idx = int(np.argmin(irms_all))
    best_l_idx = int(np.argmin(lrms_all))
    rank1, best = full(rank1_idx), full(best_idx)
    keep_idx = [int(i) for i in order[:keep]]
    poses = [full(i) for i in keep_idx]
    return {"rank1": rank1["metrics"], "best_in_list": best["metrics"],
            "best_rank": int(np.where(order == best_idx)[0][0]) + 1,
            "n_candidates": int(len(scores)),
            "floor_L_rmsd": float(floor), "floor_rotation": int(floor_i),
            "centroid_aligned_I_rmsd_diagnostic": float(diag_irmsd),
            "best_L_rmsd": float(lrms_all[best_l_idx]),
            "rank1_L_rmsd": float(lrms_all[rank1_idx]),
            "grid_shape": list(srch.shape), "box_ok": bool(srch.box_ok),
            "seconds": secs, "n_poses": len(poses),
            "scoring_error": float(rank1["metrics"]["I_rmsd"]
                                   - best["metrics"]["I_rmsd"]),
            "search_error_L": float(lrms_all[best_l_idx] - floor),
            "scoring_error_L": float(lrms_all[rank1_idx] - lrms_all[best_l_idx]),
            "_poses": poses, "_native": native, "_masks": masks,
            # The UNSELECTED candidate set. `poses` above is order[:keep], i.e. chosen BY the
            # grid score, so any ceiling computed on it is a fact about scoring, not search.
            # These expose all n_candidates so a caller can ask what the SEARCH generated.
            "_all": {"grid": scores, "I_rmsd": irms_all, "L_rmsd": lrms_all},
            "_full": full}


def rescore(rec: Structure, lig: Structure, arm: dict, n: int = RESCORE_N) -> dict:
    """Rerank the SAME pose list five ways. Search held fixed; only the score changes."""
    from rem.docking.freeenergy import free_energy
    poses = arm["_poses"][:n]
    native, masks = arm["_native"], arm["_masks"]
    rq = score.charges(rec.res_names, rec.atom_names)
    keys_cache: Dict[int, List[str]] = {}
    cols: Dict[str, List[float]] = {k: [] for k in ("grid", "pair", "ve", "greedy", "Z")}
    tw: List[int] = []
    for pi, p in enumerate(poses):
        lg = _as_struct(lig, p["coords"])
        cols["grid"].append(-p["grid_score"])          # higher grid score = better
        cols["pair"].append(score.pair_energy(
            rec.coords, rec.elements, rq, lg.coords, lg.elements,
            score.charges(lg.res_names, lg.atom_names))["total"])
        prob = build_from_case({"r_b": rec, "l_b": lg}, side="r", bound=True,
                               max_residues=REPACK_RES, n_chi1=N_CHI1, n_chi2=N_CHI2)
        if len(prob.res_keys) < 2:
            for k in ("ve", "greedy", "Z"):
                cols[k].append(cols["pair"][-1])
            continue
        g, _e = prob.to_factorgraph()
        ex = prob.solve_exact(g)
        gr = prob.solve_greedy(g, restarts=20)
        fe = free_energy(prob, energy_graph=g)
        tw.append(ex["treewidth"])
        cols["ve"].append(ex["energy"])
        cols["greedy"].append(gr["energy"])
        cols["Z"].append(fe["F"])
    out: Dict[str, object] = {"n_rescored": len(poses),
                              "treewidth_median": float(np.median(tw)) if tw else None,
                              "treewidth_max": int(max(tw)) if tw else None}
    for k, v in cols.items():
        if len(v) != len(poses):
            continue
        top = int(np.argmin(v))
        m = poses[top]["metrics"]
        out[k] = {"rank1_I_rmsd": m["I_rmsd"], "rank1_quality": m["quality"],
                  "rank1_f_nat": m["f_nat"], "picked_pose": top}
    best_i = int(np.argmin([p["metrics"]["I_rmsd"] for p in poses]))
    out["best_available"] = {"I_rmsd": poses[best_i]["metrics"]["I_rmsd"],
                             "quality": poses[best_i]["metrics"]["quality"],
                             "pose": best_i}
    if len(cols["ve"]) == len(poses) and len(cols["greedy"]) == len(poses):
        gaps = np.array(cols["greedy"]) - np.array(cols["ve"])
        out["greedy_gap_max"] = float(gaps.max())
        out["greedy_gap_n_nonzero"] = int((gaps > 1e-9).sum())
        out["greedy_changed_rank1"] = bool(
            int(np.argmin(cols["greedy"])) != int(np.argmin(cols["ve"])))
    return out


def refine_case(rec: Structure, lig: Structure, arm: dict) -> dict:
    """Algorithm 3 on the rank-1 pose. Reports I-RMSD before and after."""
    from rem.docking.flexible import FlexibleRefiner, pose_set
    p = arm["_poses"][0]
    lg = _as_struct(lig, p["coords"])
    rf = FlexibleRefiner(rec, lg, max_res_per_side=4, n_chi1=N_CHI1, n_chi2=N_CHI2)
    r = rf.refine(pose_set(n_trans=10, trans_step=0.6, n_rot=4, rot_deg=3.0, seed=5))
    m = capri.capri_metrics(rec, lig, arm["_native"], r["coords"], arm["_masks"])
    return {"before_I_rmsd": p["metrics"]["I_rmsd"], "after_I_rmsd": m["I_rmsd"],
            "before_quality": p["metrics"]["quality"], "after_quality": m["quality"],
            "treewidth": int(r["max_treewidth"]), "n_poses": int(r["n_poses"]),
            "energy_gain": float(r["identity_energy"] - r["energy"]),
            "seconds": float(r["seconds"])}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rotations", type=int, default=ROTATIONS)
    ap.add_argument("--top-per-rot", type=int, default=TOP_PER_ROT)
    ap.add_argument("--rescore-per-class", type=int, default=20)
    ap.add_argument("--refine-per-class", type=int, default=10)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--per-class", type=int, default=0,
                    help="stratified subset: this many cases per difficulty class")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args(argv)

    cls = json.load(open("benchmarks/db5_classification.json"))["usable"]
    cls = [e for e in cls if e["class"] in ("rigid", "medium", "difficult")]
    cls.sort(key=lambda e: e["id"])
    if a.per_class:
        picked, seen = [], {}
        for e in cls:
            seen.setdefault(e["class"], []).append(e)
        for c, es in seen.items():
            step = max(1, len(es) // a.per_class)
            picked += es[::step][:a.per_class]
        cls = sorted(picked, key=lambda e: e["id"])
    if a.limit:
        cls = cls[:a.limit]
    if a.nworkers > 1:
        cls = [e for i, e in enumerate(cls) if i % a.nworkers == a.worker]
    rots = rotation_set(a.rotations, seed=1)
    sp = rotation_set_spacing(
        __import__("rem.docking.rigid", fromlist=["x"]).random_quaternions(a.rotations, 1))
    print(f"  {len(cls)} complexes   {a.rotations} rotations   spacing {SPACING} A/voxel")
    print(f"  rotation set nn angle: mean {sp['mean_nn_deg']:.2f} median "
          f"{sp['median_nn_deg']:.2f} max {sp['max_nn_deg']:.2f} deg")

    # stratified subsets, drawn evenly across each class after sorting by id
    by_class: Dict[str, List[str]] = {}
    for e in cls:
        by_class.setdefault(e["class"], []).append(e["id"])
    def stratum(k):
        out = set()
        for c, ids in by_class.items():
            if not ids:
                continue
            step = max(1, len(ids) // max(k, 1))
            out |= set(ids[::step][:k])
        return out
    resc_set, ref_set = stratum(a.rescore_per_class), stratum(a.refine_per_class)
    print(f"  rescoring subset {len(resc_set)} cases; refinement subset {len(ref_set)} "
          f"(medium/difficult only)")
    print(f"  {'case':6s} {'class':10s} {'arm':3s} {'floorL':>6s} {'bestL':>6s} "
          f"{'bestI':>6s} {'rk1I':>6s} {'q_rank1':>10s} {'s':>6s}")

    results, t_start = [], time.perf_counter()
    for n, e in enumerate(cls, 1):
        cid = e["id"]
        try:
            case = load_case(cid)
        except Exception as exc:                       # noqa: BLE001
            results.append({"id": cid, "class": e["class"], "error": str(exc)[:200]})
            continue
        rec_entry: Dict[str, object] = {"id": cid, "class": e["class"],
                                        "irmsd_unbound": e["irmsd"]}
        seed = abs(hash(cid)) % (2 ** 31)
        for arm_tag, r_key, l_key in (("bb", "r_b", "l_b"), ("uu", "r_u", "l_u")):
            try:
                arm = run_arm(case[r_key], case[l_key], rots, seed,
                              top_per_rot=a.top_per_rot)
            except Exception as exc:                   # noqa: BLE001
                rec_entry[arm_tag] = {"error": str(exc)[:200]}
                continue
            print(f"  {cid:6s} {e['class']:10s} {arm_tag:3s} "
                  f"{arm['floor_L_rmsd']:6.2f} {arm['best_L_rmsd']:6.2f} "
                  f"{arm['best_in_list']['I_rmsd']:6.2f} {arm['rank1']['I_rmsd']:6.2f} "
                  f"{arm['rank1']['quality']:>10s} {arm['seconds']:6.1f}"
                  f"  [{arm['n_candidates']} cand]",
                  flush=True)
            if arm_tag == "uu":
                if cid in resc_set:
                    try:
                        rec_entry["rescore"] = rescore(case["r_u"], case["l_u"], arm)
                    except Exception as exc:           # noqa: BLE001
                        rec_entry["rescore"] = {"error": str(exc)[:200]}
                if cid in ref_set and e["class"] in ("medium", "difficult"):
                    try:
                        rec_entry["refine"] = refine_case(case["r_u"], case["l_u"], arm)
                    except Exception as exc:           # noqa: BLE001
                        rec_entry["refine"] = {"error": str(exc)[:200]}
            rec_entry[arm_tag] = {k: v for k, v in arm.items() if not k.startswith("_")}
        results.append(rec_entry)
        if n % 10 == 0:
            json.dump({"config": {"rotations": a.rotations, "spacing": SPACING,
                                  "keep": KEEP, "rescore_n": RESCORE_N,
                                  "repack_residues": REPACK_RES,
                                  "rotation_spacing": sp},
                       "results": results}, open(a.out, "w"), indent=1)
            print(f"  --- {n}/{len(cls)} done, {time.perf_counter()-t_start:.0f}s ---",
                  flush=True)
    json.dump({"config": {"rotations": a.rotations, "spacing": SPACING, "keep": KEEP,
                          "rescore_n": RESCORE_N, "repack_residues": REPACK_RES,
                          "rotation_spacing": sp},
               "results": results}, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}  ({len(results)} complexes, "
          f"{time.perf_counter()-t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

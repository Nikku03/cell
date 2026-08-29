"""Summarize the DB5 docking run: CAPRI accuracy by class, with the errors separated.

Reads the worker shards and prints the tables. Every table says what it is measuring and
what it is NOT; a rate with no denominator and no control is not a result.
"""
from __future__ import annotations

import glob
import json
import sys
from typing import Dict, List

import numpy as np

CLASSES = ("rigid", "medium", "difficult")
QUAL = ("high", "medium", "acceptable", "incorrect")

# CAPRI's L-RMSD thresholds. floor_L is a valid lower bound on the L-RMSD any pose built
# from the rotation set can reach, so comparing it against these says which quality tiers
# the SAMPLING can still reach at all, before search or scoring get a turn.
L_HIGH, L_MEDIUM, L_ACCEPTABLE = 1.0, 5.0, 10.0


def load(pattern: str = "benchmarks/shards/db5_dock_w*.json") -> tuple:
    rows, cfg = [], None
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        cfg = cfg or d["config"]
        rows += d["results"]
    return rows, cfg


def _ok(q: str) -> bool:
    return q in ("high", "medium", "acceptable")


def main() -> int:
    rows, cfg = load(sys.argv[1] if len(sys.argv) > 1 else
                     "benchmarks/shards/db5_dock_w*.json")
    rows = [r for r in rows if "error" not in r]
    print(f"  DB5 docking run: {len(rows)} complexes")
    if cfg:
        sp = cfg.get("rotation_spacing", {})
        print(f"  {cfg['rotations']} rotations (nn angle mean "
              f"{sp.get('mean_nn_deg', float('nan')):.2f} deg), "
              f"{cfg['spacing']} A/voxel, top {cfg.get('keep')} poses kept")
        print(f"  rescoring on {cfg.get('rescore_n')} poses, repacking "
              f"{cfg.get('repack_residues')} interface residues")

    # ---- 1. the bound holds ---------------------------------------------------------
    viol = [(r["id"], a, r[a]["floor_L_rmsd"], r[a]["best_L_rmsd"])
            for r in rows for a in ("bb", "uu")
            if a in r and "error" not in r[a]
            and r[a]["best_L_rmsd"] < r[a]["floor_L_rmsd"] - 1e-6]
    print(f"\n  BOUND CHECK: floor_L <= best_L on every row -- "
          f"{len(viol)} violations of {2*len(rows)}")
    for v in viol[:5]:
        print(f"      VIOLATION {v}")

    # ---- 1b. the discretization floor, per case ------------------------------------------
    # WHAT THIS DOES AND DOES NOT ESTABLISH. floor_L bounds L-RMSD only. CAPRI grants
    # "acceptable" for f_nat >= 0.1 AND (L-RMSD <= 10 OR I-RMSD <= 4), so floor_L > 10
    # closes the L-RMSD route and does NOT by itself close the I-RMSD route -- I-RMSD is
    # superimposed over interface atoms and can be far better than L-RMSD for the same
    # pose. A case is therefore marked floor-limited on the L-route, and the number of
    # floor-limited cases that reached "acceptable" anyway is printed as the check on how
    # conservative that marking is. Claiming more than this is how the first version of
    # this floor got it wrong.
    print(f"\n  DISCRETIZATION FLOOR at {cfg['rotations'] if cfg else '?'} rotations "
          f"(floor_L: the best L-RMSD ANY pose from this rotation set can reach)")
    floor_rows = []
    for r in rows:
        a = r.get("uu")
        if not a or "error" in a:
            continue
        floor_rows.append((r["id"], r["class"], a["floor_L_rmsd"],
                           a["best_L_rmsd"], a["best_in_list"]["I_rmsd"],
                           a["best_in_list"]["quality"], a["rank1"]["quality"]))
    if floor_rows:
        fl = np.array([x[2] for x in floor_rows])
        print(f"      n={len(fl)}   min {fl.min():.2f}   median {np.median(fl):.2f}   "
              f"mean {fl.mean():.2f}   max {fl.max():.2f} A")
        print(f"      tiers the SAMPLING can still reach (floor_L vs CAPRI L-RMSD bars):")
        for name, bar in (("high", L_HIGH), ("medium", L_MEDIUM),
                          ("acceptable", L_ACCEPTABLE)):
            n_ok = int((fl <= bar).sum())
            print(f"        floor_L <= {bar:5.1f} ({name:10s}): {n_ok:3d}/{len(fl)} cases")
        limited = [x for x in floor_rows if x[2] > L_ACCEPTABLE]
        print(f"\n      FLOOR-LIMITED on the L-route (floor_L > {L_ACCEPTABLE} A): "
              f"{len(limited)}/{len(floor_rows)}")
        for cid, cl, f, bL, bI, bq, r1q in limited:
            print(f"        {cid:6s} {cl:10s} floor_L {f:6.2f}  best_L {bL:6.2f}  "
                  f"best_I {bI:6.2f}  best quality {bq}")
        got = [x for x in limited if _ok(x[5])]
        if limited:
            print(f"      of those, reached 'acceptable' anyway via the I-RMSD route: "
                  f"{len(got)}/{len(limited)}"
                  f"  <- if non-zero, the marking is conservative, as designed")
        print(f"      worst floors: " + ", ".join(
            f"{x[0]}({x[2]:.1f})" for x in sorted(floor_rows, key=lambda z: -z[2])[:6]))
    excluded = {x[0] for x in floor_rows if x[2] > L_ACCEPTABLE}

    # ---- 2. CAPRI success by class and arm --------------------------------------------
    print(f"\n  CAPRI success (acceptable or better), rank-1 vs best-available")
    print(f"  HEADLINE excludes the {len(excluded)} floor-limited case(s); they are "
          f"reported separately above.")
    print(f"  {'class':10s} {'n':>4s} | {'BB rank1':>9s} {'BB best':>9s} | "
          f"{'UU rank1':>9s} {'UU best':>9s}")
    for c in CLASSES:
        sub = [r for r in rows if r["class"] == c and r["id"] not in excluded]
        if not sub:
            continue
        cells = []
        for a in ("bb", "uu"):
            ok = [r for r in sub if a in r and "error" not in r[a]]
            if not ok:
                cells += ["-", "-"]
                continue
            r1 = sum(_ok(r[a]["rank1"]["quality"]) for r in ok)
            bs = sum(_ok(r[a]["best_in_list"]["quality"]) for r in ok)
            cells += [f"{r1}/{len(ok)}", f"{bs}/{len(ok)}"]
        print(f"  {c:10s} {len(sub):4d} | {cells[0]:>9s} {cells[1]:>9s} | "
              f"{cells[2]:>9s} {cells[3]:>9s}")

    # ---- 3. the three errors, decomposed ------------------------------------------------
    print(f"\n  ERROR DECOMPOSITION in L-RMSD (the metric where the floor is a valid bound)")
    print(f"  {'class':10s} {'arm':3s} {'floor':>7s} {'best':>7s} {'rank1':>7s}"
          f" | {'search':>7s} {'scoring':>7s}   (medians, A)")
    for c in CLASSES:
        for a in ("bb", "uu"):
            ok = [r[a] for r in rows if r["class"] == c and a in r and "error" not in r[a]]
            if not ok:
                continue
            f = np.median([x["floor_L_rmsd"] for x in ok])
            b = np.median([x["best_L_rmsd"] for x in ok])
            r1 = np.median([x["rank1_L_rmsd"] for x in ok])
            se = np.median([x["search_error_L"] for x in ok])
            sc = np.median([x["scoring_error_L"] for x in ok])
            print(f"  {c:10s} {a:3s} {f:7.2f} {b:7.2f} {r1:7.2f} | {se:7.2f} {sc:7.2f}")

    # ---- 4. rescoring ablation ---------------------------------------------------------
    resc = [r for r in rows if "rescore" in r and "error" not in r["rescore"]]
    print(f"\n  RESCORING the same {cfg.get('rescore_n') if cfg else '?'} poses five ways "
          f"({len(resc)} complexes). Search held fixed; only the score changes.")
    if resc:
        print(f"  {'score':8s} {'median I-RMSD':>14s} {'success':>9s}   what it is")
        what = {"grid": "Katchalski-Katzir shape (the baseline that made the list)",
                "pair": "Lennard-Jones + Coulomb at deposited rotamers",
                "ve": "Algorithm 2: exact repacking by variable elimination",
                "greedy": "the same repacking by best-of-20 restarts",
                "Z": "Algorithm 4: -RT ln Z over the rotamer ensemble"}
        for k in ("grid", "pair", "ve", "greedy", "Z"):
            vals = [r["rescore"][k] for r in resc if k in r["rescore"]]
            if not vals:
                continue
            med = np.median([v["rank1_I_rmsd"] for v in vals])
            ok = sum(_ok(v["rank1_quality"]) for v in vals)
            print(f"  {k:8s} {med:14.2f} {ok:5d}/{len(vals):3d}   {what[k]}")
        best = [r["rescore"]["best_available"] for r in resc]
        print(f"  {'CEILING':8s} {np.median([b['I_rmsd'] for b in best]):14.2f} "
              f"{sum(_ok(b['quality']) for b in best):5d}/{len(best):3d}   "
              f"the best pose present in the rescored set at all")
        gaps = [r["rescore"].get("greedy_gap_max", 0.0) for r in resc]
        nz = [r["rescore"].get("greedy_gap_n_nonzero", 0) for r in resc]
        chg = [r["rescore"].get("greedy_changed_rank1", False) for r in resc]
        tw = [r["rescore"]["treewidth_max"] for r in resc
              if r["rescore"].get("treewidth_max") is not None]
        print(f"\n  VE vs GREEDY on the repacking inside the rescoring:")
        print(f"      poses where greedy MISSED the exact optimum: {sum(nz)} of "
              f"{sum(len([1]) * (cfg.get('rescore_n') or 0) for _ in resc)}")
        print(f"      largest greedy gap seen: {max(gaps) if gaps else 0.0:.6f} kcal/mol")
        print(f"      cases where greedy changed which pose ranked first: "
              f"{sum(chg)} of {len(chg)}")
        if tw:
            print(f"      repacking treewidth: median {np.median(tw):.0f}, max {max(tw)}")

    # ---- 5. flexible refinement -----------------------------------------------------------
    ref = [r for r in rows if "refine" in r and "error" not in r["refine"]]
    print(f"\n  FLEXIBLE REFINEMENT (Algorithm 3) on the rank-1 UU pose, "
          f"{len(ref)} medium/difficult cases")
    if ref:
        b = np.array([r["refine"]["before_I_rmsd"] for r in ref])
        a2 = np.array([r["refine"]["after_I_rmsd"] for r in ref])
        print(f"      I-RMSD before {np.median(b):.2f}  after {np.median(a2):.2f}  "
              f"(median); improved on {(a2 < b - 1e-9).sum()} of {len(ref)}")
        print(f"      median energy gain {np.median([r['refine']['energy_gain'] for r in ref]):.3f}"
              f" kcal/mol; treewidth max {max(r['refine']['treewidth'] for r in ref)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

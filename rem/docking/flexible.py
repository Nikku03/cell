"""Algorithm 3 -- REM-Cluster: flexible refinement by CONDITIONING on the pose.

THE STRUCTURAL PROBLEM, stated before the solution, because the solution is only
interesting once the problem is admitted.

Rigid docking (Algorithm 1) finds a pose with the side chains frozen at their deposited
rotamers. Repacking (Algorithm 2) finds the optimal side chains with the pose frozen. The
thing you actually want is the joint optimum over (pose, all rotamers on BOTH chains), and
that is not the composition of the two: moving the pose changes which rotamers are good,
and changing the rotamers changes which pose is good.

Write the pose as a variable P and the interface side chains as variables X_1..X_n. Every
cross-interface energy term mentions P, so P is adjacent to EVERY X_i. P is a hub, the
graph is a star plus the interface contacts, and the treewidth is n. The governing law
then says the cost is d^n, which is the wall, and no elimination ordering avoids it --
this is a genuine property of the problem, not a deficiency of the ordering heuristic.

THE HONEST ANSWER IS CONDITIONING, not a cleverer contraction. Fix P to each of its d_P
values in turn; conditioned on P, the pose terms become constants and the graph collapses
back to the interface contact graph, whose treewidth is small and MEASURED. Total cost

    d_P  x  d ** treewidth(interface | P)

which is exact -- it is a full enumeration of the outer variable, not a heuristic -- and it
is the textbook cutset-conditioning move. What it buys is not a lower asymptotic cost; it
is that the inner problem is solved to GUARANTEED optimality for every pose, so the poses
are compared on their true best conformations rather than on whatever a local search
happened to reach.

TWO-SIDED, and that is the point of doing this at all. Algorithm 2 repacks one chain
against a frozen partner. Here BOTH chains' interface residues are variables at once, so
receptor and ligand side chains can rearrange around each other. That is what makes the
contact graph genuinely bipartite-plus-intra and what makes its treewidth worth measuring
rather than assuming.

WHAT verify() MUST SHOW -- PREDECLARED, BEFORE ANY NUMBER IS RUN.
  C1  EXACTNESS. On an instance small enough to enumerate, the conditioned elimination
      optimum over the FULL joint space (pose x every rotamer on both chains) must equal
      explicit enumeration of that same space. GATE: |difference| < 1e-9.
  C2  MONOTONICITY. The pose set always contains the identity pose, so refinement can
      never return an energy worse than repacking at the input pose.
      GATE: E_refined <= E_at_input_pose + 1e-9.
  C3  TREEWIDTH IS MEASURED, not assumed, for the TWO-SIDED interface graph, across
      interface sizes. Reported per size; GATE: no instance hits the treewidth wall at the
      sizes benchmarked, and the width is reported whether it is flattering or not.
  C4  POSITIVE CONTROL. Start from a pose displaced 1.0 A from native and refine.
      GATE: the refined pose's RMSD to native must be STRICTLY LESS than the displaced
      pose's. A refiner that cannot walk back a displacement it was handed cannot be
      trusted to improve one it was not.
  C5  NEGATIVE CONTROL. Start from a pose displaced 25 A from native -- outside the pose
      set's reach -- and refine. GATE (inverted): the refined RMSD must remain > 10 A. If
      refinement "recovers" native from 25 A with a pose set that only spans a couple of
      Angstroms, the harness is leaking the answer and C4 means nothing.
  C6  REPORTED, not gated: how much of the refinement gain comes from the pose move and
      how much from the two-sided repacking. Decomposed, because a combined number that
      does not say which half moved is not a result.
      RUN ON BOTH BOUND AND UNBOUND STRUCTURES, because the first run used bound ones and
      the answer was degenerate: exact two-sided repacking contributed 0.0000 kcal/mol and
      100% of the gain was the pose move. That is not a fact about repacking, it is a fact
      about bound structures -- their deposited side chains ARE the crystallographic
      optimum, so rotamer offset 0 is already the answer and there is nothing to find.
      Unbound side chains are in the wrong rotamers, which is what makes a case medium or
      difficult, so the unbound arm is the only one in which the repacking half of C6 can
      be non-zero. verify(bound=False) runs it.
"""
from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rem.factorgraph import FactorGraph
from rem.docking.data import Structure, interface_residues, rmsd, _canonical_ids
from rem.docking.repack import RepackProblem, residue_rotamers
from rem.docking.rigid import apply_pose, quat_to_matrix, random_quaternions
from rem.docking.score import CUTOFF


def _relabel(s: Structure, tag: str) -> Structure:
    """Prefix residue ids so receptor and ligand keys cannot collide when combined.

    DB5 chain letters are not unique across the two components (1ACB's ligand is blank
    unbound), so merging two Structures without relabelling silently fuses residues.
    """
    return Structure(s.coords, s.atom_names,
                     np.array([f"{tag}|{r}" for r in s.res_ids], dtype=object),
                     s.res_names, s.elements)


def _concat(a: Structure, b: Structure) -> Structure:
    return Structure(np.vstack([a.coords, b.coords]),
                     np.concatenate([a.atom_names, b.atom_names]),
                     np.concatenate([a.res_ids, b.res_ids]),
                     np.concatenate([a.res_names, b.res_names]),
                     np.concatenate([a.elements, b.elements]))


def pose_set(n_trans: int = 6, trans_step: float = 0.75, n_rot: int = 4,
             rot_deg: float = 4.0, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Small rigid perturbations around the input pose. Entry 0 is ALWAYS the identity.

    Identity first is what makes C2 (monotonicity) meaningful: the refiner is always
    allowed to decline to move, so it can never do worse than the pose it was given.
    """
    poses: List[Tuple[np.ndarray, np.ndarray]] = [(np.eye(3), np.zeros(3))]
    rng = np.random.default_rng(seed)
    for k in range(n_trans):
        d = rng.normal(size=3)
        poses.append((np.eye(3), trans_step * d / np.linalg.norm(d)))
    for q in random_quaternions(n_rot, seed=seed + 17):
        axis = q[1:]
        nrm = np.linalg.norm(axis)
        if nrm < 1e-12:
            continue
        axis = axis / nrm
        th = np.radians(rot_deg)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        poses.append((np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K),
                      np.zeros(3)))
    return poses


class FlexibleRefiner:
    """Joint (pose, two-sided rotamers) optimisation by conditioning on the pose."""

    def __init__(self, receptor: Structure, ligand: Structure,
                 max_res_per_side: int = 4, n_chi1: int = 3, n_chi2: int = 2,
                 cutoff: float = CUTOFF, iface_cutoff: float = 6.0):
        self.receptor = _relabel(receptor, "R")
        self.ligand_ref = _relabel(ligand, "L")
        self.n_chi1, self.n_chi2, self.cutoff = n_chi1, n_chi2, cutoff
        self.lig_centre = ligand.coords.mean(axis=0)

        self.keys: Dict[str, List[str]] = {}
        for tag, mob, partner in (("R", self.receptor, self.ligand_ref),
                                  ("L", self.ligand_ref, self.receptor)):
            cand = interface_residues(mob, partner, cutoff=iface_cutoff)
            scored = [(len(residue_rotamers(mob, k, n_chi1, n_chi2)), k) for k in cand]
            scored = [(n, k) for n, k in scored if n > 1]
            scored.sort(reverse=True)
            self.keys[tag] = [k for _, k in scored[:max_res_per_side]]

    def ligand_at(self, R: np.ndarray, t: np.ndarray) -> Structure:
        return Structure(apply_pose(self.ligand_ref.coords, R, t, centre=self.lig_centre),
                         self.ligand_ref.atom_names, self.ligand_ref.res_ids,
                         self.ligand_ref.res_names, self.ligand_ref.elements)

    def problem_at(self, R: np.ndarray, t: np.ndarray) -> RepackProblem:
        """Both chains' interface residues mobile at once, conditioned on this pose."""
        lig = self.ligand_at(R, t)
        combined = _concat(self.receptor, lig)
        empty = Structure(np.zeros((0, 3)), np.array([], dtype=object),
                          np.array([], dtype=object), np.array([], dtype=object),
                          np.array([], dtype=object))
        ids = _canonical_ids(combined)
        want = set(self.keys["R"]) | set(self.keys["L"])
        keys = [k for k in dict.fromkeys(ids) if k in want]
        return RepackProblem(combined, empty, keys, self.n_chi1, self.n_chi2, self.cutoff)

    def refine(self, poses: Optional[Sequence] = None) -> dict:
        """Exact inner optimum for EVERY pose; the best pair wins. Cost d_P * d^treewidth."""
        poses = pose_set() if poses is None else list(poses)
        best, rows = None, []
        t0 = time.perf_counter()
        for pi, (R, t) in enumerate(poses):
            prob = self.problem_at(R, t)
            g, edges = prob.to_factorgraph()
            ex = prob.solve_exact(g)
            rows.append({"pose_index": pi, "energy": ex["energy"],
                         "treewidth": ex["treewidth"], "n_edges": len(edges),
                         "n_configs": ex["n_configs"]})
            if best is None or ex["energy"] < best["energy"]:
                best = {"pose_index": pi, "R": R, "t": t, "energy": ex["energy"],
                        "assignment": ex["assignment"], "treewidth": ex["treewidth"],
                        "n_configs": ex["n_configs"]}
        best["seconds"] = time.perf_counter() - t0
        best["n_poses"] = len(poses)
        best["per_pose"] = rows
        best["max_treewidth"] = max(r["treewidth"] for r in rows)
        best["identity_energy"] = rows[0]["energy"]
        best["coords"] = self.ligand_at(best["R"], best["t"]).coords
        return best

    def brute_force_joint(self, poses: Sequence) -> float:
        """Reference for C1: enumerate pose x EVERY rotamer assignment, no elimination."""
        best = np.inf
        for R, t in poses:
            prob = self.problem_at(R, t)
            g, _ = prob.to_factorgraph()
            names = prob.res_keys
            for combo in itertools.product(*[range(len(prob.rot[r])) for r in names]):
                e = prob.energy_of(dict(zip(names, combo)), g)
                if e < best:
                    best = e
        return float(best)


# --------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------

def verify(case_id: str = "1A2K", verbose: bool = True, bound: bool = True) -> dict:
    """Run C1-C6. Bars are fixed in the module docstring, above, before any number.

    bound=False runs the whole suite on the UNBOUND components, which is the only arm in
    which the repacking half of C6 can be non-zero (see the C6 note in the docstring).
    """
    from rem.docking.data import load_case
    say = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    tag = "b" if bound else "u"
    out: Dict[str, object] = {"case": case_id, "bound": bool(bound)}
    case = load_case(case_id)
    rec, lig = case[f"r_{tag}"], case[f"l_{tag}"]
    native = lig.coords.copy()
    say(f"  structures: {'BOUND' if bound else 'UNBOUND'} ({'r_'+tag}, {'l_'+tag})")

    ref = FlexibleRefiner(rec, lig, max_res_per_side=2, n_chi1=3, n_chi2=2)
    small_poses = pose_set(n_trans=2, trans_step=0.75, n_rot=1, rot_deg=4.0)
    say(f"  case {case_id}: interface residues "
        f"R={len(ref.keys['R'])} L={len(ref.keys['L'])}, {len(small_poses)} poses")

    # ---- C1: exactness over the FULL joint space ------------------------------------------
    r1 = ref.refine(small_poses)
    bf = ref.brute_force_joint(small_poses)
    d1 = abs(r1["energy"] - bf)
    out["C1_err"], out["C1"] = float(d1), d1 < 1e-9
    say(f"\n  C1 joint optimum over pose x all rotamers")
    say(f"      conditioned elimination {r1['energy']:.8f}")
    say(f"      explicit enumeration    {bf:.8f}")
    say(f"      |diff| {d1:.3e}   treewidth {r1['max_treewidth']}   "
        f"{'PASS' if out['C1'] else 'FAIL'}")

    # ---- C2: monotonicity, identity pose is always available -------------------------------
    big = FlexibleRefiner(rec, lig, max_res_per_side=4, n_chi1=3, n_chi2=2)
    poses = pose_set(n_trans=6, trans_step=0.75, n_rot=4, rot_deg=4.0)
    r2 = big.refine(poses)
    slack = r2["identity_energy"] - r2["energy"]
    out["C2_gain"], out["C2"] = float(slack), slack >= -1e-9
    say(f"\n  C2 monotonicity: refinement cannot be worse than the input pose")
    say(f"      at the identity pose {r2['identity_energy']:.5f}   "
        f"best over {r2['n_poses']} poses {r2['energy']:.5f}")
    say(f"      gain {slack:+.5f} kcal/mol (must be >= 0)   "
        f"{'PASS' if out['C2'] else 'FAIL'}")

    # ---- C3: MEASURED treewidth of the two-sided interface graph ---------------------------
    say(f"\n  C3 two-sided interface treewidth, MEASURED across sizes")
    say(f"      per side  total res   edges   density   treewidth   configs      ms")
    tw_rows, hit_wall = [], False
    for k in (1, 2, 3, 4, 5, 6):
        try:
            rf = FlexibleRefiner(rec, lig, max_res_per_side=k, n_chi1=3, n_chi2=2)
            p0 = rf.problem_at(np.eye(3), np.zeros(3))
            g, edges = p0.to_factorgraph()
            n = len(p0.res_keys)
            if n < 2:
                continue
            t0 = time.perf_counter()
            ex = p0.solve_exact(g)
            ms = (time.perf_counter() - t0) * 1e3
            dens = 2 * len(edges) / (n * (n - 1))
            tw_rows.append((k, n, len(edges), dens, ex["treewidth"], ex["n_configs"], ms))
            say(f"      {k:8d}  {n:9d}  {len(edges):6d}   {dens:7.2f}   {ex['treewidth']:9d}"
                f"   {ex['n_configs']:.2e}  {ms:6.1f}")
        except MemoryError as e:
            hit_wall = True
            say(f"      {k:8d}  TREEWIDTH WALL: {str(e)[:70]}")
    out["C3_rows"] = tw_rows
    out["C3"] = (not hit_wall) and len(tw_rows) >= 3
    say(f"      C3 {'PASS' if out['C3'] else 'FAIL'}  "
        f"(wall hit: {hit_wall}; {len(tw_rows)} sizes measured)")

    # ---- C4: POSITIVE CONTROL, walk back a 1.0 A displacement -------------------------------
    say(f"\n  C4 POSITIVE CONTROL: refine from a pose displaced 1.0 A from native")
    rng = np.random.default_rng(3)
    d = rng.normal(size=3); d = 1.0 * d / np.linalg.norm(d)
    disp = Structure(lig.coords + d, lig.atom_names, lig.res_ids, lig.res_names,
                     lig.elements)
    rf4 = FlexibleRefiner(rec, disp, max_res_per_side=4, n_chi1=3, n_chi2=2)
    r4 = rf4.refine(pose_set(n_trans=10, trans_step=0.5, n_rot=4, rot_deg=3.0, seed=5))
    before4 = rmsd(disp.coords, native)
    after4 = rmsd(r4["coords"], native)
    out["C4_before"], out["C4_after"] = float(before4), float(after4)
    out["C4"] = after4 < before4
    say(f"      RMSD to native  before {before4:.4f} A  ->  after {after4:.4f} A   "
        f"{'PASS' if out['C4'] else 'FAIL'}")

    # ---- C5: NEGATIVE CONTROL, 25 A away is out of reach --------------------------------------
    say(f"\n  C5 NEGATIVE CONTROL: refine from a pose displaced 25 A from native")
    d5 = rng.normal(size=3); d5 = 25.0 * d5 / np.linalg.norm(d5)
    far = Structure(lig.coords + d5, lig.atom_names, lig.res_ids, lig.res_names,
                    lig.elements)
    rf5 = FlexibleRefiner(rec, far, max_res_per_side=4, n_chi1=3, n_chi2=2)
    r5 = rf5.refine(pose_set(n_trans=10, trans_step=0.5, n_rot=4, rot_deg=3.0, seed=5))
    after5 = rmsd(r5["coords"], native)
    out["C5_after"], out["C5"] = float(after5), after5 > 10.0
    say(f"      RMSD to native  before {rmsd(far.coords, native):.4f} A  ->  "
        f"after {after5:.4f} A   must stay > 10   {'PASS' if out['C5'] else 'FAIL'}")

    # ---- C6: decompose the gain -----------------------------------------------------------
    say(f"\n  C6 where does the refinement gain come from? (reported, not gated)")
    p_id = big.problem_at(np.eye(3), np.zeros(3))
    g_id, _ = p_id.to_factorgraph()
    e_start = p_id.energy_of({r: 0 for r in p_id.res_keys}, g_id)   # deposited rotamers
    e_repack = r2["identity_energy"]                                # repack, pose fixed
    e_joint = r2["energy"]                                          # pose moved too
    out["C6"] = {"deposited": float(e_start), "repack_only": float(e_repack),
                 "joint": float(e_joint),
                 "from_repacking": float(e_start - e_repack),
                 "from_pose_move": float(e_repack - e_joint)}
    say(f"      deposited rotamers, input pose   {e_start:10.4f}")
    say(f"      + two-sided exact repacking      {e_repack:10.4f}   "
        f"({e_start - e_repack:+.4f} from repacking)")
    say(f"      + best pose in the set           {e_joint:10.4f}   "
        f"({e_repack - e_joint:+.4f} from the pose move)")
    tot = e_start - e_joint
    if abs(tot) > 1e-9:
        say(f"      -> {100 * (e_start - e_repack) / tot:.1f}% of the total gain is "
            f"repacking, {100 * (e_repack - e_joint) / tot:.1f}% is the pose move")

    gates = ["C1", "C2", "C3", "C4", "C5"]
    out["all_pass"] = all(bool(out[k]) for k in gates)
    say(f"\n  {'ALL GATES PASS' if out['all_pass'] else 'GATE FAILURE'}: "
        + "  ".join(f"{k}={'pass' if out[k] else 'FAIL'}" for k in gates))
    return out


if __name__ == "__main__":
    import sys
    if "--both" in sys.argv:
        for b in (True, False):
            print(f"\n{'='*90}\n  Algorithm 3 verify -- "
                  f"{'BOUND' if b else 'UNBOUND'} structures\n{'='*90}")
            verify(bound=b)
    else:
        verify(bound="--unbound" not in sys.argv)

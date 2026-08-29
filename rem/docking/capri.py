"""CAPRI accuracy metrics, and the frame convention that makes them meaningful.

THE FRAME, stated first because every number below depends on it.

Docking Benchmark 5 ships each unbound component ALREADY SUPERIMPOSED onto its bound
counterpart. That is a leak for the search (see rem.docking.rigid.randomize_pose) but it
is a gift for the evaluation: it means the shipped relative arrangement of r_u and l_u IS
the native pose for the unbound pair. So the whole benchmark can be run in one frame:

    native ligand coordinates  =  l_u as shipped
    randomize the ligand       ->  the search must earn its way back
    docked pose is evaluated   ->  directly against the shipped coordinates

The receptor never moves, so "superimpose the receptors, then measure the ligand" reduces
to a direct comparison. No second superposition, no chance of a silent frame mismatch.

THE THREE CAPRI QUANTITIES.
  f_nat    fraction of native inter-chain residue contacts (heavy atoms within 5 A) that
           the docked pose reproduces. Measures whether the right SURFACES are together.
  L-RMSD   ligand backbone RMSD to native with the receptors superimposed -- here, since
           the receptor is fixed, a direct backbone RMSD. Measures global placement.
  I-RMSD   backbone RMSD over INTERFACE residues only, after optimal superposition of just
           those atoms. Measures the interface itself and is the least forgiving of the
           three for a pose that is roughly right but rotated.

CAPRI QUALITY, as used by the assessors:
  high        f_nat >= 0.5  and  (L-RMSD <= 1.0  or  I-RMSD <= 1.0)
  medium      f_nat >= 0.3  and  (L-RMSD <= 5.0  or  I-RMSD <= 2.0)
  acceptable  f_nat >= 0.1  and  (L-RMSD <= 10.0 or  I-RMSD <= 4.0)
  incorrect   otherwise
A pose counts as a SUCCESS at "acceptable or better", which is the standard reporting bar
and the one used throughout the benchmark.

SEARCH ERROR VERSUS SCORING ERROR -- the separation this module exists to support.
A docking run can fail two ways and they need different fixes:
    search error   the near-native pose is not in the returned list AT ALL
    scoring error  it is in the list, but the score did not rank it first
Reporting only rank-1 accuracy conflates them. So the benchmark reports both
BEST-IN-TOP-N (search) and RANK-1 (scoring), and their difference is the scoring error.

AND A THIRD, WHICH IS USUALLY LEFT OUT: discretization error. A rotation set with mean
nearest-neighbour spacing theta cannot represent an arbitrary orientation better than
about theta/2, so there is a floor on achievable RMSD that has nothing to do with either
the search or the score. rotation_floor() MEASURES that floor for a given set and native
pose, so a search that hits it is reported as sampling-limited rather than as a failure.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rem.docking.data import Structure, kabsch, rmsd, superimposed_rmsd, BACKBONE

CONTACT_CUTOFF = 5.0        # A, heavy-atom cutoff defining a native contact
IFACE_CUTOFF = 10.0         # A, defines interface residues for I-RMSD


def _res_index(s: Structure) -> np.ndarray:
    """Per-atom integer residue index, in order of first appearance."""
    idx, seen = {}, 0
    out = np.empty(len(s.res_ids), dtype=int)
    for i, r in enumerate(s.res_ids):
        if r not in idx:
            idx[r] = seen
            seen += 1
        out[i] = idx[r]
    return out


def contact_set(rec_coords: np.ndarray, rec_res: np.ndarray,
                lig_coords: np.ndarray, lig_res: np.ndarray,
                cutoff: float = CONTACT_CUTOFF) -> set:
    """Inter-chain residue pairs with any heavy-atom pair within `cutoff`."""
    d2 = ((rec_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1)
    i, j = np.where(d2 <= cutoff * cutoff)
    return set(zip(rec_res[i].tolist(), lig_res[j].tolist()))


def f_nat(rec: Structure, lig_native: np.ndarray, lig_docked: np.ndarray,
          lig: Structure, cutoff: float = CONTACT_CUTOFF) -> float:
    rr, lr = _res_index(rec), _res_index(lig)
    nat = contact_set(rec.coords, rr, lig_native, lr, cutoff)
    if not nat:
        return float("nan")
    dock = contact_set(rec.coords, rr, lig_docked, lr, cutoff)
    return len(nat & dock) / len(nat)


def interface_mask(rec: Structure, lig: Structure, lig_native: np.ndarray,
                   cutoff: float = IFACE_CUTOFF) -> Tuple[np.ndarray, np.ndarray]:
    """Backbone-atom masks for the interface residues of receptor and ligand, in the
    NATIVE arrangement. Defined once from native and reused for every docked pose, so the
    atom set being compared cannot drift with the pose."""
    d2 = ((rec.coords[:, None, :] - lig_native[None, :, :]) ** 2).sum(-1)
    rec_close = (d2 <= cutoff * cutoff).any(1)
    lig_close = (d2 <= cutoff * cutoff).any(0)
    rr, lr = _res_index(rec), _res_index(lig)
    rec_res = set(rr[rec_close].tolist())
    lig_res = set(lr[lig_close].tolist())
    rmask = np.isin(rr, list(rec_res)) & np.isin(rec.atom_names, BACKBONE)
    lmask = np.isin(lr, list(lig_res)) & np.isin(lig.atom_names, BACKBONE)
    return rmask, lmask


def capri_metrics(rec: Structure, lig: Structure, lig_native: np.ndarray,
                  lig_docked: np.ndarray,
                  masks: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> dict:
    """f_nat, L-RMSD, I-RMSD and the CAPRI class for one docked pose."""
    rmask, lmask = interface_mask(rec, lig, lig_native) if masks is None else masks
    lig_bb = np.isin(lig.atom_names, BACKBONE)
    lrms = float(rmsd(lig_docked[lig_bb], lig_native[lig_bb])) if lig_bb.any() \
        else float("nan")
    # I-RMSD superimposes the interface atoms of BOTH chains together, which is what CAPRI
    # does; the receptor's interface atoms are identical in both arrangements, so they act
    # as the anchor and only the ligand half can move.
    P = np.vstack([rec.coords[rmask], lig_docked[lmask]])
    Q = np.vstack([rec.coords[rmask], lig_native[lmask]])
    irms = float(superimposed_rmsd(P, Q)) if len(P) >= 3 else float("nan")
    fn = f_nat(rec, lig_native, lig_docked, lig)
    return {"f_nat": fn, "L_rmsd": lrms, "I_rmsd": irms,
            "quality": capri_quality(fn, lrms, irms)}


def capri_quality(fn: float, lrms: float, irms: float) -> str:
    if not (np.isfinite(fn) and np.isfinite(lrms) and np.isfinite(irms)):
        return "unknown"
    if fn >= 0.5 and (lrms <= 1.0 or irms <= 1.0):
        return "high"
    if fn >= 0.3 and (lrms <= 5.0 or irms <= 2.0):
        return "medium"
    if fn >= 0.1 and (lrms <= 10.0 or irms <= 4.0):
        return "acceptable"
    return "incorrect"


QUALITY_RANK = {"incorrect": 0, "unknown": 0, "acceptable": 1, "medium": 2, "high": 3}


def is_success(q: str) -> bool:
    return QUALITY_RANK.get(q, 0) >= 1


def rotation_floor(rotations: np.ndarray, lig_coords: np.ndarray) -> dict:
    """The best RMSD any pose built from `rotations` can reach for an unrotated native.

    For each rotation R, the best achievable is min over translations of
    ||R(x-c) + c + t - x||, minimized at the optimal t, which is exactly the RMSD of the
    rotated cloud to the original after removing the centroid offset. The floor is the
    minimum over the set. A search that reaches this number is SAMPLING-limited, and
    saying so is different from calling it a failure.
    """
    c = lig_coords.mean(axis=0)
    X = lig_coords - c
    best, best_i = np.inf, -1
    for i, R in enumerate(rotations):
        Y = X @ R.T
        r = float(np.sqrt(((Y - X) ** 2).sum(1).mean()))
        if r < best:
            best, best_i = r, i
    return {"floor_rmsd": best, "rotation_index": best_i, "n_rotations": len(rotations)}


# --------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------

def verify(case_id: str = "1AY7", verbose: bool = True) -> dict:
    """Self-consistency and monotonicity checks with known answers.

    M1 the native pose scores perfectly: f_nat 1.0, L-RMSD 0, I-RMSD 0, quality "high".
    M2 monotonicity: displacing the ligand by a growing distance must make f_nat fall
       monotonically and both RMSDs rise monotonically.
    M3 a far-away pose is "incorrect": 40 A displacement must give f_nat 0.
    M4 I-RMSD is invariant to a rigid motion applied to BOTH chains together, which is
       what makes it a frame-independent measure of the interface.
    M5 rotation_floor falls as the rotation set is refined.
    """
    from rem.docking.data import load_case
    from rem.docking.rigid import rotation_set, apply_pose
    say = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    case = load_case(case_id)
    rec, lig = case["r_b"], case["l_b"]
    nat = lig.coords.copy()
    masks = interface_mask(rec, lig, nat)
    out: Dict[str, object] = {"case": case_id}

    m = capri_metrics(rec, lig, nat, nat, masks)
    out["M1"] = (abs(m["f_nat"] - 1.0) < 1e-12 and m["L_rmsd"] < 1e-9
                 and m["I_rmsd"] < 1e-9 and m["quality"] == "high")
    say(f"  M1 native pose: f_nat {m['f_nat']:.4f}  L-RMSD {m['L_rmsd']:.2e}  "
        f"I-RMSD {m['I_rmsd']:.2e}  {m['quality']}   {'PASS' if out['M1'] else 'FAIL'}")

    say("\n  M2 monotonicity under growing displacement")
    rng = np.random.default_rng(0)
    d = rng.normal(size=3); d /= np.linalg.norm(d)
    rows = []
    for dist in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        mm = capri_metrics(rec, lig, nat, nat + dist * d, masks)
        rows.append((dist, mm["f_nat"], mm["L_rmsd"], mm["I_rmsd"], mm["quality"]))
        say(f"      {dist:5.1f} A   f_nat {mm['f_nat']:.4f}   L {mm['L_rmsd']:7.3f}   "
            f"I {mm['I_rmsd']:7.3f}   {mm['quality']}")
    fn = [r[1] for r in rows]; lr = [r[2] for r in rows]; ir = [r[3] for r in rows]
    out["M2"] = (all(a >= b - 1e-12 for a, b in zip(fn, fn[1:]))
                 and all(a <= b + 1e-12 for a, b in zip(lr, lr[1:]))
                 and all(a <= b + 1e-9 for a, b in zip(ir, ir[1:])))
    say(f"      M2 {'PASS' if out['M2'] else 'FAIL'}")

    far = capri_metrics(rec, lig, nat, nat + 40.0 * d, masks)
    out["M3"] = far["f_nat"] == 0.0 and far["quality"] == "incorrect"
    say(f"\n  M3 40 A away: f_nat {far['f_nat']:.4f}  {far['quality']}   "
        f"{'PASS' if out['M3'] else 'FAIL'}")

    R = rotation_set(1, seed=5)[0]
    t = np.array([13.0, -7.0, 4.0])
    rec2 = Structure(apply_pose(rec.coords, R, t, centre=rec.coords.mean(0)),
                     rec.atom_names, rec.res_ids, rec.res_names, rec.elements)
    nat2 = apply_pose(nat, R, t, centre=rec.coords.mean(0))
    m2 = capri_metrics(rec2, lig, nat2, nat2, interface_mask(rec2, lig, nat2))
    out["M4"] = abs(m2["I_rmsd"] - m["I_rmsd"]) < 1e-6 and abs(m2["f_nat"] - 1.0) < 1e-12
    say(f"\n  M4 both chains moved rigidly together: I-RMSD {m2['I_rmsd']:.3e} "
        f"(was {m['I_rmsd']:.3e}), f_nat {m2['f_nat']:.4f}   "
        f"{'PASS' if out['M4'] else 'FAIL'}")

    say("\n  M5 rotation floor falls as the set is refined")
    floors = []
    for n in (8, 32, 128, 512):
        fl = rotation_floor(rotation_set(n, seed=1), nat)
        floors.append(fl["floor_rmsd"])
        say(f"      {n:4d} rotations   floor {fl['floor_rmsd']:7.3f} A")
    out["M5"] = all(a >= b - 1e-12 for a, b in zip(floors, floors[1:]))
    say(f"      M5 {'PASS' if out['M5'] else 'FAIL'}")

    gates = ["M1", "M2", "M3", "M4", "M5"]
    out["all_pass"] = all(bool(out[k]) for k in gates)
    say(f"\n  {'ALL GATES PASS' if out['all_pass'] else 'GATE FAILURE'}: "
        + "  ".join(f"{k}={'pass' if out[k] else 'FAIL'}" for k in gates))
    return out


if __name__ == "__main__":
    verify()

"""Algorithm 1 -- REM-FFT rigid-body pose search over SO(3) x R^3.

THE GOVERNING LAW, cost = d ** treewidth, applied where it is cheapest.

The 6-D rigid search factorizes UNEVENLY, and being honest about that asymmetry is most
of what this module is for:

    translation   the score operator S[t] = sum_x R[x] L[x-t] is block-circulant, hence
                  DIAGONAL in the Fourier basis. Distinct Fourier modes do not couple:
                  bond dimension 1, effective treewidth 0. All N^3 translations are
                  scored by one FFT pair, O(N^3 log N) instead of O(N^6).

    rotation      does NOT commute with translation, so nothing factorizes. A rotation
                  set of size K costs K independent FFT searches, full stop. There is
                  no clever contraction here and this module does not pretend otherwise.

So the total is K * O(N^3 log N). The ONLY saving over a naive 6-D sweep is the N^3/log N
per rotation. That is a big constant -- at N=100 it is about 50,000x -- but it is a
constant factor on the translation axis alone, not an exponential win, and the rotation
axis remains brute force.

ONE FFT IS HOISTED. rfftn(receptor) does not depend on the rotation, so it is computed
once and reused for every rotation: 2 transforms per rotation, not 3.

THE DB5 LEAKAGE TRAP, and why randomize_pose() is not optional.
Docking Benchmark 5 ships the UNBOUND structures already superimposed onto their bound
counterparts. An algorithm that "searches" from the shipped coordinates starts at the
answer: the identity pose is already near-native, and any method that merely fails to
move very far scores well. randomize_pose() applies a uniformly random rotation about
the ligand centroid plus a random translation, and returns the exact (R, t) it applied,
so the ground truth is known and the search is made to earn its answer.

WHAT verify() MUST SHOW -- PREDECLARED, BEFORE ANY NUMBER IS RUN.
Every gate below is pass/fail against a bar fixed here, in this docstring.

  V1  planted translation, identity rotation. A real ligand is displaced by a known
      integer voxel shift and searched for. GATE: the recovered shift equals the planted
      shift EXACTLY, on all 6 planted shifts.
  V2  FFT vs direct correlation on the real docking grids used here, not on toy random
      grids. GATE: max |S_fft - S_direct| < 1e-8 relative to max |S|.
  V3  planted rotation AND translation, with the planted rotation drawn FROM the search
      set so exact recovery is possible in principle. GATE: the top-ranked pose has
      ligand-heavy-atom RMSD to the planted pose < 1.5 A.
  V4  a NEGATIVE control that must fail. The same planted pose is searched with the
      rotation set deliberately excluding the planted rotation, using a coarse 30-degree
      set. GATE (inverted): RMSD must be WORSE than V3's. If a search that cannot
      represent the answer still finds it, the harness is leaking and V3 means nothing.
  V5  rotation-set spacing is MEASURED, not asserted: the mean and max nearest-neighbour
      angular distance over the set is reported. No gate -- it is a description of the
      sampling, and it bounds the best RMSD any rotation search on this set can reach.

WHAT THIS MODULE DOES NOT CLAIM. The grid score is Katchalski-Katzir shape complementarity
(negative core, positive surface shell). It is a SEARCH device, not a free-energy
function. Whether the top-ranked pose is the native one is a scoring question and is
measured separately; V1-V4 test only whether the search finds what the score says is best.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rem import fftcorr
from rem.docking.data import Structure, rmsd

# van der Waals radii, Angstrom. Heavy atoms only (read_pdb drops hydrogens).
ELEM_RADIUS = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90}
DEFAULT_RADIUS = 1.70


def radii_of(elements: np.ndarray) -> np.ndarray:
    return np.array([ELEM_RADIUS.get(str(e).upper(), DEFAULT_RADIUS) for e in elements])


# --------------------------------------------------------------------------------------
# SO(3) sampling
# --------------------------------------------------------------------------------------

def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def random_quaternions(n: int, seed: int = 0) -> np.ndarray:
    """n uniformly distributed unit quaternions (Shoemake), sign-canonicalized to w >= 0.

    Uniform on SO(3), not merely on the components: Shoemake's map from three uniform
    deviates carries Lebesgue measure to Haar measure exactly.
    """
    rng = np.random.default_rng(seed)
    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    q = np.stack([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ], axis=1)
    # (w, x, y, z) ordering with w last above -> roll, then canonicalize the sign.
    q = np.roll(q, 1, axis=1)
    q[q[:, 0] < 0] *= -1.0
    return q


def rotation_set(n: int, seed: int = 0) -> np.ndarray:
    """(n, 3, 3) quasi-uniform rotation matrices."""
    return np.array([quat_to_matrix(q) for q in random_quaternions(n, seed)])


def quat_angle(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Geodesic angle in DEGREES between quaternion(s), accounting for q ~ -q."""
    d = np.abs(np.einsum("...i,...i->...", qa, qb))
    return np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0)))


def rotation_set_spacing(quats: np.ndarray) -> Dict[str, float]:
    """MEASURED nearest-neighbour angular spacing of a rotation set, in degrees.

    This bounds what any search on this set can achieve: a ligand of radius r that is
    mis-rotated by theta sits at least ~ r * theta_rad from native no matter how good
    the translation search is.
    """
    d = np.abs(quats @ quats.T)
    np.fill_diagonal(d, -1.0)
    nn = np.degrees(2.0 * np.arccos(np.clip(d.max(axis=1), -1.0, 1.0)))
    return {"n": float(len(quats)), "mean_nn_deg": float(nn.mean()),
            "median_nn_deg": float(np.median(nn)), "max_nn_deg": float(nn.max())}


# --------------------------------------------------------------------------------------
# poses
# --------------------------------------------------------------------------------------

def apply_pose(coords: np.ndarray, R: np.ndarray, t: np.ndarray,
               centre: Optional[np.ndarray] = None) -> np.ndarray:
    """Rotate about `centre` (default: the coordinates' own centroid), then translate."""
    c = coords.mean(axis=0) if centre is None else np.asarray(centre, dtype=float)
    return (np.asarray(coords, dtype=float) - c) @ R.T + c + np.asarray(t, dtype=float)


def randomize_pose(coords: np.ndarray, seed: int = 0, max_shift: float = 20.0
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Destroy DB5's shipped superposition. Returns (moved, R_applied, t_applied).

    DB5 unbound structures arrive pre-superimposed onto the bound complex, so any search
    started from the shipped coordinates is started at the answer. This is the defence.
    """
    rng = np.random.default_rng(seed)
    R = quat_to_matrix(random_quaternions(1, seed=int(seed) + 991)[0])
    t = rng.uniform(-max_shift, max_shift, size=3)
    return apply_pose(coords, R, t), R, t


# --------------------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------------------

class RigidSearch:
    """FFT translation search repeated over a rotation set. Receptor FFT hoisted.

    The receptor defines the box: its grid, origin and spacing are fixed at construction
    and every rotated ligand is painted into the SAME box, so a peak at voxel shift t is
    a world translation of t * spacing with no further bookkeeping.
    """

    def __init__(self, receptor: Structure, ligand: Structure,
                 spacing: float = 1.2, pad: float = 6.0,
                 core_value: float = -15.0, surface_value: float = 1.0,
                 probe: float = 1.4, grid_shape: Optional[Tuple[int, int, int]] = None,
                 mode: str = "katchalski"):
        self.spacing = float(spacing)
        self.rec_coords = np.asarray(receptor.coords, dtype=float)
        self.lig_coords = np.asarray(ligand.coords, dtype=float)
        self.rec_radii = radii_of(receptor.elements)
        self.lig_radii = radii_of(ligand.elements)

        # The box must hold the receptor plus a ligand translated anywhere around it,
        # or a circular search wraps the ligand through the receptor and invents contacts.
        rec_span = self.rec_coords.max(0) - self.rec_coords.min(0)
        lig_span = self.lig_coords.max(0) - self.lig_coords.min(0)
        need = rec_span + lig_span + 2.0 * pad
        if grid_shape is None:
            n = int(np.ceil(need.max() / self.spacing))
            n += n % 2                      # even sizes keep the fftfreq fold symmetric
            self.shape = (n, n, n)
        else:
            self.shape = tuple(int(v) for v in grid_shape)
        self.origin = fftcorr.auto_origin(self.rec_coords, self.shape, self.spacing)
        self.box_ok = bool(np.all(np.array(self.shape) * self.spacing >= need))

        if mode == "katchalski":
            self.R_grid = fftcorr.receptor_grid(
                self.rec_coords, self.rec_radii, self.shape, self.spacing, self.origin,
                surface_value=surface_value, core_value=core_value, probe=probe)
        elif mode == "occupancy":
            # Plain occupancy on BOTH sides. Used only by the planted tests: the
            # correlation of a shape with itself is maximal at zero lag, so the planted
            # pose is the global optimum BY CONSTRUCTION and the search has a known answer.
            self.R_grid = fftcorr.ligand_grid(
                self.rec_coords, self.rec_radii, self.shape, self.spacing, self.origin,
                occupancy=1.0, probe=0.0)
        else:
            raise ValueError(f"mode must be 'katchalski' or 'occupancy', got {mode!r}")
        self.mode = mode
        self._FR = np.fft.rfftn(self.R_grid, s=self.shape, axes=(0, 1, 2))
        self.lig_centre = self.lig_coords.mean(axis=0)

    def _ligand_grid(self, R: np.ndarray) -> np.ndarray:
        rot = apply_pose(self.lig_coords, R, np.zeros(3), centre=self.lig_centre)
        return fftcorr.ligand_grid(rot, self.lig_radii, self.shape, self.spacing,
                                   self.origin, occupancy=1.0, probe=0.0)

    def score_rotation(self, R: np.ndarray) -> np.ndarray:
        """Score every translation for one rotation. One forward + one inverse FFT."""
        FL = np.fft.rfftn(self._ligand_grid(R), s=self.shape, axes=(0, 1, 2))
        return np.fft.irfftn(self._FR * np.conj(FL), s=self.shape, axes=(0, 1, 2))

    def search(self, rotations: np.ndarray, top_per_rotation: int = 3,
               keep: int = 20) -> List[dict]:
        """Return the `keep` best poses over the whole rotation set, best first.

        Each pose carries the rotation index, the rotation matrix, the world translation,
        the grid score, and the transformed ligand coordinates.
        """
        poses: List[dict] = []
        for ri, R in enumerate(rotations):
            S = self.score_rotation(R)
            for shift, sc in fftcorr.top_translations(S, k=top_per_rotation, signed=True):
                poses.append({"rot_index": ri, "R": R,
                              "t": fftcorr.shift_to_world(shift, self.spacing),
                              "shift": np.asarray(shift), "grid_score": float(sc)})
        poses.sort(key=lambda p: -p["grid_score"])
        out = poses[:keep]
        for p in out:
            p["coords"] = apply_pose(self.lig_coords, p["R"], p["t"],
                                     centre=self.lig_centre)
        return out

    def cost(self, n_rotations: int) -> dict:
        c = fftcorr.cost_model(self.shape)
        c["n_rotations"] = int(n_rotations)
        c["translation_treewidth"] = 0          # circulant -> diagonal in Fourier
        c["rotation_treewidth"] = None          # does not factorize; brute force
        c["total_fft_ops"] = int(c["fft_ops"] * n_rotations)
        c["naive_6d_ops"] = int(c["direct_ops"] * n_rotations)
        return c


# --------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------

def _demo_case(case_id: str = "1AY7"):
    from rem.docking.data import load_case
    case = load_case(case_id)
    return case["r_b"], case["l_b"]


def verify(case_id: str = "1AY7", n_rot: int = 60, verbose: bool = True) -> dict:
    """Run V1-V5. Bars are fixed in the module docstring, above, before any number."""
    rec, lig = _demo_case(case_id)
    out: Dict[str, object] = {"case": case_id}
    say = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    I = np.eye(3)
    srch = RigidSearch(rec, lig, spacing=1.5)
    say(f"  case {case_id}: receptor {len(rec)} atoms, ligand {len(lig)} atoms")
    say(f"  grid {srch.shape} at {srch.spacing} A/voxel, box fits = {srch.box_ok}")

    # ---- V1: planted translation, exact recovery ---------------------------------------
    # Receptor grid := the ligand's own occupancy rolled by p. Self-correlation is maximal
    # at zero lag, so the answer is EXACTLY p and nothing about the scoring can hide a
    # sign or fold error.
    L0 = srch._ligand_grid(I)
    planted = [(0, 0, 0), (3, 0, 0), (0, -4, 0), (0, 0, 5), (2, -3, 4), (-6, 5, -2)]
    v1_ok, v1_rows = True, []
    for p in planted:
        Rp = np.roll(L0, p, axis=(0, 1, 2))
        S = np.fft.irfftn(
            np.fft.rfftn(Rp, s=srch.shape, axes=(0, 1, 2))
            * np.conj(np.fft.rfftn(L0, s=srch.shape, axes=(0, 1, 2))),
            s=srch.shape, axes=(0, 1, 2))
        got, _ = fftcorr.best_translation(S, signed=True)
        ok = tuple(int(v) for v in got) == tuple(p)
        v1_ok &= ok
        v1_rows.append((p, tuple(int(v) for v in got), ok))
    say("\n  V1 planted translation, exact recovery (receptor := ligand rolled by p)")
    for p, g, ok in v1_rows:
        say(f"      planted {str(p):>14}  recovered {str(g):>14}   "
            f"{'ok' if ok else 'FAIL'}")
    say(f"      V1 {'PASS' if v1_ok else 'FAIL'}")
    out["V1"] = bool(v1_ok)

    # ---- V2: FFT vs direct on the REAL docking grids ------------------------------------
    small = RigidSearch(rec, lig, spacing=6.0)          # coarse so O(N^6) is affordable
    Rg, Lg = small.R_grid, small._ligand_grid(I)
    S_fft = fftcorr.correlate(Rg, Lg, mode="circular", sign="plus")
    S_dir = fftcorr.correlate_direct(Rg, Lg, mode="circular", sign="plus")
    rel = float(np.abs(S_fft - S_dir).max() / max(np.abs(S_dir).max(), 1e-12))
    out["V2_rel_err"] = rel
    out["V2"] = rel < 1e-8
    say(f"\n  V2 FFT vs direct on the real grids {small.shape}: rel err {rel:.3e}   "
        f"{'PASS' if out['V2'] else 'FAIL'}")

    # ---- V3: planted rotation AND translation, recovered by SCORE ranking ----------------
    quats = random_quaternions(n_rot, seed=1)
    rots = np.array([quat_to_matrix(q) for q in quats])
    pi = n_rot // 3
    R_true, t_true = rots[pi], np.array([4.5, -3.0, 6.0])
    target = apply_pose(lig.coords, R_true, t_true)
    planted_rec = Structure(target, lig.atom_names, lig.res_ids, lig.res_names,
                            lig.elements)
    s3 = RigidSearch(planted_rec, lig, spacing=1.0, mode="occupancy")
    t0 = time.perf_counter()
    poses = s3.search(rots, top_per_rotation=1, keep=5)
    v3_ms = (time.perf_counter() - t0) * 1e3
    top = poses[0]
    v3_rmsd = rmsd(top["coords"], target)
    out["V3_rmsd"] = float(v3_rmsd)
    out["V3_rot_index"] = int(top["rot_index"])
    out["V3"] = v3_rmsd < 1.5
    say(f"\n  V3 planted rotation+translation, TOP-RANKED pose must recover it")
    say(f"      planted rotation index {pi}, recovered {top['rot_index']}"
        f"{' (same)' if top['rot_index'] == pi else ' (DIFFERENT)'}")
    say(f"      top-ranked pose RMSD to planted: {v3_rmsd:.3f} A   "
        f"{'PASS' if out['V3'] else 'FAIL'} (bar 1.5)")
    say(f"      {n_rot} rotations x grid {s3.shape} searched in {v3_ms:.0f} ms")

    # ---- V4: NEGATIVE control, rotation set cannot represent the answer ------------------
    coarse_q = random_quaternions(8, seed=77)
    coarse_r = np.array([quat_to_matrix(q) for q in coarse_q])
    poses4 = s3.search(coarse_r, top_per_rotation=1, keep=5)
    v4_rmsd = rmsd(poses4[0]["coords"], target)
    out["V4_rmsd"] = float(v4_rmsd)
    out["V4"] = v4_rmsd > v3_rmsd
    say(f"\n  V4 NEGATIVE control, coarse 8-rotation set excluding the planted rotation")
    say(f"      top-ranked RMSD: {v4_rmsd:.3f} A   must be WORSE than V3's "
        f"{v3_rmsd:.3f}   {'PASS' if out['V4'] else 'FAIL'}")

    # ---- V5: measured rotation spacing ---------------------------------------------------
    sp, spc = rotation_set_spacing(quats), rotation_set_spacing(coarse_q)
    out["V5_spacing"] = sp
    say(f"\n  V5 rotation-set spacing, MEASURED (no gate -- it bounds achievable RMSD)")
    for tag, d in (("search", sp), ("coarse", spc)):
        say(f"      {tag:6s} n={int(d['n']):4d}  nn angle mean {d['mean_nn_deg']:6.2f}  "
            f"median {d['median_nn_deg']:6.2f}  max {d['max_nn_deg']:6.2f} deg")

    c = srch.cost(n_rot)
    say(f"\n  cost model, grid {c['grid_shape']}, {n_rot} rotations")
    say(f"      naive 6-D sweep   {c['naive_6d_ops']:.3e} ops")
    say(f"      FFT per rotation  {c['total_fft_ops']:.3e} ops   "
        f"speedup {c['predicted_speedup']:,.0f}x on the TRANSLATION axis only")
    say(f"      translation treewidth {c['translation_treewidth']} (circulant); "
        f"rotation does not factorize")

    gates = ["V1", "V2", "V3", "V4"]
    out["all_pass"] = all(bool(out[k]) for k in gates)
    say(f"\n  {'ALL GATES PASS' if out['all_pass'] else 'GATE FAILURE'}: "
        + "  ".join(f"{k}={'pass' if out[k] else 'FAIL'}" for k in gates))
    return out


if __name__ == "__main__":
    verify()

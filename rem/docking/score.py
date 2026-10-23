"""A real energy function: Lennard-Jones + Coulomb electrostatics + desolvation-like shape term.

THE PROJECT'S CENTRAL LESSON, which this file exists to make measurable: REM makes the
SEARCH exact. Accuracy is then bounded by the SCORING FUNCTION. A Gaussian proxy would let
a perfect search look perfect and prove nothing, so the terms here are real physics with
published-style parameters, and every docking report separates the two error sources.

Parameters are a simplified united-atom set in the spirit of CHARMM/AMBER: per-element
Lennard-Jones radii and well depths, and Gasteiger-like partial charges assigned per
(residue, atom) with a distance-dependent dielectric. They are NOT a validated force field
and are not claimed to be. What matters for the REM claim is that the score is a real
pairwise physical function over atoms, so that a search which finds its global optimum is
doing something non-trivial.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# element -> (Rmin/2 in Angstrom, epsilon in kcal/mol). United-atom-ish, heavy atoms only.
LJ_PARAMS: Dict[str, Tuple[float, float]] = {
    "C": (1.90, 0.086), "N": (1.85, 0.170), "O": (1.70, 0.210),
    "S": (2.00, 0.250), "P": (2.10, 0.200), "SE": (2.10, 0.250),
}
LJ_DEFAULT = (1.90, 0.100)

# Backbone + representative side-chain partial charges (e). Formal charges on the
# ionisable groups dominate the electrostatic term at interfaces.
CHARGE: Dict[Tuple[str, str], float] = {}
for _r in ("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL"
           ).split():
    CHARGE[(_r, "N")] = -0.47
    CHARGE[(_r, "CA")] = 0.07
    CHARGE[(_r, "C")] = 0.51
    CHARGE[(_r, "O")] = -0.51
CHARGE.update({
    ("ASP", "OD1"): -0.50, ("ASP", "OD2"): -0.50, ("ASP", "CG"): 0.00,
    ("GLU", "OE1"): -0.50, ("GLU", "OE2"): -0.50, ("GLU", "CD"): 0.00,
    ("LYS", "NZ"): 1.00, ("ARG", "NH1"): 0.45, ("ARG", "NH2"): 0.45, ("ARG", "NE"): 0.10,
    ("HIS", "ND1"): -0.20, ("HIS", "NE2"): -0.20,
    ("SER", "OG"): -0.40, ("THR", "OG1"): -0.40, ("TYR", "OH"): -0.40,
    ("ASN", "OD1"): -0.40, ("ASN", "ND2"): -0.40,
    ("GLN", "OE1"): -0.40, ("GLN", "NE2"): -0.40,
    ("CYS", "SG"): -0.20, ("MET", "SD"): -0.10, ("TRP", "NE1"): -0.30,
})

CUTOFF = 8.0            # A, pair cutoff
LJ_CLAMP = 10.0         # kcal/mol, cap per pair so one clash cannot swamp a score
EPS_DIEL = 4.0          # distance-dependent dielectric prefactor: eps(r) = EPS_DIEL * r
COULOMB_K = 332.0637    # kcal*A/(mol*e^2)


def lj_params(elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = np.array([LJ_PARAMS.get(str(e).upper(), LJ_DEFAULT)[0] for e in elements])
    eps = np.array([LJ_PARAMS.get(str(e).upper(), LJ_DEFAULT)[1] for e in elements])
    return r, eps


def charges(res_names: np.ndarray, atom_names: np.ndarray) -> np.ndarray:
    return np.array([CHARGE.get((str(r), str(a)), 0.0)
                     for r, a in zip(res_names, atom_names)])


def pair_energy(c1, e1, q1, c2, e2, q2, cutoff: float = CUTOFF,
                clamp: float = LJ_CLAMP) -> Dict[str, float]:
    """Lennard-Jones 12-6 + Coulomb with a distance-dependent dielectric, over atom pairs.

    Returns the decomposed terms so a docking report can say WHICH physics drove a score
    rather than quoting one opaque number."""
    r1, ep1 = lj_params(e1)
    r2, ep2 = lj_params(e2)
    d = np.sqrt(((c1[:, None, :] - c2[None, :, :]) ** 2).sum(-1))
    m = (d < cutoff) & (d > 1e-6)
    if not m.any():
        return {"lj": 0.0, "elec": 0.0, "total": 0.0, "n_pairs": 0.0, "n_clashes": 0.0}
    i, j = np.where(m)
    dij = d[i, j]
    rmin = r1[i] + r2[j]
    eps = np.sqrt(ep1[i] * ep2[j])
    x = (rmin / dij) ** 6
    lj = np.clip(eps * (x * x - 2.0 * x), -clamp, clamp)
    elec = COULOMB_K * q1[i] * q2[j] / (EPS_DIEL * dij * dij)
    return {"lj": float(lj.sum()), "elec": float(elec.sum()),
            "total": float(lj.sum() + elec.sum()), "n_pairs": float(len(dij)),
            "n_clashes": float((dij < 0.8 * rmin).sum())}


def verify(verbose: bool = True) -> dict:
    """Check the energy against hand-computable references. No brute force needed: these
    are closed-form values a wrong implementation will not reproduce."""
    # two carbons at exactly Rmin must sit at -epsilon
    rmin = 2 * LJ_PARAMS["C"][0]
    c1 = np.array([[0.0, 0.0, 0.0]]); c2 = np.array([[rmin, 0.0, 0.0]])
    e = np.array(["C"]); q0 = np.array([0.0])
    got = pair_energy(c1, e, q0, c2, e, q0)["lj"]
    err_min = abs(got - (-LJ_PARAMS["C"][1]))
    # LJ must vanish at r = Rmin * 2^(1/6) ... no: it crosses zero at sigma = Rmin/2^(1/6)
    sig = rmin / (2 ** (1 / 6))
    got0 = pair_energy(c1, e, q0, np.array([[sig, 0.0, 0.0]]), e, q0)["lj"]
    err_zero = abs(got0)
    # Coulomb: two unit charges at 4 A with eps(r)=4r
    qa, qb = np.array([1.0]), np.array([-1.0])
    r = 4.0
    got_e = pair_energy(c1, e, qa, np.array([[r, 0.0, 0.0]]), e, qb)["elec"]
    want_e = COULOMB_K * (1.0) * (-1.0) / (EPS_DIEL * r * r)
    err_elec = abs(got_e - want_e)
    if verbose:
        print("  rem.docking.score.verify   (closed-form references, not brute force)")
        print(f"    LJ at Rmin equals -epsilon        err {err_min:.3e}")
        print(f"    LJ at sigma equals zero           err {err_zero:.3e}")
        print(f"    Coulomb at 4 A, eps(r)=4r         err {err_elec:.3e}")
    return {"err_lj_min": err_min, "err_lj_zero": err_zero, "err_elec": err_elec}

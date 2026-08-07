"""WHAT DOES ONE MARK COST, AND DOES THE ANSWER DEPEND ON WHICH MARK?

THE ARITHMETIC THE REACH RESULT CHANGES.  `nexus_methyl_propagate` measured the displacement from a
5-methylcytosine falling ~15x within 15 A. If that holds, you never solve a chromosome to place one mark --
you solve a LOCAL REGION, and the chromosome-scale cost is (marks per chromosome) x (one local solve), not
(marks) x (whole structure). Those differ by many orders of magnitude, so the region size is the entire cost
question and it is measurable rather than arguable.

    how large must the solved region be before the answer stops changing?

That is asked here directly: solve inside a region of radius Rreg with the boundary held, sweep Rreg, and
compare against the full-structure solve, which is available exactly. The smallest Rreg that reproduces the
full answer IS the per-mark cost, and it is measured, not chosen.

AND THE ANSWER SHOULD NOT BE THE SAME FOR EVERY MARK, which is the second half of this file.  Methylation is
a STERIC mark: a neutral methyl group that pushes its neighbours. But chromatin's other marks are not like
that, and the difference is not cosmetic:

    5-methylcytosine     +1 carbon on DNA, charge NEUTRAL. Pure sterics. What was measured before.
    histone acetylation  REMOVES a lysine's +1 charge and adds bulk. Acetylation is the mark that loosens
                         the histone tail's grip on DNA, and the grip is electrostatic -- a +1 tail against
                         a phosphate backbone. So the important part of this mark is the CHARGE, not the bulk.
    phosphorylation      ADDS -2. The largest electrostatic perturbation chromatin carries.
    histone methylation  adds bulk and KEEPS the +1 (mono/di/tri-methyl-lysine stays cationic), so it is
                         sterically like 5mC and electrostatically like nothing at all.

THIS MATTERS BECAUSE THE TWO KINDS HAVE ALREADY BEEN MEASURED TO BEHAVE DIFFERENTLY.  `nexus_dna_spheres`
found a CHARGE change captured only 35% of its energy inside 4 A and never reached 90% at any radius,
because the atom count in a shell grows as r^2 while the interaction falls as r^-2. A steric change has no
such tail. So the prediction, stated before running: steric marks will converge at a small region and charge
marks will not, and the cost of "simulating chromatin marks" is really two very different costs.

THE PHYSICS IS THE SAME NETWORK AS BEFORE so the numbers are comparable: the all-atom anisotropic elastic
network of 1KX5, springs from the structure's own packing, exact reference by conjugate gradients with the
rigid modes projected out. Only the FORCE differs between marks:

    steric   Lennard-Jones from the added group, capped, on neighbours within a cutoff
    charge   Coulomb from the charge change on every charged atom, with NEXUS's own eps(r)=r,
             giving U = KE q_i q_j / r^2 and hence a force of -2 KE q_i q_j / r^3

Both are central and pairwise against the marked atom, so net force and torque are exactly zero in both and
neither can produce rigid-body drift that would be mistaken for reach.

CONTROLS
    full_solve          the same mark solved on the whole structure. The reference every region is graded
                        against; without it "the answer stopped changing" only means the region stopped
                        growing, which is not the same thing.
    screened_charge     the charge marks repeated with Debye screening at 150 mM (7.9 A). Real chromatin is
                        in salt. If charge marks need a huge region unscreened and a small one screened,
                        then the cost is set by whether the model includes salt -- a modelling choice, not
                        a property of chromatin, and it should not be reported as one.
    boundary_artifact   a region solved with its boundary fixed is stiffer than the real thing, so it
                        UNDERSTATES displacement near the edge. Convergence is therefore judged on the
                        near-field only (inside Rreg/2), where the boundary cannot be doing the work.

PREDECLARED, before any number:
    steric converges by ~20 A and charge does not
        -> per-mark cost is small for methylation-like marks and large for acetylation/phosphorylation,
           and any chromosome-scale estimate has to be quoted per mark TYPE.
    both converge at a similar small radius
        -> one cost covers all marks and the chromosome estimate is a single number.
    neither converges below the whole structure
        -> locality fails, per-mark cost is the whole-structure cost, and chromosome scale is out of reach
           by this route regardless of hardware.

-> outputs/orphan/chromatin_mark_cost.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexus_dna_spheres import fetch_pdb, load, VDW, KE  # noqa: E402
from nexus_methyl_propagate import (elastic_network, rigid_basis, project, exact_solve,  # noqa: E402
                                    methyl_position, methyl_forces, FCAP)

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
RREG = (8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0)
DEBYE = 7.9
SEED = 41
NEAR_FRAC = 0.5          # convergence judged inside Rreg*NEAR_FRAC, away from the held boundary
CHR22_MB = 50.8
NUC_REPEAT = 200
# charge model for the marked residue, same coarse convention as nexus_dna_spheres
Q_PROT = {"NZ": 0.80, "OG": -0.40, "OG1": -0.40, "OD1": -0.55, "OD2": -0.55, "OE1": -0.55, "OE2": -0.55,
          "O": -0.50, "N": -0.16, "C": 0.50, "OP1": -0.78, "OP2": -0.78, "P": 1.17, "O5'": -0.50,
          "O3'": -0.50, "O2": -0.45, "O4": -0.45, "O6": -0.45, "N2": -0.30, "N4": -0.30, "N6": -0.30,
          "N1": -0.20, "N3": -0.20, "N7": -0.20, "OH": -0.40, "NH1": 0.40, "NH2": 0.40, "CZ": 0.60}


def charges_of(t):
    q = np.zeros(len(t["nm"]))
    for i, (n, r) in enumerate(zip(t["nm"], t["rs"])):
        if r == "HOH":
            q[i] = -0.80
        elif r == "MN":
            q[i] = 2.0
        elif r == "CL":
            q[i] = -1.0
        else:
            q[i] = Q_PROT.get(n, 0.0)
    return q


def charge_forces(co, q, k, dq, debye=None):
    """Force field from changing atom k's charge by dq, under NEXUS's own U = KE q_i q_j / r^2.

    dU/dr = -2 KE q_i q_j / r^3, so the force on atom j is +2 KE q_j dq / r^3 along (r_j - r_k). The
    reaction is placed on atom k, and because every pair is central against k the net force and net torque
    are exactly zero -- the same guarantee the steric mark has, so the two are comparable.
    """
    d = co - co[k]
    r = np.linalg.norm(d, axis=1)
    r = np.maximum(r, 0.5)
    sc = np.ones_like(r) if debye is None else np.exp(-r / debye)
    mag = 2.0 * KE * q * dq * sc / r ** 3
    mag[k] = 0.0
    f = mag[:, None] * (d / r[:, None])
    f[k] = -f.sum(0)
    return f


def region_solve(K, co, f, k, Rreg, Q):
    """Solve inside a ball of radius Rreg around the mark, with everything outside held fixed.

    Holding the exterior is what makes the block non-singular, and it is also the approximation being
    measured: a fixed boundary is stiffer than the real surroundings, so it UNDERSTATES motion near the
    edge. Hence convergence is judged on the near field only.
    """
    d = np.linalg.norm(co - co[k], axis=1)
    inside = np.where(d < Rreg)[0]
    dof = np.sort(np.concatenate([3 * inside + a for a in range(3)]))
    A = K[dof][:, dof].tocsc()
    b = f.ravel()[dof]
    t0 = time.time()
    u_loc = sp.linalg.spsolve(A, b)
    dt = time.time() - t0
    u = np.zeros(K.shape[0])
    u[dof] = u_loc
    return u, len(inside), dt


def near_err(u, u_ex, co, k, Rreg):
    d = np.linalg.norm(co - co[k], axis=1)
    m = d < Rreg * NEAR_FRAC
    if m.sum() < 5:
        return float("nan"), 0
    A = np.linalg.norm(u.reshape(-1, 3)[m], axis=1)
    E = np.linalg.norm(u_ex.reshape(-1, 3)[m], axis=1)
    den = np.abs(E).sum()
    return (float(np.abs(A - E).sum() / den) if den > 0 else float("nan")), int(m.sum())


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("WHAT DOES ONE CHROMATIN MARK COST, AND DOES IT DEPEND ON WHICH MARK?")
    report("=" * 100)
    report("  Displacement from a methyl falls ~15x within 15 A, so a mark should not need the whole")
    report("  structure solved -- only a local region. How large that region must be IS the per-mark cost,")
    report("  and it is measured here against the exact full-structure answer rather than assumed.")
    report("  Marks are not interchangeable: 5mC is STERIC and neutral, acetylation REMOVES a +1 charge,")
    report("  phosphorylation ADDS -2. A charge change was already shown to have a tail a steric change")
    report("  does not have, so the prediction stated before running is that they will cost differently.")

    t = load(fetch_pdb(), keep_water=False)
    co, nm, rs = t["co"], t["nm"], t["rs"]
    rad = np.array([VDW.get(e, 1.70) for e in t["el"]])
    n = len(co)
    K, npair = elastic_network(co)
    Q = rigid_basis(co)
    q = charges_of(t)
    report(f"\n  1KX5 no waters: {n} atoms, {npair} springs. Same network as the propagation run, so the")
    report("  numbers are directly comparable to it.")

    # --- define the marks -------------------------------------------------------------------------------
    dc5 = [i for i in range(n) if rs[i] == "DC" and nm[i] == "C5"]
    lysnz = [i for i in range(n) if rs[i] == "LYS" and nm[i] == "NZ" and not t["dna"][i]]
    serog = [i for i in range(n) if rs[i] in ("SER", "THR") and nm[i] in ("OG", "OG1") and not t["dna"][i]]
    # pick sites in the DNA-histone interface region so every mark is in comparable surroundings
    def most_buried(cands):
        best, bi = -1, cands[0]
        for i in cands:
            d = np.linalg.norm(co - co[i], axis=1)
            c = int((d < 10.0).sum())
            if c > best:
                best, bi = c, i
        return bi

    i_c5 = most_buried(dc5)
    i_nz = most_buried(lysnz)
    i_og = most_buried(serog)

    # steric methyl on cytosine C5
    same = np.linalg.norm(co - co[i_c5], axis=1) < 3.0
    imap = {nm[j]: j for j in np.where(same)[0]}
    imap["C5"] = i_c5
    f_methyl = methyl_forces(co, rad, methyl_position(co, imap), i_c5)

    MARKS = [
        ("5mC          steric, neutral", f_methyl, None),
        ("acetyl-K     +1 -> 0 charge", charge_forces(co, q, i_nz, -0.80), i_nz),
        ("acetyl-K     screened 150mM", charge_forces(co, q, i_nz, -0.80, debye=DEBYE), i_nz),
        ("phospho-S    0 -> -2 charge", charge_forces(co, q, i_og, -2.00), i_og),
        ("phospho-S    screened 150mM", charge_forces(co, q, i_og, -2.00, debye=DEBYE), i_og),
    ]
    SITE = {"5mC          steric, neutral": i_c5}

    report(f"\n  MARK SITES (chosen as the most buried of each type, so surroundings are comparable):")
    report(f"    5mC on DC C5 atom {i_c5} | acetyl on LYS NZ atom {i_nz} | phospho on "
           f"{rs[i_og]} {nm[i_og]} atom {i_og}")

    res = {"n_atoms": n, "n_springs": int(npair), "marks": {}}
    report("\n  REGION SIZE NEEDED. Relative error of the near field (inside Rreg/2, away from the held")
    report("  boundary) against the EXACT full-structure solve. Lower is better; the smallest region that")
    report("  reaches ~0.05 is the per-mark cost.")
    hdr = "    " + f"{'mark':<32}" + "".join(f"{r:>8.0f}A" for r in RREG) + f"{'full |u|max':>13}"
    report(hdr)
    for label, f, site in MARKS:
        k = SITE.get(label, site)
        u_ex, info, resid = exact_solve(K, Q, f)
        if info != 0 or resid > 1e-6:
            report(f"    {label:<32}  exact solve failed (info {info}, resid {resid:.1e}); skipped")
            continue
        umax = float(np.linalg.norm(u_ex.reshape(-1, 3), axis=1).max())
        row, tim, sizes = [], [], []
        for R in RREG:
            u, nin, dt = region_solve(K, co, f, k, R, Q)
            e, _ = near_err(u, u_ex, co, k, R)
            row.append(e)
            tim.append(dt)
            sizes.append(nin)
        report(f"    {label:<32}" + "".join(f"{e:>9.3f}" for e in row) + f"{umax:>13.3e}")
        res["marks"][label.split()[0] + ("_screened" if "screened" in label else "")] = {
            "site": int(k), "near_err": row, "solve_s": tim, "atoms_in_region": sizes,
            "full_umax": umax}

    report("\n  COST OF ONE REGION SOLVE (seconds, sparse direct solve on this box)")
    report("    " + f"{'Rreg':<8}" + "".join(f"{r:>10.0f}A" for r in RREG))
    any_key = next(iter(res["marks"]))
    report("    " + f"{'atoms':<8}" + "".join(f"{s:>11}" for s in res["marks"][any_key]["atoms_in_region"]))
    report("    " + f"{'seconds':<8}" + "".join(f"{s:>11.3f}" for s in res["marks"][any_key]["solve_s"]))

    report("\n  READING")
    def converge_R(key):
        m = res["marks"].get(key)
        if not m:
            return None
        for R, e in zip(RREG, m["near_err"]):
            if np.isfinite(e) and e < 0.05:
                return R
        return None
    r_ster = converge_R("5mC")
    r_ac = converge_R("acetyl-K")
    r_acs = converge_R("acetyl-K_screened")
    r_ph = converge_R("phospho-S")
    r_phs = converge_R("phospho-S_screened")
    report(f"    region needed:  5mC {r_ster}   acetyl {r_ac}   acetyl+salt {r_acs}   "
           f"phospho {r_ph}   phospho+salt {r_phs}")
    if r_ster and (r_ac is None or r_ac > r_ster):
        report("    STERIC AND CHARGE MARKS DO NOT COST THE SAME. A neutral methyl settles in a small")
        report("    region; a charge change does not, for the same reason its ENERGY had no convergent")
        report("    tail -- shell population grows as r^2 while the interaction falls as r^-2. Any")
        report("    chromosome-scale number must therefore be quoted PER MARK TYPE, and a cost derived")
        report("    from methylation alone would understate acetylation and phosphorylation badly.")
    if r_acs and (r_ac is None or r_acs < r_ac):
        report("    AND SALT IS WHAT DECIDES THE CHARGE MARKS. Screened at 150 mM they localise; unscreened")
        report("    they do not. So the cost of an acetylation is set by whether the model includes")
        report("    screening -- a modelling choice, not a fact about chromatin, and it should not be")
        report("    reported as one.")

    OUT.mkdir(parents=True, exist_ok=True)
    res["log"] = log
    json.dump(res, open(OUT / "chromatin_mark_cost.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'chromatin_mark_cost.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

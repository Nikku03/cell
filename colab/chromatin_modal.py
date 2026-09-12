"""STOP INTEGRATING TIME. Diagonalise once, then every mark and every instant is nearly free.

THE WALL THIS ATTACKS.  Sphere tiling made one static solve cheaper -- total arithmetic scales as H^1.2, so
4 A spheres beat 20 A ones sevenfold. But it does nothing about the real cost of 4D, which is not the solve,
it is the NUMBER of solves: ~10^9 timesteps for a microsecond, times ~350M marks genome-wide. Making each
step cheaper cannot fix needing a billion of them. Spatial decomposition is the wrong axis.

THE RIGHT AXIS, and it was already in this repo.  The system being solved is LINEAR: K u = f. A linear system
is never integrated -- it is diagonalised. Write the displacement in the eigenbasis of K and each mode
evolves in closed form,

    u(t) = sum_k  (e_k . f / omega_k^2) (1 - cos(omega_k t)) e_k        (undamped)

so the trajectory at ANY time is an expression, not a simulation. There is no timestep, and t = 1 ns costs
exactly what t = 1 s costs. `rouse_gate` already used this at the chromatin scale (GNM, closed form, no
integrator); it was never carried down to the atomistic marks work, which has been re-solving from scratch
for every mark.

TWO SPEEDUPS, AND THEY COMPOUND
    per mark   the eigenbasis is a property of the STRUCTURE, not of the mark. Diagonalise once, and each
               new mark costs a projection onto k modes -- O(k N) -- instead of a full iterative solve.
               Every one of the ~350M genome-wide marks is a different f on the SAME K.
    per instant  free. The time axis stops being a loop.

WHAT HAS TO BE TRUE FOR IT TO WORK, and it is the only real question here: HOW MANY MODES.  The full basis is
3N modes and computing all of them is worse than solving directly. The bet is that a mark's response lives
almost entirely in the low-frequency, collective modes -- which is also the physical claim, since those are
the soft global motions the earlier runs kept finding. If 200 modes reproduce the exact displacement field,
the method is a several-thousand-fold speedup. If it takes 10,000, it is not a speedup at all.

That is measured here against the exact solve, which is available, so the answer is a number rather than an
argument.

    approx_k(f) = sum_{j<=k} (e_j . f / lambda_j) e_j          the k-mode reconstruction of K^-1 f

CONTROLS
    exact_solve     the same force solved iteratively to 1e-10. Every reconstruction is graded against it,
                    so "converged" cannot mean "stopped changing".
    random_modes    k modes drawn at random from the computed set rather than the lowest k. If those do as
                    well, the gain is dimensionality not physics, and the LOW-frequency claim is wrong.
    two mark types  a steric methyl and a charge change. The earlier mark-cost run found these have very
                    different local/global balance, so a mode count fitted to one may not serve the other,
                    and quoting a single k would hide that.

PREDECLARED, before any number:
    a few hundred modes reach ~5% of the exact field
        -> the trajectory problem dissolves: per-mark cost drops by orders of magnitude AND time becomes
           closed form. This is the speedup NEXUS was supposed to deliver and it is real.
    it takes thousands of modes
        -> the response is not low-rank, the eigendecomposition costs more than it saves, and the honest
           answer is that this route does not beat solving each mark directly.
    low modes beat random modes only slightly
        -> the win is rank, not physics; the method still works but the "collective motion" story is wrong
           and should not be told.

-> outputs/orphan/chromatin_modal.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexus_dna_spheres import fetch_pdb, load, VDW, KE  # noqa: E402
from nexus_methyl_propagate import (elastic_network, rigid_basis, project, exact_solve,  # noqa: E402
                                    methyl_position, methyl_forces)
from chromatin_mark_cost import charges_of, charge_forces, DEBYE  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
NMODE = 300              # how many of the lowest modes to compute; the sweep runs inside this
KS = (10, 25, 50, 100, 150, 200, 300)
SEED = 61
GENOME_MARKS = 3.5e8     # ~42M methylation + ~310M histone marks, diploid
NUCLEOSOMES = 3.1e7


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("STOP INTEGRATING TIME -- diagonalise once, then marks and instants are nearly free")
    report("=" * 100)
    report("  Sphere tiling made one static solve cheaper but did nothing about 4D, whose cost is the")
    report("  NUMBER of solves: ~10^9 steps per microsecond times ~350M marks. Cheaper steps cannot fix")
    report("  needing a billion of them. But the system is LINEAR, and a linear system is diagonalised,")
    report("  not integrated: each mode evolves in closed form, so any instant is an expression. The")
    report("  eigenbasis belongs to the STRUCTURE, so it is computed once and every mark reuses it.")
    report("  The only real question is HOW MANY MODES, and that is measured here against the exact field.")

    t = load(fetch_pdb(), keep_water=False)
    co = t["co"]
    rad = np.array([VDW.get(e, 1.70) for e in t["el"]])
    n = len(co)
    K, npair = elastic_network(co)
    Q = rigid_basis(co)
    report(f"\n  1KX5 no waters: {n} atoms, {3*n} degrees of freedom, {npair} springs")

    # --- the modes: lowest NMODE of K, rigid modes shifted away so they are not returned ----------------
    report(f"\n  Computing the lowest {NMODE} modes (sparse Lanczos)...")
    # K is singular in the six rigid modes, so shift-invert at exactly 0 has nothing to factorise. The fix
    # is a SCALAR shift, which keeps the matrix sparse: factorise K + eps*I and discard the six near-zero
    # modes afterwards.
    #
    # The first version projected the rigid modes out with 1e3 * (Q @ Q.T). Q is 40,875 x 6 and DENSE, so
    # that product is a dense 40,875 x 40,875 matrix -- 1.7e9 entries, ~20 GB -- built as a "sparse" one.
    # It was OOM-killed at 15.6 GB. A projector is an operator and must never be materialised.
    t1 = time.time()
    vals, vecs = eigsh(K.tocsc(), k=NMODE, sigma=-1e-3, which="LM")
    t_eig = time.time() - t1
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    keep = vals > 1e-6           # drops the six rigid modes the scalar shift lets through
    vals, vecs = vals[keep], vecs[:, keep]
    report(f"    {vecs.shape[1]} internal modes in {t_eig:.1f} s | "
           f"eigenvalue range {vals[0]:.3e} to {vals[-1]:.3e}")

    # --- the marks ---------------------------------------------------------------------------------------
    q = charges_of(t)
    dc5 = [i for i in range(n) if t["rs"][i] == "DC" and t["nm"][i] == "C5"]
    i_c5 = dc5[len(dc5) // 2]
    same = np.linalg.norm(co - co[i_c5], axis=1) < 3.0
    imap = {t["nm"][j]: j for j in np.where(same)[0]}
    imap["C5"] = i_c5
    lysnz = [i for i in range(n) if t["rs"][i] == "LYS" and t["nm"][i] == "NZ"]
    i_nz = lysnz[len(lysnz) // 2]
    MARKS = [("5mC     steric", methyl_forces(co, rad, methyl_position(co, imap), i_c5)),
             ("acetyl  charge", charge_forces(co, q, i_nz, -0.80, debye=DEBYE))]

    rng = np.random.default_rng(SEED)
    res = {"n_atoms": n, "n_modes": int(vecs.shape[1]), "t_eig": t_eig, "marks": {}}
    report("\n  RECONSTRUCTION ERROR against the exact solve, using the lowest k modes.")
    report("  random_k draws k modes at random from the same computed set -- if that does as well, the")
    report("  gain is rank rather than physics and the 'soft collective motion' story is wrong.")
    report("    " + f"{'mark':<18}{'arm':<12}" + "".join(f"{k:>9}" for k in KS))
    for name, f in MARKS:
        u_ex, info, resid = exact_solve(K, Q, f)
        if info != 0 or resid > 1e-6:
            report(f"    {name}: exact solve failed; skipped")
            continue
        b = project(Q, f.ravel())
        proj = vecs.T @ b                      # e_j . f for every computed mode
        ne = np.linalg.norm(u_ex)
        low, rnd = [], []
        for k in KS:
            k = min(k, vecs.shape[1])
            u_k = vecs[:, :k] @ (proj[:k] / vals[:k])
            low.append(float(np.linalg.norm(project(Q, u_k) - u_ex) / ne))
            sel = rng.choice(vecs.shape[1], size=k, replace=False)
            u_r = vecs[:, sel] @ (proj[sel] / vals[sel])
            rnd.append(float(np.linalg.norm(project(Q, u_r) - u_ex) / ne))
        report(f"    {name:<18}{'lowest k':<12}" + "".join(f"{e:>9.3f}" for e in low))
        report(f"    {'':<18}{'random k':<12}" + "".join(f"{e:>9.3f}" for e in rnd))
        res["marks"][name.split()[0]] = {"lowest": low, "random": rnd, "ks": list(KS)}

    # --- cost ------------------------------------------------------------------------------------------
    report("\n  COST, and the comparison that matters.")
    report("  Per mark, the modal route is a projection onto k modes: 2*k*3N flops. The iterative route")
    report("  measured in chromatin_sphere_scaling was 3.99e10 flops per mark at its cheapest.")
    ITER_FLOPS = 3.99e10
    for k in (100, 200, 400):
        mf = 2.0 * k * 3 * n
        report(f"    k={k:>4}: {mf:>10.3g} flops/mark   speedup {ITER_FLOPS/mf:>9.0f}x")
    res["iter_flops_per_mark"] = ITER_FLOPS

    def first_k(arr):
        for k, e in zip(KS, arr):
            if e < 0.05:
                return k
        return None
    kk = {m: first_k(v["lowest"]) for m, v in res["marks"].items()}
    report(f"\n  modes needed for 5% accuracy: " + ", ".join(f"{m} {v}" for m, v in kk.items()))

    good = [v for v in kk.values() if v]
    report("\n  READING")
    if good:
        kbest = max(good)
        mf = 2.0 * kbest * 3 * n
        sp_up = ITER_FLOPS / mf
        report(f"    {kbest} modes reach 5% of the exact field, so a mark costs {mf:.3g} flops instead of")
        report(f"    {ITER_FLOPS:.3g} -- a {sp_up:.0f}x speedup, and the eigendecomposition is paid ONCE")
        report(f"    per structure ({t_eig:.0f} s here), not once per mark.")
        gf = GENOME_MARKS * mf
        report(f"    GENOME-WIDE, all {GENOME_MARKS:.2g} marks: {gf:.3g} flops = {gf/1e13:.3g} s on one GPU")
        report(f"    at 10 TFLOPS, versus {GENOME_MARKS*ITER_FLOPS/1e13/86400:.3g} DAYS the iterative way.")
        report("    AND THE TIME AXIS COSTS NOTHING EXTRA. Each mode evolves as a closed-form oscillation,")
        report("    so a trajectory is evaluated at whatever instants are wanted rather than stepped to.")
        report("    The 10^9-timestep multiplier that made 4D impossible does not appear at all.")
        report("    THE LIMIT IS THE HARMONIC APPROXIMATION, and it should be stated with the number: this")
        report("    is exact for the linear network already being solved, so it is a better ALGORITHM for")
        report("    the same equations, not a new physics claim. It cannot describe bond breaking, large")
        report("    conformational change, or anything anharmonic -- for those the modes are wrong however")
        report("    many are kept.")
    else:
        report(f"    NO MODE COUNT UP TO {NMODE} REACHES 5%. The response is not low-rank, the")
        report("    eigendecomposition costs more than it saves, and this route does not beat solving")
        report("    each mark directly. The timestep wall stands.")

    OUT.mkdir(parents=True, exist_ok=True)
    res["log"] = log
    json.dump(res, open(OUT / "chromatin_modal.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'chromatin_modal.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

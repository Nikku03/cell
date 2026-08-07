"""DO MANY SMALL CONNECTED SPHERES BEAT ONE BIG SOLVE? -- the H^2 question, measured.

THE CLAIM UNDER TEST, and it is a good one.  If small spheres are connected well enough to pass their
results and their boundary state to each other, the network of them is equivalent to one large sphere --
without ever paying for a large sphere. `nexus_methyl_propagate` already established the first half: the
same tiling, corrected each sweep against neighbours rather than against itself alone, reached the exact
displacement field to 1 part in 10^7. Connected small spheres ARE the big solve.

So the question left is entirely arithmetic, and on paper it favours small spheres:

    work per iteration ~ (number of spheres) x (atoms per sphere)^2 ~ N x H^3
    iterations to converge ~ L / H          (one-level Schwarz, CG-accelerated, no coarse space)
    TOTAL ~ N x L x H^2

H SQUARED. Smaller spheres should cost LESS in total, not more, because the extra iterations grow only
linearly while the per-iteration cost falls cubically. If that holds, the whole "bigger region is more
accurate but more expensive" framing was the wrong axis, and the right move is small spheres iterated.

A CORRECTION TO SOMETHING I SAID EARLIER, because it changes what this scheme requires.  I described the
working scheme as needing "a whole-structure residual" and contrasted it with local communication. That was
misleading. The residual is f - K u, and K couples only atoms within the network cutoff (6 A). Forming it is
NEAREST-NEIGHBOUR MESSAGE PASSING, not a global reduction. There is no all-to-all step and nothing that
serialises. What the failing scheme lacked was not globality -- it was the boundary FORCES from its
neighbours' current state. Spheres that share coordinates in their overlap but not forces at the seam
converge to a structure cut apart at every boundary. That distinction is the whole difference, and it is a
local one, so the scheme parallelises exactly as the proposal assumes.

WHAT IS MEASURED
    for each sphere size H: the tiling, the iterations to a FIXED tolerance, the wall clock, and an
    implementation-independent FLOP count. Against two references:
        direct      one sparse direct solve of the whole structure -- the "big sphere" being avoided
        cg_plain    unpreconditioned CG, i.e. no spheres at all, to show what the tiling is buying

WHY FLOPS AND NOT ONLY SECONDS, and this is the trap in the measurement.  At H = 4 A a nucleosome needs tens
of thousands of spheres, and a Python loop over tens of thousands of tiny dense solves is dominated by
interpreter overhead rather than by arithmetic. Wall clock would then show small spheres losing badly and
that would be an artefact of the language, not a property of the method. So the arithmetic is counted
directly and BOTH are reported. If they disagree, the disagreement is the finding: the method scales as the
flops say, and the seconds say what this particular implementation does.

PREDECLARED, before any number:
    total flops fall as H shrinks, roughly as H^2
        -> the claim is right. Small connected spheres are cheaper than one big solve as well as equivalent
           to it, and sphere size should be chosen small.
    total flops flatten or rise as H shrinks
        -> the iteration count is eating the savings; there is an optimal H and it is measured here.
    iterations grow faster than L/H
        -> convergence is worse than one-level theory predicts, the H^2 argument fails at its first step,
           and a coarse space (which IS a global operation) becomes necessary after all.

-> outputs/orphan/chromatin_sphere_scaling.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator, splu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexus_dna_spheres import fetch_pdb, load, VDW  # noqa: E402
from nexus_methyl_propagate import (elastic_network, rigid_basis, project, exact_solve,  # noqa: E402
                                    methyl_position, methyl_forces, build_spheres,
                                    partition_weights, RC)

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
HS = (4.0, 5.0, 6.0, 8.0, 12.0, 16.0, 20.0)
TOL = 1e-6
MAXIT = 6000
SEED = 53


def factor_small(K, spheres):
    """Cholesky-style dense inverse per interior block, plus the exact flop cost of building and applying it.

    Counted rather than timed: at small H there are tens of thousands of blocks and a Python loop over them
    measures the interpreter, not the method.
    """
    facs, build_flops, apply_flops = [], 0.0, 0.0
    for inner, _ in spheres:
        dof = np.sort(np.concatenate([3 * inner + a for a in range(3)]))
        m = len(dof)
        A = K[dof][:, dof].toarray()
        A[np.diag_indices_from(A)] += 1e-9
        facs.append((dof, np.linalg.inv(A)))
        build_flops += (2.0 / 3.0) * m ** 3        # one factorisation
        apply_flops += 2.0 * m ** 2                # one solve, per application
    return facs, build_flops, apply_flops


def complete_coverage(co, sph, H, overlap=0.35):
    """Add spheres centred on any atom the grid missed, until every atom is in some interior.

    A cubic grid of spacing Rin covers the INFINITE lattice, but a molecule is not a box: atoms sitting in
    surface pockets and at concave features fall outside every grid ball. At H = 4 A that left 137 atoms
    uncovered and the radius was skipped entirely -- which would have silently removed the smallest sphere
    size from a sweep whose whole purpose is to ask how small spheres can go. Patching by atom-centred
    spheres costs a few extra blocks and makes the small radii testable rather than absent.
    """
    n = len(co)
    Rin = H * (1.0 - overlap)
    covered = np.zeros(n, bool)
    for inner, _ in sph:
        covered[inner] = True
    added = 0
    while not covered.all():
        j = int(np.argmax(~covered))
        d = np.linalg.norm(co - co[j], axis=1)
        inner = np.where(d < Rin)[0]
        if len(inner) == 0:
            inner = np.array([j])
        sph.append((inner, np.where(d < H)[0]))
        covered[inner] = True
        added += 1
    return sph, added


def run_tiling(K, Q, f, co, H, u_ex, nnz):
    sph = build_spheres(co, H)
    n = len(co)
    sph, added = complete_coverage(co, sph, H)
    covered = np.zeros(n, bool)
    for inner, _ in sph:
        covered[inner] = True
    if not covered.all():
        return {"H": H, "skipped": f"{int((~covered).sum())} atoms in no sphere interior"}
    w = partition_weights(sph, n)
    ws = np.sqrt(np.repeat(w, 3))
    t0 = time.time()
    facs, bflops, aflops = factor_small(K, sph)
    t_build = time.time() - t0

    n3 = K.shape[0]

    def M(v):
        out = np.zeros(n3)
        vs = ws * v
        for dof, Ainv in facs:
            out[dof] += Ainv.dot(vs[dof])
        return project(Q, ws * out)

    op = LinearOperator((n3, n3), matvec=lambda v: project(Q, K.dot(project(Q, v))), dtype=float)
    pre = LinearOperator((n3, n3), matvec=M, dtype=float)
    b = project(Q, f.ravel())
    it = {"k": 0}
    t0 = time.time()
    u, info = cg(op, b, M=pre, rtol=TOL, maxiter=MAXIT, callback=lambda x: it.__setitem__("k", it["k"] + 1))
    t_solve = time.time() - t0
    u = project(Q, u)
    err = float(np.linalg.norm(u - u_ex) / max(np.linalg.norm(u_ex), 1e-30))
    # per iteration: one preconditioner application + one sparse matvec (2 flops per nonzero)
    per_it = aflops + 2.0 * nnz
    return {"H": H, "n_spheres": len(sph), "patched": added, "iters": it["k"], "rel_err": err,
            "info": int(info),
            "atoms_per_sphere": float(np.mean([len(i) for i, _ in sph])),
            "build_flops": bflops, "per_iter_flops": per_it,
            "total_flops": bflops + per_it * it["k"],
            "t_build": t_build, "t_solve": t_solve, "t_total": t_build + t_solve}


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("DO MANY SMALL CONNECTED SPHERES BEAT ONE BIG SOLVE?")
    report("=" * 100)
    report("  Established already: the same tiling, with each sphere corrected against its NEIGHBOURS'")
    report("  current state rather than against itself alone, reaches the exact field to 1e-7. Connected")
    report("  small spheres ARE the big solve. What is left is arithmetic, and on paper it favours them:")
    report("  work per iteration ~ N H^3, iterations ~ L/H, so TOTAL ~ N L H^2. Smaller should be cheaper.")
    report("  AND THE COMMUNICATION IS LOCAL. The residual f - K u couples only atoms within the network")
    report(f"  cutoff ({RC:.0f} A) -- nearest-neighbour message passing, no global reduction, nothing that")
    report("  serialises. An earlier note here called it a 'whole-structure residual', which was misleading.")
    report("  What the failing scheme lacked was boundary FORCES from its neighbours, not globality.")

    t = load(fetch_pdb(), keep_water=False)
    co = t["co"]
    rad = np.array([VDW.get(e, 1.70) for e in t["el"]])
    n = len(co)
    K, npair = elastic_network(co)
    Q = rigid_basis(co)
    nnz = K.nnz
    report(f"\n  1KX5 no waters: {n} atoms, {npair} springs, {nnz} matrix nonzeros")

    dc5 = [i for i in range(n) if t["rs"][i] == "DC" and t["nm"][i] == "C5"]
    i_c5 = dc5[len(dc5) // 2]
    same = np.linalg.norm(co - co[i_c5], axis=1) < 3.0
    imap = {t["nm"][j]: j for j in np.where(same)[0]}
    imap["C5"] = i_c5
    f = methyl_forces(co, rad, methyl_position(co, imap), i_c5)

    u_ex, info, resid = exact_solve(K, Q, f)
    if info != 0 or resid > 1e-6:
        report(f"  *** reference solve failed (info {info}, resid {resid:.1e}); nothing is comparable")
        return 1
    report(f"  reference solve converged, residual {resid:.1e}")

    # --- references ------------------------------------------------------------------------------------
    report("\n  REFERENCES")
    t1 = time.time()
    Kc = K.tocsc() + sp.eye(K.shape[0], format="csc") * 1e-8
    lu = splu(Kc)
    _ = lu.solve(f.ravel())
    t_direct = time.time() - t1
    report(f"    direct sparse solve of the whole structure: {t_direct:.2f} s   (the 'big sphere')")
    op = LinearOperator(K.shape, matvec=lambda v: project(Q, K.dot(project(Q, v))), dtype=float)
    itp = {"k": 0}
    t1 = time.time()
    _, _ = cg(op, project(Q, f.ravel()), rtol=TOL, maxiter=MAXIT,
              callback=lambda x: itp.__setitem__("k", itp["k"] + 1))
    t_cg = time.time() - t1
    report(f"    unpreconditioned CG, no spheres at all: {itp['k']} iters, {t_cg:.2f} s   "
           f"({2.0*nnz*itp['k']:.3g} flops)")

    # --- the sweep -------------------------------------------------------------------------------------
    report("\n  SPHERE SIZE SWEEP, all to the same tolerance")
    report(f"    {'H':>5}{'spheres':>10}{'atoms/sph':>11}{'iters':>8}{'rel err':>11}"
           f"{'TOTAL flops':>14}{'seconds':>10}")
    rows = []
    for H in HS:
        r = run_tiling(K, Q, f, co, H, u_ex, nnz)
        if "skipped" in r:
            report(f"    {H:>5.0f}  SKIPPED: {r['skipped']}")
            continue
        rows.append(r)
        report(f"    {H:>5.0f}{r['n_spheres']:>10}{r['atoms_per_sphere']:>11.1f}{r['iters']:>8}"
               f"{r['rel_err']:>11.2e}{r['total_flops']:>14.3g}{r['t_total']:>10.2f}"
               f"{r['patched']:>9}")

    report("\n  READING")
    if len(rows) >= 2:
        sm, lg = rows[0], rows[-1]
        # iteration scaling: theory says iters ~ 1/H for one-level CG
        hs = np.array([r["H"] for r in rows], float)
        it = np.array([r["iters"] for r in rows], float)
        fl = np.array([r["total_flops"] for r in rows], float)
        p_it = float(np.polyfit(np.log(hs), np.log(it), 1)[0])
        p_fl = float(np.polyfit(np.log(hs), np.log(fl), 1)[0])
        report(f"    iterations scale as H^{p_it:+.2f}   (one-level theory predicts H^-1.00)")
        report(f"    TOTAL FLOPS scale as H^{p_fl:+.2f}   (the H^2 claim predicts H^+2.00)")
        if p_fl > 0.5:
            report("    THE CLAIM HOLDS. Total arithmetic FALLS as the spheres shrink, because the extra")
            report("    iterations grow only linearly while per-iteration cost falls cubically. Small")
            report("    connected spheres are not a compromise against a big solve -- they are cheaper AND")
            report("    they reach the same answer, which is what makes the scheme worth having.")
        elif p_fl < -0.5:
            report("    THE CLAIM FAILS: total arithmetic RISES as spheres shrink, so the iteration count is")
            report("    eating the savings and there is a floor on useful sphere size.")
        else:
            report("    TOTAL COST IS ROUGHLY FLAT IN H over this range, so sphere size can be chosen for")
            report("    memory and parallelism rather than for arithmetic.")
        best = min(rows, key=lambda r: r["total_flops"])
        report(f"    cheapest by arithmetic: H = {best['H']:.0f} A at {best['total_flops']:.3g} flops")
        fastest = min(rows, key=lambda r: r["t_total"])
        report(f"    fastest by wall clock:  H = {fastest['H']:.0f} A at {fastest['t_total']:.2f} s "
               f"vs {t_direct:.2f} s for the direct solve")
        if best["H"] != fastest["H"]:
            report("    THOSE DISAGREE, AND THE DISAGREEMENT IS THE POINT. Small H means tens of thousands")
            report("    of tiny dense solves inside a Python loop, so wall clock measures the interpreter")
            report("    rather than the method. The flop column is what a real implementation would pay and")
            report("    the seconds column is what this one does; neither alone is the honest answer.")

    OUT.mkdir(parents=True, exist_ok=True)
    res = {"test": "chromatin_sphere_scaling", "n_atoms": n, "nnz": int(nnz),
           "t_direct": t_direct, "cg_plain_iters": itp["k"], "t_cg_plain": t_cg,
           "rows": rows, "log": log}
    json.dump(res, open(OUT / "chromatin_sphere_scaling.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'chromatin_sphere_scaling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

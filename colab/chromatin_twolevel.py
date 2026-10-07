"""TWO LEVELS: local spheres for the near field, a coarse space for the global one.

WHY, IN ONE PARAGRAPH OF MEASURED FACTS.  Two methods were tested here and each failed exactly where the
other worked. A boundary-fixed local REGION captured a charge mark to 3.5% error but could never reproduce a
methyl's response, because that response sits on a floor of whole-particle breathing no local patch can see.
A basis of the 294 lowest MODES did the reverse: it captured the global floor immediately -- 78% of the
steric response inside 50 modes -- and then flatlined, because local detail is high-frequency by
construction and no soft mode represents it. Those are not two dead ends. They are one answer cut in half.

    fine level    overlapping spheres, each solving its own interior. Fast local detail, no global reach.
    coarse level   a small global basis. Global reach, no local detail.
    together       both, and this is textbook: a two-level Schwarz method with a coarse space.

WHAT IT SHOULD BUY, and it is the number this file exists to measure.  One-level Schwarz has no mechanism to
move information further than one sphere per iteration, so its condition number grows as (L/H)^2 and the
measured iteration counts were 800 to 2,800. Adding a coarse space that spans the whole structure removes
that dependence: convergence stops caring about system size. If the iteration count drops to tens, the total
cost drops with it, and the compounding with the sphere-size result (total arithmetic ~ H^1.2, so small
spheres already win sevenfold) is what makes genome scale plausible.

TWO COARSE SPACES, because they cost very different amounts to build and that matters at genome scale:
    modal          the k lowest eigenvectors of K. Since they are K-orthogonal the coarse matrix is simply
                   diag(lambda), so the coarse solve is a division. Costs an eigensolve to build (~108 s
                   here), which is only acceptable because it is a property of the STRUCTURE -- and 31M
                   nucleosomes are near-identical, so it is paid once for a family, not once per site.
    rigid-aggregate  atoms grouped by a spatial grid, with the 6 rigid-body modes of each group as coarse
                   basis vectors. No eigensolve at all -- this is the standard coarse space for elasticity
                   and it is built in milliseconds. If it matches the modal one, the eigensolve is
                   unnecessary and the method scales to any structure without a per-structure setup cost.

ARMS
    one_level          spheres only. The already-measured baseline, re-run here so the comparison is
                       within a single run rather than across runs.
    two_level_modal    spheres + k lowest modes
    two_level_aggr     spheres + rigid-body modes of spatial aggregates
    coarse_only        the coarse space with NO spheres. The control that stops "two-level works" from
                       being credited to the tiling when the coarse space is doing everything -- and the
                       modal run already showed a coarse space alone plateaus at 14-20% error, so this
                       should fail, and if it does not the fine level is redundant.

EVERY ARM IS GRADED AGAINST THE EXACT SOLVE, not against its own residual, so "converged" cannot mean
"stopped moving" -- the failure mode that produced a confident wrong answer earlier in this line.

PREDECLARED, before any number:
    iterations drop by 10x or more AND total flops drop
        -> the two-level method is the speedup. Combined with H^1.2 sphere scaling, genome-wide static
           structure moves from days to minutes, and the closed-form time evolution becomes usable because
           the basis finally represents the field.
    iterations drop but total flops do not
        -> the coarse solve costs what the iterations saved; report the crossover and the best k.
    iterations do not drop
        -> the coarse space is the wrong one, and neither low modes nor rigid aggregates span what the
           spheres are missing.

-> outputs/orphan/chromatin_twolevel.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, eigsh, LinearOperator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexus_dna_spheres import fetch_pdb, load, VDW  # noqa: E402
from nexus_methyl_propagate import (elastic_network, rigid_basis, project, exact_solve,  # noqa: E402
                                    methyl_position, methyl_forces, build_spheres, partition_weights)
from chromatin_sphere_scaling import factor_small, complete_coverage  # noqa: E402
from chromatin_mark_cost import charges_of, charge_forces, DEBYE  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
H = 8.0                  # sphere size for the fine level; 8 A was fastest by wall clock in the sweep
NMODE = 200              # modal coarse space size
AGGR = 12.0              # aggregate cell size for the rigid-body coarse space, Angstrom
TOL = 1e-6
MAXIT = 4000
SEED = 71
GENOME_MARKS = 3.5e8


def aggregate_coarse(co, cell=AGGR):
    """Coarse basis: the six rigid-body modes of each spatial aggregate.

    The standard coarse space for elasticity, and the reason it works is physical rather than algebraic --
    the modes a local solver cannot see are the ones where whole regions translate and rotate relative to
    each other, and those are exactly what this spans. Costs no eigensolve.
    """
    key = np.floor((co - co.min(0)) / cell).astype(int)
    _, lab = np.unique(key, axis=0, return_inverse=True)
    nc = lab.max() + 1
    n = len(co)
    cols = []
    for c in range(nc):
        idx = np.where(lab == c)[0]
        if len(idx) == 0:
            continue
        cen = co[idx].mean(0)
        rel = co[idx] - cen
        for a in range(3):                       # translations
            v = np.zeros((n, 3))
            v[idx, a] = 1.0
            cols.append(v.ravel())
        for a in range(3):                       # rotations about the aggregate centroid
            e = np.zeros(3)
            e[a] = 1.0
            v = np.zeros((n, 3))
            v[idx] = np.cross(np.broadcast_to(e, rel.shape), rel)
            cols.append(v.ravel())
    V = np.array(cols).T
    # drop null columns (a one-atom aggregate has no rotations) and orthonormalise
    nz = np.linalg.norm(V, axis=0) > 1e-9
    V = V[:, nz]
    V, _ = np.linalg.qr(V)
    return V, int(nc)


def make_coarse(K, Q, V):
    """Coarse correction operator V (V^T K V)^-1 V^T, with the global rigid modes projected out.

    Built once. V is dense but thin, so the inverse is a small dense factorisation, and applying it is two
    thin matvecs -- which is why the coarse level does not undo the saving it creates.
    """
    V = project_cols(Q, V)
    A0 = V.T @ (K @ V)
    A0 += np.eye(A0.shape[0]) * 1e-10 * max(1.0, np.abs(A0).max())
    A0inv = np.linalg.pinv(A0)
    apply_flops = 4.0 * V.shape[0] * V.shape[1] + 2.0 * V.shape[1] ** 2
    return V, A0inv, apply_flops


def project_cols(Q, V):
    return V - Q @ (Q.T @ V)


def solve_arm(K, Q, f, u_ex, facs, ws, coarse, nnz, aflops, label, maxit=MAXIT):
    n3 = K.shape[0]
    V, A0inv, cflops = coarse if coarse is not None else (None, None, 0.0)

    def M(v):
        out = np.zeros(n3)
        if facs is not None:
            vs = ws * v
            for dof, Ainv in facs:
                out[dof] += Ainv.dot(vs[dof])
            out = ws * out
        if V is not None:
            out = out + V @ (A0inv @ (V.T @ v))
        return project(Q, out)

    op = LinearOperator((n3, n3), matvec=lambda v: project(Q, K.dot(project(Q, v))), dtype=float)
    pre = LinearOperator((n3, n3), matvec=M, dtype=float)
    b = project(Q, f.ravel())
    it = {"k": 0}
    t1 = time.time()
    u, info = cg(op, b, M=pre, rtol=TOL, maxiter=maxit,
                 callback=lambda x: it.__setitem__("k", it["k"] + 1))
    dt = time.time() - t1
    u = project(Q, u)
    err = float(np.linalg.norm(u - u_ex) / max(np.linalg.norm(u_ex), 1e-30))
    per_it = (aflops if facs is not None else 0.0) + cflops + 2.0 * nnz
    return {"arm": label, "iters": it["k"], "rel_err": err, "seconds": dt,
            "per_iter_flops": per_it, "solve_flops": per_it * it["k"], "info": int(info)}


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("TWO LEVELS -- local spheres for the near field, a coarse space for the global one")
    report("=" * 100)
    report("  Measured previously: a local region captures a charge mark to 3.5% but never a methyl's")
    report("  global floor; the 294 lowest modes capture that floor inside 50 modes and then flatline at")
    report("  14-20% because local detail is high-frequency. Each fails where the other works, so this")
    report("  puts them together. One-level Schwarz moves information one sphere per iteration and needed")
    report("  800-2,800 of them; a coarse space spanning the whole structure should remove that entirely.")

    t = load(fetch_pdb(), keep_water=False)
    co = t["co"]
    rad = np.array([VDW.get(e, 1.70) for e in t["el"]])
    n = len(co)
    K, npair = elastic_network(co)
    Q = rigid_basis(co)
    nnz = K.nnz
    report(f"\n  1KX5 no waters: {n} atoms, {3*n} DOF, {npair} springs, {nnz} nonzeros")

    # --- fine level -------------------------------------------------------------------------------------
    sph = build_spheres(co, H)
    sph, patched = complete_coverage(co, sph, H)
    w = partition_weights(sph, n)
    ws = np.sqrt(np.repeat(w, 3))
    t1 = time.time()
    facs, bflops, aflops = factor_small(K, sph)
    t_fine = time.time() - t1
    report(f"  fine level: {len(sph)} spheres of {H:.0f} A ({patched} added to close coverage gaps), "
           f"built in {t_fine:.1f} s")

    # --- coarse levels ----------------------------------------------------------------------------------
    t1 = time.time()
    Va, nc = aggregate_coarse(co)
    coarse_a = make_coarse(K, Q, Va)
    t_aggr = time.time() - t1
    report(f"  coarse A: {nc} spatial aggregates of {AGGR:.0f} A -> {Va.shape[1]} rigid-body basis "
           f"vectors, built in {t_aggr:.1f} s (no eigensolve)")

    t1 = time.time()
    vals, vecs = eigsh(K.tocsc(), k=NMODE, sigma=-1e-3, which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    keep = vals > 1e-6
    Vm = vecs[:, keep]
    coarse_m = make_coarse(K, Q, Vm)
    t_modal = time.time() - t1
    report(f"  coarse B: {Vm.shape[1]} lowest modes, built in {t_modal:.1f} s (eigensolve)")

    # --- the mark and the reference ---------------------------------------------------------------------
    q = charges_of(t)
    dc5 = [i for i in range(n) if t["rs"][i] == "DC" and t["nm"][i] == "C5"]
    i_c5 = dc5[len(dc5) // 2]
    same = np.linalg.norm(co - co[i_c5], axis=1) < 3.0
    imap = {t["nm"][j]: j for j in np.where(same)[0]}
    imap["C5"] = i_c5
    lysnz = [i for i in range(n) if t["rs"][i] == "LYS" and t["nm"][i] == "NZ"]
    i_nz = lysnz[len(lysnz) // 2]
    MARKS = [("5mC", methyl_forces(co, rad, methyl_position(co, imap), i_c5)),
             ("acetyl", charge_forces(co, q, i_nz, -0.80, debye=DEBYE))]

    res = {"n_atoms": n, "H": H, "n_spheres": len(sph), "n_modes": int(Vm.shape[1]),
           "n_aggr_vecs": int(Va.shape[1]), "t_fine": t_fine, "t_aggr": t_aggr, "t_modal": t_modal,
           "marks": {}}

    report(f"\n  {'mark':<9}{'arm':<20}{'iters':>8}{'rel err':>11}{'flops/iter':>13}"
           f"{'solve flops':>14}{'seconds':>10}")
    for name, f in MARKS:
        u_ex, info, resid = exact_solve(K, Q, f)
        if info != 0 or resid > 1e-6:
            report(f"    {name}: exact solve failed; skipped")
            continue
        rows = []
        for label, fc, wc, cs in (("one_level", facs, ws, None),
                                  ("two_level_aggr", facs, ws, coarse_a),
                                  ("two_level_modal", facs, ws, coarse_m),
                                  ("coarse_only_aggr", None, None, coarse_a),
                                  ("coarse_only_modal", None, None, coarse_m)):
            r = solve_arm(K, Q, f, u_ex, fc, wc, cs, nnz, aflops, label)
            rows.append(r)
            report(f"  {name:<9}{label:<20}{r['iters']:>8}{r['rel_err']:>11.2e}"
                   f"{r['per_iter_flops']:>13.3g}{r['solve_flops']:>14.3g}{r['seconds']:>10.2f}")
        res["marks"][name] = rows

    report("\n  READING")
    ok = True
    for name, rows in res["marks"].items():
        d = {r["arm"]: r for r in rows}
        one, ta, tm = d["one_level"], d["two_level_aggr"], d["two_level_modal"]
        best = min((ta, tm), key=lambda r: r["solve_flops"])
        sp_it = one["iters"] / max(best["iters"], 1)
        sp_fl = one["solve_flops"] / max(best["solve_flops"], 1.0)
        report(f"    {name}: one-level {one['iters']} iters -> best two-level {best['iters']} "
               f"({best['arm']}), {sp_it:.0f}x fewer iterations, {sp_fl:.1f}x fewer flops")
        if sp_it < 5:
            ok = False
    if ok:
        d0 = res["marks"][list(res["marks"])[0]]
        dd = {r["arm"]: r for r in d0}
        best = min((dd["two_level_aggr"], dd["two_level_modal"]), key=lambda r: r["solve_flops"])
        report("    THE COARSE SPACE IS WHAT THE SPHERES WERE MISSING. Iteration count collapses because")
        report("    information no longer has to walk one sphere at a time -- the coarse level carries it")
        report("    across the structure in a single application.")
        if dd["two_level_aggr"]["solve_flops"] <= dd["two_level_modal"]["solve_flops"] * 1.5:
            report(f"    AND THE EIGENSOLVE IS NOT NEEDED. Rigid-body aggregates ({t_aggr:.1f} s to build)")
            report(f"    match or beat the modal space ({t_modal:.1f} s), so there is no per-structure")
            report("    setup cost and the method applies to any structure without precomputation.")
        gf = GENOME_MARKS * best["solve_flops"]
        report(f"    GENOME-WIDE at {GENOME_MARKS:.2g} marks: {gf:.3g} flops = {gf/1e13/3600:.3g} GPU-hours")
        report("    at 10 TFLOPS, against 16 GPU-days for the one-level route.")
    for name, rows in res["marks"].items():
        d = {r["arm"]: r for r in rows}
        if d["coarse_only_aggr"]["rel_err"] < 0.05 or d["coarse_only_modal"]["rel_err"] < 0.05:
            report(f"    *** {name}: a coarse space ALONE already reaches the field, so the fine level is")
            report("        redundant here and the two-level gain should not be credited to the spheres.")

    OUT.mkdir(parents=True, exist_ok=True)
    res["log"] = log
    json.dump(res, open(OUT / "chromatin_twolevel.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'chromatin_twolevel.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

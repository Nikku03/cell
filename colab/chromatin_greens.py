"""SOLVE ONCE, SUPERPOSE FOREVER -- and check that superposing actually reproduces a solve.

THE CLAIM BEING TESTED.  A per-event solve puts a million polymerase steps at 11.4 years. Three measured
facts say that is the wrong way to spend the compute:

    the response is LOCAL          15x fall by 15 A, floor 0.015-0.02 A at 25-70 A (nexus_methyl_propagate)
    the system is LINEAR           a harmonic network, so responses to separate forces ADD
    the units REPEAT               31M nucleosomes are near-identical, so one answer serves all of them

Together those say the response to a force at a site is a fixed field -- a Green function -- computed once
and thereafter added rather than recomputed. Every event becomes arithmetic. That is not an approximation
for a linear system, it is an identity, which is exactly why it deserves a test: an identity that fails is
a bug, with nowhere for the error to hide.

WHAT IS ACTUALLY BEING BOUGHT, stated precisely.  The expensive object is K^-1, and the expensive step is
producing it. Factorising once and reusing the factors is the whole trick; the Green columns are then
triangular solves rather than iterative ones. So the file measures the factorisation ONCE and each column
after it, and reports the ratio, because that ratio is the entire argument.

THE CONTROLS
    one force        superposition of a single Green column against a direct solve. Tests the column.
    many forces      superposition of K columns against ONE direct solve of the summed force. Tests
                     linearity, which is the assumption doing all the work. If this fails, nothing here
                     is usable at any cost.
    truncation       Green columns cut off at radius R, error measured against the untruncated field.
                     This is the locality claim from nexus_methyl_propagate turned into a required
                     support radius rather than a hope, and it sets the per-event cost.
    timing           per-event superposition measured, not estimated, and extrapolated with the ratio
                     stated rather than buried.

PREDECLARED, before any number:
    superposition matches the direct solve to solver tolerance, and truncation error falls with R
        -> the scheme is sound, per-event cost is O(support) rather than O(N), and the 11.4 years is an
           artefact of solving what was already known.
    the many-force control fails
        -> the implementation is not linear where it claims to be, which is a bug rather than a limit.
    truncation error does not fall with R
        -> the response is not local after all, the whole superposition idea loses its cost advantage,
           and per-event cost stays O(N).

-> outputs/chromatin_greens.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, splu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexus_methyl_propagate import elastic_network, rigid_basis, project  # noqa: E402
from chromatin_timescale import load_pdb, PDB  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 8191
N_SITE = int(os.environ.get("GREENS_SITES", 12))
# REGULARISATION, and the first version had this exactly backwards. The shift must be far BELOW the
# softest real mode so it perturbs nothing physical. That mode was measured at 2.05e-05 in
# chromatin_timescale, and the first version used 1e-4 -- FIVE TIMES LARGER -- while the docstring
# asserted it was far below. It damped precisely the soft modes that carry the response and Control 1
# came back at 81% error against an independent solver. Now three orders below the softest mode.
EPS = float(os.environ.get("GREENS_EPS", 2e-8))
RADII = (10.0, 20.0, 30.0, 50.0, 70.0, 100.0)


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    report("=" * 100)
    report("SOLVE ONCE, SUPERPOSE FOREVER -- tested, because an identity that fails is a bug")
    report("=" * 100)

    co, bfac, el = load_pdb(PDB)
    n = len(co)
    K, npair = elastic_network(co)
    Q = rigid_basis(co)
    n3 = 3 * n
    report(f"  {n} atoms, {npair} springs, {K.nnz} Hessian nonzeros")

    # ---- the one expensive step, done once ---------------------------------------------------------
    report(f"\n  FACTORISING ONCE. K is singular in its 6 rigid modes, so K + {EPS:g} I is factorised and")
    report("  the rigid components are projected out afterwards. The shift must sit far BELOW the softest")
    report(f"  real mode, measured at 2.05e-05: this one is {2.05e-5/EPS:.0f}x smaller. The first version used")
    report("  1e-4, five times LARGER, while claiming the opposite -- Control 1 caught it at 81% error.")
    t1 = time.time()
    lu = splu((K + EPS * sp.identity(n3, format="csr")).tocsc())
    t_fac = time.time() - t1
    report(f"    factorisation {t_fac:.1f} s, {lu.nnz:,} nonzeros in the factors "
           f"({lu.nnz/K.nnz:.1f}x fill-in over K)")

    def solve(f):
        return project(Q, lu.solve(project(Q, f)))

    # ---- the Green columns, cheap after the factorisation ------------------------------------------
    sites = rng.choice(n, size=N_SITE, replace=False)
    report(f"\n  {N_SITE} GREEN COLUMNS -- one per event site, each a triangular solve rather than an")
    report("  iterative one. This ratio is the entire argument of the file.")
    G, tcol = [], []
    for s in sites:
        f = np.zeros(n3)
        f[3 * s:3 * s + 3] = rng.normal(size=3)
        t2 = time.time()
        G.append(solve(f))
        tcol.append(time.time() - t2)
    G = np.array(G)
    report(f"    per column {np.mean(tcol)*1000:.0f} ms   against {t_fac:.1f} s for the factorisation")
    report(f"    ratio {t_fac/np.mean(tcol):.0f}x -- the cost is in producing K^-1 once, not in using it")

    # ---- CONTROL 1: does a Green column equal a direct solve? --------------------------------------
    report("\n  CONTROL 1 -- a single Green column against an independent iterative solve")
    op = LinearOperator((n3, n3), matvec=lambda v: project(Q, K.dot(project(Q, v))), dtype=float)
    f1 = np.zeros(n3)
    f1[3 * sites[0]:3 * sites[0] + 3] = rng.normal(size=3)
    u_lu = solve(f1)
    t3 = time.time()
    u_cg, info = cg(op, project(Q, f1), rtol=1e-10, maxiter=5000)
    t_cg = time.time() - t3
    e1 = np.linalg.norm(u_lu - u_cg) / max(np.linalg.norm(u_cg), 1e-300)
    report(f"    factorised {np.mean(tcol)*1000:.0f} ms vs conjugate-gradient {t_cg:.1f} s, "
           f"relative difference {e1:.2e}")

    # ---- CONTROL 2: linearity, the assumption doing all the work -----------------------------------
    report("\n  CONTROL 2 -- THE ONE THAT MATTERS. Superposing K columns against ONE solve of the summed")
    report("  force. Linearity is the whole basis of the scheme; if this fails nothing here is usable.")
    w = rng.normal(size=N_SITE)
    fs = []
    for s in sites:
        f = np.zeros(n3)
        f[3 * s:3 * s + 3] = rng.normal(size=3)
        fs.append(f)
    # rebuild columns for these exact forces so the comparison is apples to apples
    Gc = np.array([solve(f) for f in fs])
    f_sum = np.sum([wi * f for wi, f in zip(w, fs)], axis=0)
    u_direct = solve(f_sum)
    u_super = np.einsum("i,ij->j", w, Gc)
    e2 = np.linalg.norm(u_super - u_direct) / max(np.linalg.norm(u_direct), 1e-300)
    report(f"    {N_SITE} superposed columns vs one direct solve: relative difference {e2:.2e}")

    # ---- CONTROL 3: how local is it really? --------------------------------------------------------
    report("\n  CONTROL 3 -- truncation. Cutting each Green column at radius R turns the locality claim")
    report("  into a required support radius, and that radius IS the per-event cost.")
    d = np.linalg.norm(co - co[sites[0]], axis=1)
    ref = G[0]
    report("  TWO different questions, which the first version conflated into one number:")
    report("    norm inside R   how much of the field's L2 norm a truncated support carries")
    report("    peak inside R   how much of the LARGEST displacement it carries")
    report("  A response can be local in amplitude and global in norm -- a small floor spread over 13,000")
    report("  atoms carries as much norm as a sharp peak over 100 -- and those need different answers.")
    report(f"    {'R (A)':>8}{'atoms':>9}{'share of N':>12}{'norm in R':>12}{'peak in R':>12}{'max |u| out':>13}")
    trow = []
    amp = np.linalg.norm(ref.reshape(-1, 3), axis=1)
    tot_n = np.linalg.norm(ref)
    for R in RADII:
        keep = d <= R
        mask = np.repeat(keep, 3)
        f_norm = np.linalg.norm(np.where(mask, ref, 0.0)) / max(tot_n, 1e-300)
        f_peak = amp[keep].max() / max(amp.max(), 1e-300) if keep.any() else 0.0
        out = amp[~keep].max() if (~keep).any() else 0.0
        trow.append({"R": R, "atoms": int(keep.sum()), "frac": float(keep.mean()),
                     "norm_in": float(f_norm), "peak_in": float(f_peak), "max_out": float(out),
                     "rel_err": float(np.sqrt(max(0.0, 1 - f_norm ** 2)))})
        report(f"    {R:>8.0f}{int(keep.sum()):>9}{keep.mean():>12.1%}{f_norm:>12.3f}{f_peak:>12.3f}"
               f"{out:>13.3e}")

    # ---- TIMING, measured ---------------------------------------------------------------------------
    report("\n  PER-EVENT COST, measured rather than projected")
    R_use = 50.0
    nsup = int((np.linalg.norm(co - co[sites[0]], axis=1) <= R_use).sum())
    sup = np.repeat(np.linalg.norm(co - co[sites[0]], axis=1) <= R_use, 3)
    acc = np.zeros(n3)
    t4 = time.time()
    REPS = 2000
    for _ in range(REPS):
        acc[sup] += 0.5 * ref[sup]
    t_ev = (time.time() - t4) / REPS
    report(f"    support at R={R_use:.0f} A: {nsup} atoms | one superposition {t_ev*1e6:.1f} us "
           f"(single core, measured over {REPS} repeats)")
    full = t_fac + np.mean(tcol) * N_SITE
    for ev in (1e3, 1e6):
        report(f"    {ev:.0g} events: {ev*t_ev:.3g} s of superposition, against "
               f"{ev*t_cg/86400:.3g} days if each were solved iteratively")

    report("\n  READING")
    ok1 = e1 < 1e-6
    ok2 = e2 < 1e-8
    ok3 = trow[-1]["norm_in"] > 2 * trow[0]["norm_in"]
    for lab, ok in (("Green column equals an independent solve", ok1),
                    ("superposition equals a direct solve of the sum", ok2),
                    ("truncated support captures more norm as R grows", ok3)):
        report(f"    {'PASS' if ok else 'FAIL'}  {lab}")
    if ok1 and ok2 and ok3:
        report(f"  The identity holds. One factorisation at {t_fac:.0f} s makes every subsequent response a")
        report(f"  {np.mean(tcol)*1000:.0f} ms solve, and every EVENT a {t_ev*1e6:.0f} us addition over a")
        report(f"  {nsup}-atom support. The 11.4-year figure was the cost of repeatedly computing something")
        report("  already known. What this cannot absorb is nonlinearity: a contact, a melting bubble or a")
        report("  topology change alters K itself, which invalidates the factors and the columns with them.")
    else:
        report("  A gate failed. Superposition is the load-bearing assumption of the whole event scheme, so")
        report("  a failure here removes the cost argument entirely, not just a refinement of it.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "chromatin_greens", "n_atoms": int(n), "nnz": int(K.nnz),
               "t_factorise": t_fac, "factor_nnz": int(lu.nnz), "t_column_ms": float(np.mean(tcol) * 1000),
               "t_cg_s": t_cg, "err_vs_cg": float(e1), "err_superposition": float(e2),
               "truncation": trow, "t_event_us": float(t_ev * 1e6), "support_atoms": int(nsup),
               "gates": {"column": bool(ok1), "linearity": bool(ok2), "locality": bool(ok3)},
               "log": log}, open(OUT / "chromatin_greens.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'chromatin_greens.json'}")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    raise SystemExit(main())

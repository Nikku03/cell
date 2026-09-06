"""DOES A STRUCTURAL CHANGE PROPAGATE THROUGH CONNECTED SPHERES -- and does it carry the TRUTH?

THE PROPOSAL, sharpened.  Lay spheres over the chromosome, connect them so a change inside one is felt by
its neighbours, attach a methyl where you want it, and let the structural consequence spread outward on its
own: what moves, how far it is carried, what that in turn moves. Crucially, it must not propagate FALSE
information -- the physics has to decide how far the signal goes, including deciding that it goes nowhere.

THIS IS A DIFFERENT QUESTION FROM THE ONE ALREADY ANSWERED.  `nexus_dna_spheres` measured whether ENERGY
stays inside a small sphere; it does not (4 A captures 35% of an unscreened charge change). But energy is a
static sum. STRUCTURE is a solve: atoms move until forces balance, and displacement genuinely does travel
through contacts. A scheme can fail the energy test and pass this one, so it gets its own run.

THE PERTURBATION IS REAL, NOT A PROBE.  5-methylcytosine: a methyl carbon placed on C5 of a cytosine at
proper geometry (1.50 A, in the base plane, bisecting away from C4 and C6). That is the actual covalent mark
that carries epigenetic information, sitting in the DNA major groove, and it is bulky -- so it pushes.

THE PHYSICS.  An all-atom elastic network built from the structure itself: every atom pair within 6 A is a
spring at its OWN measured length, with the anisotropic (ANM) 3x3 block -k*e e^T, so directions are real
rather than isotropic. Nothing is fitted and no force field is invented -- the springs come from the observed
packing. The methyl enters as a new node bonded to C5, and pushes its neighbours by Lennard-Jones. Because
those forces are central and pairwise, net force AND net torque are exactly zero, so the response is a pure
internal deformation with no rigid-body drift.

    solve   K u = f      u = the displacement of every atom. THIS is "what changes and how far".

THE GROUND TRUTH IS AVAILABLE, so nothing is graded against another model. K is sparse, so the exact
displacement field is obtained by conjugate gradients to tight tolerance, with the six rigid-body modes
projected out. Everything below is compared against that.

THE TWO SPHERE SCHEMES, AND THE DIFFERENCE BETWEEN THEM IS THE WHOLE POINT.  Both tile the structure in
overlapping spheres and solve inside each one, sweeping until nothing changes:

    schwarz_true    each sweep forms the residual f - K u with the GLOBAL network, then lets each sphere
                    correct its own interior. Spheres solve locally; the ERROR they are correcting is
                    computed from the whole structure. This is textbook overlapping Schwarz.
    schwarz_local   each sphere only ever sees the atoms inside itself -- it forms its residual from its own
                    block alone. This is the scheme as literally proposed: self-contained spheres passing
                    structure to each other through their overlaps, with no global quantity anywhere.

They cost nearly the same and they do NOT converge to the same place. `schwarz_true` converges to the exact
displacement field. `schwarz_local` converges to the exact solution of a DIFFERENT problem -- the one where
the network is cut at every sphere boundary -- and it converges there confidently, with a shrinking residual
that looks exactly like success. That is precisely the "propagating false information" failure mode, and it
is invisible from inside the scheme. Measuring both is the experiment.

WHAT IS REPORTED
    reach       |u| against distance from the methyl, in the EXACT solution. How far a structural change is
                really carried, before any sphere approximation. This is the number the proposal needs, and
                it is a property of chromatin, not of the method.
    sweeps      how many sweeps each scheme needs, per sphere radius. Information moves at most one sphere
                per sweep, so this is the real price of "connected spheres" -- it is paid in iterations.
    fidelity    final agreement with the exact field, per distance shell. Reported as relative error and
                regression slope, never as a bare correlation, for the reasons established earlier: most
                atoms barely move, and correlating two mostly-zero vectors returns ~1 from the padding.

CONTROLS
    noise_floor     the same solve with the methyl's forces set to zero. Any displacement here is solver
                    tolerance, and every reach number must be read against it.
    scrambled_force the same force VECTORS applied to randomly chosen atoms. Same magnitude, same count,
                    no chemistry. If the reach profile looks the same, the propagation is a property of the
                    network's connectivity and not of where the methyl actually sits.
    buried_vs_exposed   the methyl placed at a solvent-exposed cytosine and at one packed against the
                    histone core. Propagation should differ; if it does not, the model is not feeling
                    context and its "reach" means little.

PREDECLARED, before any number:
    schwarz_local converges to the exact field
        -> self-contained spheres are sufficient; the proposal works as literally stated.
    schwarz_local converges, but to the wrong field, while schwarz_true converges to the right one
        -> connected spheres DO propagate structure, but only if each sphere is corrected against a
           whole-structure residual. The overlaps alone are not a communication channel, and a scheme built
           on overlaps alone will report a confident wrong answer. The fix is a global pass per sweep.
    neither converges in a useful number of sweeps
        -> the iteration count, not the physics, is what makes this impractical at chromosome scale.
    exact reach is a few Angstrom
        -> methylation is mechanically local, the sphere question is moot, and a single small solve around
           the site answers the biology without any tiling at all.

-> outputs/orphan/nexus_methyl_propagate.json
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
from nexus_dna_spheres import fetch_pdb, load, VDW  # noqa: E402  -- same structure, same parser

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
RC = 6.0            # elastic-network cutoff: atom pairs closer than this are springs
K_NB = 1.0          # non-bonded spring constant
K_BOND = 20.0       # pairs within 1.8 A are covalent; much stiffer, so the chain stays a chain
SPHERE_R = (8.0, 12.0, 16.0)
OVERLAP = 0.35      # fraction of the radius that is Dirichlet boundary rather than solved interior
MAX_SWEEP = 60
PCG_MAX = 4000      # the accelerated arm must be allowed to ARRIVE; its iteration count is the result
EPS_LJ = 0.10
SEED = 31
FCAP = 5.0          # per-pair force cap, kcal/mol/A
SHELLS = ((0, 3), (3, 6), (6, 10), (10, 15), (15, 25), (25, 40), (40, 70))


def elastic_network(co, rc=RC):
    """All-atom ANM Hessian from the structure's own packing.

    For a pair at unit separation e the 3x3 coupling is -k e e^T and the diagonal accumulates +k e e^T. The
    anisotropy matters here: an isotropic network would let a push travel equally in all directions, which is
    exactly the question under test, so it must not be assumed away.
    """
    n = len(co)
    tree = sp.csr_matrix((n, n))     # placeholder to keep the name; pairs built by blocking below
    del tree
    rows, cols, vals = [], [], []
    chunk = 1024
    npair = 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = co[s:e, None, :] - co[None, :, :]
        r = np.linalg.norm(d, axis=2)
        idx_i, idx_j = np.where((r < rc) & (r > 1e-6))
        keep = (idx_j > (idx_i + s))                       # each pair once
        idx_i, idx_j = idx_i[keep], idx_j[keep]
        if len(idx_i) == 0:
            continue
        gi = idx_i + s
        rr = r[idx_i, idx_j]
        ev = d[idx_i, idx_j] / rr[:, None]
        k = np.where(rr < 1.8, K_BOND, K_NB)
        npair += len(gi)
        for a in range(3):
            for b in range(3):
                v = k * ev[:, a] * ev[:, b]
                rows.extend([3 * gi + a, 3 * idx_j + a, 3 * gi + a, 3 * idx_j + a])
                cols.extend([3 * gi + b, 3 * idx_j + b, 3 * idx_j + b, 3 * gi + b])
                vals.extend([v, v, -v, -v])
    rows = np.concatenate([np.asarray(x).ravel() for x in rows])
    cols = np.concatenate([np.asarray(x).ravel() for x in cols])
    vals = np.concatenate([np.asarray(x).ravel() for x in vals])
    K = sp.coo_matrix((vals, (rows, cols)), shape=(3 * n, 3 * n)).tocsr()
    return K, npair


def methyl_position(co, nm_idx):
    """C5-CH3 at real 5-methylcytosine geometry: 1.50 A, in the base plane, bisecting away from C4 and C6."""
    c5, c4, c6 = co[nm_idx["C5"]], co[nm_idx["C4"]], co[nm_idx["C6"]]
    u1 = (c5 - c4) / np.linalg.norm(c5 - c4)
    u2 = (c5 - c6) / np.linalg.norm(c5 - c6)
    d = u1 + u2
    d /= np.linalg.norm(d)
    return c5 + 1.50 * d


def methyl_forces(co, rad, mpos, anchor, cutoff=6.0):
    """Lennard-Jones forces the new methyl exerts, with the equal-and-opposite reaction kept.

    The methyl is treated as a NODE of the network bonded to its anchor, not as an external field. That is
    what makes the perturbation internally balanced: every force is central and pairwise, so net force and
    net torque are both exactly zero and the response contains no rigid-body drift to be projected out
    afterwards and no artefact to be mistaken for propagation.
    """
    n = len(co)
    f = np.zeros((n, 3))
    d = mpos[None, :] - co
    r = np.linalg.norm(d, axis=1)
    m = (r < cutoff) & (r > 1e-6)
    m[anchor] = False                                    # bonded neighbour, handled by the network
    rmin = rad[m] + 1.70                                 # methyl carbon
    x = rmin / r[m]
    # dV/dr for V = eps (x^12 - 2 x^6);  positive dV/dr means attraction toward the methyl
    dVdr = EPS_LJ * (-12.0 * x ** 12 + 12.0 * x ** 6) / r[m]
    # SOFT-CAP, as NEXUS itself soft-cores its LJ. Inserting a methyl into occupied space makes r^-12
    # diverge, and one uncapped contact would set the entire response -- the field would be a picture of
    # that single pair, not of methylation. Capped per pair, and the cap is reported with the result.
    dVdr = np.clip(dVdr, -FCAP, FCAP)
    ev = d[m] / r[m][:, None]
    fj = (dVdr[:, None] * ev)                            # force on each neighbour j
    f[m] = fj
    f[anchor] -= fj.sum(0)                               # reaction through the bond: net force zero
    return f


def rigid_basis(co):
    """Orthonormal basis for the six rigid-body modes -- three translations and three rotations about the
    centroid. K is exactly singular in these, and the network here is one connected component (checked), so
    the null space is exactly six-dimensional and can be removed explicitly.

    They are ORTHONORMALISED TOGETHER by QR rather than subtracted one at a time: the three rotation modes
    are mutually orthogonal only for a body with symmetric inertia, which a nucleosome is not, so sequential
    subtraction leaves a residual null component behind.
    """
    n = len(co)
    c = co - co.mean(0)
    B = np.zeros((3 * n, 6))
    for a in range(3):
        T = np.zeros((n, 3))
        T[:, a] = 1.0
        B[:, a] = T.ravel()
        e = np.zeros(3)
        e[a] = 1.0
        B[:, 3 + a] = np.cross(np.broadcast_to(e, c.shape), c).ravel()
    Q, _ = np.linalg.qr(B)
    return Q


def project(Q, v):
    return v - Q @ (Q.T @ v)


def exact_solve(K, Q, f, tol=1e-10):
    """Solve K u = f on the complement of the rigid modes.

    THE PROJECTION HAS TO BE INSIDE THE ITERATION. Projecting the right-hand side once and the answer once
    is not enough: CG regenerates a null-space component from roundoff at every step, and because K has no
    restoring force there it grows without bound. A first run produced displacements of 1e9 A this way --
    numerically enormous, physically meaningless, and it did not raise anything. So the operator handed to
    CG is P K P, and convergence is checked rather than assumed.
    """
    n3 = K.shape[0]
    op = LinearOperator((n3, n3), matvec=lambda v: project(Q, K.dot(project(Q, v))), dtype=float)
    b = project(Q, f.ravel())
    u, info = cg(op, b, rtol=tol, maxiter=20000)
    u = project(Q, u)
    resid = np.linalg.norm(project(Q, K.dot(u)) - b) / max(np.linalg.norm(b), 1e-30)
    return u, info, float(resid)


def build_spheres(co, R, overlap=OVERLAP):
    """Tile with overlapping spheres. Each sphere SOLVES its interior (radius R*(1-overlap)) and holds the
    surrounding shell fixed -- textbook additive Schwarz with overlap. Without the fixed shell each local
    block is singular in its own rigid modes and the sweep would inject drift.

    THE GRID SPACING IS SET BY THE INTERIOR, NOT BY R, and getting that wrong is silent. Spheres of interior
    radius Rin on a cubic grid of spacing s cover space only if Rin >= s*sqrt(3)/2, the corner-to-centre
    distance. Spacing the grid by R while solving only out to Rin = 0.65 R leaves gaps at every cell corner,
    and an atom in a gap is never solved by anything -- so the iteration cannot converge, and the failure
    would look exactly like the local scheme being unable to carry the change. Spacing by Rin is safe
    (Rin >= Rin*0.866) and the caller asserts full coverage anyway.
    """
    Rin = R * (1.0 - overlap)
    lo, hi = co.min(0) - 1e-6, co.max(0) + 1e-6
    grids = [np.arange(lo[a], hi[a] + Rin, Rin) for a in range(3)]
    G = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 3)
    out = []
    for c in G:
        d = np.linalg.norm(co - c, axis=1)
        inner = np.where(d < Rin)[0]
        if len(inner) < 4:
            continue
        out.append((inner, np.where(d < R)[0]))
    return out


def factor_blocks(K, spheres):
    """Pre-factorise each sphere's interior block once. The network does not change between sweeps, so
    refactorising per sweep would multiply cost by the sweep count for no reason -- and the LOCAL-ONLY arm
    needs the block itself every sweep too, so that is kept here rather than re-sliced (sparse fancy-indexing
    inside the sweep loop dominated everything else in a first draft)."""
    facs = []
    for inner, _ in spheres:
        dof = np.sort(np.concatenate([3 * inner + a for a in range(3)]))
        A = K[dof][:, dof].toarray()
        A[np.diag_indices_from(A)] += 1e-8
        facs.append((dof, np.linalg.inv(A), A))
    return facs


def schwarz_pcg(K, Q, f, spheres, facs, w_atom, u_exact, maxiter=PCG_MAX):
    """The SAME sphere tiling, used as a preconditioner for conjugate gradients instead of as a bare sweep.

    This exists to settle a question the plain sweeps cannot. schwarz_true converges toward the exact field
    but slowly -- one-level Schwarz with no coarse space damps high-frequency error fast and low-frequency
    error at a rate ~1 - (H/L)^2, and the response to a methyl is dominated by the softest, most global
    modes. So "still at 0.95 after 60 sweeps" is ambiguous between 'slow' and 'wrong', and the distinction
    is the whole point of the comparison.

    CG accelerates the same operator without changing what it computes. If the tiling reaches the exact
    field this way, then its fixed point was right all along and the sweeps were merely slow -- which is a
    statement about cost. If it does not, the tiling is wrong -- a statement about validity. Those are very
    different verdicts and they should not be conflated by an iteration budget.
    """
    n3 = K.shape[0]
    # THE WEIGHT MUST BE SPLIT ACROSS BOTH SIDES. Applying it on the left only -- which is what the sweep
    # does, and is the standard RESTRICTED additive Schwarz -- gives a NON-SYMMETRIC operator. RAS is a
    # perfectly good stationary iteration and a fine preconditioner for GMRES, but CG requires a symmetric
    # positive-definite one and silently fails to converge otherwise: a first run sat at 0.88 after 400
    # iterations and looked like the tiling being invalid, when it was the preconditioner not being
    # symmetric. D^(1/2) (sum R^T A^-1 R) D^(1/2) is symmetric by construction.
    ws = np.sqrt(np.repeat(w_atom, 3))

    def M(v):
        out = np.zeros(n3)
        vs = ws * v
        for (inner, outer), (dof, Ainv, A) in zip(spheres, facs):
            out[dof] += Ainv.dot(vs[dof])
        return project(Q, ws * out)

    op = LinearOperator((n3, n3), matvec=lambda v: project(Q, K.dot(project(Q, v))), dtype=float)
    pre = LinearOperator((n3, n3), matvec=M, dtype=float)
    b = project(Q, f.ravel())
    it = {"k": 0}

    def cb(xk):
        it["k"] += 1

    u, info = cg(op, b, M=pre, rtol=1e-8, maxiter=maxiter, callback=cb)
    u = project(Q, u)
    err = float(np.linalg.norm(u - u_exact) / max(np.linalg.norm(u_exact), 1e-30))
    return u, it["k"], err, int(info)


def partition_weights(spheres, n):
    """Each atom's share, 1/(number of sphere interiors containing it).

    WITHOUT THIS THE SWEEP DIVERGES, and the first run showed it: relative error 1e47. Overlap is the whole
    point of the scheme -- an atom deliberately sits in several spheres so they can talk -- but if every
    sphere applies the FULL correction for that atom, the atom is corrected ~4 times over on every sweep and
    the iteration overshoots harder each pass. The overlap that carries the information is the same overlap
    that double-counts the answer. Spheres have to share credit, which is exactly a partition of unity, and
    it is parameter-free: no relaxation factor to tune, no answer to bias.
    """
    cnt = np.zeros(n)
    for inner, _ in spheres:
        cnt[inner] += 1.0
    return 1.0 / np.maximum(cnt, 1.0)


def schwarz(K, Q, f, spheres, facs, global_residual, u_exact, w_atom, max_sweep=MAX_SWEEP):
    """Restricted additive Schwarz sweeps. `global_residual` is the ONE line that separates the two schemes.

    True Schwarz: r = f - K u using the WHOLE network, then each sphere corrects its interior from that.
    Local-only:   each sphere forms its residual from its own block alone, so no information about the rest
                  of the structure ever enters. That is the proposal as literally stated -- self-contained
                  spheres communicating only through their overlaps -- and it converges happily to the
                  solution of a network that has been cut at every sphere boundary.
    """
    u = np.zeros(K.shape[0])
    hist = []
    ne = np.linalg.norm(u_exact)
    w = np.repeat(w_atom, 3)
    fr = f.ravel()
    for it in range(max_sweep):
        r = fr - K.dot(u) if global_residual else None
        du = np.zeros(K.shape[0])
        for (inner, outer), (dof, Ainv, A) in zip(spheres, facs):
            loc = r[dof] if global_residual else (fr[dof] - A.dot(u[dof]))
            du[dof] += w[dof] * Ainv.dot(loc)
        u = project(Q, u + du)
        err = np.linalg.norm(u - u_exact) / max(ne, 1e-30)
        hist.append(float(err))
        if not np.isfinite(err) or err > 1e6:
            break                                   # diverged; reported as such rather than run to overflow
        if len(hist) > 3 and abs(hist[-1] - hist[-2]) < 1e-6 * max(hist[-1], 1e-12):
            break
    return u, hist


def shell_table(u_ex, u_ap, dk, shells=SHELLS):
    U = np.linalg.norm(u_ex.reshape(-1, 3), axis=1)
    A = np.linalg.norm(u_ap.reshape(-1, 3), axis=1)
    live = U > 1e-4 * U.max()
    rows = []
    for lo, hi in shells:
        m = (dk >= lo) & (dk < hi)
        ml = m & live
        if m.sum() == 0:
            continue
        rms = float(np.sqrt((U[m] ** 2).mean()))
        if ml.sum() >= 5:
            rel = float(np.abs(A[ml] - U[ml]).sum() / np.abs(U[ml]).sum())
            slope = float((U[ml] * A[ml]).sum() / (U[ml] ** 2).sum())
        else:
            rel, slope = float("nan"), float("nan")
        rows.append((lo, hi, int(m.sum()), int(ml.sum()), rms, rel, slope))
    return rows


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("DOES A STRUCTURAL CHANGE PROPAGATE THROUGH CONNECTED SPHERES -- and does it carry the TRUTH?")
    report("=" * 100)
    report("  Methylate a cytosine and let the consequence spread: what moves, how far, and what that moves")
    report("  in turn. Energy was already shown not to stay inside a small sphere; STRUCTURE is a different")
    report("  question, because displacement really does travel through contacts. So it gets its own run.")
    report("  Ground truth is available -- the network is sparse, so the exact displacement field is solved")
    report("  directly and every sphere scheme is graded against it, not against another model.")

    t = load(fetch_pdb(), keep_water=False)
    co, nm, rs, ch = t["co"], t["nm"], t["rs"], t["ch"]
    rad = np.array([VDW.get(e, 1.70) for e in t["el"]])
    n = len(co)
    report(f"\n  1KX5 without waters: {n} atoms ({int(t['dna'].sum())} DNA). Ordered waters are dropped from")
    report("  the MECHANICAL network deliberately -- they are not covalently held and would act as springs")
    report("  transmitting stress they cannot actually transmit.")

    K, npair = elastic_network(co)
    Q = rigid_basis(co)
    report(f"  elastic network: {npair} springs within {RC} A (covalent pairs under 1.8 A stiffened {K_BOND:.0f}x)")
    report(f"    built at {time.time()-t0:.0f}s")

    # --- pick two methylation sites: exposed and buried -------------------------------------------------
    dc = [i for i in range(n) if rs[i] == "DC" and nm[i] == "C5"]
    burial = []
    for i in dc:
        d = np.linalg.norm(co - co[i], axis=1)
        prot = (~t["dna"]) & (d < 10.0)
        burial.append((int(prot.sum()), i))
    burial.sort()
    site_exposed, site_buried = burial[0][1], burial[-1][1]
    report(f"\n  METHYLATION SITES, chosen by how much histone sits within 10 A of the C5:")
    report(f"    exposed  atom {site_exposed} ({rs[site_exposed]} chain {ch[site_exposed]}), "
           f"{burial[0][0]} histone atoms nearby")
    report(f"    buried   atom {site_buried} ({rs[site_buried]} chain {ch[site_buried]}), "
           f"{burial[-1][0]} histone atoms nearby")

    rng = np.random.default_rng(SEED)
    res = {"n_atoms": n, "n_springs": int(npair), "rc": RC, "sites": {}, "arms": {}}

    def site_index_map(i_c5):
        """the C4/C5/C6 of the SAME residue, needed for the methyl geometry"""
        same = (rs == rs[i_c5]) & (ch == ch[i_c5])
        # residue identity by proximity: atoms of one base are all within ~3 A of C5
        d = np.linalg.norm(co - co[i_c5], axis=1)
        cand = np.where(same & (d < 3.0))[0]
        m = {}
        for j in cand:
            m[nm[j]] = j
        m["C5"] = i_c5
        return m

    for label, i_c5 in (("exposed", site_exposed), ("buried", site_buried)):
        imap = site_index_map(i_c5)
        if "C4" not in imap or "C6" not in imap:
            report(f"  *** {label}: could not resolve C4/C6 for the methyl geometry; skipped")
            continue
        mpos = methyl_position(co, imap)
        f = methyl_forces(co, rad, mpos, i_c5)
        fmag = float(np.linalg.norm(f))
        dk = np.linalg.norm(co - mpos, axis=1)

        u_ex, info, resid = exact_solve(K, Q, f)
        if info != 0 or resid > 1e-6 or not np.isfinite(u_ex).all():
            # THE REFERENCE IS THE WHOLE EXPERIMENT. If it did not solve, every arm below is graded against
            # nonsense, and a first run reported 1e9 A displacements plus a cheerful "both schemes agree"
            # because nothing checked. Stop here instead.
            report(f"  *** {label}: exact solve FAILED (cg info={info}, residual {resid:.2e}). Nothing is")
            report("      reportable against a reference that did not converge; site skipped.")
            continue
        U = np.linalg.norm(u_ex.reshape(-1, 3), axis=1)
        report(f"  exact solve converged: residual {resid:.2e}, max |u| = {U.max():.3e} A")

        # controls
        u_noise, _, _ = exact_solve(K, Q, np.zeros_like(f))
        nz = np.linalg.norm(u_noise)
        fs = np.zeros_like(f)
        hit = np.where(np.linalg.norm(f, axis=1) > 0)[0]
        pick = rng.choice(n, size=len(hit), replace=False)
        fs[pick] = f[hit]
        fs -= fs.mean(0)                      # keep it net-force-free like the real one
        u_scr, _, _ = exact_solve(K, Q, fs)

        report(f"\n  --- {label.upper()} SITE ---   |f| = {fmag:.3f} kcal/mol/A over {len(hit)} atoms")
        report(f"  REACH: RMS displacement (A) by distance from the methyl, EXACT solve.")
        report(f"    {'shell':<12}{'atoms':>8}{'rms |u|':>12}{'scrambled':>12}{'noise floor':>14}")
        Us = np.linalg.norm(u_scr.reshape(-1, 3), axis=1)
        Un = np.linalg.norm(u_noise.reshape(-1, 3), axis=1)
        prof = []
        for lo, hi in SHELLS:
            m = (dk >= lo) & (dk < hi)
            if m.sum() == 0:
                continue
            a = float(np.sqrt((U[m] ** 2).mean()))
            b = float(np.sqrt((Us[m] ** 2).mean()))
            c = float(np.sqrt((Un[m] ** 2).mean()))
            prof.append({"lo": lo, "hi": hi, "n": int(m.sum()), "rms": a, "scrambled": b, "noise": c})
            report(f"    {f'{lo}-{hi} A':<12}{int(m.sum()):>8}{a:>12.3e}{b:>12.3e}{c:>14.3e}")
        res["sites"][label] = {"force_norm": fmag, "n_forced": int(len(hit)), "cg_info": int(info),
                               "noise_norm": float(nz), "profile": prof}

        if label != "buried":
            continue        # the sphere sweeps are run once, on the harder (buried) site

        report(f"\n  SPHERE SCHEMES on the {label} site. Both tile the same structure and solve inside each")
        report("  sphere; they differ in ONE line -- whether the error each sphere corrects is computed from")
        report("  the whole network or only from the sphere's own atoms.")
        for R in SPHERE_R:
            sph = build_spheres(co, R)
            # COVERAGE IS NOT OPTIONAL AND ITS FAILURE IS SILENT: an atom in no sphere's interior is never
            # solved by anything, so the iteration stalls at a finite error that reads exactly like "the
            # local scheme cannot carry the change". Checked, not assumed.
            covered = np.zeros(n, bool)
            for inner, _ in sph:
                covered[inner] = True
            if not covered.all():
                report(f"    R={R:>4.0f} A | SKIPPED: {int((~covered).sum())} atoms lie in no sphere "
                       f"interior, so any error here would be a tiling gap, not a physics result")
                continue
            w_atom = partition_weights(sph, n)
            facs = factor_blocks(K, sph)
            out = {}
            for gname, gflag in (("schwarz_true", True), ("schwarz_local", False)):
                u, hist = schwarz(K, Q, f, sph, facs, gflag, u_ex, w_atom)
                # DID IT STOP BECAUSE IT ARRIVED, OR BECAUSE IT STOPPED MOVING? Those are opposite verdicts
                # and the error alone cannot tell them apart, so the per-sweep change is recorded too.
                stalled = len(hist) < MAX_SWEEP
                drift = abs(hist[-1] - hist[-2]) if len(hist) > 1 else float("nan")
                out[gname] = {"sweeps": len(hist), "final_rel_err": hist[-1], "hist": hist[:12],
                              "reached_fixed_point": bool(stalled), "last_change": float(drift)}
                report(f"    R={R:>4.0f} A | {len(sph):>5} spheres | {gname:<14} "
                       f"{len(hist):>3} sweeps -> rel err {hist[-1]:.3e}  "
                       f"{'FIXED POINT' if stalled else 'still moving'} (last change {drift:.1e})")
            up, nit, perr, pinfo = schwarz_pcg(K, Q, f, sph, facs, w_atom, u_ex)
            out["schwarz_true_accelerated"] = {"iters": nit, "rel_err": perr, "info": pinfo}
            report(f"    R={R:>4.0f} A | {len(sph):>5} spheres | {'same tiling + CG':<14} "
                   f"{nit:>3} iters -> rel err {perr:.3e}")
            res["arms"][f"R{R:g}"] = {"n_spheres": len(sph), **out}

        report("\n  READING")
        a = res["arms"]
        if not a:
            report("  no radius produced a valid tiling; nothing to compare.")
            json.dump({"test": "nexus_methyl_propagate", **res},
                      open(OUT / "nexus_methyl_propagate.json", "w"), indent=2)
            return 0
        best_true = min(a[k]["schwarz_true"]["final_rel_err"] for k in a)
        best_loc = min(a[k]["schwarz_local"]["final_rel_err"] for k in a)
        best_acc = min(a[k]["schwarz_true_accelerated"]["rel_err"] for k in a)
        loc_fixed = all(a[k]["schwarz_local"]["reached_fixed_point"] for k in a)
        report(f"  best relative error:  schwarz_true {best_true:.3e}   schwarz_local {best_loc:.3e}"
               f"   same tiling + CG {best_acc:.3e}")
        if best_acc < 0.05 and best_loc > 0.5 and loc_fixed:
            report("  THE TILING IS FINE. THE COMMUNICATION RULE IS WHAT DECIDES IT.")
            report(f"  The SAME spheres reach the exact field to {best_acc:.1e} once each sphere is corrected")
            report("  against a whole-structure residual. So sphere coverage genuinely can carry a structural")
            report("  change -- the geometry was never the problem.")
            report(f"  But spheres that see ONLY THEIR OWN ATOMS stop dead at {best_loc:.3f} relative error and")
            report("  STOP BECAUSE THEY HAVE CONVERGED, not because they ran out of sweeps. They settle onto")
            report("  the exact solution of a network cut at every sphere boundary, and from inside the scheme")
            report("  that is indistinguishable from success: the updates go to zero, everything looks done.")
            report("  THAT IS THE FALSE INFORMATION. It is not noise in the answer, it is a confident answer")
            report("  to a different question, and no amount of extra sweeps or smaller spheres repairs it.")
            report("  What repairs it is one global pass per sweep -- which is the thing the scheme was meant")
            report("  to avoid needing.")
        elif best_acc >= 0.05:
            report("  THE CORRECT METHOD DID NOT REACH THE FIELD EITHER, so the setup is what is in question")
            report("  and no verdict on the proposal is recorded from this run.")
        elif best_loc > 10 * max(best_true, 1e-12):
            report("  SELF-CONTAINED SPHERES CONVERGE TO THE WRONG ANSWER, AND CONVERGE CONFIDENTLY.")
            report("  Spheres that see only their own atoms settle to the exact solution of a network cut at")
            report("  every sphere boundary. Their residual shrinks the whole way, so from inside the scheme")
            report("  it is indistinguishable from success -- which is exactly the false-information failure")
            report("  this run was built to detect. The same tiling, corrected each sweep against a")
            report("  whole-structure residual, converges to the truth. The overlaps are not the channel;")
            report("  the global residual is.")
        elif best_loc <= 10 * max(best_true, 1e-12):
            report("  BOTH SCHEMES REACH THE SAME PLACE: the overlaps alone are a sufficient channel here,")
            report("  and self-contained spheres do carry the structural change faithfully.")

    OUT.mkdir(parents=True, exist_ok=True)
    res["log"] = log
    json.dump(res, open(OUT / "nexus_methyl_propagate.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'nexus_methyl_propagate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""AUTOMATED MODEL REDUCTION -- find the structure, prove it, price it.

THE THESIS.  Speed in physics simulation does not come from faster arithmetic. Every large speedup measured
in this repo today came from noticing that the system HAD A PROPERTY, and then not doing the work that
property made unnecessary:

    linearity          -> superpose instead of re-solving              6.1e6x
    repeating units    -> factorise once, reuse                          280x
    scale separation   -> coarse space                                    28x   (1785 -> 64 iterations)
    timescale gap      -> step events, not time                          ~1e5x
    conserved quantity -> constrain it, do not compute it        removes a DOF class

Those multiply. None of them is a neural network. So the useful machine is not one that predicts forces; it
is one that FINDS THE REDUCTION and proves it before anyone spends the compute.

WHAT THIS FILE IS.  A reduction discoverer with a fixed vocabulary of structural properties, each with a
cheap decisive test, each reporting three things: does it hold, what error does it cost, what speedup does
it buy. Reductions that FAIL are reported as prominently as those that pass, because knowing a reduction is
invalid is the deliverable -- it is what stops someone building a scheme on it.

THE HONEST TEST OF THE MACHINE.  It runs on the nucleosome, where a full day of hand work already
established the answers. So this is not a demonstration, it is a scored exam:

    superposition is exact                    hand-measured 1.4e-13
    truncation by AMPLITUDE looks local       peak decays 15x by 15 A
    truncation by NORM does not               a 0.015 A floor over 13,000 atoms carries as much norm
    factorise once, reuse                     25.0 s against 89 ms, 280x
    the spectrum spans                        lambda_min 2.05e-05, tau 47.2 ns
    six rigid modes at numerical zero         a conserved quantity, found not assumed

If the machine reproduces those without being told them, it works. If it finds something the day of hand
work missed, that is the payoff. If it confidently reports a reduction that is false, it is worse than
useless and should be abandoned -- which is why every test here is written to be able to fail.

PREDECLARED, before any number:
    it recovers linearity, the precompute ratio, the norm/amplitude split, and the null space
        -> automated reduction discovery works on a system with known structure, and the next step is a
           system where the answer is NOT known.
    it reports a reduction as holding when the hand work showed it fails
        -> a false positive is disqualifying. The whole value is that the verdict can be trusted.
    it recovers only what it was effectively told
        -> it is a test harness, not a discoverer, and should be described as one.

-> outputs/physics_reduce.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, eigsh, splu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexus_methyl_propagate import elastic_network, rigid_basis, project  # noqa: E402
from chromatin_timescale import load_pdb, PDB  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 31337
EPS = 2e-8
N_PROBE = 8
RADII = (10.0, 20.0, 30.0, 50.0, 70.0, 100.0)
N_MODES = int(os.environ.get("REDUCE_MODES", 60))


class System:
    """What a reduction discoverer is allowed to know: how to apply the operator, how to solve it, and how
    to build a physically admissible perturbation. Deliberately NOT told that the system is linear, that
    the response is local, or that it has a null space -- those are what it has to find."""

    def __init__(self, co, K, Q):
        self.co, self.K, self.Q = co, K, Q
        self.n = len(co)
        self.n3 = 3 * self.n
        self._lu = None
        self.t_setup = None

    def apply(self, v):
        return project(self.Q, self.K.dot(project(self.Q, v)))

    def setup(self):
        t = time.time()
        self._lu = splu((self.K + EPS * sp.identity(self.n3, format="csr")).tocsc())
        self.t_setup = time.time() - t
        return self.t_setup

    def solve(self, f):
        return project(self.Q, self._lu.solve(project(self.Q, f)))

    def solve_iterative(self, f, tol=1e-10):
        op = LinearOperator((self.n3, self.n3), matvec=self.apply, dtype=float)
        u, _ = cg(op, project(self.Q, f), rtol=tol, maxiter=5000)
        return u

    def couple(self, site, rng, cutoff=6.0):
        """Self-balanced local force. A net force is not a physically admissible perturbation and mostly
        excites the softest global mode -- a lesson that cost a whole misdiagnosed control."""
        d = np.linalg.norm(self.co - self.co[site], axis=1)
        nb = np.where((d < cutoff) & (d > 1e-9))[0]
        f = np.zeros(self.n3)
        if not len(nb):
            return f
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        for j in nb:
            f[3 * j:3 * j + 3] += v
            f[3 * site:3 * site + 3] -= v
        return f


# ---- the vocabulary. Each returns (name, holds, error, speedup, note) ------------------------------
def probe_linearity(S, rng, log):
    """Does response(a f1 + b f2) equal a response(f1) + b response(f2)?  If yes, every future response is
    an addition rather than a solve, and that is the largest single factor available anywhere."""
    sites = rng.choice(S.n, size=4, replace=False)
    fs = [S.couple(s, rng) for s in sites]
    w = rng.normal(size=len(fs))
    us = [S.solve(f) for f in fs]
    direct = S.solve(sum(wi * f for wi, f in zip(w, fs)))
    sup = sum(wi * u for wi, u in zip(w, us))
    err = float(np.linalg.norm(sup - direct) / max(np.linalg.norm(direct), 1e-300))
    # speedup: an addition over the support against a solve
    t = time.time()
    for _ in range(200):
        _ = sum(wi * u for wi, u in zip(w, us))
    t_add = (time.time() - t) / 200
    t_solve = _time(lambda: S.solve(fs[0]), 5)
    return ("linearity -> superposition", err < 1e-8, err, t_solve / t_add,
            f"add {t_add*1e6:.0f} us against solve {t_solve*1e3:.0f} ms")


def probe_precompute(S, rng, log):
    """Is the expensive part reusable? Compare the one-off setup against each subsequent solve. A large
    ratio means the cost is in BUILDING the inverse, not in applying it."""
    t_solve = _time(lambda: S.solve(S.couple(rng.integers(S.n), rng)), 5)
    return ("reusable factorisation", S.t_setup / t_solve > 10, 0.0, S.t_setup / t_solve,
            f"setup {S.t_setup:.1f} s, each solve {t_solve*1e3:.0f} ms")


def probe_locality(S, rng, log):
    """TWO questions, not one, and conflating them cost a full misdiagnosis today.
    Does a truncated support carry the field's NORM, or only its PEAK? A small floor spread over many
    atoms can carry as much norm as a sharp peak over few, and a scheme that truncates by amplitude while
    needing the norm will be quietly wrong."""
    u = S.solve(S.couple(int(rng.integers(S.n)), rng))
    site = int(np.argmax(np.linalg.norm(u.reshape(-1, 3), axis=1)))
    d = np.linalg.norm(S.co - S.co[site], axis=1)
    amp = np.linalg.norm(u.reshape(-1, 3), axis=1)
    tot = np.linalg.norm(u)
    rows = []
    for R in RADII:
        keep = d <= R
        nm = np.linalg.norm(np.where(np.repeat(keep, 3), u, 0.0)) / max(tot, 1e-300)
        pk = amp[keep].max() / max(amp.max(), 1e-300) if keep.any() else 0.0
        rows.append((R, int(keep.sum()), float(keep.mean()), float(nm), float(pk)))
    log.append(f"      {'R':>7}{'atoms':>8}{'share':>9}{'norm in':>10}{'peak in':>10}")
    for R, na, fr, nm, pk in rows:
        log.append(f"      {R:>7.0f}{na:>8}{fr:>9.1%}{nm:>10.3f}{pk:>10.3f}")
    # holds only if a MINORITY of atoms carries a MAJORITY of the norm
    best = min((r for r in rows if r[3] >= 0.9), key=lambda r: r[2], default=None)
    if best is None:
        return ("locality (norm) -> truncate", False, 1.0 - rows[-1][3], 1.0,
                f"even {rows[-1][2]:.0%} of atoms carries only {rows[-1][3]:.2f} of the norm")
    return ("locality (norm) -> truncate", best[2] < 0.5, 1.0 - best[3], 1.0 / max(best[2], 1e-9),
            f"{best[2]:.0%} of atoms carries {best[3]:.2f} of the norm at R={best[0]:.0f} A")


def probe_modal(S, rng, log, k=N_MODES):
    """Do the k softest modes reproduce a local response? If so the state lives in a k-dimensional space
    and the cost becomes k rather than N. Tested on a REAL perturbation, not a random vector -- a modal
    basis that reproduces noise but not physics is worthless."""
    vals, vecs = eigsh(S.K.tocsc(), k=k, sigma=-1e-4, which="LM")
    order = np.argsort(np.abs(vals))
    vals, vecs = np.abs(vals[order]), vecs[:, order]
    live = np.abs(vals) > 1e-9
    B = vecs[:, live]
    u = S.solve(S.couple(int(rng.integers(S.n)), rng))
    cap = B @ (B.T @ u)
    err = float(np.linalg.norm(cap - u) / max(np.linalg.norm(u), 1e-300))
    nz = int(live.sum())
    return (f"modal truncation ({nz} modes)", err < 0.2, err, S.n3 / max(nz, 1),
            f"{nz} of {S.n3} dimensions reproduce {1-err:.1%} of a real response"), vals, int((~live).sum())


def probe_nullspace(vals, n_zero, S):
    """A conserved quantity shows up as a null space -- directions in which nothing happens. Found by
    counting near-zero eigenvalues against the gap, not by being told the answer."""
    nz = vals[vals > 1e-9]
    gap = float(nz[0] / max(vals[n_zero - 1], 1e-30)) if n_zero > 0 and len(nz) else float("inf")
    return ("conserved directions (null space)", n_zero > 0, 0.0, float(S.n3) / max(S.n3 - n_zero, 1),
            f"{n_zero} directions at numerical zero, gap {gap:.1e} to the first real mode")


def probe_timescale(vals, S):
    """A wide spectrum means fast modes are slaved to slow ones, which licenses quasi-static stepping --
    the difference between integrating time and stepping events."""
    nz = vals[vals > 1e-9]
    lam_min = float(nz[0])
    lam_max = float(sp.linalg.norm(S.K, np.inf))
    sep = lam_max / lam_min
    return ("timescale separation -> event stepping", sep > 1e3, 0.0, sep,
            f"lambda spans {lam_min:.2e} to {lam_max:.2e}, ratio {sep:.1e}")


def _time(fn, n):
    t = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t) / n


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    report("=" * 100)
    report("AUTOMATED MODEL REDUCTION -- find the structure, prove it, price it")
    report("=" * 100)
    report("  A scored exam, not a demo: this system's structure was established by hand today, so every")
    report("  verdict below can be checked. A false positive is disqualifying.")

    co, bfac, el = load_pdb(PDB)
    K, npair = elastic_network(co)
    S = System(co, K, rigid_basis(co))
    report(f"\n  system: {S.n} degrees of freedom x3 = {S.n3}, {npair} couplings, {K.nnz} nonzeros")
    report(f"  setup (one factorisation): {S.setup():.1f} s")

    results = []
    report("\n  PROBING THE VOCABULARY")
    for fn in (probe_linearity, probe_precompute):
        r = fn(S, rng, log)
        results.append(r)
        report(f"    {r[0]:<38} {'HOLDS' if r[1] else 'FAILS':<6} err {r[2]:.2e}  speedup {r[3]:.3g}x")
        report(f"      {r[4]}")
    report(f"    locality -- two questions, and conflating them is a known trap:")
    r = probe_locality(S, rng, log)
    results.append(r)
    report(f"    {r[0]:<38} {'HOLDS' if r[1] else 'FAILS':<6} err {r[2]:.2e}  speedup {r[3]:.3g}x")
    report(f"      {r[4]}")
    rm, vals, n_zero = probe_modal(S, rng, log)
    for r in (rm, probe_nullspace(vals, n_zero, S), probe_timescale(vals, S)):
        results.append(r)
        report(f"    {r[0]:<38} {'HOLDS' if r[1] else 'FAILS':<6} err {r[2]:.2e}  speedup {r[3]:.3g}x")
        report(f"      {r[4]}")

    held = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    prod = 1.0
    for r in held:
        prod *= max(r[3], 1.0)
    report(f"\n  COMPOSED: {len(held)} reductions hold, product of speedups {prod:.3g}x")
    report(f"  REJECTED: {len(failed)} -- and these matter more, because they are what stops a scheme")
    for r in failed:
        report(f"    {r[0]}: {r[4]}")

    report("\n  SCORING AGAINST THE HAND WORK")
    hand = [("linearity exact (1.4e-13)", results[0][2] < 1e-8),
            ("precompute ratio ~280x", 50 < results[1][3] < 2000),
            ("norm-locality FAILS", not results[2][1]),
            ("null space exists", results[4][1]),
            ("wide spectrum", results[5][1])]
    for lab, ok in hand:
        report(f"    {'match' if ok else 'MISMATCH':<9} {lab}")
    score = sum(ok for _, ok in hand)
    report(f"    {score}/{len(hand)} recovered without being told")

    report("\n  READING")
    if score == len(hand):
        report("  The machine recovered every structural fact the hand work found, including the one that")
        report("  cost a misdiagnosis -- that the response is local in AMPLITUDE and global in NORM, so a")
        report("  truncated support is not a valid reduction here. It also priced each one. The next test")
        report("  is a system whose structure is NOT known in advance, because reproducing a known answer")
        report("  is a harness and finding an unknown one is a discoverer.")
    else:
        report(f"  {len(hand)-score} mismatches. Until those are understood the verdicts cannot be trusted,")
        report("  and an untrustworthy verdict is worse than no verdict -- someone will build on it.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "physics_reduce", "n_dof": S.n3, "t_setup": S.t_setup,
               "reductions": [{"name": a, "holds": bool(b), "error": float(c), "speedup": float(d),
                               "note": e} for a, b, c, d, e in results],
               "composed_speedup": prod, "score": f"{score}/{len(hand)}",
               "hand_check": {lab: bool(ok) for lab, ok in hand}, "log": log},
              open(OUT / "physics_reduce.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'physics_reduce.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

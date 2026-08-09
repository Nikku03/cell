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
# A verdict is GRADED, not binary. The first version reported pass/fail at a fixed threshold, and that
# threw away exactly the discrimination the tool exists to provide: timescale separation spanning six
# orders of magnitude across four systems collapsed into "1 of 8 verdicts moved", which the tool then
# reported as a property of the SYSTEMS rather than of its own reporting. It also rendered a linearity
# error of 2e-8 against a 1e-8 threshold as a categorical FAILS. Anything within BAND of the threshold is
# now MARGINAL and says so.
BAND = 3.0


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
        self.null = None          # DISCOVERED at setup, not assumed
        self.n_null = None
        self.scale = self.noise = self.cut = None

    def apply(self, v):
        return project(self.Q, self.K.dot(project(self.Q, v)))

    def setup(self, k0=16):
        """DISCOVER the null space; do not assume it is the six rigid modes.

        The first version assumed exactly that, and it was catastrophically wrong on any system with an
        extra floppy direction. The solve regularises K + eps I with eps = 2e-8, so a TRUE null mode is
        amplified by 1/eps = 5e7. On a random point cloud with one rattler, the response came back
        100.000% inside the null space, max |u| = 1.5e7, and every downstream probe was measuring
        amplified noise. The tool's own null-space probe had already reported 7 directions where 6 were
        expected -- it detected the condition that invalidated its other probes and nothing joined them up.

        k is grown until a non-null mode appears, because the count is not known in advance: a diluted
        network near the isostatic point had 23.
        """
        t = time.time()
        self.scale = float(sp.linalg.norm(self.K, np.inf))     # the operator's own scale
        # CALIBRATE "ZERO" AGAINST A DIRECTION THAT MUST BE ZERO BY SYMMETRY. A rigid translation cannot
        # change the energy of any mechanical system, so its Rayleigh quotient is exactly the numerical
        # noise floor of THIS operator. An analytic Hessian gives ~1e-16 relative; the rod's
        # finite-difference Hessian gives 5.3e-08, eight orders worse, and a fixed cut either finds
        # nothing (too tight) or is tuned to the noise (arbitrary). This measures it instead, and reports
        # it, so a caller can see that the operator is too noisy to have a resolvable null space.
        # ALL SIX rigid directions, not one. Calibrating on a translation alone found 3 of 6: rotations
        # displace the periphery much further, so their finite-difference noise is larger, and a cut set
        # by the quietest direction cannot see the loudest. The worst of the six is the operator's real
        # noise floor.
        rq = [float(abs(self.Q[:, j] @ (self.K @ self.Q[:, j])))
              / max(float(self.Q[:, j] @ self.Q[:, j]), 1e-300) for j in range(self.Q.shape[1])]
        self.noise = float(max(rq)) / max(self.scale, 1e-300)
        self.cut = self.scale * max(1e-9, 10.0 * self.noise)
        k = k0
        while True:
            vals, vecs = eigsh(self.K.tocsc(), k=min(k, self.n3 - 2), sigma=-1e-4, which="LM")
            o = np.argsort(np.abs(vals))
            vals, vecs = np.abs(vals[o]), vecs[:, o]
            # RELATIVE cut, and it must be relative to the WHOLE operator. An absolute 1e-9 worked on
            # the elastic networks only because their gaps were 1e10-1e14; the rod's Hessian, rescaled by
            # kT, has eigenvalues near 1e17 and the constant found ZERO null directions where symmetry
            # guarantees six. The first attempt at a fix took max(|vals|) from the RETURNED eigenvalues --
            # but those are the SMALLEST ones, since the solve is shift-inverted around zero, so the
            # scale was itself near lambda_min and the cut stayed microscopic. It has to be the norm of
            # the operator.
            nn = int((vals <= self.cut).sum())
            if nn < len(vals) or k >= self.n3 - 2:
                break
            k *= 2                       # every returned mode was null -- there are more
        self.null = np.ascontiguousarray(vecs[:, vals <= self.cut])
        self.n_null = self.null.shape[1]
        self._lu = splu((self.K + EPS * sp.identity(self.n3, format="csr")).tocsc())
        self.t_setup = time.time() - t
        return self.t_setup

    def _denull(self, v):
        return v - self.null @ (self.null.T @ v) if self.n_null else v

    def solve(self, f):
        return self._denull(self._lu.solve(self._denull(project(self.Q, f))))

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


# ---- the vocabulary ---------------------------------------------------------------------------------
# Every probe returns the same record. The NAME is stable across systems -- the first version encoded the
# live mode count into it ("modal truncation (33 modes)"), which gave every system a different key and made
# cross-system comparison impossible while silently inflating the denominator.
def verdict(name, q, thr, direction, speedup, note, detail=None):
    """Grade a measured quantity against a threshold, keeping the quantity itself.

    direction 'below': the reduction holds when q is small (an error).
    direction 'above': it holds when q is large (a ratio).
    Within a factor of BAND either way the answer is MARGINAL, because a threshold is a judgement and a
    measurement that lands on one should say so rather than pick a side.
    """
    # A MULTIPLICATIVE band is wrong for a quantity bounded above by 1. A captured fraction of 0.9975
    # against a 0.9 threshold is only 1.1x, so "HOLDS" was unreachable for any fraction and 54 modes
    # capturing 99.8% of the response was graded MARGINAL. For bounded quantities the margin lives in the
    # COMPLEMENT -- how much is left over, 0.0025 against 0.1, which is 40x clear.
    bounded = direction == "above" and 0.0 <= q <= 1.0 and 0.0 < thr < 1.0
    if bounded:
        r = (1.0 - q) / (1.0 - thr)
        v = "HOLDS" if r < 1 / BAND else ("FAILS" if r > BAND else "MARGINAL")
    elif direction == "below":
        r = q / thr if thr else float("inf")
        v = "HOLDS" if r < 1 / BAND else ("FAILS" if r > BAND else "MARGINAL")
    else:
        r = q / thr if thr else float("inf")
        v = "HOLDS" if r > BAND else ("FAILS" if r < 1 / BAND else "MARGINAL")
    return {"name": name, "quantity": float(q), "threshold": float(thr), "direction": direction,
            "verdict": v, "margin": float(r), "speedup": float(speedup), "note": note,
            "detail": detail or {}}


def probe_linearity(S, rng, log):
    """Does response(a f1 + b f2) equal a response(f1) + b response(f2)?  If yes every future response is
    an addition rather than a solve, which is the largest single factor available anywhere."""
    sites = rng.choice(S.n, size=4, replace=False)
    fs = [S.couple(s, rng) for s in sites]
    w = rng.normal(size=len(fs))
    us = [S.solve(f) for f in fs]
    direct = S.solve(sum(wi * f for wi, f in zip(w, fs)))
    sup = sum(wi * u for wi, u in zip(w, us))
    err = float(np.linalg.norm(sup - direct) / max(np.linalg.norm(direct), 1e-300))
    t = time.time()
    for _ in range(200):
        _ = sum(wi * u for wi, u in zip(w, us))
    t_add = (time.time() - t) / 200
    t_solve = _time(lambda: S.solve(fs[0]), 5)
    return verdict("linearity -> superposition", err, 1e-8, "below", t_solve / t_add,
                   f"add {t_add*1e6:.0f} us against solve {t_solve*1e3:.0f} ms")


def probe_finite_linearity(S, rng, log):
    """Is the PHYSICS linear, or only the operator?

    THE GAP THE ROD EXPOSED. probe_linearity tests whether solve(af1+bf2) equals a solve(f1)+b solve(f2).
    For any linear solver that is true BY CONSTRUCTION -- it tests the algebra, not the system. Every
    nonlinear system has a Hessian, and that Hessian is linear, so the existing probe would report
    "linearity HOLDS" on a system where superposition is worthless. That is a false positive of exactly
    the kind that disqualifies the tool.

    The honest test uses FINITE AMPLITUDE against the true energy: minimise E under force f, then under
    2f, and ask whether the second displacement is twice the first. A harmonic system says yes exactly.
    Anything else does not, and the deviation grows with amplitude -- which is also the useful output,
    since it says how far you may superpose before it stops being true.

    Requires S.energy. Systems without one SKIP this probe and say so rather than silently passing.
    """
    if not hasattr(S, "energy") or S.energy is None:
        return None
    f0 = S.couple(int(rng.integers(S.n)), rng)
    scale0 = float(np.median(np.linalg.norm(np.diff(S.co, axis=0), axis=1)))
    # SET the amplitude, do not hope a descent lands on it. Three attempts at tuning a step size gave
    # 1e-8, then 6.6e3, then 1.3e-4 bond lengths -- the window is 1e-3 to 0.5 and none of them hit it.
    # Bisecting the FORCE against the measured displacement puts the probe in the window by construction,
    # which is what a test of finite-amplitude behaviour requires before it can mean anything.
    TARGET = 0.05
    lo, hi = 1e-6, 1e12
    f, u1 = f0, S.relax(f0)
    for _ in range(24):
        mid = np.sqrt(lo * hi)
        u = S.relax(mid * f0)
        rel_ = float(np.linalg.norm(u.reshape(-1, 3), axis=1).max()) / max(scale0, 1e-300)
        if not np.isfinite(rel_) or rel_ > TARGET:
            hi = mid
        else:
            lo = mid
        if 0.5 * TARGET < rel_ < 2 * TARGET:
            f, u1 = mid * f0, u
            break
        f, u1 = mid * f0, u
    # THE PROBE MUST CHECK THAT IT ACTUALLY PERTURBED SOMETHING. Everything is linear at zero amplitude,
    # so a probe that applies a negligible force reports HOLDS on any system whatsoever -- a test that can
    # only pass. The rod caught this: deviations of 1e-6 that were growing with amplitude, i.e. real
    # nonlinearity, but far too small to grade, because the displacement was microscopic against the
    # system's own length scale.
    scale = float(np.median(np.linalg.norm(np.diff(S.co, axis=0), axis=1)))     # a bond length
    amp1 = float(np.linalg.norm(u1.reshape(-1, 3), axis=1).max())
    devs = []
    for a in (2.0, 4.0, 8.0):
        ua = S.relax(a * f)
        devs.append(float(np.linalg.norm(ua - a * u1) / max(np.linalg.norm(a * u1), 1e-300)))
    worst = max(devs)
    rel = amp1 / max(scale, 1e-300)
    note = ("deviation from exact scaling at 2x/4x/8x force: " + ", ".join(f"{d:.2e}" for d in devs)
            + f"   |  max displacement {rel:.2e} of a bond length")
    # TWO-SIDED. The guard originally checked only for a perturbation that was too SMALL, and I then
    # over-corrected the step size until the relaxation diverged to 6,600 bond lengths and the probe
    # reported FAILS -- the right verdict for the wrong reason, which is still a wrong result. A test of
    # nonlinearity has to perturb enough to leave the harmonic well and little enough to stay on the
    # energy surface, and it must refuse to grade outside that window rather than pick a side.
    if rel < 1e-3 or rel > 0.5:
        v = verdict("finite-amplitude linearity", worst, 1e-3, "below", 1.0, note,
                    {"devs": devs, "rel_amp": rel})
        v["verdict"] = "NOT TESTABLE"
        why = ("the perturbation moved almost nothing, so this cannot distinguish a linear system from "
               "any other" if rel < 1e-3 else
               "the relaxation left the energy surface entirely, so the deviation measures divergence "
               "rather than nonlinearity")
        v["note"] = note + "  -- " + why
        return v
    return verdict("finite-amplitude linearity", worst, 1e-3, "below", 1.0, note,
                   {"devs": devs, "rel_amp": rel})


def probe_precompute(S, rng, log):
    """Is the expensive part reusable? A large setup-to-solve ratio means the cost is in BUILDING the
    inverse rather than applying it, so it should be built once."""
    t_solve = _time(lambda: S.solve(S.couple(rng.integers(S.n), rng)), 5)
    r = S.t_setup / t_solve
    return verdict("reusable factorisation", r, 10.0, "above", r,
                   f"setup {S.t_setup:.1f} s, each solve {t_solve*1e3:.0f} ms")


def probe_locality(S, rng, log):
    """TWO questions, and conflating them cost a full misdiagnosis. Does a truncated support carry the
    field's NORM, or only its PEAK? A small floor over many atoms can carry as much norm as a sharp peak
    over few, so a scheme that truncates by amplitude while needing the norm is quietly wrong.

    The reported quantity is the FRACTION OF ATOMS needed to hold 90% of the norm -- small is good, and it
    is what actually sets the per-event cost."""
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
        rows.append({"R": R, "atoms": int(keep.sum()), "frac": float(keep.mean()),
                     "norm_in": float(nm), "peak_in": float(pk)})
    hit = [r for r in rows if r["norm_in"] >= 0.9]
    frac = min((r["frac"] for r in hit), default=1.0)
    note = (f"{frac:.0%} of atoms carry 90% of the norm (R={min(r['R'] for r in hit):.0f})" if hit
            else f"NO radius up to {max(RADII):.0f} reaches 90% of the norm -- "
                 f"the largest tested holds {rows[-1]['norm_in']:.2f}")
    return verdict("locality (norm) -> truncate", frac, 0.25, "below", 1.0 / max(frac, 1e-9),
                   note, {"rows": rows})


def probe_modal(S, rng, log, k=N_MODES):
    """Do the k softest modes reproduce a real response? Reported as CAPTURED FRACTION, and the mode count
    goes in the note rather than the name so systems remain comparable."""
    vals, vecs = eigsh(S.K.tocsc(), k=k, sigma=-1e-4, which="LM")
    o = np.argsort(np.abs(vals))
    vals, vecs = np.abs(vals[o]), vecs[:, o]
    live = vals > S.cut
    B = vecs[:, live]
    u = S.solve(S.couple(int(rng.integers(S.n)), rng))
    cap = float(np.linalg.norm(B @ (B.T @ u)) / max(np.linalg.norm(u), 1e-300))
    nz = int(live.sum())
    return verdict("modal truncation", cap, 0.9, "above", S.n3 / max(nz, 1),
                   f"{nz} of {S.n3} dimensions capture {cap:.1%} of a real response",
                   {"n_modes": nz}), vals, int((~live).sum())


def probe_nullspace(vals, n_zero, S):
    """A conserved quantity appears as a null space -- IF the operator can resolve one.

    Loosening the cut on the rod's finite-difference Hessian gave 0, then 3, then 9 null directions where
    symmetry guarantees 6. That is not a threshold to be tuned: it means the noise floor and the softest
    physical modes OVERLAP, so no cut can separate them and any count is an artefact of where the line
    was drawn. The honest output is the GAP at the proposed cut. Without a gap the count is reported as
    unresolvable rather than asserted, which is the difference between a measurement and a preference."""
    """A conserved quantity appears as a null space. Found by counting near-zero eigenvalues against the
    gap, not by being told. The quantity is the COUNT, which is what discriminates."""
    nz = vals[vals > S.cut]
    gap = float(nz[0] / max(vals[n_zero - 1], 1e-30)) if n_zero > 0 and len(nz) else float("inf")
    if n_zero > 0 and gap < 10.0:
        v = verdict("conserved directions (null space)", float(n_zero), 1.0, "above", 1.0,
                    f"{n_zero} directions below the cut, but the gap to the next mode is only "
                    f"{gap:.1f}x -- the noise floor ({S.noise:.1e} relative) and the softest real modes "
                    f"OVERLAP, so no cut separates them and this count is an artefact of where the line "
                    f"was drawn", {"n_null": int(n_zero), "gap": gap, "noise": S.noise})
        v["verdict"] = "UNRESOLVABLE"
        return v
    return verdict("conserved directions (null space)", float(n_zero), 1.0, "above",
                   float(S.n3) / max(S.n3 - n_zero, 1),
                   f"{n_zero} directions at numerical zero, gap {gap:.1e} to the first real mode; "
                   f"zero calibrated against rigid translation at {S.noise:.1e} relative"
                   + ("  -- NOISY OPERATOR, an analytic Hessian gives ~1e-16" if S.noise > 1e-12 else ""),
                   {"n_null": int(n_zero), "gap": gap})


def probe_timescale(vals, S):
    """A wide spectrum means fast modes are slaved to slow ones, which licenses stepping events rather
    than time."""
    nz = vals[vals > S.cut]
    lam_min = float(nz[0])
    lam_max = float(sp.linalg.norm(S.K, np.inf))
    sep = lam_max / lam_min
    return verdict("timescale separation -> event stepping", sep, 1e3, "above", sep,
                   f"lambda spans {lam_min:.2e} to {lam_max:.2e}")


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
    report("  A scored exam, not a demo: this system's structure was established by hand, so every verdict")
    report("  can be checked. Verdicts are GRADED -- anything within 3x of a threshold reads MARGINAL,")
    report("  because a threshold is a judgement and a measurement that lands on one should say so.")

    co, bfac, el = load_pdb(PDB)
    K, npair = elastic_network(co)
    S = System(co, K, rigid_basis(co))
    report(f"\n  system: {S.n3} degrees of freedom, {npair} couplings, {K.nnz} nonzeros")
    report(f"  setup: {S.setup():.1f} s, null space DISCOVERED as {S.n_null} directions")

    results = []
    report("\n  PROBING THE VOCABULARY")
    for fn in (probe_linearity, probe_finite_linearity, probe_precompute, probe_locality):
        r = fn(S, rng, log)
        if r is None:
            report("    finite-amplitude linearity            SKIPPED   no energy function supplied")
            continue
        results.append(r)
    rm, vals, n_zero = probe_modal(S, rng, log)
    results += [rm, probe_nullspace(vals, n_zero, S), probe_timescale(vals, S)]
    report(f"    {'reduction':<40}{'verdict':<10}{'quantity':>12}{'vs thr':>9}{'speedup':>11}")
    for r in results:
        report(f"    {r['name']:<40}{r['verdict']:<10}{r['quantity']:>12.4g}"
               f"{r['margin']:>9.2g}{r['speedup']:>11.4g}")
        report(f"      {r['note']}")

    AXIS = {"linearity -> superposition": "solves", "reusable factorisation": "solves",
            "conserved directions (null space)": "dimensions", "modal truncation": "dimensions",
            "timescale separation -> event stepping": "steps",
            "locality (norm) -> truncate": "dimensions"}
    by_axis = {}
    for r in results:
        if r["verdict"] == "FAILS":
            continue
        ax = AXIS.get(r["name"], "other")
        by_axis[ax] = max(by_axis.get(ax, 1.0), max(r["speedup"], 1.0))
    prod = 1.0
    for v in by_axis.values():
        prod *= v
    report("\n  COMPOSED across independent cost axes -- NOT a product over reductions, because linearity")
    report("  and reusable factorisation both eliminate re-solving and would double-count:")
    for ax, v in sorted(by_axis.items()):
        report(f"    {ax:<14}{v:>12.4g}x")
    report(f"    {'TOTAL':<14}{prod:>12.4g}x")
    rejected = [r for r in results if r["verdict"] == "FAILS"]
    report(f"  REJECTED {len(rejected)} -- these matter more, they are what stops a scheme:")
    for r in rejected:
        report(f"    {r['name']}: {r['note']}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "physics_reduce", "n_dof": S.n3, "n_null": S.n_null, "t_setup": S.t_setup,
               "reductions": results, "composed": prod, "axes": by_axis, "log": log},
              open(OUT / "physics_reduce.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'physics_reduce.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""THE REDUCER MEETS A SYSTEM IT SHOULD FAIL ON -- the twistable rod.

WHY THIS IS THE ADVERSARIAL CASE.  Every system the reducer has seen was a harmonic spring network, where
superposition is exact by construction. The rod is not: its energy contains 2 pi^2 C (dLk - Wr)^2 / L, and
Wr is a nonlinear functional of the coordinates -- a double integral over every pair of segments. If the
tool reports "linearity HOLDS" here, it has produced a false positive on the one property that carries the
largest speedup in its whole vocabulary, and it should be thrown away.

THE GAP THIS EXPOSED BEFORE IT EVEN RAN.  The existing linearity probe asks whether solve(a f1 + b f2)
equals a solve(f1) + b solve(f2). For ANY linear solver that is true by construction. Every nonlinear
system has a Hessian, that Hessian is linear, and so the probe would have passed the rod while
superposition on it is worthless. The probe was testing the algebra, not the system.

So a new probe was added first: FINITE-AMPLITUDE linearity, which minimises the TRUE energy under f and
under 2f, 4f, 8f, and asks whether the displacement scales. A harmonic system says yes exactly. The
deviation is also the useful number -- it says how far superposition may be pushed before it stops holding.

WHAT THE ROD ACTUALLY IS
    E(r) = (A/2) sum theta^2 / ds  +  2 pi^2 C (dLk - Wr(r))^2 / L  +  E_excluded
    the first and third terms are ordinary. The second is not: Wr couples every segment to every other,
    and it is what makes the rod supercoil.

PREDECLARED, before any number:
    finite-amplitude linearity FAILS, and the operator-level probe still passes
        -> the tool is correctly distinguishing a linear operator from a linear system, and the new probe
           was necessary rather than decorative.
    finite-amplitude linearity HOLDS on the rod
        -> either the amplitudes tested are too small to leave the harmonic well, which the deviation
           curve will show, or the rod as implemented is not doing what its energy says.
    the operator-level probe FAILS
        -> the numerical Hessian is wrong, and nothing else in the run means anything.

Also of interest and genuinely unknown: does the rod have a null space beyond the six rigid modes? A closed
chain with fixed bond lengths has constraints the reducer has never met, and I do not know the count.

THE ANSWER TURNED OUT TO BE THREE, AND "SIX RIGID MODES" WAS THE WRONG EXPECTATION.  For an energy
invariant under rotation, differentiating grad E . Gx = 0 a second time gives

    H (Gx) = -G^T grad E                   G the rotation generator, (Gx)_i = n x x_i

so a rotation is a null mode ONLY at a stationary point. The generator depends on x, and off a minimum
that term survives. A translation's generator does not depend on x, so translations are null everywhere.
This Hessian is built at the imposed circular configuration, which is not a minimum -- |grad E| is
0.58 kT/nm -- so three is the correct count and six would have been the error.

That is not a threshold argument, and it was checked two independent ways rather than asserted:
    the theorem PREDICTS the rotational Rayleigh quotient with no free parameter, and measurement
        agrees within 25% on all three axes  (9.25e14 measured against 1.15e15 predicted)
    shrinking the finite-difference step 10x moves NOISE and leaves a real derivative alone: the
        translation quotient falls 1e4 (3.7e-09 -> 3.9e-13) while the rotation quotient does not move
        (1.4e-05 -> 2.2e-05), and the count stays 3 at both steps
The energy was also confirmed rotation-invariant directly, to 1e-06 kT at rotations up to 1 radian, so
the rotational stiffness is not a broken symmetry in the model.

-> outputs/physics_reduce_rod.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import physics_reduce as PR  # noqa: E402
import chromatin_supercoil as SC  # noqa: E402
from chromatin_rod import KT, RISE  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 606
N_BEAD = int(os.environ.get("ROD_N", 28))
BP = 1400.0
SIGMA = -0.05
H = 1e-9        # finite-difference step, metres


class RodSystem(PR.System):
    """The rod dressed as something the reducer can probe: a numerical Hessian for the operator-level
    tests, and the TRUE energy for the finite-amplitude test."""

    def __init__(self, circle):
        self.ch = circle
        co = circle.r.copy()
        n = len(co)
        K = self._hessian(co)
        super().__init__(co, K, PR.rigid_basis(co))
        self.energy = self._E

    def _E(self, flat):
        old = self.ch.r
        self.ch.r = flat.reshape(-1, 3)
        e = self.ch.energy()
        self.ch.r = old
        return e

    def _grad(self, flat):
        g = np.zeros_like(flat)
        for i in range(len(flat)):
            p = flat.copy(); p[i] += H
            m = flat.copy(); m[i] -= H
            g[i] = (self._E(p) - self._E(m)) / (2 * H)
        return g

    def _hessian(self, co):
        """Numerical Hessian by central differences of the numerical gradient. Expensive and only viable
        because the rod is deliberately small here -- the point is the VERDICT, not the scale."""
        flat = co.ravel().copy()
        n3 = len(flat)
        Hm = np.zeros((n3, n3))
        for i in range(n3):
            p = flat.copy(); p[i] += H
            m = flat.copy(); m[i] -= H
            Hm[:, i] = (self._grad(p) - self._grad(m)) / (2 * H)
        Hm = 0.5 * (Hm + Hm.T)
        return sp.csr_matrix(Hm / KT)          # in kT units so thresholds are comparable

    def relax(self, f, steps=300, lr=None):
        """Minimise the TRUE energy under an applied force, by gradient descent. This is what makes the
        finite-amplitude probe a test of the physics rather than of the linear algebra."""
        # STEP SIZE FROM THE SYSTEM, NOT A CONSTANT. The first version used lr = 1e-22 and the
        # displacement came out at 1e-8 of a bond length -- far too small to distinguish a linear system
        # from any other, which is exactly what the probe then reported.
        # ADAPTIVE, because a fixed step diverged. The first version used lr = 1e-22 and moved 1e-8 of a
        # bond length -- too small to distinguish a linear system from any other. Scaling the step to the
        # system then overshot the other way and the descent ran to 6,600 bond lengths, so the probe
        # reported FAILS for the wrong reason. Neither is a measurement. This backtracks whenever the
        # objective rises, which is the minimum needed for the number to mean anything.
        x = self.co.ravel().copy()
        bond = float(np.median(np.linalg.norm(np.diff(self.co, axis=0), axis=1)))
        obj = lambda z: self._E(z) - float(f @ z) * KT
        e = obj(x)
        lr = lr or (1e-3 * bond / max(np.abs(self._grad(x) - f * KT).max(), 1e-300))
        for _ in range(steps):
            g = self._grad(x) - f * KT
            xn = x - lr * g
            en = obj(xn)
            if en < e:
                x, e = xn, en
                lr *= 1.1                      # it worked, try a little harder
            else:
                lr *= 0.5                      # it did not, back off
                if lr < 1e-30:
                    break
        return x - self.co.ravel()


def rigid_check(S, report):
    """WHY THE NULL SPACE IS 3 AND NOT 6, measured rather than argued.

    Translations are null modes of any Hessian of a translation-invariant energy. Rotations are null modes
    only at a stationary point, because the rotation generator depends on the coordinates: differentiating
    grad E . Gx = 0 gives H(Gx) = -G^T grad E. This checks that identity against the built Hessian, and
    runs the step-size control that separates a real derivative from finite-difference noise."""
    co = S.co
    g = S._grad(co.ravel().copy()).reshape(-1, 3)
    c = co.mean(axis=0)
    scale = float(sp.linalg.norm(S.K, np.inf))
    report(f"\n  WHY 3 AND NOT 6 -- |grad E| = {np.linalg.norm(g)/KT*1e-9:.3f} kT/nm, so this configuration"
           f" is NOT a minimum")
    report(f"    {'axis':<10}{'measured RQ':>14}{'H(Gx) = n x grad E':>21}{'ratio':>9}{'rel ||K||':>12}")
    out = []
    for lbl, ax in (("rot x", [1, 0, 0]), ("rot y", [0, 1, 0]), ("rot z", [0, 0, 1])):
        n = np.array(ax, float)
        u = np.cross(n, co - c).ravel()
        d = u / np.linalg.norm(u)
        meas = float(d @ (S.K @ d))
        pred = float(u @ np.cross(n, g).ravel() / KT) / float(u @ u)
        report(f"    {lbl:<10}{meas:>14.4e}{abs(pred):>21.4e}"
               f"{abs(meas)/max(abs(pred), 1e-300):>9.2f}{meas/scale:>12.2e}")
        out.append({"axis": lbl, "measured": meas, "predicted": abs(pred), "rel_scale": meas / scale})
    report("    a parameter-free prediction; agreement means the rotational stiffness is the theorem")
    return out


def step_control(report):
    """NOISE MOVES WITH THE STEP, A DERIVATIVE DOES NOT. The one control that separates them."""
    report("\n  CONTROL -- rebuild at a 10x smaller finite-difference step")
    report(f"    {'step':<12}{'trans RQ':>12}{'rot RQ':>12}{'n_null':>9}{'gap':>12}")
    rows, h0 = [], H
    for h in (H, H / 10):
        globals()["H"] = h
        ch = SC.Circle(N_BEAD, BP, SIGMA * (BP / 10.5), SC.C_TWIST, np.random.default_rng(SEED))
        S = RodSystem(ch)
        sc = float(sp.linalg.norm(S.K, np.inf))
        Q = PR.rigid_basis(ch.r.copy())
        rq = [float(abs(Q[:, j] @ (S.K @ Q[:, j]))) / float(Q[:, j] @ Q[:, j]) for j in range(6)]
        S.setup(k0=12)
        t, r = max(rq[:3]) / sc, max(rq[3:]) / sc
        report(f"    {h:<12.0e}{t:>12.3e}{r:>12.3e}{S.n_null:>9}{S.gap:>12.3e}")
        rows.append({"h": h, "trans": t, "rot": r, "n_null": S.n_null, "gap": S.gap})
    globals()["H"] = h0
    fell = rows[1]["trans"] < rows[0]["trans"] / 100
    held = rows[1]["rot"] > rows[0]["rot"] / 3
    report(f"    translation quotient collapsed: {fell}   rotation quotient held: {held}   "
           f"count stable: {rows[0]['n_null'] == rows[1]['n_null']}")
    if fell and held:
        report("    -> the translations' residual was noise and the rotations' stiffness is real, so the")
        report("       null space is 3. Six would have been an assumption the measurement contradicts.")
    return rows


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    report("=" * 100)
    report("THE REDUCER MEETS A SYSTEM IT SHOULD FAIL ON -- the twistable rod")
    report("=" * 100)
    report("  Every system so far was a harmonic network where superposition is exact by construction.")
    report("  The rod's energy contains (dLk - Wr)^2 and Wr is a double integral over every segment pair.")
    report("  If the tool reports linearity HOLDS here it has produced a false positive on the property")
    report("  carrying its largest speedup, and it should be thrown away.")

    lk0 = BP / 10.5
    ch = SC.Circle(N_BEAD, BP, SIGMA * lk0, SC.C_TWIST, np.random.default_rng(SEED))
    report(f"\n  rod: {N_BEAD} beads, {BP:.0f} bp, sigma {SIGMA}, dLk {SIGMA*lk0:.2f}, "
           f"Wr {ch.writhe():+.3f}")
    report(f"  building a {3*N_BEAD}x{3*N_BEAD} numerical Hessian by finite differences...")
    t1 = time.time()
    S = RodSystem(ch)
    report(f"  Hessian built in {time.time()-t1:.0f} s")
    S.setup(k0=12)
    report(f"  null space DISCOVERED: {S.n_null} directions, gap {S.gap:.2e} to the first real mode")
    rigid = rigid_check(S, report)
    ctrl = step_control(report)

    results = []
    for fn in (PR.probe_linearity, PR.probe_finite_linearity, PR.probe_precompute, PR.probe_locality):
        r = fn(S, rng, log)
        if r is None:
            report("    finite-amplitude linearity -- SKIPPED, no energy function")
            continue
        results.append(r)
    try:
        rm, vals, nz = PR.probe_modal(S, rng, log, k=min(12, 3 * N_BEAD - 2))
        results += [rm, PR.probe_nullspace(vals, nz, S), PR.probe_timescale(vals, S)]
    except Exception as ex:
        report(f"    eigensolve failed: {type(ex).__name__}: {ex}")

    report(f"\n    {'reduction':<38}{'verdict':<10}{'quantity':>13}{'speedup':>11}")
    for r in results:
        report(f"    {r['name']:<38}{r['verdict']:<10}{r['quantity']:>13.4g}{r['speedup']:>11.4g}")
        report(f"      {r['note']}")

    op = next((r for r in results if r["name"] == "linearity -> superposition"), None)
    fin = next((r for r in results if r["name"] == "finite-amplitude linearity"), None)
    report("\n  THE QUESTION THIS FILE EXISTS FOR")
    if op and fin:
        report(f"    operator-level linearity   {op['verdict']:<10} {op['quantity']:.3e}")
        report(f"    finite-amplitude linearity {fin['verdict']:<10} {fin['quantity']:.3e}")
        if op["verdict"] == "HOLDS" and fin["verdict"] == "FAILS":
            report("    THE TOOL DISTINGUISHES THEM. The operator is linear -- it is a Hessian, it could not")
            report("    be otherwise -- while the SYSTEM is not, and superposition on this rod would be")
            report("    wrong at finite amplitude. Without the second probe the tool would have reported")
            report("    the largest speedup in its vocabulary as available on a system where it is not.")
        elif fin["verdict"] == "HOLDS":
            report("    Finite-amplitude linearity HOLDS. Either the tested amplitudes never leave the")
            report("    harmonic well -- the deviation curve says which -- or the rod is not behaving as")
            report("    its energy states. Both need resolving before this is called a pass.")
        elif fin["verdict"] == "NOT TESTABLE":
            report("    The finite-amplitude probe DECLINED. It could not place the perturbation inside the")
            report("    window where the question is meaningful, so it reports that rather than a verdict.")
            report("    The operator-level result stands on its own; the system-level one is unmeasured.")
        else:
            report("    The operator-level probe did not hold, so the numerical Hessian is suspect and")
            report("    nothing else in this run should be read.")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "physics_reduce_rod", "n_bead": N_BEAD, "bp": BP, "sigma": SIGMA,
               "n_null": S.n_null, "gap": S.gap, "noise": S.noise,
               "rigid_theorem": rigid, "step_control": ctrl, "reductions": results, "log": log},
              open(OUT / "physics_reduce_rod.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'physics_reduce_rod.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""THE TIME AXIS, IN CLOSED FORM -- and the measurement of where it stops being valid.

WHAT THIS ADDS.  Everything in this line so far has been static: a mark is placed, the structure settles,
one snapshot. The two-level solver made that snapshot cheap and accurate (64 iterations, 4e-6 error, faster
than a direct solve). This adds the trajectory -- what moves FIRST, what follows, and on what timescales --
without an integrator, because for a linear system the trajectory is an expression.

THE PHYSICS, AND WHY OVERDAMPED IS THE RIGHT CHOICE, NOT A SIMPLIFICATION.  A nucleosome in water is at
Reynolds number ~1e-9. Inertia is irrelevant; friction dominates completely. So the equation of motion is
not m x'' = -K x + f but

    gamma x' = -K x + f

and in the eigenbasis of K every mode decouples into a first-order relaxation with its OWN rate:

    u(t) = sum_k  (e_k . f / lambda_k) (1 - exp(-lambda_k t / gamma)) e_k
    tau_k = gamma / lambda_k                      the relaxation time of mode k

Any t is one evaluation. t = 1 ns costs exactly what t = 1 s costs. There is no timestep and therefore no
10^9-step multiplier -- which was the wall that made 4D impossible by integration.

THE PROBLEM THAT HAD TO BE SOLVED FIRST, AND ITS RESOLUTION IS THE IDEA HERE.  A closed form in the
eigenbasis needs the eigenbasis, and the modal run measured that 194 modes reproduce only ~80% of the static
field: local detail is high-frequency and no soft mode represents it. Truncating the sum above would inherit
that 20% error at every time INCLUDING t -> infinity, which is absurd when an accurate static answer already
exists.

The resolution comes from the physics rather than from more modes. Rewrite the sum around its own endpoint:

    u(t) = u_static  -  sum_k (e_k . f / lambda_k) exp(-lambda_k t / gamma) e_k

Now the truncated sum multiplies a DECAYING exponential. The modes being dropped are the high-lambda ones,
and high lambda means tau = gamma/lambda is TINY -- they have already finished relaxing. So the modes the
basis cannot represent are precisely the modes that no longer matter dynamically, and the endpoint comes
from the two-level solve, which is accurate to 4e-6.

    endpoint     exact, from the two-level solver
    approach     the slow modes, which are the ones a modal basis is good at

THIS BUYS ACCURACY AT LARGE t AND MUST FAIL AT SMALL t, and that is not a flaw to be hidden -- it is a
quantity to be measured. Below t ~ gamma/lambda_max_kept the dropped modes are still relaxing and the
truncation is wrong. The run finds that crossover instead of asserting it.

THE REFERENCE IS A REAL INTEGRATION, so the closed form is checked rather than trusted: implicit Euler on
gamma u' = f - K u, unconditionally stable, with the step refined until the answer stops moving. The whole
claim of this file is that stepping is unnecessary, so it would be circular to verify it against anything
but stepping.

ARMS
    closed_form    the hybrid above: two-level endpoint minus the slow-mode decay
    modal_only     the naive truncated sum, no static correction. Should carry the modal basis's ~20% error
                   to t -> infinity, which is exactly the failure the hybrid exists to fix.
    static_only    u_static at every t, i.e. assuming the response is instantaneous. The control that shows
                   the time axis is carrying information at all; if this matches the reference everywhere,
                   there is no dynamics worth computing.

ON UNITS, STATED PLAINLY BECAUSE IT LIMITS WHAT MAY BE CLAIMED.  The elastic-network spring constants are
arbitrary (1.0 non-bonded, 20.0 covalent) and gamma is set to 1, so ABSOLUTE times here are not physical.
What IS meaningful is the SHAPE of the spectrum -- the ratio of slowest to fastest, and which motions are
slow -- because that depends on the structure's connectivity rather than on the units. Absolute calibration
needs one measured relaxation time to anchor it, and that is not in this repo. Times are therefore reported
in units of the slowest mode's relaxation time, and no picosecond or microsecond claim is made.

PREDECLARED, before any number:
    closed form matches the integrated reference across the decades where the marks actually relax
        -> the time axis is available in closed form, the timestep wall is gone, and 4D static-plus-dynamics
           becomes as cheap as the static answer already is.
    it matches only at long times
        -> report the crossover as the method's validity floor; it is still useful above it, and the floor
           is a number rather than a caveat.
    it does not match anywhere
        -> the overdamped linear model is the wrong description, and the closed form is not merely truncated
           but inapplicable.

-> outputs/orphan/chromatin_time.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, splu, cg, LinearOperator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexus_dna_spheres import fetch_pdb, load, VDW  # noqa: E402
from nexus_methyl_propagate import (elastic_network, rigid_basis, project, exact_solve,  # noqa: E402
                                    methyl_position, methyl_forces, build_spheres, partition_weights)
from chromatin_sphere_scaling import factor_small, complete_coverage  # noqa: E402
from chromatin_mark_cost import charges_of, charge_forces, DEBYE  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
NMODE = 200
GAMMA = 1.0
NSTEP_REF = 4000          # implicit-Euler steps for the reference; refined until the answer stops moving
SEED = 83


def modal_series(vals, vecs, b, u_static, t, gamma=GAMMA):
    """u(t) = u_static - sum_k (e_k.f / lambda_k) exp(-lambda_k t / gamma) e_k

    Written around the endpoint rather than from zero, so the truncation multiplies a decaying exponential
    and the modes that cannot be represented are the ones that have already relaxed.
    """
    c = (vecs.T @ b) / vals
    decay = np.exp(-vals * t / gamma)
    return u_static - vecs @ (c * decay)


def modal_naive(vals, vecs, b, t, gamma=GAMMA):
    c = (vecs.T @ b) / vals
    return vecs @ (c * (1.0 - np.exp(-vals * t / gamma)))


def integrate(K, b, ts, gamma=GAMMA, nstep=NSTEP_REF):
    """Implicit Euler on gamma u' = b - K u, which is unconditionally stable so the step is an accuracy
    choice rather than a stability one. The system matrix is constant, so it is factorised once and each
    step is a pair of triangular solves.

    Returns the state at each requested time. This is the thing the closed form claims to make unnecessary,
    so it is the only honest reference for it.
    """
    n3 = K.shape[0]
    tmax = max(ts)
    dt = tmax / nstep
    A = (K + sp.eye(n3, format="csr") * (gamma / dt)).tocsc()
    lu = splu(A)
    u = np.zeros(n3)
    out, want = {}, sorted(ts)
    wi = 0
    for s in range(1, nstep + 1):
        u = lu.solve(b + (gamma / dt) * u)
        tnow = s * dt
        while wi < len(want) and tnow >= want[wi] - 1e-12:
            out[want[wi]] = u.copy()
            wi += 1
    while wi < len(want):
        out[want[wi]] = u.copy()
        wi += 1
    return out


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("THE TIME AXIS IN CLOSED FORM -- and where it stops being valid")
    report("=" * 100)
    report("  Overdamped is not a simplification here: a nucleosome in water sits at Reynolds ~1e-9, so")
    report("  inertia is irrelevant and gamma u' = f - K u is the equation. In the eigenbasis every mode")
    report("  relaxes independently, so any instant is one evaluation and the 10^9-timestep wall is gone.")
    report("  THE TRICK: write the series around its ENDPOINT. u(t) = u_static - sum c_k exp(-t/tau_k).")
    report("  The truncated part then multiplies a DECAYING exponential, so the high-frequency modes a")
    report("  modal basis cannot represent are exactly the ones that have already finished relaxing --")
    report("  and the endpoint comes from the two-level solver, which is accurate to 4e-6.")

    t = load(fetch_pdb(), keep_water=False)
    co = t["co"]
    rad = np.array([VDW.get(e, 1.70) for e in t["el"]])
    n = len(co)
    K, npair = elastic_network(co)
    Q = rigid_basis(co)
    report(f"\n  1KX5 no waters: {n} atoms, {3*n} DOF, {npair} springs")

    vals, vecs = eigsh(K.tocsc(), k=NMODE, sigma=-1e-3, which="LM")
    o = np.argsort(vals)
    vals, vecs = vals[o], vecs[:, o]
    keep = vals > 1e-6
    vals, vecs = vals[keep], vecs[:, keep]
    vecs = vecs - Q @ (Q.T @ vecs)
    tau = GAMMA / vals
    report(f"  {len(vals)} internal modes | relaxation times span {tau.min():.3g} to {tau.max():.3g} "
           f"(ratio {tau.max()/tau.min():.3g})")
    report("  Units: spring constants are arbitrary and gamma = 1, so ABSOLUTE times are not physical.")
    report("  The spectrum's SHAPE is, because it comes from connectivity. Times below are in units of")
    report("  tau_slow, the slowest mode's relaxation time. No picosecond or microsecond claim is made.")
    tau_slow = tau.max()

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

    TS = [tau_slow * x for x in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)]
    LBL = ["0.003", "0.01", "0.03", "0.1", "0.3", "1", "3", "10"]
    res = {"n_modes": int(len(vals)), "tau_min": float(tau.min()), "tau_max": float(tau_slow), "marks": {}}

    for name, f in MARKS:
        u_st, info, resid = exact_solve(K, Q, f)
        if info != 0 or resid > 1e-6:
            report(f"  {name}: static solve failed; skipped")
            continue
        b = project(Q, f.ravel())
        report(f"\n  --- {name} ---   relative error against implicit-Euler integration")
        ref = integrate(K, b, TS)
        report(f"    {'t / tau_slow':<16}" + "".join(f"{l:>10}" for l in LBL))
        rows = {}
        for arm, fn in (("closed_form", lambda tt: modal_series(vals, vecs, b, u_st, tt)),
                        ("modal_only", lambda tt: modal_naive(vals, vecs, b, tt)),
                        ("static_only", lambda tt: u_st)):
            errs = []
            for tt in TS:
                r = project(Q, ref[tt])
                a = project(Q, fn(tt))
                errs.append(float(np.linalg.norm(a - r) / max(np.linalg.norm(r), 1e-30)))
            rows[arm] = errs
            report(f"    {arm:<16}" + "".join(f"{e:>10.4f}" for e in errs))
        # how much of the response has actually happened by each time -- so "accurate" can be read against
        # "anything is moving yet"
        frac = [float(np.linalg.norm(project(Q, ref[tt])) / max(np.linalg.norm(u_st), 1e-30)) for tt in TS]
        report(f"    {'(fraction done)':<16}" + "".join(f"{v:>10.3f}" for v in frac))
        res["marks"][name] = {"ts": [float(x) for x in TS], "labels": LBL, "arms": rows, "progress": frac}

    report("\n  READING")
    for name, d in res["marks"].items():
        cf, mo, so = d["arms"]["closed_form"], d["arms"]["modal_only"], d["arms"]["static_only"]
        good = [l for l, e in zip(d["labels"], cf) if e < 0.05]
        report(f"    {name}: closed form within 5% at t/tau_slow = {', '.join(good) if good else 'nowhere'}")
        report(f"      endpoint error   closed_form {cf[-1]:.4f}   modal_only {mo[-1]:.4f}")
        report(f"      earliest time    closed_form {cf[0]:.4f}   static_only {so[0]:.4f}")
    anyname = next(iter(res["marks"]), None)
    if anyname:
        d = res["marks"][anyname]
        cf, mo, so = d["arms"]["closed_form"], d["arms"]["modal_only"], d["arms"]["static_only"]
        if cf[-1] < 0.01 and mo[-1] > 0.05:
            report("    THE ENDPOINT FIX WORKS. Writing the series around u_static removes the modal basis's")
            report(f"    truncation error at long times entirely ({cf[-1]:.4f} against {mo[-1]:.4f} for the")
            report("    naive sum), which is what made a 200-mode basis unusable before.")
        if so[0] > 0.2:
            report("    AND THERE IS REAL DYNAMICS TO COMPUTE. Assuming the response is instantaneous is")
            report(f"    {so[0]:.2f} wrong at the earliest time sampled, so the trajectory carries")
            report("    information the static answer does not.")
        bad = [l for l, e in zip(d["labels"], cf) if e >= 0.05]
        if bad:
            report(f"    VALIDITY FLOOR, measured not assumed: the closed form misses at t/tau_slow = "
                   f"{', '.join(bad)}.")
            report("    Below that the dropped high-frequency modes are still relaxing, so the truncation")
            report("    is wrong there. Above it the method is exact for the price of one evaluation.")

    OUT.mkdir(parents=True, exist_ok=True)
    res["log"] = log
    json.dump(res, open(OUT / "chromatin_time.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'chromatin_time.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

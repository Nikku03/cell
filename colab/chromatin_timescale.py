"""IS THE TIME AXIS CREDIBLE?  -- the one missing calibration, and it decides the whole event scheme.

THE PROBLEM.  chromatin_time.py measured relaxation times spanning tau_min 7.2 to tau_max 48,794 and the
cost model then argued that quasi-static EVENT stepping is legitimate because the structure re-equilibrates
between events -- a polymerase adds one base every ~30 ms and the elastic response is "much faster". But
those tau are in the SOLVER'S UNITS. The spring constants are K_NB = 1.0 and K_BOND = 20.0, which are
arbitrary, and gamma was never written down at all. tau = gamma / lambda with both sides uncalibrated is not
a time. It is a number.

So the claim "relaxation is fast against 30 ms" has never been checked, and the entire event-driven
architecture rests on it. If relaxation is actually SLOWER than the interval between events, quasi-static
stepping is invalid and the whole scheme has to be replaced by real integration -- which the cost model
already showed costs decades.

TWO CONSTANTS, EACH FROM SOMETHING MEASURED RATHER THAN CHOSEN.

    THE SPRING SCALE, from crystallographic B-factors.  An elastic network at temperature T fluctuates by
    <u_i^2> = kT * tr[K^+]_ii, and the crystal recorded exactly that: B_i = (8 pi^2 / 3) <u_i^2>. So the
    scale s that turns model springs into kcal/mol/A^2 is fixed by requiring the model's mean fluctuation to
    equal the structure's own mean B-factor. Nothing is fitted per atom; it is one number.

    THE FRICTION, from Stokes drag.  gamma_i = 6 pi eta a with water at 37 C (eta = 0.69 mPa s) and an atomic
    radius of 1.9 A. This is the standard free-draining estimate and it IGNORES HYDRODYNAMIC COUPLING, which
    makes it an OVERESTIMATE of the drag on a collectively moving domain -- so the relaxation times it gives
    are upper bounds, which is the conservative direction for the question being asked.

THE TRACE IS COMPUTED, NOT TRUNCATED.  The mean fluctuation needs tr(K^+) over the non-rigid space, and
that sum is dominated by the softest modes but not exhausted by them -- the modal run measured that 194
modes reproduce only ~80% of a static field. Truncating would inflate the spring scale and shorten every
relaxation time, i.e. it would bias the answer toward the conclusion this file is trying to test. So the
trace is estimated by Hutchinson probing with exact projected solves instead, which has no truncation.

PREDECLARED, before any number:
    tau_max is well under 30 ms
        -> the timescale separation holds, quasi-static event stepping is legitimate, and the cost model's
           hours-to-days figures stand.
    tau_max is comparable to or longer than 30 ms
        -> it does not hold. The slowest collective modes have not relaxed between polymerase steps, the
           structure carries memory the scheme discards, and event stepping is wrong for transcription --
           whatever it does for slower processes like replication.
    tau_max lands within a factor of ~10 of 30 ms
        -> the separation is too thin to assert. It would then depend on which modes actually couple to the
           polymerase, and the honest answer is that the scheme needs a per-mode criterion rather than a
           global one.

-> outputs/chromatin_timescale.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexus_methyl_propagate import elastic_network, rigid_basis, project  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
PDB = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/1kx5.pdb")

# ---- physical constants, all standard --------------------------------------------------------------
T_K = 310.0                  # 37 C
KB = 1.380649e-23            # J/K
KT_J = KB * T_K              # J
KCAL_MOL = 6.9477e-21        # J per kcal/mol (i.e. 1 kcal/mol in joules per molecule)
KT_KCAL = KT_J / KCAL_MOL    # kT in kcal/mol -- ~0.616 at 37 C
ETA = 0.69e-3                # Pa s, water at 37 C
ATOM_RADIUS = 1.9e-10        # m, a heavy atom
NPROBE = 24                  # Hutchinson probes for tr(K^+)
POLY_INTERVAL = 1.0 / 33.0   # s between polymerase steps, 33 bp/s


def load_pdb(path):
    """Heavy atoms with their B-factors. Waters and hetero-atoms dropped -- the network is the particle."""
    co, b, el = [], [], []
    for ln in open(path):
        if not ln.startswith("ATOM"):
            continue
        e = ln[76:78].strip() or ln[12:16].strip()[0]
        if e == "H":
            continue
        co.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        b.append(float(ln[60:66]))
        el.append(e)
    return np.array(co), np.array(b), el


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("IS THE TIME AXIS CREDIBLE? -- calibrating the springs and the friction, then checking 30 ms")
    report("=" * 100)
    report("  tau = gamma / lambda was reported with BOTH sides uncalibrated. The event-driven architecture")
    report("  rests entirely on relaxation being fast against a polymerase step, and that has never been")
    report("  checked. Two constants, each from something measured.")

    co, bfac, el = load_pdb(PDB)
    n = len(co)
    report(f"\n  {PDB.name}: {n} heavy atoms | B-factor mean {bfac.mean():.2f} A^2, "
           f"median {np.median(bfac):.2f}, range {bfac.min():.1f}-{bfac.max():.1f}")
    K, npair = elastic_network(co)
    report(f"  elastic network: {npair} springs within 6 A (K_NB=1.0, K_BOND=20.0, model units)")
    Q = rigid_basis(co)

    # ---- tr(K^+) by Hutchinson, no truncation --------------------------------------------------------
    n3 = 3 * n
    op = LinearOperator((n3, n3), matvec=lambda v: project(Q, K.dot(project(Q, v))), dtype=float)
    rng = np.random.default_rng(7)
    report(f"\n  estimating tr(K^+) over the {n3-6}-dimensional non-rigid space by Hutchinson probing")
    report(f"  ({NPROBE} probes, exact projected CG each -- no modal truncation, which would bias the")
    report("  spring scale upward and every relaxation time downward)")
    ests = []
    for p in range(NPROBE):
        z = project(Q, rng.choice([-1.0, 1.0], size=n3))
        x, info = cg(op, z, rtol=1e-8, maxiter=5000)
        ests.append(float(z @ x))
        if (p + 1) % 8 == 0:
            report(f"    probe {p+1}/{NPROBE}  running estimate {np.mean(ests):.4g}  "
                   f"(se {np.std(ests)/np.sqrt(len(ests)):.2g})  {time.time()-t0:.0f}s")
    tr_pinv = float(np.mean(ests))
    se = float(np.std(ests) / np.sqrt(len(ests)))
    report(f"  tr(K^+) = {tr_pinv:.4g} +/- {se:.2g}  (model units)")

    # ---- the spring scale, from B-factors ------------------------------------------------------------
    # <u_i^2> summed over atoms = kT * tr(K^+_real) = (kT / s) * tr(K^+_model)
    # mean B = (8 pi^2 / 3) * mean <u_i^2>
    msf_model_mean = tr_pinv / n                       # per atom, model units, per kT
    s_kcal = (8 * np.pi ** 2 / 3) * KT_KCAL * msf_model_mean / bfac.mean()
    report(f"\n  THE SPRING SCALE, fixed by one number rather than fitted per atom")
    report(f"    kT at {T_K:.0f} K                        {KT_KCAL:.3f} kcal/mol")
    report(f"    model MSF per atom (per kT)         {msf_model_mean:.4g} A^2")
    report(f"    observed mean B-factor              {bfac.mean():.2f} A^2")
    report(f"    -> spring scale s                   {s_kcal:.4g} kcal/mol/A^2")
    report(f"    i.e. the soft non-bonded spring is  {s_kcal:.3g} kcal/mol/A^2 and the covalent one "
           f"{20*s_kcal:.3g}")
    report("    For orientation, a real covalent stretch is ~300-700 kcal/mol/A^2, so a value far below")
    report("    that would say the network is too soft and every time below is too long.")

    # ---- the friction, from Stokes -------------------------------------------------------------------
    gamma = 6 * np.pi * ETA * ATOM_RADIUS
    report(f"\n  THE FRICTION, from Stokes drag in water at {T_K-273:.0f} C")
    report(f"    eta {ETA*1e3:.2f} mPa s, atomic radius {ATOM_RADIUS*1e10:.1f} A")
    report(f"    -> gamma per atom {gamma:.3e} kg/s")
    report("    Free-draining, so hydrodynamic coupling is ignored. That OVERESTIMATES the drag on a")
    report("    collectively moving domain, making every relaxation time below an upper bound -- the")
    report("    conservative direction for the question being asked.")

    # ---- the spectrum, and the answer ----------------------------------------------------------------
    report(f"\n  slowest non-rigid modes ({time.time()-t0:.0f}s elapsed)")
    vals = eigsh(op, k=12, sigma=None, which="SM", return_eigenvectors=False, tol=1e-6, maxiter=20000)
    vals = np.sort(np.abs(vals))
    lam = vals[vals > 1e-8]
    lam_min = float(lam[0])
    lam_max_est = float(sp.linalg.norm(K, np.inf))
    S_SI = s_kcal * KCAL_MOL / 1e-20                   # kcal/mol/A^2 -> J/m^2 = N/m
    tau_slow = gamma / (lam_min * S_SI)
    tau_fast = gamma / (lam_max_est * S_SI)
    report(f"    lambda_min (non-rigid)  {lam_min:.4g} model units -> {lam_min*S_SI:.3e} N/m")
    report(f"    lambda_max (bound)      {lam_max_est:.4g} model units -> {lam_max_est*S_SI:.3e} N/m")
    report(f"\n  RELAXATION TIMES IN SECONDS")
    report(f"    slowest mode   tau_max = {tau_slow:.3e} s   ({tau_slow*1e9:.3f} ns)")
    report(f"    fastest mode   tau_min = {tau_fast:.3e} s   ({tau_fast*1e15:.3f} fs)")
    report(f"    polymerase step interval          {POLY_INTERVAL:.3e} s  ({POLY_INTERVAL*1e3:.0f} ms)")
    ratio = POLY_INTERVAL / tau_slow
    report(f"    SEPARATION: the polymerase interval is {ratio:.3g}x the slowest relaxation time")

    report("\n  READING")
    if ratio > 100:
        report(f"  THE SEPARATION HOLDS, by {ratio:.3g}x. Every elastic mode in the particle has relaxed")
        report("  completely before the polymerase takes its next step, so nothing is lost by refusing to")
        report("  resolve the interval. Quasi-static event stepping is legitimate and the cost model's")
        report("  hours-to-days figures stand on a measured foundation rather than an assumption.")
        report("  The claim this licenses is narrow and worth stating exactly: it covers the ELASTIC")
        report("  response of one nucleosome-scale particle. It says nothing about the slow modes of a")
        report("  megabase rod, whose relaxation grows steeply with contour length and which is the")
        report("  quantity that actually has to clear 30 ms at chromosome scale.")
    elif ratio > 10:
        report(f"  TOO THIN TO ASSERT, at {ratio:.3g}x. The separation is in the right direction but not by")
        report("  enough to wave through. It would need a per-mode criterion rather than a global one.")
    else:
        report(f"  THE SEPARATION FAILS at {ratio:.3g}x. The slowest modes have NOT relaxed between")
        report("  polymerase steps, the structure carries memory the quasi-static scheme discards, and")
        report("  event stepping is not valid for transcription at this rate.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "chromatin_timescale", "n_atoms": int(n), "n_springs": int(npair),
               "b_mean": float(bfac.mean()), "tr_pinv": tr_pinv, "tr_pinv_se": se,
               "spring_scale_kcal_mol_A2": float(s_kcal), "gamma_kg_s": float(gamma),
               "lambda_min": lam_min, "lambda_max_bound": lam_max_est,
               "tau_max_s": float(tau_slow), "tau_min_s": float(tau_fast),
               "polymerase_interval_s": POLY_INTERVAL, "separation_ratio": float(ratio),
               "log": log}, open(OUT / "chromatin_timescale.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'chromatin_timescale.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

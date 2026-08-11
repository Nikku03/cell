"""THE ENGINE: one eigendecomposition, and both the 3D map and the 4D dynamics fall out.

THE COMPUTE PROBLEM, stated honestly.  This container has four CPUs and no GPU. A Langevin simulation of
a 1,400-bead chromosome, ensembled over enough loop configurations to make a contact map, is hours of
integration per condition and the repo has already spent an arc on that road. So the model is built the
other way round.

THE IDEA.  A chromosome held together by harmonic bonds -- backbone plus cohesin loops -- is a GAUSSIAN
NETWORK, and a Gaussian network is never integrated. It is diagonalised. Write

    U = (k/2) * sum_bonds |r_i - r_j|^2          ->    L = the graph Laplacian of those bonds

and then everything the model has to produce is a function of L alone:

    3D   <R_ij^2> = b^2 * (L+_ii + L+_jj - 2 L+_ij)      the effective resistance between i and j
         P_contact ~ <R_ij^2>^(-3/2)                     Gaussian end-to-end density at short range
    4D   each eigenmode p relaxes independently at rate k*lambda_p/gamma, so
         MSD_i(t) = sum_p (2 b^2 / lambda_p) (1 - exp(-t/tau_p)) v_p(i)^2      tau_p = gamma/(k lambda_p)

One O(N^3) decomposition, about a second at N = 1400, and after it every pair contact and every
timepoint is arithmetic. That is the whole compute argument: the expensive object is computed once and
the time axis costs nothing, which is the only way 4D is affordable here.

WHAT THIS LOOP DOES AND DOES NOT DO.  It builds the engine and checks it against physics that is known
in advance -- identities and textbook exponents, not chromatin. No Hi-C is touched. A model validated
only against the thing it was tuned on is unfalsifiable, so the engine has to pass a test whose answer
was fixed before the code existed.

PREDECLARED, before any number:

    P1 THE IDEAL CHAIN IS EXACT   with no loops, <R_ij^2> must equal b^2 |i-j| to numerical precision.
                This is an IDENTITY, not a fit: the effective resistance between two nodes of a path
                graph is the number of edges between them. Relative error must be below 1e-8.
                fails -> the Laplacian machinery is wrong and every number after it is decoration.
    P2 THE FREE CHAIN GIVES THE TEXTBOOK EXPONENT, AND IT IS THE WRONG ONE FOR CHROMATIN
                a Gaussian chain has P(s) ~ s^-1.5. The engine must reproduce -1.5 +/- 0.05, AND that
                value must sit OUTSIDE the window loop 33 measured on real chromatin, [-1.16, -0.76].
                Both halves are the gate. The first says the engine is right; the second says a bare
                polymer is NOT a chromosome, so whatever closes that gap -- loops, confinement -- is
                doing real work and can be credited for it later rather than assumed.
    P3 THE DYNAMICS ARE ROUSE   monomer MSD must scale as t^0.5 in the intermediate regime, 0.5 +/- 0.05.
                This is the fourth dimension arriving in closed form. If the modal solution does not
                reproduce the one exponent Rouse dynamics is famous for, the time axis is not physics.
    P4 CONFINEMENT SATURATES BOTH   adding a harmonic tether (L -> L + c I, a nucleus of finite size)
                must make <R^2> plateau with separation AND make MSD plateau in time.
                A chromosome in a nucleus cannot diffuse forever or stretch forever. A model that lets
                it do either is not confined, whatever else it gets right.

CONTROLS: the ideal-chain identity as an exact positive control; textbook exponents fixed before the
run; the confined and unconfined engines compared on the same chain so the plateau cannot come from
anything else; and the free-chain exponent checked against loop 33's measured window so the gap the
rest of the project has to close is quantified up front.

-> outputs/loop_polymer.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 1904
N_BEADS = 1400          # chr21 at 25 kb, mappable part -- the real target size
B2 = 1.0                # b^2 = 3kT/k, the bond mean-square; sets the length unit
CONFINE = 3e-3          # tether strength; sets the confinement radius
GATE_IDENT = 1e-8
GATE_GAUSS = (-1.55, -1.45)
GATE_MSD = (0.45, 0.55)
CHROMATIN_WINDOW = (-1.16, -0.76)   # loop 33, measured


def laplacian(n, loops=(), confine=0.0):
    """Backbone path graph, optional loop bonds, optional harmonic tether to a fixed frame."""
    L = np.zeros((n, n))
    idx = np.arange(n - 1)
    L[idx, idx + 1] -= 1.0
    L[idx + 1, idx] -= 1.0
    L[idx, idx] += 1.0
    L[idx + 1, idx + 1] += 1.0
    for i, j in loops:
        if i == j:
            continue
        L[i, j] -= 1.0
        L[j, i] -= 1.0
        L[i, i] += 1.0
        L[j, j] += 1.0
    if confine > 0:
        L = L + confine * np.eye(n)
    return L


def r2_matrix(L, confined):
    """<R_ij^2> = b^2 (G_ii + G_jj - 2 G_ij).

    Unconfined, L is singular (the constant vector is a free translation) so the PSEUDO-inverse is the
    right object and the null direction is projected out. Confined, L is positive definite and a plain
    inverse is correct -- and the two must not be mixed up, because using the pseudo-inverse on a
    confined network silently removes exactly the mode that produces the plateau."""
    G = np.linalg.inv(L) if confined else np.linalg.pinv(L, hermitian=True)
    d = np.diag(G)
    return B2 * (d[:, None] + d[None, :] - 2.0 * G)


def ps_slope(R2, lo, hi):
    """Contact probability against separation, from P ~ <R^2>^(-3/2), fitted in log-log."""
    n = len(R2)
    s, p = [], []
    for d in range(1, n):
        i = np.arange(0, n - d)
        v = R2[i, i + d]
        v = v[v > 0]
        if len(v):
            s.append(d)
            p.append(np.mean(v ** -1.5))
    s, p = np.array(s, float), np.array(p)
    m = (s >= lo) & (s <= hi) & (p > 0)
    return float(np.polyfit(np.log10(s[m]), np.log10(p[m]), 1)[0]), s, p


def msd_curve(L, confined, times, probe):
    """MSD of one monomer, in closed form from the modes -- no integrator anywhere.

    The zero mode of an unconfined network is free translation of the whole chain: it contributes
    ordinary diffusion of the centre of mass, not internal motion, and is excluded so that what is
    measured is the chain's own relaxation."""
    w, V = np.linalg.eigh(L)
    keep = w > 1e-10 if not confined else np.ones(len(w), bool)
    w, V = w[keep], V[:, keep]
    amp = 2.0 * B2 * (V[probe, :] ** 2) / w
    return np.array([float(np.sum(amp * (1.0 - np.exp(-t * w)))) for t in times])


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    say("=" * 100)
    say("THE ENGINE -- one eigendecomposition, then the 3D map and the 4D dynamics are arithmetic")
    say("=" * 100)
    say("  Four CPUs, no GPU. So the chromosome is not integrated, it is diagonalised: a harmonic")
    say("  network is a linear system, and a linear system has closed forms for both its equilibrium")
    say("  contacts and its trajectory. No Hi-C is touched here -- the engine is checked against")
    say("  physics whose answer was fixed before the code existed.")

    n = N_BEADS
    t0 = time.time()
    L = laplacian(n)
    R2 = r2_matrix(L, confined=False)
    t_solve = time.time() - t0
    say(f"\n  chain of {n:,} beads; one decomposition took {t_solve:.2f}s and yields all "
        f"{n*(n-1)//2:,} pair distances")

    # ---- P1 the identity ---------------------------------------------------------------------------
    i, j = np.triu_indices(n, 1)
    want = B2 * (j - i).astype(float)
    err = float(np.max(np.abs(R2[i, j] - want) / want))
    p1 = bool(err < GATE_IDENT)
    say(f"\n  P1 THE IDEAL CHAIN IS EXACT -- <R_ij^2> vs the identity b^2|i-j|")
    say(f"     max relative error {err:.3e} over {len(i):,} pairs   (gate {GATE_IDENT:.0e})")
    say(f"     the effective resistance between two nodes of a path graph IS the edge count between")
    say(f"     them, so this is an identity and not a fit")
    say(f"     P1 {'PASS' if p1 else 'FAIL'}")

    # ---- P2 the textbook exponent, and the gap to real chromatin -----------------------------------
    sl, _, _ = ps_slope(R2, 4, n // 4)
    in_gauss = GATE_GAUSS[0] <= sl <= GATE_GAUSS[1]
    outside_chromatin = not (CHROMATIN_WINDOW[0] <= sl <= CHROMATIN_WINDOW[1])
    p2 = bool(in_gauss and outside_chromatin)
    say(f"\n  P2 THE FREE CHAIN GIVES THE TEXTBOOK EXPONENT, AND IT IS THE WRONG ONE FOR CHROMATIN")
    say(f"     simulated P(s) slope {sl:.4f}   (Gaussian theory -1.5, gate "
        f"[{GATE_GAUSS[0]}, {GATE_GAUSS[1]}])")
    say(f"     real chromatin measured in loop 33: -0.964, window "
        f"[{CHROMATIN_WINDOW[0]}, {CHROMATIN_WINDOW[1]}]")
    say(f"     the bare polymer is {abs(sl - (-0.964)):.3f} away from the chromosome, and OUTSIDE its")
    say(f"     window: {outside_chromatin}. That gap is what loops and confinement have to close, and")
    say(f"     quantifying it now means they can be credited later instead of assumed.")
    say(f"     P2 {'PASS' if p2 else 'FAIL'}  (engine correct AND bare polymer is not a chromosome)")

    # ---- P3 Rouse dynamics -------------------------------------------------------------------------
    times = np.logspace(-1, 4, 40)
    msd = msd_curve(L, False, times, probe=n // 2)
    m = (times >= 3) & (times <= 3e3)
    a_msd = float(np.polyfit(np.log10(times[m]), np.log10(msd[m]), 1)[0])
    p3 = bool(GATE_MSD[0] <= a_msd <= GATE_MSD[1])
    say(f"\n  P3 THE DYNAMICS ARE ROUSE -- monomer MSD exponent {a_msd:.4f}   "
        f"(theory 0.5, gate [{GATE_MSD[0]}, {GATE_MSD[1]}])")
    say(f"     computed from the modes in closed form; the time axis costs one exponential per mode")
    say(f"     and no timesteps at all, which is the entire reason 4D is affordable here")
    say(f"     P3 {'PASS' if p3 else 'FAIL'}")

    # ---- P4 confinement ----------------------------------------------------------------------------
    Lc = laplacian(n, confine=CONFINE)
    R2c = r2_matrix(Lc, confined=True)
    far = np.array([np.mean(R2c[np.arange(0, n - d), np.arange(0, n - d) + d])
                    for d in range(n // 4, n // 2, 10)])
    plateau_r2 = float(np.std(far) / np.mean(far))
    msdc = msd_curve(Lc, True, times, probe=n // 2)
    late = (times >= 1e3)
    plateau_msd = float(np.polyfit(np.log10(times[late]), np.log10(msdc[late]), 1)[0])
    p4 = bool(plateau_r2 < 0.05 and plateau_msd < 0.15)
    say(f"\n  P4 CONFINEMENT SATURATES BOTH -- harmonic tether c = {CONFINE}, a nucleus of finite size")
    say(f"     <R^2> at large separation: relative spread {plateau_r2:.4f}  (flat = plateau, gate < 0.05)")
    say(f"     free chain at the same separations rises without limit; confined one stops at "
        f"{far.mean():.1f} b^2")
    say(f"     MSD late-time exponent {plateau_msd:.4f}  (0 = plateau, gate < 0.15; free chain gives "
        f"{a_msd:.2f})")
    say(f"     P4 {'PASS' if p4 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"P1 ideal chain is exact": p1, "P2 textbook exponent, wrong for chromatin": p2,
             "P3 dynamics are Rouse": p3, "P4 confinement saturates both": p4}
    for k, v in gates.items():
        say(f"  {k:<44}{'PASS' if v else 'FAIL'}")
    if all(gates.values()):
        say(f"  THE ENGINE IS PHYSICS AND IT IS CHEAP. One {t_solve:.2f}s decomposition of a {n:,}-bead")
        say(f"  chain gives every pair distance and every timepoint, checked against an exact identity")
        say(f"  ({err:.0e}), the Gaussian exponent ({sl:.3f}), the Rouse exponent ({a_msd:.3f}) and")
        say(f"  confinement in both space and time.")
        say(f"  AND IT IS NOT YET A CHROMOSOME. Its P(s) is {sl:.3f} where chr21 measures -0.964. The")
        say(f"  bare polymer is the null model, not the answer, and the distance is now on the record.")
    else:
        say("  THE ENGINE IS NOT SOUND. Nothing should be built on it until these pass, because every")
        say("  chromatin number downstream would inherit the fault.")
    say("=" * 100)

    man = RM.manifest(inputs=["closed form, no external input"], available=n, used=n,
                      selection="all", seed=SEED,
                      controls=["ideal-chain identity as an exact positive control",
                                "textbook exponents fixed before the run",
                                "confined vs unconfined on the same chain",
                                "free-chain exponent compared to loop 33's measured window"],
                      note="validated only against physics known in advance; no Hi-C is touched, so "
                           "the engine cannot have been tuned to the thing it will be scored on")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_polymer", "manifest": man, "gates": gates,
               "n_beads": n, "solve_seconds": t_solve, "identity_max_rel_err": err,
               "ps_slope_free": sl, "chromatin_window": list(CHROMATIN_WINDOW),
               "msd_exponent": a_msd, "confined_r2_spread": plateau_r2,
               "confined_msd_late_exponent": plateau_msd, "log": log},
              open(OUT / "loop_polymer.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_polymer.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

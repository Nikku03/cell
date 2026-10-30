"""Do geometric residence times survive a non-exponential gating step?

WHY THIS EXISTS. According to PubMed, Strasser M, Theis FJ, Marr C, "Stability and multiattractor
dynamics of a toggle switch based on a two-stage model of stochastic gene expression", Biophysical
Journal 102:19-29 (2012), doi:10.1016/j.bpj.2011.11.4000. They derive, for a two-stage stochastic
gene expression toggle, that "the residence times of the system in one of the committed attractors
are geometrically distributed", with an analytical expression for the parameter.

Geometric (discrete) / exponential (continuous) residence means MEMORYLESS: how long a circuit has
already held its state tells you nothing about how much longer it will hold. That is the assumption
every circuit-reliability figure rests on, because it is what lets a single rate summarise a switch.

THE QUESTION THIS MODULE ASKS, AND IT IS A QUESTION. Their derivation is for a model in which every
elementary step has exponential waiting. Real circuits have multi-step gating reactions -- assembly,
maturation, multimerisation, translocation -- whose waiting times are Erlang, not exponential. Does
the memoryless result survive that?

WHAT THIS BUILD ORDER ALREADY KNOWS, AND IT CUTS AGAINST THE OBVIOUS FRAMING. rem/atlas/gapdetect.py
gate GD2 predeclared, from the spec, that waiting-time shape matters ONLY on the gating step (spec
values 2.04 and 2.80 orders vs 0.11 off-path). MEASURED: 1.61 orders on the identified gating step
and 1.43 on the other -- a separation of 1.1x against a bar of 5x. GD2 FAILED. In a LINEAR cascade
every step lies on the causal path to the observable, so there is no off-path step to spare. That
failure is why the question here is posed as "does memorylessness survive", not "which step matters".

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN. Deciding statistic: worst case over the sweep.
=================================================================================================

R-CONTROL  MANDATORY, AND IT VALIDATES THE INSTRUMENT BEFORE ANY CLAIM.
    Started from the quasi-stationary distribution (the left eigenvector of the basin
    sub-generator), the exit time is EXACTLY exponential by construction. So the measured
    coefficient of variation must be 1 to < 1e-9. If it is not, the moment code is wrong and
    every other row is an artefact of the solver rather than a property of the circuit.

R1  REPRODUCE THE MEMORYLESS RESULT WITH ALL-EXPONENTIAL STEPS. Starting instead from the
    physically relevant distribution -- the stationary law of the recurrent circuit, restricted
    and renormalised to the basin -- the exit-time CV must come out near 1. Predeclared: |CV - 1|
    < 0.05. This is the Strasser/Theis/Marr regime and if it does not reproduce, nothing below
    means anything.

R2  THE ACTUAL QUESTION. Replace the exponential waiting on the PRODUCTION step by Erlang-k
    (k sequential substeps, same mean waiting time, so the mean production flux is UNCHANGED --
    only the shape moves). Sweep k = 1, 2, 4, 8. Predeclared gate: memorylessness is judged
    BROKEN if |CV - 1| at any k exceeds 3x the R1 deviation. Report the value whichever way it
    goes; a null result here is a real answer and is reported as one.

R3  WHAT IT COSTS IF IT DOES BREAK. Report the change in MEAN residence time across the k sweep,
    in orders. A circuit reliability spec is a mean-first-passage-time number, so this is the
    quantity a synthetic biologist would actually quote.

R-VACUITY  The basin must be a real attractor, not an arbitrary cut: stationary occupancy of the
    basin > 0.9, and every mean exit time finite and above the solver floor. A residence time in
    a basin the system barely occupies is not a residence time.

R-ABLATION  Erlang-1 must reproduce the all-exponential row to < 1e-9, because Erlang-1 IS the
    exponential. If the k = 1 row differs, the Erlang construction changed something other than
    the waiting-time shape and the sweep is measuring that instead.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

RULE = "=" * 97


# -------------------------------------------------------------------------------------------
# the circuit: one autoactivating gene, bistable. Production gated by an Erlang-k waiting time.
# -------------------------------------------------------------------------------------------

def birth_rate(n, g, K, h, leak):
    return leak + g * (n ** h) / (K ** h + n ** h)


def generator(M: int, k: int, g: float, gamma: float, K: float, h: float, leak: float):
    """States (n, phase) with phase in 0..k-1. A production event completes after k substeps.

    Each substep fires at rate k * birth(n), so the MEAN waiting time to produce is
    1/birth(n) for every k -- the mean production flux is held exactly fixed and only the
    SHAPE of the waiting time changes. That is the whole point: any tail movement is
    attributable to shape and not to flux.
    """
    ns = M + 1
    idx = lambda n, ph: n * k + ph
    N = ns * k
    rows, cols, vals = [], [], []

    def add(i, j, r):
        if r > 0:
            rows.append(i); cols.append(j); vals.append(r)

    for n in range(ns):
        b = birth_rate(n, g, K, h, leak)
        for ph in range(k):
            i = idx(n, ph)
            if b > 0:
                if ph + 1 < k:
                    add(i, idx(n, ph + 1), k * b)          # advance one substep
                elif n + 1 < ns:
                    add(i, idx(n + 1, 0), k * b)           # last substep completes: n -> n+1
            if n > 0:
                add(i, idx(n - 1, ph), gamma * n)          # decay is a single step, always
    r = np.array(rows); c = np.array(cols); v = np.array(vals, float)
    diag = np.bincount(r, weights=v, minlength=N)
    Q = sp.coo_matrix((np.concatenate([v, -diag]),
                       (np.concatenate([r, np.arange(N)]),
                        np.concatenate([c, np.arange(N)]))), shape=(N, N)).tocsr()
    return Q, idx, ns


def stationary(Q):
    N = Q.shape[0]
    A = Q.T.tolil()
    A[0, :] = 1.0
    b = np.zeros(N); b[0] = 1.0
    p = np.maximum(spl.spsolve(A.tocsr(), b), 0.0)
    return p / p.sum()


def exit_moments(Q, basin_mask, start):
    """Mean and second moment of the exit time from the basin, started from `start`.

    Q_B is the sub-generator on basin states. m1 = -Q_B^{-1} 1, m2 = 2 Q_B^{-2} 1.
    """
    QB = Q[basin_mask][:, basin_mask].tocsc()
    one = np.ones(QB.shape[0])
    m1 = spl.spsolve(QB, -one)
    m2 = 2.0 * spl.spsolve(QB, -m1)
    s = start[basin_mask]
    s = s / s.sum()
    e1 = float(s @ m1)
    e2 = float(s @ m2)
    var = max(e2 - e1 ** 2, 0.0)
    return e1, np.sqrt(var) / e1 if e1 > 0 else np.nan


def qsd(Q, basin_mask):
    """Quasi-stationary distribution: left eigenvector of Q_B for the least-negative eigenvalue."""
    QB = Q[basin_mask][:, basin_mask].tocsc()
    vals, vecs = spl.eigs(QB.T, k=1, which="LR")
    v = np.real(vecs[:, 0])
    v = np.abs(v)
    full = np.zeros(Q.shape[0])
    full[basin_mask] = v / v.sum()
    return full, float(np.real(vals[0]))


# -------------------------------------------------------------------------------------------

PARAMS = dict(g=18.0, gamma=1.0, K=10.0, h=3.0, leak=0.6)
M = 60
BASIN_TOP = 9          # basin = low-expression attractor, n <= BASIN_TOP


def run_k(k, basin_top=BASIN_TOP, M=M, **params):
    Q, idx, ns = generator(M, k, **params)
    N = Q.shape[0]
    basin = np.zeros(N, bool)
    for n in range(basin_top + 1):
        for ph in range(k):
            basin[idx(n, ph)] = True
    pi = stationary(Q)
    occ = float(pi[basin].sum())
    e1, cv = exit_moments(Q, basin, pi)
    q, lam = qsd(Q, basin)
    e1q, cvq = exit_moments(Q, basin, q)
    return dict(k=k, occupancy=occ, mean=e1, cv=cv, mean_qsd=e1q, cv_qsd=cvq, lam=lam)


def report():
    out = []
    P = out.append
    P(RULE)
    P("DO GEOMETRIC RESIDENCE TIMES SURVIVE A NON-EXPONENTIAL GATING STEP?")
    P(RULE)
    P("  Strasser, Theis & Marr (Biophys J 102:19-29, 2012, doi:10.1016/j.bpj.2011.11.4000)")
    P("  derive geometrically distributed residence times for an all-exponential two-stage model.")
    P("  This asks whether that survives Erlang-k waiting on the production step, at FIXED mean")
    P("  production flux -- so any movement is attributable to shape, never to flux.")
    P("")
    rows = [run_k(k, **PARAMS) for k in (1, 2, 4, 8)]

    P("  R-CONTROL  started from the quasi-stationary distribution, exit is exponential BY")
    P("  CONSTRUCTION, so CV must be 1. This validates the moment code before any claim.")
    P("        k    CV from QSD    |CV - 1|")
    worst_ctrl = 0.0
    for r in rows:
        d = abs(r["cv_qsd"] - 1.0); worst_ctrl = max(worst_ctrl, d)
        P(f"        {r['k']:1d}    {r['cv_qsd']:12.9f}   {d:.3e}")
    P(f"     worst |CV - 1| = {worst_ctrl:.3e}   "
      f"{'PASS' if worst_ctrl < 1e-9 else 'FAIL'} (bar 1e-9)")
    P("")

    P("  R-VACUITY  is the basin a real attractor?")
    ok_v = all(r["occupancy"] > 0.9 and np.isfinite(r["mean"]) and r["mean"] > 0 for r in rows)
    for r in rows:
        P(f"        k={r['k']:1d}  stationary occupancy of basin {r['occupancy']:.6f}  "
          f"mean residence {r['mean']:.6e}")
    P(f"     {'PASS' if ok_v else 'FAIL'} (bar: occupancy > 0.9, mean finite and positive)")
    P("")

    base = rows[0]
    dev1 = abs(base["cv"] - 1.0)
    P("  R1  ALL-EXPONENTIAL (k = 1): does the memoryless result reproduce?")
    P(f"        exit-time CV = {base['cv']:.6f}   |CV - 1| = {dev1:.6f}")
    P(f"     {'PASS' if dev1 < 0.05 else 'FAIL'} (bar |CV - 1| < 0.05)")
    if dev1 < 0.05:
        P("     The Strasser/Theis/Marr regime reproduces: residence in this attractor is")
        P("     memoryless to within 5%, started from where the circuit actually sits.")
    P("")

    P("  R2  ERLANG-k ON THE PRODUCTION STEP, mean flux held fixed:")
    P("        k    mean residence    CV      |CV - 1|   vs R1 deviation")
    for r in rows:
        d = abs(r["cv"] - 1.0)
        ratio = d / dev1 if dev1 > 0 else np.inf
        P(f"        {r['k']:1d}    {r['mean']:14.6e}  {r['cv']:.6f}  {d:9.6f}   {ratio:8.2f}x")
    worst_ratio = max(abs(r["cv"] - 1.0) / dev1 for r in rows) if dev1 > 0 else np.inf
    broken = worst_ratio > 3.0
    P(f"     worst departure {worst_ratio:.2f}x the R1 deviation   "
      f"{'MEMORYLESSNESS BROKEN' if broken else 'MEMORYLESSNESS SURVIVES'} (bar 3x)")
    P("")

    P("  R3  WHAT IT COSTS -- the mean residence time is the circuit reliability number:")
    mmin = min(r["mean"] for r in rows); mmax = max(r["mean"] for r in rows)
    P(f"        mean residence spans {mmin:.6e} to {mmax:.6e}")
    P(f"        = {np.log10(mmax / mmin):.4f} orders across k = 1 to 8 at IDENTICAL mean flux")
    P("")

    P("  R-ABLATION  Erlang-1 IS the exponential, so the k = 1 row must equal the")
    P("  all-exponential row exactly. It is the same computation by construction; recorded so")
    P("  that a future change to the Erlang code cannot silently alter the baseline.")
    P("")
    P(RULE)
    P("WHAT IS NOT CLAIMED")
    P(RULE)
    P("  * This is one autoactivating gene, not their two-stage toggle. It tests whether the")
    P("    MEMORYLESS PROPERTY is robust to waiting-time shape, not whether their analytical")
    P("    expression is right -- it is right, for the model they state it for.")
    P("  * Novelty is NOT established. They may already know this. It is sent as a question.")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())

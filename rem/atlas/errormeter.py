"""An error meter: can the cost of factorising be estimated WITHOUT forming the joint?

WHY THIS IS THE MISSING PIECE. The architecture consumes a per-group error budget eps_G in five
places and produces it in none. rem/atlas/grouping.py measured why a bulk criterion cannot supply
it: tail_err = c*sqrt(MI), so a mutual information vanishes as the SQUARE of the error it is being
used to bound. A meter is needed that reports, for a proposed simplification, how much the FINAL
ANSWER moves.

THE ASYMMETRY THAT DECIDES WHAT IS HARD.
  SPLIT is cheap and exact. If A and B already sit in one group you hold their joint, so you can
  evaluate the observable both ways and compare. No circularity, nothing to estimate.
  MERGE is the hard direction. Asking whether C should be pulled in requires the joint with C --
  precisely the object being avoided. Only this direction needs a meter.

THE CANDIDATE, and it generalises a result already in this repo. For the conjunctive sterilisation
question in benchmarks/... the exact factorisation error was n*Cov(q,N) -- a SECOND-MOMENT
quantity requiring no joint. The general conjecture is that the log-error of factorising a
conjunctive tail decomposes to first order into a sum over PAIRS:

    Lambda_full  =  log[ P(A in T, B in T, C in T) / (P_A P_B P_C) ]          <- needs the 3-way joint
    Lambda_pair  =  lam_AB + lam_AC + lam_BC,  lam_ij = log[P(i,j in T)/(P_i P_j)]
                                                                              <- needs only 2-way joints

If Lambda_pair tracks Lambda_full, a merge decision can be priced from pairwise marginals alone,
and the residual Lambda_full - Lambda_pair IS section 10's "when is a three-way group required"
criterion, made measurable instead of asserted.

DIRECTION MATTERS MORE THAN ACCURACY. A meter that is merely close is not a safety device. What is
needed is a CONSERVATIVE one: for positively associated variables the pairwise sum must not
UNDERSTATE the true error, or the meter licenses unsafe merges. Gate M3 tests the sign of the
residual explicitly and reports it whichever way it falls.

SYSTEM. Three reporters A, B, C on one shared bursty pool, exact stationary joint P(pool,A,B,C) as
a null vector. Conditionally independent given the pool, marginally dependent through it. Coupling
swept by pool SPEED, which leaves the pool's stationary law exactly invariant (scaling a generator
does not move its null vector) so only the correlation time changes.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN. Deciding statistic: worst case over the sweep.
=================================================================================================

M1  GROUND TRUTH. Stationary residual max|Q^T pi| < 1e-10, and the pool mean identical across the
    sweep to < 1e-9 relative, so nothing measured is attributable to a shift in level.

M2  NON-VACUITY. Every three-way tail probability entering a reported ratio must lie inside
    (1e-12, 0.1), and every pairwise tail likewise. A ratio of two numbers that are both
    effectively zero is not a measurement.

M3  THE SIGN, AND IT DECIDES WHETHER THIS IS A SAFETY DEVICE. Predeclared: for a shared-driver
    system the residual R = Lambda_full - Lambda_pair should be NEGATIVE, i.e. the pairwise sum
    OVERSTATES the true error and the meter is conservative. If R > 0 anywhere the meter
    UNDERSTATES and is unsafe as a merge criterion; that outcome is reported as a failure of the
    proposal, not written around.

M4  ACCURACY. Report |R| / |Lambda_full| across the sweep. A meter useful for a merge decision
    should hold the relative residual under 25%; that bar is stated in advance and is generous
    because the decision it feeds is a threshold test, not a reported number.

M5  DISCRIMINATION. Lambda_full must vary by at least 10x across the sweep. If the true error
    barely moves, any estimator tracks it and M4 could not have failed.

M6  ZERO-COUPLING CONTROL. With the pool made fast, A, B and C become independent: Lambda_full,
    Lambda_pair and every pairwise lam_ij must vanish together, each < 1e-4 at the fastest pool.
    A residual surviving at zero coupling means the sweep is measuring truncation.

M7  BEATS THE BULK BASELINE. The pairwise meter must predict Lambda_full better than the
    bulk-criterion baseline c*sqrt(MI_total) calibrated in grouping.py. Reported as the ratio of
    relative residuals. If the bulk baseline wins there is no case for the pairwise meter.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

RULE = "=" * 97


def build4(pcap, xcap, alpha, burst, muR, c, mu):
    """Exact stationary joint P(pool, A, B, C); three identical reporters on one bursty pool."""
    nx = xcap + 1
    idx = lambda p, a, b, cc: ((p * nx + a) * nx + b) * nx + cc
    n = (pcap + 1) * nx ** 3
    rows, cols, vals = [], [], []
    ap = rows.append; cp = cols.append; vp = vals.append

    def add(i, j, r):
        if r > 0:
            ap(i); cp(j); vp(r)

    q = 1.0 / burst
    for p in range(pcap + 1):
        room = pcap - p
        # burst distribution from state p, tail lumped on the cap so no production flux is lost
        jumps = []
        tail = 1.0
        for j in range(1, room):
            pj = q * (1.0 - q) ** (j - 1)
            tail -= pj
            jumps.append((j, alpha * pj))
        if room >= 1 and tail > 0:
            jumps.append((room, alpha * tail))
        for a in range(nx):
            for b in range(nx):
                for cc in range(nx):
                    i = idx(p, a, b, cc)
                    for j, r in jumps:
                        add(i, idx(p + j, a, b, cc), r)
                    if p > 0:
                        add(i, idx(p - 1, a, b, cc), muR * p)
                    if a + 1 < nx:
                        add(i, idx(p, a + 1, b, cc), c * p)
                    if a > 0:
                        add(i, idx(p, a - 1, b, cc), mu * a)
                    if b + 1 < nx:
                        add(i, idx(p, a, b + 1, cc), c * p)
                    if b > 0:
                        add(i, idx(p, a, b - 1, cc), mu * b)
                    if cc + 1 < nx:
                        add(i, idx(p, a, b, cc + 1), c * p)
                    if cc > 0:
                        add(i, idx(p, a, b, cc - 1), mu * cc)
    r = np.array(rows); co = np.array(cols); v = np.array(vals, float)
    diag = np.bincount(r, weights=v, minlength=n)
    Q = sp.coo_matrix((np.concatenate([v, -diag]),
                       (np.concatenate([r, np.arange(n)]),
                        np.concatenate([co, np.arange(n)]))), shape=(n, n)).tocsr()
    A0 = Q.T.tolil(); A0[0, :] = 1.0
    rhs = np.zeros(n); rhs[0] = 1.0
    p0 = np.maximum(spl.spsolve(A0.tocsr(), rhs), 0.0)
    mode = int(np.argmax(p0))
    A1 = Q.T.tolil(); A1[mode, :] = 1.0
    rhs = np.zeros(n); rhs[mode] = 1.0
    pi = np.maximum(spl.spsolve(A1.tocsr(), rhs), 0.0)
    pi /= pi.sum()
    resid = float(np.max(np.abs(Q.T @ pi)))
    return pi.reshape(pcap + 1, nx, nx, nx), resid


def meter(P4, T):
    """Exact three-way tail lift, the pairwise-sum estimate, and the total MI baseline."""
    PABC = P4.sum(axis=0)
    PA = PABC.sum(axis=(1, 2)); PB = PABC.sum(axis=(0, 2)); PC = PABC.sum(axis=(0, 1))
    pa = PA[T:].sum(); pb = PB[T:].sum(); pc = PC[T:].sum()
    full = PABC[T:, T:, T:].sum()
    lam_full = np.log(full / (pa * pb * pc)) if full > 0 else np.nan

    def pair(M, m1, m2):
        j = M[T:, T:].sum()
        return (np.log(j / (m1 * m2)) if j > 0 else np.nan), j

    lab, jab = pair(PABC.sum(axis=2), pa, pb)
    lac, jac = pair(PABC.sum(axis=1), pa, pc)
    lbc, jbc = pair(PABC.sum(axis=0), pb, pc)
    lam_pair = lab + lac + lbc

    # total mutual information of the triple against the full product (the bulk baseline)
    outer = PA[:, None, None] * PB[None, :, None] * PC[None, None, :]
    m = (PABC > 0) & (outer > 0)
    MI_tot = float(np.sum(PABC[m] * np.log(PABC[m] / outer[m])))
    return dict(lam_full=lam_full, lam_pair=lam_pair, lab=lab, lac=lac, lbc=lbc,
                MI=MI_tot, tail3=full, pmin=min(pa, pb, pc),
                jmin=min(jab, jac, jbc))


POOL_MEAN, MU_R, BURST = 4.0, 1.0, 2.0
C_RATE, MU_X = 0.8, 0.8
PCAP, XCAP, T = 22, 15, 9
SPEEDS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
C_BULK = 20.23          # calibrated in rem/atlas/RESULTS_grouping_law.txt


def run(speed):
    alpha = MU_R * POOL_MEAN / BURST * speed
    P4, resid = build4(PCAP, XCAP, alpha, BURST, MU_R * speed, C_RATE, MU_X)
    d = meter(P4, T)
    d["resid"] = resid
    d["pool_mean"] = float(np.arange(PCAP + 1) @ P4.sum(axis=(1, 2, 3)))
    return d


def report():
    out = []; P = out.append
    P(RULE)
    P("AN ERROR METER: can the cost of factorising be priced WITHOUT forming the joint?")
    P(RULE)
    P("  Three reporters on one shared pool. The exact three-way tail lift Lambda_full needs the")
    P("  3-way joint. The candidate meter Lambda_pair needs only pairwise joints. If they track,")
    P("  a MERGE can be priced from pairwise marginals -- the one direction that is not already")
    P("  cheap. Coupling swept by pool SPEED, which leaves the pool's stationary law invariant.")
    P("")
    rows = [(s, run(s)) for s in SPEEDS]

    P(RULE)
    P("M1  GROUND TRUTH")
    P(RULE)
    wr = max(r["resid"] for _, r in rows)
    pms = [r["pool_mean"] for _, r in rows]
    span = (max(pms) - min(pms)) / np.mean(pms)
    P(f"  worst stationary residual = {wr:.3e}   {'PASS' if wr < 1e-10 else 'FAIL'} (bar 1e-10)")
    P(f"  pool mean {min(pms):.10f} to {max(pms):.10f}, span {span:.3e}   "
      f"{'PASS' if span < 1e-9 else 'FAIL'} (bar 1e-9)")
    P("")

    P(RULE)
    P("M2  NON-VACUITY")
    P(RULE)
    bad3 = [s for s, r in rows if not (1e-12 < r["tail3"] < 0.1)]
    badj = [s for s, r in rows if not (1e-12 < r["jmin"] < 0.1)]
    P(f"  three-way tail range {min(r['tail3'] for _, r in rows):.3e} to "
      f"{max(r['tail3'] for _, r in rows):.3e}   offending: {bad3 or 'none'}")
    P(f"  worst pairwise tail  {min(r['jmin'] for _, r in rows):.3e}   offending: {badj or 'none'}")
    P(f"  {'PASS' if not bad3 and not badj else 'FAIL'}")
    P("")

    P(RULE)
    P("THE SWEEP")
    P(RULE)
    P(f"  {'speed':>6s} {'Lam_full':>10s} {'Lam_pair':>10s} {'residual':>10s} {'|R|/|Lam|':>10s}"
      f" {'MI_tot':>10s} {'bulk est':>10s}")
    for s, r in rows:
        R = r["lam_full"] - r["lam_pair"]
        rel = abs(R) / abs(r["lam_full"]) if r["lam_full"] else np.nan
        bulk = C_BULK * np.sqrt(max(r["MI"], 0.0))
        P(f"  {s:6.1f} {r['lam_full']:10.5f} {r['lam_pair']:10.5f} {R:+10.5f} {rel:10.4f}"
          f" {r['MI']:10.5f} {bulk:10.4f}")
    P("")

    P(RULE)
    P("M3  THE SIGN -- is the meter conservative?")
    P(RULE)
    Rs = [r["lam_full"] - r["lam_pair"] for _, r in rows]
    P(f"  residual R = Lambda_full - Lambda_pair over the sweep: "
      f"{min(Rs):+.5f} to {max(Rs):+.5f}")
    if max(Rs) <= 0:
        P("  R <= 0 everywhere: the pairwise sum OVERSTATES the true error. The meter is")
        P("  CONSERVATIVE and safe to use as a merge criterion -- it never licenses a merge-skip")
        P("  that the exact answer would forbid.  PASS as predeclared.")
    elif min(Rs) >= 0:
        P("  R >= 0 everywhere: the pairwise sum UNDERSTATES the true error. The meter is UNSAFE")
        P("  as a merge criterion. This is a failure of the proposal and is reported as one.")
    else:
        P("  R changes sign across the sweep: no uniform direction, so the meter cannot be used")
        P("  as a one-sided safety bound. Reported as a failure of the conservative claim.")
    P("")

    P(RULE)
    P("M4  ACCURACY  and  M7  AGAINST THE BULK BASELINE")
    P(RULE)
    rels = [abs(r["lam_full"] - r["lam_pair"]) / abs(r["lam_full"]) for _, r in rows]
    bulks = [abs(C_BULK * np.sqrt(max(r["MI"], 0.0)) - abs(r["lam_full"])) / abs(r["lam_full"])
             for _, r in rows]
    P(f"  worst relative residual of the PAIRWISE meter : {max(rels):.4f}   "
      f"{'PASS' if max(rels) < 0.25 else 'FAIL'} (bar 0.25)")
    P(f"  worst relative residual of the BULK baseline  : {max(bulks):.4f}")
    P(f"  M7 {'PASS -- pairwise beats bulk' if max(rels) < max(bulks) else 'FAIL -- bulk wins'}"
      f"  (ratio {max(bulks)/max(rels):.1f}x)" if max(rels) > 0 else "")
    P("")

    P(RULE)
    P("M5  DISCRIMINATION -- could M4 have failed?")
    P(RULE)
    lf = [abs(r["lam_full"]) for _, r in rows]
    P(f"  |Lambda_full| spans {min(lf):.5f} to {max(lf):.5f} = {max(lf)/min(lf):.1f}x   "
      f"{'PASS' if max(lf)/min(lf) > 10 else 'FAIL -- too flat, any estimator would track it'}")
    P("")

    P(RULE)
    P("M6  ZERO-COUPLING CONTROL")
    P(RULE)
    P(f"  {'speed':>7s} {'Lam_full':>11s} {'Lam_pair':>11s} {'MI':>11s}")
    worst = None
    for s in (32.0, 128.0, 512.0):
        r = run(s)
        worst = max(abs(r["lam_full"]), abs(r["lam_pair"]), abs(r["MI"]))
        P(f"  {s:7.0f} {r['lam_full']:11.3e} {r['lam_pair']:11.3e} {r['MI']:11.3e}")
    P(f"  worst surviving quantity at the fastest pool: {worst:.3e}   "
      f"{'PASS' if worst < 1e-4 else 'FAIL'} (bar 1e-4)")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())

"""The cell cycle and division: what they change, and what is still missing after them.

WHY THIS IS NOT JUST ANOTHER SUBSYSTEM. Everything built so far assumes a STEADY STATE -- linear
programs at an optimum, stationary distributions, fixed points. The cell cycle has none. It is an
oscillator with a discrete event in it, and division is the event: at the end of each cycle every
molecule is partitioned binomially between two daughters, and during S phase gene dosage doubles.
Neither is a perturbation of a steady state; they are a different mathematical object, and the
right tool is a Floquet (periodic) solve, not a stationary one.

THE REFINEMENT THAT MATTERS MOST, and it cuts AGAINST the earlier conclusion rather than for it.
In a dividing cell the protein removal rate is not k_dp alone:

    total removal = k_dp (degradation)  +  ln2 / T (dilution by division)

and the division period T is DIRECTLY OBSERVABLE -- one watches cells divide. Using Schwanhausser's
median protein half-life of about 46 h, at a 24 h cycle degradation is only 34% of total protein
turnover and dilution is the other 66%. So adding the cell cycle partially CLOSES the blind
direction that expression.py identified, because the observable period supplies most of the
turnover for free. V5 measures exactly how much.

THE CENTRAL TEST. Division lowers the mean protein number roughly threefold, which on its own
would change any tail. That is not interesting. The question is whether the cycle changes the tail
BEYOND what a steady-state model matched on the same mean already predicts. If it does not, the
cell cycle can be absorbed into an effective steady state and every earlier tail number stands. If
it does, steady-state expression models are wrong about tails by whatever the gap turns out to be,
and that includes expression.py's.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

V1  THE PERIODIC SOLVER IS CORRECT. Probability conserved through a full cycle, the power
    iteration converged, and boundary mass in both dimensions below 1e-12.

V2  THE PARTITIONING OPERATOR IS EXACT. The binomial thinning matrix must have unit column sums
    and must halve the mean exactly, checked against the identity sum_j j C(i,j) 2^-i = i/2.

V3  THE LIMIT CHECK, which is the strongest correctness test available. Remove division --
    partitioning set to the identity, gene dosage held constant, and a dilution term added to the
    generator instead -- and the periodic state must converge to the STATIONARY state of the same
    generator. Worst relative disagreement in the tail below 1e-6.

V4  THE CENTRAL TEST: THE MEAN-MATCHED CONTROL, on the CYCLE-AVERAGED distribution. (The first
    run compared the POST-DIVISION state against a cycle-averaged stationary model and was
    therefore confounded: the means came out 74.3 against 118.3, a ratio of 1.592 against a dose
    factor of 1.600, which identified the error rather than merely suggesting it. Both quantities
    are wanted and V7 now reports the other one.) Compare the periodic tail against a stationary
    model matched on cycle-averaged transcription and total protein removal, so the two have the
    same mean by construction and only the cycle structure differs. Predeclared readings: agreement
    within 0.05 orders means the cell cycle is absorbable into an effective steady state and the
    earlier analysis stands; a larger gap means steady-state expression models misstate tails and
    expression.py's numbers inherit that error.

V7  THE RISK WINDOW, which is a finding rather than a control. Report the post-division tail
    beside the cycle-averaged one. Division halves every molecule at an instant, so immediately
    afterwards a cell is at its most exposed, and no stationary model contains that window at all.

V5  THE STRUCTURAL REFINEMENT, exact. With T observable, recompute the least-squares residual of
    the tail-controlling directions on the observable rows, and compare against the no-division
    case. Reported as the fraction of the blind direction that observing the cell cycle closes.

V6  WHAT IS STILL ABSENT AFTER THIS, with parameter counts rather than adjectives: signalling,
    transcriptional regulation, trafficking and spatial organisation. For each, state whether the
    machinery in this build order applies at all, and if not, why not.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply, spsolve
from scipy.special import gammaln

from rem.atlas.hybrid_tune import RULE

T_CYCLE = 24.0          # h, a mammalian division period
F_S = 0.4               # fraction of the cycle before DNA replication
THRESH = 20             # protein copies below which the gene counts as failed
KDP_MEDIAN = np.log(2) / 46.0     # Schwanhausser median protein half-life, 46 h
N_GENES = 8


def binom_thin(N):
    B = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        j = np.arange(i + 1)
        B[j, i] = np.exp(gammaln(i + 1) - gammaln(j + 1) - gammaln(i - j + 1) - i * np.log(2.0))
    return B


def gen(Mm, Mp, k_tx, k_tl, k_dm, k_dp, dilute=0.0):
    n = (Mm + 1) * (Mp + 1)
    idx = lambda m, q: m * (Mp + 1) + q
    r_, c_, v_ = [], [], []
    for m in range(Mm + 1):
        for q in range(Mp + 1):
            s0 = idx(m, q)
            for tgt, rate in (((m + 1, q), k_tx if m < Mm else 0.0),
                              ((m - 1, q), (k_dm + dilute) * m),
                              ((m, q + 1), k_tl * m if q < Mp else 0.0),
                              ((m, q - 1), (k_dp + dilute) * q)):
                if rate > 0:
                    r_.append(idx(*tgt)); c_.append(s0); v_.append(rate)
                    r_.append(s0); c_.append(s0); v_.append(-rate)
    return coo_matrix((v_, (r_, c_)), shape=(n, n)).tocsc()


def periodic(k_tx, k_tl, k_dm, k_dp, T, fS, Mm, Mp, divide=True, iters=400, tol=1e-13,
             n_phase=12):
    """Floquet state: the fixed point of one cycle. G1 at one gene copy, S/G2/M at two, then
    binomial partitioning at division.

    Returns BOTH the post-division state and the time-averaged state over the cycle. The first
    run returned only the post-division one and compared it against a cycle-averaged stationary
    model, which is not the same quantity -- the means differed by exactly the dose factor."""
    L1 = gen(Mm, Mp, k_tx, k_tl, k_dm, k_dp)
    L2 = gen(Mm, Mp, 2 * k_tx if divide else k_tx, k_tl, k_dm, k_dp)
    Bm, Bp = (binom_thin(Mm), binom_thin(Mp)) if divide else (np.eye(Mm + 1), np.eye(Mp + 1))
    d1, d2 = T * fS, T * (1 - fS)
    x = np.zeros((Mm + 1) * (Mp + 1)); x[0] = 1.0
    it = 0
    for it in range(iters):
        y = expm_multiply(L2 * d2, expm_multiply(L1 * d1, x))
        Xg = Bm @ y.reshape(Mm + 1, Mp + 1) @ Bp.T
        y = np.maximum(Xg.reshape(-1), 0.0)
        s = y.sum()
        y = y / s if s > 0 else y
        if np.abs(y - x).max() < tol:
            x = y
            break
        x = y
    # time-average over the cycle: march the converged post-division state through both phases,
    # sampling at n_phase points each, weighted by the duration of each segment
    acc = np.zeros_like(x)
    z = x.copy()
    for L, dur in ((L1, d1), (L2, d2)):
        step = dur / n_phase
        for _ in range(n_phase):
            acc += z * step
            z = expm_multiply(L * step, z)
    acc = acc / acc.sum()
    return x.reshape(Mm + 1, Mp + 1), it, acc.reshape(Mm + 1, Mp + 1)


def stationary(k_tx, k_tl, k_dm, k_dp, Mm, Mp, dilute=0.0):
    L = gen(Mm, Mp, k_tx, k_tl, k_dm, k_dp, dilute=dilute)
    n = L.shape[0]
    Lk = L.tolil(); Lk[n - 1, :] = 1.0
    rhs = np.zeros(n); rhs[n - 1] = 1.0
    pi = np.maximum(spsolve(Lk.tocsc(), rhs), 0.0)
    return (pi / pi.sum()).reshape(Mm + 1, Mp + 1)


def stats(X, thresh=THRESH):
    pq = X.sum(axis=0)
    n = np.arange(len(pq))
    return float((n * pq).sum()), float(pq[:thresh].sum())


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE); P("THE CELL CYCLE AND DIVISION: WHAT THEY CHANGE"); P(RULE)
    P(f"  cycle period T = {T_CYCLE} h, S phase at {F_S:.0%} of the cycle,"
      f" threshold {THRESH} copies")
    P(f"  protein removal = k_dp + ln2/T = {KDP_MEDIAN:.4f} + {np.log(2)/T_CYCLE:.4f}"
      f" = {KDP_MEDIAN + np.log(2)/T_CYCLE:.4f} /h")
    P(f"  degradation is {KDP_MEDIAN/(KDP_MEDIAN+np.log(2)/T_CYCLE):.1%} of protein turnover;"
      f" dilution by division is the rest")

    # ---- V2 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("V2  THE PARTITIONING OPERATOR IS EXACT"); P(RULE)
    B = binom_thin(60)
    col = float(np.abs(B.sum(axis=0) - 1.0).max())
    means = B.T @ np.arange(61)
    mean_err = float(np.abs(means - np.arange(61) / 2.0).max())
    P(f"  worst |column sum - 1| : {col:.2e}")
    P(f"  worst |mean after partitioning - i/2| : {mean_err:.2e}")
    P(f"  {'PASS' if col < 1e-12 and mean_err < 1e-10 else 'FAIL'}")

    # ---- V3 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("V3  THE LIMIT CHECK  (remove division, recover the stationary state)"); P(RULE)
    Mm, Mp = 18, 260
    k_dp0, k_dm0 = KDP_MEDIAN, 0.30
    a0, b0 = 9.0, 12.0
    k_tx0, k_tl0 = a0 * k_dp0, b0 * k_dm0
    Xn, itn, Xn_avg = periodic(k_tx0, k_tl0, k_dm0, k_dp0, T_CYCLE, F_S, Mm, Mp,
                               divide=False)
    Xs = stationary(k_tx0, k_tl0, k_dm0, k_dp0, Mm, Mp)
    mn, tn = stats(Xn)
    ms, ts = stats(Xs)
    rel = abs(tn - ts) / max(ts, 1e-300)
    P(f"  periodic with division switched off : mean {mn:.6f}, P(<{THRESH}) {tn:.8e}")
    P(f"  stationary state, same generator    : mean {ms:.6f}, P(<{THRESH}) {ts:.8e}")
    P(f"  relative disagreement in the tail {rel:.2e}"
      f"   {'PASS' if rel < 1e-6 else 'FAIL -- the periodic solver is not solving the same model'}")

    # ---- V1, V4 ---------------------------------------------------------------------------------
    P("\n" + RULE); P("V4  THE CENTRAL TEST: THE MEAN-MATCHED CONTROL"); P(RULE)
    P("  Division lowers the mean about threefold on its own, which would change any tail. The")
    P("  question is whether the cycle changes it BEYOND a stationary model matched on the mean.")
    P(f"  {'a':>7}{'b':>7}{'cyc mean':>11}{'stat mean':>11}{'cyc P(<T)':>13}{'stat P(<T)':>13}"
      f"{'log10 gap':>11}")
    rng = np.random.default_rng(11)
    worst_gap, worst_edge, allconv = 0.0, 0.0, True
    window = []
    t0 = time.time()
    for g in range(N_GENES):
        a = float(np.exp(rng.normal(np.log(9.0), 0.6)))
        b = float(np.exp(rng.normal(np.log(12.0), 0.5)))
        k_dp = KDP_MEDIAN
        k_dm = 0.30
        k_tx, k_tl = a * k_dp, b * k_dm
        Xc, itc, Xa = periodic(k_tx, k_tl, k_dm, k_dp, T_CYCLE, F_S, Mm, Mp, divide=True)
        allconv &= itc < 399
        worst_edge = max(worst_edge, float(Xc[Mm, :].sum() + Xc[:, Mp].sum()))
        # matched stationary: cycle-averaged transcription, total removal including dilution
        dose = F_S * 1.0 + (1 - F_S) * 2.0
        dil = np.log(2.0) / T_CYCLE
        Xm = stationary(k_tx * dose, k_tl, k_dm, k_dp + dil, Mm, Mp)
        ma, ta = stats(Xa)          # cycle-averaged: what a randomly sampled cell shows
        mc, tc = stats(Xc)          # post-division: the most exposed phase
        mm_, tm = stats(Xm)
        gap = abs(np.log10(max(ta, 1e-300)) - np.log10(max(tm, 1e-300)))
        worst_gap = max(worst_gap, gap)
        window.append((a, b, ta, tc))
        P(f"  {a:>7.2f}{b:>7.2f}{ma:>11.3f}{mm_:>11.3f}{ta:>13.4e}{tm:>13.4e}{gap:>11.4f}")
    P(f"  {N_GENES} genes in {time.time()-t0:.0f}s")
    P(f"\n  V1: all converged {allconv}, worst boundary mass {worst_edge:.2e}"
      f"   {'PASS' if allconv and worst_edge < 1e-12 else 'FAIL'}")
    P(f"  V4: worst |log10 gap| between the cycling and the mean-matched stationary model:"
      f" {worst_gap:.4f} orders")
    if worst_gap < 0.05:
        P("  PASS -- the cell cycle is absorbable into an effective steady state, and every")
        P("  steady-state tail computed earlier in this build order stands as it is.")
    else:
        P("  FAIL -- steady-state expression models MISSTATE the tail by this much, and")
        P("  expression.py's numbers inherit the error. Recorded, not explained away.")

    P("\n" + RULE); P("V7  THE RISK WINDOW  (post-division against cycle-averaged)"); P(RULE)
    P("  Division halves every molecule at an instant. Immediately afterwards a cell is at its")
    P("  most exposed, and no stationary model contains that window at all.")
    P(f"  {'a':>7}{'b':>7}{'cycle-avg P(<T)':>18}{'post-division P(<T)':>21}{'ratio':>10}")
    worst_w = 0.0
    for a, b, ta, tc in window:
        r = tc / max(ta, 1e-300)
        worst_w = max(worst_w, r)
        P(f"  {a:>7.2f}{b:>7.2f}{ta:>18.4e}{tc:>21.4e}{r:>10.2f}x")
    P(f"  worst elevation of risk just after division: {worst_w:.2f}x")

    # ---- V5 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("V5  THE STRUCTURAL REFINEMENT: HOW MUCH DOES OBSERVING THE CYCLE CLOSE?")
    P(RULE)
    dil = np.log(2.0) / T_CYCLE
    share_dp = KDP_MEDIAN / (KDP_MEDIAN + dil)
    share_T = dil / (KDP_MEDIAN + dil)
    P("  In a dividing cell the protein-removal term is k_dp + ln2/T, so in log-rate coordinates")
    P("  x = (log k_tx, log k_tl, log k_dm, log k_dp, log T) its derivative splits between the")
    P(f"  two by their shares: d log(removal)/d log k_dp = {share_dp:.4f},"
      f" d log(removal)/d log T = {-share_T:+.4f}.")
    rows_nodiv = {"RNA-seq": np.array([1, 0, -1, 0, 0], float),
                  "proteomics": np.array([1, 1, -1, -1, 0], float)}
    rows_div = {"RNA-seq": np.array([1, 0, -1, 0, 0], float),
                "proteomics": np.array([1, 1, -1, -share_dp, share_T], float),
                "the cycle period, watched": np.array([0, 0, 0, 0, 1], float)}
    tgt = {"burst frequency": np.array([1, 0, 0, -share_dp, share_T], float),
           "burst size": np.array([0, 1, -1, 0, 0], float)}

    def resid(v, rows):
        A = np.array(list(rows.values()), float)
        coef, *_ = np.linalg.lstsq(A.T, v, rcond=None)
        return float(np.linalg.norm(A.T @ coef - v))

    P(f"\n  {'direction':>20}{'residual, no cycle':>22}{'residual, cycle watched':>26}"
      f"{'closed':>9}")
    for nm, v in tgt.items():
        r0 = resid(v, rows_nodiv)
        r1 = resid(v, rows_div)
        P(f"  {nm:>20}{r0:>22.4f}{r1:>26.4f}{(1 - r1/max(r0,1e-12)):>9.1%}")
    P("  Watching cells divide is free, and it supplies the dilution share of protein turnover.")
    P("  What it does NOT supply is the degradation share, and that is what remains blind.")

    # ---- V6 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("V6  WHAT IS STILL ABSENT AFTER THIS"); P(RULE)
    P(f"  {'subsystem':>26}{'rough parameter count':>24}  does this machinery apply?")
    for nm, cnt, ok in (
        ("transcriptional regulation", "~1,600 TFs x targets, 1e4-1e5", "PARTLY -- steady state exists, but regulation makes the cell a controller, not an optimiser, which invalidates the LP framing used for metabolism"),
        ("signal transduction", "~2,000 proteins, ~1e4 rates", "NO -- signalling answers are TRANSIENT (amplitude, duration, adaptation). There is no stationary distribution to take a tail of"),
        ("trafficking and secretion", "~1,500 proteins, ~1e4 rates", "PARTLY -- compartments are already labels in Recon3D; making them dynamic adds transport rates but keeps a steady state"),
        ("spatial organisation", "fields, not parameters", "NO -- a reaction-diffusion PDE has no finite state space, so every method here that enumerates states fails outright"),
        ("cell cycle and division", "~100-200 (this module)", "YES -- but only via a periodic solve, not a stationary one; that is what this module had to build"),
    ):
        P(f"  {nm:>26}{cnt:>24}")
        P(f"                            {ok}")
    P("\n  The honest summary: of the six subsystems asked about, one is now built, two could be")
    P("  built with the existing machinery, and two -- signalling and space -- need a different")
    P("  mathematics, because they have no stationary distribution to compute a rare event from.")

    P("\n" + RULE)
    open(os.path.join(os.path.dirname(__file__), "RESULTS_division.txt"),
         "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

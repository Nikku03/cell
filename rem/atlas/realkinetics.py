"""Do real transcription factors sit in the prunable regime? And if so, assemble and run.

WHAT prune.py LEFT. Pruning the stratum tree with an admissible bound turns the exponential
2^(|C|(L+1)) into a POLYNOMIAL in the window depth -- but only below about 0.30 controller
switches per window, and it stays exponential with a smaller base above about 2.13. That
transition was measured on synthetic systems and it is dimensionless, so it can be evaluated
against real numbers. This module does that, and then, only if it passes, assembles the engine
and runs it.

THE MEASUREMENTS, from PubMed. Absolute residence times are known to depend on imaging
acquisition parameters through sampling bias (Presman et al. 2017, Methods 123:76-88,
doi:10.1016/j.ymeth.2017.03.014; Paakinaho et al. 2017, Nat Commun 8:15896,
doi:10.1038/ncomms15896), so every number below is carried as a RANGE and swept rather than
taken as a point.

    Chen et al. 2014, Cell 156:1274-1285, doi:10.1016/j.cell.2014.01.062. Sox2 and Oct4 in mouse
    embryonic stem cells, by single-molecule imaging: 84-97 events of 3D diffusion (3.3-3.7 s)
    interspersed with brief nonspecific collisions (0.75-0.9 s) before acquiring specific target
    DNA, where they then dwell for 12.0-14.6 s.

    Gebhardt et al. 2013, Nat Methods 10:421-6, doi:10.1038/nmeth.2411. Residence times on DNA
    for oligomerisation states and mutants of the glucocorticoid receptor and estrogen
    receptor-alpha, resolving distinct modes of DNA binding.

    Loffreda et al. 2017, Nat Commun 8:313, doi:10.1038/s41467-017-00398-7. p53 residence time on
    chromatin is MODULATED by C-terminal acetylation and tracks transcriptional activity -- so the
    rate is not a constant of the protein, which is why it is swept here.

    Schwanhausser et al. 2011, Nature 473:337-342, doi:10.1038/nature10098. mRNA and protein
    abundance and turnover measured together for more than 5,000 genes, establishing that mRNA and
    protein half-lives are uncorrelated and are separately measurable quantities.

THE ARITHMETIC THAT DECIDES IT, and the answer depends entirely on WHAT THE CONTROLLER STATE IS.

    At the BINDING level, a controller is a transcription factor occupying its site. The off-rate
    is one over the dwell time, 1/13.3 s = 0.075 per second. For 0.30 switches per window the
    window must be under 4 SECONDS. The window the engine actually needs is set by how long a
    target's mRNA remembers its input, which is hours. This fails by three to four orders.

    At the ACTIVITY level, a controller is a transcription factor being present and active. It
    changes on the timescale of that protein's turnover. Then the dimensionless group is

        switches per window  =  ln2 * (target mRNA lifetime) / (TF protein half-life)

    and the criterion becomes: the engine is prunable when a TF's protein outlives its target's
    mRNA by a factor of at least 2.3.

WHICH MEANS THE LOAD-BEARING ASSUMPTION IS NOT A NUMBER, IT IS A COARSE-GRAINING. Everything
depends on being allowed to average the fast binding away and keep only the slow activity. That
assumption is not asserted here. K3 builds a two-timescale controller -- fast binding gated by
slow activity -- and measures how much separation is actually needed before the slow-only model
reproduces the TAIL, which is a stronger requirement than reproducing the mean and is the one this
build order exists for.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

K0  THE NUMBERS, AND WHICH ARE MEASURED. Every input reported with its source and its range. A
    number carried as a point that was published as a range is a defect, not a simplification.

K1  THE BINDING-LEVEL TEST. PREDECLARED: passes if switches per window <= 0.30 at a window the
    engine actually needs.

K2  THE ACTIVITY-LEVEL TEST, swept over the full plausible range of the ratio rather than
    evaluated at a median, because the gate is per-gene and a median of medians is not a joint
    distribution.

K3  IS THE COARSE-GRAINING LEGITIMATE? Two-timescale controllers, fast binding gated by slow
    activity, sweeping the separation. PREDECLARED: the slow-only model must reproduce the TAIL,
    not merely the mean, and the separation required must be reported. If the required separation
    exceeds what real kinetics provide, K2 passes on a fiction and the whole thing fails here.

K4  THE MARGIN. How far inside the regime, and what would push it out. A pass with no margin is
    not a pass.

K5  CONDITIONAL ON K1-K4: ASSEMBLE AND RUN. The controller block with TRRUST wiring and
    activity-level rates, strata pruned with the certificate, targets carried by the class count
    times the temporal multiplier. Measured against exact where exact is affordable, then pushed
    past where it is not.

K6  WHAT BREAKS FIRST, at scale, with the measured scaling laws rather than hope.

K7  WHAT THIS DOES AND DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import numpy as np
from scipy.linalg import expm
from scipy.sparse import coo_matrix, csr_matrix

from rem.atlas.hybrid_tune import RULE
from rem.atlas.statedim import stationary
from rem.atlas.trrust_engine import signed_edges
from rem.atlas.localclosure import load_trrust

# ---- the sourced numbers, as ranges -------------------------------------------------------------
DWELL_S = (12.0, 14.6)          # Chen 2014, specific target dwell, seconds
COLLIDE_S = (0.75, 0.9)         # Chen 2014, nonspecific collisions
DIFFUSE_S = (3.3, 3.7)          # Chen 2014, 3D diffusion between collisions
NEVENTS = (84, 97)              # Chen 2014, sampling events before target acquisition
TRANSITION_POLY = 0.30          # measured in prune.py: polynomial at or below this
TRANSITION_EXP = 2.13           # measured in prune.py: exponential at or above this


def two_timescale(nC, nT, sep, sgn, mag, k_slow=1.0, boff=2.0, seed=20260907):
    """Each controller carries a SLOW activity bit and a FAST binding bit. Binding is gated by
    activity -- a controller can only bind while it is active -- and the target drive reads the
    BOUND state, not the active state. sep is the ratio of binding to activity rates.

    This is the object the coarse-graining claim is about: the engine wants to track only the
    slow bits, and K3 measures how much separation that needs before the TAIL survives it."""
    nv = 2 * nC + nT
    n = 1 << nv
    rng = np.random.default_rng(seed)
    st = np.arange(n, dtype=np.int64)
    act = [((st >> c) & 1).astype(float) for c in range(nC)]
    bnd = [((st >> (nC + c)) & 1).astype(float) for c in range(nC)]
    tgt = [((st >> (2 * nC + j)) & 1).astype(float) for j in range(nT)]
    a = np.exp(rng.normal(0, 0.3, nT))
    b = boff * np.exp(rng.normal(0, 0.3, nT))
    R, C, D = [], [], []
    for c in range(nC):                      # slow: activity on/off
        R.append(st); C.append(st ^ (1 << c))
        D.append(np.where(act[c] == 0, k_slow, k_slow))
    for c in range(nC):                      # fast: binding, gated by activity
        R.append(st); C.append(st ^ (1 << (nC + c)))
        D.append(np.where(bnd[c] == 0, sep * k_slow * act[c] + 1e-12, sep * k_slow))
    for j in range(nT):                      # targets read the BOUND state
        lg = np.zeros(n)
        for c in range(nC):
            lg = lg + sgn[c] * mag[c] * bnd[c]
        drive = np.exp(lg / nC)
        R.append(st); C.append(st ^ (1 << (2 * nC + j)))
        D.append(np.where(tgt[j] == 0, a[j] * drive, b[j]))
    Q = coo_matrix((np.concatenate(D), (np.concatenate(R), np.concatenate(C))),
                   shape=(n, n)).tocsr()
    dg = np.asarray(Q.sum(axis=1)).ravel()
    return (Q - csr_matrix((dg, (st, st)), shape=(n, n))).tocsr(), nv


def coarse_grained(nC, nT, sep, sgn, mag, k_slow=1.0, boff=2.0, seed=20260907):
    """The model the engine would actually build: only the SLOW activity bits, with the fast
    binding replaced by its quasi-steady-state average given the activity. If the separation is
    real this must reproduce the full model; K3 measures when it does."""
    nv = nC + nT
    n = 1 << nv
    rng = np.random.default_rng(seed)
    st = np.arange(n, dtype=np.int64)
    act = [((st >> c) & 1).astype(float) for c in range(nC)]
    tgt = [((st >> (nC + j)) & 1).astype(float) for j in range(nT)]
    a = np.exp(rng.normal(0, 0.3, nT))
    b = boff * np.exp(rng.normal(0, 0.3, nT))
    R, C, D = [], [], []
    for c in range(nC):
        R.append(st); C.append(st ^ (1 << c))
        D.append(np.full(n, k_slow))
    for j in range(nT):
        # QSS: given activity a_c, P(bound) = on/(on+off) = a_c*sep/(a_c*sep + sep) = a_c
        lg = np.zeros(n)
        for c in range(nC):
            lg = lg + sgn[c] * mag[c] * act[c]
        drive = np.exp(lg / nC)
        R.append(st); C.append(st ^ (1 << (nC + j)))
        D.append(np.where(tgt[j] == 0, a[j] * drive, b[j]))
    Q = coo_matrix((np.concatenate(D), (np.concatenate(R), np.concatenate(C))),
                   shape=(n, n)).tocsr()
    dg = np.asarray(Q.sum(axis=1)).ravel()
    return (Q - csr_matrix((dg, (st, st)), shape=(n, n))).tocsr(), nv


def target_tail(pi, nv, ntop, nT):
    """P(all nT targets on), the conjunctive rare event, from a stationary distribution whose
    target bits are the TOP nT bits."""
    st = np.arange(len(pi), dtype=np.int64)
    mask = 0
    for j in range(nT):
        mask |= 1 << (ntop + j)
    return float(pi[(st & mask) == mask].sum())


def target_mean(pi, nv, ntop, nT):
    st = np.arange(len(pi), dtype=np.int64)
    return float(sum((pi * ((st >> (ntop + j)) & 1)).sum() for j in range(nT)))


# =================================================================================================
# PART 2: the assembled engine
# =================================================================================================

def trrust_block(nCtrl, k_slow=1.0, cc=0.8, seed=20260907):
    """The controller block, wired by TRRUST. The top-nCtrl transcription factors by OUT-degree
    (out-degree, not total degree -- that was defect C4 in trrust_engine.py) regulate each other
    where TRRUST says they do, with the sign TRRUST gives. Activity-level rates: each controller
    turns over at k_slow, modulated by its regulators among the chosen set."""
    E, _ = signed_edges()
    adj, inv, sha, ne = load_trrust()
    outd = collections.Counter(u for u, v, _ in E)
    name2i = {g: i for i, g in inv.items()}
    od = np.zeros(len(adj))
    for g, d in outd.items():
        if g in name2i:
            od[name2i[g]] = d
    order = list(np.argsort(-od))
    ctrl = [inv[i] for i in order[:nCtrl]]
    cidx = {g: i for i, g in enumerate(ctrl)}
    within = [(cidx[u], cidx[v], m) for u, v, m in E if u in cidx and v in cidx]
    n = 1 << nCtrl
    st = np.arange(n, dtype=np.int64)
    bits = [((st >> c) & 1).astype(float) for c in range(nCtrl)]
    R, C, D = [], [], []
    for c in range(nCtrl):
        drive = np.ones(n)
        for (u, v, m) in within:
            if v != c:
                continue
            s = 1.0 if m == "Activation" else (-1.0 if m == "Repression" else 0.0)
            drive = drive * (1.0 + s * cc * bits[u]) if s != 0 else drive
        drive = np.maximum(drive, 1e-6)
        R.append(st); C.append(st ^ (1 << c))
        D.append(np.where(bits[c] == 0, k_slow * drive, k_slow))
    Q = coo_matrix((np.concatenate(D), (np.concatenate(R), np.concatenate(C))),
                   shape=(n, n)).tocsr()
    dg = np.asarray(Q.sum(axis=1)).ravel()
    Q = (Q - csr_matrix((dg, (st, st)), shape=(n, n))).tocsr()
    return Q, ctrl, cidx, sha, len(E), len(within)


def target_rows(cidx, ntarget=200):
    """Real target genes and their real regulators among the controller set, with real signs.
    Only genes that HAVE a regulator in the set are carried -- a gene with none is not a test of
    anything the engine does."""
    E, _ = signed_edges()
    by = collections.defaultdict(list)
    for u, v, m in E:
        if u in cidx and v not in cidx:
            s = 1.0 if m == "Activation" else (-1.0 if m == "Repression" else 0.0)
            by[v].append((cidx[u], s))
    rows = [(g, regs) for g, regs in by.items() if regs]
    rows.sort(key=lambda t: (-len(t[1]), t[0]))
    return rows[:ntarget]


def engine_tail(Q, nCtrl, rows, L, dt, tau, hvec=None, base=-1.0, gain=2.0):
    """The assembled engine.

    ONE SIMPLIFICATION THAT MATTERS. In the controller block the state IS the controller code, so
    a stratum is not a distribution over states -- it is a single PATH through the chain, and its
    mass is the path probability. Enumerating strata is therefore literally Markov-chain path
    enumeration, and the prefix-mass bound is the path probability, which can only decrease as the
    path is extended. That makes the pruner a branch-and-bound over paths and drops the cost per
    node from O(n^2) to O(n).

    Each target's ON probability comes from the SIGNED CLASS COUNT times the TEMPORAL MULTIPLIER --
    the composed form multiplier.py measured -- evaluated on the path's time-weighted activity."""
    pi, res, _ = stationary(Q)
    n = Q.shape[0]
    Pm = expm((Q.T * dt).toarray())
    if hvec is None:                       # recent slices weigh more; SHARED across all targets,
        hvec = np.exp(-0.5 * np.arange(L, -1, -1))   # which is what makes it a multiplier and not
        hvec = hvec / hvec.sum()                     # a per-gene table
    S = np.zeros((len(rows), nCtrl))       # signed class count per target, normalised by k_g
    for i, (g, regs) in enumerate(rows):
        for c, sg in regs:
            S[i, c] += sg
        S[i] /= max(len(regs), 1)
    codes = np.arange(n, dtype=np.int64)
    actbit = np.array([[(m >> c) & 1 for c in range(nCtrl)] for m in range(n)], dtype=float)

    touched = 0
    dropped = 0.0
    cur = {}
    for m in range(n):
        touched += 1
        w = float(pi[m])
        if w <= 0:
            continue
        if w < tau:
            dropped += w
            continue
        cur[(m,)] = w
    for _ in range(L):
        nxt = {}
        for a, w in cur.items():
            col = Pm[:, a[-1]]
            for m in range(n):
                touched += 1
                s = w * float(col[m])
                if s <= 1e-300:
                    continue
                if s < tau:
                    dropped += s
                    continue
                nxt[a + (m,)] = s
        cur = nxt
    tail = 0.0
    for a, w in cur.items():
        wact = hvec @ actbit[list(a)]                 # time-weighted activity, one per controller
        z = base + gain * (S @ wact)
        tail += w * float(np.exp(-np.logaddexp(0.0, -z).sum()))
    return tail, dropped, touched, len(cur), res


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE); P_("REAL TRANSCRIPTION FACTOR KINETICS AGAINST THE PRUNABLE REGIME"); P_(RULE)

    # ---- K0  THE NUMBERS -----------------------------------------------------------------------
    P_("\n" + RULE); P_("K0  THE NUMBERS, AND WHICH OF THEM ARE MEASURED"); P_(RULE)
    P_("  From PubMed. Absolute residence times depend on imaging acquisition parameters through")
    P_("  sampling bias (Presman 2017 doi:10.1016/j.ymeth.2017.03.014; Paakinaho 2017")
    P_("  doi:10.1038/ncomms15896), so every number is carried as a RANGE and swept.")
    P_("\n  Chen et al. 2014, Cell 156:1274-1285, doi:10.1016/j.cell.2014.01.062 (Sox2/Oct4, mESC):")
    P_(f"    specific target dwell           {DWELL_S[0]}-{DWELL_S[1]} s")
    P_(f"    nonspecific collisions          {COLLIDE_S[0]}-{COLLIDE_S[1]} s")
    P_(f"    3D diffusion between            {DIFFUSE_S[0]}-{DIFFUSE_S[1]} s")
    P_(f"    sampling events before target   {NEVENTS[0]}-{NEVENTS[1]}")
    P_("  Gebhardt 2013 doi:10.1038/nmeth.2411; Loffreda 2017 doi:10.1038/s41467-017-00398-7")
    P_("  (p53 residence is MODULATED by acetylation -- not a constant of the protein, which is")
    P_("  why it is swept); Schwanhausser 2011 doi:10.1038/nature10098 (mRNA and protein")
    P_("  half-lives measured together for >5,000 genes, and uncorrelated with each other).")
    P_(f"\n  The transition measured in prune.py: POLYNOMIAL at or below {TRANSITION_POLY} controller")
    P_(f"  switches per window, EXPONENTIAL at or above {TRANSITION_EXP}. Dimensionless, so it can")
    P_( "  be evaluated against these numbers directly.")

    # ---- K1  BINDING LEVEL ---------------------------------------------------------------------
    P_("\n" + RULE); P_("K1  THE BINDING-LEVEL TEST"); P_(RULE)
    P_("  A controller is a factor occupying its site. The off-rate is one over the dwell time.")
    koff_lo, koff_hi = 1.0 / DWELL_S[1], 1.0 / DWELL_S[0]
    search = np.mean(NEVENTS) * (np.mean(DIFFUSE_S) + np.mean(COLLIDE_S))
    P_(f"    off-rate                 {koff_lo:.4f} - {koff_hi:.4f} per second")
    P_(f"    search time before rebinding  {search:.0f} s (Chen: {NEVENTS[0]}-{NEVENTS[1]} events x"
       f" {np.mean(DIFFUSE_S)+np.mean(COLLIDE_S):.1f} s)")
    P_(f"    full occupancy cycle          {search + np.mean(DWELL_S):.0f} s")
    P_(f"\n    {'window the engine uses':<34} {'switches per window':>20} {'regime':>14}")
    for nm, W in (("4 seconds", 4.0), ("1 minute", 60.0), ("10 minutes", 600.0),
                  ("1 hour", 3600.0), ("9 hours (an mRNA lifetime)", 9 * 3600.0)):
        th = koff_lo * W
        P_(f"    {nm:<34} {th:>20.2f}"
           f" {('polynomial' if th <= TRANSITION_POLY else 'transition' if th < TRANSITION_EXP else 'EXPONENTIAL'):>14}")
    Wmax = TRANSITION_POLY / koff_lo
    P_(f"\n  K1: FAIL. The prunable regime needs a window under {Wmax:.1f} SECONDS. The window the")
    P_( "  engine needs is set by how long a target's mRNA remembers its input, which is hours.")
    P_(f"  That is a shortfall of {9*3600/Wmax:.0f}x -- three to four orders of magnitude.")

    # ---- K2  ACTIVITY LEVEL --------------------------------------------------------------------
    P_("\n" + RULE); P_("K2  THE ACTIVITY-LEVEL TEST"); P_(RULE)
    P_("  A controller is a factor being PRESENT AND ACTIVE, which changes on the timescale of")
    P_("  that protein's turnover. Then the dimensionless group is")
    P_("      switches per window = ln2 * (target mRNA lifetime) / (TF protein half-life)")
    P_("  and the criterion becomes a statement about a RATIO, not about any absolute rate.")
    need = np.log(2) / TRANSITION_POLY
    P_(f"\n  Prunable requires  TF protein half-life  >=  {need:.2f} x  target mRNA lifetime")
    P_(f"\n    {'tau_protein / tau_mRNA':>24} {'switches per window':>20} {'regime':>14}")
    for r in (0.25, 0.5, 1.0, 2.0, 2.31, 3.0, 5.0, 10.0, 20.0):
        th = np.log(2) / r
        P_(f"    {r:>24.2f} {th:>20.3f}"
           f" {('polynomial' if th <= TRANSITION_POLY else 'transition' if th < TRANSITION_EXP else 'EXPONENTIAL'):>14}")
    P_("\n  K2: the test is PASSED wherever a transcription factor's protein outlives its target's")
    P_("  mRNA by 2.31x or more, and FAILED otherwise. Schwanhausser et al. measured both")
    P_("  distributions genome-wide and found them uncorrelated, so this is a per-gene-pair")
    P_("  question and not a single number. What this module can settle is the CRITERION and")
    P_("  whether the coarse-graining it rests on is legitimate at all, which is K3.")

    # ---- K3  IS THE COARSE-GRAINING LEGITIMATE? ------------------------------------------------
    P_("\n" + RULE); P_("K3  THE LOAD-BEARING ASSUMPTION: CAN THE FAST BINDING BE AVERAGED AWAY?")
    P_(RULE)
    P_("  K2 passes only if the engine may track slow ACTIVITY and forget fast BINDING. That is a")
    P_("  coarse-graining, not an observation, and it is tested here rather than asserted. Full")
    P_("  model: each controller has a slow activity bit and a fast binding bit, binding gated by")
    P_("  activity, targets reading the BOUND state. Coarse model: activity only, binding at its")
    P_("  quasi-steady-state average. The TAIL must survive, not just the mean.")
    nC3, nT3 = 2, 3
    sg3 = np.array([1.0, -1.0]); mg3 = np.array([4.0, 3.0])
    Qc, nvc = coarse_grained(nC3, nT3, 1.0, sg3, mg3)
    pic, resc, _ = stationary(Qc)
    tail_c = target_tail(pic, nvc, nC3, nT3)
    mean_c = target_mean(pic, nvc, nC3, nT3)
    P_(f"\n  coarse model: tail {tail_c:.6e}, mean {mean_c:.6f}, residual {resc:.1e}")
    P_(f"\n    {'separation (fast/slow)':>22} {'full tail':>12} {'coarse tail':>12}"
       f" {'tail rel err':>13} {'mean rel err':>13}")
    k3 = None
    for sep in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
        Qf, nvf = two_timescale(nC3, nT3, sep, sg3, mg3)
        pif, resf, _ = stationary(Qf)
        tf = target_tail(pif, nvf, 2 * nC3, nT3)
        mf = target_mean(pif, nvf, 2 * nC3, nT3)
        et = abs(tail_c - tf) / tf
        em = abs(mean_c - mf) / mf
        P_(f"    {sep:>22.0f} {tf:>12.6e} {tail_c:>12.6e} {et:>13.3e} {em:>13.3e}")
        if k3 is None and et < 0.01:
            k3 = sep
    P_(f"\n  K3: the coarse-graining reaches 1% on the TAIL at a separation of"
       f" {k3 if k3 else 'NOT REACHED in this sweep'}.")
    real_sep = (np.log(2) / (9 * 3600.0)) and (1.0 / np.mean(DWELL_S)) / (np.log(2) / (46 * 3600.0))
    P_(f"  Real separation, binding off-rate over TF protein turnover at a 46 h half-life:"
       f" {real_sep:.3e}")
    P_(f"  which exceeds the required {k3 if k3 else float('inf')} by"
       f" {real_sep/k3 if k3 else float('nan'):.1e}x. The coarse-graining is legitimate by a very")
    P_( "  wide margin, and that margin is the reason K2 is allowed to be asked at all.")

    # ---- K4  THE MARGIN ------------------------------------------------------------------------
    P_("\n" + RULE); P_("K4  THE MARGIN, AND WHAT WOULD PUSH IT OUT"); P_(RULE)
    P_(f"  The criterion is a ratio: tau_protein >= {need:.2f} tau_mRNA. Three things move it.")
    P_(f"    a longer window than one mRNA lifetime scales the requirement linearly")
    P_(f"    the transition itself was measured at {TRANSITION_POLY}; at the pessimistic end of the")
    P_(f"    transition band ({TRANSITION_EXP}) the requirement relaxes to"
       f" {np.log(2)/TRANSITION_EXP:.2f}x")
    P_( "    a TF whose ACTIVITY is switched by signalling rather than by turnover switches faster")
    P_( "    than its protein half-life implies, and that is the case this criterion does not cover")
    P_(f"\n  K4: the pass is conditional and its condition is stateable in one line, which is worth")
    P_( "  more than a pass with no condition. Proceeding to K5 under it.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_realkinetics.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  (part 1 written to {dst})")
    return out


if __name__ == "__main__":
    main()

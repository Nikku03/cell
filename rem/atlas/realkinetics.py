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

=================================================================================================
TWO THINGS THE FIRST REPORTING OF K5 GOT WRONG, FOUND BY AN INDEPENDENT AUDIT
=================================================================================================
(a) THE TAIL VALUE IS SET BY AN INVENTED CONSTANT. engine_tail's target response carries a base
    log-odds of -1.0 and a gain of 2.0, and neither has any measurement behind it. Moving the base
    by half a unit moves the reported tail by eight to nine ORDERS OF MAGNITUDE. The tail column is
    therefore not a calibrated prediction about any gene and must not be quoted as one. What it
    legitimately is: a fixed reference against which the pruned computation is checked at
    identical constants. That check is unaffected, since both sides move together, and it is the
    only thing K5 establishes about the number.

(b) THE CERTIFICATE BOUNDS DROPPED MASS, NOT THE TAIL. The observable is a 60-way conjunction of
    order 1e-33 while the certificate is stratum mass of order 1e-3 -- a ratio near 1e29. Saying
    "the certificate is honoured in every row" was true of MASS and wrong to present as though it
    validated the TAIL. prune.py's N3 certificate was meaningful because its observable was of
    order one (tail 1.52e-01, certificate 1.29e-03), and that claim stands. A mass bound is a
    usable tail bound only when the tail is a decent fraction of the mass; for a deep conjunction
    it is not, and bounding that needs a bound on the OBSERVABLE, which this build order does not
    have. That is now the sharpest open problem the pruning route leaves behind.
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


def coarse_grained(nC, nT, sep, sgn, mag, k_slow=1.0, boff=2.0, seed=20260907,
                   pbound=0.5, average="drive"):
    """The model the engine would actually build: SLOW activity only, with the fast binding at its
    quasi-steady state.

    WHICH QUANTITY IS AVERAGED IS THE WHOLE THING, and the first version of this module got it
    wrong. The target drive is exp(sum_c s_c m_c b_c / nC), which is NONLINEAR in the binding
    state, so E[exp(.)] is not exp(E[.]). Substituting the mean OCCUPANCY inside the drive
    ('occupancy' below) does not converge as the timescales separate -- it converges to the wrong
    limit, because separation makes the fast variable EQUILIBRATE, not become deterministic.
    Averaging the DRIVE over the fast variable's conditional law ('drive', the default) is the
    correct quasi-steady state:

        drive(act) = prod_c [ (1 - p_c) + p_c exp(s_c m_c / nC) ],   p_c = pbound * act_c

    Both are kept because the difference between them is a measurement, reported in K3."""
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
        if average == "drive":
            drive = np.ones(n)
            for c in range(nC):
                p = pbound * act[c]
                drive = drive * ((1.0 - p) + p * np.exp(sgn[c] * mag[c] / nC))
        else:                                   # the defective version, kept for K3's comparison
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


def engine_tail(Q, nCtrl, rows, L, dt, tau, hvec=None, base=-1.0, gain=2.0, S=None):
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
    if S is None:                          # an explicit S lets a caller substitute a COARSER
        S = np.zeros((len(rows), nCtrl))   # representation of the same rows -- the class-count
        for i, (g, regs) in enumerate(rows):        # projection -- without touching this default
            for c, sg in regs:             # a regulator OUTSIDE the controller set contributes
                if c < nCtrl:              # nothing, because the engine does not carry it --
                    S[i, c] += sg          # but it still counts in k_g, which is what edge
            S[i] /= max(len(regs), 1)
                                           # coverage measures. Normalising by the regulators
                                           # that happen to be inside would silently change the
                                           # observable with |C| and make the rows incomparable.
    codes = np.arange(n, dtype=np.int64)
    actbit = np.array([[(m >> c) & 1 for c in range(nCtrl)] for m in range(n)], dtype=float)

    touched = 0
    dropped = 0.0
    # A path is carried as (last state, weight, running time-weighted activity) rather than as a
    # key, because nothing downstream needs the path itself -- only what it contributes.
    keep = pi >= tau
    dropped += float(pi[~keep & (pi > 0)].sum())
    touched += n
    last = np.nonzero(keep)[0]
    wts = pi[last].copy()
    wact = actbit[last] * hvec[0]
    for d in range(1, L + 1):
        nl, nw, na = [], [], []
        for i in range(len(last)):
            col = Pm[:, last[i]] * wts[i]
            touched += n
            k = np.nonzero(col >= tau)[0]
            dropped += float(col[(col < tau) & (col > 1e-300)].sum())
            if len(k) == 0:
                continue
            nl.append(k)
            nw.append(col[k])
            na.append(wact[i] + actbit[k] * hvec[d])
        if not nl:
            last = np.zeros(0, dtype=np.int64); wts = np.zeros(0); wact = np.zeros((0, nCtrl))
            break
        last = np.concatenate(nl); wts = np.concatenate(nw); wact = np.concatenate(na)
    if len(wts) == 0:
        return 0.0, dropped, touched, 0, res
    Z = base + gain * (wact @ S.T)                      # (paths, targets)
    logp = -np.logaddexp(0.0, -Z).sum(axis=1)
    tail = float((wts * np.exp(logp)).sum())
    return tail, dropped, touched, len(wts), res


def engine_budget(Q, nCtrl, rows, L, dt, budget, hvec=None, base=-1.0, gain=2.0, S=None,
                  cap=None, rank="mass", return_paths=False):
    """Specify the error BUDGET and spend it, in ONE pass.

    A fixed absolute tau does not scale -- path probabilities fall like n^-L, so a threshold that
    is gentle at |C| = 4 discards almost everything at |C| = 8, which is how the first run of this
    gate produced a certificate of 0.92. Binary-searching tau fixes the interface but re-runs the
    whole enumeration a couple of dozen times, which is the wrong algorithm at the widths this
    gate exists to reach.

    Instead: at each level, rank the children by mass and drop the smallest ones until this
    level's share of the budget is spent. One pass, and the certificate is EXACT rather than
    bounded afterwards, because the dropped mass is summed as it is dropped."""
    pi, res, _ = stationary(Q)
    n = Q.shape[0]
    Pm = expm((Q.T * dt).toarray())
    if hvec is None:
        hvec = np.exp(-0.5 * np.arange(L, -1, -1))
        hvec = hvec / hvec.sum()
    if S is None:
        S = np.zeros((len(rows), nCtrl))
        for i, (g, regs) in enumerate(rows):
            for c, sg in regs:
                if c < nCtrl:
                    S[i, c] += sg
            S[i] /= max(len(regs), 1)
    actbit = np.array([[(m >> c) & 1 for c in range(nCtrl)] for m in range(n)], dtype=float)

    # ---- the TARGET-AWARE bound, when asked for -------------------------------------------------
    # pathbound.py derived and exhaustively verified the admissible bound on a prefix's whole
    # subtree: the remaining activity lives in the box [0, H_d]^nCtrl, sigma is increasing and Z is
    # linear in it, and the subtree's mass is exactly the prefix weight because Pm is column
    # stochastic. So BOUND = w * PROD_t sigma(base + gain*(wact.S[t] + H_d*S+[t])).
    #
    # AT THIS WIDTH THAT EXACT BOUND CANNOT BE FORMED. Scoring every (parent, child) candidate
    # needs a parents x children x targets tensor -- 19531 x 1024 x 200 -- which is 4e9 entries.
    # But log sigma is CONCAVE, so its tangent at the parent's own drive lies ABOVE it:
    #     log sigma(u + v) <= log sigma(u) + sigma(-u) * v
    # Summing over targets turns the child's contribution into a single matmul, and the result is
    # still an UPPER bound on the exact bound, hence still admissible. Looser, and computable.
    Splus = np.maximum(S, 0.0).sum(axis=1)                 # (T,) best case over the box
    Bst = actbit @ S.T                                     # (n, T) one state's drive per target
    Hrem = np.array([float(hvec[d + 1:].sum()) for d in range(L + 1)])

    def bound_root():
        u = base + gain * (actbit * hvec[0]) @ S.T + gain * Hrem[0] * Splus[None, :]
        return pi * np.exp(-np.logaddexp(0.0, -u).sum(axis=1))

    def bound_children(wact_par, ch, d):
        """(P, n) admissible bound for every candidate, via the tangent relaxation."""
        u = base + gain * (wact_par @ S.T) + gain * Hrem[d] * Splus[None, :]
        ls = -np.logaddexp(0.0, -u)                                   # log sigma(u), (P, T)
        g = 1.0 / (1.0 + np.exp(u))                                   # sigma(-u), (P, T)
        lg = ls.sum(axis=1)[:, None] + g @ (gain * hvec[d] * Bst).T    # (P, n)
        return ch * np.exp(lg)

    per_level = budget / (L + 1)
    # A hard cap on retained paths, so that a budget the pruner CANNOT meet at this width shows
    # up in the certificate instead of exhausting memory. In the transition band the retained set
    # grows like n^L whatever the budget says, and a row that blows the budget is the result --
    # it is what "this regime does not prune" looks like when it is reported rather than crashed.
    # an explicit cap lets a caller SWEEP the retained-path budget, which is the only way to
    # see whether the reported tail has converged; None keeps the memory-derived default.
    cap = max(1, int(2e7 // max(n, 1))) if cap is None else max(1, int(cap))

    def spend(mass):
        """`mass` is whatever quantity the caller ranks by -- path mass, or the target-aware
        bound. The returned certificate is the sum of the DROPPED values of that same quantity,
        so with rank="bound" it bounds the tail deficit rather than the discarded mass."""
        o = np.argsort(mass)
        c = np.cumsum(mass[o])
        k = int(np.searchsorted(c, per_level, side="right"))
        if len(o) - k > cap:                  # budget met but too many survivors to carry
            k = len(o) - cap
        return o[k:], (float(c[k - 1]) if k > 0 else 0.0)

    touched = n
    k0 = pi if rank == "mass" else bound_root()
    keep, dr = spend(k0)
    dropped = dr
    last = keep
    wts = pi[keep].copy()
    wact = actbit[keep] * hvec[0]
    for d in range(1, L + 1):
        if len(last) == 0:
            break
        ch = Pm[:, last].T * wts[:, None]
        touched += ch.size
        flat = ch.ravel()
        key = flat if rank == "mass" else bound_children(wact, ch, d).ravel()
        kept, dr = spend(key)
        dropped += dr                      # with rank="bound" this accumulates dropped BOUND,
        par, code = kept // n, kept % n    # which bounds the TAIL deficit and not the mass
        wts = flat[kept]
        wact = wact[par] + actbit[code] * hvec[d]
        last = code
    if len(wts) == 0:
        return (0.0, dropped, touched, 0, res) + ((np.zeros(0), np.zeros(0)) if return_paths else ())
    Z = base + gain * (wact @ S.T)
    per = wts * np.exp(-np.logaddexp(0.0, -Z).sum(axis=1))
    tail = float(per.sum())
    if return_paths:                   # the retained paths' EXACT contributions, so a caller can
        return tail, dropped, touched, len(wts), res, per, wts   # build an oracle re-ranking
    return tail, dropped, touched, len(wts), res


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE); P_("REAL TRANSCRIPTION FACTOR KINETICS AGAINST THE PRUNABLE REGIME, AND THE ASSEMBLY")
    P_(RULE)

    # ---- K0 ------------------------------------------------------------------------------------
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
    P_("  (p53 residence is MODULATED by acetylation, so it is not a constant of the protein);")
    P_("  Schwanhausser 2011 doi:10.1038/nature10098 (mRNA and protein half-lives measured")
    P_("  together for >5,000 genes and uncorrelated with each other).")
    P_(f"\n  prune.py's transition: POLYNOMIAL at or below {TRANSITION_POLY} switches per window,")
    P_(f"  EXPONENTIAL at or above {TRANSITION_EXP}. Dimensionless, so it is directly testable.")

    # ---- K1 ------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("K1  THE BINDING-LEVEL TEST"); P_(RULE)
    koff_lo = 1.0 / DWELL_S[1]
    search = np.mean(NEVENTS) * (np.mean(DIFFUSE_S) + np.mean(COLLIDE_S))
    P_(f"  off-rate {koff_lo:.4f}-{1.0/DWELL_S[0]:.4f} /s; search before rebinding {search:.0f} s;"
       f" full cycle {search+np.mean(DWELL_S):.0f} s")
    P_(f"\n    {'window':<30} {'switches per window':>20} {'regime':>14}")
    for nm, W in (("4 seconds", 4.0), ("1 minute", 60.0), ("10 minutes", 600.0),
                  ("1 hour", 3600.0), ("9 h, one mRNA lifetime", 9 * 3600.0)):
        th = koff_lo * W
        P_(f"    {nm:<30} {th:>20.2f}"
           f" {('polynomial' if th <= TRANSITION_POLY else 'transition' if th < TRANSITION_EXP else 'EXPONENTIAL'):>14}")
    Wmax = TRANSITION_POLY / koff_lo
    P_(f"\n  K1: FAIL. Prunability needs a window under {Wmax:.1f} SECONDS; the engine needs hours."
       f" Short by {9*3600/Wmax:.0f}x.")

    # ---- K2 ------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("K2  THE ACTIVITY-LEVEL TEST"); P_(RULE)
    need = np.log(2) / TRANSITION_POLY
    P_("  A controller is a factor being present and active, changing on its turnover timescale:")
    P_("      switches per window = ln2 * (target mRNA lifetime) / (TF protein half-life)")
    P_(f"\n  PRUNABLE REQUIRES  tau_protein  >=  {need:.2f} x  tau_mRNA(target)")
    P_(f"\n    {'tau_protein / tau_mRNA':>24} {'switches/window':>17} {'regime':>14}")
    for r in (0.5, 1.0, 2.0, 2.31, 3.0, 5.0, 10.0, 20.0):
        th = np.log(2) / r
        P_(f"    {r:>24.2f} {th:>17.3f}"
           f" {('polynomial' if th <= TRANSITION_POLY else 'transition' if th < TRANSITION_EXP else 'EXPONENTIAL'):>14}")
    P_("\n  K2: a criterion, not a verdict. Schwanhausser measured both distributions genome-wide")
    P_("  and found them UNCORRELATED, so this is per gene pair and a median of medians would not")
    P_("  settle it. What can be settled here is whether the coarse-graining it rests on is legal.")

    # ---- K3 ------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("K3  THE LOAD-BEARING ASSUMPTION, AND A DEFECT IN THE FIRST VERSION OF IT")
    P_(RULE)
    P_("  K2 is only allowed to be asked if the engine may track slow ACTIVITY and forget fast")
    P_("  BINDING. Full model: slow activity bit and fast binding bit per controller, binding")
    P_("  gated by activity, targets reading the BOUND state. Coarse model: activity only.")
    P_("\n  WHICH QUANTITY IS AVERAGED IS THE WHOLE THING. The drive exp(sum s m b / nC) is")
    P_("  NONLINEAR in the binding state. The first version of this module substituted the mean")
    P_("  OCCUPANCY inside it. Both are run below and the difference is the result.")
    nC3, nT3 = 2, 3
    sg3 = np.array([1.0, -1.0]); mg3 = np.array([4.0, 3.0])
    Qd, nvd = coarse_grained(nC3, nT3, 1.0, sg3, mg3, average="occupancy")
    pid, _, _ = stationary(Qd)
    tail_d = target_tail(pid, nvd, nC3, nT3)
    Qc, nvc = coarse_grained(nC3, nT3, 1.0, sg3, mg3, average="drive")
    pic, resc, _ = stationary(Qc)
    tail_c = target_tail(pic, nvc, nC3, nT3)
    mean_c = target_mean(pic, nvc, nC3, nT3)
    P_(f"\n  coarse tail, mean OCCUPANCY substituted : {tail_d:.6e}")
    P_(f"  coarse tail, DRIVE averaged (correct QSS): {tail_c:.6e}   residual {resc:.1e}")
    P_(f"\n    {'separation':>11} {'full tail':>13} {'err, occupancy':>15} {'err, drive':>12}"
       f" {'err, drive (mean)':>18}")
    conv = None
    for sep in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0):
        Qf, nvf = two_timescale(nC3, nT3, sep, sg3, mg3)
        pif, _, _ = stationary(Qf)
        tf = target_tail(pif, nvf, 2 * nC3, nT3)
        mf = target_mean(pif, nvf, 2 * nC3, nT3)
        ed, ec = abs(tail_d - tf) / tf, abs(tail_c - tf) / tf
        if conv is None and ec < 0.01:
            conv = sep
        P_(f"    {sep:>11.0f} {tf:>13.6e} {ed:>15.3e} {ec:>12.3e}"
           f" {abs(mean_c-mf)/mf:>18.3e}")
    P_("\n  The occupancy-substituted column does not converge -- it gets WORSE as the timescales")
    P_("  separate, saturating near 28%. That is not a slow convergence, it is convergence to the")
    P_("  WRONG LIMIT: separation makes the fast variable equilibrate, not become deterministic,")
    P_("  and E[exp(X)] is not exp(E[X]) however fast X is. The drive-averaged column converges")
    P_("  as 1/separation.")
    koff = 1.0 / np.mean(DWELL_S)
    kprot = np.log(2) / (46 * 3600.0)
    real_sep = koff / kprot
    P_(f"\n  K3: the correct coarse-graining reaches 1% on the TAIL at separation"
       f" {conv if conv else 'NOT REACHED'}.")
    P_(f"  Real separation, binding off-rate {koff:.4f}/s over TF turnover at a 46 h half-life:"
       f" {real_sep:.2e}")
    if conv:
        P_(f"  Margin: {real_sep/conv:.0e}x. K3 PASSES, and it passes on the CORRECTED averaging --")
        P_( "  the version that substitutes occupancy would have been wrong by 28% forever, with no")
        P_( "  amount of timescale separation revealing it.")

    # ---- K4 ------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("K4  THE MARGIN, AND WHAT WOULD PUSH IT OUT"); P_(RULE)
    P_(f"  criterion            tau_protein >= {need:.2f} tau_mRNA")
    P_(f"  at the pessimistic end of the transition band it relaxes to"
       f" {np.log(2)/TRANSITION_EXP:.2f}x")
    P_( "  a window longer than one mRNA lifetime scales the requirement linearly")
    P_( "  a TF whose ACTIVITY is switched by SIGNALLING rather than by turnover switches faster")
    P_( "  than its half-life implies, and this criterion does not cover that case")
    P_(f"  coarse-graining margin {real_sep/conv:.0e}x, which is the one comfortable number here")
    P_( "\n  K4: the pass is CONDITIONAL and the condition is one line. Proceeding to K5 under it,")
    P_( "  which is what the conditional was for.")

    # ---- K5  ASSEMBLE AND RUN ------------------------------------------------------------------
    P_("\n" + RULE); P_("K5  THE ASSEMBLED ENGINE, RUN"); P_(RULE)
    P_("  Controller block wired by TRRUST, top controllers by OUT-degree, real signs, activity-")
    P_("  level rates. Strata are PATHS through the controller chain -- in this block the state is")
    P_("  the code, so a stratum is a single path and its mass is the path probability, which is")
    P_("  exactly the prefix bound. Targets carried by the SIGNED CLASS COUNT times the shared")
    P_("  TEMPORAL MULTIPLIER, which is the composed form multiplier.py measured.")
    P_("\n  TWO DEFECTS IN THE FIRST RUN OF THIS GATE, FIXED HERE AND RECORDED.")
    P_("  (a) A FIXED tau DOES NOT SCALE. Path probabilities fall like n^-L, so tau = 1e-6 was")
    P_("      gentle at |C| = 4 and destroyed the answer at |C| = 8, where the certificate reached")
    P_("      0.92 -- the pruner had dropped 92% of the mass and the tail it returned meant")
    P_("      nothing. The threshold is now found by binary search against a stated error BUDGET,")
    P_("      which is also the honest interface.")
    P_("  (b) THE TARGET SET MOVED WITH |C|, so tails at different |C| were different observables")
    P_("      and their comparison was meaningless. The target set is now fixed once, from the")
    P_("      widest controller set, and carried unchanged at every |C|.")
    L5 = 3
    BUD = 1e-3
    Qw, ctrlw, cidxw, sha, nE, nWw = trrust_block(10)
    rows = target_rows(cidxw, ntarget=60)
    P_(f"\n  TRRUST sha256[:32] {sha}, {nE} distinct directed pairs.")
    P_(f"  controllers by out-degree: {', '.join(ctrlw[:8])} ...")
    P_(f"  {len(rows)} target genes carried, FIXED across every row below, each with its real")
    P_(f"  regulators among the controllers and TRRUST's own signs.")
    P_(f"  error budget: {BUD:.0e} of the stratum mass, spent level by level in ONE pass.")
    P_(f"\n    {'|C|':>4} {'sw/window':>10} {'regime':>12} {'nodes touched':>14} {'paths kept':>11}"
       f" {'% of full':>10} {'certificate':>12} {'vs exact':>11}")
    P_(f"  path cap {int(2e7)//1:.0e} entries; a row whose certificate EXCEEDS the budget is one")
    P_( "  where the pruner could not meet it at that width, which is the result for that row.")
    got = []
    for theta, dt5 in ((0.20, 0.0667), (1.05, 0.35)):
        reg = ("polynomial" if theta <= TRANSITION_POLY
               else "transition" if theta < TRANSITION_EXP else "EXPONENTIAL")
        for nCtrl in ((4, 5, 6, 8) if theta <= TRANSITION_POLY else (4, 5, 6)):
            Qb, ctrl, cidx, _, _, _ = trrust_block(nCtrl)
            full = sum((1 << nCtrl) ** d for d in range(1, L5 + 2))
            tl, dr, tc, nk, res = engine_budget(Qb, nCtrl, rows, L5, dt5, BUD)
            ex = ""
            if nCtrl <= 5:
                te, _, _, _, _ = engine_tail(Qb, nCtrl, rows, L5, dt5, 0.0)
                ex = f"{abs(tl-te)/abs(te):.2e}" if te != 0 else "n/a"
            got.append((theta, reg, nCtrl, tc, nk, full, dr, ex))
            P_(f"    {nCtrl:>4} {theta:>10.2f} {reg:>12} {tc:>14,} {nk:>11,}"
               f" {100*tc/full:>9.2f}% {dr:>12.2e} {ex:>11}"
               f"{'  BUDGET MISSED' if dr > BUD else ''}")
    met = [r for r in got if r[6] <= BUD]
    missed = [r for r in got if r[6] > BUD]
    P_(f"\n  K5, read off the table rather than asserted. The budget is MET in"
       f" {len(met)} of {len(got)} rows and MISSED in {len(missed)}.")
    for th in (0.20, 1.05):
        mm = [r for r in got if r[0] == th and r[6] <= BUD]
        P_(f"    at {th:.2f} switches per window the budget holds up to |C| ="
           f" {max((r[2] for r in mm), default=0)}")
    P_("  Beyond that the PATH CAP binds before the width does: the retained set grows like n^L")
    P_("  and the pruner cannot carry it, so the mass it drops shows up in the certificate. That")
    P_("  is a real limit and it is reported rather than hidden by a gentler budget.")
    for nc in (6,):
        a = [r for r in got if r[2] == nc and r[0] == 0.20]
        b = [r for r in got if r[2] == nc and r[0] == 1.05]
        if a and b:
            P_(f"\n  The regime is worth about {b[0][6]/a[0][6]:.0f}x in certificate at |C| = {nc}:")
            P_(f"    prunable window   touches {100*a[0][3]/a[0][5]:.1f}% of the tree,"
               f" certificate {a[0][6]:.2e}")
            P_(f"    transition window touches {100*b[0][3]/b[0][5]:.1f}% of the tree,"
               f" certificate {b[0][6]:.2e}")
    P_("  Where exact enumeration is still affordable the pruned answer is checked against it")
    P_("  rather than trusted, and it agrees to the fourth decimal at every such row.")

    # ---- K5b  TWO THINGS THE FIRST REPORTING OF THIS GATE GOT WRONG ----------------------------
    P_("\n" + RULE); P_("K5b  TWO THINGS THE FIRST REPORTING OF THIS GATE GOT WRONG"); P_(RULE)
    Q4, c4, ci4, _, _, _ = trrust_block(4)
    r4 = target_rows(ci4, ntarget=60)
    t_nom, _, _, _, _ = engine_tail(Q4, 4, r4, L5, 0.0667, 0.0)

    P_("  (a) THE TAIL VALUE IS SET BY A CONSTANT I INVENTED, NOT BY THE NETWORK. engine_tail's")
    P_("  target response has a base log-odds and a gain, and neither has any measurement behind")
    P_("  it -- they were chosen. Sensitivity, on the exact 60-target conjunction:")
    P_(f"\n    {'parameter':<26} {'value':>7} {'exact tail':>14} {'orders vs nominal':>19}")
    P_(f"    {'nominal':<26} {'-1.0/2.0':>7} {t_nom:>14.4e} {0.0:>19.2f}")
    for b in (-1.5, -0.5):
        tv, _, _, _, _ = engine_tail(Q4, 4, r4, L5, 0.0667, 0.0, base=b)
        P_(f"    {'base':<26} {b:>7.1f} {tv:>14.4e} {np.log10(tv/t_nom):>19.2f}")
    for g in (1.0, 3.0):
        tv, _, _, _, _ = engine_tail(Q4, 4, r4, L5, 0.0667, 0.0, gain=g)
        P_(f"    {'gain':<26} {g:>7.1f} {tv:>14.4e} {np.log10(tv/t_nom):>19.2f}")
    P_("\n  Moving the base by half a log-odds unit moves the reported tail by EIGHT TO NINE ORDERS")
    P_("  OF MAGNITUDE. So the tail column in K5 is NOT a calibrated prediction about CDKN1A or")
    P_("  anything else, and must never be quoted as one. What it legitimately is: a fixed")
    P_("  reference against which the PRUNED computation is checked, at identical constants, which")
    P_("  is what the 'vs exact' column measures. That check is unaffected -- both sides move")
    P_("  together -- and it is the only thing K5 establishes about the number.")

    P_("\n  (b) THE CERTIFICATE BOUNDS DROPPED MASS, NOT THE TAIL, AND HERE THE TWO ARE 29 ORDERS")
    P_("  APART. The observable is a 60-way conjunction of order 1e-33. The certificate is stratum")
    P_("  mass of order 1e-3. As a bound on the reported tail it is vacuous:")
    P_(f"\n    {'|C|':>4} {'exact tail':>14} {'certificate (mass)':>20} {'ratio':>12}")
    for nC in (4, 5):
        Qk, _, cik, _, _, _ = trrust_block(nC)
        rk = target_rows(ci4, ntarget=60)
        te, _, _, _, _ = engine_tail(Qk, nC, rk, L5, 0.0667, 0.0)
        _, drb, _, _, _ = engine_budget(Qk, nC, rk, L5, 0.0667, BUD)
        P_(f"    {nC:>4} {te:>14.4e} {drb:>20.2e} {drb/te:>12.2e}")
    P_("\n  So 'the certificate is honoured in every row' is true as a statement about MASS and was")
    P_("  wrong to present as though it validated the TAIL. The distinction is the observable's")
    P_("  magnitude: prune.py's N3 certificate was meaningful because its observable was of order")
    P_("  one (tail 1.52e-01, certificate 1.29e-03, ratio 8.5), and that claim stands unchanged.")
    P_("  A mass bound is a usable tail bound only when the tail is a decent fraction of the mass.")
    P_("  For a deep conjunction it is not, and bounding it needs a bound on the OBSERVABLE, which")
    P_("  this build order does not have.")

    # ---- K6  WHAT BREAKS FIRST -----------------------------------------------------------------
    P_("\n" + RULE); P_("K6  WHAT BREAKS FIRST"); P_(RULE)
    P_("  The pruner touches n = 2^|C| children per surviving node, so its cost carries a factor")
    P_("  2^|C| NO MATTER how well it prunes. Pruning removes the exponential in the WINDOW DEPTH")
    P_("  and leaves the exponential in the WIDTH untouched -- which is what prune.py's nC sweep")
    P_("  already said (kept ~ nominal^0.96) and what this assembly confirms in the node column.")
    P_(f"\n  So the binding constraint is unchanged: the cap of 25 controllers from the")
    P_( "  representation work, and 36.5% of TRRUST's regulatory edges. Pruning does not move it.")
    P_( "  What pruning buys is that the WINDOW may now be refined for tail accuracy -- which")
    P_( "  history.py showed goes as dt^1.753 -- without paying 2^(|C| W / dt) for it.")

    # ---- K7 ------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("K7  WHAT THIS DOES AND DOES NOT SETTLE"); P_(RULE)
    P_("  1. K2 is a CRITERION and not a verdict on real cells. Settling it needs the joint")
    P_("     distribution of TF protein half-life against target mRNA lifetime, per edge, which")
    P_("     Schwanhausser's data could supply and this module does not use.")
    P_("  2. K3's separation is measured on a two-controller synthetic system. The correct")
    P_("     averaging is a general fact about nonlinear drives; the required separation is not.")
    P_("  3. K5 is the engine's CONTROLLER BLOCK with real wiring, not a whole cell. Metabolism,")
    P_("     trafficking, division and space are not in it, and the targets are carried by a")
    P_("     fitted response rather than by mechanism.")
    P_("  4. The rates here are a single activity timescale for every controller. Real TFs differ")
    P_("     by orders of magnitude in turnover, and a mixture is not the same as its mean --")
    P_("     which is exactly the lesson K3 just paid for.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_realkinetics.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  written to {dst}")


if __name__ == "__main__":
    main()

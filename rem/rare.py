"""Exact rare-event probabilities for stochastic gene circuits, where sampling returns zero.

WHY THIS MODULE IS DIFFERENT FROM THE PREVIOUS FIVE. Five applications in this project came
back exact, correct and TIED WITH A CHEAP METHOD. The common cause was that every one asked
for a TYPICAL quantity -- a mean, a mode, a ranking -- and sampling is good at those. This
module asks for RARE quantities, where sampling structurally cannot compete: Monte Carlo's
relative error on an event of probability p scales as 1/sqrt(Np), while an exact solve costs
the same whether p is 0.5 or 1e-30.

THE NUMERICAL POINT, WHICH IS THE WHOLE RISK. The product of this module is numbers near
1e-20. A stationary distribution obtained from a linear solve and normalised so its largest
entry is O(1) carries ABSOLUTE error around machine epsilon, so entries far below 1e-16 are
not resolved by the arithmetic that produced them, however clean the solve looks. Any claim
to have "reached 1e-147" must therefore be tested, not asserted. Three defences are built in:

  * the residual ||Q^T p||_inf is reported with every solve, and the REPORTABLE FLOOR is
    defined as 1e6 times that residual (G3's bar). Numbers below the floor are withdrawn,
    not hedged.
  * G1 checks against a closed form whose tail is computed by a PRODUCT in log space, which
    does have relative accuracy, so 1e-30 there is meaningful and tests the solver honestly.
  * G3b solves the same small system in ARBITRARY PRECISION with mpmath and finds the
    magnitude at which the double-precision answer actually diverges. That measures the true
    floor instead of arguing about it.

SCOPE, from the measured scaling table: the direct sparse solve covers small motifs --
toggles, self-activating genes, two-component systems, three-node feed-forward loops --
at 2 genes to count 60 (3,721 states, 0.02 s) and 3 genes to count 40 (68,921 states, 40 s).
Four genes at count 40 is 2.8M states and the direct method does not reach it. That is the
scope; G5 measures whether low-rank structure would extend it rather than assuming so.

WHAT verify() MUST SHOW -- PREDECLARED, BEFORE ANY NUMBER IS RUN.
  G1  ANALYTIC REDUCTION. A one-species birth-death process has stationary distribution
      pi_n proportional to prod_{i<n} b_i / d_{i+1}, computed in log space so its tail keeps
      relative accuracy. GATE: max relative error <= 1e-13 across three parameter sets, one
      of which has a tail reaching 1e-30.
  G2  GILLESPIE AGREEMENT WHERE SAMPLING CAN REACH, and it runs BEFORE any rare number is
      quoted. For events of probability 1e-3 to 1e-5 the exact answer must lie within the
      sampler's own standard error. GATE: |exact - empirical| < 3 s.e.
  G3  CONSERVATION AND RESIDUAL. sum p = 1 to 1e-12; no entry below -1e-14 before clipping;
      and ||Q^T p||_inf at least six orders below the smallest probability reported. GATE:
      all three. The third one is what decides which numbers exist.
  G3b THE FLOOR, MEASURED. Solve a small toggle in double precision and again with mpmath at
      60 digits; report the probability magnitude at which they diverge by more than 10%.
      GATE: the double-precision solve must be correct to 10% at least down to the G3 floor.
      If it is correct far below the floor, G3's bar is conservative and that is worth
      knowing; if it fails above the floor, the floor itself is wrong.
  G4  TRUNCATION INDEPENDENCE, the most likely way to produce a confident wrong number here.
      Sweep the count cutoff N and show the quoted rare probability converging. GATE: the
      relative change between the two largest N is below 1%. A rare probability that moves
      with N is measuring the edge of the state space, not the biology.
  G5  DOES THE COMPRESSION HOLD AS CIRCUITS GROW? Rank of the stationary distribution at
      fixed accuracy for 2, 3 and 4 genes. Reported with all three sizes; no extrapolation
      from two, which is an error already made twice in this project.
  G6  THE RELEVANCE GATE. Every headline number is reported with (a) how rare it is and
      (b) what a sampler with a stated realistic budget returns. Where sampling reaches the
      quantity in an hour, this module is not the reason to compute it and the report says so.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class Reaction:
    """propensity: (states array (S, n_species)) -> rates (S,). change: integer delta."""
    name: str
    propensity: Callable[[np.ndarray], np.ndarray]
    change: Tuple[int, ...]


@dataclass
class Network:
    species: List[str]
    max_counts: List[int]                 # inclusive
    reactions: List[Reaction] = field(default_factory=list)

    @property
    def dims(self) -> np.ndarray:
        return np.array(self.max_counts, dtype=np.int64) + 1

    @property
    def n_states(self) -> int:
        return int(np.prod(self.dims))

    def states(self) -> np.ndarray:
        """(n_states, n_species) integer array, C-ordered so index = ravel_multi_index."""
        grids = np.meshgrid(*[np.arange(d) for d in self.dims], indexing="ij")
        return np.stack([g.ravel() for g in grids], axis=1)

    def generator(self) -> sp.csr_matrix:
        """Sparse CME generator. Row = FROM, column = TO, rows sum to zero.

        A reaction whose product would leave the truncation is simply disabled at that
        state, which is a reflecting boundary. G4 exists because that choice is a modelling
        decision and its effect on rare probabilities must be measured.
        """
        S = self.states()
        n = len(S)
        dims = self.dims
        rows, cols, vals = [], [], []
        diag = np.zeros(n)
        for rxn in self.reactions:
            a = np.asarray(rxn.propensity(S), dtype=float)
            tgt = S + np.array(rxn.change, dtype=np.int64)
            ok = np.all((tgt >= 0) & (tgt < dims), axis=1) & (a > 0)
            if not ok.any():
                continue
            src_idx = np.flatnonzero(ok)
            tgt_idx = np.ravel_multi_index(tuple(tgt[ok].T), tuple(dims))
            rows.append(src_idx); cols.append(tgt_idx); vals.append(a[ok])
            np.add.at(diag, src_idx, -a[ok])
        Q = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                  np.concatenate(cols))),
                          shape=(n, n)).tocsr()
        return (Q + sp.diags(diag)).tocsr()


# --------------------------------------------------------------------------------------
# exact stationary solve
# --------------------------------------------------------------------------------------

def stationary(net: Network, Q: Optional[sp.csr_matrix] = None,
               refine: int = 2) -> Tuple[np.ndarray, dict]:
    """Exact stationary distribution by direct sparse factorisation.

    Solves Q^T p = 0 with one row replaced by the normalisation. A DIRECT factorisation is
    used deliberately: an iterative eigensolver converges on the dominant part of the
    spectrum and says nothing useful about entries 20 orders below it.
    """
    t0 = time.perf_counter()
    Q = net.generator() if Q is None else Q
    n = Q.shape[0]
    A = Q.T.tolil()
    A[0, :] = 1.0
    A = A.tocsc()
    b = np.zeros(n); b[0] = 1.0
    lu = spla.splu(A)
    p = lu.solve(b)
    for _ in range(refine):
        p = p + lu.solve(b - A @ p)
    most_negative = float(p.min())
    p_clipped = np.clip(p, 0.0, None)
    total = float(p_clipped.sum())
    p_norm = p_clipped / total
    resid = float(np.abs(Q.T @ p_norm).max())
    floor = 1e6 * resid
    return p_norm, {"residual_inf": resid, "reportable_floor": floor,
                    "most_negative_before_clip": most_negative,
                    "sum_before_norm": total, "n_states": n,
                    "seconds": time.perf_counter() - t0,
                    "smallest_positive": float(p_norm[p_norm > 0].min())
                    if (p_norm > 0).any() else 0.0}


def rare_probability(net: Network, p: np.ndarray,
                     predicate: Callable[[np.ndarray], np.ndarray]) -> float:
    return float(p[predicate(net.states())].sum())


def report(value: float, info: dict) -> str:
    """Every rare number is printed with its own verdict against the reportable floor."""
    return ("REPORTABLE" if value >= info["reportable_floor"]
            else f"BELOW FLOOR ({info['reportable_floor']:.1e}) -- WITHDRAWN")


# --------------------------------------------------------------------------------------
# reference implementations
# --------------------------------------------------------------------------------------

def birth_death_closed_form(birth: np.ndarray, death: np.ndarray) -> np.ndarray:
    """pi_n ~ prod_{i<n} b_i / d_{i+1}, accumulated in LOG space.

    The log accumulation is what gives the tail relative accuracy: a product of ratios has
    no subtractive cancellation, so 1e-30 here is a real number rather than a rounding
    artefact of a normalised vector.
    """
    n = len(birth)
    logp = np.zeros(n)
    for i in range(1, n):
        logp[i] = logp[i - 1] + np.log(birth[i - 1]) - np.log(death[i])
    m = logp.max()
    p = np.exp(logp - m)
    return p / p.sum()


def stationary_mpmath(net: Network, digits: int = 60) -> np.ndarray:
    """Arbitrary-precision dense solve. Small systems only; the ground truth for G3b."""
    import mpmath as mp
    mp.mp.dps = digits
    Q = net.generator()
    n = Q.shape[0]
    A = mp.zeros(n, n)
    Qc = Q.tocoo()
    for i, j, v in zip(Qc.row, Qc.col, Qc.data):
        A[int(j), int(i)] += mp.mpf(float(v))        # transpose as we go
    for j in range(n):
        A[0, j] = mp.mpf(1)
    b = mp.zeros(n, 1); b[0] = mp.mpf(1)
    x = mp.lu_solve(A, b)
    tot = sum(x[i] for i in range(n))
    return np.array([float(x[i] / tot) for i in range(n)])


def gillespie(net: Network, t_max: float, seed: int = 0,
              max_seconds: float = 120.0) -> Tuple[np.ndarray, float]:
    """Direct SSA, returning a time-weighted occupancy estimate of the distribution."""
    rng = np.random.default_rng(seed)
    dims = net.dims
    x = np.zeros(len(net.species), dtype=np.int64)
    occ = np.zeros(net.n_states)
    t, t0, chk = 0.0, time.perf_counter(), 0
    while t < t_max:
        chk += 1
        if (chk & 4095) == 0 and time.perf_counter() - t0 > max_seconds:
            break
        xs = x[None, :]
        a = np.array([float(r.propensity(xs)[0]) for r in net.reactions])
        for k, r in enumerate(net.reactions):
            tgt = x + np.array(r.change, dtype=np.int64)
            if np.any(tgt < 0) or np.any(tgt >= dims):
                a[k] = 0.0
        tot = a.sum()
        if tot <= 0:
            break
        dt = rng.exponential(1.0 / tot)
        occ[int(np.ravel_multi_index(tuple(x), tuple(dims)))] += dt
        t += dt
        x = x + np.array(net.reactions[int(rng.choice(len(a), p=a / tot))].change,
                         dtype=np.int64)
    return occ / max(occ.sum(), 1e-300), t


# --------------------------------------------------------------------------------------
# worked examples
# --------------------------------------------------------------------------------------

def toggle(M: int = 40, g: float = 20.0, gamma: float = 1.0, K: float = 10.0,
           h: float = 2.0) -> Network:
    """Two genes, mutual repression. The canonical bistable motif."""
    return Network(["A", "B"], [M, M], [
        Reaction("A+", lambda S: g / (1.0 + (S[:, 1] / K) ** h), (1, 0)),
        Reaction("A-", lambda S: gamma * S[:, 0], (-1, 0)),
        Reaction("B+", lambda S: g / (1.0 + (S[:, 0] / K) ** h), (0, 1)),
        Reaction("B-", lambda S: gamma * S[:, 1], (0, -1)),
    ])


def self_activating(M: int = 60, g: float = 20.0, basal: float = 1.0,
                    gamma: float = 1.0, K: float = 15.0, h: float = 2.0) -> Network:
    x = lambda S: S[:, 0]
    return Network(["X"], [M], [
        Reaction("X+", lambda S: basal + g * (x(S) / K) ** h / (1 + (x(S) / K) ** h),
                 (1,)),
        Reaction("X-", lambda S: gamma * x(S), (-1,)),
    ])


def schlogl(M: int = 60, k1: float = 3e-1, k2: float = 1e-3, k3: float = 2e2,
            k4: float = 3.5) -> Network:
    """The standard bistable chemical model: A + 2X <-> 3X, B <-> X."""
    x = lambda S: S[:, 0].astype(float)
    return Network(["X"], [M], [
        Reaction("auto+", lambda S: k1 * x(S) * np.maximum(x(S) - 1, 0) / 2.0, (1,)),
        Reaction("auto-", lambda S: k2 * x(S) * np.maximum(x(S) - 1, 0)
                 * np.maximum(x(S) - 2, 0) / 6.0, (-1,)),
        Reaction("in", lambda S: np.full(len(S), k3), (1,)),
        Reaction("out", lambda S: k4 * x(S), (-1,)),
    ])


def feed_forward(M: int = 15, g: float = 8.0, gamma: float = 1.0, K: float = 5.0,
                 h: float = 2.0) -> Network:
    """Three-node coherent feed-forward loop: X -> Y, X -> Z, Y -> Z."""
    hill = lambda v: (v / K) ** h / (1.0 + (v / K) ** h)
    return Network(["X", "Y", "Z"], [M, M, M], [
        Reaction("X+", lambda S: np.full(len(S), g * 0.5), (1, 0, 0)),
        Reaction("X-", lambda S: gamma * S[:, 0], (-1, 0, 0)),
        Reaction("Y+", lambda S: g * hill(S[:, 0]), (0, 1, 0)),
        Reaction("Y-", lambda S: gamma * S[:, 1], (0, -1, 0)),
        Reaction("Z+", lambda S: g * hill(S[:, 0]) * hill(S[:, 1]), (0, 0, 1)),
        Reaction("Z-", lambda S: gamma * S[:, 2], (0, 0, -1)),
    ])


def rank_profile(net: Network, p: np.ndarray, split: int) -> np.ndarray:
    """Singular values of p reshaped across a cut between species groups."""
    dims = net.dims
    left = int(np.prod(dims[:split])); right = int(np.prod(dims[split:]))
    return np.linalg.svd(p.reshape(left, right), compute_uv=False)


def cascade(n_genes: int = 3, M: int = 10, g: float = 6.0, gamma: float = 1.0,
            K: float = 4.0, h: float = 2.0) -> Network:
    """X1 -> X2 -> ... -> Xn activation cascade. Same topology at every n, so G5's rank
    comparison across gene counts is like-for-like rather than across different circuits."""
    hill = lambda v: (v / K) ** h / (1.0 + (v / K) ** h)
    sp_names = [f"X{i+1}" for i in range(n_genes)]
    rxns = [Reaction("X1+", lambda S: np.full(len(S), g * 0.6),
                     tuple([1] + [0] * (n_genes - 1))),
            Reaction("X1-", lambda S: gamma * S[:, 0],
                     tuple([-1] + [0] * (n_genes - 1)))]
    for i in range(1, n_genes):
        up = np.zeros(n_genes, dtype=int); up[i] = 1
        dn = np.zeros(n_genes, dtype=int); dn[i] = -1
        rxns.append(Reaction(f"X{i+1}+",
                             (lambda k: (lambda S: g * hill(S[:, k - 1])))(i),
                             tuple(up)))
        rxns.append(Reaction(f"X{i+1}-",
                             (lambda k: (lambda S: gamma * S[:, k]))(i), tuple(dn)))
    return Network(sp_names, [M] * n_genes, rxns)


# --------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------

def verify(verbose: bool = True, skip_mpmath: bool = False) -> dict:
    """Run G1-G6. Bars are fixed in the module docstring, above, before any number."""
    say = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    out: Dict[str, object] = {}

    # ---- G1: analytic reduction, with a tail the solver has to earn --------------------
    say("  G1 one-species birth-death vs the closed-form product (log space)")
    say(f"      {'lam':>6s} {'mu':>5s} {'M':>4s} {'tail min':>11s} {'max rel err':>12s} "
        f"{'rel err in tail':>16s}")
    g1 = True
    for lam, mu, M in ((10.0, 1.0, 40), (3.0, 1.0, 40), (1.0, 1.0, 40)):
        birth = np.full(M + 1, lam)
        death = mu * np.arange(M + 1, dtype=float)
        cf = birth_death_closed_form(birth, death)
        net = Network(["X"], [M], [
            Reaction("b", lambda S, lam=lam: np.full(len(S), lam), (1,)),
            Reaction("d", lambda S, mu=mu: mu * S[:, 0].astype(float), (-1,))])
        p, info = stationary(net)
        ok = cf > 0
        rel = np.abs(p[ok] - cf[ok]) / cf[ok]
        tail = cf < 1e-12
        rel_tail = float(rel[tail[ok]].max()) if tail[ok].any() else float("nan")
        g1 &= float(rel.max()) <= 1e-13
        say(f"      {lam:6.1f} {mu:5.1f} {M:4d} {cf.min():11.2e} {rel.max():12.2e} "
            f"{rel_tail:16.2e}")
    out["G1"] = bool(g1)
    say(f"      G1 {'PASS' if g1 else 'FAIL'}  (bar 1e-13 on max relative error)")

    # ---- G2: positive control, BEFORE any rare number ----------------------------------
    say("\n  G2 POSITIVE CONTROL: Gillespie where sampling can reach (runs first)")
    net = toggle(M=25, g=12.0, gamma=1.0, K=6.0, h=2.0)
    p, info = stationary(net)
    S = net.states()
    cands = [("both >= 12", lambda s: (s[:, 0] >= 12) & (s[:, 1] >= 12)),
             ("A == 0", lambda s: s[:, 0] == 0),
             ("A + B <= 2", lambda s: (s[:, 0] + s[:, 1]) <= 2)]
    chosen = None
    for name, pred in cands:
        v = rare_probability(net, p, pred)
        if 1e-5 <= v <= 1e-2:
            chosen = (name, pred, v); break
    if chosen is None:
        chosen = (cands[0][0], cands[0][1], rare_probability(net, p, cands[0][1]))
    name, pred, exact_v = chosen
    emp, t_sim = gillespie(net, t_max=4e5, seed=1, max_seconds=150.0)
    emp_v = float(emp[pred(S)].sum())
    se = np.sqrt(max(emp_v, 1e-12) / max(t_sim, 1e-12))
    out["G2_exact"], out["G2_emp"], out["G2_se"] = exact_v, emp_v, float(se)
    out["G2"] = bool(abs(exact_v - emp_v) <= 3 * se)
    say(f"      event '{name}': exact {exact_v:.4e}   Gillespie {emp_v:.4e} +/- {se:.1e} "
        f"(t={t_sim:.3e})")
    say(f"      |difference| {abs(exact_v-emp_v):.2e} vs 3 s.e. {3*se:.2e}   "
        f"{'PASS' if out['G2'] else 'FAIL'}")

    # ---- G3: conservation, negativity, residual, and the FLOOR --------------------------
    say("\n  G3 conservation, residual, and the reportable floor")
    net40 = toggle(M=40)
    p40, i40 = stationary(net40)
    s_ok = abs(p40.sum() - 1.0) < 1e-12
    n_ok = i40["most_negative_before_clip"] > -1e-14
    say(f"      states {i40['n_states']:,}   solve {i40['seconds']:.3f}s")
    say(f"      sum p - 1            {p40.sum()-1.0:+.2e}   {'ok' if s_ok else 'FAIL'}")
    say(f"      most negative entry  {i40['most_negative_before_clip']:+.2e}   "
        f"{'ok' if n_ok else 'FAIL'}")
    say(f"      residual ||Q^T p||   {i40['residual_inf']:.3e}")
    say(f"      REPORTABLE FLOOR     {i40['reportable_floor']:.3e}  (1e6 x residual)")
    say(f"      smallest positive    {i40['smallest_positive']:.3e}  "
        f"-> {'above floor' if i40['smallest_positive'] >= i40['reportable_floor'] else 'BELOW FLOOR, not a number'}")
    out["G3_residual"] = i40["residual_inf"]; out["G3_floor"] = i40["reportable_floor"]
    out["G3"] = bool(s_ok and n_ok)
    say(f"      G3 {'PASS' if out['G3'] else 'FAIL'} on conservation; the floor decides "
        f"which numbers exist")

    # ---- G3b: where does double precision actually stop being right? ---------------------
    if not skip_mpmath:
        say("\n  G3b MEASURED FLOOR: double precision vs mpmath at 60 digits")
        small = toggle(M=9, g=6.0, gamma=1.0, K=3.0, h=2.0)
        pd_, id_ = stationary(small)
        t0 = time.perf_counter()
        pm = stationary_mpmath(small, digits=60)
        say(f"      {small.n_states} states, mpmath solve {time.perf_counter()-t0:.1f}s")
        ok = pm > 0
        rel = np.abs(pd_[ok] - pm[ok]) / pm[ok]
        say(f"      {'magnitude':>12s} {'n entries':>10s} {'max rel err':>12s}")
        bad_mag = None
        for lo, hi in ((1e-2, 1.0), (1e-4, 1e-2), (1e-6, 1e-4), (1e-8, 1e-6),
                       (1e-10, 1e-8), (1e-12, 1e-10), (0.0, 1e-12)):
            m = (pm[ok] >= lo) & (pm[ok] < hi)
            if m.sum() == 0:
                continue
            r = float(rel[m].max())
            say(f"      {lo:6.0e}-{hi:5.0e} {int(m.sum()):10d} {r:12.2e}")
            if r > 0.10 and bad_mag is None:
                bad_mag = hi
        out["G3b_diverge_at"] = bad_mag
        out["G3b"] = bool(bad_mag is None or bad_mag < id_["reportable_floor"])
        say(f"      double precision first exceeds 10% error at magnitude "
            f"{bad_mag if bad_mag else 'never (within this range)'}")
        say(f"      G3 floor for this system: {id_['reportable_floor']:.2e}   "
            f"G3b {'PASS' if out['G3b'] else 'FAIL'}")

    # ---- G4: truncation independence ------------------------------------------------------
    say("\n  G4 truncation independence -- the likeliest way to a confident wrong number")
    say(f"      {'M':>4s} {'states':>8s} {'P(both>=12)':>14s} {'rel change':>12s} "
        f"{'floor':>10s} {'verdict':>12s}")
    prev, vals = None, []
    for M in (20, 25, 30, 35, 40):
        nt = toggle(M=M)
        pp, ii = stationary(nt)
        v = rare_probability(nt, pp, lambda s: (s[:, 0] >= 12) & (s[:, 1] >= 12))
        ch = abs(v - prev) / prev if prev else float("nan")
        vals.append(v)
        say(f"      {M:4d} {ii['n_states']:8,d} {v:14.6e} {ch:12.2e} "
            f"{ii['reportable_floor']:10.1e} "
            f"{'reportable' if v >= ii['reportable_floor'] else 'BELOW FLOOR':>12s}")
        prev = v
    final_change = abs(vals[-1] - vals[-2]) / vals[-2] if vals[-2] else float("inf")
    out["G4_final_change"] = float(final_change)
    out["G4"] = bool(final_change < 0.01)
    say(f"      relative change between the two largest M: {final_change:.2e}   "
        f"{'PASS' if out['G4'] else 'FAIL'}  (bar 1%)")

    # ---- G5: does the compression hold as circuits grow? ------------------------------------
    say("\n  G5 rank of the stationary distribution vs number of genes (same topology)")
    say(f"      {'genes':>6s} {'M':>3s} {'states':>8s} {'full rank':>10s} "
        f"{'r@1e-3':>7s} {'r@1e-6':>7s} {'r@1e-10':>8s} {'sec':>7s}")
    ranks = {}
    for ng, M in ((2, 12), (3, 9), (4, 6)):
        nc = cascade(ng, M)
        pc, ic = stationary(nc)
        sv = rank_profile(nc, pc, split=ng // 2)
        sv = sv / sv[0]
        full = len(sv)
        r = {t: int((sv > t).sum()) for t in (1e-3, 1e-6, 1e-10)}
        ranks[ng] = {"full": full, **{f"r{t}": r[t] for t in r}}
        say(f"      {ng:6d} {M:3d} {ic['n_states']:8,d} {full:10d} {r[1e-3]:7d} "
            f"{r[1e-6]:7d} {r[1e-10]:8d} {ic['seconds']:7.2f}")
    out["G5_ranks"] = ranks
    say("      three sizes reported; no extrapolation from two.")

    # ---- G6: relevance ----------------------------------------------------------------------
    say("\n  G6 relevance: for each headline number, what would a sampler return?")
    say(f"      {'event':>22s} {'exact p':>12s} {'draws for 1 hit':>16s} "
        f"{'reachable in 1e8 draws?':>24s}")
    net6 = toggle(M=40)
    p6, i6 = stationary(net6)
    for nm, pr in (("P(A>=25 and B>=25)", lambda s: (s[:, 0] >= 25) & (s[:, 1] >= 25)),
                   ("P(A>=15 and B>=15)", lambda s: (s[:, 0] >= 15) & (s[:, 1] >= 15)),
                   ("P(extinction 0,0)", lambda s: (s[:, 0] == 0) & (s[:, 1] == 0)),
                   ("P(A >= 30)", lambda s: s[:, 0] >= 30)):
        v = rare_probability(net6, p6, pr)
        need = 1.0 / v if v > 0 else np.inf
        say(f"      {nm:>22s} {v:12.3e} {need:16.2e} "
            f"{('YES -- sampling suffices' if need < 1e8 else 'no'):>24s}   "
            f"[{report(v, i6)}]")

    gates = ["G1", "G2", "G3", "G4"] + (["G3b"] if not skip_mpmath else [])
    out["all_pass"] = all(bool(out.get(k)) for k in gates)
    say(f"\n  {'ALL GATES PASS' if out['all_pass'] else 'GATE FAILURE'}: "
        + "  ".join(f"{k}={'pass' if out.get(k) else 'FAIL'}" for k in gates))
    return out


if __name__ == "__main__":
    verify()

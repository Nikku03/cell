"""Exact aggregate and rare-event distributions by convolution of heterogeneous units.

THE PROBLEM. N independent units, unit i carrying its own discrete distribution over
integer outcomes (a loss, a count, a delay). What is P(sum_i X_i >= t) when t sits far
out in the tail -- so far that a Monte Carlo run of 10^7 draws never once gets there and
returns a hard, uninformative ZERO?

THE ANSWER IS A CONVOLUTION, AND THE CONVOLUTION IS BUCKET ELIMINATION ON A CHAIN.
Introduce the running partial sums S_k = X_1 + ... + X_k. The factor graph is

    X_1   X_2   X_3        ...
     |     |     |
    (S_1)-(S_2)-(S_3)- ... -(S_N)          factor k couples (S_{k-1}, X_k, S_k)

Cut the chain anywhere. The ONLY thing that crosses the cut is the single variable S_k:
everything to the left is summarised by the running total. So

    separator size (treewidth) = 1
    bond dimension across cut k = d_k = |support(S_k)|
    cost = d ** treewidth  =  d_k ** 1  per cut, times the unit's own support m_{k+1}

which is exactly THE GOVERNING LAW at treewidth 1. Total cost O(N * d_max * m_max):
linear in the number of units, linear in the aggregate support. That -- and nothing more
exotic -- is why the deep tail is reachable exactly. aggregate_distribution() MEASURES and
logs the bond dimension after every step so the claim is a number, not a story.

WHAT BREAKS THIS. Independence. If the units are coupled by a dependency graph of
treewidth w, the cost is d**w and the chain argument is gone. The one cheap escape is a
common latent factor: condition on it, aggregate each conditional (still a chain), and mix
-- that is mixture_aggregate(), and it costs (number of latent states) x the chain.

THE THREE CONVOLUTIONS, AND WHY THE MODULE SHIPS ALL THREE. They agree in the bulk and
disagree by 300 orders of magnitude in the tail, which is the whole point:

  convolve_direct  linear space, O(La*Lb). Sums of products of non-negative numbers, so
                   there is NO cancellation and RELATIVE accuracy survives all the way
                   down to the float64 subnormal floor, 4.94e-324. Below that: exactly 0.
  convolve_fft     O(N log N) but it is a sum of ~N complex terms with cancelling signs.
                   Its error floor is ABSOLUTE, about eps * max(result) ~ 1e-17, not
                   relative. Anything smaller than that is round-off noise -- verify()
                   measures FFT entries that come out NEGATIVE, which no probability is.
                   Use it for the bulk, never for the tail.
  convolve_log     log space via logaddexp. Floor is exp(-1.8e308): no practical floor at
                   all. Needed once the answer drops past 1e-308, e.g. a 600-unit
                   portfolio whose tail is 1e-400 and which linear space reports as 0.0.

GROUND TRUTH FOR A NUMBER MONTE CARLO CANNOT SEE. Claiming "the exact tail is 1e-30" is
worthless unless something independent confirms it, and neither MC nor brute-force
enumeration (4^24 = 2.8e14 joint outcomes) can. So the demo portfolio's probabilities are
exact rationals k/10000, and exact_integer_aggregate() convolves the INTEGER NUMERATORS in
Python bigint arithmetic -- no floats, no logs, no FFT anywhere in that code path. The
resulting Fraction is the exact answer. verify() compares it to the float64 log-space
answer at the 1e-30 threshold.

Every claim here is checked by verify(): brute-force enumeration for the pmf, exact
rational arithmetic for the deep tail, and Monte Carlo as the positive control that both
methods are computing the same quantity where MC still works.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "Unit", "IntUnit", "Portfolio", "Aggregate",
    "as_unit", "convolve_direct", "convolve_fft", "convolve_log",
    "aggregate_distribution", "tail_probability", "log_tail_probability",
    "log_tail_curve",
    "monte_carlo_tail", "brute_force_aggregate", "mixture_aggregate",
    "value_at_risk", "expected_shortfall",
    "exact_integer_aggregate", "exact_rational_tail",
    "demo_portfolio", "underflow_report", "verify",
]

# ---------------------------------------------------------------------------------------
# Representable floors. These are the numbers the module quotes when it says "floor".
# ---------------------------------------------------------------------------------------
TINY_NORMAL = float(np.finfo(np.float64).tiny)        # 2.2250738585072014e-308
TINY_SUBNORMAL = float(np.nextafter(0.0, 1.0))        # 4.9406564584124654e-324
LOG_UNDERFLOW = math.log(TINY_SUBNORMAL)              # -744.44...; exp() below this is 0.0
LOG_SPACE_FLOOR = -float(np.finfo(np.float64).max)    # -1.7976931348623157e308


def underflow_report() -> Dict[str, float]:
    """The representable floor of each convolution mode. Quoted, not guessed."""
    return {
        "float64_smallest_normal": TINY_NORMAL,
        "float64_smallest_subnormal": TINY_SUBNORMAL,
        "linear_convolution_floor": TINY_SUBNORMAL,
        "log_space_floor_as_log_p": LOG_SPACE_FLOOR,
        "log_space_floor_as_p": 0.0,           # exp(-1.8e308) is not representable at all
        "fft_convolution_floor_is_relative": float(np.finfo(np.float64).eps),
    }


def _logsumexp(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return -math.inf
    m = float(np.max(a))
    if m == -math.inf:
        return -math.inf
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.sum(np.exp(a - m))))


def _safe_exp(x: float) -> float:
    """exp() that returns 0.0 on underflow instead of raising or warning."""
    if x <= LOG_UNDERFLOW:
        return 0.0
    if x > 709.78:
        return math.inf
    return math.exp(x)


def _safe_log(a: np.ndarray) -> np.ndarray:
    """log() of a possibly-noisy linear array; non-positive entries -> -inf."""
    a = np.asarray(a, dtype=float)
    out = np.full(a.shape, -math.inf)
    pos = a > 0.0
    out[pos] = np.log(a[pos])
    return out


# ---------------------------------------------------------------------------------------
# Unit distributions
# ---------------------------------------------------------------------------------------
@dataclass
class Unit:
    """One unit's distribution over CONSECUTIVE integers offset .. offset+len(pmf)-1.

    Zero entries are allowed (a unit may be supported on {0, 3, 7}); the contiguous array
    is just the storage. Units are heterogeneous by construction: different offsets,
    different lengths, different probabilities."""
    offset: int
    pmf: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        p = np.asarray(self.pmf, dtype=float).ravel()
        if p.size == 0:
            raise ValueError("unit pmf is empty")
        if not np.all(np.isfinite(p)):
            raise ValueError("unit pmf has non-finite entries")
        if np.any(p < 0):
            raise ValueError("unit pmf has negative entries")
        s = float(p.sum())
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"unit pmf must sum to 1, got {s!r}")
        self.pmf = p / s
        self.offset = int(self.offset)

    def __len__(self) -> int:
        return int(self.pmf.size)

    @property
    def support(self) -> np.ndarray:
        return np.arange(self.offset, self.offset + self.pmf.size)

    @property
    def mean(self) -> float:
        return float(self.support @ self.pmf)

    @property
    def var(self) -> float:
        m = self.mean
        return float(((self.support - m) ** 2) @ self.pmf)

    @property
    def min_positive(self) -> float:
        pos = self.pmf[self.pmf > 0]
        return float(pos.min()) if pos.size else 0.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        cdf = np.cumsum(self.pmf)
        cdf[-1] = 1.0
        return np.searchsorted(cdf, rng.random(n), side="right") + self.offset


def as_unit(x: Any) -> Unit:
    """Coerce a Unit / {value: prob} dict / (offset, pmf) pair / bare pmf array."""
    if isinstance(x, Unit):
        return x
    if isinstance(x, Mapping):
        keys = sorted(int(k) for k in x)
        lo, hi = keys[0], keys[-1]
        p = np.zeros(hi - lo + 1)
        for k, v in x.items():
            p[int(k) - lo] += float(v)
        return Unit(lo, p)
    if isinstance(x, tuple) and len(x) == 2 and np.ndim(x[0]) == 0 and np.ndim(x[1]) == 1:
        return Unit(int(x[0]), np.asarray(x[1], dtype=float))
    return Unit(0, np.asarray(x, dtype=float))


@dataclass
class IntUnit:
    """A unit whose probabilities are EXACT rationals num[j] / den. No floats."""
    offset: int
    num: List[int]
    den: int

    def __post_init__(self) -> None:
        self.num = [int(v) for v in self.num]
        self.den = int(self.den)
        if any(v < 0 for v in self.num):
            raise ValueError("negative numerator")
        if sum(self.num) != self.den:
            raise ValueError(f"numerators sum to {sum(self.num)}, not den={self.den}")

    def to_unit(self) -> Unit:
        return Unit(self.offset, np.asarray(self.num, dtype=float) / self.den)


@dataclass
class Portfolio:
    """The same heterogeneous units in two forms: float64 and exact rational."""
    units: List[Unit]
    int_units: List[IntUnit]

    def __len__(self) -> int:
        return len(self.units)

    @property
    def max_sum(self) -> int:
        return sum(u.offset + len(u) - 1 for u in self.units)

    @property
    def mean(self) -> float:
        return float(sum(u.mean for u in self.units))

    @property
    def sd(self) -> float:
        return math.sqrt(sum(u.var for u in self.units))


# ---------------------------------------------------------------------------------------
# The three convolutions
# ---------------------------------------------------------------------------------------
def convolve_direct(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Linear-space convolution, O(La*Lb) multiply-adds.

    Every term is a product of two non-negative numbers and every accumulation is a sum of
    non-negative numbers, so there is no cancellation: the RELATIVE error of a tail entry
    stays ~ n*eps no matter how small the entry is, right down to the subnormal floor."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < b.size:
        a, b = b, a
    out = np.zeros(a.size + b.size - 1)
    for j in range(b.size):
        bj = b[j]
        if bj != 0.0:
            out[j:j + a.size] += a * bj
    return out


def convolve_fft(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """O(N log N) convolution through the Fourier domain. Raw output, NOT clipped.

    The result is a sum of ~N complex terms whose signs cancel, so its error is ABSOLUTE
    (about eps * max|result|), not relative. Deep-tail entries are pure round-off noise and
    can be negative. Left unclipped on purpose so verify() can count the negatives."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size + b.size - 1
    nfft = 1 << max(1, (n - 1)).bit_length()
    out = np.fft.irfft(np.fft.rfft(a, nfft) * np.fft.rfft(b, nfft), nfft)[:n]
    return out


def convolve_log(la: Sequence[float], lb: Sequence[float]) -> np.ndarray:
    """Convolution in LOG space: out[k] = log sum_j exp(la[k-j] + lb[j]).

    Uses np.logaddexp, which is exact to a fraction of an ulp of the log, so the answer
    is meaningful at log p = -1e5 where linear space has been 0.0 for 99000 decades."""
    la = np.asarray(la, dtype=float)
    lb = np.asarray(lb, dtype=float)
    if la.size < lb.size:
        la, lb = lb, la
    out = np.full(la.size + lb.size - 1, -math.inf)
    for j in range(lb.size):
        if lb[j] == -math.inf:
            continue
        seg = out[j:j + la.size]
        np.logaddexp(seg, la + lb[j], out=seg)
    return out


# ---------------------------------------------------------------------------------------
# The aggregate distribution
# ---------------------------------------------------------------------------------------
@dataclass
class Aggregate:
    """The exact pmf of sum_i X_i over consecutive integers offset .. offset+L-1."""
    offset: int
    pmf: np.ndarray            # linear space; underflows to 0 below 4.94e-324
    logpmf: np.ndarray         # log space; the authoritative copy for the deep tail
    method: str
    info: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.logpmf.size)

    @property
    def support(self) -> np.ndarray:
        return np.arange(self.offset, self.offset + len(self))

    @property
    def total_mass(self) -> float:
        return _safe_exp(_logsumexp(self.logpmf))

    @property
    def mean(self) -> float:
        return float(self.support @ self.pmf)

    @property
    def var(self) -> float:
        m = self.mean
        return float(((self.support - m) ** 2) @ self.pmf)


def aggregate_distribution(unit_distributions: Sequence[Any],
                           method: str = "auto",
                           verbose: bool = False) -> Aggregate:
    """EXACT pmf of the sum of independent heterogeneous units, by convolution.

    method: "log" (default whenever the tail can underflow float64), "direct" (linear),
            "fft", or "auto" which picks log vs direct from a measured underflow budget:
            the product of each unit's smallest positive probability is a lower bound on
            the smallest attainable joint probability; if that is below the float64 normal
            floor, linear space cannot represent the answer and log space is used.

    Logs the bond dimension (= |support| of the running partial sum) after every step.
    That is d in cost = d ** treewidth, with treewidth 1 for a chain of partial sums."""
    units = [as_unit(u) for u in unit_distributions]
    if not units:
        raise ValueError("no units given")

    log_budget = 0.0
    for u in units:
        mp = u.min_positive
        log_budget += math.log(mp) if mp > 0 else -math.inf
    if method == "auto":
        method = "log" if log_budget < math.log(TINY_NORMAL) else "direct"
    if method not in ("log", "direct", "fft"):
        raise ValueError(f"unknown method {method!r}")

    t0 = time.perf_counter()
    offset = units[0].offset
    bonds = [len(units[0])]
    ops = 0

    if method == "log":
        cur = _safe_log(units[0].pmf)
        for u in units[1:]:
            lu = _safe_log(u.pmf)
            ops += cur.size * lu.size
            cur = convolve_log(cur, lu)
            offset += u.offset
            bonds.append(int(cur.size))
        logpmf = cur
        with np.errstate(under="ignore"):
            pmf = np.exp(logpmf)
    else:
        conv = convolve_direct if method == "direct" else convolve_fft
        cur = units[0].pmf.copy()
        for u in units[1:]:
            ops += cur.size * u.pmf.size
            cur = conv(cur, u.pmf)
            offset += u.offset
            bonds.append(int(cur.size))
        pmf = cur
        logpmf = _safe_log(cur)

    dt = time.perf_counter() - t0
    info = {
        "method": method,
        "n_units": len(units),
        "support_size": int(logpmf.size),
        "bond_dimensions": bonds,
        "max_bond_dimension": int(max(bonds)),
        "treewidth": 1,
        "cost_multiply_adds": int(ops),
        "time_s": dt,
        "log_underflow_budget": log_budget,
        "linear_space_sufficient": bool(log_budget >= math.log(TINY_NORMAL)),
    }
    if verbose:
        print(f"    aggregate: {len(units)} units, method={method}, "
              f"support {logpmf.size}, max bond dimension {max(bonds)}, "
              f"treewidth 1  ->  cost = d^1 per cut = {ops} multiply-adds, {dt*1e3:.1f} ms")
    return Aggregate(offset, pmf, logpmf, method, info)


def _as_aggregate(pmf: Any, offset: int = 0) -> Aggregate:
    if isinstance(pmf, Aggregate):
        return pmf
    arr = np.asarray(pmf, dtype=float)
    return Aggregate(int(offset), arr, _safe_log(arr), "given", {})


def log_tail_probability(pmf: Any, threshold: float, offset: int = 0,
                         side: str = "upper") -> float:
    """log P(sum >= threshold)  (side="upper")  or  log P(sum <= threshold)  ("lower").

    Summed with logsumexp, so the answer is meaningful far below the float64 floor.
    Returns -inf when no support satisfies the condition."""
    agg = _as_aggregate(pmf, offset)
    L = len(agg)
    if side == "upper":
        i = int(math.ceil(threshold)) - agg.offset
        if i >= L:
            return -math.inf
        return _logsumexp(agg.logpmf[max(i, 0):])
    if side == "lower":
        j = int(math.floor(threshold)) - agg.offset
        if j < 0:
            return -math.inf
        return _logsumexp(agg.logpmf[:min(j + 1, L)])
    raise ValueError(f"side must be 'upper' or 'lower', got {side!r}")


def tail_probability(pmf: Any, threshold: float, offset: int = 0,
                     side: str = "upper") -> float:
    """EXACT P(sum >= threshold) as a float64.

    Underflows to 0.0 below 4.94e-324 -- that is a property of float64, not of the
    computation. Use log_tail_probability() for anything deeper."""
    return _safe_exp(log_tail_probability(pmf, threshold, offset=offset, side=side))


def log_tail_curve(pmf: Any, offset: int = 0, side: str = "upper") -> np.ndarray:
    """log P(sum >= t) for EVERY t in the support, in one O(L) reverse pass.

    Same answer as calling log_tail_probability() at each t (verified against it), but
    linear rather than quadratic. np.logaddexp.accumulate is the log-space cumulative
    sum, so no partial sum is ever exponentiated and the deep tail is preserved."""
    agg = _as_aggregate(pmf, offset)
    if side == "upper":
        return np.logaddexp.accumulate(agg.logpmf[::-1])[::-1]
    if side == "lower":
        return np.logaddexp.accumulate(agg.logpmf)
    raise ValueError(f"side must be 'upper' or 'lower', got {side!r}")


def value_at_risk(pmf: Any, alpha: float, offset: int = 0) -> int:
    """Smallest integer t with P(sum <= t) >= alpha. Exact, by scanning the exact cdf."""
    agg = _as_aggregate(pmf, offset)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    c = np.cumsum(agg.pmf)
    idx = int(np.searchsorted(c, alpha * c[-1], side="left"))
    idx = min(idx, len(agg) - 1)
    return int(agg.offset + idx)


def expected_shortfall(pmf: Any, alpha: float, offset: int = 0) -> float:
    """E[sum | sum >= VaR_alpha], computed exactly from the pmf (not sampled)."""
    agg = _as_aggregate(pmf, offset)
    v = value_at_risk(agg, alpha)
    i = v - agg.offset
    w = agg.pmf[i:]
    tot = float(w.sum())
    if tot <= 0.0:
        return float(v)
    return float((np.arange(v, v + w.size) @ w) / tot)


def mixture_aggregate(weights: Sequence[float],
                      components: Sequence[Any],
                      method: str = "auto") -> Aggregate:
    """Aggregate under a COMMON LATENT FACTOR: sum_s w_s * (chain aggregate given s).

    Units are independent only CONDITIONAL on the latent state. Cost is (number of latent
    states) x the chain cost -- the dependency graph has treewidth 2 (each unit touches
    the latent and the running sum), so the law reads cost = d^1 per cut x |latent|."""
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("mixture weights must be non-negative")
    s = float(w.sum())
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"mixture weights must sum to 1, got {s!r}")
    w = w / s
    comps = [c if isinstance(c, Aggregate) else aggregate_distribution(c, method=method)
             for c in components]
    if len(comps) != w.size:
        raise ValueError("weights and components have different lengths")
    lo = min(c.offset for c in comps)
    hi = max(c.offset + len(c) for c in comps)
    out = np.full(hi - lo, -math.inf)
    for wi, c in zip(w, comps):
        if wi <= 0.0:
            continue
        a = c.offset - lo
        seg = out[a:a + len(c)]
        np.logaddexp(seg, c.logpmf + math.log(wi), out=seg)
    with np.errstate(under="ignore"):
        pmf = np.exp(out)
    return Aggregate(lo, pmf, out, "mixture",
                     {"n_components": len(comps), "treewidth": 1,
                      "latent_states": len(comps),
                      "support_size": int(out.size)})


# ---------------------------------------------------------------------------------------
# Brute force -- a genuinely different algorithm. No convolution, no FFT, no logs.
# ---------------------------------------------------------------------------------------
def brute_force_aggregate(unit_distributions: Sequence[Any]) -> Aggregate:
    """Enumerate EVERY joint outcome, multiply its per-unit probabilities, bin by sum.

    O(prod_i m_i) and completely independent of the convolution code path: it never calls
    convolve_* and never touches an FFT. This is the ground truth for verify() (a)."""
    units = [as_unit(u) for u in unit_distributions]
    probs = [[float(v) for v in u.pmf] for u in units]
    vals = [[u.offset + j for j in range(len(u))] for u in units]
    acc: Dict[int, float] = {}
    for combo in itertools.product(*[range(len(p)) for p in probs]):
        p = 1.0
        s = 0
        for i, j in enumerate(combo):
            p *= probs[i][j]
            s += vals[i][j]
        acc[s] = acc.get(s, 0.0) + p
    lo, hi = min(acc), max(acc)
    arr = np.zeros(hi - lo + 1)
    for k, v in acc.items():
        arr[k - lo] = v
    return Aggregate(lo, arr, _safe_log(arr), "bruteforce",
                     {"n_joint_outcomes": int(np.prod([len(p) for p in probs])),
                      "support_size": int(arr.size)})


def brute_force_tail(unit_distributions: Sequence[Any], threshold: float) -> float:
    """P(sum >= threshold) by enumeration. Never calls tail_probability()."""
    units = [as_unit(u) for u in unit_distributions]
    probs = [[float(v) for v in u.pmf] for u in units]
    vals = [[u.offset + j for j in range(len(u))] for u in units]
    tot = 0.0
    for combo in itertools.product(*[range(len(p)) for p in probs]):
        p = 1.0
        s = 0
        for i, j in enumerate(combo):
            p *= probs[i][j]
            s += vals[i][j]
        if s >= threshold:
            tot += p
    return tot


# ---------------------------------------------------------------------------------------
# Exact rational ground truth for the deep tail -- Python bigints, zero floating point
# ---------------------------------------------------------------------------------------
def exact_integer_aggregate(int_units: Sequence[IntUnit]) -> Tuple[List[int], int, int]:
    """Convolve integer NUMERATORS with arbitrary-precision integer arithmetic.

    Returns (numerators, offset, total_denominator). The pmf entry for value
    offset+k is exactly numerators[k] / total_denominator -- a rational, no rounding
    anywhere. This is the only way to have ground truth at p ~ 1e-30, where Monte Carlo
    sees nothing and enumeration of 4^24 joint outcomes is out of reach."""
    if not int_units:
        raise ValueError("no units given")
    cur = list(int_units[0].num)
    off = int(int_units[0].offset)
    den = int(int_units[0].den)
    for u in int_units[1:]:
        new = [0] * (len(cur) + len(u.num) - 1)
        for i, a in enumerate(cur):
            if a:
                for j, b in enumerate(u.num):
                    if b:
                        new[i + j] += a * b
        cur = new
        off += u.offset
        den *= u.den
    return cur, off, den


def exact_rational_tail(int_units: Sequence[IntUnit], threshold: float) -> Fraction:
    """EXACT P(sum >= threshold) as a Fraction. No float64 is used to produce it."""
    num, off, den = exact_integer_aggregate(int_units)
    i = int(math.ceil(threshold)) - off
    if i >= len(num):
        return Fraction(0, 1)
    return Fraction(sum(num[max(i, 0):]), den)


# ---------------------------------------------------------------------------------------
# Monte Carlo -- the comparison, and the positive control
# ---------------------------------------------------------------------------------------
def monte_carlo_tail(unit_distributions: Sequence[Any],
                     threshold: Union[float, Sequence[float]],
                     n_samples: int,
                     seed: int = 0,
                     chunk: int = 1_000_000) -> Dict[str, Any]:
    """Estimate P(sum >= threshold) by sampling. threshold may be a list of thresholds,
    all evaluated on the SAME sample set, which is what makes the bulk agreement a genuine
    positive control for the deep-tail zero.

    Returns hits, estimate, standard error, and -- when hits == 0 -- the only honest
    statement Monte Carlo can make: the rule-of-three 95% upper bound 3/n."""
    units = [as_unit(u) for u in unit_distributions]
    scalar = np.ndim(threshold) == 0
    thr = np.atleast_1d(np.asarray(threshold, dtype=float))
    rng = np.random.default_rng(seed)
    cdfs = []
    offs = []
    for u in units:
        c = np.cumsum(u.pmf)
        c[-1] = 1.0
        cdfs.append(c)
        offs.append(u.offset)

    hits = np.zeros(thr.size, dtype=np.int64)
    max_seen = -(2 ** 62)
    done = 0
    t0 = time.perf_counter()
    while done < n_samples:
        m = int(min(chunk, n_samples - done))
        tot = np.zeros(m, dtype=np.int64)
        for c, o in zip(cdfs, offs):
            tot += np.searchsorted(c, rng.random(m), side="right").astype(np.int64) + o
        max_seen = max(max_seen, int(tot.max()))
        for k in range(thr.size):
            hits[k] += int(np.count_nonzero(tot >= thr[k]))
        done += m
    dt = time.perf_counter() - t0

    est = hits / float(n_samples)
    se = np.sqrt(np.maximum(est * (1.0 - est), 0.0) / float(n_samples))
    rule_of_three = 3.0 / float(n_samples)
    ci95_upper = np.where(hits == 0, rule_of_three, est + 1.96 * se)
    out: Dict[str, Any] = {
        "n_samples": int(n_samples),
        "seed": int(seed),
        "threshold": thr[0] if scalar else thr,
        "hits": int(hits[0]) if scalar else hits,
        "estimate": float(est[0]) if scalar else est,
        "stderr": float(se[0]) if scalar else se,
        "ci95_upper": float(ci95_upper[0]) if scalar else ci95_upper,
        "returned_exact_zero": bool(hits[0] == 0) if scalar else (hits == 0),
        "rule_of_three_bound": rule_of_three,
        "max_sample_sum": int(max_seen),
        "time_s": dt,
    }
    return out


# ---------------------------------------------------------------------------------------
# The demo portfolio: heterogeneous, reproducible, exactly rational
# ---------------------------------------------------------------------------------------
def demo_portfolio(n_units: int = 40, seed: int = 20260829,
                   denominator: int = 10000) -> Portfolio:
    """N heterogeneous unit loss distributions with probabilities that are exact
    multiples of 1/denominator, so the same portfolio can be run through float64
    convolution AND through exact bigint rational convolution."""
    rng = np.random.default_rng(seed)
    units: List[Unit] = []
    iunits: List[IntUnit] = []
    for _ in range(n_units):
        vmax = int(rng.integers(3, 10))
        k = int(min(rng.integers(1, 4), vmax))
        vals = sorted(int(v) for v in
                      rng.choice(np.arange(1, vmax + 1), size=k, replace=False))
        nums = []
        for j, _v in enumerate(vals):
            lo_n = max(2, denominator // 50)
            hi_n = max(lo_n + 2, denominator * 7 // 50)
            nums.append(int(rng.integers(lo_n, hi_n)) // (j + 1) + 1)
        arr = [0] * (vals[-1] + 1)
        for v, nu in zip(vals, nums):
            arr[v] = nu
        arr[0] = denominator - sum(nums)
        if arr[0] <= 0:
            raise ValueError("degenerate unit; adjust generator")
        iu = IntUnit(0, arr, denominator)
        iunits.append(iu)
        units.append(iu.to_unit())
    return Portfolio(units, iunits)


# ---------------------------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------------------------
def _random_small_units(rng: np.random.Generator, n: int) -> List[Unit]:
    us = []
    for _ in range(n):
        m = int(rng.integers(2, 5))
        p = rng.random(m) ** 2 + 1e-3
        if m >= 3 and rng.random() < 0.4:
            p[int(rng.integers(1, m - 1))] = 0.0      # a hole in the support
        p = p / p.sum()
        us.append(Unit(int(rng.integers(-2, 3)), p))
    return us


def verify(verbose: bool = True, n_samples: int = 10_000_000,
           seed: int = 0) -> Dict[str, Any]:
    """Check every claim this module makes against an independent ground truth."""
    rng = np.random.default_rng(seed)
    res: Dict[str, Any] = {}
    if verbose:
        print("  rem.aggregate.verify")

    # ---- (a) exact pmf vs BRUTE-FORCE ENUMERATION -------------------------------------
    e_log = e_dir = e_fft = e_mass = 0.0
    e_tail = e_var = e_es = 0.0
    n_cases = 0
    for _ in range(24):
        units = _random_small_units(rng, int(rng.integers(2, 6)))
        bf = brute_force_aggregate(units)
        for meth, slot in (("log", "log"), ("direct", "direct"), ("fft", "fft")):
            agg = aggregate_distribution(units, method=meth)
            assert agg.offset == bf.offset and len(agg) == len(bf), "support mismatch"
            err = float(np.max(np.abs(agg.pmf - bf.pmf)))
            if slot == "log":
                e_log = max(e_log, err)
                e_mass = max(e_mass, abs(float(agg.pmf.sum()) - 1.0))
                lo, hi = bf.offset, bf.offset + len(bf)
                for t in range(lo, hi + 1):
                    e_tail = max(e_tail, abs(tail_probability(agg, t)
                                             - brute_force_tail(units, t)))
                for al in (0.5, 0.9, 0.99):
                    e_var = max(e_var, abs(value_at_risk(agg, al)
                                           - value_at_risk(bf, al)))
                    e_es = max(e_es, abs(expected_shortfall(agg, al)
                                         - expected_shortfall(bf, al)))
            elif slot == "direct":
                e_dir = max(e_dir, err)
            else:
                e_fft = max(e_fft, err)
        n_cases += 1

    # mixture (common latent factor) against enumeration of the same mixture
    e_mix = 0.0
    for _ in range(6):
        w = rng.random(2)
        w = w / w.sum()
        sets = [_random_small_units(rng, 3) for _ in range(2)]
        mix = mixture_aggregate(w, sets)
        bfs = [brute_force_aggregate(s) for s in sets]
        lo = min(b.offset for b in bfs)
        hi = max(b.offset + len(b) for b in bfs)
        ref = np.zeros(hi - lo)
        for wi, b in zip(w, bfs):
            ref[b.offset - lo:b.offset - lo + len(b)] += wi * b.pmf
        assert mix.offset == lo and len(mix) == ref.size
        e_mix = max(e_mix, float(np.max(np.abs(mix.pmf - ref))))

    if verbose:
        print(f"    (a) exact pmf vs BRUTE-FORCE ENUMERATION over all joint outcomes, "
              f"{n_cases} heterogeneous instances")
        print(f"          max |pmf_log    - pmf_bruteforce|      {e_log:.3e}")
        print(f"          max |pmf_direct - pmf_bruteforce|      {e_dir:.3e}")
        print(f"          max |pmf_fft    - pmf_bruteforce|      {e_fft:.3e}")
        print(f"          max |sum(pmf) - 1|                     {e_mass:.3e}")
        print(f"          max |P(S>=t) - bruteforce P(S>=t)|     {e_tail:.3e}  "
              f"(all thresholds)")
        print(f"          max |VaR - bruteforce VaR|             {e_var:.3e}")
        print(f"          max |ES  - bruteforce ES|              {e_es:.3e}")
        print(f"          max |mixture - bruteforce mixture|     {e_mix:.3e}  "
              f"(common latent factor)")
    res["a"] = {"max_err_log": e_log, "max_err_direct": e_dir, "max_err_fft": e_fft,
                "max_err_mass": e_mass, "max_err_tail": e_tail, "max_err_var": e_var,
                "max_err_es": e_es, "max_err_mixture": e_mix, "n_instances": n_cases}

    # ---- (b) THE HEADLINE ------------------------------------------------------------
    pf = demo_portfolio()
    agg = aggregate_distribution(pf.units, method="log")
    ai = agg.info
    logtail = log_tail_curve(agg)
    e_curve = float(np.max(np.abs(logtail - np.array(
        [log_tail_probability(agg, t) for t in agg.support]))))

    # pick a bulk threshold (p ~ 1e-2), a mid threshold (p ~ 1e-5), a deep one (p ~ 1e-30)
    def _closest(target_log: float) -> int:
        return int(agg.support[int(np.argmin(np.abs(logtail - target_log)))])
    t_bulk = _closest(math.log(1e-2))
    t_mid = _closest(math.log(1e-5))
    t_deep = _closest(math.log(1e-30))

    mc = monte_carlo_tail(pf.units, [t_bulk, t_mid, t_deep], n_samples=n_samples, seed=1)
    p_bulk = _safe_exp(log_tail_probability(agg, t_bulk))
    p_mid = _safe_exp(log_tail_probability(agg, t_mid))
    p_deep = _safe_exp(log_tail_probability(agg, t_deep))

    # exact rational ground truth for all three, from bigint arithmetic
    fr_bulk = exact_rational_tail(pf.int_units, t_bulk)
    fr_mid = exact_rational_tail(pf.int_units, t_mid)
    fr_deep = exact_rational_tail(pf.int_units, t_deep)
    rel_bulk = abs(p_bulk - float(fr_bulk)) / float(fr_bulk)
    rel_mid = abs(p_mid - float(fr_mid)) / float(fr_mid)
    rel_deep = abs(p_deep - float(fr_deep)) / float(fr_deep)
    # exact rational total mass: the integer numerators must sum to den^N EXACTLY
    _num, _off, _den = exact_integer_aggregate(pf.int_units)
    rational_mass_exact = (sum(_num) == _den)

    z_bulk = abs(mc["estimate"][0] - p_bulk) / max(mc["stderr"][0], 1e-300)
    z_mid = abs(mc["estimate"][1] - p_mid) / max(mc["stderr"][1], 1e-300)
    if verbose:
        print(f"    (b) HEADLINE: {len(pf)} heterogeneous units, "
              f"support size {ai['support_size']}, max bond dimension "
              f"{ai['max_bond_dimension']}, treewidth 1")
        print(f"          exact aggregate built in {ai['time_s']*1e3:.1f} ms, "
              f"{ai['cost_multiply_adds']} multiply-adds; sum(pmf)-1 = "
              f"{float(agg.pmf.sum())-1.0:+.3e}")
        print(f"          Monte Carlo: {n_samples:,} samples, seed 1, "
              f"{mc['time_s']:.1f} s, largest sum ever sampled S={mc['max_sample_sum']}")
        print(f"          POSITIVE CONTROL (bulk, both methods must agree):")
        print(f"            S >= {t_bulk:3d}   exact {p_bulk:.6e}   "
              f"MC {mc['estimate'][0]:.6e} +- {mc['stderr'][0]:.1e}  "
              f"({mc['hits'][0]:,} hits, {z_bulk:.2f} sigma)")
        print(f"            S >= {t_mid:3d}   exact {p_mid:.6e}   "
              f"MC {mc['estimate'][1]:.6e} +- {mc['stderr'][1]:.1e}  "
              f"({mc['hits'][1]:,} hits, {z_mid:.2f} sigma)")
        print(f"          DEEP TAIL (this is the claim):")
        print(f"            S >= {t_deep:3d}   exact {p_deep:.6e}")
        print(f"            S >= {t_deep:3d}   MC    {mc['estimate'][2]:.1f}  "
              f"EXACTLY ZERO -- {mc['hits'][2]} hits in {n_samples:,} samples")
        print(f"            all MC can say at 95%: p < 3/n = "
              f"{mc['rule_of_three_bound']:.3e}, which is "
              f"{math.log10(mc['rule_of_three_bound']/p_deep):.0f} orders of magnitude "
              f"above the true value")
        print(f"          GROUND TRUTH for the deep tail (exact bigint rational "
              f"arithmetic, no floats):")
        print(f"            exact rational P(S >= {t_deep}) = {float(fr_deep):.12e}")
        print(f"            float64 log-space           = {p_deep:.12e}   "
              f"relative error {rel_deep:.3e}")
        print(f"            (bulk rel err {rel_bulk:.3e}, mid rel err {rel_mid:.3e})")
        print(f"            exact rational total mass sum(numerators) == "
              f"denominator^{len(pf)} : {rational_mass_exact}   (integer identity, "
              f"no float64 anywhere)")
        print(f"          max |log_tail_curve - log_tail_probability| over all "
              f"{len(agg)} thresholds: {e_curve:.3e}")
    res["b"] = {"n_units": len(pf), "t_bulk": t_bulk, "t_mid": t_mid, "t_deep": t_deep,
                "p_bulk": p_bulk, "p_mid": p_mid, "p_deep": p_deep,
                "mc_bulk": float(mc["estimate"][0]), "mc_mid": float(mc["estimate"][1]),
                "mc_deep": float(mc["estimate"][2]),
                "mc_hits": [int(h) for h in mc["hits"]],
                "n_samples": n_samples, "z_bulk": z_bulk, "z_mid": z_mid,
                "mc_max_sample_sum": mc["max_sample_sum"],
                "rule_of_three": mc["rule_of_three_bound"],
                "exact_rational_deep": float(fr_deep),
                "rel_err_deep_vs_rational": rel_deep,
                "rel_err_bulk_vs_rational": rel_bulk,
                "rel_err_mid_vs_rational": rel_mid,
                "aggregate_info": ai, "mc_time_s": mc["time_s"],
                "rational_mass_exact": bool(rational_mass_exact),
                "max_err_tail_curve": e_curve}

    # ---- (c) numerical care ----------------------------------------------------------
    a_dir = aggregate_distribution(pf.units, method="direct")
    a_fft = aggregate_distribution(pf.units, method="fft")
    ex = agg.pmf
    bands = [(1e-3, 1.0), (1e-6, 1e-3), (1e-12, 1e-6), (1e-20, 1e-12), (0.0, 1e-20)]
    band_rows = []
    for lo_b, hi_b in bands:
        m = (ex > lo_b) & (ex <= hi_b)
        if not m.any():
            continue
        rf = float(np.max(np.abs(a_fft.pmf[m] - ex[m]) / ex[m]))
        rd = float(np.max(np.abs(a_dir.pmf[m] - ex[m]) / ex[m]))
        band_rows.append({"lo": lo_b, "hi": hi_b, "n": int(m.sum()),
                          "rel_fft": rf, "rel_direct": rd})
    bulk_mask = ex > 1e-6
    deep_mask = (ex > 0) & (ex < 1e-20)
    rel_fft_bulk = float(np.max(np.abs(a_fft.pmf[bulk_mask] - ex[bulk_mask])
                                / ex[bulk_mask]))
    rel_fft_deep = float(np.max(np.abs(a_fft.pmf[deep_mask] - ex[deep_mask])
                                / ex[deep_mask])) if deep_mask.any() else float("nan")
    rel_dir_bulk = float(np.max(np.abs(a_dir.pmf[bulk_mask] - ex[bulk_mask])
                                / ex[bulk_mask]))
    rel_dir_deep = float(np.max(np.abs(a_dir.pmf[deep_mask] - ex[deep_mask])
                                / ex[deep_mask])) if deep_mask.any() else float("nan")
    n_neg_fft = int(np.count_nonzero(a_fft.pmf < 0))
    fft_deep = float(a_fft.pmf[t_deep - a_fft.offset])
    dir_deep = float(a_dir.pmf[t_deep - a_dir.offset])
    fft_noise = float(np.max(np.abs(a_fft.pmf[a_fft.pmf < 0]))) if n_neg_fft else 0.0
    fft_abs_floor = float(np.finfo(np.float64).eps * float(ex.max()))
    # close the loop: the LINEAR convolution's deep tail against the exact rational too
    rel_dir_rational = abs(tail_probability(a_dir, t_deep) - float(fr_deep)) / float(fr_deep)
    rel_fft_rational = abs(tail_probability(a_fft, t_deep) - float(fr_deep)) / float(fr_deep)

    # a portfolio deep enough that LINEAR space cannot represent the answer at all
    pf_big = demo_portfolio(n_units=600, seed=7)
    t_lin0 = time.perf_counter()
    a_big_lin = aggregate_distribution(pf_big.units, method="direct")
    t_lin = time.perf_counter() - t_lin0
    t_log0 = time.perf_counter()
    a_big_log = aggregate_distribution(pf_big.units, method="log")
    t_log = time.perf_counter() - t_log0
    curve_big = log_tail_curve(a_big_log)
    target_big = -400.0 * math.log(10.0)
    t_big = int(a_big_log.support[int(np.argmin(np.abs(curve_big - target_big)))])
    lin_big = tail_probability(a_big_lin, t_big)
    log_big = log_tail_probability(a_big_log, t_big)
    # deepest threshold with any mass at all, in log space
    t_max_big = int(a_big_log.offset + len(a_big_log) - 1)
    log_max_big = float(curve_big[-1])
    fl = underflow_report()

    if verbose:
        print(f"    (c) numerical care -- MEASURED, not asserted")
        print(f"          max RELATIVE error vs exact log-space pmf, by size of the "
              f"probability being computed:")
        print(f"            {'band':>22}  {'entries':>7}  {'FFT conv':>11}  "
              f"{'direct conv':>11}")
        for r in band_rows:
            print(f"            {r['lo']:>9.0e} .. {r['hi']:<9.0e}  {r['n']:>7}  "
                  f"{r['rel_fft']:>11.2e}  {r['rel_direct']:>11.2e}")
        print(f"          FFT pmf entries that are NEGATIVE: {n_neg_fft} of "
              f"{a_fft.pmf.size}, largest magnitude {fft_noise:.3e} "
              f"(a probability cannot be negative; that IS the round-off floor)")
        print(f"          at S = {t_deep}: exact pmf {ex[t_deep-agg.offset]:.6e} | "
              f"direct {dir_deep:.6e} | FFT {fft_deep:+.6e}  <- FFT is pure noise here")
        print(f"          P(S >= {t_deep}) vs the EXACT RATIONAL: log-space rel err "
              f"{rel_deep:.3e} | direct rel err {rel_dir_rational:.3e} | FFT rel err "
              f"{rel_fft_rational:.3e}")
        print(f"          REPRESENTABLE FLOOR")
        print(f"            float64 smallest normal    {fl['float64_smallest_normal']:.6e}"
              f"   smallest subnormal {fl['float64_smallest_subnormal']:.6e}")
        print(f"            direct (linear) convolution: no cancellation, so RELATIVE "
              f"accuracy holds to the subnormal floor {TINY_SUBNORMAL:.3e}, then 0")
        print(f"            FFT convolution: ABSOLUTE floor ~ eps*max(p) = "
              f"{fft_abs_floor:.3e}; everything below that is noise")
        print(f"            log space: floor is log p = {LOG_SPACE_FLOOR:.3e}, i.e. no "
              f"practical floor")
        print(f"          600-unit portfolio ({len(a_big_log)} support points, "
              f"log conv {t_log*1e3:.0f} ms, linear conv {t_lin*1e3:.0f} ms):")
        print(f"            S >= {t_big}:  log space log P = {log_big:.3f} -> P = 10^"
              f"{log_big/math.log(10):.1f}   |   linear P = {lin_big:.1f}  <- float64 "
              f"underflow, log space REQUIRED")
        print(f"            S >= {t_max_big} (the maximum): log P = {log_max_big:.3f} "
              f"-> P = 10^{log_max_big/math.log(10):.1f}")
    res["c"] = {"rel_fft_bulk": rel_fft_bulk, "rel_fft_deep": rel_fft_deep,
                "rel_direct_bulk": rel_dir_bulk, "rel_direct_deep": rel_dir_deep,
                "bands": band_rows,
                "n_negative_fft_entries": n_neg_fft, "fft_noise_floor": fft_noise,
                "fft_absolute_floor": fft_abs_floor,
                "rel_direct_vs_rational_deep": rel_dir_rational,
                "rel_fft_vs_rational_deep": rel_fft_rational,
                "fft_at_deep_threshold": fft_deep, "direct_at_deep_threshold": dir_deep,
                "big_n_units": len(pf_big), "big_threshold": t_big,
                "big_log_tail": log_big, "big_linear_tail": lin_big,
                "big_max_threshold": t_max_big, "big_log_tail_at_max": log_max_big,
                "big_time_log_s": t_log, "big_time_direct_s": t_lin,
                "floors": fl}

    # ---- (d) is that 1e-15 the ALGORITHM or the INPUTS? -------------------------------
    # In (b) each unit probability is k/10000, which is NOT a dyadic rational, so the
    # float64 unit pmfs already differ from the exact rationals by ~eps before any
    # convolution happens. Repeat with denominator 1024: every k/1024 is EXACTLY
    # representable in binary, so the float64 inputs equal the rational inputs bit for
    # bit and whatever error remains is the convolution's own.
    pf_dy = demo_portfolio(n_units=40, seed=20260829, denominator=1024)
    inputs_exact = all(Fraction(float(v)) == Fraction(n, 1024)
                       for u, iu in zip(pf_dy.units, pf_dy.int_units)
                       for v, n in zip(u.pmf, iu.num))
    a_dy = aggregate_distribution(pf_dy.units, method="log")
    c_dy = log_tail_curve(a_dy)
    t_dy = int(a_dy.support[int(np.argmin(np.abs(c_dy - math.log(1e-30))))])
    fr_dy = exact_rational_tail(pf_dy.int_units, t_dy)
    p_dy = _safe_exp(log_tail_probability(a_dy, t_dy))
    rel_dy = abs(p_dy - float(fr_dy)) / float(fr_dy)
    if verbose:
        print(f"    (d) algorithm error isolated (dyadic probabilities k/1024, so the "
              f"float64 inputs are BIT-EXACT rationals: {inputs_exact})")
        print(f"          S >= {t_dy}:  exact rational {float(fr_dy):.12e}   "
              f"float64 log-space {p_dy:.12e}")
        print(f"          relative error {rel_dy:.3e}  = {rel_dy/float(np.finfo(float).eps):.1f} "
              f"eps, accumulated over 40 convolutions and a {len(a_dy)}-term logsumexp")
    res["d"] = {"inputs_bit_exact": bool(inputs_exact), "threshold": t_dy,
                "exact_rational": float(fr_dy), "float64": p_dy, "rel_err": rel_dy}

    # ---- (e) the law -----------------------------------------------------------------
    bonds = ai["bond_dimensions"]
    if verbose:
        print(f"    (e) THE LAW: cost = d ** treewidth")
        print(f"          the running partial sum is the ONLY variable crossing any cut "
              f"-> treewidth 1")
        print(f"          bond dimension after each unit: {bonds[0]}, {bonds[1]}, "
              f"{bonds[2]}, ... , {bonds[-1]}  (= |support(S_k)|)")
        print(f"          cost = sum_k d_k^1 * m_k = {ai['cost_multiply_adds']} "
              f"multiply-adds for {ai['n_units']} units; a joint frontier would be "
              f"prod_i m_i ~ 10^"
              f"{sum(math.log10(len(u)) for u in pf.units):.0f} outcomes")
        print(f"          REM is not a quantum computer: this is cheap only because the "
              f"units are INDEPENDENT, i.e. the dependency graph is a chain.")

    ok = (e_log < 1e-14 and e_dir < 1e-14 and e_mass < 1e-14 and e_tail < 1e-14
          and e_var <= 1.0 and e_es < 1e-12 and e_mix < 1e-14 and e_curve < 1e-9
          and rational_mass_exact
          and rel_deep < 1e-10 and rel_bulk < 1e-12
          and int(mc["hits"][2]) == 0 and z_bulk < 5.0 and z_mid < 5.0
          and 1e-34 < p_deep < 1e-26
          and rel_fft_deep > 1e3 and rel_dir_deep < 1e-9
          and rel_dir_rational < 1e-10 and rel_fft_rational > 1e3
          and lin_big == 0.0 and math.isfinite(log_big)
          and inputs_exact and rel_dy < 1e-12)
    res["ok"] = bool(ok)
    if verbose:
        print(f"    OVERALL {'PASS' if ok else 'FAIL'}")
    return res


if __name__ == "__main__":
    verify()

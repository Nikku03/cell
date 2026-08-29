"""Tail risk: exact probabilities for rare events under DEPENDENT losses.

WHY THIS IS THE ONE PLACE EXACTNESS OBVIOUSLY BEATS SAMPLING.

Monte Carlo's relative error on an event of probability p is about sqrt((1-p)/(N p)). To
get 10% relative error on p = 1e-6 you need roughly 1e8 samples. Exact summation does not
care how rare the event is: it costs the same for p = 0.5 and for p = 1e-12. So the tail is
where an exact method has an argument that is not merely aesthetic.

THE CONSTRUCTION. n risk factors X_1..X_n, each discrete over `d` loss levels, coupled by a
dependency graph. The question is P(sum X_i >= threshold). Introduce running totals
S_0 = 0, S_i = S_(i-1) + X_i as variables, with a DETERMINISTIC factor on (S_(i-1), X_i, S_i).
Eliminating in the order X_1, S_1, X_2, S_2, ... contracts the whole thing, and the width is
set by the dependency graph plus one for the running total. Cost

    n_bins  x  d ** (treewidth of the dependency graph + 1)

Linear in how finely the loss axis is discretized, exponential in how tangled the
dependencies are. Both halves of that are MEASURED below, not asserted.

WHAT THIS IS NOT. The loss axis is DISCRETIZED. A continuous-loss model computed on a grid
of bins is an approximation of the continuous problem no matter how exactly the discrete
sum is performed, and refining the grid costs linearly. Every number here is exact FOR THE
DISCRETE MODEL; the discretization error is a separate quantity and this module does not
pretend to have removed it. Nor is REM a quantum computer: with a densely dependent
portfolio the treewidth is n and the cost is d^n, which is the wall, and T6 measures it.

WHAT verify() MUST SHOW -- PREDECLARED, BEFORE ANY NUMBER IS RUN.
  T1  the exact loss distribution by elimination vs explicit enumeration of every joint
      outcome. GATE: max |difference| < 1e-12 over the whole distribution.
  T2  the distribution is a distribution. GATE: |sum - 1| < 1e-12 and no negative mass.
  T3  MONTE CARLO, the thing being beaten, measured rather than assumed. Report MC's
      relative error against the exact tail at several sample sizes and depths.
      GATE: at the deepest tail reported, MC with 1e5 samples must have relative error
      above 10%. If sampling were accurate there, the case for exactness would be weak and
      this module should say so instead of claiming a win.
  T4  POSITIVE CONTROL. With every dependency factor removed, the exact answer must equal
      the independent convolution of the marginals, computed by a completely separate code
      path. GATE: max |difference| < 1e-12.
  T5  cost vs DISCRETIZATION at fixed dependency structure. GATE: fitted log-log slope of
      time vs n_bins in [0.6, 1.6] -- linear, as claimed.
  T6  THE WALL. Treewidth and cost as the dependency graph goes from a chain to complete.
      GATE: the treewidth must grow at least linearly in n for the complete graph (slope
      >= 0.8), so the module demonstrates its own limit rather than only its strength.
"""
from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rem.factorgraph import FactorGraph, logsumexp

LOG0 = -1e300          # a log-probability standing in for zero


def _log(p: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(p), LOG0, dtype=float)
    p = np.asarray(p, dtype=float)
    np.log(p, out=out, where=p > 0)
    return out


def make_portfolio(n: int, d: int = 4, topology: str = "chain", seed: int = 0,
                   coupling: float = 0.8) -> dict:
    """n risk factors over d discrete loss levels 0..d-1, plus a dependency graph.

    topology "chain"      X_i couples to X_(i+1) only -- treewidth 1
             "complete"   every pair couples -- treewidth n-1
             "none"       no coupling at all; used by the T4 positive control
    """
    rng = np.random.default_rng(seed)
    marg = rng.dirichlet(np.full(d, 2.0), size=n)          # each factor's own law
    edges: List[Tuple[int, int]] = []
    if topology == "chain":
        edges = [(i, i + 1) for i in range(n - 1)]
    elif topology == "complete":
        edges = list(itertools.combinations(range(n), 2))
    elif topology != "none":
        raise ValueError(f"unknown topology {topology!r}")
    tabs = {e: coupling * rng.normal(size=(d, d)) for e in edges}
    return {"n": n, "d": d, "marginals": marg, "edges": edges, "tables": tabs,
            "topology": topology}


def _risk_graph(pf: dict) -> FactorGraph:
    """Just the risk factors and their dependencies; log-weights, unnormalized."""
    g = FactorGraph()
    for i in range(pf["n"]):
        g.add_var(f"x{i}", pf["d"])
        g.add_factor([f"x{i}"], _log(pf["marginals"][i]))
    for (i, j), tab in pf["tables"].items():
        g.add_factor([f"x{i}", f"x{j}"], tab)
    return g


def loss_distribution(pf: dict, n_bins: Optional[int] = None) -> dict:
    """P(total loss = k) for every k, EXACTLY, by elimination with a running-total variable.

    The running total S_i is a variable of cardinality n_bins and the factor on
    (S_(i-1), X_i, S_i) is deterministic, so the elimination width is the dependency
    treewidth plus one. The distribution is obtained by clamping the final total to each
    value in turn -- n_bins eliminations, hence the linear dependence on discretization
    that T5 measures.
    """
    n, d = pf["n"], pf["d"]
    total_max = n * (d - 1)
    n_bins = total_max + 1 if n_bins is None else int(n_bins)
    t0 = time.perf_counter()

    g = FactorGraph()
    for i in range(n):
        g.add_var(f"x{i}", d)
        g.add_factor([f"x{i}"], _log(pf["marginals"][i]))
    for (i, j), tab in pf["tables"].items():
        g.add_factor([f"x{i}", f"x{j}"], tab)
    for i in range(n):
        g.add_var(f"s{i}", n_bins)
    # s0 = x0
    t = np.full((d, n_bins), LOG0)
    for a in range(d):
        if a < n_bins:
            t[a, a] = 0.0
    g.add_factor(["x0", "s0"], t)
    for i in range(1, n):
        t = np.full((n_bins, d, n_bins), LOG0)
        for s in range(n_bins):
            for a in range(d):
                if s + a < n_bins:
                    t[s, a, s + a] = 0.0
        g.add_factor([f"s{i-1}", f"x{i}", f"s{i}"], t)

    logZ, _a, info = g.eliminate("sum")
    # clamp the final total to each bin: one extra unary per value
    logp = np.empty(n_bins)
    for k in range(n_bins):
        gk = FactorGraph()
        for v, c in g.cards.items():
            gk.add_var(v, c)
        for f in g.factors:
            gk.add_factor(list(f.vars), f.table)
        pin = np.full(n_bins, LOG0); pin[k] = 0.0
        gk.add_factor([f"s{n-1}"], pin)
        logp[k] = gk.eliminate("sum")[0]
    p = np.exp(logp - logZ)
    return {"pmf": p, "logZ": float(logZ), "treewidth": int(info["treewidth"]),
            "n_bins": n_bins, "seconds": time.perf_counter() - t0,
            "largest_table": int(info["largest_table"])}


def tail_probability(pmf: np.ndarray, threshold: int) -> float:
    return float(pmf[threshold:].sum())


def brute_force_pmf(pf: dict) -> np.ndarray:
    """Explicit enumeration of every joint outcome. The reference; exponential."""
    n, d = pf["n"], pf["d"]
    pmf = np.zeros(n * (d - 1) + 1)
    lm = [_log(pf["marginals"][i]) for i in range(n)]
    tot = []
    for combo in itertools.product(range(d), repeat=n):
        w = sum(lm[i][combo[i]] for i in range(n))
        for (i, j), tab in pf["tables"].items():
            w += tab[combo[i], combo[j]]
        tot.append((sum(combo), w))
    ws = np.array([w for _s, w in tot])
    ws = np.exp(ws - ws.max()); ws /= ws.sum()
    for (s, _w), wi in zip(tot, ws):
        pmf[s] += wi
    return pmf


def independent_pmf(pf: dict) -> np.ndarray:
    """Convolution of the marginals. A completely separate code path, for T4."""
    pmf = np.array([1.0])
    for i in range(pf["n"]):
        pmf = np.convolve(pmf, pf["marginals"][i])
    return pmf


def monte_carlo_tail(pf: dict, threshold: int, n_samples: int, seed: int = 0) -> float:
    """Sample the portfolio and estimate the tail. Requires an independent portfolio --
    with dependencies there is no exact sampler without first doing the exact work, which
    is itself part of the point."""
    rng = np.random.default_rng(seed)
    tot = np.zeros(n_samples, dtype=int)
    for i in range(pf["n"]):
        tot += rng.choice(pf["d"], size=n_samples, p=pf["marginals"][i])
    return float((tot >= threshold).mean())


def _slope(xs, ys) -> float:
    return float(np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)[0])


def verify(verbose: bool = True) -> dict:
    """Run T1-T6. Bars are fixed in the module docstring, above, before any number."""
    say = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    out: Dict[str, object] = {}

    pf = make_portfolio(6, d=4, topology="chain", seed=0)
    r = loss_distribution(pf)
    bf = brute_force_pmf(pf)
    d1 = float(np.abs(r["pmf"] - bf).max())
    out["T1_err"], out["T1"] = d1, d1 < 1e-12
    say(f"  T1 exact pmf, elimination vs enumeration of all {pf['d']**pf['n']:,} outcomes")
    say(f"      max |diff| {d1:.3e}   treewidth {r['treewidth']}   "
        f"largest table {r['largest_table']:,}   {'PASS' if out['T1'] else 'FAIL'}")

    tot = float(r["pmf"].sum()); neg = float(r["pmf"].min())
    out["T2"] = abs(tot - 1.0) < 1e-12 and neg >= -1e-15
    say(f"\n  T2 it is a distribution: sum {tot:.15f}, min mass {neg:.3e}   "
        f"{'PASS' if out['T2'] else 'FAIL'}")

    # ---- T3: Monte Carlo, measured -----------------------------------------------------
    say("\n  T3 Monte Carlo vs exact in the tail (independent portfolio, so MC is valid)")
    ipf = make_portfolio(12, d=4, topology="none", seed=3)
    ex = independent_pmf(ipf)
    say(f"      {'threshold':>9s} {'exact p':>12s} " +
        " ".join(f"{'N=' + f'{N:.0e}':>16s}" for N in (1e4, 1e5, 1e6)))
    deep_err = None
    for th in (20, 26, 30, 33):
        p = float(ex[th:].sum())
        cells = []
        for N in (1e4, 1e5, 1e6):
            mc = monte_carlo_tail(ipf, th, int(N), seed=7)
            rel = abs(mc - p) / p if p > 0 else float("nan")
            cells.append(f"{mc:.2e}({rel*100:5.1f}%)")
            if th == 33 and N == 1e5:
                deep_err = rel
        say(f"      {th:9d} {p:12.3e} " + " ".join(f"{c:>16s}" for c in cells))
    out["T3_deep_rel_err"] = float(deep_err) if deep_err is not None else None
    out["T3"] = bool(deep_err is not None and deep_err > 0.10)
    say(f"      MC relative error at the deepest tail with 1e5 samples: "
        f"{deep_err*100:.1f}% (bar > 10%)   {'PASS' if out['T3'] else 'FAIL'}")
    say("      exact costs the same whatever p is; MC's cost scales as 1/p")

    # ---- T4: POSITIVE CONTROL, independence --------------------------------------------
    ipf6 = make_portfolio(6, d=4, topology="none", seed=0)
    r4 = loss_distribution(ipf6)
    d4 = float(np.abs(r4["pmf"] - independent_pmf(ipf6)).max())
    out["T4_err"], out["T4"] = d4, d4 < 1e-12
    say(f"\n  T4 POSITIVE CONTROL: with dependencies removed, elimination must equal the "
        f"convolution\n      max |diff| {d4:.3e}   {'PASS' if out['T4'] else 'FAIL'}")

    # ---- T5: linear in discretization ----------------------------------------------------
    say("\n  T5 cost vs DISCRETIZATION (n_bins) at fixed structure")
    bins, secs = [], []
    base = make_portfolio(6, d=4, topology="chain", seed=1)
    for nb in (19, 38, 76, 152):
        rr = loss_distribution(base, n_bins=nb)
        bins.append(nb); secs.append(rr["seconds"])
        say(f"      n_bins {nb:4d}   {rr['seconds']*1e3:8.1f} ms   "
            f"largest table {rr['largest_table']:,}")
    s5 = _slope(np.log10(bins), np.log10(secs))
    out["T5_slope"], out["T5"] = s5, 0.6 <= s5 <= 1.6
    say(f"      log-log slope {s5:.3f} (bar 0.6-1.6, i.e. linear)   "
        f"{'PASS' if out['T5'] else 'FAIL'}")

    # ---- T6: THE WALL ---------------------------------------------------------------------
    say("\n  T6 THE WALL: treewidth as the dependency graph goes chain -> complete")
    ns, tw_chain, tw_comp = [], [], []
    for n in (4, 6, 8, 10, 12):
        ns.append(n)
        tw_chain.append(_risk_graph(make_portfolio(n, 4, "chain", 0)).treewidth())
        tw_comp.append(_risk_graph(make_portfolio(n, 4, "complete", 0)).treewidth())
        say(f"      n={n:3d}   chain tw {tw_chain[-1]:2d} (cost 4^{tw_chain[-1]}"
            f" = {4.0**tw_chain[-1]:.1e})   complete tw {tw_comp[-1]:2d} "
            f"(cost 4^{tw_comp[-1]} = {4.0**tw_comp[-1]:.1e})")
    s6 = _slope(ns, tw_comp)
    out["T6_slope"], out["T6"] = s6, s6 >= 0.8
    say(f"      slope of complete-graph treewidth vs n: {s6:.3f} (bar >= 0.8)   "
        f"{'PASS' if out['T6'] else 'FAIL'}")

    gates = ["T1", "T2", "T3", "T4", "T5", "T6"]
    out["all_pass"] = all(bool(out[k]) for k in gates)
    say(f"\n  {'ALL GATES PASS' if out['all_pass'] else 'GATE FAILURE'}: "
        + "  ".join(f"{k}={'pass' if out[k] else 'FAIL'}" for k in gates))
    say("\n  THE HONEST SUMMARY: exactness wins in the tail because its cost is independent"
        "\n  of the event's rarity while Monte Carlo's scales as 1/p. It wins only while the"
        "\n  dependency treewidth stays small; a densely dependent portfolio costs d^n and"
        "\n  T6 measures that. And every number is exact FOR THE DISCRETE MODEL -- the"
        "\n  discretization of a continuous loss axis is a separate error, not removed here.")
    return out


if __name__ == "__main__":
    verify()

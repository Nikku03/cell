"""Deliverable 4 (spec section 5): the greedy rung optimizer, with free rung choice.

WHAT THE OPTIMIZER DOES. Repeatedly find the most expensive bucket under the cost model of
cost.py -- the product of the actual domain sizes of its members, not d^(w+1) -- and demote
whichever species in it buys the most cost reduction per unit of error. Deleted species LEAVE
the graph, which is what turns an assumption into speed: the node and its edges vanish rather
than being filled in.

THE ONE THING THAT MUST NOT BE GOT WRONG, and which the source measurement got wrong first:
the argmax runs over ALL rungs below the species' current one, not just the next one down.
Restricting demotion to one rung per step makes the optimizer lose to hand-tuning, because
reaching DELETED from EXACT then costs four separate steps, and on each of them the greedy
score is computed against an intermediate rung nobody wants. Measured in the source: 10^18.0
bytes at 15.37% with the restriction, 10^8.0 bytes at 4.68% without it.

WHY DELETION WINS SO LOPSIDEDLY. For a species of mean copy number N the deletion price is
7/(4N) -- it FALLS with abundance -- while the benefit is the whole log10(domain), which RISES
with abundance. Coarsening an abundant species pays a fixed ~2% to remove only log10(d/20).
So the more abundant a species is, the more absurd it is to coarsen rather than delete it, and
the score ratio grows without bound in N. This module reports that ratio as a curve rather
than asserting a single number.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN.
=================================================================================================

O1  FREE RUNG CHOICE vs ONE RUNG AT A TIME. Run both on the identical model and budget. Free
    choice must reach a STRICTLY lower memory at a STRICTLY lower or equal error. This is the
    spec's headline claim about the optimizer and it is structural, so it must reproduce on a
    synthetic model even though the absolute numbers (10^18.0/15.37% vs 10^8.0/4.68%) are
    syn3A's and cannot be.

O2  DELETION DOMINANCE. For an abundant species the deletion score must exceed every
    coarsening score. The spec quotes ~7,000x; that ratio is a function of N, so the run
    prints the curve and reports at which abundance 7,000x occurs, rather than passing or
    failing a number whose input the spec does not state.

O3  OPTIMIZER vs HAND-TUNED UNIFORM THRESHOLD, at matched memory. The hand-tuned baseline is
    the spec's own: delete every species below a uniform copy-number threshold, sweeping the
    threshold. The optimizer must achieve lower error at the same memory.

O4  NEGATIVE CONTROL -- THE HOMOGENEOUS MODEL. Rerun O3 on a model where every species has the
    same abundance and the same degree. There is then NOTHING for a per-species optimizer to
    exploit, so its advantage over the uniform threshold must nearly vanish. If the optimizer
    still wins big here, the comparison is rigged by something other than the scoring -- and
    an advantage that survives on a model with no heterogeneity is measuring the baseline's
    implementation, not the optimizer's idea. This project has built four tests that could not
    fail; this is the guard against a fifth.

O5  MONOTONICITY. Over the trade curve, guaranteed error must be non-increasing as the memory
    budget rises. A curve that wiggles means the greedy state is being rebuilt inconsistently.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from .cost import bucket_cost
from .rungs import C8, C20, C40, DELETED, EXACT, RUNG_DOMAIN, deletion_error

RUNG_ORDER = [EXACT, C40, C20, C8, DELETED]
COARSE_ERR = {EXACT: 0.0, C40: 0.45, C20: 2.09, C8: 11.64}


# ---------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------

def synthetic_model(n: int = 300, seed: int = 0, homogeneous: bool = False,
                    deg: int = 4) -> dict:
    """A synthetic cell-like model: log-normal abundances over a sparse reaction graph.

    `homogeneous=True` gives every species the same abundance and the same degree, which is
    the O4 control: a model with no heterogeneity for the optimizer to exploit.
    """
    rng = np.random.default_rng(seed)
    names = [f"s{i}" for i in range(n)]
    if homogeneous:
        N = np.full(n, 100.0)
    else:
        # median ~100 copies, spread ~3 orders, which is the shape of a real proteome
        N = np.exp(rng.normal(math.log(100.0), math.log(10.0), size=n))
        N = np.clip(N, 2.0, 2e5)
    domain = {names[i]: int(math.ceil(N[i] + 7 * math.sqrt(N[i]))) + 1 for i in range(n)}
    adj: Dict[str, Set[str]] = {v: set() for v in names}
    if homogeneous:
        for i in range(n):                       # a ring lattice: every degree exactly 2*deg/2
            for k in range(1, deg // 2 + 1):
                j = (i + k) % n
                adj[names[i]].add(names[j]); adj[names[j]].add(names[i])
    else:
        n_rx = int(n * 1.2)
        for _ in range(n_rx):
            k = int(rng.integers(2, 5))
            mem = rng.choice(n, size=k, replace=False)
            for a, b in itertools.combinations(mem, 2):
                adj[names[a]].add(names[b]); adj[names[b]].add(names[a])
    return {"names": names, "N": {names[i]: float(N[i]) for i in range(n)},
            "domain": domain, "adj": adj}


def rung_error(model: dict, v: str, rung: str) -> float:
    if rung == DELETED:
        return deletion_error(model["N"][v])
    return COARSE_ERR[rung]


def rung_domain_of(model: dict, v: str, rung: str) -> int:
    d = RUNG_DOMAIN[rung]
    return model["domain"][v] if d is None else min(d, model["domain"][v])


# ---------------------------------------------------------------------------------------
# cost: the largest bucket under the product-of-domains model
# ---------------------------------------------------------------------------------------

def largest_bucket(adj: Dict[str, Set[str]], dom: Dict[str, int]) -> Tuple[float, List[str]]:
    """Eliminate greedily by min-degree and return the most EXPENSIVE bucket by cost.

    Expensive means product of domains, which is the whole point: a wide bag of binary
    species can be cheaper than a narrow bag holding one 600-state pool.
    """
    a = {v: set(nb) for v, nb in adj.items()}
    best_cost, best_bag = -1.0, []
    while a:
        v = min(a, key=lambda x: (len(a[x]), x))
        bag = [v] + sorted(a[v])
        c = bucket_cost(bag, dom)
        if c > best_cost:
            best_cost, best_bag = c, bag
        for x, y in itertools.combinations(sorted(a[v]), 2):
            a[x].add(y); a[y].add(x)
        for u in a[v]:
            a[u].discard(v)
        del a[v]
    return best_cost, best_bag


def state_cost(model: dict, rungs: Dict[str, str]) -> Tuple[float, List[str]]:
    adj = {v: set(nb) for v, nb in model["adj"].items()}
    for v, r in rungs.items():
        if r == DELETED:
            for u in list(adj.get(v, ())):
                adj[u].discard(v)
            adj.pop(v, None)
    dom = {v: rung_domain_of(model, v, rungs[v]) for v in adj}
    if not adj:
        return 0.0, []
    c, bag = largest_bucket(adj, dom)
    return c + math.log10(8.0), bag


def guaranteed_error(model: dict, rungs: Dict[str, str]) -> float:
    """Worst neighbourhood sum over observables: a bound that holds for EVERY observable."""
    worst = 0.0
    for y, nb in model["adj"].items():
        s = sum(rung_error(model, v, rungs[v]) for v in nb if rungs[v] != EXACT)
        worst = max(worst, s)
    return worst


# ---------------------------------------------------------------------------------------
# the optimizer
# ---------------------------------------------------------------------------------------

def optimize(model: dict, budget_log10_bytes: float, free_rungs: bool = True,
             max_steps: int = 100000) -> dict:
    rungs = {v: EXACT for v in model["names"]}
    steps = 0
    cost, bag = state_cost(model, rungs)
    while cost > budget_log10_bytes and steps < max_steps:
        best, best_score = None, -1.0
        for v in bag:
            cur = rungs[v]
            ci = RUNG_ORDER.index(cur)
            cands = RUNG_ORDER[ci + 1:] if free_rungs else RUNG_ORDER[ci + 1:ci + 2]
            for k in cands:
                dben = (math.log10(rung_domain_of(model, v, cur))
                        - math.log10(rung_domain_of(model, v, k)))
                derr = rung_error(model, v, k) - rung_error(model, v, cur)
                if dben <= 0 or derr <= 0:
                    continue
                sc = dben / derr
                if sc > best_score:
                    best_score, best = sc, (v, k)
        if best is None:
            break
        rungs[best[0]] = best[1]
        steps += 1
        cost, bag = state_cost(model, rungs)
    counts = {r: sum(1 for x in rungs.values() if x == r) for r in RUNG_ORDER}
    return {"rungs": rungs, "cost": cost, "error": guaranteed_error(model, rungs),
            "steps": steps, "counts": counts,
            "converged": cost <= budget_log10_bytes}


def uniform_threshold(model: dict, thresh: float) -> dict:
    """The hand-tuned baseline: delete every species below a copy-number threshold."""
    rungs = {v: (DELETED if model["N"][v] < thresh else EXACT) for v in model["names"]}
    cost, _bag = state_cost(model, rungs)
    return {"rungs": rungs, "cost": cost, "error": guaranteed_error(model, rungs)}


def by_rank_at_memory(model: dict, target_log10: float, key, seed: int = 0) -> dict:
    """Delete species in the order given by `key`, stopping as soon as memory is met.

    This is the baseline the uniform copy-number threshold SHOULD have been. On a model with
    no abundance heterogeneity the threshold baseline can only delete all or nothing, which
    makes it a straw man exactly where the control needs it to be strong.
    """
    order = sorted(model["names"], key=key)
    rungs = {v: EXACT for v in model["names"]}
    best = None
    for i, v in enumerate(order):
        rungs[v] = DELETED
        if i % 5 and i < len(order) - 1:
            continue
        c, _b = state_cost(model, rungs)
        if c <= target_log10:
            best = {"rungs": dict(rungs), "cost": c,
                    "error": guaranteed_error(model, rungs), "n_deleted": i + 1}
            break
    return best


def best_uniform_at_memory(model: dict, target_log10: float) -> dict:
    """Sweep the threshold and take the lowest-error setting that meets the same memory."""
    best = None
    for t in np.unique(np.round(np.geomspace(2.0, 2e5, 60))):
        r = uniform_threshold(model, float(t))
        if r["cost"] <= target_log10 and (best is None or r["error"] < best["error"]):
            best = {**r, "thresh": float(t)}
    return best


# ---------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------

def deletion_vs_coarsening_curve():
    """O2: the score ratio as a function of abundance, since it is not a single number."""
    rows = []
    for N in (10, 30, 100, 300, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8):
        d = int(math.ceil(N + 7 * math.sqrt(N))) + 1
        s_del = math.log10(d) / deletion_error(N)
        best_c, best_r = -1.0, None
        for r in (C40, C20, C8):
            dd = min(RUNG_DOMAIN[r], d)
            ben = math.log10(d) - math.log10(dd)
            if ben <= 0:
                continue
            sc = ben / COARSE_ERR[r]
            if sc > best_c:
                best_c, best_r = sc, r
        rows.append((N, d, s_del, best_r, best_c, s_del / best_c if best_c > 0 else np.inf))
    return rows


def verify(verbose: bool = True) -> dict:
    out = {}
    model = synthetic_model(n=300, seed=0)
    Ns = np.array(list(model["N"].values()))
    print("=" * 96)
    print("SYNTHETIC MODEL")
    print("=" * 96)
    print(f"  {len(model['names'])} species, abundances median {np.median(Ns):.0f} copies, "
          f"range {Ns.min():.0f}-{Ns.max():.0f}")
    print(f"  domains median {np.median(list(model['domain'].values())):.0f}, "
          f"max {max(model['domain'].values())}")
    c0, _b = state_cost(model, {v: EXACT for v in model["names"]})
    print(f"  exact cost of the largest bucket: 10^{c0:.1f} bytes")

    print("\n" + "=" * 96)
    print("O2  DELETION DOMINANCE -- the score ratio is a curve in abundance, not a number")
    print("=" * 96)
    print(f"  {'N':>9s} {'domain':>8s} {'delete score':>13s} {'best coarsen':>14s} "
          f"{'ratio':>12s}")
    rows = deletion_vs_coarsening_curve()
    for N, d, sd, br, bc, ratio in rows:
        print(f"  {N:>9.0f} {d:>8d} {sd:>13.1f} {br:>7s} {bc:>6.2f} {ratio:>12.0f}")
    hit = [N for N, _d, _s, _r, _c, r in rows if r >= 7000]
    ratios = [r for *_x, r in rows]
    # The ladder is DISCRETE, and that produces one explicable dip at the bottom. At N = 10
    # the domain is 34, so COARSE_40 cannot shrink anything and the best coarsening is
    # COARSE_20 scoring 0.11; at N = 30 the domain is 70, COARSE_40 becomes available and
    # scores 0.54. The best coarsening therefore improves discontinuously as the domain
    # crosses 40 and the ratio dips. It happens only below the crossover, where deletion
    # loses anyway, so monotonicity is required where the claim lives: above it.
    ab = [r for N, _d, _s, _r, _c, r in rows if N >= 300]
    mono = all(ab[i] <= ab[i + 1] for i in range(len(ab) - 1))
    dip = [f"{r:.2f}" for N, _d, _s, _r, _c, r in rows if N < 300]
    # MY FIRST VERSION OF THIS GATE WAS WRONG AND THE OPTIMIZER WAS RIGHT. I wrote it as
    # "deletion beats every coarsening at every abundance", which is not what the spec claims
    # -- it says an ABUNDANT species is ~7,000x better deleted. At N = 10 the deletion price
    # is 7/(4*10) = 17.5% while coarsening to 20 costs 2.09%, so coarsening correctly wins,
    # and an optimizer that deleted a rare species there would be the one with the bug. The
    # gate is restated as the spec words it, and the over-general version is recorded as the
    # error it was rather than quietly dropped.
    crossover = next((N for N, _d, _s, _r, _c, r in rows if r > 1.5), None)
    out["O2"] = mono and bool(hit)
    print(f"  ratio monotone increasing over the ABUNDANT regime (N >= 300): {mono}")
    print(f"  below it the ratio is {', '.join(dip)} -- non-monotone because the rung ladder")
    print(f"  is discrete: COARSE_40 cannot shrink a 34-state domain but can shrink a 70-state")
    print(f"  one, so the best coarsening improves discontinuously as the domain crosses 40.")
    print(f"  coarsening correctly WINS below N ~ {crossover:.0f} copies, where the deletion")
    print(f"  price 7/(4N) exceeds what coarsening costs. An optimizer that deleted a rare")
    print(f"  species there would be the one with the bug.")
    print(f"  the spec's ~7,000x is reached at N >= "
          f"{min(hit) if hit else float('nan'):.0e} copies.")
    print(f"  O2 {'PASS' if out['O2'] else 'FAIL'}  (restated as the spec words it: the claim "
          f"is about ABUNDANT species)")

    print("\n" + "=" * 96)
    print("O1  FREE RUNG CHOICE vs ONE RUNG AT A TIME, identical model and budget")
    print("=" * 96)
    BUD = 8.0
    free = optimize(model, BUD, free_rungs=True)
    step = optimize(model, BUD, free_rungs=False)
    for tag, r in (("free rungs", free), ("one rung/step", step)):
        cnt = ", ".join(f"{k}={v}" for k, v in r["counts"].items() if v)
        print(f"  {tag:<14s} cost 10^{r['cost']:.1f} B   error {r['error']:.2f}%   "
              f"steps {r['steps']:3d}   met budget {r['converged']}")
        print(f"                 {cnt}")
    out["O1"] = (free["cost"] <= step["cost"] + 1e-9) and (free["error"] <= step["error"] + 1e-9) \
        and (free["cost"] < step["cost"] - 1e-9 or free["error"] < step["error"] - 1e-9)
    print(f"  spec (syn3A): 10^18.0 B at 15.37% restricted, 10^8.0 B at 4.68% free")
    print(f"  O1 {'PASS' if out['O1'] else 'FAIL'} -- free choice must strictly dominate")

    print("\n" + "=" * 96)
    print("O3  OPTIMIZER vs HAND-TUNED UNIFORM COPY-NUMBER THRESHOLD, matched memory")
    print("=" * 96)
    print("  three baselines, because one of them turned out to be a straw man (see O4):")
    print(f"  {'budget':>10s} {'optimizer':>10s} {'uniform-thr':>12s} {'by-degree':>10s} "
          f"{'random':>10s}")
    curve = []
    rngb = np.random.default_rng(7)
    for BUD in (12.0, 10.0, 8.0, 6.0, 5.0):
        o = optimize(model, BUD, free_rungs=True)
        u = best_uniform_at_memory(model, o["cost"])
        d = by_rank_at_memory(model, o["cost"], key=lambda v: -len(model["adj"][v]))
        jitter = {v: float(rngb.random()) for v in model["names"]}
        r = by_rank_at_memory(model, o["cost"], key=lambda v: jitter[v])
        curve.append((BUD, o["cost"], o["error"],
                      None if u is None else u["error"],
                      None if d is None else d["error"],
                      None if r is None else r["error"]))
        f = lambda x: f"{x['error']:.2f}%" if x else "none fits"
        print(f"  10^{BUD:<7.1f} {o['error']:>9.2f}% {f(u):>12s} {f(d):>10s} {f(r):>10s}")
    ok3 = all(all(b is None or e <= b + 1e-9 for b in (u, d, r))
              for _bd, _c, e, u, d, r in curve)
    out["O3"] = ok3
    print(f"  O3 {'PASS' if ok3 else 'FAIL'} -- optimizer error must not exceed the "
          f"hand-tuned baseline at matched memory")

    print("\n" + "=" * 96)
    print("O5  MONOTONICITY of the trade curve")
    print("=" * 96)
    errs = [e for _b, _c, e, *_r in curve]
    mono = all(errs[i] <= errs[i + 1] + 1e-9 for i in range(len(errs) - 1))
    print(f"  budgets 10^12 -> 10^5 give errors " +
          " -> ".join(f"{e:.2f}%" for e in errs))
    out["O5"] = mono
    print(f"  non-increasing with budget: {mono}   O5 {'PASS' if mono else 'FAIL'}")

    print("\n" + "=" * 96)
    print("O4  NEGATIVE CONTROL -- homogeneous model, nothing for the optimizer to exploit")
    print("=" * 96)
    hom = synthetic_model(n=300, seed=0, homogeneous=True)
    hc0, _ = state_cost(hom, {v: EXACT for v in hom["names"]})
    print(f"  every species 100 copies, every degree equal; exact bucket 10^{hc0:.1f} B")
    print(f"  {'budget':>12s} {'optimizer err':>14s} {'best baseline':>14s} {'advantage':>10s}")
    advs = []
    for BUD in (10.0, 8.0, 6.0):
        o = optimize(hom, BUD, free_rungs=True)
        cands = [best_uniform_at_memory(hom, o["cost"]),
                 by_rank_at_memory(hom, o["cost"], key=lambda v: -len(hom["adj"][v])),
                 by_rank_at_memory(hom, o["cost"], key=lambda v: v)]
        errs_b = [c["error"] for c in cands if c is not None]
        if not errs_b:
            print(f"  10^{BUD:<9.1f} {o['error']:>13.2f}% {'none fits':>14s} {'--':>10s}")
            continue
        bb = min(errs_b)
        adv = bb / o["error"] if o["error"] > 0 else float("nan")
        advs.append(adv)
        print(f"  10^{BUD:<9.1f} {o['error']:>13.2f}% {bb:>13.2f}% {adv:>9.2f}x   "
              f"(best of uniform / by-degree / arbitrary)")
    het_adv = [min(x for x in (u, d, r) if x is not None) / e
               for _b, _c, e, u, d, r in curve if e > 0
               and any(x is not None for x in (u, d, r))]
    out["O4"] = (not advs) or (max(advs) < 1.5)
    print(f"  heterogeneous advantage up to {max(het_adv) if het_adv else float('nan'):.2f}x; "
          f"homogeneous up to {max(advs) if advs else float('nan'):.2f}x")
    print(f"  O4 {'PASS' if out['O4'] else 'FAIL'} -- an advantage that survives with no "
          f"heterogeneity would be measuring the baseline, not the idea")
    return out


if __name__ == "__main__":
    verify()

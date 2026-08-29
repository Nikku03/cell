"""Clustered topology -- cost flat in system size when clusters are small and local.

THE GOVERNING LAW, which this module exists to MEASURE rather than assert:

    cost = d ** treewidth        d = states per variable

Entanglement across a cut, bond dimension, edges crossing the cut and treewidth are the
same number. REM is NOT a quantum computer; nothing here breaks cryptography. What it does
do is exploit the one structural fact that actually pays: if a system is a chain of small
densely-coupled CLUSTERS joined by NARROW interfaces, then the treewidth is set by the
cluster and the interface, and is completely independent of how many clusters there are.
Adding clusters costs LINEAR time and ZERO extra width.

THE CONSTRUCTION (explicit, because the claim is only as good as the graph it is made on).

  Variables are named  c{k}v{i}  for cluster k = 0..K-1 and local slot i = 0..m-1, each
  with d states. Every variable gets a random unary factor, so none is isolated.

  INTERNAL (dense): every pair (i < j) inside a cluster gets a random d x d pairwise
  factor. Cluster k is therefore a CLIQUE on m vertices -- treewidth m-1 all by itself.
  This is deliberately the worst case locally: nothing about the flat scaling below comes
  from the clusters being internally easy.

  INTERFACE (narrow), topology="chain":
      ports of cluster k       A_k = local slots  m-s .. m-1   (the LAST s)
      ports of cluster k+1     B_k = local slots  0 .. s-1     (the FIRST s)
      interface_coupling="matching": s edges,  A_k[j] -- B_k[j]
      interface_coupling="complete": s*s edges, every A_k[j] -- B_k[j']
  Consecutive clusters share NOTHING else. s = interface_size is the number of edges (or
  the number of vertices, for "matching") crossing every cut, i.e. the bond dimension
  exponent of the chain.

  CONTRAST, topology="all-to-all": the ports are the FIRST s slots of every cluster and
  EVERY pair of clusters is coupled. With s = 1 the K port variables form a K-CLIQUE, so
  treewidth >= K-1 and the cost is d^(K-1). Same variables, same cluster internals, same
  cardinality -- only the wiring changed, and the cost goes from flat to exponential in K.
  Without this contrast "the treewidth stayed flat" would be unfalsifiable.

THE WIDTHS BELOW ARE EXACT, NOT HEURISTIC. Each cluster is a clique on m vertices, so
treewidth >= m-1 holds for EVERY ordering, by a lower bound that owes nothing to the greedy
search. For a matching interface of width s < m the greedy ordering ATTAINS m-1 (measured
for m = 3..6, s = 1..m-1, K = 2..16), so "treewidth 4" below is the treewidth, not an upper
bound on it.

TWO INDEPENDENT REFERENCES, because a check that reuses the code under test is worthless.

  1. brute_force(): naive enumeration over every joint assignment, written in PURE PYTHON
     (math.fsum / math.log, plain tuple indexing). It shares no code path with
     FactorGraph.eliminate -- no elimination order, no buckets, no numpy broadcasting.
     Exponential, so it only reaches ~12 variables.

  2. chain_reference(): a CLUSTER-LEVEL TRANSFER-MATRIX DP. It collapses each cluster into
     one super-variable with d^m states, builds the d^m x d^m interface matrices, and
     sweeps left to right in min-plus (mode="min") or log-sum-exp (mode="sum") algebra.
     This is a different algorithm at a different granularity -- no variable elimination
     order exists in it at all -- and it stays cheap where brute force cannot go, so the
     headline instances (160+ variables, 2^160 assignments) are checked against ground
     truth and not merely against themselves.
"""
from __future__ import annotations

import itertools
import math
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rem.factorgraph import FactorGraph

_NAME = re.compile(r"^c(\d+)v(\d+)$")


def var_name(cluster: int, slot: int) -> str:
    return f"c{cluster}v{slot}"


def parse_name(name: str) -> Tuple[int, int]:
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"{name!r} is not a clustered-graph variable name")
    return int(m.group(1)), int(m.group(2))


# --------------------------------------------------------------------------- build
def build_clustered_graph(n_clusters: int, cluster_size: int, d: int = 2,
                          interface_size: int = 1, seed: int = 0, *,
                          topology: str = "chain",
                          interface_coupling: str = "matching",
                          internal_scale: float = 1.0,
                          interface_scale: float = 1.0) -> FactorGraph:
    """Build the clustered factor graph described in the module docstring.

    n_clusters      K, how many clusters
    cluster_size    m, variables per cluster; each cluster is an internal CLIQUE
    d               states per variable
    interface_size  s, how wide the interface between coupled clusters is
    topology        "chain"      -- cluster k couples only to cluster k+1  (the claim)
                    "all-to-all" -- every pair of clusters couples        (the contrast)
    interface_coupling  "matching" (s edges) or "complete" (s*s edges)

    The graph attribute `spec` records every argument so downstream references can
    rebuild the structure without re-deriving it from the factor list.
    """
    if n_clusters < 1 or cluster_size < 1 or d < 2:
        raise ValueError("need n_clusters >= 1, cluster_size >= 1, d >= 2")
    if not (1 <= interface_size <= cluster_size):
        raise ValueError(f"interface_size must be in 1..cluster_size ({cluster_size})")
    if topology not in ("chain", "all-to-all"):
        raise ValueError("topology must be 'chain' or 'all-to-all'")
    if interface_coupling not in ("matching", "complete"):
        raise ValueError("interface_coupling must be 'matching' or 'complete'")

    rng = np.random.default_rng(seed)
    g = FactorGraph()
    for k in range(n_clusters):
        for i in range(cluster_size):
            g.add_var(var_name(k, i), d)

    # unary on every variable -- no isolated vertices, and the min is non-degenerate
    for k in range(n_clusters):
        for i in range(cluster_size):
            g.add_factor([var_name(k, i)], rng.normal(size=d))

    # INTERNAL: dense clique inside each cluster
    for k in range(n_clusters):
        for i, j in itertools.combinations(range(cluster_size), 2):
            g.add_factor([var_name(k, i), var_name(k, j)],
                         internal_scale * rng.normal(size=(d, d)))

    # INTERFACE
    s = interface_size

    def couple(kj: int, aslots: Sequence[int], kk: int, bslots: Sequence[int]):
        if interface_coupling == "matching":
            pairs = list(zip(aslots, bslots))
        else:
            pairs = [(a, b) for a in aslots for b in bslots]
        for a, b in pairs:
            g.add_factor([var_name(kj, a), var_name(kk, b)],
                         interface_scale * rng.normal(size=(d, d)))

    if topology == "chain":
        tail = list(range(cluster_size - s, cluster_size))   # ports of the left cluster
        head = list(range(s))                                # ports of the right cluster
        for k in range(n_clusters - 1):
            couple(k, tail, k + 1, head)
    else:                                                    # all-to-all contrast
        ports = list(range(s))
        for kj, kk in itertools.combinations(range(n_clusters), 2):
            couple(kj, ports, kk, ports)

    g.spec = {"n_clusters": n_clusters, "cluster_size": cluster_size, "d": d,
              "interface_size": s, "seed": seed, "topology": topology,
              "interface_coupling": interface_coupling}
    return g


def plant_ground_state(graph: FactorGraph, seed: int = 0,
                       bonus: float = 6.0) -> Dict[str, int]:
    """POSITIVE CONTROL. Plant a known global optimum in an existing clustered graph.

    A random assignment is drawn, and `bonus` is SUBTRACTED from the single entry of every
    factor that this assignment selects. Any deviation from it turns off the bonus in every
    factor touching a changed variable, and with bonus >> the O(1) spread of the random
    tables the planted assignment becomes the unique argmin. Returns it.

    This exists because rule 6 of the spec is real: a null result ("treewidth grew, cost
    exploded") is indistinguishable from a broken harness unless the SAME pipeline is shown
    to recover a signal that is definitely there. It also checks the argmin at sizes where
    brute force cannot go -- 160 variables, 3^160 assignments."""
    rng = np.random.default_rng(seed)
    planted = {v: int(rng.integers(graph.cards[v])) for v in graph.cards}
    for f in graph.factors:
        f.table[tuple(planted[v] for v in f.vars)] -= bonus
    return planted


# --------------------------------------------------------------------------- solve
def solve(graph: FactorGraph, mode: str = "min", order: Optional[Sequence[str]] = None,
          max_table: float = 2e8) -> dict:
    """Run FactorGraph.eliminate and report the value together with the cost law.

    Returns value, assignment, treewidth, the predicted cost d**treewidth, the largest
    intermediate table actually built, and the wall time split into ordering and
    elimination. Treewidth is logged for every instance because it, and not n_vars,
    is what predicts the cost."""
    t0 = time.perf_counter()
    if order is None:
        order, _ = graph.best_order()
    t_order = time.perf_counter() - t0

    t1 = time.perf_counter()
    value, assignment, info = graph.eliminate(mode, order=order, max_table=max_table)
    t_elim = time.perf_counter() - t1

    d = max(graph.cards.values()) if graph.cards else 1
    tw = info["treewidth"]
    return {"value": value, "assignment": assignment, "mode": mode,
            "treewidth": tw, "d": d, "cost_d_pow_tw": float(d) ** tw,
            "largest_table": info["largest_table"],
            "n_vars": info["n_vars"], "n_factors": info["n_factors"],
            "order_seconds": t_order, "elim_seconds": t_elim,
            "seconds": t_order + t_elim, "order": info["order"]}


# ------------------------------------------------------------- reference 1: brute force
def brute_force(graph: FactorGraph, mode: str = "min"):
    """Naive enumeration over every joint assignment. PURE PYTHON, no numpy reductions,
    no elimination order, no buckets -- genuinely independent of the code under test.
    Cost d^n_vars, so this is for tiny instances only."""
    names = list(graph.cards)
    tables = [(f.vars, f.table.tolist()) for f in graph.factors]
    pos = {v: i for i, v in enumerate(names)}

    def energy(combo) -> float:
        parts = []
        for vs, tab in tables:
            t = tab
            for v in vs:
                t = t[combo[pos[v]]]
            parts.append(t)
        return math.fsum(parts)

    ranges = [range(graph.cards[v]) for v in names]
    if mode == "min":
        best, arg = math.inf, None
        for combo in itertools.product(*ranges):
            e = energy(combo)
            if e < best:
                best, arg = e, dict(zip(names, combo))
        return best, arg
    if mode == "sum":
        vals = [energy(combo) for combo in itertools.product(*ranges)]
        mx = max(vals)
        return mx + math.log(math.fsum(math.exp(v - mx) for v in vals)), None
    raise ValueError("mode must be 'min' or 'sum'")


# ------------------------------------------- reference 2: cluster transfer-matrix DP
def _lse(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def chain_reference(graph: FactorGraph, mode: str = "min") -> float:
    """Cluster-level transfer-matrix DP for topology="chain".

    Each cluster becomes ONE super-variable with S = d^m states; the model is then a
    plain chain over K super-variables and is solved by a left-to-right sweep in min-plus
    or log-sum-exp algebra. No elimination order, no bucket, no factor pool -- a different
    algorithm at a different granularity from FactorGraph.eliminate, and cheap enough to
    check instances with 2^160 assignments that brute force cannot touch.

    Cost O(K * d^(2m)) -- exponential in the CLUSTER, linear in the number of clusters,
    which is the same governing law from the other side."""
    spec = getattr(graph, "spec", None)
    if spec is None or spec["topology"] != "chain":
        raise ValueError("chain_reference needs a graph from build_clustered_graph(topology='chain')")
    K, m, d = spec["n_clusters"], spec["cluster_size"], spec["d"]
    S = d ** m
    digits = np.array(list(itertools.product(range(d), repeat=m)), dtype=int)  # (S, m)

    local: List[List] = [[] for _ in range(K)]        # factors inside one cluster
    cross: Dict[int, List] = {}                       # factors spanning k, k+1
    for f in graph.factors:
        ks = sorted({parse_name(v)[0] for v in f.vars})
        if len(ks) == 1:
            local[ks[0]].append(f)
        elif len(ks) == 2 and ks[1] == ks[0] + 1 and len(f.vars) == 2:
            cross.setdefault(ks[0], []).append(f)
        else:
            raise ValueError(f"factor over {f.vars} is not chain-local")

    def local_vec(k: int) -> np.ndarray:
        out = np.zeros(S)
        for f in local[k]:
            slots = [parse_name(v)[1] for v in f.vars]
            idx = tuple(digits[:, i] for i in slots)
            out += f.table[idx]
        return out

    def cross_mat(k: int) -> np.ndarray:
        out = np.zeros((S, S))
        for f in cross.get(k, []):
            (ka, ia), (kb, ib) = (parse_name(v) for v in f.vars)
            tab = f.table if ka == k else f.table.T
            ia, ib = (ia, ib) if ka == k else (ib, ia)
            out += tab[digits[:, ia][:, None], digits[:, ib][None, :]]
        return out

    v = local_vec(0)
    for k in range(1, K):
        M = cross_mat(k - 1) + v[:, None]
        v = (np.min(M, axis=0) if mode == "min" else _lse(M, axis=0)) + local_vec(k)
    return float(v.min() if mode == "min" else _lse(v, axis=0))


# --------------------------------------------------------------------------- scaling
def scaling_table(n_clusters_list: Sequence[int] = (2, 4, 8, 16, 32),
                  cluster_size: int = 5, d: int = 3, interface_size: int = 2,
                  seed: int = 0, mode: str = "min", *,
                  topology: str = "chain", interface_coupling: str = "matching",
                  max_table: float = 2e8, repeats: int = 3,
                  verbose: bool = True, label: str = "") -> List[dict]:
    """Measure treewidth and wall time as n_clusters grows with cluster_size held fixed.

    Timing is best-of-`repeats` (minimum, not mean) so the numbers report the machine's
    capability and not its scheduling noise. Rows that exceed max_table are recorded with
    wall=True and the predicted cost d**treewidth, which is the honest way to report the
    treewidth wall -- it is the governing law biting, not a bug."""
    rows = []
    for K in n_clusters_list:
        g = build_clustered_graph(K, cluster_size, d, interface_size, seed,
                                  topology=topology,
                                  interface_coupling=interface_coupling)
        tw = g.treewidth()
        row = {"n_clusters": K, "cluster_size": cluster_size, "d": d,
               "interface_size": interface_size, "topology": topology,
               "interface_coupling": interface_coupling,
               "n_vars": len(g.cards), "n_factors": len(g.factors),
               "treewidth": tw, "cost_d_pow_tw": float(d) ** tw}
        best = None
        wall = False
        for _ in range(repeats):
            try:
                r = solve(g, mode, max_table=max_table)
            except MemoryError:
                wall = True
                break
            best = r if best is None else (r if r["seconds"] < best["seconds"] else best)
        if wall or best is None:
            row.update({"wall": True, "seconds": float("nan"),
                        "order_seconds": float("nan"), "elim_seconds": float("nan"),
                        "largest_table": None, "value": None})
        else:
            row.update({"wall": False, "seconds": best["seconds"],
                        "order_seconds": best["order_seconds"],
                        "elim_seconds": best["elim_seconds"],
                        "largest_table": best["largest_table"],
                        "value": best["value"]})
        row["sec_per_cluster"] = (row["seconds"] / K) if not row["wall"] else float("nan")
        rows.append(row)

    if verbose:
        head = label or f"{topology} / {interface_coupling}"
        print(f"    {head}:  cluster_size={cluster_size}  d={d}  "
              f"interface_size={interface_size}  mode={mode}")
        print(f"      {'K':>4} {'n_vars':>7} {'n_fac':>7} {'tw':>4} {'d^tw':>12} "
              f"{'max_table':>10} {'sec':>9} {'sec/K':>9}")
        for r in rows:
            if r["wall"]:
                print(f"      {r['n_clusters']:>4} {r['n_vars']:>7} {r['n_factors']:>7} "
                      f"{r['treewidth']:>4} {r['cost_d_pow_tw']:>12.3e} "
                      f"{'WALL':>10} {'--':>9} {'--':>9}")
            else:
                print(f"      {r['n_clusters']:>4} {r['n_vars']:>7} {r['n_factors']:>7} "
                      f"{r['treewidth']:>4} {r['cost_d_pow_tw']:>12.3e} "
                      f"{r['largest_table']:>10} {r['seconds']:>9.4f} "
                      f"{r['sec_per_cluster']:>9.5f}")
    return rows


def interface_scan(interface_sizes: Sequence[int] = (1, 2, 3, 4, 5, 6),
                   n_clusters: int = 6, cluster_size: int = 6, d: int = 2,
                   seed: int = 0, mode: str = "min",
                   interface_coupling: str = "complete",
                   verbose: bool = True) -> List[dict]:
    """Second contrast: hold the SIZE fixed and widen the interface. Same n_vars, same
    cluster_size, same K -- only the number of edges crossing each cut changes, and the
    treewidth (hence d^treewidth) climbs with it."""
    rows = []
    for s in interface_sizes:
        g = build_clustered_graph(n_clusters, cluster_size, d, s, seed,
                                  topology="chain", interface_coupling=interface_coupling)
        tw = g.treewidth()
        r = solve(g, mode)
        rows.append({"interface_size": s, "treewidth": tw, "d": d,
                     "cost_d_pow_tw": float(d) ** tw,
                     "largest_table": r["largest_table"], "n_vars": r["n_vars"],
                     "n_factors": r["n_factors"], "seconds": r["seconds"]})
    if verbose:
        print(f"    widening the interface (coupling={interface_coupling}): "
              f"K={n_clusters} clusters of {cluster_size}, d={d} -- size is CONSTANT")
        print(f"      {'s':>3} {'n_vars':>7} {'n_fac':>7} {'tw':>4} {'d^tw':>12} "
              f"{'max_table':>10} {'sec':>9}")
        for r in rows:
            print(f"      {r['interface_size']:>3} {r['n_vars']:>7} {r['n_factors']:>7} "
                  f"{r['treewidth']:>4} {r['cost_d_pow_tw']:>12.3e} "
                  f"{r['largest_table']:>10} {r['seconds']:>9.4f}")
    return rows


# --------------------------------------------------------------------------- verify
def verify(verbose: bool = True, seed: int = 0) -> dict:
    """(a) exactness vs two independent references, (b) the flat-treewidth headline,
    (c) the contrast that makes (b) falsifiable."""
    out: dict = {}

    # ---- (a1) tiny instances vs PURE-PYTHON BRUTE FORCE -----------------------------
    e_min = e_sum = e_arg = 0.0
    tiny_tw = []
    cases = []
    for K, m, d, s, topo, coup in [
            (2, 3, 2, 1, "chain", "matching"),
            (3, 3, 2, 1, "chain", "matching"),
            (3, 3, 2, 2, "chain", "complete"),
            (2, 4, 2, 2, "chain", "matching"),
            (4, 2, 3, 1, "chain", "matching"),
            (3, 2, 3, 1, "chain", "complete"),
            (4, 3, 2, 1, "all-to-all", "matching"),
            (3, 3, 2, 2, "all-to-all", "complete"),
            (5, 2, 2, 1, "all-to-all", "matching")]:
        for sd in (seed, seed + 1):
            g = build_clustered_graph(K, m, d, s, sd, topology=topo,
                                      interface_coupling=coup)
            tiny_tw.append(g.treewidth())
            r = solve(g, "min")
            bmin, barg = brute_force(g, "min")
            e_min = max(e_min, abs(r["value"] - bmin))
            # the returned assignment must actually achieve the returned value
            tot = math.fsum(float(f.table[tuple(r["assignment"][v] for v in f.vars)])
                            for f in g.factors)
            e_arg = max(e_arg, abs(tot - r["value"]))
            rs = solve(g, "sum")
            bsum, _ = brute_force(g, "sum")
            e_sum = max(e_sum, abs(rs["value"] - bsum))
            cases.append((K, m, d, s, topo, coup))
    out["n_bruteforce_cases"] = len(cases)
    out["max_err_min_vs_bruteforce"] = e_min
    out["max_err_logZ_vs_bruteforce"] = e_sum
    out["max_err_argmin_consistency"] = e_arg
    out["bruteforce_treewidths"] = (min(tiny_tw), max(tiny_tw))

    # ---- (a2) LARGE instances vs the CLUSTER TRANSFER-MATRIX DP ---------------------
    e_min_ref = e_sum_ref = 0.0
    ref_rows = []
    for K, m, d, s, coup in [(4, 4, 2, 1, "matching"), (8, 4, 2, 2, "matching"),
                             (16, 5, 2, 2, "matching"), (32, 5, 2, 2, "matching"),
                             (8, 4, 3, 1, "matching"), (6, 4, 2, 2, "complete"),
                             (12, 5, 2, 3, "complete")]:
        g = build_clustered_graph(K, m, d, s, seed, topology="chain",
                                  interface_coupling=coup)
        rmin = solve(g, "min")
        rsum = solve(g, "sum")
        amin = chain_reference(g, "min")
        asum = chain_reference(g, "sum")
        e_min_ref = max(e_min_ref, abs(rmin["value"] - amin))
        e_sum_ref = max(e_sum_ref, abs(rsum["value"] - asum))
        ref_rows.append((K, m, d, s, coup, rmin["n_vars"], rmin["treewidth"]))
    out["n_transfer_matrix_cases"] = len(ref_rows)
    out["max_err_min_vs_transfer_matrix"] = e_min_ref
    out["max_err_logZ_vs_transfer_matrix"] = e_sum_ref
    out["largest_checked_n_vars"] = max(r[5] for r in ref_rows)

    # ---- (a3) LOG SPACE, where a probability-space implementation would overflow ------
    g_big = build_clustered_graph(256, 5, 2, 2, seed, topology="chain",
                                  interface_coupling="matching")
    r_big = solve(g_big, "sum")
    a_big = chain_reference(g_big, "sum")
    out["logspace_n_vars"] = r_big["n_vars"]
    out["logspace_logZ"] = r_big["value"]
    out["logspace_err"] = abs(r_big["value"] - a_big)
    out["logspace_overflows_in_prob_space"] = bool(r_big["value"] > 709.78)

    if verbose:
        print("  rem.clusters.verify")
        print("  (a) EXACTNESS vs two independent references")
        print(f"    naive pure-python enumeration, {len(cases)} clustered instances, "
              f"treewidth {min(tiny_tw)}-{max(tiny_tw)}")
        print(f"      max |eliminate(min)  - brute force|          {e_min:.3e}")
        print(f"      max |eliminate(sum)  - brute force logZ|     {e_sum:.3e}")
        print(f"      max |E(argmin) - reported min|               {e_arg:.3e}")
        big = max(ref_rows, key=lambda r: r[5])
        print(f"    cluster transfer-matrix DP, {len(ref_rows)} instances up to "
              f"{big[5]} vars ({big[2]}^{big[5]} = "
              f"{big[2]**big[5]:.3e} assignments)")
        print(f"      max |eliminate(min)  - transfer-matrix DP|   {e_min_ref:.3e}")
        print(f"      max |eliminate(sum)  - transfer-matrix DP|   {e_sum_ref:.3e}")
        print(f"    log space: {out['logspace_n_vars']} vars, "
              f"logZ = {out['logspace_logZ']:.4f}  (err vs DP {out['logspace_err']:.3e}); "
              f"exp(logZ) overflows float64: {out['logspace_overflows_in_prob_space']}")

    # ---- (b) THE HEADLINE ------------------------------------------------------------
    if verbose:
        print("  (b) HEADLINE -- cluster_size fixed, n_clusters grows 2 -> 128")
    Ks = (2, 4, 8, 16, 32, 64, 128)
    head = scaling_table(Ks, cluster_size=5, d=3, interface_size=2,
                         seed=seed, mode="min", topology="chain",
                         interface_coupling="matching", repeats=3, verbose=verbose,
                         label="chain, narrow interface (the claim)")
    tws = [r["treewidth"] for r in head]
    tables = [r["largest_table"] for r in head]
    secs = [r["seconds"] for r in head]
    spc = [r["sec_per_cluster"] for r in head]
    out["headline"] = head
    out["headline_treewidths"] = tws
    out["headline_flat"] = bool(len(set(tws)) == 1 and len(set(tables)) == 1)
    out["headline_time_ratio_32_over_2"] = secs[4] / secs[0] if secs[0] > 0 else np.inf
    out["headline_time_ratio_128_over_2"] = secs[-1] / secs[0] if secs[0] > 0 else np.inf
    out["headline_sec_per_cluster_spread"] = max(spc) / min(spc)
    # least-squares fit of seconds against n_clusters; R^2 near 1 means LINEAR
    x = np.asarray(Ks, dtype=float)
    y = np.asarray(secs, dtype=float)
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    out["headline_fit_slope_sec_per_cluster"] = float(coef[0])
    out["headline_fit_intercept_sec"] = float(coef[1])
    out["headline_fit_r2"] = r2
    out["headline_doubling_ratios"] = [secs[i + 1] / secs[i] for i in range(len(secs) - 1)]
    if verbose:
        print(f"      treewidth {tws} -> {'FLAT' if out['headline_flat'] else 'NOT FLAT'}"
              f"; largest table {tables[0]} constant = d^(tw+1) = 3^5")
        print(f"      64x the clusters (2 -> 128) cost "
              f"{out['headline_time_ratio_128_over_2']:.2f}x the time "
              f"(linear = 64x; exponential in K would be 3^126)")
        print(f"      doubling ratios T(2K)/T(K) = "
              f"{[round(v, 2) for v in out['headline_doubling_ratios']]}  (linear -> 2.0)")
        print(f"      least-squares fit  sec = {coef[0]:.3e} * K + {coef[1]:.3e}, "
              f"R^2 = {r2:.5f};  sec/cluster spread {out['headline_sec_per_cluster_spread']:.2f}x")

    # ---- (c) THE CONTRAST ------------------------------------------------------------
    if verbose:
        print("  (c) CONTRAST -- same clusters, same d, only the WIRING changes")
    ata = scaling_table((2, 4, 8, 12, 16, 20), cluster_size=5, d=3, interface_size=1,
                        seed=seed, mode="min", topology="all-to-all",
                        interface_coupling="matching", max_table=5e6, repeats=1,
                        verbose=verbose, label="all-to-all clusters (the contrast)")
    out["contrast_all_to_all"] = ata
    ata_tws = [r["treewidth"] for r in ata]
    out["contrast_treewidths"] = ata_tws
    out["contrast_grows"] = bool(ata_tws[-1] > ata_tws[0] and
                                 ata_tws == sorted(ata_tws))
    out["contrast_hits_wall"] = bool(any(r["wall"] for r in ata))
    if verbose:
        print(f"      treewidth {ata_tws} -> "
              f"{'GROWS with n_clusters' if out['contrast_grows'] else 'did NOT grow'}"
              f"; predicted cost d^tw up to {ata[-1]['cost_d_pow_tw']:.3e}")
        walls = [r['n_clusters'] for r in ata if r['wall']]
        print(f"      treewidth wall (max_table=5e6) hit at K = {walls}")

    scan = interface_scan((1, 2, 3, 4, 5, 6), n_clusters=6, cluster_size=6, d=2,
                          seed=seed, mode="min", interface_coupling="complete",
                          verbose=verbose)
    out["contrast_interface_scan"] = scan
    scan_tws = [r["treewidth"] for r in scan]
    out["interface_scan_treewidths"] = scan_tws
    out["interface_scan_grows"] = bool(scan_tws[-1] > scan_tws[0])
    if verbose:
        print(f"      treewidth {scan_tws} -> "
              f"{'GROWS with interface width' if out['interface_scan_grows'] else 'flat'}"
              f"  (n_vars constant at {scan[0]['n_vars']})")
        print("  cost = d ** treewidth. Flat treewidth is a property of the WIRING, "
              "not of the size.")

    # ---- (d) POSITIVE CONTROL -------------------------------------------------------
    planted_rows = []
    for K, m, d, s, topo in [(3, 3, 2, 1, "chain"), (8, 4, 3, 2, "chain"),
                             (32, 5, 3, 2, "chain"), (16, 4, 2, 2, "chain"),
                             (8, 4, 3, 1, "all-to-all")]:
        g = build_clustered_graph(K, m, d, s, seed, topology=topo)
        truth = plant_ground_state(g, seed=seed + 11)
        r = solve(g, "min")
        exact = all(r["assignment"][v] == truth[v] for v in truth)
        planted_energy = math.fsum(float(f.table[tuple(truth[v] for v in f.vars)])
                                   for f in g.factors)
        planted_rows.append({"n_clusters": K, "cluster_size": m, "d": d,
                             "topology": topo, "n_vars": r["n_vars"],
                             "treewidth": r["treewidth"], "recovered": bool(exact),
                             "gap": r["value"] - planted_energy})
    out["positive_control"] = planted_rows
    out["positive_control_all_recovered"] = bool(all(p["recovered"] for p in planted_rows))
    out["positive_control_max_gap"] = max(abs(p["gap"]) for p in planted_rows)
    if verbose:
        print("  (d) POSITIVE CONTROL -- a planted ground state must come back exactly")
        for p in planted_rows:
            print(f"      K={p['n_clusters']:>3} m={p['cluster_size']} d={p['d']} "
                  f"{p['topology']:>11}  {p['n_vars']:>4} vars  tw={p['treewidth']:>2}  "
                  f"argmin == planted: {p['recovered']}   "
                  f"E(argmin)-E(planted) = {p['gap']:+.3e}")
        print(f"      all recovered: {out['positive_control_all_recovered']}  "
              f"(so a null in (c) is the topology, not a dead pipeline)")
    return out


if __name__ == "__main__":                                    # pragma: no cover
    verify(verbose=True)

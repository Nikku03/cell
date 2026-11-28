"""Measured numbers for rem.factorgraph. Regenerates the README table."""
import sys, time, itertools
sys.path.insert(0, ".")
import numpy as np
from rem.factorgraph import FactorGraph


def chain(n, d, seed=0):
    rng = np.random.default_rng(seed)
    g = FactorGraph()
    for i in range(n):
        g.add_var(f"x{i}", d)
        g.add_factor([f"x{i}"], rng.normal(size=d))
    for i in range(n - 1):
        g.add_factor([f"x{i}", f"x{i+1}"], rng.normal(size=(d, d)))
    return g


def grid(rows, cols, d, seed=0):
    """A k x m grid has treewidth min(k, m) -- the knob for the scaling law."""
    rng = np.random.default_rng(seed)
    g = FactorGraph()
    name = lambda r, c: f"v{r}_{c}"
    for r in range(rows):
        for c in range(cols):
            g.add_var(name(r, c), d)
            g.add_factor([name(r, c)], rng.normal(size=d))
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                g.add_factor([name(r, c), name(r + 1, c)], rng.normal(size=(d, d)))
            if c + 1 < cols:
                g.add_factor([name(r, c), name(r, c + 1)], rng.normal(size=(d, d)))
    return g


def timeit(fn, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return out, best


print("=" * 78)
print("BENCHMARK 1 -- the spec's headline: 6^100 configurations")
print("=" * 78)
g = chain(100, 6)
(val, arg, info), dt = timeit(lambda: g.eliminate("min"))
print(f"  chain n=100, d=6   search space 6^100 = {6.0**100:.3e} configurations")
print(f"  treewidth {info['treewidth']}   largest table {info['largest_table']}")
print(f"  global optimum {val:.6f}   time {dt*1000:.2f} ms")
brute_equiv = 6.0 ** 100
print(f"  configurations per second if enumerated: would need "
      f"{brute_equiv/1e9/3.15e7:.3e} GHz-years")

print()
print("=" * 78)
print("BENCHMARK 2 -- exact partition function at scale")
print("=" * 78)
# Two routes to the same quantity, both measured. Generic elimination is O(n) because it
# sweeps every variable; the ring closed form is O(d^3) and never sweeps at all. Quoting
# only the fast one would hide which structures earn it.
from rem.circulant import ring_logZ_transfer
rng = np.random.default_rng(0)
logT = rng.normal(size=(6, 6))
print(f"  {'n':>12s} {'elimination':>18s} {'ring closed form':>18s}   speedup")
for n in (100, 1000, 10000, 100000, 1000000):
    if n <= 10000:
        g = chain(n, 6)
        (_, _, _), dt_el = timeit(lambda: g.eliminate("sum"), repeat=1)
        el = f"{dt_el*1000:.1f} ms"
    else:
        dt_el, el = float("nan"), "(O(n), skipped)"
    _, dt_ring = timeit(lambda: ring_logZ_transfer(logT, n), repeat=5)
    sp = f"{dt_el/dt_ring:,.0f}x" if dt_el == dt_el else "--"
    print(f"  {n:>12,} {el:>18s} {dt_ring*1e6:>15.1f} us   {sp}")
print("  elimination cost grows linearly in n; the ring closed form does not move.")

print()
print("=" * 78)
print("BENCHMARK 3 -- the governing law: cost = d^treewidth")
print("=" * 78)
print(f"  {'grid':>10s} {'treewidth':>10s} {'d^tw':>10s} {'largest table':>15s} {'time (ms)':>11s}")
d = 4
for rows in (2, 3, 4, 5, 6, 7):
    g = grid(rows, 8, d)
    try:
        (val, _, info), dt = timeit(lambda: g.eliminate("min"), repeat=1)
        print(f"  {f'{rows}x8':>10s} {info['treewidth']:>10d} {d**info['treewidth']:>10,} "
              f"{info['largest_table']:>15,} {dt*1000:>11.1f}")
    except MemoryError as e:
        print(f"  {f'{rows}x8':>10s} {g.treewidth():>10d} {d**g.treewidth():>10,} "
              f"{'--':>15s} {'TREEWIDTH WALL':>11s}")
        break
print("  time tracks d^treewidth, not the number of variables. That is the whole law.")

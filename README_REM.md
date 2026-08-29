# REM — exact inference and optimization over structured problems

```
cost = d ** treewidth          d = states per variable
```

Entanglement across a cut, bond dimension, edges crossing the cut, and treewidth are the
same number. When it is small, REM is exact and fast. When the dependency graph is an
expander (treewidth ∝ n), nothing is efficient — that is a property of the problem, not a
limit of this code.

**REM is not a quantum computer.** It cannot factor RSA or break strong cryptography.

## Status

| module | state | verified against |
|---|---|---|
| `rem.factorgraph` | done | brute force, max error **1.8e-15** over 40 random instances |
| `rem.circulant` | done | brute force **1.1e-14**, dense slogdet **7.1e-15** |
| `rem.clusters` | not started | |
| `rem.fftcorr` | not started | |
| `rem.bp` | not started | |
| `rem.docking` (4 algorithms) | not started | |

`pytest tests/` — 39 passed.

## Measured

**The law itself.** Grid graphs, d = 4; time tracks `d^treewidth`, not variable count:

| grid | treewidth | d^tw | largest table | time |
|---|---|---|---|---|
| 2×8 | 2 | 16 | 64 | 0.7 ms |
| 3×8 | 3 | 64 | 256 | 1.0 ms |
| 4×8 | 4 | 256 | 1,024 | 1.7 ms |
| 5×8 | 5 | 1,024 | 4,096 | 2.9 ms |
| 6×8 | 8¹ | 65,536 | 262,144 | 12.3 ms |

¹ greedy min-fill/min-degree returned 8 where the true treewidth of a 6×8 grid is 6. These
orderings are heuristics; the reported width is an upper bound and the cost follows the
width actually used, not the optimum.

**Global optimum over 6^100 configurations** (chain, treewidth 1): **6.5 ms**.
Enumeration would need 2.1e61 GHz-years.

**Ring partition function**, `Z = tr(Tⁿ) = Σλᵢⁿ`, cost O(d³) independent of n:

| n | space | time |
|---|---|---|
| 10³ | 6^1000 | 30 µs |
| 10⁶ | 6^1000000 | **17 µs** |
| 10¹⁵ | 6^10¹⁵ | 59 µs |

Only for a **homogeneous** ring. A heterogeneous one needs the O(n·d³) matrix product.

## Two bugs found by measurement, not by reading

Both were found because the benchmark printed a number that did not match the claimed
complexity, and both first diagnoses were wrong.

1. **Elimination looked O(n²)** — 8 ms → 504 ms → 58 s for n = 100 → 1,000 → 10,000.
   First guess: the factor pool was rescanned per step. Fixed with bucket indexing;
   **it barely helped** (58 s → 52 s), so the guess was wrong.
2. **Profiling showed the real cause: order selection, not elimination.** At n = 4,000,
   min-fill took 5,949 ms and the elimination it was ordering took 126 ms. Replaced the
   rescan with a lazy heap that revalidates stale scores on pop: **5,949 ms → 28 ms**,
   same orderings, verification unchanged.

## The pathwidth trap

The classic bug is a running joint over every variable seen so far, which costs
`d^pathwidth` — always ≥ treewidth, often far worse. `eliminate()` keeps a **bucket-indexed
factor list** and touches only factors mentioning the variable being eliminated.
`verify_pathwidth_trap()` asserts a 60-variable chain (d=4) keeps its largest intermediate
table at **16 = d²**, where a joint frontier would build 4^60.

## Conventions

A factor holds a real table `phi`. `eliminate("min")` returns `min Σ phi` plus the argmin;
`eliminate("sum")` returns `log Σ exp(Σ phi)`. So `phi = −E/kT` gives log Z and `phi = E`
gives the ground state — one table type, two semirings.

Every module ships a `verify()` that checks it against brute force and prints max error.

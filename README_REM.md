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

`pytest tests/` — **562 passed, 1 skipped**.

| module | verified against | max error |
|---|---|---|
| `rem.factorgraph` | brute force, 40 random instances + 40 with factor-free variables | **3.6e-15** |
| `rem.circulant` | brute force / dense slogdet | **1.1e-14** / 7.1e-15 |
| `rem.clusters` | brute force + an independent chain contraction | see tests |
| `rem.fftcorr` | O(N⁶) direct correlation; Fourier-diagonality built explicitly | **4.2e-16** |
| `rem.bp` | exact elimination on trees, enumeration on loops | see tests |
| `rem.rna` | explicit enumeration of nested structures | see tests |
| `rem.phylo` | enumeration of ancestral assignments | see tests |
| `rem.hp` | enumeration of every self-avoiding walk; admissibility audit of the pruning bound | exact |
| `rem.tailrisk` | enumeration of every joint outcome; separate convolution path | **1.9e-16** |
| `rem.aggregate` | exact bigint **rational** arithmetic — no float64 anywhere in the reference | **3.1e-15** |
| `rem.docking.score` | closed form (LJ at Rmin, LJ at σ, Coulomb) | **7.6e-17** |
| `rem.docking.repack` (Alg 2) | exhaustive enumeration of all rotamer assignments | **0.0** |
| `rem.docking.rigid` (Alg 1) | planted translation / planted rotation + negative control | exact |
| `rem.docking.flexible` (Alg 3) | enumeration of pose × all rotamers | **0.0** |
| `rem.docking.freeenergy` (Alg 4) | enumeration of ln Z and of marginals | **7.1e-15** |
| `rem.docking.capri` | native pose, monotonicity, frame invariance | M1–M5b |

## What the four docking algorithms actually are

1. **REM-FFT rigid search.** Translation factorizes exactly — the score operator is
   circulant, hence diagonal in Fourier, bond dimension 1, effective treewidth 0 — so one
   FFT pair scores all N³ translations. **Rotation does not factorize**; K rotations cost K
   independent searches. The saving is N³/log N on the translation axis only (≈3,000× at
   the grid sizes used here), not an exponential win.
2. **REM-VE side-chain repacking.** Interface rotamers become a factor graph; elimination
   returns the guaranteed global optimum.
3. **REM-Cluster flexible refinement.** The pose is a hub adjacent to every side chain, so
   treewidth is *n* and no ordering avoids it. The honest answer is cutset conditioning:
   fix the pose, and the graph collapses to the interface contact graph. Cost
   `d_pose × d^treewidth`, exact.
4. **REM-Z binding free energy.** The same graph, the other semiring: `eliminate("sum")` on
   `φ = −E/RT` is exactly `ln Z`, no sampling, no convergence criterion.

## What failed, and was kept in the record

This is the part that matters. Eight failures are recorded below, and none had its bar
quietly moved. They are not all the same kind of thing, so the breakdown matters:
**four were predeclared gates that failed outright** (Z4, Z4b, M5, and T5 — the last
voided rather than failed, for the reason in its row); **one** was a measurement declared
reported-not-gated that turned out incapable of failing (Z7 v1); **two** were construction
errors in the benchmark's own reference quantities; **one** was a library bug reported
externally and confirmed before any change was made.

| gate | what failed | why |
|---|---|---|
| Alg 4 **Z4** | high-T limit off by 3.9e-2 vs a 1e-3 bar | the residual is exactly −⟨E⟩/RT, and ⟨E⟩ = 77.5 kcal/mol, so the bar was unreachable when written. Replaced by **Z4c**, which gates only dimensionless cumulant exponents (−1, −2, −3) — no units, so no energy scale can make it unreachable. |
| Alg 4 **Z4b** | the repair failed too, at 7.4% vs 1% | −⟨E⟩/RT is only the *first* cumulant; the second is 7.7% of it at that temperature. Defect L committed inside its own fix. |
| Alg 4 **Z7** (v1) | reported "identical ordering" and could not have reported otherwise | it ranked problems of 3–7 residues, so both orderings were the residue count. Rebuilt to rank 8 decoy poses of the *same* residue set. |
| `capri` **M5** | 128 rotations scored *worse* than 32 | `rotation_set(n)` drew *n independent* quaternions, so a bigger set was a different set, not a refinement. Fixed in the **sampler** (prefixes now nested), not the gate. |
| DB5 driver | "search error" came out negative | it subtracted a full-ligand RMSD floor from an interface RMSD. |
| DB5 driver | the replacement floor was **not a lower bound** — 7.65 Å "floor" where the search achieved 4.75 Å | it placed each rotation at the translation optimal for a *different* objective. A quantity at the argmin of A is not a bound on B. |
| `tailrisk` **T5** | claimed cost linear in discretization; it is quadratic | Marked **VOID**, not failed — its wall-clock statistic gave 1.945 / 1.127 / 1.783 / 1.835 on unchanged code under varying machine load. A gate whose verdict flips between identical runs is not measuring the code. **T5b gates the largest intermediate table, not runtime**: a deterministic property of the contraction that cannot be flattered by a quiet machine. It is exactly `n_bins²·d` on every size, log-log slope **2.000000**. |
| `factorgraph` | `eliminate("sum")` under-counted ln Z by `log(card)` per factor-free variable | reported externally with a minimal repro. The 40-instance sweep could not see it because `random_graph` gave every variable a unary. Fixed; `random_graph` now takes `n_free` and `verify()` covers the case by name and by sweep. |

Twelve machinery defects are in `colab/gate_guard.py` — lettered A–M, skipping F — each
recorded with the false sentence it caused a run to print. Z4, Z4b and every other failed gate stay in the table above and in the
modules' own output, each with its reason; none was deleted or relabelled as passing.

### The n² in `tailrisk` is a spec error, not an implementation one — and it is now closed

Worth separating, because the two call for different fixes. The running total enters as a
factor on `(S_{i-1}, X_i, S_i)`, which the generic dense-factor formalism stores as a full
`n_bins × d × n_bins` table — measured largest table equals `n_bins²·d` to a ratio of 1.00.
But the operation that factor *represents* is a convolution. Nothing in `rem.tailrisk` is
coded wrong; the cost comes from specifying the sum as a generic factor rather than as a
convolution primitive.

`rem.aggregate` is that primitive, and it closes the gap. Aggregation becomes bucket
elimination on a chain in which the running partial sum is the *only* variable crossing any
cut — **treewidth 1** — so cost is `O(N · d_max · m_max)`, linear in the number of units:

- 40 heterogeneous units, support 197, bond dimension 197, **22,605 multiply-adds, 0.6 ms**.
  A joint frontier over the same units would be ~10³⁰ outcomes.
- Verified against **exact bigint rational** arithmetic (no float64 in the reference):
  `P(S ≥ 156) = 1.550583465440e-30`, float64 log-space relative error **3.05e-15**.

**But `n log n` is the wrong fix for this problem, and the module measures why.** FFT
convolution is `O(N log N)` as advertised, and numerically unusable in the tail: its error
floor is *absolute*, ~`eps·max(p)` ≈ 1.1e-17, so everything below that is noise. At
`S = 156` the FFT relative error is **5.4e+13** against 3.05e-15 for direct convolution, and
36 of 197 FFT entries come out **negative** — a probability cannot be negative, and that is
the round-off floor made visible. Direct convolution is `O(N·m)` with no cancellation, so
relative accuracy holds to the subnormal floor; log-space has no practical floor and is
*required* past float64 underflow — a 600-unit portfolio reaches `P = 10^-400.1`, where
linear convolution returns exactly 0.0.

So the complexity argument and the numerical argument point opposite ways, and for
rare-event work the numerical one wins.

## Measured

**Algorithm 2's exactness is real and cheap.** 84 DB5 interface instances, sizes 4–16
residues: solved exactly **84/84**, treewidth wall hit **0/84**, largest instance
2.8×10¹² configurations contracted in 1.8 s at treewidth 8. The contact graphs are *dense*
— density 0.79 at 4 residues falling to 0.49 at 16 — and treewidth rises with interface
size (median 2 → 6, max 8), so the low-treewidth assumption is not free.

**The greedy tie is real, and it is NOT a bound-structure artifact.** This was challenged
in review on the grounds that bound side chains are already optimally packed, so re-run as
a paired sweep — same cases, same interface sizes, same rotamer library, with only the
side-chain provenance moving:

| arm | instances | greedy missed the optimum | deposited already optimal | exact beats deposited by |
|---|---|---|---|---|
| bound | 84 | **0** | **1/84** | mean 45.1, max 336.5 kcal/mol |
| unbound | 83 | **0** | **0/83** | mean 153.5, max 721.9 kcal/mol |

Half of the challenge held and half did not, and the half that failed was **my
explanation**. I had written that in a bound structure the deposited rotamers *are* the
crystallographic optimum so nothing could be found. Measured: the deposited conformation
was optimal in **1 of 84** bound instances, and exact repacking improved it by 45 kcal/mol
on average. Under this energy function — Lennard-Jones plus Coulomb, no solvation — the
crystal conformation is simply not the minimum. What *did* hold is the direction: unbound
side chains are more mispacked, never optimal (0/83), with the gap **3.4× larger**.

So repacking does substantial work on both arms, and greedy still matches the exact
optimum on **167 of 167** instances. The guarantee does not bite at interface sizes 4–16.

**The baseline is not cheating**, which had to be checked before that negative could be
believed. Crippling greedy makes it miss, as it must:

| restarts | 1 | 2 | 5 | 20 |
|---|---|---|---|---|
| missed the optimum | **2/12** | 1/12 | 1/12 | **0/12** |

Single-restart greedy misses by up to **+16.35 kcal/mol**. So the problem is genuinely
non-trivial and the solver is genuinely independent — 20 restarts simply buys the same
answer exactness does, more cheaply, at these sizes.

**Algorithm 3's 0.0000 kcal/mol repacking gain** was measured on a bound structure and
rests on the same falsified explanation; `flexible.verify(bound=False)` re-runs it unbound.

**Two-sided interface graphs are sparser than one-sided ones**: density 0.29–0.40 and
treewidth 2–4 at 12 residues, against 0.49–0.79 and up to 8 for one-sided repacking.

**Where exactness genuinely wins: the tail.** Monte Carlo's relative error scales as
1/√(Np); exact summation is indifferent to rarity. On a 40-unit portfolio, `P(S ≥ 156) =
1.55e-30` exactly — while Monte Carlo with **10⁶ samples returned exactly zero**, 0 hits,
and could say only `p < 3e-6` at 95%, which is **24 orders of magnitude** above the true
value. This holds only while the dependency treewidth stays small: complete-graph
portfolios have treewidth n−1 and cost 4ⁿ, which `tailrisk`'s `T6` measures.

**Where the law says no: HP lattice folding.** NP-hard, and the turn-variable encoding has
treewidth linear in chain length — no structural saving exists. Confining the chain to a
strip of width W moves the exponential from the *length* to the *width*: cost linear in L,
exponential in W. Same law, different geometry.

## Docking Benchmark 5

271 complexes parsed; **270 usable**. Measured difficulty split against published DB5:

| class | n | measured | published |
|---|---|---|---|
| rigid | 163 | 60.4% | ~65% |
| medium | 45 | 16.7% | ~20% |
| difficult | 62 | 23.0% | ~15% |

Getting there required four real fixes, each of which had silently produced plausible
wrong numbers: DB5 renames chains between bound and unbound (keying on the letter matched
zero atoms and read as missing data); residue numbering differs (26% of cases excluded
until Needleman–Wunsch alignment per chain); crystallographic oligomers map several copies
to one chain (`dict.update` let the last win, giving I-RMSD 24 Å); and the copy-assignment
cost must be *direct* RMSD, not superimposed, because superposition scores all copies of an
oligomer alike.

*Docking run in progress — CAPRI accuracy by class, with search, scoring and discretization
error separated, lands here.*

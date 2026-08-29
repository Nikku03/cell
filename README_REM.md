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

`pytest tests/` — **395 passed**.

| module | verified against | max error |
|---|---|---|
| `rem.factorgraph` | brute force, 40 random instances + 40 with factor-free variables | **3.6e-15** |
| `rem.circulant` | brute force / dense slogdet | **1.1e-14** / 7.1e-15 |
| `rem.clusters` | brute force + an independent chain contraction | see tests |
| `rem.fftcorr` | O(N⁶) direct correlation; Fourier-diagonality built explicitly | **4.2e-16** |
| `rem.bp` | exact elimination on trees, enumeration on loops | see tests |
| `rem.rna` | explicit enumeration of nested structures | see tests |
| `rem.phylo` | enumeration of ancestral assignments | see tests |
| `rem.hp` | enumeration of every self-avoiding walk | H1–H5 |
| `rem.tailrisk` | enumeration of every joint outcome; separate convolution path | **1.9e-16** |
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

This is the part that matters. Nine predeclared gates failed; none had its bar quietly
moved.

| gate | what failed | why |
|---|---|---|
| Alg 4 **Z4** | high-T limit off by 3.9e-2 vs a 1e-3 bar | the residual is exactly −⟨E⟩/RT, and ⟨E⟩ = 77.5 kcal/mol, so the bar was unreachable when written. Replaced by **Z4c**, which gates only dimensionless cumulant exponents (−1, −2, −3) — no units, so no energy scale can make it unreachable. |
| Alg 4 **Z4b** | the repair failed too, at 7.4% vs 1% | −⟨E⟩/RT is only the *first* cumulant; the second is 7.7% of it at that temperature. Defect L committed inside its own fix. |
| Alg 4 **Z7** (v1) | reported "identical ordering" and could not have reported otherwise | it ranked problems of 3–7 residues, so both orderings were the residue count. Rebuilt to rank 8 decoy poses of the *same* residue set. |
| `capri` **M5** | 128 rotations scored *worse* than 32 | `rotation_set(n)` drew *n independent* quaternions, so a bigger set was a different set, not a refinement. Fixed in the **sampler** (prefixes now nested), not the gate. |
| DB5 driver | "search error" came out negative | it subtracted a full-ligand RMSD floor from an interface RMSD. |
| DB5 driver | the replacement floor was **not a lower bound** — 7.65 Å "floor" where the search achieved 4.75 Å | it placed each rotation at the translation optimal for a *different* objective. A quantity at the argmin of A is not a bound on B. |
| `tailrisk` **T5** | claimed cost linear in discretization; it is quadratic | the transfer factor is a dense `n_bins × d × n_bins` table even though the operation is a convolution. Marked **VOID**, not failed — its wall-clock statistic gave 1.945 / 1.127 / 1.783 / 1.835 on unchanged code under varying machine load. A gate whose verdict flips between identical runs is not measuring the code. **T5b** gates the deterministic largest table instead: exactly `n_bins²·d`, slope **2.000000**. |
| `factorgraph` | `eliminate("sum")` under-counted ln Z by `log(card)` per factor-free variable | reported externally with a minimal repro. The 40-instance sweep could not see it because `random_graph` gave every variable a unary. Fixed; `random_graph` now takes `n_free` and `verify()` covers the case by name and by sweep. |

Nine more machinery defects (A–M) are in `colab/gate_guard.py`, with the failing sentence
each one printed.

## Measured

**Algorithm 2's guarantee is real, and at these sizes it buys nothing.** 84 DB5 interface
instances, sizes 4–16 residues:

- solved exactly **84/84**, treewidth wall hit **0/84**
- largest instance: 2.8×10¹² configurations contracted in 1.8 s at treewidth 8
- **greedy (best-of-20 restarts, iterated to convergence) also found the optimum on 84/84**

That is a negative result for the advantage claim and is reported as one. The contact
graphs are *dense* — density 0.79 at 4 residues falling to 0.49 at 16 — and treewidth rises
with interface size (median 2 → 6, max 8), so the low-treewidth assumption is not free.

**Algorithm 3, where the gain comes from.** On a bound structure, exact two-sided repacking
contributed **0.0000 kcal/mol** and 100% of the refinement gain was the pose move — the
deposited rotamers already are the crystallographic optimum. Repacking can only pay on
*unbound* structures.

**Two-sided interface graphs are sparser than one-sided ones**: density 0.29–0.40 and
treewidth 2–4 at 12 residues, against 0.49–0.79 and up to 8 for one-sided repacking.

**Where exactness genuinely wins: the tail.** Monte Carlo's relative error scales as
1/√(Np); exact summation is indifferent to rarity. At p = 2.0×10⁻⁶, MC with 10⁴ and 10⁵
samples returned **zero** (100% error) and with 10⁶ samples was 102% off. The exact answer
costs the same as at p = 0.5 — while the dependency treewidth stays small. Complete-graph
portfolios have treewidth n−1 and cost 4ⁿ; `T6` measures that wall.

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

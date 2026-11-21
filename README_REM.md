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

**For Algorithm 3, the same challenge *did* hold.** Its 0.0000 kcal/mol repacking gain was
a bound-structure artifact, and the two arms invert completely:

| arm | from repacking | from the pose move |
|---|---|---|
| bound | +0.0000 (0.0%) | +1.5122 (**100.0%**) |
| unbound | +44.1416 (**86.3%**) | +7.0270 (13.7%) |

Two-sided exact repacking earns its place on unbound components and does nothing on bound
ones. So the review was right about Algorithm 3 and wrong about Algorithm 2 — different
mechanisms, and pooling them would have hidden both.

### Exactness is not accuracy: C4 fails on unbound

The sharpest result in the repo, and a failure. Algorithm 3's positive control refines from
a pose displaced 1.0 Å from native:

| arm | before | after | |
|---|---|---|---|
| bound | 1.000 Å | **0.535 Å** | PASS |
| unbound | 1.000 Å | **1.228 Å** | **FAIL** |

The verdict stands and the bar is not moved. This is **not** a search failure — C1 shows
the conditioned elimination returns the exact joint optimum over pose × rotamers to
`0.000e+00`. It is a scoring failure. The refiner finds the true minimum of the energy it
was given, and on unbound components that minimum is *not at the native pose*. Optimizing
an imperfect function exactly moves you exactly to the wrong place. No better search fixes
this; only a better energy does.

That is the same search-versus-scoring split the DB5 benchmark is built to expose, showing
up in miniature — and it is the honest limit of everything above: this project can tell you
it found the true optimum of its model, and cannot tell you the model is right.

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

## REM on a real human cell axis: exact ribosome maps

The one place in this project where REM meets a real cell. Hard rods of footprint 10 codons
on real human CDS, weights from **measured** tRNA abundance, scored against **measured**
ribosome profiling — everything HEK293-T so the tRNA pool and the footprints come from one
cell type.

| | source |
|---|---|
| CDS | Ensembl GRCh38 r112, 19,553 usable gene symbols |
| tRNA abundance | GEO GSE152621, mim-tRNAseq, `Hsap_HEK293T` rep1+rep2 |
| ribosome profiling | GEO GSE290865, total translatome, P-site counts per CDS nucleotide |

**The machinery is exact and linear.** Three independent solvers — forward-backward
hard-rod recursion, explicit enumeration of every valid configuration, and a
`rem.factorgraph` chain contracted with `eliminate("sum")` — agree to **1.8e-15** on both
occupancy and log Z. Cost is linear in transcript length, fitted slope **0.958**: a
2,997-codon transcript solves exactly in **6 ms**.

*The third solver initially disagreed by 1.3–2.5 in log Z. Two errors, both mine: a rod at
codon i needs the state at i−1 to be ≥ ell−1, not ≥ ell (requiring ell forbids the tightest
legal packing and undercounts Z), and the initial empty state must start at gap ell, not
gap 1. Having three solvers rather than two is what localised it to the construction rather
than the physics.*

**And the biology test is VOID — not failed.** M4 required the prediction to beat its own
codon-shuffled null, and it lost systematically. But that verdict was obtained with the
density unset: `codon_logweights` normalises to geometric mean 1, so the model ran at 0.068
ribosomes/codon — **68% of close packing** — against data at 5–10%. The one parameter that
decides whether exclusion matters at all was uncontrolled, so ledger rule A applies: this is
VOID, not FAIL.

**Re-run with the density set, and the conclusion returns — now established rather than
asserted.** Each transcript's load is taken from its own measured total P-site counts over
its length, rescaled so the median gene sits at the target; total counts is one number per
gene while what is scored is rank order *along* the gene, which Spearman makes
scale-invariant, so the calibration quantity is orthogonal to the predicted one.

| target ρ | % of close packing | realised/target | median real | median shuffled | paired | win frac |
|---|---|---|---|---|---|---|
| 0.005 | 5% | 1.0000 | −0.0211 | +0.0003 | −0.0226 | 35.5% |
| 0.010 | 10% | 1.0000 | −0.0211 | +0.0011 | −0.0228 | 34.5% |
| 0.020 | 20% | 1.0000 | −0.0193 | +0.0027 | −0.0231 | 34.1% |
| 0.040 | 40% | 1.0000 | −0.0154 | +0.0026 | −0.0206 | 37.0% |
| uncalibrated | ~68% | — | −0.0208 | +0.0014 | −0.0225 | 32.2% |

**D1 PASS** — the calibration lands exactly (max deviation 0.0000). **D2 FAIL** — at
physiological ρ = 0.01 the model beats its own codon-shuffled null on 34.5% of genes,
p = 1.00. **D3 PASS** — the sign of the paired difference is identical at every density
including the uncalibrated one.

So voiding M4 was procedurally right and **not load-bearing**: the density was uncontrolled,
which is enough to void a verdict, but controlling it did not change the answer. tRNA-
abundance dwell is weakly anti-predictive of P-site density across the whole range from 5%
to 68% of close packing. Crowding mitigates it slightly (−0.0211 → −0.0154 by 40% packing)
and never flips it.

The original uncalibrated numbers, kept for the record:

| predictor | median Spearman vs measured P-site density |
|---|---|
| REM occupancy from w = 1/W | **−0.0224** |
| codon-shuffled null | −0.0000 |
| position-only baseline (no sequence at all) | **+0.0548** |

Real beat its own shuffle on **375/1200 genes (31.2%)** — worse than chance. A predictor
that knows only *where you are in the transcript*, carrying no sequence information
whatsoever, beats the codon model outright.

**M4b localises the failure**, and it is the search-versus-scoring split again:

| | median Spearman |
|---|---|
| REM occupancy from w = 1/W | −0.0225 |
| raw 1/W with no REM at all | −0.0223 |
| raw W, sign flipped | +0.0221 |

`|REM − raw| = 0.0002`. The solver reproduces its input's correlation to four decimal
places, so it is propagating the signal faithfully and **the input physics is what is
wrong** — tRNA-abundance dwell is weakly anti-predictive of P-site density in this data.
The tAI weights are not the problem either: they rank GAC, ATG, AAA, AAG highest and ATA,
TCA, TTG, TCG, CTA, TTA lowest, which are the textbook rare codons.

**M6 found no asymmetry to explain.** Around the slowest 10% of codons the measured
upstream/downstream densities differ by −0.0235 at z = −1.7 — not significant. So the
equilibrium model's inability to queue was never the binding limitation here.

This is the DB5 result in a second domain, and it was predicted before the run: exact
machinery, wrong model. REM will compute the true optimum of whatever physics you hand it.

## When does REM earn its keep? The crowding crossover

The sharpest methodological result here, and it came from a challenge: *REM only pays when
things are genuinely jammed, and we keep pointing it at roomy problems.* That is now
measured rather than asserted.

Exclusion is the **only** thing the exact hard-rod machinery buys over an independent-site
model. Holding the density fixed with a per-transcript fugacity and comparing the two:

| density (rib/codon) | % of close packing | error if you ignore exclusion | contact correlation g |
|---|---|---|---|
| 0.005 | 5% | **0.9%** | 1.047 |
| 0.010 | 10% | 1.8% | 1.099 |
| 0.020 | 20% | 3.6% | 1.219 |
| 0.040 | 40% | 7.6% | 1.562 |
| 0.060 | 60% | 12.7% | 2.171 |
| 0.080 | 80% | **23.4%** | 3.555 |
| 0.095 | 95% | **55.0%** | 15.49 |

Real monosome density is 0.005–0.01 per codon — **5–10% of close packing**, where ignoring
exclusion entirely costs under 2%. The machinery was idle. It becomes decisive only above
roughly half of close packing. That is the crossover, and it says exactly where to point
this tool.

**It also exposed a defect in my own earlier run.** `codon_logweights` normalises to
geometric mean 1 and *nothing set the density*: the model ran at 0.068 ribosomes/codon,
which is **68% of close packing**, against dilute data. The one parameter controlling
whether exclusion matters at all was left unset. A fugacity is now solved per transcript.

**The two-point function is validated against exact theory.** `contact_pairs` computes
P(rod at *i* AND rod at *i+ell*) — the collision event — from the exact two-point function
rather than a product of marginals:

| density | measured g | exact lattice g | rel. dev. |
|---|---|---|---|
| 0.005 | 1.0471 | 1.0471 | 5.3e-06 |
| 0.020 | 1.2194 | 1.2195 | 9.9e-05 |
| 0.080 | 3.5550 | 3.5714 | 4.6e-03 |

Getting there cost two wrong reference values, both mine. **C4** predeclared `|g−1| < 0.02`
in the dilute limit and failed at 1.0471 — defect L for the third time, an absolute
tolerance not derived from the quantity's own theory; 1.0 is never the correct limit at
finite density. **C4b** then used the *continuum* Tonks form `1/(1−ρℓ)` and showed
deviations growing 5e-3 → 2.9e-1 with density. A systematic drift is what a wrong
*reference* looks like; a wrong *implementation* gives a flat noise floor. This is a
lattice model, where the gap between neighbouring rods is a non-negative integer, giving
`g = 1/(1 − ρ(ℓ−1))` — which matches to four to six decimals at every density.

**The collision test is not done, and the blocker is data, not method.** Searched GEO
thoroughly for human disome / collided-ribosome profiling. Four routes, each blocked for a
specific reason: `GSE133393` has the matched monosome/disome pair and a real signal (8,973
transcripts, median disome/monosome **0.083**) but keys on legacy UCSC `uc` transcript IDs,
and UCSC has migrated hg19 to ENST so the mapping table is no longer served — a length-based
join resolves only 483 of 9,128 unambiguously. `GSE201364` deposits only a metagene average.
`GSE145723` and `GSE282964`/`GSE275336` deposit only raw reads. `GSE299329` has gene symbols
but 40 peaks total. What would unblock it: a legacy UCSC `uc`→symbol table, or any disome
dataset with per-transcript counts keyed on symbols or ENST.

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

### The docking run: the search works, the score does not

60 complexes, 20 per class, 2000 rotations (nn spacing 10.75°), 1.5 Å/voxel. The
discretization floor holds as a valid bound on all 120 arms.

| class | n | BB rank-1 | BB best | UU rank-1 | UU best |
|---|---|---|---|---|---|
| rigid | 20 | 0/20 | 13/20 | 0/20 | 16/20 |
| medium | 20 | 0/20 | 17/20 | 0/20 | 8/20 |
| difficult | 20 | **1/20** | 12/20 | 0/20 | 6/20 |

**One rank-1 success in 120 arms.** But an acceptable pose is *present in the search output*
in 42/60 bound and 30/60 unbound arms. The search generates the answer; the score never
picks it. In L-RMSD medians the split is stark — search error 4.3–9.2 Å against scoring
error 38.7–56.7 Å, five to ten times larger.

**Sampling is not the constraint.** Floors: median 2.18 Å, max 6.68 Å; 60/60 cases could
reach *acceptable*, 56/60 *medium*, and only 5/60 *high*. Nothing was floor-limited.

**Rescoring cannot rescue it, and the ceiling says why.** Reranking the same 20 poses five
ways — grid, pair energy, exact VE repacking, greedy repacking, −RT ln Z — gives 0/58 every
time. The *ceiling*, the best pose present in that shortlist at all, is 12.26 Å median and
also scores 0/58. The near-native pose is usually not in the top 20 to begin with, so this
is an argument for keeping more candidates, not for a better rescorer.

**Greedy again ties exactness**: 0 of 1160 rescored poses where greedy missed the VE
optimum, and it never changed which pose ranked first.

**Flexible refinement makes it slightly worse**: I-RMSD 17.70 → 17.78 median, improving on
10 of 28 medium/difficult cases. That is the C4 result at benchmark scale — exact
optimization of an imperfect energy moves confidently in the wrong direction.

The one-line summary of the whole benchmark: **this pipeline's failure is scoring, not
search, and REM addresses search.** Every exactness guarantee in this repo is real and none
of them touches the thing that is actually broken.

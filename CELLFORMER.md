# CELLFORMER — an AlphaFold-shaped architecture for the cell, and the ablation table that judges it

The request was for a transformer test with AlphaFold's level of architectural commitment, not a data dump into
a stock encoder. This document states the analogy precisely, says where it holds and where it breaks, and — most
importantly — defines the ablation table **before** anything is trained, because an architecture that cannot show
each of its parts earning its keep has not been tested, only run.

---

## 0. Why a stock transformer would be the wrong answer here

Dumping the 5,120 × 8,246 perturbation matrix into a sequence model and training it to reconstruct rows would
produce a number, and that number would be uninterpretable. Three measurements from this project say why:

| measured | consequence for a naive model |
|---|---|
| a **frequency baseline** (name the globally most-moved genes) scores 0.31–0.35 recall and beat seven mechanistic methods | any model that learns the marginal gene distribution scores well without learning anything conditional |
| the shared stress core is **37.4%** of held-out response variance, and removing it makes methods score *worse* | the benchmark rewards predicting damage, not predicting which gene was damaged |
| **40.6%** of responsive genes move for exactly one perturbation | the specific tail is unlearnable from response data alone; it has to come from mechanism |

So the architecture's job is not to fit the matrix. It is to **carry mechanism into the prediction** in a way that
can be switched off and measured. That is exactly what AlphaFold's design does: it does not learn structure from
sequence by brute force, it builds coevolution and geometric consistency into the computation graph.

---

## 1. The analogy, stated exactly

AlphaFold's central insight is that **coevolution across an MSA constrains residue contacts**, and that the
network should read that signal through a pair representation kept geometrically consistent.

The corresponding insight here: **co-response across perturbations constrains functional coupling.** A
Perturb-seq matrix is an MSA whose "species" are interventions.

| AlphaFold 2 | Cellformer | why the mapping holds |
|---|---|---|
| residues *r* | genes *g* | the entities being related |
| MSA rows: homologous sequences | perturbation rows: 5,120 knockouts | each row is one independent piece of evidence about how genes covary |
| MSA representation `m[s,r,c]` | response representation `m[p,g,c]` | same shape, same role |
| pair representation `z[r,r',c]` | gene-pair representation `z[g,g',c]` | initialised from the reaction network / complexes / PPI instead of from residue index |
| **triangle inequality** on distances | **stoichiometric mass balance** `S·v = 0` | the hard physical law the output must obey |
| structure module → 3D coordinates | flux module → reaction fluxes `v` | the physically-constrained output object |
| Invariant Point Attention (SE(3)) | null-space projection of `v` onto `null(S)` | the "allowed manifold" the output is forced onto |
| recycling ×3 | recycling ×3 | iterative refinement of both representations |
| distogram auxiliary head | co-response auxiliary head | supervises the pair representation directly |
| pLDDT confidence | per-perturbation confidence → **abstention** | ties to build 2, where 58% of perturbations do nothing |
| MSA cropping to 256 residues | crop to 256 genes × 128 perturbations | identical trick, identical reason (memory) |

### Where the analogy breaks, and what is done about it

Being explicit about this matters more than the parts that work.

1. **Distance is metric; stoichiometry is not.** AlphaFold's triangle update is justified because distances obey
   the triangle inequality — a local, three-body constraint that can be applied to `z[i,j]` from `z[i,k]` and
   `z[k,j]`. Mass balance is *global* and *linear*, not local and metric. Applying a literal triangle update to
   gene pairs would be cargo-culting the form without the justification.
   **What is done instead:** the pair update propagates along the **bipartite gene–reaction graph**. `z[i,j]` is
   updated from the reactions that genes *i* and *j* participate in. That is the real locality structure of this
   problem, and it is the honest analogue — same role (inject the physical constraint into the pair
   representation), different and correct mechanism.

2. **AlphaFold's output is deterministic given inputs; a cell's is not.** Protein structure is (mostly) a
   function of sequence. A knockout's transcriptional response is stochastic, cell-state dependent, and — by this
   project's own measurement — **58.2% of the time it is nothing at all**. A model forced to always emit a
   response is wrong by construction on the majority of inputs.
   **What is done instead:** the confidence head is not decorative. It gates the output, and the model is scored
   *with abstention allowed*, using build 2's calibration machinery.

3. **AlphaFold had 170,000 structures; we have 5,120 perturbations.** Three orders of magnitude less. Any claim
   of "AlphaFold-scale" would be false. The architecture is AlphaFold-*shaped*; the regime is small-data, so the
   inductive biases (pair representation from a known network, physical constraint as a loss) matter *more*, not
   less — that is the whole argument for building them in rather than learning them.

---

## 2. Architecture

```
INPUTS
  m0 [P, G, c_m]   response representation, from measured z-scores + learned gene/perturbation embeddings
  z0 [G, G, c_z]   pair representation, initialised from FIVE measured relations:
                     shares a reaction   (Human-GEM, 12,931 reactions)
                     shares a complex    (2,039 complexes)
                     PPI edge            (191,447 edges, index space verified)
                     co-expression       (16,374 genes)
                     signed regulation   (1,451 genes; the 91% unsigned layer is a SEPARATE channel and
                                          flagged, because it overlaps real K562 regulation at chance)
  r0 [N, c_r]      reaction representation: stoichiometry column, kcat, Σ abundance, EC class

CELLFORMER BLOCK  × L                                      (the Evoformer analogue)
  a. row attention over genes, biased by z        m[p,:,:] ← attn(m[p,:,:], bias=z)     ← pair informs response
  b. column attention over perturbations          m[:,g,:] ← attn(m[:,g,:])             ← the "coevolution" read
  c. outer-product mean                           z ← z + OPM(m)                        ← response informs pair
  d. reaction-graph pair update                   z[i,j] ← z[i,j] + Σ_r B[i,r]B[j,r]W z ← the physics injection
  e. transition (gated FFN) on both m and z

FLUX MODULE                                                (the structure module analogue)
  per-gene capacity        e = softplus(Linear(m̄))                  m̄ = mean over perturbation axis
  per-reaction capacity    c_r = GPR-fold(e)                        isozymes add, complexes take the min
  differentiable flux      v = Π_null(S) · (c_r ⊙ w)                projection onto the mass-balance manifold
  recycle                  m ← m + Embed(v);  z ← z + Embed(v ⊗ v)

HEADS
  response      Δexpression per (perturbation, gene)
  confidence    P(this perturbation moves ≥ 5 specific genes)        → abstention gate
  kcat          log10 k_app per reaction                             → the parameter-estimation task
  co-response   pair-level auxiliary, supervising z directly         (the distogram analogue)

LOSSES
  L = L_response + λ_c·L_confidence + λ_k·L_kcat + λ_p·L_pair + λ_phys·‖S·v‖²
```

**Recycling** runs 3 passes, gradients through the last only — AlphaFold's exact scheme, and the reason is the
same: it buys iterative refinement at the cost of one backward pass.

**Cropping.** 256 genes × 128 perturbations per training example. This is not a shortcut around a limitation; it
is AlphaFold's own training procedure (it crops to 256 residues), and it makes the pair representation
256×256×32 = 8 MB instead of 8,246²×32 = 4.3 GB. The sandbox has **torch 2.12.1, CPU only, no GPU**, so this is
what makes the model trainable at all, and the crop size is reported rather than buried.

---

## 3. The test is the ablation table, not the score

A single number from a complex model proves nothing. AlphaFold's own case rests on its ablation figure — each
architectural commitment removed, one at a time, and shown to cost accuracy. That is the standard being adopted.

Every row is trained and scored identically, on the **same held-out perturbations**, with the **same harness**
this project has used throughout (tide-removed specific movers, top-50 predictions, recall and precision).

| # | ablation | what it tests |
|---|---|---|
| 0 | **full model** | — |
| 1 | no pair representation (`z` removed, row attention unbiased) | does the network prior contribute anything? |
| 2 | pair repr **shuffled** (same density, wrong partners) | is it the *content* of the network or just its existence? |
| 3 | no reaction-graph pair update (step d removed) | does the physics injection earn its place? |
| 4 | no recycling (1 pass instead of 3) | does iterative refinement help at this data scale? |
| 5 | no flux module (heads read `m` directly) | does the mass-balance manifold constrain usefully? |
| 6 | no physical loss (`λ_phys = 0`) | is `S·v = 0` doing work, or is the module decorative? |
| 7 | no column attention (perturbations treated independently) | is the "coevolution" read real? |
| 8 | no auxiliary heads (response loss only) | do the auxiliary losses regularise? |
| 9 | **shuffled labels** | the sanity floor — must collapse to chance |
| 10 | **frequency baseline** | the bar seven mechanistic methods have already failed |

**Pre-registered pass criterion.** The full model must beat the frequency baseline on held-out perturbations with
a paired bootstrap CI excluding zero. Anything less is reported as a failure, exactly as builds 1, 4, 6, 7, 8 and
10 were.

**Pre-registered interpretation rule, set now to prevent motivated reading later.** If the full model beats
frequency but ablations 1, 3, 5 and 6 do not cost anything, then the architecture is not what produced the gain
and the honest report is "a transformer with capacity beat the baseline; none of its biology mattered". That
outcome will be stated in exactly those words if it occurs.

---

## 4. What this cannot do, stated in advance

- It will not learn the one-off tail. 40.6% of responsive genes move for a single perturbation; no model learns a
  rule from one instance.
- It will not fix the essentiality ceiling. That is 0.561 and structural (build 10), and no amount of
  representation learning changes which reactions carry flux.
- It is trained on 5,120 examples on a CPU. If it beats frequency, the margin will be modest, and the ablations
  matter more than the margin.

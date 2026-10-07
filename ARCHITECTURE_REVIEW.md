# Review of the hierarchical adaptive-factorization cell engine

Six independent analytic lenses, each finding adversarially refuted, then synthesised.
37 agents, 0 errors. Every number below was verified directly against a file in this repo
before being written down; the agent findings themselves are not taken on trust.

## 1. Correction to an earlier statement of mine

I previously wrote that section 30's three inequalities are "already contradicted by
measurement". That is too strong, and the refutation round narrowed it correctly:

- **Refuted, for the metabolic layer only.** Human-GEM's 147 subsystems share 60.0% of species;
  median boundary fraction **0.821**, rising to **0.849** when refined to subsystem x compartment.
  `NOTES_rem_cell.md`: *"Finer partitions have proportionally more boundary, not less."*
  Metabolism does not decompose into narrow-interface groups by any criterion tried here.
- **Unmeasured, not refuted, globally.** The intersection carrying full dynamical state AND a
  metabolic reaction AND a signed TF regulator AND a surviving k_cat is **87 genes, 2.49% of
  proteome mass**. The three inequalities cannot be evaluated at scale because almost nothing in
  the model carries enough state to evaluate them on.

Both point the same way, but the epistemic status differs and the distinction matters.

## 2. The deepest defect, and I missed it

**`eps_G` has no producer.** The per-group error budget is CONSUMED by five separate sections --
the representation selector (6-11), merge (12), split (14), memory priority (22) and the error
budget (23) -- and PRODUCED by none of the thirty.

> An adaptive method without an a posteriori error estimator is not adaptive. It is arbitrary
> with extra bookkeeping.

This is the correct one-line verdict on the design, and it is sharper than anything the
tail-versus-mean analysis produced. The quantum analogy is what conceals it: dynamic entanglement
grouping works in MPS/DMRG because the truncation error is *computed* -- discarded Schmidt
coefficients give a certified bound and the bond dimension is exactly the cost. `C_ij = w_I I +
w_S S + w_U U + w_N N + w_E E` has no error theorem, no certificate and no relation to cost. The
metaphor is not decorative; it implies a guarantee the replacement heuristic does not carry.

This repo's own `rem/clusters.py` docstring already states the correct correspondence:
entanglement across a cut, bond dimension, edges crossing the cut and treewidth are the same
number. Section 4 is not that quantity.

## 3. Two measured facts that break specific sections

**Section 17's causal frontier is a linearization used 50x outside its measured validity.**
`rem/reach.py:194-195`: the module *"failed linearity at eps = 0.10 (worst deviation 0.0225
against a 0.01 bar)"*; the largest perturbation at which response is linear to 1% is **eps = 0.02**.
A transcriptional burst is a factor-of-two change, eps ~ 1. R1 was rewritten as a precondition
rather than a report for exactly this reason.

**Section 23's additive error bound has no valid horizon where the loop gain exceeds 1.**
`outputs/loop_growth_loop.json` g3: `F_prime = -1.1187`, so |F'| = **1.1187** and the fixed point
REPELS. Section 23 sums per-group errors as if independent, but groups are split precisely
*because* their residual coupling was judged small -- so the ignored correlations are the ones the
split created. Above unit loop gain they are amplified across the cut without bound.

## 4. What the repo already knew, and it is worse than the architecture review

`NOTES_rem_cell.md` section 4.3 lists fatal objections that no part of this review raised:

- **Four of seven propagation links are at or below their own nulls.** Kinase -> TF-substrate
  co-dependency 0.9713, CI [0.8716, 1.0839]; the sham control on random pairs scores 0.9959, so
  the real edges perform *below random*. Power is demonstrated in the same file (BioPlex 3.0
  scores 1.0884 on 75,639 pairs through identical machinery).
- **TF -> transcription is worse than a coin.** Real edge signs AUC 0.5465 against shuffled signs
  0.5494 +/- 0.0079, with publication count at 0.5536 beating both.
- **The chromatin polymer scores below a distance-only null** (0.8229 vs 0.8283), consumes ~60% of
  the cost model's wall time, and its boundary message has no subscribers.
- **The model cannot be validated even if built.** The persistence bar exists on one channel
  covering 2.3% of rows; loop 197 queried 5,792 released human ENCODE experiments and found
  nothing above 4 matched points outside the A549 positive control.

## 5. What survives as genuinely new

Not the coupling score, and not the entanglement metaphor. The best idea in the document is the
one it does not advertise: **splitting on the requested observable** (14-15) -- build the active
subgraph by backward relevance from Y and split when `|Y_coupled - Y_split|` is small. That is
dual-weighted-residual goal-oriented adaptivity (Becker-Rannacher), mature in finite elements and
essentially absent from cell simulation, where adaptivity is almost always state-error-driven.
Transplanting it is a real contribution.

Everything else -- event-driven multi-timescale scheduling (Gibson-Bruck 2000), the frontier
(DRGEP path-flux propagation, term for term), representation ladders, hierarchical
zoom-and-compress, priority degradation under a memory cap -- is standard practice in its home
field and should be presented as integration, not discovery. Dynamic grouping for the CME is
published as rank-adaptive tensor-train work, with a *better* merge criterion.

## 6. The load-bearing assumption, restated

Not modularity. **Detectable** modularity:

> There exists a function of quantities computable at cost o(C_exact) whose value certifies the
> error of a proposed factorization.

Modularity you cannot locate buys nothing. This is falsifiable but not verifiable: it can be
killed on systems small enough to have exact ground truth, and cannot be confirmed at whole-cell
scale, because confirming it requires the reference solve the design exists to avoid.

## 7. One piece of the missing estimator, measured

`rem/atlas/grouping.py` and `grouping_law.py` supply a partial producer for the `eps_G` that
section 4 lacks, for one observable class. On an exact joint with ground truth and a driver whose
stationary mean and variance are invariant to 7.25e-15:

    tail_err = c * sqrt(MI),    c -> 20.23

Derivable rather than fitted: a mutual information is `-0.5 log(1-rho^2) ~ rho^2/2`, QUADRATIC in
the coupling, because it averages under P; a tail lift is LINEAR in it. So section 14's threshold
is wrong by a square root, always in the unsafe direction. To hold conjunctive tail error below
1e-2 requires `MI < 2.4e-7`, not `MI < 1e-2` -- a factor of 40,912.

Measured at a threshold calibrated the way the architecture would calibrate it (tau = 0.001352,
the largest MI keeping the joint MEAN accurate to 1%): the joint TAIL is wrong by 47.6%, a 59x
amplification from the quantity the threshold controls to the quantity it does not.

## 8. Honest limits of this review

- **Zero findings survived the adversarial round, and I built that bias in.** The refuters were
  instructed to "default to refuted=true if uncertain", which biases toward killing findings. The
  synthesis agent flagged it unprompted: *"The survivor list handed to me is empty. That is a fact
  about the refutation round, not a clean bill of health."* The seven defects it then raised were
  never put to a refuter at all.
- The square-root law is measured on one system, one tail depth, one coupling mechanism. The
  exponent argument is general for weak dependence; the constant `c = 20.23` is not.
- Three gates in the supporting build failed, two of them because I set bars that could not be met
  by any outcome. Recorded in `rem/atlas/RESULTS_grouping.txt` and the commit log.

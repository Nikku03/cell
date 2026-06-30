# TF→operator, specificity-ordered: where sequence prediction works (E. coli)

Strategy (as proposed): learn each TF's operator, count how many genome
promoters match it (N_sites = "positions to attach to"), rank ascending (most
specific first), and validate target prediction against RegulonDB at each level.
Leak-free (PWM from train-half targets, scored on held-out half).

## The result splits TFs into two clean regimes

| regime | TFs | mean N_sites | mean precision | mean recall |
|---|---|---|---|---|
| **very specific (≤100 sites)** | 20 | 33 | **0.33** | 0.04 |
| specific (100–400) | 10 | 189 | 0.07 | 0.05 |
| moderate (400–1000) | 10 | 732 | 0.03 | 0.20 |
| **degenerate (>1000 sites)** | 23 | 2974 | 0.03 | **0.68** |

It is a precision/recall seesaw driven entirely by operator information content:
- **Specific operators (high info, ~13–15 bits) → few genome matches → high
  precision.** When they hit, it's usually a real target. They miss many targets
  (low recall) because the match bar is strict.
- **Degenerate operators (low info, ~7–9 bits) → match almost everywhere → recall
  ~1.0 but precision = base rate (useless).**

## The specific end — your "fewest positions, perfectly aligned" wins

| TF | N_sites | info (bits) | precision |
|---|---|---|---|
| gntR | 9 | 13.8 | **0.78** |
| qseB | 10 | 13.4 | 0.60 |
| cytR | 11 | 13.2 | 0.55 |
| torR | 11 | 14.8 | 0.55 |
| trpR | 16 | 14.4 | 0.44 |
| paaX | 16 | 13.2 | 0.44 |
| metJ | 20 | 14.2 | 0.35 |
| tyrR | 21 | 14.5 | 0.33 |

These are exactly the specific metabolic/local repressors (gntR, cytR, trpR,
metJ, tyrR, torR…). Their operators carry enough information that you can place
their handful of sites from sequence with real precision (0.3–0.8) — a genuine,
usable win.

## The degenerate end — the global regulators, unpredictable from sequence

CRP (N_sites 4669/4689, info 7.0), FNR (4675, 8.0), Fis (4674, 8.4), IHF (4607,
8.2), H-NS (4634, 9.4), ArcA (4662, 7.9), Fur (4661, 9.3). Their operators match
*essentially every promoter*. Recall is 1.0 only because they "predict" the whole
genome — precision is the base rate. These are the nucleoid-shaping / global
catabolite regulators, and sequence cannot pin them. (They are also the regulators
of the "universal/housekeeping" background — the same global set hits everything,
which is why those genes can't be assigned a *specific* TF from sequence.)

## The synthesis — which route per TF
The specificity rank gives a decision rule for building the regulatory layer:
- **Specific-operator TFs (~20–30, N_sites ≲ 100, info ≳ 13 bits):** place their
  sites directly from sequence (precision 0.3–0.8). Do these first, ascending.
- **Global/degenerate TFs (CRP, FNR, IHF, H-NS, Fis, ArcA, Fur, σ70 background):**
  sequence is hopeless — use the **functional edge** (co-expression + co-fitness,
  AUC 0.63) and treat them as the constitutive/global layer.

So the honest, working pipeline is hybrid: **sequence for the specific minority,
functional data for the global majority.** The specificity number tells you,
per TF, which bucket it's in — no guessing.

Files: colab/tf_specificity_rank.py, outputs/orphan/tf_specificity_rank.{png,json}.

# Transformer design: from "classify essentiality" to "complete the cell"

## 0. The trap, and why it tells us the answer

The obvious build — a transformer that eats W1,W2,W3,W4 scores and outputs
essential/not — **cannot work, and the reason is diagnostic.** Our logistic
regression on those four numbers already sits at AUC ~0.92 / essCov ~0.50. Four
scalars carry almost no interaction structure; a transformer over four numbers
is a heater. We proved repeatedly that the ceiling is **missing data**, not
missing model capacity. So adding capacity on the *outputs* is wasted.

But look at *which* data is missing vs abundant:

| signal | examples | nature |
|---|---|---|
| binary essentiality labels | ~210k, partly feba-derived/noisy | **scarce, dirty** |
| W3 FBA | 15 orgs | scarce |
| **W4 fitness scores** | **33.8M (gene × condition), graded, clean** | **ABUNDANT** |

The whole project treated all four as small datasets to ensemble. **That is the
mistake.** One of them — the raw fitness tensor — is enormous and clean. The
genius move is to stop predicting the scarce label and start **completing the
abundant tensor**, then *derive* essentiality as a readout.

## 1. The reframe: tensor completion, not classification

Define the object the cell actually is:

```
F[organism, gene, condition] = fitness defect when this gene is disrupted,
                               in this organism, in this condition
```

33.8M entries observed (48 orgs × ~150 conditions × thousands of genes), the
vast majority unobserved. **The model's job is to complete this tensor.**

Essentiality is then *derived*, not predicted:
- **always-essential** = F strongly negative across ALL conditions
- **conditionally-essential** = F negative in some conditions, ~0 in others
- **dispensable** = F ≈ 0 everywhere

This single reframe fixes four things at once:
1. **Scarcity → abundance.** Training target is 33.8M graded values, not 210k
   dirty binaries. Representation learning rides the dense signal.
2. **Conditionality becomes native.** The output IS condition-resolved. The
   ~69% conditional soup we could never crack is now the model's *primary* job,
   not an afterthought.
3. **All 4 wheels unify** (see §3): W1/W3 become encoder inputs, W4 becomes the
   target, W2 becomes emergent.
4. **Leakage controllable.** Train on raw `fit` scores; evaluate essentiality on
   *independent* truth (DeJesus mtub, published Keio). The dense target is not
   the dirty label, so circularity is broken by construction.

## 2. The architecture — three levels, each justified by a finding

### Level 1 — Genome-context gene encoder ("genes as tokens")
- Each gene → a token. Token features:
  - **ESM-2 protein embedding** (frozen backbone, 35M model). *Why:* gives
    cross-organism generalization at the sequence level — a new organism's
    proteins are in-distribution because ESM saw millions of proteins. This is
    how we beat W1's degradation on novel organisms.
  - genomic coordinate / strand / intergenic gap (positional).
  - family/EC/domain annotation tokens (the W3 metabolic context, the TF family).
- A transformer (~6 layers) attends **across the ordered genome**.
  - *Why ordered-genome attention:* it natively captures operon structure and
    **local-regulator adjacency** — the *one* regulatory signal we proved is
    recoverable (precision 0.51 on divergent pairs). The model learns it instead
    of us hand-coding adjacency. The global-regulator edges (Wunderlich-Mirny
    wall) stay unrecoverable — the model won't hallucinate them because they
    aren't in the input.
- Output: a context-aware gene vector `g`.

### Level 2 — Compositional condition encoder
- A condition = its media composition. *We have `MediaComponents`: 12,308 rows.*
- Each compound → learned embedding; condition = attention-pooled **set** of
  compound embeddings + categoricals (aerobic, temperature, stress).
  - *Why compositional (set of compounds), not a condition-ID lookup:* a
    condition never seen in training (LB + new drug) is encodable as
    (LB compounds) ∪ (drug) → the model generalizes **across conditions**, which
    a lookup table fundamentally cannot. This is the analogue of ESM for
    chemistry.
- Output: condition vector `c`.

### Level 3 — Cross-attention completion core + multi-task heads
- Gene `g` cross-attends to the condition's compound set `c`.
  - *Why this is beautiful:* the attention LEARNS gene↔nutrient couplings — a
    biotin-synthesis gene attends to "biotin present" and predicts high fitness
    (dispensable) when it is, low when absent. **This is the metabolic gap-fill
    logic (W3) learned from data instead of hand-coded**, and it extends to
    non-metabolic couplings (efflux gene ↔ drug present) we could never model.
- Small per-organism bias embedding (GC, genome size, clade) — captures
  baseline; for a novel organism set to nearest-neighbor or learned default.
- **Heads (multi-task):**
  1. **Fitness regression** (PRIMARY): predict `F(gene,condition)`. Huber loss,
     weighted by the measurement's |t| (trust strong measurements). 33.8M
     examples carry all representation learning.
  2. **Essentiality** (auxiliary): pooled readout → P(essential). Supervised by
     *independent* labels only. Small weight; it calibrates, doesn't drive.
  3. **Cell-layout category** (auxiliary): chassis / metabolic / regulator /
     transport / conditional. Supervised by our §assembly layer assignment.
     Injects the structural prior we validated (the generalizable 31%).

## 3. How all four wheels dissolve into one model

| wheel | role in the transformer |
|---|---|
| W1 sequence | the ESM gene-encoder backbone (Level 1) |
| W2 conservation | **emergent** — orthologous genes get similar ESM embeddings → similar predictions, *without* an explicit conservation feature. (And we proved transfer = conservation, so we don't hard-wire it.) |
| W3 FBA / metabolic | annotation tokens on the gene + the gene↔nutrient cross-attention learns necessity from data |
| W4 fitness | the **training target** (Level 3 primary head) |

The "4 wheels of limited data" problem evaporates because we stop trying to
*learn from* four small sets. One big set (W4) teaches the representation; the
others become structure (inputs, priors, emergent geometry).

## 4. Stress-testing the design (what breaks, honestly)

- **Novel-organism generalization:** works for genes resembling known protein
  families (the conserved core + most metabolism) because ESM has seen them.
  **Fails for true orphans** — no escape; but we don't make it worse, and the
  layout head still tags "looks structural."
- **Condition extrapolation:** compositional encoder makes "far" smaller, but a
  condition chemically unlike anything in training is still extrapolation.
  Honest limit.
- **The conditional 69%:** this is the part the model can newly reach — but only
  for conditions *near the measured ones*. It cannot invent the fitness of a gene
  in a condition no relative was ever tested in. The data ceiling becomes a
  *condition-coverage* ceiling, which is softer (compositional) but real.
- **Will it beat the logistic on the static 60-org call?** Modestly (+0.03–0.06
  AUC), because that call is near the data ceiling. **The real wins are new
  capabilities the logistic cannot do at all:** condition-resolved fitness, and
  higher novel-organism floor via ESM+context. We should sell it on those, not
  on beating 0.92.

## 5. Eval protocol (leakage-proof)

- **Held-out ORGANISMS** (leave-organism-out): the only honest test of
  generalization. Train on 47 feba orgs, predict the 48th's fitness + essentiality.
- **Held-out CONDITIONS**: mask whole conditions, test compositional generalization.
- **Independent essentiality**: DeJesus mtub, published Keio — never the
  feba-derived labels for the final number.
- Report: fitness regression R per held-out org; essentiality AUC/essCov on
  independent truth; explicit "seen vs novel organism" split.

## 6. Staged build with go/no-go gates (don't overbuild)

- **Stage 0** (1 day): precompute ESM-2 embeddings for ~288k genes; tensorize
  feba fitness + MediaComponents into training triples.
- **Stage 1 — sanity** (2 days): simplest completion — ESM(gene) + compound-bag →
  MLP → fitness. **GATE:** does it beat the per-gene-mean and per-condition-mean
  baselines on held-out conditions? If not, the framing is wrong — stop.
- **Stage 2 — genome context** (3 days): add the genes-as-tokens transformer.
  **GATE:** does context raise held-ORGANISM essentiality AUC above the current
  logistic's novel-org estimate (~0.65)?
- **Stage 3 — compositionality** (3 days): set-attention condition encoder +
  cross-attention. **GATE:** held-out-CONDITION fitness R > 0.3.
- **Stage 4 — multitask + clean eval** (2 days): add essentiality + layout heads;
  final leakage-proof numbers on DeJesus.

Each gate is a real off-ramp. If Stage 1 fails the baseline, we learned the
tensor isn't completable from sequence+chemistry and we stop — cheaply.

## 7. The one-sentence design

**Don't build a classifier over four scores; build a gene×condition fitness-tensor
completion transformer — ESM-encoded genes in genomic context, compositionally-
encoded conditions, cross-attended — trained on the 33.8M dense fitness
measurements, with essentiality and cell-layout as derived heads. It turns our
scarce-label problem into a data-rich self-supervised one, makes conditionality
native, absorbs all four wheels, and is honestly bounded by condition-coverage
rather than model capacity.**

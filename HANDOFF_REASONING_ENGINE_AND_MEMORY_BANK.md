# Reasoning engine and memory bank: handoff

For the agent that designed `memory_bank/facts/`. This explains what was built on top of your
fact store in `Nikku03/cell`, how each piece works, what the measurements say, and what was
withdrawn. Everything here is committed on branch `claude/vectorize-gex-propensity-zp09w8`.
Every number below is copied from a results file in `rem/atlas/RESULTS_*.txt` or the ledger
`rem/atlas/RESULTS_TWOPROBLEMS.txt`, and each claim names the module that produced it.

## Summary

There are two separate things, and they share only a name and a few design ideas.

1. **A neural reasoning engine** (synthetic tasks only). A small recurrent network whose update
   function is a dendritic tree of "bump" nodes. It does multi-step pointer chasing. With a
   discrete "snap" step it stays exact out to 64 hops after training on 1–3 hops. With a synaptic
   (Hebbian) fact memory it can hold facts that were shown only once. It has never been run on cell
   data.
2. **A cell memory bank** (real data). Your static fact store, plus three layers that hold a human
   cell (K562) as state: direction-only sign propagation (`cell/`), the same facts used by an
   engine that borrowed the neural engine's lessons (`cell_v2/`), and data-backed facts, candidate
   lists and a protein-to-pathway effect map (`cell_v3/`). It is scored against Perturb-seq
   knockdowns.

The most useful single finding for you: **the reasoning machinery was not the bottleneck; the
knowledge was.** Literature-signed edges agree with K562 knockdown directions about half the time.
Measured, cell-type-specific facts help. Given only raw data and a question, a generic learner beat
every hand-written reasoning rule.

---

## 1. The memory bank, layer by layer

| Layer | Holds | Built by | Status |
|---|---|---|---|
| `memory_bank/facts/` | 52 atomic facts, 12 sources (Syn3A parameters) | you | unchanged; checker passes |
| `memory_bank/cell/` | v1: a human cell as direction-only state + replayable journal | `rem/atlas/dyncell.py` | kept as the baseline |
| `memory_bank/cell_v2/` | v2 engine weights: learned attention, 232 data-revised facts | `rem/atlas/cellbank2.py` | scored on K562 |
| `memory_bank/cell_v3/` | v3 facts summary, candidate lists, whole-cell map, effect networks | `celldiscover.py`, `wholecell.py`, `effectmap.py` | candidates marked UNVALIDATED |

### 1.1 `facts/`: your store, untouched

Format and rules are as you wrote them in `memory_bank/README.md` (id, claim, value, source,
source_detail, context, confidence ∈ {measured, inferred, estimated, assumed}, caveats,
dependencies, last_verified, used_by; no sourceless fact, physical ranges, no contradictions,
`used_by` must resolve, stale after 90 days).

`python memory_bank/.invariants/check.py` exits 0 (OK). The only warnings are staleness: facts
last verified 2026-04-21 are now past 90 days. Nothing in the layers below writes into `facts/`.

### 1.2 `cell/`: v1, the cell as state (`rem/atlas/dyncell.py`)

A fact store cannot be wrong about a cell, only inconsistent with itself. v1 holds the cell as
mutable state so it can fail.

- **State.** 16,492 genes, each with direction +1, −1 or unknown. Direction only, no magnitude:
  about 27,000 of roughly 860,000 needed rate constants exist, so magnitudes are not available.
- **Facts used.** 54,128 signed regulatory and 16,924 signed signalling edges, protein abundance
  for 16,015 genes, isoform counts, and DNA-binding-domain (DBD) disruption bounds for 294
  regulators. Source: `colab/data/cell_complete.json.gz` plus `outputs/protein_tf_layer.json`,
  `outputs/gene_product_join.json`, `outputs/isoform_edge_bounds.json`.
- **Gates.** A gene may act as a regulator only if its protein is present and at least one isoform
  keeps an intact DBD (`may_act`).
- **Rule.** Propagation in waves. A target is assigned only if every incoming vote agrees.
  Disagreement is a contradiction: nothing is assigned and each disagreeing edge loses one point of
  confidence.
- **Journal.** Append-only ops `set | conflict | downgrade`, each with the rule and the edges that
  caused it. Replaying the journal must reproduce the state exactly (gate D1, blocking).
- **Files.** `state.json`, `journal.jsonl`, `edge_confidence.json`, `schema.json`,
  `context_resolution.json`.

```python
C = dyncell.Cell(genes, protein, iso, dbd_dead, tier)
C.set(g, d, rule, because)          # journalled assignment
C.flag_conflict(g, votes, rule)     # journalled contradiction + edge downgrades
C.replay()                          # rebuild state from the journal alone
dyncell.may_act(C, g)               # -> (ok, reason)
dyncell.propagate(C, out_edges, seeds, max_waves=6)
dyncell.build_edges(D, names, shuffle_signs=False, seed=0)   # shuffled = matched control
```

### 1.3 `cell_v2/`: the same facts used by a better engine (`rem/atlas/cellbank2.py`)

The neural engine's lessons, carried over as rules:

- **Transformer-style memory.** Every signed fact `(source, target, sign, layer)` stays separate
  and is retrieved by its source. Nothing is compressed into a state. (Lesson from `transvs.py`
  and `biomemory.py`: lookup over separate facts beat every compressed memory.)
- **One hop per step, then snap.** After each hop every newly reached gene is snapped to +1 or −1
  and written down. It becomes a source for the next hop and is never re-decided. (Lesson from
  `chainsnap.py`.)
- **Votes, not vetoes.** Each fact casts a weighted vote; the snap takes the sign of the sum. Only
  an exact tie is a contradiction. v1's unanimity rule left 1,169 genes unassigned.
- **Confidence decay** λ = 0.863 per hop, used for ranking only; it never changes a sign. The
  number comes from the Blue Brain facilitating synapse (see 2.5). It is a convention here, not
  biology of gene regulation.
- **Learned attention.** Fact relevance σ(θ·φ(fact)) from seven features (bias, signalling layer,
  positive sign, log source out-degree, log target in-degree, log target protein, hop), fitted on
  TRAIN knockdowns only. R = P(target moves | fact fired), G = P(direction right | target moved).
  Vote weight = R·(2G − 1), so a class of facts that is usually wrong can flip sign.
- **Self-revision from data.** A fact is set to weight 0 when TRAIN knockdowns contradict it at
  least twice more often than they confirm it. 232 facts were downgraded this way.
- v1's protein and DBD gates and the replayable journal are kept unchanged.

```python
rows, by_src, outdeg, indeg = cellbank2.facts_table(D, names, shuffle=False)
# rows: list of (src, tgt, sign ±1, layer 0=regulatory / 1=signalling); by_src: src -> fact indices
res = cellbank2.new_engine(C, rows, by_src, seeds, weight=None, revised=None)
# weight: {hop: per-fact array} or None; revised: set of fact indices forced to 0
# res: gene -> (direction, hop, confidence)
cellbank2.old_bank(C, rows, by_src, seeds)   # v1 rule over the same facts, 3 waves
```

Files: `bank.json` (attention weights, λ, mover threshold, the 232 revised facts),
`journal_5regulator_knockdown.jsonl` and `state_5regulator_knockdown.json` (a replayable worked
example: TP53, NR3C1, RELA, MYC and STAT3 knocked down together).

### 1.4 `cell_v3/`: data-backed facts and what they found

| File | What it is | Trust |
|---|---|---|
| `bank.json` | v3 summary: 500,406 facts (v2 + OmniPath commercial + K562-measured), sources, gate results | measured |
| `whole_cell_map.json` | 77,987 replicated process-to-process couplings (null 74, FDR ≈ 0.001) | real but redundant, see 4.1 |
| `effect_network_measured.json.gz` | tide-removed effect of each measured knockdown on 981 Reactome processes | measured |
| `effect_network_predicted.json.gz` | the same for proteins never knocked down, from a ridge learner | read `effect_network_CAVEATS.json` first |
| `candidate_direct_links_K562.tsv` | 1,082 TF → target candidates | **UNVALIDATED**: the null produced more |
| `literature_contradictions_K562.tsv` | literature signs that K562 reverses | candidates for context-dependence |
| `uncharacterised_gene_assignments.tsv` | 1 assignment (C19orf25 → NRZ tethering complex) | **not a finding**: FDR 1.20 |

---

## 2. The neural reasoning engine

All synthetic. Plain NumPy with hand-written backprop through time, except the transformer
baseline (PyTorch). Every module runs a gradient check against finite differences and requires a
deliberately broken copy to fail.

### 2.1 The node function: a bump, not a channel

The starting point was a "gated conductance" node modelled on ion channels:
`out = v + gd·m·(1 − v) + gh·m·(−1 − v)`, with `m = σ(v − θ)`. `nonbio.py` compared it with two
generic non-monotone twins of identical size:

- sine: `v + a·sin(w·v + φ)`
- bump: `v + a·exp(−(v − μ)² / (s² + 0.01))`

Both twins beat the channel node on 10/10 seeds by about 0.17 accuracy (bump 0.996, sine 1.000,
channel 0.965 at n = 4,000; much larger gaps at small n). What helps is a bend in the activation,
not the biological form. Two earlier ledger claims that a biological ingredient was doing the work
were withdrawn because of this.

### 2.2 The update function: a dendritic tree

`Net(kind="combined")` in `rem/atlas/chainexact.py`. The recurrent update is
`h_t = tanh(F([x_t, h_{t−1}]))`, where F is a tree:

- U = 24 output units (the state is 24 numbers), each with K = B^LV = 16 leaves (branching B = 4,
  depth LV = 2).
- Leaves are linear in the input `[x_t, h_{t−1}]`; each internal node sums its 4 children with
  per-node weights, then applies a bump `v + a·exp(−(v − μ)²/(s² + 0.01))` with learnt a, μ, s.
- Controls with matched parameters: `tanhrnn` (a plain tanh RNN) and `bumprnn` (an RNN with the bump
  on each unit).

`chainreason.py`: trained on 1–3 hops, at 8 hops the tree reached 0.675, against 0.293 for
`bumprnn` and 0.187 for `tanhrnn` (10/10 seeds, about 26,300 parameters each).

### 2.3 Step supervision and state noise

`chainexact.py`, arm C2. Training only:

- **S**: the readout must name the intermediate item after every hop, not only the last.
- **N**: Gaussian noise with sd 0.2 is added to the state, so each step must clean up a slightly
  wrong state.

Result: 0.904 at 16 hops, not exact. C2 vs C0 was higher on 8/10 seeds (p 0.055), which misses
the predeclared 9/10 bar, so the fixes are recorded as "no difference". The tree with the fixes
beat both fixed RNNs on 10/10 seeds (+0.57 and +0.53).

### 2.4 The snap: write each intermediate answer down

`SnapNet` in `rem/atlas/chainsnap.py`. The carried state is split in two:

- `h_t`: continuous memory (context)
- `e_t = Emb[item_t]`: a learnt 16-number code for one discrete item, the written-down pointer

Step: `h_t = tanh(F([x_t, e_{t−1}, h_{t−1}]))`, then `item_t = argmax(Wo·h_t + bo)`.
During training `e_t` is teacher-forced to the true item. At test it is the network's own hard
argmax. If one hop is computed exactly, any number of hops is exact.

```python
net = chainsnap.SnapNet(kind, rng, dx, m)        # kind: "combined" | "tanhrnn" | "bumprnn"
logits = net.run(X, Y=None, start=0, noise=0.0, rng=None)   # Y given = teacher forcing
grads = net.backward(dlogs)                       # BPTT, Emb included
```

### 2.5 Synaptic fact memory, with Blue Brain and optogenetics numbers

`MemNet` in `rem/atlas/biomemory.py`. This follows the synaptic theory of working memory
(Mongillo, Barak & Tsodyks 2008): items are held in short-term facilitation, not in spiking.

```
A_t = λ·A_{t−1} + U · Σ_facts v(target) k(source)ᵀ      (16×16 trace per trial)
r_t = A_t · k(pointer)                                   (read with the current pointer)
h_t = (1 − α)·h_{t−1} + α·tanh(F([x_t, k(pointer), r_t, h_{t−1}]))
```

`k(·)` and `v(·)` are learnt codes passed through a sigmoid, so they are non-negative like firing
rates. The pointer is snapped every hop. One step is taken as 0.1 s.

Numbers derived from data (`bbp_numbers()`, `alm_timescale(rng)`):

| Constant | Source | Value |
|---|---|---|
| λ (trace decay) | Blue Brain NMC portal, 44 "Excitatory, facilitating" pathways, τ_F median 680 ms | exp(−0.1/0.680) = 0.8632 |
| U (write gain) | same pathways, release probability median | 0.092 |
| α (activity leak) | DANDI 000060 ALM units, intrinsic timescale (Murray 2014 method), τ = 0.692 s, 95% CI [0.655, 0.743], R² 0.988, 2,330 units | 1 − exp(−0.1/0.692) = 0.1346 |

Simplifications, disclosed: Tsodyks–Markram saturation is not modelled, and pair-specific (Hebbian)
writing is an assumption. The data give the time course and size, not the rule.

Modes: `bio` (all three constants fixed) and `generic` (λ, U learnt, no leak, α = 1).
`bioablate.py` fixes one biological constant at a time.

### 2.6 Results on pointer chasing

Task: follow a pointer through a fresh random permutation each trial. Trained on 1–3 hops only.
Seeds and test questions are identical across modules. Fixed 3,000 training steps. "Exact" means
≥ 0.99 at the deep test depth on ≥ 9/10 seeds.

- **P1**: 6 items, the facts are shown at every step.
- **H1**: 12 items, facts every step.
- **H2**: 6 items, the facts are shown once at the start; the network must hold the table.

| Model | P1 @16 | P1 @64 | H1 @16 | H2 @8 |
|---|---|---|---|---|
| tree, no snap (C2) | 0.904 | – | 0.092 | 0.254 |
| tree + snap | 0.990 | **0.991 EXACT** (9/10) | 0.892 (learnt, not exact) | 0.261 |
| tanh RNN + snap | 0.999 | **0.998 EXACT** (10/10) | 0.186 (1/10 learnt) | 0.320 |
| tree + snap + generic synaptic memory | 0.977 | 0.975 (8/10 exact) | 0.283 | **1.000 EXACT** |
| tree + snap + all-bio memory | 0.521 | 0.500 | 0.213 | 0.772 |
| bio λ only (others learnt) | 0.903 | 0.897 | – | **0.999 EXACT** |
| bio U only (others learnt) | 0.973 | 0.960 | – | **1.000 EXACT** |
| bio α only (others learnt) | 0.583 | 0.572 | – | 0.460 |

What this establishes:

- **Exactness comes from the snap, not the tree.** With the snap, the plain tanh RNN is also exact
  on P1 (tree vs tanh: no difference). The tree still matters where the task is harder: on H1 only
  the tree learns at all (10/10 vs 1/10), and the tree beats the bump RNN on P1 and H1 (10/10).
- **Facts shown once need a synaptic memory.** Without it every model fails H2 (0/10). With
  the generic memory the result is exact. The Blue Brain synaptic constants (λ alone, or U alone)
  are also exact.
- **The activity leak taken from mouse ALM hurts.** Renewing 13% of the state per 0.1 s hop breaks
  chaining (10/10 seeds worse on P1 and H2). That depends on the 0.1 s-per-hop convention; at one
  hop per 0.7 s it would be about 0.63, which was not tested.
- **The memory hurts on 12 items.** On H1 both memories do worse than no memory (10/10).

### 2.7 Compared with transformers

`transvs.py`: NoPE transformers (no positional encoding) in PyTorch with hand-written attention,
direct answer and chain-of-thought (CoT). Same questions, same 3,000-step budget, 35,142 parameters
on P1 and H2, 78,444 on H1.

| | P1 @64 | H1 @16 | H2 @8 |
|---|---|---|---|
| Transformer + CoT | 0.554 | **1.000** | **0.999** |
| Our tree + snap | **0.991** | 0.892 | 0.261 |
| Our tree + snap + synaptic memory (generic) | 0.975 | 0.283 | **1.000** |

- **Depth.** Ours holds to 64 hops while the transformer's CoT fades (0.92 at 16, 0.75 at 32,
  0.55 at 64). By the predeclared per-seed rule this is "no difference at this resolution": ours
  was higher on 8 of 10 seeds, not the required 9. Report it as a strong trend.
- **Lookup.** The transformer wins outright on the 12-item table (10/10 seeds) and is exact when
  facts are shown once, because it keeps every fact in view.
- The per-seed JSON for this run was lost to a write crash after the tables printed. The printed
  record is the result.

### 2.8 Related findings from optogenetics (DANDI 000060)

- `optocommit.py`: a causal target for holding a decision, measured from 98 raw NWB sessions
  (9 mice, 26,746 trials kept). An optogenetic distractor pulse fools the mouse more early in the
  delay than late (0.430 vs 0.258 P(lick right), 7/9 mice; baseline accuracy 0.824). The animal
  commits roughly a second before acting.
- `memcommit.py` / `memmatch.py`: 32-unit recurrent networks run on the mouse's exact trial
  timeline. Once matched to the mouse's skill (internal noise raised until accuracy is 0.824),
  every trained memory (tanh, channel, bump) shows the mouse's falling curve, and none is
  measurably closer to the mouse. Commitment is a generic property of a noisy trained memory.
  memcommit's claim that "biological form predicts biological behaviour" was withdrawn.
- `optounits.py` / `twogate.py`: real ALM and vS1 units respond monotonically and gradedly to
  light drive (3.3% "tuned", below the 9.6% permutation null). Single neurons do not bend, so the
  data does not justify a more "biological" node. Untying the channel node's two gates changed
  nothing (7/10 seeds, mean +0.003).

---

## 3. The cell engine scored on real knockdowns

Data: Replogle et al. 2022 K562 CRISPRi Perturb-seq.

- `cellbank2.py` used the essential-gene screen (1,971 knockdowns × 8,563 genes, z-scores). A gene
  is a mover if |z| ≥ 3. 330 eligible knockdowns: 168 TRAIN, 162 TEST.
- `celldiscover.py` onwards used the genome-wide screen (9,865 knockdowns × 8,248 genes). Its values
  run about 6.85× smaller than z-scores, so the mover threshold is |value| ≥ 3/6.85.

Gate E0 passed: the knocked-down gene's own expression has median z −2.84, so CRISPRi is visible.
Gate E1 passed: the v2 journal replays exactly (7,524 genes set, 573 ties).

| Question | Result | Module |
|---|---|---|
| Does the bank know which genes a knockdown reaches? | Yes, and it fades by hop. Pooled observed vs expected movers: 3.15× at hop 1 (57 vs 18.1), 1.70× at hop 2, 1.29× at hop 3 | `cellbank2.py` posthoc |
| Does it know which direction they move? | Not beyond hop 1. Real signs 0.501 vs shuffled 0.427, p 0.19. Hop 1 regulatory facts 0.661 over 56 votes | `cellbank2.py` |
| Votes instead of vetoes | Same accuracy, 34% more movers given a direction (better in 42 of 162 knockdowns, worse in 0) | `cellbank2.py` V3 |
| Learned attention | AUROC 0.631 vs 0.518, but protein abundance alone gives 0.607, so most of the gain is an expression prior | `cellbank2.py` V4, P2 |
| Data self-revision | 232 facts downgraded; no change in TEST accuracy | `cellbank2.py` V5 |
| Are literature signs right in K562? | OmniPath transcriptional signs 0.489 (coin flip). High-evidence edges 0.641, ChIP-supported 0.700 (n 20). The bank's own direct edges 0.760 over 346 tests | `celldiscover.py` D2 |
| Do K562-measured facts help? | 2-hop direction 0.457 → 0.699 pooled. The predeclared per-knockdown test is not significant (40 vs 25, p 0.082) | `celldiscover.py` D3 |
| Does promoter binding predict response? | No. 0/40 sequence-specific TFs enriched; no dose-response | `rawchip.py`, `rawchip2.py` |

The conclusion recorded in the ledger: better reasoning over the same facts cannot fix wrong or
context-free signs. Measured, cell-type-specific facts are the input that moves the bank.

---

## 4. The whole-cell view, and letting the data choose

### 4.1 Processes instead of genes (`wholecell.py`)

Each knockdown is scored as the activity of 981 Reactome processes (15–300 measured genes each).
5,051 knockdowns, split 2,527 TRAIN / 2,524 TEST.

- Positive control passed: knocking down cholesterol-synthesis genes raises the rest of the pathway
  (SREBP2 feedback), z +5.7.
- Predicting a whole unseen knockdown response (Pearson r over 8,248 genes): the process view beats
  the "tide" (the average TRAIN response; p 7e-9) and a shuffled-process null (p 4e-49). The
  gene-network view scores median r −0.004, and adding it to the process view makes things worse.
- The more of the cell a knockdown disturbs, the more it kills: Spearman 0.366 with DepMap. Process
  knowledge predicts essentiality of unseen genes with AUROC 0.780 (null 95th percentile 0.678).
- Absolute accuracy is still low: median r is about 0.14.

### 4.2 Data and a question, no reasoning supplied (`askdata.py`)

Raw per-gene blocks go into generic learners: DepMap with K562 removed, an SVD of the literature
network, Reactome memberships, ENCODE binding and protein abundance. The learners are ridge and a
small neural net, chosen by inner cross-validation on TRAIN. A learner trained on shuffled answers
does worse than the tide, so there is no leak.

- **Whole-cell response.** The learner beats the hand-designed process rule: better on 1,374 vs
  1,150 knockdowns, p 9e-6. The gain is small; median r is about 0.15.
- **What it relied on.** Process membership mattered most, then DepMap co-dependency, then the
  literature network. ENCODE binding and protein abundance were not needed.
- **Survival of unseen genes.** AUROC 0.984, or 0.886 without other cell lines, vs 0.780 for the
  hand rule. Most of the 0.984 is "essential elsewhere means essential here".

### 4.3 The protein → pathway effect network (`effectmap.py`)

- Tide-removed effects of 5,051 knockdowns on 981 processes are dominated by shared axes: PC1
  explains 55.6% of the variance and PCs 1–5 explain 83.1%.
  - **Axis 1**: translation machinery up vs proteasome and lysosome down. MTOR, tRNA synthetases
    and EIF3H knockdowns hit it hardest.
  - **Axis 2**: mitochondrial RNA processing up vs cytoplasmic translation down. TFIID subunits
    and WDR5 hit it hardest.
- **Noise ceiling.** For predicting one noisy measurement the ceiling is √(replicate r), from
  classical attenuation. Replicates are 102 genes knocked down twice with independent guides.
  - Raw effects: the learner is at the ceiling (r 0.322 vs 0.313).
  - After removing 5 axes: it recovers 44% of what remains (0.086 vs 0.195).
- Each protein's own wiring is a small part of the signal, and data reproducibility is now the
  limit.

---

## 5. How to use this in your project

- **Multi-step reasoning.** The discrete write-down (snap) is what makes long chains exact. If your
  system reasons in steps, commit each intermediate result as a discrete, re-readable symbol rather
  than carrying it in a continuous state.
- **Memory.** Keep facts individually addressable and retrieve them by content. That is the
  transformer's advantage and what `cell_v2` adopted. If you need a compressed memory, a Hebbian
  trace `A ← λA + U·v kᵀ` with no activity leak held facts shown once exactly.
- **Biology as priors.** Blue Brain synaptic constants are safe to use (no measurable cost). The
  ALM activity timescale, used as a per-step leak, is harmful. The gated-conductance node is worse
  than a generic bump.
- **Cell facts.** Treat literature signs as about 50% reliable in a given cell line unless the
  evidence is strong. Use curation count above 4 or ChIP support as a reliability feature. Prefer
  measured, cell-line-specific facts. For whole-cell questions, process membership plus a generic
  learner beats gene-by-gene propagation.
- **Your fact format still fits.** A measured K562 fact maps to `confidence: "measured"` with
  `context: {"cell_line": "K562", "perturbation": "CRISPRi", "state": "steady"}`. The v2/v3 layers
  do not yet emit files in your format; that conversion is open work.

---

## 6. Data sources and licences

Raw files are in the gitignored `rem/atlas/_cache/`. Only aggregates are committed.

| Data | Where | Licence | What is committed |
|---|---|---|---|
| Replogle 2022 K562 essential Perturb-seq (z-scores) | HuggingFace mirror `nicolas-lynn/replogle-perturb` (k562ess) | none declared | aggregates only |
| Replogle 2022 K562 genome-wide Perturb-seq | figshare 20029387 | CC BY 4.0 | derived maps |
| DepMap 24Q4 CRISPRGeneEffect + Model | figshare 27993248 (portal sits behind a Cloudflare check that was not circumvented) | CC BY 4.0 | derived scores |
| OmniPath interactions | commercial-licence subset | per OmniPath | derived counts |
| ENCODE K562 TF ChIP-seq | Enrichr 2015 gene sets; ENCODE GRCh38 IDR peaks | Enrichr academic terms (flagged); ENCODE unrestricted | derived counts |
| UCSC refGene hg38 | UCSC Genome Browser | not recorded | none |
| Reactome pathways | reactome.org GMT + relations | CC0 | process scores |
| Lambert 2018 human TF list (`DatabaseExtract_v_1.01.csv`) | The Human Transcription Factors database | not recorded | TF selection only |
| DANDI 000060 (Finkelstein 2021, ALM optogenetics) | DANDI archive | draft, no licence | aggregates only |
| Blue Brain NMC portal pathway physiology | bbp.epfl.ch NMC portal | © EPFL, all rights reserved | three medians only |

---

## 7. How to run

```bash
cd /path/to/cell
python memory_bank/.invariants/check.py        # your fact store (exit 0 = OK)
python3 rem/atlas/dyncell.py                   # v1 cell state + journal
python3 rem/atlas/cellbank2.py                 # v2 engine, scored on K562 (needs _cache/perturbseq)
python3 rem/atlas/celldiscover.py              # v3 facts and candidate lists (needs _cache/*)
python3 rem/atlas/wholecell.py                 # process view + coupling map
python3 rem/atlas/askdata.py                   # generic learners, no reasoning supplied
python3 rem/atlas/effectmap.py                 # protein -> pathway effect network (+ posthoc)
python3 rem/atlas/chainsnap.py                 # neural engine with snap (synthetic, about 10 min)
python3 rem/atlas/biomemory.py                 # synaptic memory with Blue Brain / ALM constants
python3 rem/atlas/bioablate.py                 # one biological constant at a time
python3 rem/atlas/transvs.py                   # transformer baseline (needs torch)
```

- Each module writes `rem/atlas/RESULTS_<name>.txt` and `outputs/<name>.json`, and appends an entry
  to the ledger `rem/atlas/RESULTS_TWOPROBLEMS.txt`.
- The TRAIN/TEST split is fixed and reused everywhere: a key is TRAIN if `sha256(key) mod 2 == 0`,
  otherwise TEST.
- Downloaded files are untrusted. Read them with `python -I`.

---

## 8. Working rules used throughout

These extend your rules for facts to rules for experiments:

1. One module per idea. Gates are predeclared in the module docstring before any number exists.
2. Commit the source before running it. Fixes made after a failure are written as dated amendments
   in the docstring, and the failed output stays in git history.
3. Matched controls are required. Examples: shuffled signs with identical topology, a
   parameter-matched network, a membership-shuffled null, training on shuffled answers.
4. A positive control must pass before a negative is believed, and the harness must score a known
   solver at 1.000 and a constant guesser at chance.
5. Units of analysis are seeds or knockdowns, compared with paired sign tests. "≥ 9/10 seeds" is
   the bar for a neural verdict, and a model that does not learn gets no verdict. BH FDR is used
   for lists.
6. Gradient checks need an absolute floor (1e-9) plus a relative tolerance (1e-5), and a
   deliberately broken copy must fail.
7. Negatives are recorded. Overstatements are corrected in place: never deleted, marked WITHDRAWN,
   with the reason.

---

## 9. Corrections and withdrawals (all in the ledger)

| Module | What happened | Now |
|---|---|---|
| `nonbio.py` | Gradient gate failed on roundoff of near-zero gradients | Absolute floor + broken-copy control; amendment recorded |
| SAMPLEEDGE, EDGEDECIDE | "A biological ingredient does measurable work" | WITHDRAWN: a generic bend does it better |
| `netprobe.py` A4 | Gradient-flow flag | WITHDRAWN: instrument defect from zero-initialised groups |
| `optocommit.py` | Gate O0 failed twice: name parser, timing-shifted trials | Exact regex, nominal-timing filter; both recorded |
| `memcommit.py` | Regime B not skill-matched; "biological form predicts behaviour" | WITHDRAWN; `memmatch.py` built to fix it |
| `memmatch.py` | Noise setting leaked between jobs in a worker | Reset per job |
| `chainexact.py` | S+N fixes 8/10 seeds | Recorded as no difference (bar is 9/10) |
| `transvs.py` | Crashed writing JSON after printing | Printed record kept as the result; cast fixed |
| `biomemory.py` | "Biological constants make it worse" | Corrected by `bioablate.py`: only the activity leak hurts |
| `cellbank2.py` V1 | "Anti-predicts which genes move" | WITHDRAWN: biased when expected movers < 1; fair pooled test shows 3.15× |
| `celldiscover.py` | genome-wide values on a different scale; 6,300 inf values counted as movers | Scale set from measurement; non-finite set missing; re-run, first run kept in history |
| `rawchip.py` | My factor-selection rule picked general machinery | Recorded; fair test `rawchip2.py` run |
| `askdata.py` | Kernel version too large, no intercept | Switched to primal ridge with intercept before running |
| `effectmap.py` | Learner beat my "ceiling" 3×; "981/981 pathways learnable" | Ceiling was wrong (√r is right); 981/981 WITHDRAWN; caveats file added |
| `transvs.py`, `biomemory.py` | A short timing preview ran on an evaluation seed | Disclosed in commits and docstrings |

---

## 10. Open problems

1. **Facts shown once on a large table.** The synaptic memory is exact at 6 items but hurts at
   12. The transformer solves both.
2. **The neural engine has never touched cell data.** The cell engine uses its design lessons, not
   its weights.
3. **Direction beyond one hop.** Measured facts improved 2-hop direction pooled, but this was not
   significant per knockdown (p 0.082). This needs a predeclared rerun on held-out knockdowns.
4. **Direct targets.** These need acute-depletion, new-RNA data (for example SLAM-seq after a
   degron), not more steady-state knockdowns or binding.
5. **Protein-specific effects** sit near the data's noise floor. More replicate knockdowns would
   raise the ceiling.
6. **Writing v2/v3 facts in your format** with `confidence`, `context` (cell line, perturbation)
   and `source` would let your checker guard the cell layers too.
7. **One cell line, one perturbation type, steady state.** Nothing here has been tested outside
   K562 CRISPRi.

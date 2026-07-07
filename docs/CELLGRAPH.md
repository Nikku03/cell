# CellGraph — a learned model of the whole cell

A learned representation of the 16,492-node multi-relational cell knowledge graph (reg / ppi / sig / codep /
lr / complex), built to answer mechanistic questions. CPU-trainable in-sandbox (the graph is small); the
architecture is a SIGN/SGC-style graph neural network — node features + multi-hop smoothed structure — with
light task heads. `colab/cellgraph.py` (`CellGraph` class + `demo()`), gated by `colab/validate_cellgraph.py`
as the 10th recovery-scorecard axis.

## Measured capabilities (all leakage-controlled)

| capability | question it answers | metric | vs random |
|---|---|---|---|
| **link prediction** | what can this protein **bind**? | PPI **AUC 0.754** (hard negatives, test edges removed) | 0.5 |
| **perturbation → downstream** | remove protein X → what changes, which way? | **sign-accuracy 0.81** on held-out signed edges | 0.5 |
| **drug polypharmacology** | give a drug → what else can it hit? | held-out drug-target **AUC 0.80** (1,433 drugs) | 0.5 |
| **structure → function** | does the wiring encode function? | essentiality/TF **AUC 0.74/0.75** (pure topology) | 0.5 |

## Live query examples (correct)

- `bind_partners("TP53")` → CTNNB1, SMAD4, AR, PPARG, APC, RB1 … (tumor-suppressor network)
- `knockout_effect("SREBF2")` → **LDLR, HMGCR, NPC1L1, PCSK9 down; ABCG5/8, LRP1 up** — textbook cholesterol regulation
- `knockout_effect("PCSK9")` → **LDLR** (secreted effector → falls back to physical partners)
- `drug_off_targets("Imatinib")` → TNF, FASLG, CASP8, AKT1 … (apoptosis axis)

## Honest limits / bugs fixed along the way

- **Two leakage bugs caught & fixed (R2):** link prediction used the full graph incl. test edges; function
  prediction used the labels as features (gave AUC 1.0). Both fixed → honest numbers above.
- **Knockout ranking (R6):** multi-hop propagation floods hubs → use *direct* (1-hop) downstream; effectors
  with no transcriptional out-edges fall back to PPI partners.
- **Perturbation is directional, not magnitude-calibrated** — it gets up/down right (81%), not "by how much."
- **Co-dependency is not predicted** (AUC 0.53) — it's fitness-correlation, a different signal than cascade.
- **Enhancement that did NOT help (R8):** adding complex-co-membership edges / 3 hops (0.754→0.755, noise).
  A real jump needs a *learned* GNN (torch) — the Colab scale path below.

## Scale path (next rounds, on Colab)

1. **Learned GNN (torch):** replace fixed SIGN propagation with a trained GraphSAGE/R-GCN → higher link AUC.
2. **Supervised perturbation on real Replogle:** train knockout→Δexpression against the measured screen
   (already validated separately by `measured_cause`), calibrating *magnitude*, not just direction.
3. **Fold in the atlas/Geneformer/scGPT embeddings on Drive** as additional node features.

# The billion-dollar bar — honest spec, gap analysis, and roadmap

## First, an honest reframe of "billion-dollar model"
No cell model today "answers everything about the cell at any precision." That is the **AI Virtual Cell**
grand challenge (CZI) — an explicit 10-year+ moonshot, **not a shipped product**. The billion-dollar
*value* in bio-AI is **not breadth** — it is **narrow + deep + validated + proprietary data**:

| player | what the model does | why it's worth billions |
|---|---|---|
| **AlphaFold / Isomorphic** | sequence → 3D structure | *solved* a defined problem, validated (CASP ~90 GDT), narrow |
| **Recursion (~$2–4B)** | perturbation → cell **morphology** (phenomics) | petabyte-scale **proprietary** imaging + wet lab |
| **Insitro** | proprietary functional genomics → **validated targets** | its own data-generation + clinical pipeline |
| **Arc Institute STATE** | **perturbation-response** prediction | trained on **100M+ perturbation cells** (much proprietary); current SOTA — *and still weak on unseen perturbations* |

**The lesson:** billion-dollar value ≈ **20% model capability + 80% proprietary data + experimental
validation + a business/clinical pipeline.** You do **not** get there with a public-data knowledge graph,
no matter how broad. That's the honest ceiling on *value*. But there is one axis where **no** billion-dollar
player competes, and where we can plausibly be **best-in-class**: a **broad, integrated, queryable,
reasoning** model of the whole cell. They are all narrow; that breadth is our lane.

## Capability axes — the bar, our score, and WHY the gap
Score 0–10 (10 = the billion-dollar bar). "Why" = data / algorithm / validation (the fixable vs the not).

| axis | billion-dollar bar | ours | gap driver |
|---|---|---|---|
| **Coverage / parts list** | ~complete proteome + interactome | **8** | near-parity; interactome still ~partial |
| **Function assignment (incl. dark)** | ~all genes, validated | **6** | good (multi-lens + structure + domain + GO); *unvalidated* |
| **Network / wiring accuracy** | high-precision, causal | **5** | mostly correlational; causal layer thin → *algorithm + data* |
| **Novel discovery** | validated novel targets | **5** | convergence produces strong candidates; **0 validated** → *validation* |
| **Perturbation prediction** | Arc-STATE level | **2** | Model 4 weak (9.3×, neg R²) → *data (proprietary perturbation) + algorithm* |
| **Condition / context response** | quantitative, many conditions | **4** | curated ontology + data-driven ARCHS4 networks; qualitative only → *data* |
| **Quantitative / dynamics / kinetics** | concentrations & rates over time | **1** | the wall — <10% measured, not closeable → *data that doesn't exist* |
| **Tissue / multicellular** | spatial, cell-cell, whole-tissue | **4** | tissue model + cell-instance bridge; no spatial → *data + algorithm* |
| **Multimodal (imaging/spatial/structure)** | phenomics + spatial + structure | **3** | structure now; **no imaging/spatial** → *proprietary data* |
| **Queryability ("ask anything")** | — (nobody has this broadly) | **2→(building)** | our biggest opportunity → *algorithm/reasoning* |
| **Validation / trust** | wet-lab + held-out, calibrated | **2** | nothing wet-lab confirmed → *validation* |

**Weighted honest read:** on **breadth + integration + reasoning + queryability** we are at or can exceed
the field. On **validated prediction + proprietary-data + quantitative dynamics** we are far behind and
**cannot** close it with code — those need wet-lab, proprietary data, or data that doesn't exist.

## What "beating the billion-dollar mark" realistically means for us
- **Achievable (code + public data + reasoning):** be the **most comprehensive, queryable, honestly-
  calibrated cell/tissue reasoning model in existence** — answer any *structural / relational / functional /
  conditional* question, with confidence and explicit "not knowable," across a broader scope than any narrow
  billion-dollar model. This is a genuine best-in-class claim on the breadth+reasoning axis.
- **Not achievable by us alone:** the *value* of a validated-target or perturbation-prediction company —
  that needs proprietary data generation + wet-lab. It's a business reality, not an algorithm.
- **Impossible for anyone today:** true quantitative dynamics (kinetics wall).

## Roadmap — ranked, start at the top
1. **Unified query/reasoning engine** ("ask the cell anything") — routes any question to the right layer,
   returns an answer + confidence + provenance + honest "not knowable." *This is the axis we can lead on,
   and it's the user's actual goal.* **← start here.**
2. **Calibrated confidence on every answer** — so "how accurate" is attached to each output.
3. **Push each weak-but-closeable axis to its public-data ceiling** — perturbation (production GNN),
   conditions (LINCS/GEO), tissue (spatial where public).
4. **Validation harness** — held-out + orthogonal-data checks → turn candidates into *trusted* answers.
5. **The value gate (non-code):** proprietary data + one wet-lab-validated novel finding.

Progress against this doc is tracked as we implement each item.

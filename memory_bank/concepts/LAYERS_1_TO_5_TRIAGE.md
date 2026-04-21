# Layers 1-5 — Triage

_Session 3 ran Layer 0 end-to-end (Phases A-E) and then went straight to Layer 6 (essentiality pipeline), which is the brief's actual goal. Layers 1-5 are NOT re-implemented in this session; they are partially covered by existing `cell_sim/` code. This file is a shortlist of what's already there and what each layer still needs when it gets real attention._

## Layer 1 — Transcription machinery

**Already in existing code**
- `cell_sim/layer3_reactions/gene_expression.py::build_gene_expression_rules` — transcription rules with precomputed duration (40 nt/s class rates lumped).
- `kinetic_params.xlsx` sheet `Gene Expression` has 30 rows of measured transcription / translation parameters (RNAP counts, elongation rates, promoter strengths).

**Missing / to be added in a full Layer 1 pass**
- RNAP holoenzyme assembly (α2ββ'σ) as explicit complex formation events. Currently RNAP and ribosomes are lumped as pools, not tracked as ProteinInstances (flagged in inventory under `gene_expression.py` verdict "adapt").
- Sigma-factor binding kinetics. `JCVISYN3A_0407` (rpoD) is the major sigma factor; only sigma factor present in Syn3A.
- Promoter strengths per gene (per `Gene Expression` kinetics sheet).
- mRNA half-lives (brief target: from Thornburg 2022 or Breuer 2019).

**Dependencies**
- Layer 0 `Genome` API provides gene coordinates and product annotations.

## Layer 2 — Translation machinery

**Already in existing code**
- `kinetic_params.xlsx` sheets `SSU Assembly` and `LSU Assembly` — 30S / 50S ribosomal assembly pathways with rates (measured).
- `tRNA Charging` sheet — 121 rows of aminoacyl-tRNA synthetase kinetics.
- Translation rule in `gene_expression.py` fires at a fixed ~12 aa/s (lumped).

**Missing**
- Ribosome as an explicit tracked complex (55+ subunits). Currently it's a pool count.
- tRNA charging stoichiometry tied to the amino acid pools (so amino acid starvation stalls translation correctly under knockouts of biosynthetic genes).
- Codon-by-codon translation modelling (probably deferred; whole-protein events are fine for essentiality).

## Layer 3 — Protein folding + complex assembly

**Already in existing code**
- `complex_formation.xlsx` — 25 known complexes with stoichiometries, init counts, assembly pathways, PDB structures.
- `cell_sim/layer2_field/real_syn3a_rules.py::make_folding_rule` and `make_complex_formation_rules` — event-driven versions.
- `scaled_proteome` sheet in `initial_concentrations.xlsx` — per-gene initial counts accounting for membership in complexes.

**Missing**
- Fact files citing each complex's stoichiometry (F1F0 ATPase, RNAP, ribosome, etc.) — needed before Layer 6 can cite why a knockout is expected to be lethal.
- Folding rate per protein (currently lumped).

## Layer 4 — Metabolism

**Already in existing code**
- `cell_sim/layer3_reactions/sbml_parser.py` — loads `Syn3A_updated.xml` (iMB155-ish).
- `kinetic_params.xlsx` sheets `Central`, `Nucleotide`, `Lipid`, `Cofactor`, `Transport` — hundreds of reactions with k_cat / K_m / E_0.
- `reversible.py` — forward + reverse Michaelis-Menten with product saturation.
- `nutrient_uptake.py` — six patched transporter k_cats (flagged in Layer 0 inventory as needing citation).

**Missing**
- Fact files for every patched transporter k_cat (6 files, each marked `confidence: estimated`).
- Check that `Syn3A_updated.xml` is in fact iMB155 — the naming is suggestive but we have not verified revision.

## Layer 5 — Biomass + division

**Already in existing code**
- None — there is no explicit biomass / division logic in the current simulator.
- `cell_sim/layer3_reactions/coupled.py` does convert pool counts to mM but does not accumulate biomass.

**Missing**
- Biomass composition vector (typical bacterial: 55% protein, 20% RNA, 3% DNA, 10% lipid, 12% other) — needs to be recorded as a fact from Breuer 2019 or Thornburg 2022.
- Division trigger: biomass doubling OR DNA replication complete. Brief target is 2 ± 0.5 h doubling.

## Why we skipped ahead to Layer 6

The brief's goal is **essentiality prediction with MCC > 0.59**. The *existing* `cell_sim/` can already run a full Syn3A simulation for short bio-time intervals (as demonstrated by `test_knockouts.py` at 0.5 s bio-time). That is enough to build a scaffolded essentiality pipeline today, even if Layers 1-5 are imperfect, and iterate the accuracy of the predictions by improving individual layers later.

Layer 6 is therefore implementable now. Layers 1-5 get attention in subsequent sessions, driven by *which genes the essentiality pipeline mispredicts* — that is a much better prioritisation signal than abstractly improving each layer.

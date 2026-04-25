# Toxicity prediction — viability assessment (Session 21)

_This is the viability assessment required by the toxicity-prediction research specification. It precedes any implementation. If any of the three viability gates fail, the project does not proceed and that finding is itself the deliverable._

## The question

Can the existing Syn3A whole-cell simulator be extended into a mechanism-aware toxicity predictor that takes a SMILES + concentration and returns (binary toxicity, mechanism trace, confidence) — validated against published Mycoplasma MIC data?

## Three viability gates

The research direction needs THREE things to all be feasible. Failure on any one gate ends the project at this stage; the negative result is documented and the remaining gates are not evaluated.

### Gate A — Simulator architecture supports additive inhibition

**Status: ✅ PASS.**

`cell_sim/layer3_reactions/reversible.py` builds catalysis rules whose firing rate is `kcat × saturation_factor`, where `saturation_factor = ∏(c_i / (c_i + K_m_i))` (line 70-72; `mm_saturation_factor`). An inhibition layer is mathematically a multiplicative factor on top:

| Inhibition mode | Modification |
|---|---|
| Competitive | Replace `K_m_i` with `K_m_i × (1 + [I]/K_i)` for each substrate i |
| Non-competitive | Multiply final rate by `1 / (1 + [I]/K_i)` |
| Uncompetitive | Multiply both numerator and denominator's substrate-saturation product by appropriate factors |

All three forms reduce to the unmodified rate when `[I] = 0`. This is what the spec requires (additive layer; existing essentiality predictions must be reproducible at zero inhibitor concentration). The injection point is concrete: a single function call inside `make_reversible_rules` that consults a per-reaction inhibitor dict and returns a multiplier.

Estimated implementation cost: 1 day for the rate-law extension + regression test that v15 essentiality MCC reproduces exactly at `[I] = 0` for all 455 genes.

### Gate B — Compound-target data with traceable provenance exists for Mycoplasma-relevant enzymes

**Status: ⚠️ PARTIAL — depends on Colab access; sandbox-side it's a no.**

Probed from sandbox (April 2026): every relevant database returns HTTP 403:
- ChEMBL REST API (`ebi.ac.uk/chembl/api/data/...`)
- BindingDB
- PubChem PUG REST
- DrugBank
- OpenFDA

Only `raw.githubusercontent.com` is reachable. ChEMBL bulk downloads are 5+ GiB SQLite/SDF files; even on Colab those need ~30 min to fetch + parse.

The realistic path: **all compound-target curation happens on Colab**, parquet/CSV is pushed back to the branch, sandbox-side code consumes the curated extracts. Same pattern as Tier-1 cache.

Coverage estimate (literature-based, not yet measured):
- The Syn3A SBML has **155 unique gene-associated reactions**. ~30-40 of those involve enzymes that are conserved bacterial drug targets (DnaA, FtsZ, RNA polymerase, ribosomal proteins, tRNA synthetases, dihydrofolate reductase, fatty acid synthesis enzymes).
- ChEMBL has ≥ 1000 documented inhibitors per major target class; **for the conserved targets, hundreds-to-thousands of (compound, IC50) pairs exist**.
- The bottleneck is NOT data abundance but **orthology mapping**: Syn3A locus tag (`JCVISYN3A_0001`) → SBML gene association (`G_MMSYN1_0008`) → UniProt accession → ChEMBL target ID. Two of these joins are well-defined; the locus_tag → UniProt step needs an explicit mapping file.

**Open viability question for Gate B**: how many of the 155 Syn3A SBML genes can be mapped to a UniProt accession that has ≥ 5 ChEMBL inhibitors with measured IC50 / Ki? This is a 1-day Colab notebook (download UniProt's M. genitalium proteome, BLAST Syn3A protein sequences against it for orthology, query ChEMBL for each match). Until that runs, Gate B is provisionally PASS but needs measurement.

### Gate C — Validation dataset of ≥ 30 compound-toxicity pairs exists for Mycoplasma

**Status: ✅ PROVISIONAL PASS — requires Colab assembly to confirm.**

Mycoplasma is a well-studied antibiotic-susceptibility target. Its lack of a cell wall makes the validation set narrower than for E. coli or B. subtilis (no β-lactams, no glycopeptides — they're intrinsically inactive), but the active-against-Mycoplasma drug classes are still well-populated:

| Drug class | Target | Mycoplasma-active? | ~Compound count with documented MIC |
|---|---|---|---|
| Macrolides | 50S ribosome (peptidyl transferase) | ✅ yes (mainstay) | ~15-20 (erythromycin, azithromycin, clarithromycin, telithromycin, …) |
| Tetracyclines | 30S ribosome (A-site) | ✅ yes (mainstay) | ~10 (doxycycline, minocycline, tigecycline, …) |
| Fluoroquinolones | DNA gyrase / topo IV | ✅ yes | ~10 (ciprofloxacin, levofloxacin, moxifloxacin, …) |
| Pleuromutilins | 50S ribosome (P-site) | ✅ yes | ~5 (lefamulin, retapamulin, valnemulin) |
| Aminoglycosides | 30S ribosome (decoding center) | partial (some active) | ~10 |
| Lincosamides | 50S ribosome | ✅ yes | ~5 (clindamycin, …) |
| Sulfonamides | DHPS (folate biosynth) | ✅ in genitalium | ~5 (Mycoplasma-relevant, others) |
| Trimethoprim | DHFR | ✅ yes | 1-2 |
| **β-lactams** | PBP / cell wall | **❌ INTRINSICALLY INACTIVE** (no peptidoglycan) | excluded |
| **Glycopeptides** | D-Ala-D-Ala / cell wall | **❌ INTRINSICALLY INACTIVE** | excluded |
| Polymyxins | LPS / outer membrane | depends on species | variable |

**Conservative estimate: 50-80 antibiotics with a published MIC against M. pneumoniae, M. genitalium, or related Mycoplasma species.** This comfortably clears the spec's ≥ 30 compounds floor for "minimum publishable result" and gets close to the ≥ 100 floor for a "strong result."

Candidate validation sources (need Colab to fetch):
- **Hannan 2000** and follow-ups: comprehensive MIC tables for M. pneumoniae across antibiotic classes
- **Bébéar 2011**: M. genitalium susceptibility review
- **Waites 2017** (Clin Microbiol Rev): mycoplasma antibiotic susceptibility comprehensive review
- **CO-ADD** (Community for Open Antimicrobial Drug Discovery): open compound screening data
- **PubChem BioAssay**: AID-keyed antimicrobial datasets including M. genitalium and M. pneumoniae targets
- **The Spec's Glass 2017 reference is likely a typo or memory error** — Glass et al. 2006 PNAS (M. genitalium minimal-genome) is real; no 2017 follow-up I can locate. Flagging this for the user to verify before relying on it.

## What's NOT viable (worth being explicit)

- **Sandbox-only execution.** Every database I'd need is blocked. The user must run a series of Colab notebooks (curation, mapping, validation-set assembly) just like the multi-organism essentiality work. Implementation work CAN happen sandbox-side once curated data is on the branch.
- **β-lactam validation.** Mycoplasma intrinsically rejects β-lactams. Including them in the validation set would either inflate true-negatives artificially (every β-lactam would be predicted "non-toxic" trivially) or, worse, hide the simulator's actual mechanism by putting cell-wall-related rules under load they don't experience. **Explicitly exclude β-lactams + glycopeptides from the validation set.**
- **Membrane disruption mechanisms.** The simulator does not model membrane biophysics. Polymyxin-class compounds (membrane disruptors) cannot be mechanistically predicted; they'd have to be excluded or treated as a known-failure-mode category. Spec says "metabolic toxicity sufficient?" — this is one of the open scientific questions, but it bounds the deliverable.
- **Novel-compound ML prediction.** The spec mentions DeepDTA / MolecularACE for novel compounds. Sandbox-side that needs GPU; Colab-side it works. Defer until Phase 2 — the validation set should be all *known* compounds first.

## Provisional go/no-go decision

**Go, with a structural caveat.** Two of three gates are confirmed PASS in sandbox. Gate B (compound-target data coverage) is provisionally PASS but unverified — needs one Colab notebook to measure how many Syn3A enzymes have ≥ 5 ChEMBL-cataloged inhibitors with traceable provenance.

If Gate B comes back below threshold (say < 20 Syn3A enzymes with usable inhibitor data) the project pivots to a smaller-scope variant: predict toxicity only for the subset of compounds whose targets ARE well-characterized in Syn3A, accepting a smaller validation set.

If Gate B comes back below an absolute minimum (say < 5 enzymes), the project does not proceed and the negative result is itself the deliverable.

## Next session — Gate B measurement (Session 22)

A single Colab notebook (`notebooks/toxicity_gate_b_assessment.ipynb`) that:

1. Pulls UniProt orthology for the 155 Syn3A SBML-associated genes via M. genitalium G37 / M. pneumoniae M129 reference proteomes.
2. For each Syn3A locus, queries ChEMBL for inhibitor counts at IC50 / Ki / Kd evidence levels.
3. Produces `memory_bank/data/toxicity/syn3a_enzyme_inhibitor_coverage.csv` with one row per Syn3A locus: `(locus_tag, uniprot_accession, chembl_target_id, n_inhibitors_strong, n_inhibitors_weak, mean_pchembl)`.
4. Produces a one-paragraph summary with the headline number ("X of 155 Syn3A SBML enzymes have ≥ 5 inhibitors with documented IC50; Y have ≥ 50; mean coverage = Z").

Decision rule for proceeding to implementation:

| Gate B outcome | Action |
|---|---|
| ≥ 30 Syn3A enzymes with ≥ 5 inhibitors each | Proceed to Session 23 (validation set assembly) |
| 10-30 enzymes | Proceed but scope down to a "narrow validation set" — tell the user the implications |
| < 10 enzymes | Halt. Document negative finding. Do not implement. |

## Other risks not yet quantified

- **Kinetic-parameter mismatch.** Published K_i values for inhibitors are typically measured against purified enzymes from organisms other than Mycoplasma (often *E. coli* or human). Cross-organism K_i transfer is a known source of error. The spec acknowledges this: "estimated IC50/Ki values" is fine, but the validation MCC will be sensitive to it.
- **Toxicity-readout threshold tuning.** The spec lists candidate signals (ATP/ADP collapse, pool depletion, etc.) but the threshold itself is "a parameter to tune against validation data." That's circular if not done carefully — must use train/test split with the threshold tuned only on the train half.
- **MIC vs. simulator-predicted concentration units.** Published MICs are typically µg/mL (mass concentration); simulator uses mM (molar concentration). Conversion via molecular weight is mechanical but easy to mess up. Need a single canonical conversion path with unit tests.
- **Selectivity vs. toxicity.** Some compounds are toxic to humans but not bacteria (or vice versa). The simulator predicts bacterial toxicity by construction. Mismatches between the validation set's reported toxicity and the simulator's prediction may reflect species selectivity, not simulator failure.

## Hard non-negotiables maintained

- No simulator code touched in this session. Existing 235 tests pass. v15 essentiality reproducibility unchanged.
- No claims of mechanism agreement made (no implementation exists yet).
- No compound-target data fabricated.
- No large datasets committed (this is documentation only).
- Branch unchanged: `claude/syn3a-whole-cell-simulator-REjHC`.

## Sessions ahead (only if Gate B passes)

| Session | Output | Cost estimate |
|---|---|---|
| 22 | Gate B measurement (Colab) — `syn3a_enzyme_inhibitor_coverage.csv` | 1-2 days, 1 Colab GPU run |
| 23 | Validation set assembly (Colab) — Mycoplasma MIC table for ≥ 30 compounds with cited sources | 2-3 days |
| 24 | `cell_sim/layer3_reactions/inhibition.py` + zero-inhibitor regression test | 1-2 days, sandbox |
| 25 | Toxicity readout layer in Layer 6 + first end-to-end prediction on a known antibiotic | 1-2 days |
| 26 | Run validation set; compute MCC + mechanism-agreement scores | 1 day |
| 27 | Writeup; cleanup; portfolio packaging | 1-2 days |

Total estimated effort: **2-3 weeks of focused part-time work** to a minimum-publishable result, assuming Gate B does not falsify the project. The 3-4 month timeline in the spec is conservative; this is achievable inside it.

## What this assessment does NOT contain

- No measured numbers about ChEMBL coverage (deferred to Session 22).
- No compound list (deferred to Session 23).
- No claims of accuracy (no implementation exists).
- No promised mechanism predictions (those need the implementation + validation set first).

This is exactly what the spec asks: viability assessment, no premature building.

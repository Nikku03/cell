# Toxicity feature/curation outputs

This directory holds curated CSVs that drive the toxicity-prediction work:

- `syn3a_enzyme_inhibitor_coverage.csv` — produced by `notebooks/toxicity_gate_b_assessment.ipynb` (Session 22). One row per Syn3A SBML-associated gene with UniProt mapping + ChEMBL inhibitor counts. Headline number: how many enzymes have ≥ 5 ChEMBL inhibitors with documented IC50.
- `validation_set.csv` — produced by Session 23 (deferred until Gate B passes). Per-compound MIC against M. pneumoniae / M. genitalium with cited sources.

Schema for `syn3a_enzyme_inhibitor_coverage.csv`:

| column | type | notes |
|---|---|---|
| sbml_gene_id | str | SBML gene association string (e.g. `G_MMSYN1_0008`) |
| sbml_locus_tag | str | derived locus tag (e.g. `JCVISYN3A_0008`) |
| reaction_short_names | str | `;`-joined SBML reaction short names this gene catalyses |
| gene_name | str | best-guess gene name from Syn3A GenBank |
| product_description | str | product description from Syn3A GenBank |
| sequence_length | int | protein length in residues |
| uniprot_accession | str | mapped UniProt accession (or empty if unmapped) |
| mapping_method | str | `direct_syn3a` / `mgenitalium_ortholog` / `mpneumoniae_ortholog` / `unmapped` |
| chembl_target_id | str | mapped ChEMBL target ID (or empty) |
| n_inhibitors_strong | int | ChEMBL activities with `pchembl_value >= 5` and IC50/Ki/Kd evidence |
| n_inhibitors_weak | int | activities below the strong threshold but still measured |
| mean_pchembl_strong | float | mean of `pchembl_value` over strong inhibitors |

## Gate B decision rule (Session 21 viability assessment)

Once `syn3a_enzyme_inhibitor_coverage.csv` lands:

| outcome | action |
|---|---|
| ≥ 30 enzymes with `n_inhibitors_strong ≥ 5` | proceed to Session 23 (validation set assembly) |
| 10-30 such enzymes | proceed but scope down to a "narrow validation set" |
| < 10 such enzymes | halt; document negative finding as the Session-22 contribution |

## Why this lives outside the repo's main data

Per the toxicity research spec's "Hard non-negotiables": **do not commit large datasets to the repo.** The coverage CSV itself is small (~155 rows, < 100 KiB), so it lives inside the repo. ChEMBL bulk dumps and per-target activity JSON blobs are NOT committed — they're regenerable from Colab via the notebook.

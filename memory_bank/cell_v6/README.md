# cell_v6: drug development joins the memory bank

Written by `rem/atlas/cellbank6.py build`. It holds everything in `cell_v5`, unchanged and in the same order, plus two drug layers:
- **What drugs do to K562 cells** (measured).
- **What is known about drugs from development**: targets, trials, warnings and adverse reactions.

## Measured drug responses: screen F (sci-Plex3)

Source: Srivatsan et al. 2020, *Science*, "Massively multiplex chemical transcriptomics at single-cell resolution". Data from the scPerturb collection on Zenodo (record 13350497), CC BY 4.0.

- 188 compounds, each at 4 doses (10 nM, 100 nM, 1 µM, 10 µM), 24 h, in K562 cells: 173,652 cells.
- Each record is one compound at one dose.

How a record is built:
1. Sum the counts of all cells given that compound and dose.
2. Convert to counts per million, then take log1p.
3. Subtract the vehicle's value (3,935 control cells).
4. Keep only genes with vehicle CPM ≥ 20 (5,924 genes).
5. Remove the mean response of all compounds at that dose.
6. Map onto screen A's genes and the shared 128 components, as for every other record.

In the files:
- The gene field reads `drug:<compound>` and the split field reads `DRUG`.
- `records_meta.json["drug"]` gives each record's compound, dose, cell count and ChEMBL ids.
- `sciplex_compounds.json` lists, per compound:
  - its ChEMBL matches;
  - its loss-of-function targets (inhibitor, antagonist, blocker and similar) and all annotated targets;
  - after `score`, the 5 knockdowns its 10 µM response most resembles.

144 of the 188 compounds match a ChEMBL molecule by name; 90 have a loss-of-function target.

## Drug-development knowledge (Open Targets Platform 26.09, marked CC0)

### `drugs.json`

One entry per ChEMBL drug: 13,128 drugs, 5,052 of them with a mechanism. Each entry holds:
- name, type and highest clinical stage;
- mechanisms (action, target Ensembl id and symbol);
- top indications;
- trial counts by status;
- up to 15 stopped trials, with the reason they stopped and the stop category;
- drug warnings (black box, withdrawal), with toxicity class, country and year;
- the 10 strongest openFDA adverse-reaction signals (log-likelihood ratio).

### `targets.json`

One entry per target gene, keyed by Ensembl id: 17,095 genes. Each entry holds:
- the drugs on the gene (action and stage);
- tractability flags;
- safety events (ToxCast, literature), with direction and dosing;
- high-quality chemical probes;
- counts of stopped trials and warnings for drugs on the gene.

### `index.sqlite`

Adds two tables:
- `drugs(chembl, name, type, max_stage, n_mechanisms, n_warnings, n_stopped)`
- `mechanisms(chembl, target_ensembl, symbol, action)`, indexed by symbol.

Open Targets marks its release CC0. The fields come from ChEMBL, ClinicalTrials.gov, DailyMed, openFDA, ToxCast and the literature; cite those sources when using them.

## Journal

`score` appends hypotheses of the form "drug X acts on K562 like knocking down gene Y". Each is marked **UNVALIDATED** and carries its sources and the drug's development record.

## Rules for users

These are the same as before. When predicting knockdowns, hide:
- all screen-A TEST records;
- the query gene's own records.

Drug records are not part of the engine's snap memory.

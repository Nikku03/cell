# M. pneumoniae M129 — cell_sim input data (Phase D2 partial)

This directory mirrors the Syn3A reference data layout at
`cell_sim/data/Minimal_Cell_ComplexFormation/input_data/`. Files
here are the **partial** product of Phase D2 conversion against the
source bundle pushed in commit `4519f00` and curated by
`scripts/curate_mpneumoniae_data.py` (run from repo root).

**Status: NOT YET RUNNABLE in cell_sim.** Two reasons:

1. Critical gaps in the source data — see `DATA_GAPS.md` for the
   complete inventory. The metabolite initial concentrations
   (Yus 2009) and the canonical complex formation table
   (Kühner 2009) are still missing because both papers' supplementary
   materials are behind Cloudflare on `science.org`.
2. The simulator's data-loading layer (`cell_sim/layer0_genome/`,
   `cell_sim/layer3_reactions/sbml_parser.py`,
   `cell_sim/layer3_reactions/kinetics.py`,
   `cell_sim/layer6_essentiality/real_simulator.py`) is hard-coded to
   the Syn3A files. Phase D3 (separate session) will add an
   `organism: str` config flag.

## Files

| File | Source | Coverage |
|---|---|---|
| `input_data/M_pneumoniae_iJW145.xml` | BioModels MODEL1301290000 (Wodke 2013, Mol Syst Biol) | 324 species, 306 reactions, 2 compartments — complete |
| `input_data/kinetic_params.xlsx` | Wodke 2013 MOESM2 Table S1 (gene-reaction-EC-pathway) | 308 reactions across 15 pathways. **Km / kcat columns blank — Wodke is FBA, not kinetic.** Marked `NEEDS_CURATION` in every Parameter Type cell. |
| `input_data/initial_concentrations.xlsx` | Maier 2011 MOESM3 Table S2 (Comparative Proteomics) | 414 protein abundances. **Intracellular Metabolites + Simulation Medium + mRNA Count sheets are empty pending Yus 2009.** |
| `input_data/complex_formation.xlsx` | Maier 2011 MOESM3 Table S7 (ribosomal proteins) | 1 complex entry (Ribosome). **All other complex sheets empty pending Kühner 2009.** |
| `DATA_GAPS.md` | this directory | per-gap inventory + impact on simulator |

## Reproducibility

```bash
cd /repo/root
python scripts/curate_mpneumoniae_data.py
```

Deterministic given the source files in
`memory_bank/data/multiorg_essentiality/raw/mpneumoniae/`
(commit `4519f00`).

## Provenance per output cell

Every populated row in `kinetic_params.xlsx` carries traceability
columns: `wodke_model_id`, `yus2009_id`, `ec_number`, `reversibility`,
`equation`. These let any reviewer trace a row back to the exact
Wodke 2013 Table S1 entry it came from.

`initial_concentrations.xlsx` `Comparative Proteomics` sheet rows
come from Maier 2011 Table S2 `Control (copies/cell)` column. Other
columns (`Gene Name`, `Gene Product`, `Protein Length`,
`Essentiality`, `Primary Function`, `Localization`) are blank because
they would require joining with the M. pneumoniae GenBank
(`memory_bank/data/multiorg_essentiality/raw/mpneumoniae_M129_NC_000912.gb`,
commit `b60c3b5`) — that join is a separate session, would land as
its own commit.

The single `complex_formation.xlsx` row (`Ribosome`) lists the 46
ribosomal protein MPN locus tags from Maier 2011 Table S7 with
stoichiometry 1 per subunit and `Init. Count` = median of the
direct-quantified copies/cell across r-proteins (10–536 range, n/d
rows excluded).

## What this is not

* Not biology in the curated sense — it is structure. Curated
  parameters require human review of the source bundle, which is
  Phase D3+ work.
* Not a runnable simulator config — it is files-on-disk in the right
  schema. Wiring requires the organism-config refactor.
* Not a measurement against Lluch-Senar 2015 essentiality — that is
  Phase D4.

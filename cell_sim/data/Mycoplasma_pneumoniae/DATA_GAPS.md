# M. pneumoniae cell_sim data — explicit gap inventory

The Phase D2 conversion (`scripts/curate_mpneumoniae_data.py`)
produced schema-compatible XLSX files using **only** the source
material that is on disk. Each gap below is named, sized, and
mapped to its blocker.

## Gap 1 — reaction-level kinetic parameters (Km, kcat)

**Schema location:** `kinetic_params.xlsx`, every sheet, columns
`Value` + `Units`.

**Status:** all 308 rows have `Value = NaN` and
`Parameter Type = "kcat (per s, NEEDS_CURATION)"`.

**Why:** Wodke 2013 (`MODEL1301290000_iJW145.xml`) is a
constraint-based / FBA model. Its `<listOfParameters>` block
contains 918 entries but they are flux bounds and stoichiometric
factors, not enzyme kinetics. M. pneumoniae has no published
genome-scale kinetic table comparable to Luthey-Schulten's Syn3A
`kinetic_params.xlsx`.

**Where the values would come from:**
* BRENDA enzyme database, per-EC-number lookup (proxy-blocked).
* Yus 2009 supplementary tables (Cloudflare-blocked).
* Per-paper hand curation against published M. pneumoniae enzyme
  characterizations — multi-week research engineering.

**Effect on the simulator:** cell_sim's MM rule factories
(`cell_sim/layer3_reactions/reversible.py`) use Km + kcat per
reaction. With Km/kcat blank, every reaction has propensity zero
and no events fire. The simulator cannot run on M. pneumoniae
until either (a) the gap is filled or (b) reaction-list-based
default kinetics are wired in (a deliberate scope decision, not
biology — would need to be flagged with `confidence: assumed` on
any downstream measurement).

## Gap 2 — intracellular metabolite initial concentrations

**Schema location:** `initial_concentrations.xlsx`,
`Intracellular Metabolites` sheet (currently empty).

**Status:** zero rows. Header columns present but no data.

**Why:** Yus et al. 2009, Science, "Impact of Genome Reduction on
Bacterial Metabolism" (DOI `10.1126/science.1177263`) is the
canonical source for M. pneumoniae intracellular metabolite levels.
science.org returns HTTP 403 (Cloudflare) from the acquisition
sandbox, and there is no PMC mirror.

**Where the values would come from:** Yus 2009 supplementary
tables (PDF + XLS supp data). Browser download is required.

**Effect on the simulator:** Even with kinetics filled in, the
simulator initializes metabolite counts from this sheet. With it
empty, every catalysis rule reads `count = 0` and refuses to
fire. Running M. pneumoniae through cell_sim requires this gap
filled before anything else.

## Gap 3 — extracellular medium composition

**Schema location:** `initial_concentrations.xlsx`,
`Simulation Medium` sheet (currently empty).

**Why:** M. pneumoniae's standard culture medium is **Hayflick
medium** (Hayflick 1965). The recipe is well-known but lives in
review papers and lab manuals, not a structured table. Per the
original spec, this is hand-typed, not downloaded.

**Where the values would come from:** Hayflick original paper
plus standard M. pneumoniae growth medium references. Manual
curation from those into the schema's `(Metabolite name, Met ID,
KEGG ID, InChI key, Conc (mM))` rows.

**Effect on the simulator:** With this empty, all extracellular
("_e") species default to zero, which means no nutrient uptake
fires. Hayflick medium is rich in glucose, amino acids,
nucleosides, fatty acids — all needed for any growth-mode
simulation.

## Gap 4 — non-ribosomal complex formation

**Schema location:** `complex_formation.xlsx`, `Complexes` sheet
(currently has only 1 row: Ribosome).

**Why:** Kühner et al. 2009, Science, "Proteome organization in a
genome-reduced bacterium" (DOI `10.1126/science.1176343`) is the
canonical source for M. pneumoniae multi-protein complex
membership (~200 complexes catalogued). science.org returns HTTP
403 (Cloudflare) from the acquisition sandbox; no PMC mirror.

**Where the values would come from:** Kühner 2009 supplementary
materials (browser download required) and/or the paper's data
deposited at the project's own repository.

**Effect on the simulator:** Folding + complex-assembly rules
populate. With most complexes missing, ComplexAssemblyDetector
predictions on M. pneumoniae would fall back to AnnotationClass
priors only — losing the largest single source of v15's MCC lift
on Syn3A.

## Gap 5 — mRNA copy numbers per gene

**Schema location:** `initial_concentrations.xlsx`, `mRNA Count`
sheet (currently empty).

**Why:** Maier 2011 Table S8 has tiling-array mRNA quantification
(733 rows: gene + average_intensity). It is on disk in the source
bundle but the values are array intensities, not copies/cell.
Conversion to absolute copy numbers requires either the absolute-
quantification calibration the Maier paper used or a separate
qPCR / RNA-seq dataset.

**Where the values would come from:** Maier 2011 Table S8 +
calibration. Or qRT-PCR / RNA-seq from a follow-up paper. Phase
D2.5 work.

**Effect on the simulator:** Gene expression layer initializes
mRNA pools from this sheet. With it empty, transcription has no
substrate; any gex-on simulation behaves as if the cell starts
with zero mRNA.

## Gap 6 — ribosomal-protein direct copies/cell ('n/d' rows)

**Schema location:** `complex_formation.xlsx`, `Complexes` sheet,
Ribosome row, `Init. Count` column.

**Status:** populated as median (43 copies/cell) of the 43 r-proteins
in Maier 2011 Table S7 that have direct quantification. 3 r-proteins
were `n/d` (not detected) and excluded.

**Why this is a soft gap rather than a blocker:** the 3 `n/d`
proteins are likely small / membrane-associated subunits that mass
spec missed. The remaining 43 r-proteins span 8 to 536 copies/cell
with median ~43 — a usable single ribosome-count value but with
~10x intra-complex variation that doesn't appear in Syn3A's
ribosome model. A more careful curation would fit the 70S ribosome
copy number (literature: ~140 ribosomes/cell for slow-growing
M. pneumoniae) directly rather than median-from-subunits.

## Summary table

| Gap | Schema | Status | Blocker | Severity |
|---|---|---|---|---|
| 1. Km/kcat | `kinetic_params.xlsx` Value/Units | empty | BRENDA + Yus 2009 (proxy-blocked) | **runtime-blocking** |
| 2. Intracellular metabolites | `initial_concentrations.xlsx` Intracellular | empty | Yus 2009 (Cloudflare) | **runtime-blocking** |
| 3. Medium | `initial_concentrations.xlsx` Simulation Medium | empty | Hayflick recipe (manual) | **runtime-blocking** |
| 4. Non-ribosomal complexes | `complex_formation.xlsx` Complexes | 1 row only | Kühner 2009 (Cloudflare) | important — degrades MCC, doesn't block run |
| 5. mRNA copies | `initial_concentrations.xlsx` mRNA Count | empty | Maier S8 + calibration | important for gex-on |
| 6. Ribosome n/d rows | `complex_formation.xlsx` Init. Count | populated by median | Maier S7 has 3 n/d | low — usable approximation |

**Bottom line:** to actually run cell_sim on M. pneumoniae, gaps
1, 2, and 3 are all blocking. Gaps 4 and 5 degrade prediction
quality. Gap 6 is acceptable as-is. Filling gaps 1–3 requires
either browser downloads of Yus 2009 + Kühner 2009 + manual
Hayflick recipe entry, or per-EC BRENDA queries against a
network-enabled environment.

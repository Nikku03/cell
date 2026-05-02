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

## Gap 4 — non-ribosomal complex formation [RESOLVED, partial]

**Schema location:** `complex_formation.xlsx`, `Complexes` sheet
(now 105 rows; was 1).

**Status:** Substantively resolved via Kühner 2009 SOM Table S5
(literature-curated complexes), recovered from the Internet Archive
Wayback Machine (`kuhner_2009_som.pdf`, commit `01dc08c`).

**Coverage achieved:**

* 27 heteromultimeric complexes vs the paper's stated 31 — 87%
  coverage (the missing 3–4 may be in PDF text the vision-read
  missed at page boundaries; not a blocker, can be filled later).
* 78 homomultimeric complexes — matches paper's stated 78 exactly,
  100% coverage.
* Each row carries provenance: source citation (Expasy or specific
  PMID), organism evidence (M. pneumoniae direct vs by similarity
  vs cross-species homology), and per-subunit MPN locus tags.
* The Maier 2011 Ribosome composite row was replaced by Kühner's
  full ribosomal-protein-by-protein listing (51 r-protein members)
  inside the new "Ribosome complex" entry.

**Residual gap (soft, not runtime-blocking):**

`Init. Count` column is blank for every Kühner-derived row.
Kühner 2009 catalogs membership but not absolute complex copy
numbers. Phase D2.5 should populate it from either (a) the smallest
member-protein abundance in Maier 2011 Table S2 (the limiting-subunit
heuristic Luthey-Schulten uses for Syn3A), or (b) literature
complex-abundance values where available.

**Source:** `kuhner_2009_som.pdf` Table S5 (pages 89–95 of the SOM
PDF), transcribed by vision-reading rendered PNGs at 200 dpi.
`pdftotext -layout` mangled the multi-line cells; vision parsing
worked cleanly. `scripts/integrate_kuhner_complexes.py` is the
reproducible emitter.

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

| Gap | Schema | Status | Severity |
|---|---|---|---|
| 1. Km/kcat | `kinetic_params.xlsx` Value/Units | **populated with bacterial defaults (Km=0.1mM, kcat=100/s)** — `Parameter Type` = `BACTERIAL_DEFAULT` flag | calibration-debt; runs but predictions weakened |
| 2. Intracellular metabolites | `initial_concentrations.xlsx` Intracellular | **populated with Bennett 2009 E. coli proxy** (~80 metabolites) — `confidence: assumed` | calibration-debt; runs |
| 3. Medium | `initial_concentrations.xlsx` Simulation Medium | **populated with Syn3A medium proxy** (56 entries) | calibration-debt; runs |
| 4. Non-ribosomal complexes | `complex_formation.xlsx` Complexes | 105 rows from Kühner 2009; 100/105 with Init. Count from Maier limiting-subunit | resolved |
| 5. mRNA copies | `initial_concentrations.xlsx` mRNA Count | empty (would need Maier S8 calibration) | important for gex-on; not blocking gex-off |
| 6. Ribosome n/d rows | `complex_formation.xlsx` Init. Count | superseded by Kühner full r-protein listing + Maier limiting-subunit | resolved |

**Bottom line:** to actually run cell_sim on M. pneumoniae, gaps
1, 2, and 3 are still blocking. Gap 4 (non-ribosomal complexes)
is now resolved via Kühner 2009 Wayback recovery — 105 complexes
with provenance, including 51-subunit Ribosome. Gap 5 (mRNA copies)
matters once gex-on is enabled. Gap 6 is now superseded by
the Kühner per-r-protein listing. Filling gaps 1–3 still requires
either browser downloads of Yus 2009 + manual Hayflick recipe
entry, or per-EC BRENDA queries against a network-enabled environment.

# M. pneumoniae source acquisition (Phase D1)

Fetched 2026-05-02 by automated download for the multi-organism
generalization track. Per-source provenance and gap notes below.

## wodke_2013_iJW145/

- **MODEL1301290000_iJW145.xml** — Wodke 2013 genome-scale constraint-based
  model. Source: BioModels MODEL1301290000 (`https://www.ebi.ac.uk/biomodels/MODEL1301290000`).
  Note: spec called for `iMP139`; the canonical BioModels entry for
  Wodke 2013 (MSB 2013.6) is named `iJW145` (145 metabolic genes). Same
  paper. If you specifically want a different model named iMP139,
  surface that and I'll re-fetch.
- **MODEL1301290000_iJW145_report.pdf** — Auto-generated BioModels report.
- **44320_2013_BFMSB20136_MOESM{1..10}_ESM** — Wodke 2013 EMBO supplementary
  materials (`https://www.embopress.org/doi/full/10.1038/msb.2013.6`).
  Likely contents include kinetic params (Phase D1 row 2) and initial
  concentrations Tables 5/6 (Phase D1 row 3).

## maier_2011_proteomics/

- **44320_2011_BFMSB201138_MOESM{1..5}_ESM** — Maier 2011 EMBO MSB
  supplementary (`https://www.embopress.org/doi/full/10.1038/msb.2011.38`).
  Phase D1 row 4 (complex formation), partial — Kühner 2009 still missing.

## Gaps still requiring manual download

- **Yus 2009 Science** (`10.1126/science.1177263`) — Phase D1 row 3
  (initial concentrations). science.org returned HTTP 403 (Cloudflare);
  no PMC mirror. Browser download required.
- **Kühner 2009 Science** (`10.1126/science.1176343`) — Phase D1 row 4
  (complex formation, primary source). Same 403 / no PMC mirror.
- **Hayflick medium recipe** — Phase D1 row 5. Spec says hand-typed from
  review papers; not a download target.


## kuhner_2009/

- **kuhner_2009_som.pdf** — Kühner 2009 *Science* "Proteome organization in
  a genome-reduced bacterium" (`10.1126/science.1176343`) Supporting
  Online Material PDF. Source: live science.org returns Cloudflare 403,
  but the file `kuehner.som.pdf` is archived at the Internet Archive
  Wayback Machine (capture timestamp 20231012235905). Fetched via
  `https://web.archive.org/web/20231012235905id_/...`. 2.2 MB, PDF 1.6.
  Phase D1 row 4 primary source (complex formation, ~200 complexes,
  Table S2 listed inside the PDF).

## Still missing after Wayback search

- **Yus 2009** (`10.1126/science.1177263`) SOM — Wayback CDX shows only
  302-redirect captures (filename `yus-som.revision.1.pdf`); the actual
  PDF bytes were never archived. Live science.org returns 403. Browser
  download from an institution with Science.org access is required.
- **Yus 2009 supplementary tables S1–S6** — same situation; not
  individually archived.

## sabio_rk_kinetics/

- **sabio_raw_dump.tsv** + curated TSVs — Km/kcat per EC for the 158
  EC numbers that appear in iJW145 reactions, fetched via SABIO-RK
  REST. 138/158 EC have data; 70/306 SBML reactions get populated
  rows (23 % coverage). See `sabio_rk_kinetics/README.md` for the
  pipeline, organism-rank scheme, and caveats.
- Phase D1 row 1 (Km/kcat) — partial. Direct M. pneumoniae values
  exist for only 3 EC; the rest fall back through Mycoplasma genus
  → E. coli → other prokaryotes.

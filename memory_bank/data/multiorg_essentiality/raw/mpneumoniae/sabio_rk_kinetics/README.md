# SABIO-RK kinetic parameters for iJW145 reactions

Generated 2026-05-02 by autonomous bulk fetch from SABIO-RK
(`https://sabiork.h-its.org/sabioRestWebServices/kineticlawsExportTsv`).
Phase D1 row 1 (Km/kcat) — see `../PROVENANCE.md` for context.

## Pipeline

1. Parsed iJW145 SBML (`MODEL1301290000`) → 306 reactions; 160 unique
   KEGG reaction IDs in `<rdf:li ...kegg.reaction/R*/>` annotations.
2. Mapped KEGG R-id → EC via KEGG REST `rest.kegg.jp/link/ec/...`
   in 10-id batches → 158 unique EC numbers covering 157/160 KEGG
   reactions.
3. For each EC, fetched all kinetic-law rows from SABIO-RK with
   fields `EntryID, ECNumber, Organism, Parameter` (4-field minimal
   set; richer fields like `Temperature`, `PubMedID` consistently
   404'd or timed out). 138/158 EC returned data; 62,820 raw rows.
4. Filtered raw rows to `Km` and `kcat` parameter types with
   positive numeric `parameter.startValue`.
5. For each EC, picked the median value at the best available
   organism rank, where rank prefers
   M. pneumoniae (0) < other Mycoplasma (1) < Mollicutes (2)
   < E. coli (3) < other model bacteria (4) < other prokaryote (5)
   < eukaryote (6).

## Coverage

- **EC numbers in curated table:** 145 (some are multi-EC complexes)
- **EC with Km filled:**   145 / 145
- **EC with kcat filled:** 105 / 145
- **EC with both:**        105
- **Direct M. pneumoniae Km:**   3 EC (rank 0)
- **Mycoplasma genus Km:**       11 EC (rank ≤ 1)
- **E. coli Km:**                70 EC (rank 3 best available)

After joining back to SBML reaction IDs:

- **Per-reaction rows:** 101
- **Unique reactions populated:** 70 / 306 SBML reactions (23 %)
- **Reactions with Km filled:**   88 rows
- **Reactions with kcat filled:** 70 rows

The 23 % per-reaction coverage is the honest ceiling from KEGG/SABIO
without manual curation: many iJW145 reactions are transport,
exchange, or biomass pseudoreactions with no EC code, and many EC
numbers have no entry in SABIO-RK at all.

## Files

- `sabio_raw_dump.tsv` — full unfiltered 62k-row SABIO-RK output (4.4 MB).
  Keep this so reruns of the curation step don't have to re-hit SABIO.
- `kinetic_params_mpne_per_ec.tsv` — one row per EC, best Km + best kcat
  with the organism the value was sourced from and rank metadata.
- `kinetic_params_mpne_per_reaction.tsv` — one row per (SBML reaction,
  EC) pair, ready to join into the cell_sim kinetic_params schema.
- `ec_query_list.txt` — the 158 EC numbers queried.
- `kegg_rid_to_ec.tsv` — KEGG R-id → EC mapping from KEGG REST.

## Caveats

- **Organism mismatch is the rule, not the exception.** Only 3 of 145
  EC numbers had any direct M. pneumoniae kinetic data in SABIO-RK.
  The bulk falls back to E. coli (rank 3) or worse. Downstream code
  should treat the `km_organism` / `kcat_organism` columns as a first-
  class confidence signal, not metadata.
- **Median-at-rank is a coarse aggregator.** SABIO-RK reports per-paper
  measurements at varying pH, temperature, isoform, and cofactor
  conditions. Picking the median throws away that variance. For the
  3 direct M. pneumoniae hits, you almost certainly want to read the
  raw rows in `sabio_raw_dump.tsv` and pick by hand.
- **Units.** All Km values are in M (molar) per SABIO. kcat in s^-1.
  No unit conversion was applied. Some kcat rows from SABIO use
  `katal*g^(-1)` (specific activity, not turnover) — those were
  *not* filtered out by the `parameter.type == "kcat"` rule alone
  because some entries label specific-activity rows as `kcat`. Check
  unit before consuming.
- **Multi-EC reactions.** Some KEGG R-ids map to multiple EC (e.g.
  the pyruvate dehydrogenase complex `1.2.4.1 2.3.1.12 1.8.1.4`).
  These appear as multiple rows in `per_reaction.tsv`; the schema
  consumer must decide whether to pick the rate-limiting EC or
  combine.

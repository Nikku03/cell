# Multi-organism essentiality supplements

Phase C (Option C) data acquisition pass 2: per-paper supplementary
tables containing the published essentiality calls for each of the
16 organisms staged in `../multi_organism_genbank/`.

## Source resolution path

For each paper, ID resolution was:

1. **Europe PMC search by DOI** (`/europepmc/webservices/rest/search?query=DOI:...`)
   to get PMID + PMCID + `hasSuppl` flag.
2. If the paper is in PMC, **Europe PMC `supplementaryFiles` endpoint**
   returns a single ZIP of all archive-included supps. Worked
   directly for 6 of the 12 PMC-resolved papers.
3. Where step 2 returned an empty stub (HTTP 200, ~160 bytes,
   not actually a ZIP), tried two fallbacks:
   - **NCBI PMC OA bulk-package** (`oa.fcgi?id=PMC<id>` →
     ftp.ncbi.nlm.nih.gov tar.gz). Works only for OA-flagged
     PMCs. 8 of 12 papers were OA.
   - **NCBI PMC `/articles/instance/<id>/bin/<file>` direct** —
     all served reCAPTCHA challenge pages instead of files
     (~1.8 KB or ~20 KB HTML masquerading as the requested mimetype).
4. Where step 2 + 3 failed, fell back to publisher CDN. PNAS,
   ASM (mBio / J Bacteriol), and Cell Press all return Cloudflare
   HTTP 403 from this network. Wayback Machine was used for the
   Yus 2009 / Kühner 2009 cases earlier; it was partially offline
   during this pass.

See `MANIFEST.tsv` for per-paper status.

## What landed (7 papers, ~11 MB total)

| organism | study | paper file with essentiality calls |
|---|---|---|
| C. crescentus | Christen 2011 MSB | msb201158-s2.xls |
| S. aureus | Chaudhuri 2009 BMC Genomics | 1471-2164-10-291-S1.xls |
| M. tuberculosis | DeJesus 2017 mBio | mbo002173137st2.xlsx (H37Rv TRANSIT output) |
| S. Typhimurium | Barquist 2013 NAR | supp_gkt148_nar-02998-f-2012-File007.xlsx |
| K. pneumoniae | Bachman 2015 mBio | mbo003152358sd1.xlsx + sd2.xls + sd3.xlsx |
| A. baylyi | de Berardinis 2008 MSB | msb200810-s2.xls |
| S. pneumoniae | van Opijnen 2009 Nat Methods | **partial** — PMC bundle only includes the SI PDF, not the original xls |

## What's still missing (9 papers)

| organism | study | reason | next-action hint |
|---|---|---|---|
| E. coli | Goodall 2018 mBio | OA tgz 404; ASM 403; PMC bin/ reCAPTCHA | Browser via institutional ASM access. Filename is `mbo001183726st1.xlsx` (essential) and `mbo001183726st4.xlsx`. |
| B. subtilis | Koo 2017 Cell Syst | Not OA; Cell Press 403 | Filename is `NIHMS851825-supplement-3.xlsx` through `-supplement-8.xlsx`; manuscript is open access via NIHMS even though Cell version is paywalled. Try Wayback or NIHMSID lookup. |
| H. influenzae | Akerley 2002 PNAS | DOI didn't resolve in EPMC | Verify DOI (probably 10.1073/pnas.012602299 or 10.1073/pnas.022473899 — both look truncated). |
| P. aeruginosa | Lee 2015 PNAS | Not OA; PNAS 403 | PMC has supp filename `pnas.1422186112.sd01.xlsx`. Browser. |
| F. tularensis | Weiss 2007 PNAS | DOI didn't resolve in EPMC | Verify DOI; paper is PNAS 104:6037-6042. |
| H. pylori | Salama 2004 J Bacteriol | Not OA; ASM 403 | PMC bundle is structured oddly (1 PDF + index.html only); supp tables may be embedded in the article text rather than separate files. |
| M. tuberculosis | (got DeJesus already, also have 4 more screens we could add) | - | - |
| S. pneumoniae xls | van Opijnen 2009 | EPMC bundle PDF only | The Tn-seq calls table was published as `nmeth.1377-S1.xls` originally; not in PMC archive. Try Nat Methods supplementary at nature.com. |
| A. baumannii | Gallagher 2015 mBio | EPMC matched wrong paper | Real DOI is 10.1128/mBio.01660-15 → PMC4659468; that PMCID returned a different study. Re-resolve by exact title "First-Generation Whole-Genome Comprehensive Mutant Library of Acinetobacter baumannii". |
| V. cholerae | Chao 2013 PNAS | DOI didn't resolve | Likely the right cite is Cameron et al. 2008 PNAS 105:8736 (Tn screen) or Chao et al. 2013 PNAS 110:E4485. Verify before retry. |

## Caveats

- **Per-paper schema is inconsistent.** Each xls/xlsx encodes
  essentiality differently: some use boolean E/NE columns, some
  use insertion density thresholds with arbitrary cutoffs, some
  publish raw transposon counts and require re-thresholding.
  Building a unified per-gene-essentiality matrix across these
  16 organisms is an engineering task in its own right
  (Phase C-2, separate from this fetch).
- **Strain mismatches.** A few studies use a different strain
  than the canonical reference (e.g. Bachman 2015 used KPNIH1
  but mapped to an internal NTUH-K2044-like reference). 1:1
  ortholog resolution will be needed before the per-gene calls
  can be joined to the GenBank annotations.
- **van Opijnen 2009 is partial.** Only the PDF is in the PMC
  archive; the per-gene Tn-seq fitness table was published as a
  separate xls and didn't get cross-deposited. Marked PARTIAL in
  the manifest; downstream code should treat this organism's
  essentiality slot as MISSING until the xls is recovered.

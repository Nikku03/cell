# Multi-organism GenBank annotation bundle

Phase C (Option C: DEG/OGEE multi-organism) data acquisition,
fetched 2026-05-02 via NCBI Entrez efetch.

DEG (`origin.tubic.org`) and OGEE (`v3.ogee.info`) were both
unreachable at fetch time (DNS / SSL / connection-timeout), so
genome annotations were pulled directly from NCBI nuccore using
`db=nuccore&rettype=gbwithparts&retmode=text`. Per-organism
essentiality study targets are listed in `MANIFEST.tsv` (column
`essentiality_study`) — those supplementary tables are tracked
separately under `../multi_organism_essentiality/` (TBD).

## Coverage

16 organisms spanning gamma-proteobacteria, firmicutes, mollicutes,
and actinobacteria. All have published transposon / TraDIS / Tn-seq
essentiality screens. Total size: ~130 MB.

| organism | accession | essentiality study |
|---|---|---|
| ecoli_K12_MG1655 | U00096.3 | Goodall 2018 mBio (TraDIS) |
| bsubtilis_168 | AL009126.3 | Koo 2017 Cell Syst |
| ccrescentus_NA1000 | CP001340.1 | Christen 2011 MSB |
| hinfluenzae_Rd_KW20 | L42023.1 | Akerley 2002 PNAS |
| saureus_N315 | BA000018.3 | Chaudhuri 2009 BMC Genomics |
| paeruginosa_PAO1 | AE004091.2 | Lee 2015 PNAS |
| ftularensis_SCHU_S4 | AJ749949.2 | Weiss 2007 PNAS |
| hpylori_26695 | AE000511.1 | Salama 2004 PNAS |
| mtuberculosis_H37Rv | AL123456.3 | DeJesus 2017 mBio (Himar1) |
| spneumoniae_TIGR4 | AE005672.3 | van Opijnen 2009 Nat Methods (Tn-seq) |
| abaumannii_AB5075 | CP008706.1 | Wang 2014; Gallagher 2015 |
| styphimurium_LT2 | AE006468.2 | Barquist 2013 NAR (TraDIS) |
| kpneumoniae_KPNIH1 | CP008827.1 | Bachman 2015 mBio |
| vcholerae_N16961_chr1 | AE003852.1 | Chao 2013 PNAS |
| mgenitalium_G37 | L43967.2 | Glass 2006 PNAS (Tn screen) |
| abaylyi_ADP1 | CR543861.1 | de Berardinis 2008 MSB |

## Files

Each `<shortname>_<accession>.gb` is a GenBank-with-parts dump
including `LOCUS`, `FEATURES` (CDS, gene, rRNA, tRNA, etc. with
/product, /gene, /locus_tag, /protein_id, /translation), and
sequence. Suitable for direct parsing with Biopython.

## Caveats

- GenBank annotations are NCBI-curated; quality varies by organism
  and assembly age. Downstream feature extraction should treat
  /product strings as noisy labels, not ground truth.
- Strain choice favors the strain used in the paired essentiality
  paper (e.g. `mtuberculosis_H37Rv` matches DeJesus 2017's H37Rv
  Himar1 screen). Where the screen used a different strain than
  the canonical reference, gene-name mapping may need 1:1 ortholog
  resolution against the reference assembly.
- `vcholerae` is chromosome 1 only (AE003852.1); chromosome 2
  (AE003853.1) is not bundled because the Chao 2013 essentiality
  screen reports per-gene calls keyed to chromosome-1 loci. Add
  chr 2 if needed for whole-genome features.
